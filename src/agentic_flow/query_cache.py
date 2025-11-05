"""
Query Result Caching Mechanism

This module implements a caching system for query results and intermediate
processing results to improve performance and reduce redundant computations.
"""

import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import OrderedDict
from threading import Lock

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """캐시 엔트리"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None  # Time To Live (초)
    
    def is_expired(self) -> bool:
        """TTL 기반 만료 확인"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl


class LRUCache:
    """
    LRU (Least Recently Used) 캐시 구현
    
    최근 사용 빈도에 따라 항목을 자동으로 제거하는 캐시입니다.
    """
    
    def __init__(self, max_size: int = 100, default_ttl: Optional[float] = None):
        """
        LRU 캐시 초기화
        
        Args:
            max_size: 최대 캐시 항목 수
            default_ttl: 기본 TTL (초, None이면 무제한)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        캐시에서 값 가져오기
        
        Args:
            key: 캐시 키
            
        Returns:
            캐시된 값 또는 None
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # 만료 확인
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None
            
            # LRU 업데이트: 마지막으로 사용된 항목을 맨 뒤로 이동
            self._cache.move_to_end(key)
            entry.last_accessed = time.time()
            entry.access_count += 1
            
            self._hits += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        캐시에 값 저장하기
        
        Args:
            key: 캐시 키
            value: 저장할 값
            ttl: TTL (초, None이면 default_ttl 사용)
        """
        with self._lock:
            ttl = ttl or self.default_ttl
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                ttl=ttl
            )
            
            # 기존 항목이 있으면 제거
            if key in self._cache:
                del self._cache[key]
            
            # 최대 크기 확인
            if len(self._cache) >= self.max_size:
                # 가장 오래 사용되지 않은 항목 제거
                self._cache.popitem(last=False)
            
            self._cache[key] = entry
            self._cache.move_to_end(key)
    
    def delete(self, key: str) -> bool:
        """
        캐시에서 항목 삭제
        
        Args:
            key: 캐시 키
            
        Returns:
            삭제 성공 여부
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """캐시 전체 비우기"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        캐시 통계 가져오기
        
        Returns:
            캐시 통계 딕셔너리
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "total_requests": total_requests
            }
    
    def cleanup_expired(self) -> int:
        """
        만료된 항목 정리
        
        Returns:
            삭제된 항목 수
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            return len(expired_keys)


class QueryCacheManager:
    """
    쿼리 결과 캐시 관리자
    
    다양한 유형의 쿼리 결과를 캐싱하고 관리합니다.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        QueryCacheManager 초기화
        
        Args:
            config: 설정 딕셔너리
                - max_result_cache_size: 결과 캐시 최대 크기
                - max_intermediate_cache_size: 중간 결과 캐시 최대 크기
                - result_cache_ttl: 결과 캐시 TTL (초)
                - intermediate_cache_ttl: 중간 결과 캐시 TTL (초)
        """
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)
        
        # 결과 캐시 (최종 쿼리 결과)
        self.result_cache = LRUCache(
            max_size=self.config.get("max_result_cache_size", 50),
            default_ttl=self.config.get("result_cache_ttl", 3600.0)  # 1시간
        )
        
        # 중간 결과 캐시 (RAG 검색 결과, SQL 생성 결과 등)
        self.intermediate_cache = LRUCache(
            max_size=self.config.get("max_intermediate_cache_size", 100),
            default_ttl=self.config.get("intermediate_cache_ttl", 1800.0)  # 30분
        )
        
        self.logger.info("QueryCacheManager initialized")
    
    def generate_cache_key(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        key_type: str = "result"
    ) -> str:
        """
        캐시 키 생성
        
        Args:
            query: 사용자 쿼리
            context: 추가 컨텍스트 (의도, 파라미터 등)
            key_type: 키 타입 ("result", "rag", "sql", "python_code")
            
        Returns:
            캐시 키 (해시)
        """
        # 쿼리와 컨텍스트를 결합하여 키 생성
        key_data = {
            "query": query.lower().strip(),
            "type": key_type,
            "context": context or {}
        }
        
        # JSON 직렬화하여 일관된 키 생성
        key_string = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        key_hash = hashlib.sha256(key_string.encode('utf-8')).hexdigest()
        
        return f"{key_type}:{key_hash}"
    
    def get_cached_result(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        캐시된 결과 가져오기
        
        Args:
            query: 사용자 쿼리
            context: 추가 컨텍스트
            
        Returns:
            캐시된 결과 또는 None
        """
        key = self.generate_cache_key(query, context, "result")
        result = self.result_cache.get(key)
        
        if result:
            self.logger.debug(f"Cache hit for query: {query[:50]}...")
        else:
            self.logger.debug(f"Cache miss for query: {query[:50]}...")
        
        return result
    
    def cache_result(
        self, 
        query: str, 
        result: Any, 
        context: Optional[Dict[str, Any]] = None,
        ttl: Optional[float] = None
    ) -> None:
        """
        쿼리 결과 캐싱
        
        Args:
            query: 사용자 쿼리
            result: 캐시할 결과
            context: 추가 컨텍스트
            ttl: TTL (초, None이면 기본값 사용)
        """
        key = self.generate_cache_key(query, context, "result")
        self.result_cache.set(key, result, ttl)
        self.logger.debug(f"Cached result for query: {query[:50]}...")
    
    def get_cached_intermediate(
        self, 
        cache_type: str, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        캐시된 중간 결과 가져오기
        
        Args:
            cache_type: 캐시 타입 ("rag", "sql", "python_code" 등)
            query: 쿼리 또는 키워드
            context: 추가 컨텍스트
            
        Returns:
            캐시된 중간 결과 또는 None
        """
        key = self.generate_cache_key(query, context, cache_type)
        return self.intermediate_cache.get(key)
    
    def cache_intermediate(
        self, 
        cache_type: str, 
        query: str, 
        result: Any, 
        context: Optional[Dict[str, Any]] = None,
        ttl: Optional[float] = None
    ) -> None:
        """
        중간 결과 캐싱
        
        Args:
            cache_type: 캐시 타입 ("rag", "sql", "python_code" 등)
            query: 쿼리 또는 키워드
            result: 캐시할 결과
            context: 추가 컨텍스트
            ttl: TTL (초, None이면 기본값 사용)
        """
        key = self.generate_cache_key(query, context, cache_type)
        self.intermediate_cache.set(key, result, ttl)
        self.logger.debug(f"Cached intermediate result ({cache_type}): {query[:50]}...")
    
    def invalidate_result(self, query: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        쿼리 결과 캐시 무효화
        
        Args:
            query: 사용자 쿼리
            context: 추가 컨텍스트
            
        Returns:
            무효화 성공 여부
        """
        key = self.generate_cache_key(query, context, "result")
        return self.result_cache.delete(key)
    
    def clear_all(self) -> None:
        """모든 캐시 비우기"""
        self.result_cache.clear()
        self.intermediate_cache.clear()
        self.logger.info("All caches cleared")
    
    def cleanup_expired(self) -> Tuple[int, int]:
        """
        만료된 항목 정리
        
        Returns:
            (결과 캐시 삭제 수, 중간 결과 캐시 삭제 수) 튜플
        """
        result_count = self.result_cache.cleanup_expired()
        intermediate_count = self.intermediate_cache.cleanup_expired()
        
        if result_count > 0 or intermediate_count > 0:
            self.logger.info(f"Cleaned up {result_count} expired result cache entries and {intermediate_count} intermediate cache entries")
        
        return (result_count, intermediate_count)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        캐시 통계 가져오기
        
        Returns:
            캐시 통계 딕셔너리
        """
        return {
            "result_cache": self.result_cache.get_stats(),
            "intermediate_cache": self.intermediate_cache.get_stats()
        }


# Global cache manager instance
_cache_manager: Optional[QueryCacheManager] = None
_cache_lock = Lock()


def get_cache_manager(config: Optional[Dict[str, Any]] = None) -> QueryCacheManager:
    """
    전역 캐시 관리자 인스턴스 가져오기 (싱글톤)
    
    Args:
        config: 설정 딕셔너리 (최초 초기화 시만 적용)
        
    Returns:
        QueryCacheManager 인스턴스
    """
    global _cache_manager
    
    with _cache_lock:
        if _cache_manager is None:
            _cache_manager = QueryCacheManager(config)
        return _cache_manager

