#!/usr/bin/env python3
"""
스키마 캐싱 시스템
메타데이터 캐싱 및 메모리 관리를 위한 고급 캐싱 시스템
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import psutil
import gc

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """캐시 엔트리 데이터 클래스"""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0


@dataclass
class CacheMetrics:
    """캐시 성능 메트릭"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    max_size_bytes: int = 0
    hit_rate: float = 0.0
    memory_usage_percent: float = 0.0


class SchemaCache:
    """
    고급 스키마 캐싱 시스템
    LRU, TTL, 메모리 관리 기능을 포함한 캐싱 시스템
    """
    
    def __init__(self, 
                 max_size: int = 1000,
                 ttl_seconds: int = 3600,
                 max_memory_mb: int = 100,
                 cleanup_interval: int = 300):
        """
        SchemaCache 초기화
        
        Args:
            max_size: 최대 캐시 엔트리 수
            ttl_seconds: 캐시 만료 시간 (초)
            max_memory_mb: 최대 메모리 사용량 (MB)
            cleanup_interval: 정리 작업 간격 (초)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cleanup_interval = cleanup_interval
        
        # LRU 캐시 (OrderedDict 사용)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # 메트릭
        self.metrics = CacheMetrics()
        self.metrics.max_size_bytes = self.max_memory_bytes
        
        # 스레드 안전성
        self._lock = threading.RLock()
        
        # 백그라운드 정리 스레드
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        
        logger.info(f"SchemaCache initialized: max_size={max_size}, ttl={ttl_seconds}s, max_memory={max_memory_mb}MB")
    
    def start_background_cleanup(self):
        """백그라운드 정리 스레드 시작"""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._stop_cleanup.clear()
            self._cleanup_thread = threading.Thread(
                target=self._background_cleanup,
                daemon=True,
                name="SchemaCacheCleanup"
            )
            self._cleanup_thread.start()
            logger.info("Background cleanup thread started")
    
    def stop_background_cleanup(self):
        """백그라운드 정리 스레드 중지"""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5)
            logger.info("Background cleanup thread stopped")
    
    def get(self, key: str) -> Optional[Any]:
        """
        캐시에서 값 조회
        
        Args:
            key: 캐시 키
            
        Returns:
            캐시된 값 또는 None
        """
        with self._lock:
            if key not in self._cache:
                self.metrics.misses += 1
                return None
            
            entry = self._cache[key]
            current_time = time.time()
            
            # TTL 확인
            if current_time - entry.timestamp > self.ttl_seconds:
                del self._cache[key]
                self.metrics.misses += 1
                self.metrics.evictions += 1
                return None
            
            # LRU 업데이트 (맨 뒤로 이동)
            self._cache.move_to_end(key)
            entry.last_access = current_time
            entry.access_count += 1
            
            # 히트율 업데이트
            self._update_hit_rate()
            
            self.metrics.hits += 1
            return entry.value
    
    def set(self, key: str, value: Any, size_bytes: Optional[int] = None) -> bool:
        """
        캐시에 값 저장
        
        Args:
            key: 캐시 키
            value: 저장할 값
            size_bytes: 값의 크기 (바이트), None이면 자동 계산
            
        Returns:
            저장 성공 여부
        """
        with self._lock:
            try:
                # 크기 계산
                if size_bytes is None:
                    size_bytes = self._estimate_size(value)
                
                # 메모리 제한 확인
                if not self._check_memory_limit(size_bytes):
                    logger.warning(f"Memory limit exceeded, cannot cache key: {key}")
                    return False
                
                # 기존 엔트리 제거 (있다면)
                if key in self._cache:
                    old_entry = self._cache[key]
                    self.metrics.total_size_bytes -= old_entry.size_bytes
                    del self._cache[key]
                
                # 새 엔트리 생성
                entry = CacheEntry(
                    key=key,
                    value=value,
                    timestamp=time.time(),
                    size_bytes=size_bytes
                )
                
                # 캐시 크기 제한 확인
                if len(self._cache) >= self.max_size:
                    self._evict_lru()
                
                # 엔트리 추가
                self._cache[key] = entry
                self.metrics.total_size_bytes += size_bytes
                
                logger.debug(f"Cached key: {key}, size: {size_bytes} bytes")
                return True
                
            except Exception as e:
                logger.error(f"Failed to cache key '{key}': {str(e)}")
                return False
    
    def delete(self, key: str) -> bool:
        """
        캐시에서 값 삭제
        
        Args:
            key: 삭제할 캐시 키
            
        Returns:
            삭제 성공 여부
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self.metrics.total_size_bytes -= entry.size_bytes
                del self._cache[key]
                logger.debug(f"Deleted cache key: {key}")
                return True
            return False
    
    def clear(self):
        """캐시 전체 삭제"""
        with self._lock:
            self._cache.clear()
            self.metrics.total_size_bytes = 0
            self.metrics.hits = 0
            self.metrics.misses = 0
            self.metrics.evictions = 0
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        with self._lock:
            self._update_memory_usage()
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self.metrics.hits,
                "misses": self.metrics.misses,
                "evictions": self.metrics.evictions,
                "hit_rate": self.metrics.hit_rate,
                "total_size_bytes": self.metrics.total_size_bytes,
                "max_memory_bytes": self.max_memory_bytes,
                "memory_usage_percent": self.metrics.memory_usage_percent,
                "ttl_seconds": self.ttl_seconds
            }
    
    def _estimate_size(self, value: Any) -> int:
        """값의 크기 추정 (바이트)"""
        try:
            import sys
            return sys.getsizeof(value)
        except:
            # 기본값 반환
            return 1024
    
    def _check_memory_limit(self, additional_size: int) -> bool:
        """메모리 제한 확인"""
        return (self.metrics.total_size_bytes + additional_size) <= self.max_memory_bytes
    
    def _evict_lru(self):
        """LRU 정책으로 엔트리 제거"""
        if not self._cache:
            return
        
        # 가장 오래된 엔트리 제거
        oldest_key, oldest_entry = self._cache.popitem(last=False)
        self.metrics.total_size_bytes -= oldest_entry.size_bytes
        self.metrics.evictions += 1
        
        logger.debug(f"Evicted LRU entry: {oldest_key}")
    
    def _update_hit_rate(self):
        """히트율 업데이트"""
        total_requests = self.metrics.hits + self.metrics.misses
        if total_requests > 0:
            self.metrics.hit_rate = self.metrics.hits / total_requests
    
    def _update_memory_usage(self):
        """메모리 사용량 업데이트"""
        if self.max_memory_bytes > 0:
            self.metrics.memory_usage_percent = (self.metrics.total_size_bytes / self.max_memory_bytes) * 100
    
    def _background_cleanup(self):
        """백그라운드 정리 작업"""
        while not self._stop_cleanup.is_set():
            try:
                self._cleanup_expired_entries()
                self._cleanup_memory_pressure()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Error in background cleanup: {str(e)}")
    
    def _cleanup_expired_entries(self):
        """만료된 엔트리 정리"""
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, entry in self._cache.items():
                if current_time - entry.timestamp > self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self._cache[key]
                self.metrics.total_size_bytes -= entry.size_bytes
                del self._cache[key]
                self.metrics.evictions += 1
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired entries")
    
    def _cleanup_memory_pressure(self):
        """메모리 압박 시 정리"""
        with self._lock:
            # 메모리 사용량 확인
            memory_usage = psutil.virtual_memory().percent
            
            if memory_usage > 90:  # 80% -> 90%로 상향 조정
                # 가장 오래된 엔트리들 제거
                entries_to_remove = max(1, len(self._cache) // 4)  # 25% 제거
                
                for _ in range(entries_to_remove):
                    if self._cache:
                        key, entry = self._cache.popitem(last=False)
                        self.metrics.total_size_bytes -= entry.size_bytes
                        self.metrics.evictions += 1
                
                # 가비지 컬렉션 강제 실행
                gc.collect()
                
                logger.info(f"Memory pressure cleanup: removed {entries_to_remove} entries")
    
    def get_most_accessed(self, limit: int = 10) -> List[Tuple[str, int]]:
        """가장 많이 접근된 엔트리 목록 반환"""
        with self._lock:
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].access_count,
                reverse=True
            )
            return [(key, entry.access_count) for key, entry in sorted_entries[:limit]]
    
    def get_largest_entries(self, limit: int = 10) -> List[Tuple[str, int]]:
        """가장 큰 엔트리 목록 반환"""
        with self._lock:
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].size_bytes,
                reverse=True
            )
            return [(key, entry.size_bytes) for key, entry in sorted_entries[:limit]]


class SchemaCacheManager:
    """
    스키마 캐시 관리자
    여러 캐시 인스턴스를 관리하고 통합 통계 제공
    """
    
    def __init__(self):
        """SchemaCacheManager 초기화"""
        self.caches: Dict[str, SchemaCache] = {}
        self.default_cache = None
        
        # 기본 캐시 생성 및 사전 로딩
        self._initialize_default_cache()
        
        logger.info("SchemaCacheManager initialized")
    
    def _initialize_default_cache(self):
        """기본 캐시 생성 및 사전 로딩"""
        try:
            # 기본 캐시 생성
            self.default_cache = self.create_cache(
                "default",
                max_size=2000,
                ttl_seconds=7200,  # 2시간
                max_memory_mb=200
            )
            
            # 사전 로딩 수행 (백그라운드에서)
            import threading
            preload_thread = threading.Thread(
                target=self._preload_in_background,
                daemon=True
            )
            preload_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to initialize default cache: {e}")
    
    def _preload_in_background(self):
        """백그라운드에서 사전 로딩 수행"""
        try:
            if self.default_cache:
                self.default_cache.preload_common_schemas()
                self.default_cache.preload_relationship_info()
                logger.info("Background preloading completed")
        except Exception as e:
            logger.error(f"Background preloading failed: {e}")
    
    def create_cache(self, name: str, **kwargs) -> SchemaCache:
        """
        새로운 캐시 인스턴스 생성
        
        Args:
            name: 캐시 이름
            **kwargs: SchemaCache 생성자 인자
            
        Returns:
            생성된 SchemaCache 인스턴스
        """
        cache = SchemaCache(**kwargs)
        self.caches[name] = cache
        
        if self.default_cache is None:
            self.default_cache = cache
        
        logger.info(f"Created cache: {name}")
        return cache
    
    def get_cache(self, name: str) -> Optional[SchemaCache]:
        """캐시 인스턴스 조회"""
        return self.caches.get(name)
    
    def get_default_cache(self) -> Optional[SchemaCache]:
        """기본 캐시 인스턴스 반환"""
        return self.default_cache
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """모든 캐시의 통계 반환"""
        stats = {}
        for name, cache in self.caches.items():
            stats[name] = cache.get_stats()
        return stats
    
    def cleanup_all(self):
        """모든 캐시 정리"""
        for cache in self.caches.values():
            cache.clear()
        logger.info("All caches cleaned up")
    
    def start_all_background_cleanup(self):
        """모든 캐시의 백그라운드 정리 시작"""
        for cache in self.caches.values():
            cache.start_background_cleanup()
    
    def stop_all_background_cleanup(self):
        """모든 캐시의 백그라운드 정리 중지"""
        for cache in self.caches.values():
            cache.stop_background_cleanup()
    
    def preload_common_schemas(self):
        """자주 사용되는 스키마 정보를 사전에 로딩"""
        try:
            from core.db import get_db_connection
            
            db_connection = get_db_connection()
            
            # 자주 사용되는 테이블들의 스키마 사전 로딩
            common_tables = [
                't_member', 't_member_login_log', 't_creator', 
                't_content', 't_payment', 't_subscription'
            ]
            
            for table_name in common_tables:
                try:
                    # 테이블 스키마 정보 로딩
                    schema_key = f"table_schema:{table_name}"
                    if not self.get(schema_key):
                        table_schema = db_connection.get_table_schema(table_name)
                        self.set(schema_key, table_schema, ttl=7200)  # 2시간 TTL
                        logger.info(f"Preloaded schema for table: {table_name}")
                    
                    # 테이블 컬럼 정보 로딩
                    columns_key = f"table_columns:{table_name}"
                    if not self.get(columns_key):
                        columns = [col['COLUMN_NAME'] for col in table_schema]
                        self.set(columns_key, columns, ttl=7200)
                        logger.info(f"Preloaded columns for table: {table_name}")
                        
                except Exception as e:
                    logger.warning(f"Failed to preload schema for {table_name}: {e}")
            
            logger.info("Common schemas preloaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to preload common schemas: {e}")
    
    def preload_relationship_info(self):
        """테이블 관계 정보를 사전에 로딩"""
        try:
            from .database_relationship_analyzer import DatabaseRelationshipAnalyzer
            from core.db import get_db_connection
            
            db_connection = get_db_connection()
            analyzer = DatabaseRelationshipAnalyzer(db_connection)
            
            # 관계 정보 사전 로딩
            relationships_key = "table_relationships"
            if not self.get(relationships_key):
                relationships = analyzer.analyze_relationships()
                self.set(relationships_key, relationships, ttl=3600)  # 1시간 TTL
                logger.info("Preloaded table relationships")
            
            # 조인 경로 정보 사전 로딩
            join_paths_key = "join_paths"
            if not self.get(join_paths_key):
                # 자주 사용되는 테이블 조합의 조인 경로 사전 계산
                common_combinations = [
                    ('t_member', 't_member_login_log'),
                    ('t_member', 't_creator'),
                    ('t_content', 't_creator'),
                    ('t_member', 't_payment')
                ]
                
                join_paths = {}
                for table1, table2 in common_combinations:
                    try:
                        path = analyzer.find_join_path([table1], [table2])
                        if path:
                            join_paths[f"{table1}_{table2}"] = path
                    except Exception as e:
                        logger.warning(f"Failed to preload join path for {table1}-{table2}: {e}")
                
                self.set(join_paths_key, join_paths, ttl=3600)
                logger.info("Preloaded join paths")
            
        except Exception as e:
            logger.error(f"Failed to preload relationship info: {e}")


