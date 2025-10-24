"""
SQL 결과 캐싱 시스템
TTL 기반 캐싱으로 SQL 실행 시간을 최적화합니다.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import threading
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """캐시 엔트리"""
    key: str
    data: Any
    created_at: float
    ttl: float
    hit_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """TTL 만료 여부 확인"""
        return time.time() - self.created_at > self.ttl

    def is_stale(self, stale_threshold: float = 300) -> bool:
        """데이터가 오래되었는지 확인 (5분)"""
        return time.time() - self.last_accessed > stale_threshold

    def touch(self):
        """접근 시간 업데이트"""
        self.last_accessed = time.time()
        self.hit_count += 1

@dataclass
class CacheMetrics:
    """캐시 성능 메트릭"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_queries: int = 0
    cache_size: int = 0
    hit_rate: float = 0.0
    avg_response_time: float = 0.0

    def update_hit_rate(self):
        """히트율 업데이트"""
        if self.total_queries > 0:
            self.hit_rate = self.hits / self.total_queries
        else:
            self.hit_rate = 0.0

class SQLCacheManager:
    """SQL 결과 캐싱 매니저"""
    
    def __init__(self, 
                 max_size: int = 1000,
                 default_ttl: float = 300,  # 5분
                 cleanup_interval: float = 60,  # 1분마다 정리
                 enable_metrics: bool = True):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.enable_metrics = enable_metrics
        
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._metrics = CacheMetrics()
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        
        # 캐시 정책 설정
        self._eviction_policy = "lru"  # LRU, LFU, TTL
        self._stale_threshold = 300  # 5분
        
        # 쿼리 패턴별 TTL 설정
        self._query_ttl_map = {
            "SELECT COUNT": 600,  # 집계 쿼리는 10분
            "SELECT * FROM t_member": 300,  # 회원 데이터는 5분
            "SELECT * FROM t_creator": 600,  # 크리에이터 데이터는 10분
            "SELECT * FROM t_post": 180,  # 포스트 데이터는 3분
            "SELECT * FROM t_member_login_log": 120,  # 로그 데이터는 2분
        }
        
        self._start_cleanup_thread()
        logger.info(f"SQL Cache Manager initialized: max_size={max_size}, default_ttl={default_ttl}s")

    def _start_cleanup_thread(self):
        """백그라운드 정리 스레드 시작"""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self._cleanup_thread.start()

    def _cleanup_worker(self):
        """백그라운드 정리 작업"""
        while not self._stop_cleanup.is_set():
            try:
                self._cleanup_expired_entries()
                self._cleanup_stale_entries()
                self._enforce_size_limit()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

    def _cleanup_expired_entries(self):
        """만료된 엔트리 정리"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self._cache[key]
                self._metrics.evictions += 1
                
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def _cleanup_stale_entries(self):
        """오래된 엔트리 정리"""
        with self._lock:
            stale_keys = [
                key for key, entry in self._cache.items()
                if entry.is_stale(self._stale_threshold)
            ]
            
            for key in stale_keys:
                del self._cache[key]
                self._metrics.evictions += 1
                
            if stale_keys:
                logger.debug(f"Cleaned up {len(stale_keys)} stale cache entries")

    def _enforce_size_limit(self):
        """캐시 크기 제한 적용"""
        if len(self._cache) <= self.max_size:
            return
            
        with self._lock:
            # LRU 정책으로 오래된 엔트리 제거
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].last_accessed
            )
            
            excess_count = len(self._cache) - self.max_size
            for i in range(excess_count):
                key, _ = sorted_entries[i]
                del self._cache[key]
                self._metrics.evictions += 1
                
            logger.debug(f"Evicted {excess_count} entries to enforce size limit")

    def _generate_cache_key(self, sql: str, params: Optional[Dict] = None) -> str:
        """SQL 쿼리와 파라미터로부터 캐시 키 생성"""
        # SQL 정규화 (공백, 대소문자 통일)
        normalized_sql = " ".join(sql.strip().split()).upper()
        
        # 파라미터 포함하여 해시 생성
        cache_data = {
            "sql": normalized_sql,
            "params": params or {}
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _get_ttl_for_query(self, sql: str) -> float:
        """쿼리 패턴에 따른 TTL 반환"""
        sql_upper = sql.upper()
        
        for pattern, ttl in self._query_ttl_map.items():
            if pattern.upper() in sql_upper:
                return ttl
                
        return self.default_ttl

    def get(self, sql: str, params: Optional[Dict] = None) -> Optional[Any]:
        """캐시에서 데이터 조회"""
        cache_key = self._generate_cache_key(sql, params)
        
        with self._lock:
            self._metrics.total_queries += 1
            
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                
                if entry.is_expired():
                    # 만료된 엔트리 제거
                    del self._cache[cache_key]
                    self._metrics.misses += 1
                    self._metrics.evictions += 1
                    logger.debug(f"Cache miss (expired): {cache_key[:8]}...")
                    return None
                
                # 캐시 히트
                entry.touch()
                self._metrics.hits += 1
                self._metrics.update_hit_rate()
                
                logger.debug(f"Cache hit: {cache_key[:8]}... (hits: {entry.hit_count})")
                return entry.data
            else:
                self._metrics.misses += 1
                self._metrics.update_hit_rate()
                logger.debug(f"Cache miss: {cache_key[:8]}...")
                return None

    def set(self, sql: str, data: Any, params: Optional[Dict] = None, 
            ttl: Optional[float] = None, metadata: Optional[Dict] = None) -> str:
        """캐시에 데이터 저장"""
        cache_key = self._generate_cache_key(sql, params)
        
        if ttl is None:
            ttl = self._get_ttl_for_query(sql)
        
        entry = CacheEntry(
            key=cache_key,
            data=data,
            created_at=time.time(),
            ttl=ttl,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._cache[cache_key] = entry
            self._metrics.cache_size = len(self._cache)
            
        logger.debug(f"Cache set: {cache_key[:8]}... (ttl: {ttl}s)")
        return cache_key

    def invalidate(self, pattern: Optional[str] = None, table: Optional[str] = None):
        """캐시 무효화"""
        with self._lock:
            if pattern:
                # SQL 패턴으로 무효화
                keys_to_remove = [
                    key for key, entry in self._cache.items()
                    if pattern.upper() in entry.metadata.get("sql", "").upper()
                ]
            elif table:
                # 테이블명으로 무효화
                keys_to_remove = [
                    key for key, entry in self._cache.items()
                    if table.upper() in entry.metadata.get("sql", "").upper()
                ]
            else:
                # 전체 캐시 무효화
                keys_to_remove = list(self._cache.keys())
            
            for key in keys_to_remove:
                del self._cache[key]
                self._metrics.evictions += 1
                
            self._metrics.cache_size = len(self._cache)
            logger.info(f"Cache invalidated: {len(keys_to_remove)} entries removed")

    def get_metrics(self) -> CacheMetrics:
        """캐시 성능 메트릭 반환"""
        with self._lock:
            self._metrics.cache_size = len(self._cache)
            return self._metrics

    def get_cache_info(self) -> Dict[str, Any]:
        """캐시 상태 정보 반환"""
        with self._lock:
            return {
                "cache_size": len(self._cache),
                "max_size": self.max_size,
                "metrics": {
                    "hits": self._metrics.hits,
                    "misses": self._metrics.misses,
                    "hit_rate": self._metrics.hit_rate,
                    "evictions": self._metrics.evictions,
                    "total_queries": self._metrics.total_queries
                },
                "entries": [
                    {
                        "key": key[:8] + "...",
                        "created_at": datetime.fromtimestamp(entry.created_at).isoformat(),
                        "ttl": entry.ttl,
                        "hit_count": entry.hit_count,
                        "is_expired": entry.is_expired()
                    }
                    for key, entry in list(self._cache.items())[:10]  # 최대 10개만 표시
                ]
            }

    def clear(self):
        """전체 캐시 클리어"""
        with self._lock:
            self._cache.clear()
            self._metrics = CacheMetrics()
            logger.info("Cache cleared")

    def shutdown(self):
        """캐시 매니저 종료"""
        self._stop_cleanup.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        logger.info("SQL Cache Manager shutdown")

# 전역 캐시 매니저 인스턴스
_cache_manager: Optional[SQLCacheManager] = None

def get_cache_manager() -> SQLCacheManager:
    """전역 캐시 매니저 인스턴스 반환"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = SQLCacheManager()
    return _cache_manager

def cache_sql_result(sql: str, params: Optional[Dict] = None, 
                    ttl: Optional[float] = None, metadata: Optional[Dict] = None):
    """SQL 결과 캐싱 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            
            # 캐시에서 조회
            cached_result = cache_manager.get(sql, params)
            if cached_result is not None:
                return cached_result
            
            # 함수 실행
            result = func(*args, **kwargs)
            
            # 결과 캐싱
            cache_manager.set(sql, result, params, ttl, metadata)
            
            return result
        return wrapper
    return decorator

if __name__ == "__main__":
    # 테스트 코드
    print("=== SQL 캐싱 시스템 테스트 ===")
    
    # 캐시 매니저 생성
    cache_manager = SQLCacheManager(max_size=100, default_ttl=10)
    
    # 테스트 데이터
    test_queries = [
        ("SELECT * FROM t_member WHERE status = 'A'", {"status": "A"}),
        ("SELECT COUNT(*) FROM t_member_login_log WHERE ins_datetime >= '2024-01-01'", {}),
        ("SELECT c.creator_name, COUNT(m.member_no) FROM t_creator c LEFT JOIN t_member m ON c.creator_no = m.creator_no GROUP BY c.creator_no", {}),
    ]
    
    # 캐시 테스트
    for i, (sql, params) in enumerate(test_queries):
        print(f"\n테스트 {i+1}: {sql[:50]}...")
        
        # 첫 번째 조회 (캐시 미스)
        start_time = time.time()
        result1 = cache_manager.get(sql, params)
        miss_time = time.time() - start_time
        
        if result1 is None:
            # 가짜 데이터 생성 및 캐싱
            fake_data = [{"id": j, "name": f"test_{j}"} for j in range(10)]
            cache_manager.set(sql, fake_data, params, metadata={"sql": sql})
            print(f"  캐시 미스: {miss_time:.4f}s (데이터 캐싱)")
        else:
            print(f"  캐시 히트: {miss_time:.4f}s")
        
        # 두 번째 조회 (캐시 히트)
        start_time = time.time()
        result2 = cache_manager.get(sql, params)
        hit_time = time.time() - start_time
        
        if result2 is not None:
            print(f"  캐시 히트: {hit_time:.4f}s")
        else:
            print(f"  캐시 미스: {hit_time:.4f}s")
    
    # 메트릭 출력
    print(f"\n=== 캐시 메트릭 ===")
    metrics = cache_manager.get_metrics()
    print(f"히트율: {metrics.hit_rate:.2%}")
    print(f"총 쿼리: {metrics.total_queries}")
    print(f"히트: {metrics.hits}")
    print(f"미스: {metrics.misses}")
    print(f"제거: {metrics.evictions}")
    
    # 캐시 정보 출력
    print(f"\n=== 캐시 정보 ===")
    cache_info = cache_manager.get_cache_info()
    print(f"캐시 크기: {cache_info['cache_size']}/{cache_info['max_size']}")
    
    # 캐시 무효화 테스트
    print(f"\n=== 캐시 무효화 테스트 ===")
    cache_manager.invalidate(table="t_member")
    print("t_member 관련 캐시 무효화 완료")
    
    # 종료
    cache_manager.shutdown()
    print("캐시 매니저 종료 완료")

