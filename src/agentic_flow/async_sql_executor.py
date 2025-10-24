"""
비동기 SQL 실행 및 성능 모니터링 시스템
SQL 실행 시간을 최적화하고 성능을 모니터링합니다.
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import threading
import queue
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.db import DatabaseManager, ReadOnlyDatabaseConnection
from agentic_flow.sql_cache import get_cache_manager, SQLCacheManager
from agentic_flow.sql_optimizer import SQLOptimizer

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    query_id: str
    sql: str
    execution_time: float
    cache_hit: bool
    optimization_applied: bool
    rows_returned: int
    memory_usage: float
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None

@dataclass
class PerformanceReport:
    """성능 리포트"""
    total_queries: int
    avg_execution_time: float
    cache_hit_rate: float
    optimization_rate: float
    slow_queries: List[PerformanceMetrics]
    error_rate: float
    time_range: Tuple[datetime, datetime]

class AsyncSQLExecutor:
    """비동기 SQL 실행기"""
    
    def __init__(self, 
                 max_workers: int = 10,
                 enable_cache: bool = True,
                 enable_optimization: bool = True,
                 enable_monitoring: bool = True):
        self.max_workers = max_workers
        self.enable_cache = enable_cache
        self.enable_optimization = enable_optimization
        self.enable_monitoring = enable_monitoring
        
        # 컴포넌트 초기화
        self.db_manager = DatabaseManager()
        self.cache_manager = get_cache_manager() if enable_cache else None
        self.optimizer = SQLOptimizer() if enable_optimization else None
        
        # 성능 모니터링
        self.metrics: List[PerformanceMetrics] = []
        self.metrics_lock = threading.Lock()
        self.performance_threshold = 5.0  # 5초 이상은 느린 쿼리
        
        # 스레드 풀
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 모니터링 설정
        self.monitoring_interval = 60  # 1분마다 리포트 생성
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        if enable_monitoring:
            self._start_monitoring()
            
        logger.info(f"AsyncSQLExecutor initialized: max_workers={max_workers}, cache={enable_cache}, optimization={enable_optimization}")

    def _start_monitoring(self):
        """성능 모니터링 시작"""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
            self.monitoring_thread.start()

    def _monitoring_worker(self):
        """성능 모니터링 백그라운드 작업"""
        while not self.stop_monitoring.is_set():
            try:
                self._generate_performance_report()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")

    def _generate_performance_report(self):
        """성능 리포트 생성"""
        with self.metrics_lock:
            if not self.metrics:
                return
                
            # 최근 1시간 데이터
            cutoff_time = time.time() - 3600
            recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
            
            if not recent_metrics:
                return
                
            # 리포트 생성
            report = PerformanceReport(
                total_queries=len(recent_metrics),
                avg_execution_time=statistics.mean([m.execution_time for m in recent_metrics]),
                cache_hit_rate=sum(1 for m in recent_metrics if m.cache_hit) / len(recent_metrics),
                optimization_rate=sum(1 for m in recent_metrics if m.optimization_applied) / len(recent_metrics),
                slow_queries=[m for m in recent_metrics if m.execution_time > self.performance_threshold],
                error_rate=sum(1 for m in recent_metrics if m.error) / len(recent_metrics),
                time_range=(datetime.fromtimestamp(min(m.timestamp for m in recent_metrics)),
                          datetime.fromtimestamp(max(m.timestamp for m in recent_metrics)))
            )
            
            # 로그 출력
            logger.info(f"Performance Report: {report.total_queries} queries, "
                       f"avg_time={report.avg_execution_time:.2f}s, "
                       f"cache_hit_rate={report.cache_hit_rate:.2%}, "
                       f"slow_queries={len(report.slow_queries)}")

    async def execute_sql(self, sql: str, params: Optional[Dict] = None, 
                         use_cache: bool = True, optimize: bool = True) -> Tuple[Any, PerformanceMetrics]:
        """비동기 SQL 실행"""
        query_id = f"query_{int(time.time() * 1000)}"
        start_time = time.time()
        
        metrics = PerformanceMetrics(
            query_id=query_id,
            sql=sql,
            execution_time=0.0,
            cache_hit=False,
            optimization_applied=False,
            rows_returned=0,
            memory_usage=0.0
        )
        
        try:
            # 캐시 확인
            if use_cache and self.cache_manager:
                cached_result = self.cache_manager.get(sql, params)
                if cached_result is not None:
                    metrics.cache_hit = True
                    metrics.execution_time = time.time() - start_time
                    self._record_metrics(metrics)
                    return cached_result, metrics
            
            # SQL 최적화
            optimized_sql = sql
            if optimize and self.optimizer:
                try:
                    optimization_result = self.optimizer.optimize_query(sql)
                    if optimization_result.optimized_sql != sql:
                        optimized_sql = optimization_result.optimized_sql
                        metrics.optimization_applied = True
                except Exception as e:
                    logger.warning(f"SQL optimization failed: {e}")
                    # 최적화 실패해도 원본 SQL로 실행
            
            # 비동기 실행
            result = await self._execute_sql_async(optimized_sql, params)
            
            # 결과 캐싱
            if use_cache and self.cache_manager and result:
                self.cache_manager.set(optimized_sql, result, params)
            
            # 메트릭 업데이트
            metrics.execution_time = time.time() - start_time
            metrics.rows_returned = len(result) if isinstance(result, list) else 1
            
            self._record_metrics(metrics)
            return result, metrics
            
        except Exception as e:
            metrics.error = str(e)
            metrics.execution_time = time.time() - start_time
            self._record_metrics(metrics)
            logger.error(f"SQL execution error: {e}")
            raise

    async def _execute_sql_async(self, sql: str, params: Optional[Dict] = None) -> Any:
        """실제 SQL 실행 (스레드 풀에서 실행)"""
        loop = asyncio.get_event_loop()
        
        def _execute():
            try:
                # ReadOnlyDatabaseConnection 인스턴스 직접 사용
                read_only_conn = self.db_manager._read_only_connection
                if not read_only_conn:
                    read_only_conn = self.db_manager._read_only_connection = ReadOnlyDatabaseConnection()
                
                with read_only_conn.get_connection() as connection:
                    cursor = connection.cursor(dictionary=True)
                    
                    if params:
                        cursor.execute(sql, params)
                    else:
                        cursor.execute(sql)
                    
                    if sql.strip().upper().startswith('SELECT'):
                        result = cursor.fetchall()
                    else:
                        result = cursor.rowcount
                    
                    cursor.close()
                    return result
                
            except Exception as e:
                logger.error(f"Database execution error: {e}")
                raise
        
        return await loop.run_in_executor(self.executor, _execute)

    def _record_metrics(self, metrics: PerformanceMetrics):
        """성능 메트릭 기록"""
        if not self.enable_monitoring:
            return
            
        with self.metrics_lock:
            self.metrics.append(metrics)
            
            # 메트릭 개수 제한 (최근 1000개만 유지)
            if len(self.metrics) > 1000:
                self.metrics = self.metrics[-1000:]

    async def execute_batch(self, queries: List[Tuple[str, Optional[Dict]]]) -> List[Tuple[Any, PerformanceMetrics]]:
        """배치 SQL 실행"""
        tasks = [
            self.execute_sql(sql, params)
            for sql, params in queries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 예외 처리
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch execution error: {result}")
                processed_results.append((None, PerformanceMetrics(
                    query_id="batch_error",
                    sql="",
                    execution_time=0.0,
                    cache_hit=False,
                    optimization_applied=False,
                    rows_returned=0,
                    memory_usage=0.0,
                    error=str(result)
                )))
            else:
                processed_results.append(result)
        
        return processed_results

    def get_performance_report(self, hours: int = 1) -> PerformanceReport:
        """성능 리포트 조회"""
        with self.metrics_lock:
            if not self.metrics:
                return PerformanceReport(
                    total_queries=0,
                    avg_execution_time=0.0,
                    cache_hit_rate=0.0,
                    optimization_rate=0.0,
                    slow_queries=[],
                    error_rate=0.0,
                    time_range=(datetime.now(), datetime.now())
                )
            
            # 지정된 시간 범위의 메트릭 필터링
            cutoff_time = time.time() - (hours * 3600)
            recent_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
            
            if not recent_metrics:
                return PerformanceReport(
                    total_queries=0,
                    avg_execution_time=0.0,
                    cache_hit_rate=0.0,
                    optimization_rate=0.0,
                    slow_queries=[],
                    error_rate=0.0,
                    time_range=(datetime.now(), datetime.now())
                )
            
            return PerformanceReport(
                total_queries=len(recent_metrics),
                avg_execution_time=statistics.mean([m.execution_time for m in recent_metrics]),
                cache_hit_rate=sum(1 for m in recent_metrics if m.cache_hit) / len(recent_metrics),
                optimization_rate=sum(1 for m in recent_metrics if m.optimization_applied) / len(recent_metrics),
                slow_queries=[m for m in recent_metrics if m.execution_time > self.performance_threshold],
                error_rate=sum(1 for m in recent_metrics if m.error) / len(recent_metrics),
                time_range=(datetime.fromtimestamp(min(m.timestamp for m in recent_metrics)),
                          datetime.fromtimestamp(max(m.timestamp for m in recent_metrics)))
            )

    def get_slow_queries(self, limit: int = 10) -> List[PerformanceMetrics]:
        """느린 쿼리 조회"""
        with self.metrics_lock:
            slow_queries = [m for m in self.metrics if m.execution_time > self.performance_threshold]
            return sorted(slow_queries, key=lambda x: x.execution_time, reverse=True)[:limit]

    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 조회"""
        if self.cache_manager:
            return self.cache_manager.get_cache_info()
        return {"cache_enabled": False}

    def clear_cache(self):
        """캐시 클리어"""
        if self.cache_manager:
            self.cache_manager.clear()
            logger.info("Cache cleared")

    def shutdown(self):
        """실행기 종료"""
        self.stop_monitoring.set()
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        self.executor.shutdown(wait=True)
        logger.info("AsyncSQLExecutor shutdown")

# 전역 실행기 인스턴스
_executor: Optional[AsyncSQLExecutor] = None

def get_async_executor() -> AsyncSQLExecutor:
    """전역 비동기 실행기 인스턴스 반환"""
    global _executor
    if _executor is None:
        _executor = AsyncSQLExecutor()
    return _executor

if __name__ == "__main__":
    # 테스트 코드
    print("=== 비동기 SQL 실행 및 성능 모니터링 테스트 ===")
    
    async def test_async_execution():
        # 실행기 생성
        executor = AsyncSQLExecutor(max_workers=5, enable_cache=True, enable_optimization=True)
        
        # 테스트 쿼리들
        test_queries = [
            ("SELECT * FROM t_member WHERE status = 'A' LIMIT 10", {}),
            ("SELECT COUNT(*) FROM t_member_login_log WHERE ins_datetime >= '2024-01-01'", {}),
            ("SELECT c.creator_name, COUNT(m.member_no) FROM t_creator c LEFT JOIN t_member m ON c.creator_no = m.creator_no GROUP BY c.creator_no LIMIT 5", {}),
        ]
        
        print("단일 쿼리 실행 테스트:")
        for i, (sql, params) in enumerate(test_queries):
            print(f"\n테스트 {i+1}: {sql[:50]}...")
            try:
                result, metrics = await executor.execute_sql(sql, params)
                print(f"  실행 시간: {metrics.execution_time:.4f}s")
                print(f"  캐시 히트: {metrics.cache_hit}")
                print(f"  최적화 적용: {metrics.optimization_applied}")
                print(f"  반환 행 수: {metrics.rows_returned}")
                if metrics.error:
                    print(f"  오류: {metrics.error}")
            except Exception as e:
                print(f"  실행 실패: {e}")
        
        print("\n배치 쿼리 실행 테스트:")
        try:
            batch_results = await executor.execute_batch(test_queries)
            print(f"배치 실행 완료: {len(batch_results)}개 쿼리")
            
            total_time = sum(metrics.execution_time for _, metrics in batch_results)
            print(f"총 실행 시간: {total_time:.4f}s")
            
        except Exception as e:
            print(f"배치 실행 실패: {e}")
        
        # 성능 리포트
        print("\n성능 리포트:")
        report = executor.get_performance_report()
        print(f"총 쿼리: {report.total_queries}")
        print(f"평균 실행 시간: {report.avg_execution_time:.4f}s")
        print(f"캐시 히트율: {report.cache_hit_rate:.2%}")
        print(f"최적화율: {report.optimization_rate:.2%}")
        print(f"느린 쿼리: {len(report.slow_queries)}개")
        print(f"오류율: {report.error_rate:.2%}")
        
        # 캐시 통계
        print("\n캐시 통계:")
        cache_stats = executor.get_cache_stats()
        print(f"캐시 크기: {cache_stats.get('cache_size', 0)}")
        print(f"히트율: {cache_stats.get('metrics', {}).get('hit_rate', 0):.2%}")
        
        # 종료
        executor.shutdown()
        print("\n실행기 종료 완료")
    
    # 비동기 테스트 실행
    asyncio.run(test_async_execution())
