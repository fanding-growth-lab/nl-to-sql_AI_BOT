#!/usr/bin/env python3
"""
Intent Classification Statistics Collection System

This module implements comprehensive statistics collection for LLMIntentClassifier
to track classification performance metrics including total classifications,
confidence scores, intent distribution, and error rates.
"""

import json
import time
import threading
import queue
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Timer
import logging
import psutil
import os

from .nodes import QueryIntent
from .statistics_persistence import get_persistence_manager, PersistenceConfig


@dataclass
class ClassificationMetrics:
    """Individual classification metrics"""
    intent: str
    confidence: float
    response_time_ms: float
    timestamp: float
    is_error: bool = False


@dataclass
class QueryInteractionMetrics:
    """
    통합 쿼리 상호작용 메트릭스
    
    Intent 분류 정보와 AutoLearning 데이터를 통합하여 저장합니다.
    이 구조는 두 시스템 간 데이터 공유를 용이하게 합니다.
    """
    # 기본 쿼리 정보 (기본값 없는 필드는 앞에)
    user_query: str
    intent: str
    intent_confidence: float
    
    # 선택적 필드들 (기본값 있는 필드는 뒤에)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    intent_reasoning: Optional[str] = None
    
    # 처리 결과
    sql_query: Optional[str] = None
    validation_passed: bool = False
    execution_success: bool = False
    execution_result_count: int = 0
    
    # 성능 메트릭스
    response_time_ms: float = 0.0
    total_processing_time_ms: float = 0.0
    
    # 학습 관련 정보
    mapping_result: Optional[Dict[str, Any]] = None
    schema_mapping: Optional[Dict[str, Any]] = None
    template_used: Optional[str] = None
    
    # 사용자 피드백
    user_feedback: Optional[str] = None
    user_satisfaction: Optional[float] = None
    
    # 타임스탬프
    timestamp: float = 0.0
    
    # 에러 정보
    is_error: bool = False
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary 변환"""
        return {
            "user_query": self.user_query,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "intent_reasoning": self.intent_reasoning,
            "sql_query": self.sql_query,
            "validation_passed": self.validation_passed,
            "execution_success": self.execution_success,
            "execution_result_count": self.execution_result_count,
            "response_time_ms": self.response_time_ms,
            "total_processing_time_ms": self.total_processing_time_ms,
            "mapping_result": self.mapping_result,
            "schema_mapping": self.schema_mapping,
            "template_used": self.template_used,
            "user_feedback": self.user_feedback,
            "user_satisfaction": self.user_satisfaction,
            "timestamp": self.timestamp,
            "is_error": self.is_error,
            "error_message": self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryInteractionMetrics':
        """Dictionary에서 객체 생성"""
        return cls(**data)


@dataclass
class BatchProcessingConfig:
    """배치 처리 설정"""
    batch_size: int = 100
    flush_interval_seconds: float = 5.0
    max_queue_size: int = 10000
    enable_memory_monitoring: bool = True
    memory_threshold_percent: float = 80.0
    sampling_rate: float = 1.0  # 1.0 = 모든 데이터 수집, 0.5 = 50% 샘플링


@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    processing_time_ms: float = 0.0
    queue_size: int = 0
    memory_usage_mb: float = 0.0
    batch_processing_rate: float = 0.0
    error_count: int = 0


class BatchProcessor:
    """고급 배치 처리기"""
    
    def __init__(self, config: BatchProcessingConfig, 
                 process_callback: Callable[[List[ClassificationMetrics]], None]):
        self.config = config
        self.process_callback = process_callback
        self.logger = logging.getLogger(__name__)
        
        # 큐 및 동기화
        self._queue = queue.Queue(maxsize=config.max_queue_size)
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # 배치 처리
        self._current_batch: List[ClassificationMetrics] = []
        self._batch_lock = threading.Lock()
        
        # 성능 모니터링
        self._performance_metrics = PerformanceMetrics()
        self._last_flush_time = time.time()
        
        # 타이머 및 워커 스레드
        self._flush_timer: Optional[Timer] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        
        # 통계
        self._total_processed = 0
        self._total_errors = 0
        self._start_time = time.time()
        
        self._start_processing()
    
    def _start_processing(self):
        """배치 처리 시작"""
        # 워커 스레드 시작
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="batch-processor-worker",
            daemon=True
        )
        self._worker_thread.start()
        
        # 모니터링 스레드 시작
        if self.config.enable_memory_monitoring:
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="batch-processor-monitor",
                daemon=True
            )
            self._monitor_thread.start()
        
        # 주기적 플러시 타이머 시작
        self._schedule_flush()
    
    def _worker_loop(self):
        """워커 스레드 메인 루프"""
        while not self._shutdown_event.is_set():
            try:
                # 큐에서 메트릭 가져오기 (타임아웃 설정)
                try:
                    metrics = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # 샘플링 적용
                if not self._should_sample():
                    continue
                
                # 배치에 추가
                with self._batch_lock:
                    self._current_batch.append(metrics)
                    
                    # 배치 크기 확인
                    if len(self._current_batch) >= self.config.batch_size:
                        self._process_batch()
                
                self._queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in worker loop: {e}")
                self._total_errors += 1
    
    def _monitor_loop(self):
        """모니터링 스레드 루프"""
        while not self._shutdown_event.is_set():
            try:
                self._update_performance_metrics()
                
                # 메모리 사용량 확인
                if self.config.enable_memory_monitoring:
                    memory_usage_percent = self._performance_metrics.memory_usage_mb  # 이미 백분율로 계산됨
                    if memory_usage_percent > self.config.memory_threshold_percent:
                        self.logger.warning(f"High memory usage: {memory_usage_percent:.1f}%")
                        self._force_flush()
                
                time.sleep(1.0)  # 1초마다 모니터링
                
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
    
    def _should_sample(self) -> bool:
        """샘플링 결정"""
        if self.config.sampling_rate >= 1.0:
            return True
        return (time.time() * 1000) % 100 < (self.config.sampling_rate * 100)
    
    def _process_batch(self):
        """배치 처리"""
        if not self._current_batch:
            return
        
        batch_start_time = time.time()
        
        try:
            # 콜백 함수 호출
            self.process_callback(self._current_batch.copy())
            
            # 통계 업데이트
            self._total_processed += len(self._current_batch)
            
            # 배치 클리어
            self._current_batch.clear()
            
            # 성능 메트릭 업데이트
            processing_time = (time.time() - batch_start_time) * 1000
            self._performance_metrics.processing_time_ms = processing_time
            
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            self._total_errors += 1
    
    def _schedule_flush(self):
        """주기적 플러시 스케줄링"""
        if self._shutdown_event.is_set():
            return
        
        self._flush_timer = Timer(
            self.config.flush_interval_seconds,
            self._scheduled_flush
        )
        self._flush_timer.start()
    
    def _scheduled_flush(self):
        """스케줄된 플러시 실행"""
        try:
            with self._batch_lock:
                if self._current_batch:
                    self._process_batch()
            
            # 다음 플러시 스케줄링
            self._schedule_flush()
            
        except Exception as e:
            self.logger.error(f"Error in scheduled flush: {e}")
    
    def _force_flush(self):
        """강제 플러시"""
        with self._batch_lock:
            if self._current_batch:
                self._process_batch()
    
    def _get_available_memory_mb(self) -> float:
        """
        사용 가능한 메모리 크기 반환 (컨테이너 환경 고려)
        
        컨테이너 환경에서는 cgroups 메모리 제한을 우선적으로 사용하고,
        일반 환경에서는 시스템 전체 메모리를 사용합니다.
        """
        try:
            import psutil
            
            # 1. cgroups 메모리 제한 확인 (컨테이너 환경)
            cgroup_limit_mb = self._get_cgroup_memory_limit_mb()
            if cgroup_limit_mb and cgroup_limit_mb > 0:
                self.logger.debug(f"Using cgroup memory limit: {cgroup_limit_mb:.1f} MB")
                return cgroup_limit_mb
            
            # 2. 시스템 전체 메모리 사용 (일반 환경)
            system_memory = psutil.virtual_memory()
            total_memory_mb = system_memory.total / 1024 / 1024
            self.logger.debug(f"Using system total memory: {total_memory_mb:.1f} MB")
            return total_memory_mb
            
        except Exception as e:
            self.logger.warning(f"Failed to get available memory: {e}, using default")
            # 폴백: 기본값 사용 (예: 8GB)
            return 8192.0
    
    def _get_cgroup_memory_limit_mb(self) -> Optional[float]:
        """
        cgroups 메모리 제한 읽기 (Docker/Kubernetes 환경)
        
        Returns:
            메모리 제한 (MB 단위) 또는 None (cgroups 미사용 환경)
        """
        try:
            # cgroups v2 경로 (최신 시스템)
            cgroup_v2_path = "/sys/fs/cgroup/memory.max"
            if os.path.exists(cgroup_v2_path):
                with open(cgroup_v2_path, 'r') as f:
                    limit_str = f.read().strip()
                    if limit_str.isdigit():
                        limit_bytes = int(limit_str)
                        if limit_bytes > 0:
                            return limit_bytes / 1024 / 1024
                    elif limit_str == "max":
                        # 제한 없음
                        return None
            
            # cgroups v1 경로 (구버전 시스템)
            cgroup_v1_paths = [
                "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                "/sys/fs/cgroup/memory.max"  # 일부 시스템에서 사용
            ]
            
            for path in cgroup_v1_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        limit_str = f.read().strip()
                        if limit_str.isdigit():
                            limit_bytes = int(limit_str)
                            # 9223372036854771712는 "무제한"을 의미 (64비트 최대값 근사치)
                            if limit_bytes > 0 and limit_bytes < 9000000000000000000:
                                return limit_bytes / 1024 / 1024
            
            # cgroups 없음 (일반 환경)
            return None
            
        except (IOError, OSError, ValueError) as e:
            # 파일 읽기 실패 또는 컨테이너 환경이 아님
            self.logger.debug(f"cgroup memory limit not available: {e}")
            return None
    
    def _update_performance_metrics(self):
        """성능 메트릭 업데이트"""
        try:
            # 큐 크기
            self._performance_metrics.queue_size = self._queue.qsize()
            
            # 메모리 사용량 (백분율로 계산)
            if self.config.enable_memory_monitoring:
                import psutil
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # 컨테이너 환경 고려: cgroups 메모리 제한 확인
                total_memory_mb = self._get_available_memory_mb()
                
                # 백분율 계산
                memory_usage_percent = (memory_mb / total_memory_mb) * 100 if total_memory_mb > 0 else 0
                self._performance_metrics.memory_usage_mb = memory_usage_percent
            else:
                self._performance_metrics.memory_usage_mb = 0.0
            
            # 배치 처리율 계산
            elapsed_time = time.time() - self._start_time
            if elapsed_time > 0:
                self._performance_metrics.batch_processing_rate = self._total_processed / elapsed_time
            
            # 오류 수
            self._performance_metrics.error_count = self._total_errors
            
        except Exception as e:
            self.logger.warning(f"Failed to update performance metrics: {e}")
    
    def add_metrics(self, metrics: ClassificationMetrics) -> bool:
        """메트릭 추가"""
        try:
            self._queue.put_nowait(metrics)
            return True
        except queue.Full:
            self.logger.warning("Queue is full, dropping metrics")
            return False
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """성능 메트릭 반환"""
        self._update_performance_metrics()
        return self._performance_metrics
    
    def shutdown(self, timeout: float = 5.0):
        """배치 처리기 종료"""
        self.logger.info("Shutting down batch processor...")
        
        # 종료 이벤트 설정
        self._shutdown_event.set()
        
        # 타이머 취소
        if self._flush_timer:
            self._flush_timer.cancel()
        
        # 강제 플러시
        self._force_flush()
        
        # 워커 스레드 종료 대기
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
        
        # 모니터링 스레드 종료 대기
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=timeout)
        
        self.logger.info(f"Batch processor shutdown complete. Processed: {self._total_processed}, Errors: {self._total_errors}")


@dataclass
class IntentClassifierStats:
    """Comprehensive statistics for intent classification"""
    total_classifications: int = 0
    total_errors: int = 0
    average_confidence: float = 0.0
    intent_distribution: Dict[str, int] = None
    confidence_distribution: Dict[str, int] = None
    response_times: Dict[str, float] = None
    last_updated: float = 0.0
    
    def __post_init__(self):
        if self.intent_distribution is None:
            self.intent_distribution = defaultdict(int)
        if self.confidence_distribution is None:
            self.confidence_distribution = defaultdict(int)
        if self.response_times is None:
            self.response_times = {
                "min": float('inf'),
                "max": 0.0,
                "avg": 0.0,
                "total": 0.0
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "total_classifications": self.total_classifications,
            "total_errors": self.total_errors,
            "error_rate": self.error_rate,
            "average_confidence": self.average_confidence,
            "intent_distribution": dict(self.intent_distribution),
            "confidence_distribution": dict(self.confidence_distribution),
            "response_times": self.response_times.copy(),
            "last_updated": self.last_updated
        }
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage"""
        if self.total_classifications == 0:
            return 0.0
        return (self.total_errors / self.total_classifications) * 100.0


class StatisticsCollector:
    """Thread-safe statistics collector for intent classification with advanced batch processing and persistence"""
    
    def __init__(self, enable_async: bool = True, batch_size: int = 100, 
                 config: Optional[BatchProcessingConfig] = None,
                 persistence_config: Optional[PersistenceConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.enable_async = enable_async
        
        # 배치 처리 설정
        if config is None:
            config = BatchProcessingConfig(
                batch_size=batch_size,
                flush_interval_seconds=5.0,
                max_queue_size=10000,
                enable_memory_monitoring=True,
                memory_threshold_percent=80.0,
                sampling_rate=1.0
            )
        self.config = config
        
        # Thread-safe statistics storage
        self._lock = threading.RLock()
        self._stats = IntentClassifierStats()
        
        # 고급 배치 처리기 초기화
        if self.enable_async:
            self._batch_processor = BatchProcessor(
                config=config,
                process_callback=self._process_batch_callback
            )
        else:
            self._batch_processor = None
        
        # 기존 동기 처리용 (fallback)
        self._pending_metrics: List[ClassificationMetrics] = []
        self._batch_lock = threading.Lock()
        
        # 성능 메트릭
        self._performance_metrics = PerformanceMetrics()
        
        # 지속성 관리자 초기화
        self._persistence_manager = get_persistence_manager()
        if persistence_config:
            # 사용자 정의 설정이 있으면 새 인스턴스 생성
            from .statistics_persistence import StatisticsPersistenceManager
            self._persistence_manager = StatisticsPersistenceManager(persistence_config)
    
    def _process_batch_callback(self, batch: List[ClassificationMetrics]) -> None:
        """배치 처리 콜백"""
        try:
            with self._lock:
                for metrics in batch:
                    self._update_stats(metrics)
            
            self.logger.debug(f"Processed batch of {len(batch)} metrics")
            
            # 지속성 저장
            self._save_to_persistence(batch)
            
        except Exception as e:
            self.logger.error(f"Error in batch processing callback: {e}")
    
    def _save_to_persistence(self, metrics_batch: List[ClassificationMetrics]) -> None:
        """지속성 저장"""
        try:
            # 현재 통계 가져오기
            current_stats = self._stats
            
            # 지속성 저장
            success = self._persistence_manager.save_statistics(current_stats, metrics_batch)
            if success:
                self.logger.debug(f"Saved {len(metrics_batch)} metrics to persistence")
            else:
                self.logger.warning("Failed to save metrics to persistence")
                
        except Exception as e:
            self.logger.error(f"Error saving to persistence: {e}")
    
    def record_classification(self, intent: str, confidence: float, 
                            response_time_ms: float, is_error: bool = False) -> None:
        """Record a single classification result"""
        metrics = ClassificationMetrics(
            intent=intent,
            confidence=confidence,
            response_time_ms=response_time_ms,
            timestamp=time.time(),
            is_error=is_error
        )
        
        if self.enable_async and self._batch_processor:
            # 고급 배치 처리 사용
            success = self._batch_processor.add_metrics(metrics)
            if not success:
                self.logger.warning("Failed to add metrics to batch processor, falling back to sync processing")
                self._process_metrics_sync(metrics)
        else:
            # 동기 처리 (fallback)
            self._process_metrics_sync(metrics)
    
    def _process_metrics_async(self, metrics: ClassificationMetrics) -> None:
        """Process metrics asynchronously"""
        try:
            with self._batch_lock:
                self._pending_metrics.append(metrics)
                
                # Process batch when size limit reached
                if len(self._pending_metrics) >= self.batch_size:
                    self._process_batch()
        except Exception as e:
            self.logger.error(f"Error in async metrics processing: {e}")
    
    def _process_metrics_sync(self, metrics: ClassificationMetrics) -> None:
        """Process metrics synchronously"""
        try:
            with self._lock:
                self._update_stats(metrics)
        except Exception as e:
            self.logger.error(f"Error in sync metrics processing: {e}")
    
    def _process_batch(self) -> None:
        """Process a batch of pending metrics"""
        if not self._pending_metrics:
            return
        
        batch = self._pending_metrics.copy()
        self._pending_metrics.clear()
        
        with self._lock:
            for metrics in batch:
                self._update_stats(metrics)
    
    def _update_stats(self, metrics: ClassificationMetrics) -> None:
        """Update statistics with new metrics"""
        # Update counters
        self._stats.total_classifications += 1
        if metrics.is_error:
            self._stats.total_errors += 1
        
        # Update intent distribution
        self._stats.intent_distribution[metrics.intent] += 1
        
        # Update confidence distribution
        confidence_level = self._get_confidence_level(metrics.confidence)
        self._stats.confidence_distribution[confidence_level] += 1
        
        # Update average confidence
        total_confidence = self._stats.average_confidence * (self._stats.total_classifications - 1)
        self._stats.average_confidence = (total_confidence + metrics.confidence) / self._stats.total_classifications
        
        # Update response times
        self._stats.response_times["min"] = min(self._stats.response_times["min"], metrics.response_time_ms)
        self._stats.response_times["max"] = max(self._stats.response_times["max"], metrics.response_time_ms)
        self._stats.response_times["total"] += metrics.response_time_ms
        self._stats.response_times["avg"] = self._stats.response_times["total"] / self._stats.total_classifications
        
        # Update timestamp
        self._stats.last_updated = metrics.timestamp
    
    def _get_confidence_level(self, confidence: float) -> str:
        """
        신뢰도 레벨 분류
        
        Args:
            confidence: 신뢰도 점수 (0.0 ~ 1.0)
            
        Returns:
            "high" (confidence >= 0.8)
            "medium" (0.5 <= confidence < 0.8)
            "low" (confidence < 0.5)
        """
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        else:
            return "low"
    
    def get_stats(self) -> IntentClassifierStats:
        """Get current statistics"""
        with self._lock:
            # Process any pending batch (fallback for sync mode)
            if not self.enable_async or not self._batch_processor:
                with self._batch_lock:
                    if self._pending_metrics:
                        self._process_batch()
            
            return self._stats
    
    def get_stats_as_json(self) -> str:
        """Get statistics as JSON string"""
        stats = self.get_stats()
        return json.dumps(stats.to_dict(), indent=2)
    
    def get_specific_metric(self, metric_type: str) -> Any:
        """Get specific metric by type"""
        stats = self.get_stats()
        
        metric_map = {
            "total_classifications": stats.total_classifications,
            "total_errors": stats.total_errors,
            "error_rate": stats.error_rate,
            "average_confidence": stats.average_confidence,
            "intent_distribution": dict(stats.intent_distribution),
            "confidence_distribution": dict(stats.confidence_distribution),
            "response_times": stats.response_times,
            "last_updated": stats.last_updated
        }
        
        return metric_map.get(metric_type)
    
    def reset_stats(self) -> None:
        """Reset all statistics"""
        with self._lock:
            self._stats = IntentClassifierStats()
            with self._batch_lock:
                self._pending_metrics.clear()
    
    def reset_specific_metric(self, metric_type: str) -> None:
        """Reset specific metric"""
        with self._lock:
            if metric_type == "total_classifications":
                self._stats.total_classifications = 0
            elif metric_type == "total_errors":
                self._stats.total_errors = 0
            elif metric_type == "average_confidence":
                self._stats.average_confidence = 0.0
            elif metric_type == "intent_distribution":
                self._stats.intent_distribution.clear()
            elif metric_type == "confidence_distribution":
                self._stats.confidence_distribution.clear()
            elif metric_type == "response_times":
                self._stats.response_times = {
                    "min": float('inf'),
                    "max": 0.0,
                    "avg": 0.0,
                    "total": 0.0
                }
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get performance metrics"""
        if self._batch_processor:
            return self._batch_processor.get_performance_metrics()
        else:
            # Fallback for sync mode
            return PerformanceMetrics()
    
    def get_batch_processing_config(self) -> BatchProcessingConfig:
        """Get batch processing configuration"""
        return self.config
    
    def update_sampling_rate(self, sampling_rate: float) -> None:
        """Update sampling rate dynamically"""
        if 0.0 <= sampling_rate <= 1.0:
            self.config.sampling_rate = sampling_rate
            self.logger.info(f"Updated sampling rate to {sampling_rate}")
        else:
            self.logger.warning(f"Invalid sampling rate: {sampling_rate}")
    
    def force_flush(self) -> None:
        """Force flush pending metrics"""
        if self._batch_processor:
            self._batch_processor._force_flush()
        else:
            # Fallback for sync mode
            with self._batch_lock:
                if self._pending_metrics:
                    self._process_batch()
    
    def load_historical_stats(self, start_time: Optional[float] = None, 
                             end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """히스토리컬 통계 데이터 로드"""
        try:
            return self._persistence_manager.load_statistics(start_time, end_time)
        except Exception as e:
            self.logger.error(f"Failed to load historical stats: {e}")
            return []
    
    def create_backup(self) -> bool:
        """백업 생성"""
        try:
            return self._persistence_manager.create_backup()
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            return False
    
    def cleanup_old_data(self) -> None:
        """오래된 데이터 정리"""
        try:
            self._persistence_manager.cleanup_old_data()
            self.logger.info("Old data cleanup completed")
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
    
    def get_storage_info(self) -> Dict[str, Any]:
        """저장소 정보 반환"""
        try:
            return self._persistence_manager.get_storage_info()
        except Exception as e:
            self.logger.error(f"Failed to get storage info: {e}")
            return {}
    
    def force_save(self) -> bool:
        """강제 저장"""
        try:
            # 현재 배치 강제 플러시
            self.force_flush()
            
            # 현재 통계 저장
            current_stats = self.get_stats()
            success = self._persistence_manager.save_statistics(current_stats, None)
            
            if success:
                self.logger.info("Force save completed successfully")
            else:
                self.logger.warning("Force save failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error in force save: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown the statistics collector"""
        try:
            # 마지막 저장
            self.force_save()
            
            # 배치 처리기 종료
            if self._batch_processor:
                self._batch_processor.shutdown()
            else:
                # Fallback for sync mode
                with self._batch_lock:
                    if self._pending_metrics:
                        self._process_batch()
            
            self.logger.info("Statistics collector shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Global statistics collector instance
_stats_collector: Optional[StatisticsCollector] = None


def get_stats_collector() -> StatisticsCollector:
    """Get the global statistics collector instance"""
    global _stats_collector
    if _stats_collector is None:
        _stats_collector = StatisticsCollector()
    return _stats_collector


def shutdown_stats_collector() -> None:
    """Shutdown the global statistics collector"""
    global _stats_collector
    if _stats_collector:
        _stats_collector.shutdown()
        _stats_collector = None


class LearningDataIntegrator:
    """
    통합 학습 데이터 관리자
    
    StatisticsCollector와 AutoLearningSystem 간 데이터 공유 및
    통합 쿼리 상호작용 메트릭스를 관리합니다.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.stats_collector = get_stats_collector()
        
        # AutoLearningSystem은 필요할 때 lazy initialization
        self._auto_learning_system: Optional[Any] = None
    
    @property
    def auto_learning_system(self):
        """AutoLearningSystem lazy initialization"""
        if self._auto_learning_system is None:
            try:
                from .auto_learning_system import AutoLearningSystem
                self._auto_learning_system = AutoLearningSystem()
                self.logger.info("AutoLearningSystem initialized for integration")
            except ImportError as e:
                self.logger.warning(f"AutoLearningSystem not available: {e}")
                return None
        return self._auto_learning_system
    
    def record_complete_query_interaction(self, metrics: QueryInteractionMetrics) -> None:
        """
        전체 쿼리 상호작용 기록 (통합)
        
        Args:
            metrics: 통합 쿼리 상호작용 메트릭스
        """
        try:
            # 1. StatisticsCollector에 Intent 분류 데이터 기록
            classification_metrics = ClassificationMetrics(
                intent=metrics.intent,
                confidence=metrics.intent_confidence,
                response_time_ms=metrics.response_time_ms,
                timestamp=metrics.timestamp,
                is_error=metrics.is_error
            )
            self.stats_collector.record_classification(
                intent=metrics.intent,
                confidence=metrics.intent_confidence,
                response_time_ms=metrics.response_time_ms,
                is_error=metrics.is_error
            )
            
            # 2. AutoLearningSystem에 쿼리 상호작용 기록
            if self.auto_learning_system:
                success = metrics.execution_success and metrics.validation_passed
                
                self.auto_learning_system.record_query_interaction(
                    user_id=metrics.user_id or "unknown",
                    query=metrics.user_query,
                    mapping_result=metrics.mapping_result or {},
                    confidence=metrics.intent_confidence,
                    success=success,
                    user_feedback=metrics.user_feedback
                )
                
                self.logger.debug(f"Recorded query interaction: {metrics.user_query[:50]}...")
            
        except Exception as e:
            self.logger.error(f"Failed to record complete query interaction: {e}")
    
    def get_unified_insights(self) -> Dict[str, Any]:
        """
        통합 인사이트 조회
        
        Returns:
            StatisticsCollector와 AutoLearningSystem의 통합 인사이트
        """
        try:
            insights = {
                "intent_classification": {},
                "query_learning": {},
                "performance_metrics": {}
            }
            
            # Intent 분류 통계
            stats = self.stats_collector.get_stats()
            insights["intent_classification"] = {
                "total_classifications": stats.total_classifications,
                "average_confidence": stats.average_confidence,
                "error_rate": stats.error_rate,
                "intent_distribution": dict(stats.intent_distribution),
                "confidence_distribution": dict(stats.confidence_distribution),
                "response_times": stats.response_times
            }
            
            # AutoLearning 인사이트
            if self.auto_learning_system:
                learning_report = self.auto_learning_system.get_learning_report()
                insights["query_learning"] = {
                    "total_queries": learning_report.get("learning_metrics", {}).get("total_queries", 0),
                    "success_rate": learning_report.get("learning_metrics", {}).get("success_rate", 0),
                    "total_patterns": learning_report.get("pattern_analysis", {}).get("total_patterns", 0),
                    "avg_confidence": learning_report.get("learning_metrics", {}).get("avg_confidence", 0)
                }
            
            # 성능 메트릭스
            performance = self.stats_collector.get_performance_metrics()
            insights["performance_metrics"] = {
                "queue_size": performance.queue_size,
                "memory_usage_percent": performance.memory_usage_mb,
                "batch_processing_rate": performance.batch_processing_rate,
                "error_count": performance.error_count
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to get unified insights: {e}")
            return {}
    
    def get_query_pattern_analysis(self) -> Dict[str, Any]:
        """
        쿼리 패턴 분석 (통합)
        
        Intent 분류와 AutoLearning 패턴을 결합하여 분석합니다.
        """
        try:
            analysis = {
                "intent_patterns": {},
                "query_patterns": {},
                "correlations": {}
            }
            
            # Intent 분포
            stats = self.stats_collector.get_stats()
            analysis["intent_patterns"] = {
                "distribution": dict(stats.intent_distribution),
                "confidence_by_intent": {}
            }
            
            # AutoLearning 패턴
            if self.auto_learning_system:
                learning_report = self.auto_learning_system.get_learning_report()
                analysis["query_patterns"] = learning_report.get("pattern_analysis", {})
            
            # 상관관계 분석 (향후 구현)
            # 예: 특정 Intent의 쿼리가 특정 패턴으로 매핑되는 빈도
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to get query pattern analysis: {e}")
            return {}
    
    def optimize_based_on_data(self) -> List[str]:
        """
        통합 데이터 기반 최적화 제안
        
        Returns:
            최적화 제안 리스트
        """
        suggestions = []
        
        try:
            # Intent 분류 최적화 제안
            stats = self.stats_collector.get_stats()
            if stats.error_rate > 10.0:
                suggestions.append(
                    f"Intent 분류 오류율이 {stats.error_rate:.1f}%로 높습니다. "
                    "모델 튜닝 또는 추가 학습 데이터 수집을 고려하세요."
                )
            
            if stats.average_confidence < 0.6:
                suggestions.append(
                    f"평균 신뢰도가 {stats.average_confidence:.2f}로 낮습니다. "
                    "프롬프트 개선 또는 모델 업그레이드를 고려하세요."
                )
            
            # AutoLearning 최적화 제안
            if self.auto_learning_system:
                learning_report = self.auto_learning_system.get_learning_report()
                learning_suggestions = learning_report.get("improvement_suggestions", [])
                suggestions.extend(learning_suggestions)
            
        except Exception as e:
            self.logger.error(f"Failed to generate optimization suggestions: {e}")
        
        return suggestions


# Global integrator instance
_integrator: Optional[LearningDataIntegrator] = None


def get_integrator() -> LearningDataIntegrator:
    """Get the global learning data integrator instance"""
    global _integrator
    if _integrator is None:
        _integrator = LearningDataIntegrator()
    return _integrator
