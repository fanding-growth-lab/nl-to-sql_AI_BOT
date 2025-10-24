#!/usr/bin/env python3
"""
최적 조인 경로 생성 및 성능 평가 시스템
조인 경로의 성능을 분석하고 최적화하는 고급 시스템
"""

import logging
import time
import statistics
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from collections import defaultdict, deque
import heapq
import json
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class OptimizationMetric(Enum):
    """최적화 메트릭 타입"""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    IO_COST = "io_cost"
    CPU_COST = "cpu_cost"
    NETWORK_COST = "network_cost"
    SCALABILITY = "scalability"


class JoinStrategy(Enum):
    """조인 전략"""
    NESTED_LOOP = "nested_loop"
    HASH_JOIN = "hash_join"
    MERGE_JOIN = "merge_join"
    SORT_MERGE = "sort_merge"
    INDEX_JOIN = "index_join"


@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    execution_time: float = 0.0
    memory_usage: int = 0
    io_cost: float = 0.0
    cpu_cost: float = 0.0
    network_cost: float = 0.0
    scalability_score: float = 0.0
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationResult:
    """최적화 결과"""
    original_path: Any
    optimized_path: Any
    improvement_ratio: float
    performance_gain: Dict[str, float]
    optimization_strategy: str
    confidence: float
    recommendations: List[str]


@dataclass
class BenchmarkResult:
    """벤치마크 결과"""
    path_id: str
    metrics: PerformanceMetrics
    query_complexity: int
    data_volume: int
    execution_success: bool
    error_message: Optional[str] = None


class JoinOptimizer:
    """
    최적 조인 경로 생성 및 성능 평가 시스템
    """
    
    def __init__(self, join_path_engine):
        """
        JoinOptimizer 초기화
        
        Args:
            join_path_engine: JoinPathEngine 인스턴스
        """
        self.join_path_engine = join_path_engine
        self.performance_history = {}
        self.benchmark_results = []
        self.optimization_cache = {}
        self.metrics_collector = PerformanceMetricsCollector()
        self.strategy_analyzer = JoinStrategyAnalyzer()
        
        # 성능 임계값 설정
        self.performance_thresholds = {
            "execution_time": 1.0,  # 1초
            "memory_usage": 100 * 1024 * 1024,  # 100MB
            "io_cost": 1000.0,
            "cpu_cost": 500.0
        }
        
        # 최적화 가중치
        self.optimization_weights = {
            OptimizationMetric.EXECUTION_TIME: 0.4,
            OptimizationMetric.MEMORY_USAGE: 0.2,
            OptimizationMetric.IO_COST: 0.2,
            OptimizationMetric.CPU_COST: 0.1,
            OptimizationMetric.SCALABILITY: 0.1
        }
        
        logger.info("JoinOptimizer initialized")
    
    def optimize_join_paths(self, join_request: Any, 
                          optimization_goals: List[OptimizationMetric] = None) -> List[OptimizationResult]:
        """
        조인 경로 최적화
        
        Args:
            join_request: 조인 요청
            optimization_goals: 최적화 목표 메트릭들
            
        Returns:
            List[OptimizationResult]: 최적화 결과들
        """
        try:
            logger.info(f"Starting join path optimization for {len(optimization_goals or [])} goals")
            
            if optimization_goals is None:
                optimization_goals = [OptimizationMetric.EXECUTION_TIME]
            
            # 기본 조인 경로 탐색
            original_paths = self.join_path_engine.find_optimal_join_paths(join_request)
            logger.info(f"Found {len(original_paths)} original join paths")
            
            if not original_paths:
                logger.warning("No join paths found for optimization")
                return []
            
            # 각 경로에 대해 최적화 수행
            optimization_results = []
            for path in original_paths:
                try:
                    optimized_result = self._optimize_single_path(path, optimization_goals)
                    if optimized_result:
                        optimization_results.append(optimized_result)
                except Exception as e:
                    logger.error(f"Failed to optimize path {path}: {str(e)}")
                    continue
            
            # 최적화 결과 정렬 및 필터링
            optimization_results = self._rank_optimization_results(optimization_results, optimization_goals)
            
            logger.info(f"Generated {len(optimization_results)} optimization results")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Failed to optimize join paths: {str(e)}", exc_info=True)
            return []
    
    def _optimize_single_path(self, path: Any, optimization_goals: List[OptimizationMetric]) -> Optional[OptimizationResult]:
        """단일 경로 최적화"""
        try:
            path_id = f"{path.path[0]}_{path.path[-1]}_{hash(tuple(path.path))}"
            
            # 성능 메트릭 수집
            original_metrics = self._collect_performance_metrics(path)
            
            # 최적화 전략 적용
            optimization_strategies = self._get_optimization_strategies(path, optimization_goals)
            
            best_optimization = None
            best_improvement = 0.0
            
            for strategy in optimization_strategies:
                try:
                    optimized_path = self._apply_optimization_strategy(path, strategy)
                    if optimized_path:
                        optimized_metrics = self._collect_performance_metrics(optimized_path)
                        improvement = self._calculate_improvement(original_metrics, optimized_metrics, optimization_goals)
                        
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_optimization = OptimizationResult(
                                original_path=path,
                                optimized_path=optimized_path,
                                improvement_ratio=improvement,
                                performance_gain=self._calculate_performance_gain(original_metrics, optimized_metrics),
                                optimization_strategy=strategy,
                                confidence=self._calculate_optimization_confidence(original_metrics, optimized_metrics),
                                recommendations=self._generate_recommendations(optimized_path, optimized_metrics)
                            )
                except Exception as e:
                    logger.debug(f"Optimization strategy {strategy} failed: {str(e)}")
                    continue
            
            return best_optimization
            
        except Exception as e:
            logger.error(f"Failed to optimize single path: {str(e)}")
            return None
    
    def _collect_performance_metrics(self, path: Any) -> PerformanceMetrics:
        """성능 메트릭 수집"""
        try:
            # 기본 메트릭 계산
            execution_time = self._estimate_execution_time(path)
            memory_usage = self._estimate_memory_usage(path)
            io_cost = self._estimate_io_cost(path)
            cpu_cost = self._estimate_cpu_cost(path)
            scalability_score = self._estimate_scalability(path)
            
            # 신뢰도 계산
            confidence = self._calculate_metrics_confidence(path)
            
            return PerformanceMetrics(
                execution_time=execution_time,
                memory_usage=memory_usage,
                io_cost=io_cost,
                cpu_cost=cpu_cost,
                network_cost=0.0,  # 로컬 DB이므로 0
                scalability_score=scalability_score,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {str(e)}")
            return PerformanceMetrics()
    
    def _estimate_execution_time(self, path: Any) -> float:
        """실행 시간 추정"""
        try:
            base_time = 0.1  # 기본 시간 (100ms)
            
            # 조인 복잡도에 따른 시간 증가
            complexity_factor = path.join_complexity * 0.05
            
            # 신뢰도에 따른 시간 감소
            confidence_factor = 1.0 - (path.confidence * 0.3)
            
            # 예상 행 수에 따른 시간 증가
            rows_factor = min(path.estimated_rows / 10000, 2.0)  # 최대 2배
            
            estimated_time = base_time * (1 + complexity_factor) * confidence_factor * (1 + rows_factor)
            
            return estimated_time
            
        except Exception as e:
            logger.error(f"Failed to estimate execution time: {str(e)}")
            return 1.0
    
    def _estimate_memory_usage(self, path: Any) -> int:
        """메모리 사용량 추정 (바이트)"""
        try:
            base_memory = 1024 * 1024  # 1MB 기본
            
            # 조인 복잡도에 따른 메모리 증가
            complexity_factor = path.join_complexity * 0.5
            
            # 예상 행 수에 따른 메모리 증가
            rows_factor = path.estimated_rows * 100  # 행당 100바이트 가정
            
            estimated_memory = int(base_memory * (1 + complexity_factor) + rows_factor)
            
            return estimated_memory
            
        except Exception as e:
            logger.error(f"Failed to estimate memory usage: {str(e)}")
            return 10 * 1024 * 1024  # 10MB 기본값
    
    def _estimate_io_cost(self, path: Any) -> float:
        """I/O 비용 추정"""
        try:
            base_io_cost = 10.0
            
            # 조인 복잡도에 따른 I/O 증가
            complexity_factor = path.join_complexity * 2.0
            
            # 예상 행 수에 따른 I/O 증가
            rows_factor = path.estimated_rows / 1000
            
            estimated_io_cost = base_io_cost * (1 + complexity_factor) + rows_factor
            
            return estimated_io_cost
            
        except Exception as e:
            logger.error(f"Failed to estimate I/O cost: {str(e)}")
            return 100.0
    
    def _estimate_cpu_cost(self, path: Any) -> float:
        """CPU 비용 추정"""
        try:
            base_cpu_cost = 5.0
            
            # 조인 복잡도에 따른 CPU 증가
            complexity_factor = path.join_complexity * 1.5
            
            # 신뢰도에 따른 CPU 감소 (더 효율적인 조인)
            confidence_factor = 1.0 - (path.confidence * 0.2)
            
            estimated_cpu_cost = base_cpu_cost * (1 + complexity_factor) * confidence_factor
            
            return estimated_cpu_cost
            
        except Exception as e:
            logger.error(f"Failed to estimate CPU cost: {str(e)}")
            return 50.0
    
    def _estimate_scalability(self, path: Any) -> float:
        """확장성 점수 추정 (0-1)"""
        try:
            base_scalability = 0.8
            
            # 조인 복잡도가 높을수록 확장성 감소
            complexity_penalty = path.join_complexity * 0.1
            
            # 신뢰도가 높을수록 확장성 증가
            confidence_bonus = path.confidence * 0.2
            
            # 예상 행 수가 적을수록 확장성 증가
            rows_penalty = min(path.estimated_rows / 100000, 0.3)
            
            scalability_score = base_scalability - complexity_penalty + confidence_bonus - rows_penalty
            
            return max(0.0, min(1.0, scalability_score))
            
        except Exception as e:
            logger.error(f"Failed to estimate scalability: {str(e)}")
            return 0.5
    
    def _calculate_metrics_confidence(self, path: Any) -> float:
        """메트릭 신뢰도 계산"""
        try:
            # 기본 신뢰도는 경로의 신뢰도 기반
            base_confidence = path.confidence
            
            # 관계 수가 많을수록 신뢰도 증가
            relationship_bonus = min(len(path.relationships) * 0.05, 0.2)
            
            # 복잡도가 적당할수록 신뢰도 증가
            complexity_factor = 1.0 - min(path.join_complexity * 0.05, 0.3)
            
            confidence = base_confidence + relationship_bonus
            confidence *= complexity_factor
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics confidence: {str(e)}")
            return 0.5
    
    def _get_optimization_strategies(self, path: Any, optimization_goals: List[OptimizationMetric]) -> List[str]:
        """최적화 전략 목록 생성"""
        strategies = []
        
        try:
            # 실행 시간 최적화
            if OptimizationMetric.EXECUTION_TIME in optimization_goals:
                strategies.extend([
                    "index_optimization",
                    "join_order_optimization",
                    "predicate_pushdown",
                    "materialized_view_usage"
                ])
            
            # 메모리 사용량 최적화
            if OptimizationMetric.MEMORY_USAGE in optimization_goals:
                strategies.extend([
                    "streaming_processing",
                    "memory_efficient_joins",
                    "temporary_table_optimization"
                ])
            
            # I/O 비용 최적화
            if OptimizationMetric.IO_COST in optimization_goals:
                strategies.extend([
                    "batch_processing",
                    "sequential_scan_optimization",
                    "buffer_pool_optimization"
                ])
            
            # CPU 비용 최적화
            if OptimizationMetric.CPU_COST in optimization_goals:
                strategies.extend([
                    "parallel_processing",
                    "cpu_efficient_algorithms",
                    "computation_optimization"
                ])
            
            # 확장성 최적화
            if OptimizationMetric.SCALABILITY in optimization_goals:
                strategies.extend([
                    "horizontal_scaling",
                    "distributed_processing",
                    "load_balancing"
                ])
            
            return strategies
            
        except Exception as e:
            logger.error(f"Failed to get optimization strategies: {str(e)}")
            return ["default_optimization"]
    
    def _apply_optimization_strategy(self, path: Any, strategy: str) -> Optional[Any]:
        """최적화 전략 적용"""
        try:
            if strategy == "index_optimization":
                return self._apply_index_optimization(path)
            elif strategy == "join_order_optimization":
                return self._apply_join_order_optimization(path)
            elif strategy == "predicate_pushdown":
                return self._apply_predicate_pushdown(path)
            elif strategy == "materialized_view_usage":
                return self._apply_materialized_view_optimization(path)
            elif strategy == "streaming_processing":
                return self._apply_streaming_optimization(path)
            elif strategy == "memory_efficient_joins":
                return self._apply_memory_efficient_joins(path)
            elif strategy == "parallel_processing":
                return self._apply_parallel_processing(path)
            else:
                return self._apply_default_optimization(path)
                
        except Exception as e:
            logger.error(f"Failed to apply optimization strategy {strategy}: {str(e)}")
            return None
    
    def _apply_index_optimization(self, path: Any) -> Any:
        """인덱스 최적화 적용"""
        try:
            # 인덱스 최적화된 경로 생성 (실제로는 복사본)
            optimized_path = self._copy_path_option(path)
            
            # 인덱스 사용으로 인한 성능 개선 시뮬레이션
            optimized_path.total_cost *= 0.7  # 30% 비용 감소
            optimized_path.confidence += 0.1  # 신뢰도 증가
            
            return optimized_path
            
        except Exception as e:
            logger.error(f"Failed to apply index optimization: {str(e)}")
            return None
    
    def _apply_join_order_optimization(self, path: Any) -> Any:
        """조인 순서 최적화 적용"""
        try:
            # 조인 순서 최적화된 경로 생성
            optimized_path = self._copy_path_option(path)
            
            # 작은 테이블부터 조인하는 최적화
            optimized_path.total_cost *= 0.8  # 20% 비용 감소
            optimized_path.confidence += 0.05  # 신뢰도 약간 증가
            
            return optimized_path
            
        except Exception as e:
            logger.error(f"Failed to apply join order optimization: {str(e)}")
            return None
    
    def _apply_predicate_pushdown(self, path: Any) -> Any:
        """조건 푸시다운 최적화 적용"""
        try:
            optimized_path = self._copy_path_option(path)
            
            # 조건 푸시다운으로 인한 성능 개선
            optimized_path.total_cost *= 0.85  # 15% 비용 감소
            optimized_path.estimated_rows *= 0.5  # 행 수 50% 감소
            
            return optimized_path
            
        except Exception as e:
            logger.error(f"Failed to apply predicate pushdown: {str(e)}")
            return None
    
    def _apply_materialized_view_optimization(self, path: Any) -> Any:
        """물리화된 뷰 최적화 적용"""
        try:
            optimized_path = self._copy_path_option(path)
            
            # 물리화된 뷰 사용으로 인한 성능 개선
            optimized_path.total_cost *= 0.6  # 40% 비용 감소
            optimized_path.confidence += 0.15  # 신뢰도 증가
            
            return optimized_path
            
        except Exception as e:
            logger.error(f"Failed to apply materialized view optimization: {str(e)}")
            return None
    
    def _apply_streaming_optimization(self, path: Any) -> Any:
        """스트리밍 처리 최적화 적용"""
        try:
            optimized_path = self._copy_path_option(path)
            
            # 스트리밍 처리로 메모리 사용량 감소
            optimized_path.total_cost *= 0.9  # 10% 비용 감소
            
            return optimized_path
            
        except Exception as e:
            logger.error(f"Failed to apply streaming optimization: {str(e)}")
            return None
    
    def _apply_memory_efficient_joins(self, path: Any) -> Any:
        """메모리 효율적 조인 최적화 적용"""
        try:
            optimized_path = self._copy_path_option(path)
            
            # 메모리 효율적인 조인으로 성능 개선
            optimized_path.total_cost *= 0.75  # 25% 비용 감소
            
            return optimized_path
            
        except Exception as e:
            logger.error(f"Failed to apply memory efficient joins: {str(e)}")
            return None
    
    def _apply_parallel_processing(self, path: Any) -> Any:
        """병렬 처리 최적화 적용"""
        try:
            optimized_path = self._copy_path_option(path)
            
            # 병렬 처리로 성능 개선
            optimized_path.total_cost *= 0.5  # 50% 비용 감소
            optimized_path.confidence += 0.1  # 신뢰도 증가
            
            return optimized_path
            
        except Exception as e:
            logger.error(f"Failed to apply parallel processing: {str(e)}")
            return None
    
    def _apply_default_optimization(self, path: Any) -> Any:
        """기본 최적화 적용"""
        try:
            optimized_path = self._copy_path_option(path)
            
            # 기본 최적화
            optimized_path.total_cost *= 0.95  # 5% 비용 감소
            optimized_path.confidence += 0.02  # 신뢰도 약간 증가
            
            return optimized_path
            
        except Exception as e:
            logger.error(f"Failed to apply default optimization: {str(e)}")
            return None
    
    def _copy_path_option(self, path: Any) -> Any:
        """경로 옵션 복사"""
        try:
            # 간단한 복사 (실제로는 더 정교한 복사 필요)
            from agentic_flow.join_path_engine import JoinPathOption
            
            return JoinPathOption(
                path=path.path.copy(),
                relationships=path.relationships.copy(),
                total_cost=path.total_cost,
                confidence=path.confidence,
                estimated_rows=path.estimated_rows,
                join_complexity=path.join_complexity,
                optimization_score=path.optimization_score
            )
            
        except Exception as e:
            logger.error(f"Failed to copy path option: {str(e)}")
            return None
    
    def _calculate_improvement(self, original_metrics: PerformanceMetrics, 
                             optimized_metrics: PerformanceMetrics, 
                             optimization_goals: List[OptimizationMetric]) -> float:
        """개선도 계산"""
        try:
            total_improvement = 0.0
            total_weight = 0.0
            
            for goal in optimization_goals:
                weight = self.optimization_weights.get(goal, 0.0)
                if weight > 0:
                    if goal == OptimizationMetric.EXECUTION_TIME:
                        improvement = (original_metrics.execution_time - optimized_metrics.execution_time) / original_metrics.execution_time
                    elif goal == OptimizationMetric.MEMORY_USAGE:
                        improvement = (original_metrics.memory_usage - optimized_metrics.memory_usage) / original_metrics.memory_usage
                    elif goal == OptimizationMetric.IO_COST:
                        improvement = (original_metrics.io_cost - optimized_metrics.io_cost) / original_metrics.io_cost
                    elif goal == OptimizationMetric.CPU_COST:
                        improvement = (original_metrics.cpu_cost - optimized_metrics.cpu_cost) / original_metrics.cpu_cost
                    elif goal == OptimizationMetric.SCALABILITY:
                        improvement = (optimized_metrics.scalability_score - original_metrics.scalability_score)
                    else:
                        improvement = 0.0
                    
                    total_improvement += improvement * weight
                    total_weight += weight
            
            return total_improvement / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate improvement: {str(e)}")
            return 0.0
    
    def _calculate_performance_gain(self, original_metrics: PerformanceMetrics, 
                                  optimized_metrics: PerformanceMetrics) -> Dict[str, float]:
        """성능 향상 계산"""
        try:
            gains = {}
            
            gains["execution_time"] = ((original_metrics.execution_time - optimized_metrics.execution_time) / original_metrics.execution_time) * 100
            gains["memory_usage"] = ((original_metrics.memory_usage - optimized_metrics.memory_usage) / original_metrics.memory_usage) * 100
            gains["io_cost"] = ((original_metrics.io_cost - optimized_metrics.io_cost) / original_metrics.io_cost) * 100
            gains["cpu_cost"] = ((original_metrics.cpu_cost - optimized_metrics.cpu_cost) / original_metrics.cpu_cost) * 100
            gains["scalability"] = (optimized_metrics.scalability_score - original_metrics.scalability_score) * 100
            
            return gains
            
        except Exception as e:
            logger.error(f"Failed to calculate performance gain: {str(e)}")
            return {}
    
    def _calculate_optimization_confidence(self, original_metrics: PerformanceMetrics, 
                                         optimized_metrics: PerformanceMetrics) -> float:
        """최적화 신뢰도 계산"""
        try:
            # 메트릭 개선도와 신뢰도 기반으로 계산
            improvement_confidence = 0.0
            
            if optimized_metrics.execution_time < original_metrics.execution_time:
                improvement_confidence += 0.2
            
            if optimized_metrics.memory_usage < original_metrics.memory_usage:
                improvement_confidence += 0.2
            
            if optimized_metrics.io_cost < original_metrics.io_cost:
                improvement_confidence += 0.2
            
            if optimized_metrics.cpu_cost < original_metrics.cpu_cost:
                improvement_confidence += 0.2
            
            if optimized_metrics.scalability_score > original_metrics.scalability_score:
                improvement_confidence += 0.2
            
            # 신뢰도 가중 평균
            confidence = (original_metrics.confidence + optimized_metrics.confidence) / 2
            
            return min(1.0, improvement_confidence + confidence * 0.5)
            
        except Exception as e:
            logger.error(f"Failed to calculate optimization confidence: {str(e)}")
            return 0.5
    
    def _generate_recommendations(self, optimized_path: Any, metrics: PerformanceMetrics) -> List[str]:
        """최적화 권장사항 생성"""
        try:
            recommendations = []
            
            # 실행 시간 기반 권장사항
            if metrics.execution_time > self.performance_thresholds["execution_time"]:
                recommendations.append("Consider adding indexes on join columns")
                recommendations.append("Optimize join order for better performance")
            
            # 메모리 사용량 기반 권장사항
            if metrics.memory_usage > self.performance_thresholds["memory_usage"]:
                recommendations.append("Use streaming processing to reduce memory usage")
                recommendations.append("Consider partitioning large tables")
            
            # I/O 비용 기반 권장사항
            if metrics.io_cost > self.performance_thresholds["io_cost"]:
                recommendations.append("Optimize disk I/O by using appropriate indexes")
                recommendations.append("Consider using materialized views for complex queries")
            
            # CPU 비용 기반 권장사항
            if metrics.cpu_cost > self.performance_thresholds["cpu_cost"]:
                recommendations.append("Use parallel processing for CPU-intensive operations")
                recommendations.append("Optimize query algorithms for better CPU efficiency")
            
            # 확장성 기반 권장사항
            if metrics.scalability_score < 0.5:
                recommendations.append("Consider horizontal scaling for better scalability")
                recommendations.append("Implement load balancing for distributed processing")
            
            # 일반 권장사항
            if optimized_path.join_complexity > 3:
                recommendations.append("Consider breaking down complex joins into simpler steps")
            
            if optimized_path.estimated_rows > 100000:
                recommendations.append("Consider data archiving for large result sets")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {str(e)}")
            return ["General optimization recommended"]
    
    def _rank_optimization_results(self, results: List[OptimizationResult], 
                                 optimization_goals: List[OptimizationMetric]) -> List[OptimizationResult]:
        """최적화 결과 순위 매기기"""
        try:
            # 개선도와 신뢰도 기반으로 정렬
            def sort_key(result):
                improvement_score = result.improvement_ratio
                confidence_score = result.confidence
                return improvement_score * 0.7 + confidence_score * 0.3
            
            return sorted(results, key=sort_key, reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to rank optimization results: {str(e)}")
            return results
    
    def benchmark_paths(self, paths: List[Any], iterations: int = 3) -> List[BenchmarkResult]:
        """경로 벤치마크 수행"""
        try:
            logger.info(f"Starting benchmark for {len(paths)} paths with {iterations} iterations each")
            
            benchmark_results = []
            
            for i, path in enumerate(paths):
                path_id = f"path_{i}_{hash(tuple(path.path))}"
                
                try:
                    # 여러 번 실행하여 평균 성능 측정
                    execution_times = []
                    memory_usages = []
                    success_count = 0
                    
                    for iteration in range(iterations):
                        try:
                            start_time = time.time()
                            
                            # 실제 쿼리 실행 시뮬레이션 (여기서는 메트릭 수집)
                            metrics = self._collect_performance_metrics(path)
                            
                            execution_time = time.time() - start_time
                            execution_times.append(execution_time)
                            memory_usages.append(metrics.memory_usage)
                            success_count += 1
                            
                        except Exception as e:
                            logger.debug(f"Benchmark iteration {iteration} failed for {path_id}: {str(e)}")
                            continue
                    
                    if success_count > 0:
                        # 평균 성능 계산
                        avg_execution_time = statistics.mean(execution_times)
                        avg_memory_usage = statistics.mean(memory_usages)
                        
                        benchmark_result = BenchmarkResult(
                            path_id=path_id,
                            metrics=PerformanceMetrics(
                                execution_time=avg_execution_time,
                                memory_usage=avg_memory_usage,
                                io_cost=metrics.io_cost,
                                cpu_cost=metrics.cpu_cost,
                                scalability_score=metrics.scalability_score,
                                confidence=metrics.confidence
                            ),
                            query_complexity=path.join_complexity,
                            data_volume=path.estimated_rows,
                            execution_success=True
                        )
                    else:
                        benchmark_result = BenchmarkResult(
                            path_id=path_id,
                            metrics=PerformanceMetrics(),
                            query_complexity=path.join_complexity,
                            data_volume=path.estimated_rows,
                            execution_success=False,
                            error_message="All benchmark iterations failed"
                        )
                    
                    benchmark_results.append(benchmark_result)
                    
                except Exception as e:
                    logger.error(f"Failed to benchmark path {path_id}: {str(e)}")
                    benchmark_results.append(BenchmarkResult(
                        path_id=path_id,
                        metrics=PerformanceMetrics(),
                        query_complexity=0,
                        data_volume=0,
                        execution_success=False,
                        error_message=str(e)
                    ))
            
            self.benchmark_results.extend(benchmark_results)
            logger.info(f"Benchmark completed: {len(benchmark_results)} results")
            
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Failed to benchmark paths: {str(e)}", exc_info=True)
            return []
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        try:
            if not self.benchmark_results:
                return {"message": "No benchmark results available"}
            
            successful_results = [r for r in self.benchmark_results if r.execution_success]
            
            if not successful_results:
                return {"message": "No successful benchmark results"}
            
            stats = {
                "total_benchmarks": len(self.benchmark_results),
                "successful_benchmarks": len(successful_results),
                "success_rate": len(successful_results) / len(self.benchmark_results),
                
                "average_execution_time": statistics.mean([r.metrics.execution_time for r in successful_results]),
                "min_execution_time": min([r.metrics.execution_time for r in successful_results]),
                "max_execution_time": max([r.metrics.execution_time for r in successful_results]),
                
                "average_memory_usage": statistics.mean([r.metrics.memory_usage for r in successful_results]),
                "average_io_cost": statistics.mean([r.metrics.io_cost for r in successful_results]),
                "average_cpu_cost": statistics.mean([r.metrics.cpu_cost for r in successful_results]),
                "average_scalability": statistics.mean([r.metrics.scalability_score for r in successful_results]),
                
                "complexity_distribution": self._calculate_complexity_distribution(successful_results),
                "performance_trends": self._calculate_performance_trends(successful_results)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get performance statistics: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_complexity_distribution(self, results: List[BenchmarkResult]) -> Dict[int, int]:
        """복잡도 분포 계산"""
        try:
            distribution = defaultdict(int)
            for result in results:
                distribution[result.query_complexity] += 1
            return dict(distribution)
        except Exception as e:
            logger.error(f"Failed to calculate complexity distribution: {str(e)}")
            return {}
    
    def _calculate_performance_trends(self, results: List[BenchmarkResult]) -> Dict[str, float]:
        """성능 트렌드 계산"""
        try:
            # 복잡도별 평균 성능
            complexity_performance = defaultdict(list)
            for result in results:
                complexity_performance[result.query_complexity].append(result.metrics.execution_time)
            
            trends = {}
            for complexity, times in complexity_performance.items():
                if len(times) > 1:
                    trends[f"complexity_{complexity}_trend"] = statistics.mean(times)
            
            return trends
            
        except Exception as e:
            logger.error(f"Failed to calculate performance trends: {str(e)}")
            return {}
    
    def clear_cache(self):
        """캐시 정리"""
        try:
            self.optimization_cache.clear()
            self.performance_history.clear()
            logger.info("Join optimizer cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")


class PerformanceMetricsCollector:
    """성능 메트릭 수집기"""
    
    def __init__(self):
        self.metrics_history = []
        self.collection_lock = threading.Lock()
    
    def collect_metrics(self, path: Any, execution_context: Dict[str, Any] = None) -> PerformanceMetrics:
        """메트릭 수집"""
        try:
            with self.collection_lock:
                # 실제 메트릭 수집 로직 (여기서는 시뮬레이션)
                metrics = PerformanceMetrics(
                    execution_time=0.1,
                    memory_usage=1024 * 1024,
                    io_cost=10.0,
                    cpu_cost=5.0,
                    network_cost=0.0,
                    scalability_score=0.8,
                    confidence=0.9
                )
                
                self.metrics_history.append(metrics)
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to collect metrics: {str(e)}")
            return PerformanceMetrics()


class JoinStrategyAnalyzer:
    """조인 전략 분석기"""
    
    def __init__(self):
        self.strategy_performance = {}
    
    def analyze_strategy(self, path: Any, strategy: JoinStrategy) -> Dict[str, Any]:
        """조인 전략 분석"""
        try:
            # 전략별 성능 특성 분석
            analysis = {
                "strategy": strategy.value,
                "estimated_cost": self._estimate_strategy_cost(strategy),
                "memory_requirements": self._estimate_memory_requirements(strategy),
                "scalability": self._estimate_strategy_scalability(strategy),
                "recommended_for": self._get_strategy_recommendations(strategy)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze strategy: {str(e)}")
            return {}
    
    def _estimate_strategy_cost(self, strategy: JoinStrategy) -> float:
        """전략별 비용 추정"""
        cost_map = {
            JoinStrategy.NESTED_LOOP: 10.0,
            JoinStrategy.HASH_JOIN: 5.0,
            JoinStrategy.MERGE_JOIN: 3.0,
            JoinStrategy.SORT_MERGE: 4.0,
            JoinStrategy.INDEX_JOIN: 2.0
        }
        return cost_map.get(strategy, 5.0)
    
    def _estimate_memory_requirements(self, strategy: JoinStrategy) -> int:
        """전략별 메모리 요구량 추정"""
        memory_map = {
            JoinStrategy.NESTED_LOOP: 1024 * 1024,  # 1MB
            JoinStrategy.HASH_JOIN: 10 * 1024 * 1024,  # 10MB
            JoinStrategy.MERGE_JOIN: 5 * 1024 * 1024,  # 5MB
            JoinStrategy.SORT_MERGE: 8 * 1024 * 1024,  # 8MB
            JoinStrategy.INDEX_JOIN: 2 * 1024 * 1024   # 2MB
        }
        return memory_map.get(strategy, 5 * 1024 * 1024)
    
    def _estimate_strategy_scalability(self, strategy: JoinStrategy) -> float:
        """전략별 확장성 추정"""
        scalability_map = {
            JoinStrategy.NESTED_LOOP: 0.3,
            JoinStrategy.HASH_JOIN: 0.7,
            JoinStrategy.MERGE_JOIN: 0.8,
            JoinStrategy.SORT_MERGE: 0.6,
            JoinStrategy.INDEX_JOIN: 0.9
        }
        return scalability_map.get(strategy, 0.5)
    
    def _get_strategy_recommendations(self, strategy: JoinStrategy) -> List[str]:
        """전략별 권장사항"""
        recommendations_map = {
            JoinStrategy.NESTED_LOOP: ["Small tables", "Simple queries"],
            JoinStrategy.HASH_JOIN: ["Medium tables", "Equality joins"],
            JoinStrategy.MERGE_JOIN: ["Large tables", "Sorted data"],
            JoinStrategy.SORT_MERGE: ["Large datasets", "Range queries"],
            JoinStrategy.INDEX_JOIN: ["Indexed columns", "High selectivity"]
        }
        return recommendations_map.get(strategy, ["General use"])


