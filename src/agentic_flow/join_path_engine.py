#!/usr/bin/env python3
"""
그래프 기반 조인 경로 탐색 엔진
복잡한 다중 테이블 조인 경로를 효율적으로 탐색하고 최적화하는 시스템
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from collections import defaultdict, deque
import heapq
import itertools

logger = logging.getLogger(__name__)


class PathOptimizationStrategy(Enum):
    """조인 경로 최적화 전략"""
    SHORTEST_PATH = "shortest_path"
    HIGHEST_CONFIDENCE = "highest_confidence"
    LOWEST_COST = "lowest_cost"
    BALANCED = "balanced"


@dataclass
class JoinPathOption:
    """조인 경로 옵션"""
    path: List[str]  # 테이블 경로
    relationships: List[Any]  # 관계 정보
    total_cost: float
    confidence: float
    estimated_rows: int
    join_complexity: int
    optimization_score: float


@dataclass
class MultiTableJoinRequest:
    """다중 테이블 조인 요청"""
    source_tables: List[str]
    target_tables: List[str]
    required_columns: List[str]
    filters: Dict[str, Any]
    optimization_strategy: PathOptimizationStrategy
    max_depth: int = 5
    max_paths: int = 10


class JoinPathEngine:
    """
    그래프 기반 조인 경로 탐색 및 최적화 엔진
    """
    
    def __init__(self, relationship_detector):
        """
        JoinPathEngine 초기화
        
        Args:
            relationship_detector: RelationshipDetector 인스턴스
        """
        self.relationship_detector = relationship_detector
        self.path_cache = {}
        self.optimization_cache = {}
        self.performance_metrics = {
            "path_searches": 0,
            "cache_hits": 0,
            "total_search_time": 0.0,
            "average_path_length": 0.0
        }
        
        # 조인 비용 계산 가중치
        self.cost_weights = {
            "relationship_confidence": 0.3,
            "table_size_factor": 0.2,
            "join_complexity": 0.2,
            "path_length": 0.2,
            "index_availability": 0.1
        }
        
        logger.info("JoinPathEngine initialized")
    
    def find_optimal_join_paths(self, request: MultiTableJoinRequest) -> List[JoinPathOption]:
        """
        다중 테이블 간의 최적 조인 경로들을 탐색
        
        Args:
            request: 조인 요청 정보
            
        Returns:
            List[JoinPathOption]: 최적화된 조인 경로 옵션들
        """
        try:
            start_time = time.time()
            self.performance_metrics["path_searches"] += 1
            
            logger.info(f"Finding optimal join paths for {len(request.source_tables)} sources to {len(request.target_tables)} targets")
            
            # 캐시 확인
            cache_key = self._generate_cache_key(request)
            if cache_key in self.path_cache:
                self.performance_metrics["cache_hits"] += 1
                logger.debug("Join paths retrieved from cache")
                return self.path_cache[cache_key]
            
            # 모든 가능한 조인 경로 탐색
            all_paths = self._explore_all_join_paths(request)
            
            # 경로 최적화
            optimized_paths = self._optimize_join_paths(all_paths, request.optimization_strategy)
            
            # 결과 제한
            final_paths = optimized_paths[:request.max_paths]
            
            # 캐시에 저장
            self.path_cache[cache_key] = final_paths
            
            # 성능 메트릭 업데이트
            search_time = time.time() - start_time
            self.performance_metrics["total_search_time"] += search_time
            if final_paths:
                avg_length = sum(len(path.path) for path in final_paths) / len(final_paths)
                self.performance_metrics["average_path_length"] = avg_length
            
            logger.info(f"Found {len(final_paths)} optimal join paths in {search_time:.3f}s")
            return final_paths
            
        except Exception as e:
            logger.error(f"Failed to find optimal join paths: {str(e)}", exc_info=True)
            return []
    
    def _explore_all_join_paths(self, request: MultiTableJoinRequest) -> List[JoinPathOption]:
        """모든 가능한 조인 경로 탐색"""
        all_paths = []
        
        try:
            # 소스 테이블과 타겟 테이블 간의 모든 조합 탐색
            for source_table in request.source_tables:
                for target_table in request.target_tables:
                    # 직접 경로 탐색
                    direct_paths = self._find_direct_paths(source_table, target_table, request.max_depth)
                    all_paths.extend(direct_paths)
                    
                    # 중간 테이블을 통한 경로 탐색
                    intermediate_paths = self._find_intermediate_paths(
                        source_table, target_table, request.max_depth
                    )
                    all_paths.extend(intermediate_paths)
            
            # 다중 소스/타겟 조합 경로 탐색
            if len(request.source_tables) > 1 or len(request.target_tables) > 1:
                multi_paths = self._find_multi_table_paths(request)
                all_paths.extend(multi_paths)
            
            logger.debug(f"Explored {len(all_paths)} total join paths")
            return all_paths
            
        except Exception as e:
            logger.error(f"Failed to explore join paths: {str(e)}")
            return []
    
    def _find_direct_paths(self, source_table: str, target_table: str, max_depth: int) -> List[JoinPathOption]:
        """두 테이블 간의 직접 경로 탐색"""
        paths = []
        
        try:
            # NetworkX를 사용한 최단 경로 탐색
            graph = self.relationship_detector.relationship_graph
            
            if source_table not in graph.nodes() or target_table not in graph.nodes():
                logger.debug(f"Tables not in graph: {source_table} or {target_table}")
                return paths
            
            # 연결성 확인 (양방향)
            has_forward_path = nx.has_path(graph, source_table, target_table)
            has_backward_path = nx.has_path(graph, target_table, source_table)
            
            if not (has_forward_path or has_backward_path):
                logger.debug(f"No path exists between {source_table} and {target_table}")
                return paths
            
            if not has_forward_path:
                logger.debug(f"No forward path from {source_table} to {target_table}, but backward path exists")
                # 역방향 경로가 있으면 경로를 뒤집어서 처리
                try:
                    backward_path = nx.shortest_path(graph, target_table, source_table)
                    path_nodes = backward_path[::-1]  # 경로 뒤집기
                    
                    if len(path_nodes) - 1 <= max_depth:
                        join_path_option = self._create_join_path_option(path_nodes)
                        if join_path_option:
                            join_path_option.join_complexity = len(path_nodes) - 1
                            paths.append(join_path_option)
                            logger.debug(f"Found reversed path: {' -> '.join(path_nodes)}")
                except nx.NetworkXNoPath:
                    pass
                return paths
            
            # 다양한 알고리즘으로 경로 탐색
            path_algorithms = [
                ("shortest", nx.shortest_path),
                ("dijkstra", lambda g, s, t: nx.dijkstra_path(g, s, t, weight='weight')),
            ]
            
            for algo_name, path_func in path_algorithms:
                try:
                    path_nodes = path_func(graph, source_table, target_table)
                    
                    if len(path_nodes) - 1 <= max_depth:
                        join_path_option = self._create_join_path_option(path_nodes)
                        if join_path_option:
                            join_path_option.join_complexity = len(path_nodes) - 1
                            paths.append(join_path_option)
                            logger.debug(f"Found path via {algo_name}: {' -> '.join(path_nodes)}")
                            
                except nx.NetworkXNoPath:
                    logger.debug(f"No path found via {algo_name}")
                    continue
                except Exception as e:
                    logger.debug(f"Path algorithm {algo_name} failed: {str(e)}")
                    continue
            
            # 경로가 없으면 중간 테이블을 통한 경로 탐색
            if not paths:
                logger.debug(f"No direct paths found, searching intermediate paths")
                intermediate_paths = self._find_intermediate_paths(source_table, target_table, max_depth)
                paths.extend(intermediate_paths)
            
            return paths
            
        except Exception as e:
            logger.error(f"Failed to find direct paths: {str(e)}")
            return []
    
    def _find_intermediate_paths(self, source_table: str, target_table: str, max_depth: int) -> List[JoinPathOption]:
        """중간 테이블을 통한 경로 탐색"""
        paths = []
        
        try:
            graph = self.relationship_detector.relationship_graph
            
            # 모든 가능한 중간 테이블 탐색 (제한적으로)
            all_nodes = list(graph.nodes())
            
            # 소스와 타겟의 이웃 노드들을 우선적으로 확인
            source_neighbors = set(graph.neighbors(source_table))
            target_neighbors = set(graph.neighbors(target_table))
            
            # 공통 이웃을 우선적으로 시도
            common_neighbors = source_neighbors.intersection(target_neighbors)
            if common_neighbors:
                logger.debug(f"Found common neighbors: {common_neighbors}")
            
            # 우선순위 테이블 목록 (공통 이웃 + 각각의 이웃)
            priority_tables = list(common_neighbors) + list(source_neighbors) + list(target_neighbors)
            priority_tables = list(set(priority_tables))  # 중복 제거
            
            # 우선순위 테이블부터 탐색
            for intermediate_table in priority_tables[:20]:  # 상위 20개만 시도
                if intermediate_table in [source_table, target_table]:
                    continue
                
                try:
                    # source -> intermediate 경로 탐색
                    try:
                        path1 = nx.shortest_path(graph, source_table, intermediate_table)
                    except nx.NetworkXNoPath:
                        # 역방향 경로 시도
                        try:
                            backward_path1 = nx.shortest_path(graph, intermediate_table, source_table)
                            path1 = backward_path1[::-1]
                        except nx.NetworkXNoPath:
                            continue
                    
                    # intermediate -> target 경로 탐색
                    try:
                        path2 = nx.shortest_path(graph, intermediate_table, target_table)
                    except nx.NetworkXNoPath:
                        # 역방향 경로 시도
                        try:
                            backward_path2 = nx.shortest_path(graph, target_table, intermediate_table)
                            path2 = backward_path2[::-1]
                        except nx.NetworkXNoPath:
                            continue
                    
                    # 경로 결합 (중간 테이블 중복 제거)
                    combined_path = path1 + path2[1:]
                    
                    if len(combined_path) - 1 <= max_depth:
                        join_path_option = self._create_join_path_option(combined_path)
                        if join_path_option:
                            join_path_option.join_complexity = len(combined_path) - 1
                            paths.append(join_path_option)
                            logger.debug(f"Found intermediate path: {' -> '.join(combined_path)}")
                            
                except Exception as e:
                    logger.debug(f"Intermediate path failed for {intermediate_table}: {str(e)}")
                    continue
            
            # 경로가 여전히 없으면 나머지 테이블들도 시도
            if not paths:
                remaining_tables = [t for t in all_nodes if t not in priority_tables and t not in [source_table, target_table]]
                for intermediate_table in remaining_tables[:10]:  # 최대 10개 더 시도
                    try:
                        path1 = nx.shortest_path(graph, source_table, intermediate_table)
                        path2 = nx.shortest_path(graph, intermediate_table, target_table)
                        combined_path = path1 + path2[1:]
                        
                        if len(combined_path) - 1 <= max_depth:
                            join_path_option = self._create_join_path_option(combined_path)
                            if join_path_option:
                                join_path_option.join_complexity = len(combined_path) - 1
                                paths.append(join_path_option)
                                
                    except (nx.NetworkXNoPath, Exception):
                        continue
            
            return paths
            
        except Exception as e:
            logger.error(f"Failed to find intermediate paths: {str(e)}")
            return []
    
    def _find_multi_table_paths(self, request: MultiTableJoinRequest) -> List[JoinPathOption]:
        """다중 테이블 조합 경로 탐색"""
        paths = []
        
        try:
            # 모든 소스-타겟 조합에 대해 경로 탐색
            for source_table in request.source_tables:
                for target_table in request.target_tables:
                    # 기존 경로 탐색 로직 사용
                    direct_paths = self._find_direct_paths(source_table, target_table, request.max_depth)
                    paths.extend(direct_paths)
            
            # 중복 제거 및 최적화
            unique_paths = self._remove_duplicate_paths(paths)
            return unique_paths
            
        except Exception as e:
            logger.error(f"Failed to find multi-table paths: {str(e)}")
            return []
    
    def _create_join_path_option(self, path_nodes: List[str]) -> Optional[JoinPathOption]:
        """경로 노드로부터 JoinPathOption 생성"""
        try:
            if len(path_nodes) < 2:
                logger.debug(f"Path too short: {len(path_nodes)} nodes")
                return None
            
            logger.debug(f"Creating join path option for: {' -> '.join(path_nodes)}")
            
            relationships = []
            total_cost = 0.0
            confidence_sum = 0.0
            
            # 경로의 각 단계에서 관계 정보 추출
            for i in range(len(path_nodes) - 1):
                source_table = path_nodes[i]
                target_table = path_nodes[i + 1]
                
                logger.debug(f"Looking for relationship: {source_table} -> {target_table}")
                
                # 관계 정보 찾기
                relationship = self._find_relationship(source_table, target_table)
                if relationship:
                    relationships.append(relationship)
                    total_cost += 1.0 / relationship.confidence  # 신뢰도가 높을수록 비용 낮음
                    confidence_sum += relationship.confidence
                    logger.debug(f"Found relationship: {relationship.source_column} -> {relationship.target_column}, confidence: {relationship.confidence}")
                else:
                    # 관계가 없으면 높은 비용 부여
                    total_cost += 10.0
                    logger.debug(f"No relationship found between {source_table} and {target_table}")
            
            if not relationships:
                logger.debug(f"No relationships found for path: {' -> '.join(path_nodes)}")
                return None
            
            # 평균 신뢰도 계산
            avg_confidence = confidence_sum / len(relationships)
            
            # 예상 행 수 계산 (간단한 추정)
            estimated_rows = self._estimate_join_result_rows(path_nodes)
            
            # 조인 복잡도 계산
            join_complexity = len(path_nodes) - 1
            
            # 최적화 점수 계산
            optimization_score = self._calculate_optimization_score(
                total_cost, avg_confidence, join_complexity, estimated_rows
            )
            
            logger.debug(f"Created join path option: cost={total_cost:.2f}, confidence={avg_confidence:.2f}, complexity={join_complexity}")
            
            return JoinPathOption(
                path=path_nodes,
                relationships=relationships,
                total_cost=total_cost,
                confidence=avg_confidence,
                estimated_rows=estimated_rows,
                join_complexity=join_complexity,
                optimization_score=optimization_score
            )
            
        except Exception as e:
            logger.error(f"Failed to create join path option: {str(e)}", exc_info=True)
            return None
    
    def _find_relationship(self, source_table: str, target_table: str) -> Optional[Any]:
        """두 테이블 간의 관계 정보 찾기"""
        try:
            for rel in self.relationship_detector.relationships:
                if rel.source_table == source_table and rel.target_table == target_table:
                    return rel
            return None
        except Exception as e:
            logger.error(f"Failed to find relationship: {str(e)}")
            return None
    
    def _estimate_join_result_rows(self, path_nodes: List[str]) -> int:
        """조인 결과 예상 행 수 추정"""
        try:
            # 간단한 추정: 각 테이블의 예상 크기를 곱하고 조인 선택도로 나눔
            estimated_rows = 1000  # 기본값
            
            for i in range(len(path_nodes) - 1):
                # 각 조인 단계에서 선택도 적용 (간단한 추정)
                estimated_rows *= 0.1  # 10% 선택도 가정
            
            return max(estimated_rows, 1)
        except Exception as e:
            logger.error(f"Failed to estimate join result rows: {str(e)}")
            return 1000
    
    def _calculate_optimization_score(self, cost: float, confidence: float, 
                                    complexity: int, estimated_rows: int) -> float:
        """최적화 점수 계산"""
        try:
            # 비용이 낮을수록, 신뢰도가 높을수록, 복잡도가 낮을수록, 예상 행 수가 적을수록 좋음
            score = (
                (1.0 / (cost + 1)) * self.cost_weights["relationship_confidence"] +
                confidence * self.cost_weights["table_size_factor"] +
                (1.0 / (complexity + 1)) * self.cost_weights["join_complexity"] +
                (1.0 / (estimated_rows + 1)) * self.cost_weights["path_length"]
            )
            
            return score
        except Exception as e:
            logger.error(f"Failed to calculate optimization score: {str(e)}")
            return 0.0
    
    def _optimize_join_paths(self, paths: List[JoinPathOption], 
                           strategy: PathOptimizationStrategy) -> List[JoinPathOption]:
        """조인 경로 최적화"""
        try:
            if not paths:
                return []
            
            # 전략에 따른 정렬
            if strategy == PathOptimizationStrategy.SHORTEST_PATH:
                sorted_paths = sorted(paths, key=lambda p: len(p.path))
            elif strategy == PathOptimizationStrategy.HIGHEST_CONFIDENCE:
                sorted_paths = sorted(paths, key=lambda p: p.confidence, reverse=True)
            elif strategy == PathOptimizationStrategy.LOWEST_COST:
                sorted_paths = sorted(paths, key=lambda p: p.total_cost)
            elif strategy == PathOptimizationStrategy.BALANCED:
                sorted_paths = sorted(paths, key=lambda p: p.optimization_score, reverse=True)
            else:
                sorted_paths = paths
            
            return sorted_paths
            
        except Exception as e:
            logger.error(f"Failed to optimize join paths: {str(e)}")
            return paths
    
    def _remove_duplicate_paths(self, paths: List[JoinPathOption]) -> List[JoinPathOption]:
        """중복 경로 제거"""
        try:
            seen_paths = set()
            unique_paths = []
            
            for path in paths:
                path_key = tuple(path.path)
                if path_key not in seen_paths:
                    seen_paths.add(path_key)
                    unique_paths.append(path)
            
            return unique_paths
        except Exception as e:
            logger.error(f"Failed to remove duplicate paths: {str(e)}")
            return paths
    
    def _generate_cache_key(self, request: MultiTableJoinRequest) -> str:
        """캐시 키 생성"""
        try:
            key_components = [
                ",".join(sorted(request.source_tables)),
                ",".join(sorted(request.target_tables)),
                str(request.max_depth),
                str(request.optimization_strategy.value)
            ]
            return "|".join(key_components)
        except Exception as e:
            logger.error(f"Failed to generate cache key: {str(e)}")
            return str(time.time())
    
    def generate_sql_from_path(self, path_option: JoinPathOption, 
                             required_columns: List[str]) -> str:
        """조인 경로로부터 SQL 쿼리 생성"""
        try:
            if not path_option.path or len(path_option.path) < 2:
                return ""
            
            # SELECT 절 생성
            select_clause = self._build_select_clause(path_option.path, required_columns)
            
            # FROM 절 생성
            from_clause = self._build_from_clause(path_option.path)
            
            # JOIN 절 생성
            join_clause = self._build_join_clause(path_option.relationships)
            
            # SQL 조합
            sql = f"{select_clause}\n{from_clause}\n{join_clause}"
            
            return sql
            
        except Exception as e:
            logger.error(f"Failed to generate SQL from path: {str(e)}")
            return ""
    
    def _build_select_clause(self, path: List[str], required_columns: List[str]) -> str:
        """SELECT 절 생성"""
        try:
            if required_columns:
                columns = ", ".join(required_columns)
            else:
                # 모든 테이블에서 모든 컬럼 선택 (실제로는 스키마 정보 필요)
                columns = "*"
            
            return f"SELECT {columns}"
        except Exception as e:
            logger.error(f"Failed to build SELECT clause: {str(e)}")
            return "SELECT *"
    
    def _build_from_clause(self, path: List[str]) -> str:
        """FROM 절 생성"""
        try:
            if not path:
                return "FROM unknown_table"
            
            return f"FROM {path[0]}"
        except Exception as e:
            logger.error(f"Failed to build FROM clause: {str(e)}")
            return "FROM unknown_table"
    
    def _build_join_clause(self, relationships: List[Any]) -> str:
        """JOIN 절 생성"""
        try:
            join_clauses = []
            
            for rel in relationships:
                join_condition = rel.join_condition or f"{rel.source_table}.{rel.source_column} = {rel.target_table}.{rel.target_column}"
                join_clause = f"JOIN {rel.target_table} ON {join_condition}"
                join_clauses.append(join_clause)
            
            return "\n".join(join_clauses)
        except Exception as e:
            logger.error(f"Failed to build JOIN clause: {str(e)}")
            return ""
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 반환"""
        try:
            metrics = self.performance_metrics.copy()
            
            if metrics["path_searches"] > 0:
                metrics["average_search_time"] = metrics["total_search_time"] / metrics["path_searches"]
                metrics["cache_hit_rate"] = metrics["cache_hits"] / metrics["path_searches"]
            else:
                metrics["average_search_time"] = 0.0
                metrics["cache_hit_rate"] = 0.0
            
            metrics["cache_size"] = len(self.path_cache)
            metrics["optimization_cache_size"] = len(self.optimization_cache)
            
            return metrics
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {str(e)}")
            return {}
    
    def clear_cache(self):
        """캐시 정리"""
        try:
            self.path_cache.clear()
            self.optimization_cache.clear()
            logger.info("Join path engine cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
    
    def optimize_for_query(self, query_context: Dict[str, Any]) -> Dict[str, Any]:
        """특정 쿼리 컨텍스트에 대한 최적화 제안"""
        try:
            suggestions = {
                "recommended_strategy": PathOptimizationStrategy.BALANCED,
                "estimated_cost": 0.0,
                "confidence": 0.0,
                "alternative_paths": []
            }
            
            # 쿼리 컨텍스트 분석
            if query_context.get("performance_critical", False):
                suggestions["recommended_strategy"] = PathOptimizationStrategy.LOWEST_COST
            elif query_context.get("accuracy_critical", False):
                suggestions["recommended_strategy"] = PathOptimizationStrategy.HIGHEST_CONFIDENCE
            elif query_context.get("simple_query", False):
                suggestions["recommended_strategy"] = PathOptimizationStrategy.SHORTEST_PATH
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to optimize for query: {str(e)}")
            return {}
