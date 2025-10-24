#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
복합 조인 경로 최적화 엔진
외래키 관계를 활용하여 최적의 조인 경로를 계산하고 최적화하는 시스템
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import networkx as nx
import heapq
from core.db import get_db_session, execute_query


class JoinType(Enum):
    """조인 유형"""
    INNER = "INNER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FULL = "FULL"


class OptimizationStrategy(Enum):
    """최적화 전략"""
    COST_BASED = "cost_based"  # 비용 기반
    PERFORMANCE_BASED = "performance_based"  # 성능 기반
    SIMPLICITY_BASED = "simplicity_based"  # 단순성 기반


@dataclass
class JoinCondition:
    """조인 조건"""
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    join_type: JoinType = JoinType.INNER
    confidence: float = 1.0


@dataclass
class JoinPath:
    """조인 경로"""
    tables: List[str]
    joins: List[JoinCondition]
    total_cost: float
    confidence: float
    estimated_rows: int = 0
    estimated_time: float = 0.0
    complexity_score: float = 0.0
    description: str = ""


@dataclass
class OptimizationResult:
    """최적화 결과"""
    best_path: JoinPath
    alternative_paths: List[JoinPath]
    optimization_notes: List[str]
    performance_metrics: Dict[str, Any]


class JoinPathOptimizer:
    """조인 경로 최적화 엔진"""
    
    def __init__(self, foreign_key_mapper=None):
        self.logger = logging.getLogger(__name__)
        self.foreign_key_mapper = foreign_key_mapper
        self.join_graph: nx.DiGraph = nx.DiGraph()
        self.table_statistics: Dict[str, Dict[str, Any]] = {}
        self._build_join_graph()
        self._collect_table_statistics()
        
    def _build_join_graph(self):
        """조인 그래프 구성"""
        self.logger.info("조인 그래프 구성 중...")
        
        if not self.foreign_key_mapper:
            self.logger.warning("외래키 매퍼가 없습니다. 기본 그래프를 구성합니다.")
            return
        
        # 신뢰도가 높은 관계만 사용 (성능 개선)
        high_confidence_fks = [fk for fk in self.foreign_key_mapper.foreign_key_mappings 
                              if fk.confidence > 0.7]
        
        self.logger.info(f"고신뢰도 외래키만 사용: {len(high_confidence_fks)}개 (전체: {len(self.foreign_key_mapper.foreign_key_mappings)}개)")
        
        # 외래키 관계를 기반으로 그래프 구성
        for fk in high_confidence_fks:
            # 양방향 엣지 추가
            self.join_graph.add_edge(
                fk.source_table, fk.target_table,
                weight=1.0 / fk.confidence,  # 신뢰도가 높을수록 가중치 낮음
                foreign_key=fk,
                join_type=JoinType.INNER
            )
            
            # 역방향 엣지도 추가 (LEFT JOIN으로)
            self.join_graph.add_edge(
                fk.target_table, fk.source_table,
                weight=1.0 / fk.confidence,
                foreign_key=fk,
                join_type=JoinType.LEFT
            )
        
        self.logger.info(f"조인 그래프 구성 완료: {self.join_graph.number_of_nodes()}개 노드, {self.join_graph.number_of_edges()}개 엣지")
    
    def _collect_table_statistics(self):
        """테이블 통계 정보 수집"""
        self.logger.info("테이블 통계 정보 수집 중...")
        
        try:
            with get_db_session() as session:
                # 모든 테이블의 행 수와 인덱스 정보 수집
                tables_query = """
                SELECT 
                    TABLE_NAME,
                    TABLE_ROWS,
                    AVG_ROW_LENGTH,
                    DATA_LENGTH,
                    INDEX_LENGTH
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = DATABASE()
                ORDER BY TABLE_NAME
                """
                
                tables_result = execute_query(tables_query)
                
                for table_row in tables_result:
                    table_name = table_row['TABLE_NAME']
                    
                    # 인덱스 정보 조회
                    indexes_query = f"""
                    SELECT 
                        INDEX_NAME,
                        COLUMN_NAME,
                        NON_UNIQUE,
                        CARDINALITY
                    FROM INFORMATION_SCHEMA.STATISTICS 
                    WHERE TABLE_SCHEMA = DATABASE() 
                    AND TABLE_NAME = '{table_name}'
                    ORDER BY INDEX_NAME, SEQ_IN_INDEX
                    """
                    
                    indexes_result = execute_query(indexes_query)
                    
                    # 인덱스 정보 정리
                    indexes = {}
                    for idx in indexes_result:
                        idx_name = idx['INDEX_NAME']
                        if idx_name not in indexes:
                            indexes[idx_name] = {
                                'columns': [],
                                'is_unique': not idx['NON_UNIQUE'],
                                'cardinality': idx['CARDINALITY']
                            }
                        indexes[idx_name]['columns'].append(idx['COLUMN_NAME'])
                    
                    # 테이블 통계 저장
                    self.table_statistics[table_name] = {
                        'row_count': table_row['TABLE_ROWS'] or 0,
                        'avg_row_length': table_row['AVG_ROW_LENGTH'] or 0,
                        'data_length': table_row['DATA_LENGTH'] or 0,
                        'index_length': table_row['INDEX_LENGTH'] or 0,
                        'indexes': indexes
                    }
                
                self.logger.info(f"테이블 통계 수집 완료: {len(self.table_statistics)}개 테이블")
                
        except Exception as e:
            self.logger.error(f"테이블 통계 수집 실패: {str(e)}")
    
    def find_optimal_join_path(self, source_tables: List[str], target_tables: List[str], 
                             strategy: OptimizationStrategy = OptimizationStrategy.COST_BASED) -> OptimizationResult:
        """최적 조인 경로 찾기"""
        self.logger.info(f"최적 조인 경로 탐색: {source_tables} -> {target_tables}")
        
        try:
            # 모든 가능한 경로 찾기
            all_paths = self._find_all_join_paths(source_tables, target_tables)
            
            if not all_paths:
                return OptimizationResult(
                    best_path=None,
                    alternative_paths=[],
                    optimization_notes=["조인 경로를 찾을 수 없습니다."],
                    performance_metrics={}
                )
            
            # 경로 최적화
            optimized_paths = self._optimize_paths(all_paths, strategy)
            
            # 최적 경로 선택
            best_path = optimized_paths[0] if optimized_paths else None
            alternative_paths = optimized_paths[1:5] if len(optimized_paths) > 1 else []
            
            # 성능 메트릭 계산
            performance_metrics = self._calculate_performance_metrics(best_path)
            
            # 최적화 노트 생성
            optimization_notes = self._generate_optimization_notes(best_path, alternative_paths)
            
            result = OptimizationResult(
                best_path=best_path,
                alternative_paths=alternative_paths,
                optimization_notes=optimization_notes,
                performance_metrics=performance_metrics
            )
            
            self.logger.info(f"최적 조인 경로 탐색 완료: {len(optimized_paths)}개 경로 발견")
            return result
            
        except Exception as e:
            self.logger.error(f"최적 조인 경로 탐색 실패: {str(e)}")
            return OptimizationResult(
                best_path=None,
                alternative_paths=[],
                optimization_notes=[f"오류: {str(e)}"],
                performance_metrics={}
            )
    
    def _find_all_join_paths(self, source_tables: List[str], target_tables: List[str]) -> List[JoinPath]:
        """모든 가능한 조인 경로 찾기"""
        all_paths = []
        
        for source_table in source_tables:
            for target_table in target_tables:
                if source_table == target_table:
                    continue
                
                # BFS로 모든 경로 찾기
                paths = self._find_paths_bfs(source_table, target_table)
                all_paths.extend(paths)
        
        return all_paths
    
    def _find_paths_bfs(self, start_table: str, end_table: str, max_depth: int = 3) -> List[JoinPath]:
        """BFS로 경로 찾기 (성능 최적화)"""
        paths = []
        queue = deque([(start_table, [start_table], [])])
        visited = set()
        max_paths = 10  # 최대 경로 수 제한
        
        while queue and len(paths) < max_paths:
            current_table, path, joins = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            if current_table == end_table:
                # 경로 완성
                join_path = JoinPath(
                    tables=path,
                    joins=joins,
                    total_cost=self._calculate_path_cost(joins),
                    confidence=self._calculate_path_confidence(joins),
                    estimated_rows=self._estimate_result_rows(path),
                    estimated_time=self._estimate_execution_time(path, joins),
                    complexity_score=self._calculate_complexity_score(path, joins),
                    description=f"{start_table} -> {end_table} 경로"
                )
                paths.append(join_path)
                continue
            
            # 다음 테이블들 탐색 (신뢰도가 높은 관계 우선)
            neighbors = list(self.join_graph.neighbors(current_table))
            # 신뢰도 순으로 정렬
            neighbors.sort(key=lambda n: self.join_graph[current_table][n].get('weight', 1.0))
            
            for next_table in neighbors[:5]:  # 상위 5개만 탐색
                if next_table in path:  # 순환 방지
                    continue
                
                # 조인 조건 찾기
                join_condition = self._find_join_condition(current_table, next_table)
                if not join_condition:
                    continue
                
                new_path = path + [next_table]
                new_joins = joins + [join_condition]
                
                # 방문 기록 (경로별로 다르게 관리)
                path_key = tuple(new_path)
                if path_key not in visited:
                    visited.add(path_key)
                    queue.append((next_table, new_path, new_joins))
        
        return paths
    
    def _find_join_condition(self, table1: str, table2: str) -> Optional[JoinCondition]:
        """두 테이블 간의 조인 조건 찾기"""
        if not self.join_graph.has_edge(table1, table2):
            return None
        
        edge_data = self.join_graph[table1][table2]
        fk = edge_data.get('foreign_key')
        join_type = edge_data.get('join_type', JoinType.INNER)
        
        if not fk:
            return None
        
        return JoinCondition(
            left_table=table1,
            left_column=fk.source_column,
            right_table=table2,
            right_column=fk.target_column,
            join_type=join_type,
            confidence=fk.confidence
        )
    
    def _calculate_path_cost(self, joins: List[JoinCondition]) -> float:
        """경로 비용 계산"""
        if not joins:
            return 0.0
        
        total_cost = 0.0
        for join in joins:
            # 조인 비용 = 테이블 크기 * 조인 복잡도
            left_size = self.table_statistics.get(join.left_table, {}).get('row_count', 1000)
            right_size = self.table_statistics.get(join.right_table, {}).get('row_count', 1000)
            
            # 조인 복잡도 (신뢰도 기반)
            complexity = 1.0 / join.confidence if join.confidence > 0 else 1.0
            
            # 조인 타입별 가중치
            type_weight = {
                JoinType.INNER: 1.0,
                JoinType.LEFT: 1.2,
                JoinType.RIGHT: 1.2,
                JoinType.FULL: 1.5
            }.get(join.join_type, 1.0)
            
            join_cost = (left_size + right_size) * complexity * type_weight
            total_cost += join_cost
        
        return total_cost
    
    def _calculate_path_confidence(self, joins: List[JoinCondition]) -> float:
        """경로 신뢰도 계산"""
        if not joins:
            return 1.0
        
        # 모든 조인의 신뢰도의 기하평균
        confidence_product = 1.0
        for join in joins:
            confidence_product *= join.confidence
        
        return confidence_product ** (1.0 / len(joins))
    
    def _estimate_result_rows(self, tables: List[str]) -> int:
        """결과 행 수 추정"""
        if not tables:
            return 0
        
        # 첫 번째 테이블의 행 수를 기준으로 추정
        first_table_rows = self.table_statistics.get(tables[0], {}).get('row_count', 1000)
        
        # 조인할 때마다 행 수가 감소한다고 가정 (보수적 추정)
        estimated_rows = first_table_rows
        for i in range(1, len(tables)):
            estimated_rows = int(estimated_rows * 0.8)  # 20% 감소
        
        return max(estimated_rows, 1)
    
    def _estimate_execution_time(self, tables: List[str], joins: List[JoinCondition]) -> float:
        """실행 시간 추정 (초)"""
        if not tables:
            return 0.0
        
        # 기본 실행 시간 (테이블 수에 비례)
        base_time = len(tables) * 0.1
        
        # 조인 복잡도에 따른 추가 시간
        join_time = 0.0
        for join in joins:
            complexity = 1.0 / join.confidence if join.confidence > 0 else 1.0
            join_time += complexity * 0.05
        
        return base_time + join_time
    
    def _calculate_complexity_score(self, tables: List[str], joins: List[JoinCondition]) -> float:
        """복잡도 점수 계산 (0-1, 낮을수록 단순)"""
        if not tables:
            return 0.0
        
        # 테이블 수에 따른 복잡도
        table_complexity = min(len(tables) / 10.0, 1.0)
        
        # 조인 수에 따른 복잡도
        join_complexity = min(len(joins) / 5.0, 1.0)
        
        # 조인 신뢰도에 따른 복잡도
        confidence_complexity = 0.0
        if joins:
            avg_confidence = sum(join.confidence for join in joins) / len(joins)
            confidence_complexity = 1.0 - avg_confidence
        
        # 가중 평균
        total_complexity = (table_complexity * 0.4 + join_complexity * 0.4 + confidence_complexity * 0.2)
        
        return min(total_complexity, 1.0)
    
    def _optimize_paths(self, paths: List[JoinPath], strategy: OptimizationStrategy) -> List[JoinPath]:
        """경로 최적화"""
        if not paths:
            return []
        
        # 전략별 정렬 기준
        if strategy == OptimizationStrategy.COST_BASED:
            paths.sort(key=lambda p: (p.total_cost, -p.confidence))
        elif strategy == OptimizationStrategy.PERFORMANCE_BASED:
            paths.sort(key=lambda p: (p.estimated_time, -p.confidence, p.total_cost))
        elif strategy == OptimizationStrategy.SIMPLICITY_BASED:
            paths.sort(key=lambda p: (p.complexity_score, p.total_cost, -p.confidence))
        else:
            # 기본: 비용 기반
            paths.sort(key=lambda p: (p.total_cost, -p.confidence))
        
        return paths
    
    def _calculate_performance_metrics(self, path: JoinPath) -> Dict[str, Any]:
        """성능 메트릭 계산"""
        if not path:
            return {}
        
        return {
            "estimated_rows": path.estimated_rows,
            "estimated_time": path.estimated_time,
            "total_cost": path.total_cost,
            "confidence": path.confidence,
            "complexity_score": path.complexity_score,
            "table_count": len(path.tables),
            "join_count": len(path.joins)
        }
    
    def _generate_optimization_notes(self, best_path: JoinPath, alternative_paths: List[JoinPath]) -> List[str]:
        """최적화 노트 생성"""
        notes = []
        
        if not best_path:
            notes.append("최적 경로를 찾을 수 없습니다.")
            return notes
        
        # 기본 정보
        notes.append(f"최적 경로: {best_path.description}")
        notes.append(f"테이블 수: {len(best_path.tables)}, 조인 수: {len(best_path.joins)}")
        notes.append(f"예상 결과 행 수: {best_path.estimated_rows:,}")
        notes.append(f"예상 실행 시간: {best_path.estimated_time:.2f}초")
        notes.append(f"신뢰도: {best_path.confidence:.2f}")
        
        # 성능 관련 조언
        if best_path.complexity_score > 0.7:
            notes.append("⚠️ 복잡한 조인 경로입니다. 성능을 고려하여 인덱스를 확인하세요.")
        
        if best_path.estimated_rows > 100000:
            notes.append("⚠️ 대용량 결과가 예상됩니다. LIMIT 절을 고려하세요.")
        
        if best_path.estimated_time > 5.0:
            notes.append("⚠️ 실행 시간이 오래 걸릴 수 있습니다. 쿼리 최적화를 고려하세요.")
        
        # 대안 경로 정보
        if alternative_paths:
            notes.append(f"대안 경로 {len(alternative_paths)}개 발견")
            for i, alt_path in enumerate(alternative_paths[:3], 1):
                notes.append(f"  {i}. {alt_path.description} (비용: {alt_path.total_cost:.2f}, 신뢰도: {alt_path.confidence:.2f})")
        
        return notes
    
    def generate_sql_from_path(self, path: JoinPath, select_columns: List[str] = None, 
                             where_conditions: List[str] = None, order_by: List[str] = None, 
                             limit: int = None) -> str:
        """조인 경로로부터 SQL 생성"""
        if not path or not path.tables:
            return ""
        
        # SELECT 절
        if select_columns:
            select_clause = ", ".join(select_columns)
        else:
            # 기본: 모든 테이블의 주요 컬럼들
            select_clause = self._generate_default_select(path.tables)
        
        # FROM 절 (첫 번째 테이블)
        from_clause = f"FROM {path.tables[0]}"
        
        # JOIN 절들
        join_clauses = []
        for join in path.joins:
            join_type = join.join_type.value
            join_clause = f"{join_type} JOIN {join.right_table} ON {join.left_table}.{join.left_column} = {join.right_table}.{join.right_column}"
            join_clauses.append(join_clause)
        
        # WHERE 절
        where_clause = ""
        if where_conditions:
            where_clause = f"WHERE {' AND '.join(where_conditions)}"
        
        # ORDER BY 절
        order_clause = ""
        if order_by:
            order_clause = f"ORDER BY {', '.join(order_by)}"
        
        # LIMIT 절
        limit_clause = ""
        if limit:
            limit_clause = f"LIMIT {limit}"
        
        # SQL 조합
        sql_parts = [
            f"SELECT {select_clause}",
            from_clause,
            *join_clauses,
            where_clause,
            order_clause,
            limit_clause
        ]
        
        # 빈 부분 제거
        sql_parts = [part for part in sql_parts if part.strip()]
        
        return "\n".join(sql_parts)
    
    def _generate_default_select(self, tables: List[str]) -> str:
        """기본 SELECT 절 생성"""
        columns = []
        
        for table in tables:
            # 각 테이블의 주요 컬럼들 추가
            if table in self.table_statistics:
                # 기본적으로 id, name, created_at 등의 컬럼을 가정
                columns.extend([
                    f"{table}.no as {table}_id",
                    f"{table}.created_at as {table}_created_at"
                ])
        
        return ", ".join(columns) if columns else "*"
    
    def get_join_recommendations(self, tables: List[str]) -> Dict[str, Any]:
        """조인 추천 정보"""
        if len(tables) < 2:
            return {"message": "최소 2개 이상의 테이블이 필요합니다."}
        
        # 모든 테이블 쌍에 대한 조인 경로 찾기
        recommendations = {}
        
        for i, table1 in enumerate(tables):
            for table2 in tables[i+1:]:
                result = self.find_optimal_join_path([table1], [table2])
                
                if result.best_path:
                    recommendations[f"{table1}_to_{table2}"] = {
                        "path": result.best_path,
                        "sql": self.generate_sql_from_path(result.best_path),
                        "performance": result.performance_metrics,
                        "notes": result.optimization_notes
                    }
        
        return recommendations
    
    def analyze_query_complexity(self, sql_query: str) -> Dict[str, Any]:
        """쿼리 복잡도 분석"""
        try:
            # 간단한 SQL 파싱 (실제로는 더 정교한 파서 필요)
            tables_mentioned = self._extract_tables_from_sql(sql_query)
            joins_mentioned = self._extract_joins_from_sql(sql_query)
            
            complexity_score = 0.0
            issues = []
            recommendations = []
            
            # 테이블 수에 따른 복잡도
            if len(tables_mentioned) > 5:
                complexity_score += 0.3
                issues.append("많은 테이블이 사용되었습니다.")
                recommendations.append("불필요한 테이블을 제거하거나 서브쿼리로 분리하세요.")
            
            # 조인 수에 따른 복잡도
            if len(joins_mentioned) > 4:
                complexity_score += 0.3
                issues.append("복잡한 조인이 사용되었습니다.")
                recommendations.append("조인을 단순화하거나 인덱스를 확인하세요.")
            
            # 성능 관련 체크
            if "ORDER BY" in sql_query.upper() and "LIMIT" not in sql_query.upper():
                issues.append("ORDER BY가 있지만 LIMIT가 없습니다.")
                recommendations.append("LIMIT 절을 추가하여 성능을 개선하세요.")
            
            return {
                "complexity_score": min(complexity_score, 1.0),
                "tables_count": len(tables_mentioned),
                "joins_count": len(joins_mentioned),
                "issues": issues,
                "recommendations": recommendations,
                "estimated_performance": "HIGH" if complexity_score < 0.3 else "MEDIUM" if complexity_score < 0.7 else "LOW"
            }
            
        except Exception as e:
            return {
                "error": f"쿼리 분석 실패: {str(e)}",
                "complexity_score": 1.0,
                "estimated_performance": "UNKNOWN"
            }
    
    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """SQL에서 테이블명 추출"""
        # 간단한 정규식으로 테이블명 추출 (실제로는 더 정교한 파서 필요)
        import re
        
        # FROM, JOIN 절에서 테이블명 추출
        table_pattern = r'(?:FROM|JOIN)\s+(\w+)'
        tables = re.findall(table_pattern, sql.upper())
        
        return list(set(tables))  # 중복 제거
    
    def _extract_joins_from_sql(self, sql: str) -> List[str]:
        """SQL에서 조인 정보 추출"""
        import re
        
        # JOIN 절 추출
        join_pattern = r'(?:INNER|LEFT|RIGHT|FULL)?\s*JOIN\s+\w+\s+ON\s+[^;]+'
        joins = re.findall(join_pattern, sql.upper())
        
        return joins

