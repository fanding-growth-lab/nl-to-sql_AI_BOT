"""
SQL 쿼리 최적화 시스템

이 모듈은 SQL 쿼리를 분석하고 자동으로 인덱스 힌트를 추가하여
쿼리 실행 시간을 단축하는 시스템을 제공합니다.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """최적화 레벨"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class IndexHint:
    """인덱스 힌트 정보"""
    table_name: str
    index_name: str
    hint_type: str  # 'USE', 'FORCE', 'IGNORE'
    reason: str
    confidence: float


@dataclass
class QueryAnalysis:
    """쿼리 분석 결과"""
    original_sql: str
    optimized_sql: str
    tables: List[str]
    joins: List[Tuple[str, str]]
    where_conditions: List[str]
    order_by_columns: List[str]
    index_hints: List[IndexHint]
    optimization_level: OptimizationLevel
    estimated_improvement: float  # 예상 성능 개선률 (%)


class SQLOptimizer:
    """SQL 쿼리 최적화 클래스"""
    
    def __init__(self):
        self.index_mapping = self._create_index_mapping()
        self.optimization_rules = self._create_optimization_rules()
        self.logger = logging.getLogger(__name__)
    
    def _create_index_mapping(self) -> Dict[str, List[str]]:
        """테이블별 사용 가능한 인덱스 매핑"""
        return {
            "t_member": [
                "idx_member_status",
                "idx_member_ins_datetime", 
                "idx_member_email",
                "idx_member_creator_no"
            ],
            "t_member_login_log": [
                "idx_login_member_no",
                "idx_login_ins_datetime",
                "idx_login_date"
            ],
            "t_creator": [
                "idx_creator_name",
                "idx_creator_status",
                "idx_creator_ins_datetime"
            ],
            "t_payment": [
                "idx_payment_status",
                "idx_payment_ins_datetime",
                "idx_payment_member_no",
                "idx_payment_price"
            ],
            "t_post": [
                "idx_post_status",
                "idx_post_ins_datetime",
                "idx_post_view_count",
                "idx_post_like_count",
                "idx_post_creator_no"
            ],
            "t_post_view_log": [
                "idx_view_post_no",
                "idx_view_member_no",
                "idx_view_ins_datetime"
            ]
        }
    
    def _create_optimization_rules(self) -> Dict[str, Dict[str, Any]]:
        """최적화 규칙 정의"""
        return {
            "date_range_queries": {
                "pattern": r"WHERE.*(ins_datetime|created_at|updated_at).*[><=]",
                "index_priority": ["ins_datetime", "created_at", "updated_at"],
                "hint_type": "USE"
            },
            "status_queries": {
                "pattern": r"WHERE.*status.*=.*['\"](A|I|D)['\"]",
                "index_priority": ["status"],
                "hint_type": "USE"
            },
            "join_optimization": {
                "pattern": r"JOIN.*ON.*=",
                "index_priority": ["foreign_key_indexes"],
                "hint_type": "USE"
            },
            "order_by_optimization": {
                "pattern": r"ORDER BY.*",
                "index_priority": ["order_by_columns"],
                "hint_type": "USE"
            },
            "aggregation_queries": {
                "pattern": r"(COUNT|SUM|AVG|MAX|MIN)\s*\(",
                "index_priority": ["group_by_columns"],
                "hint_type": "USE"
            }
        }
    
    def analyze_and_optimize(self, sql: str, optimization_level: OptimizationLevel = OptimizationLevel.INTERMEDIATE) -> QueryAnalysis:
        """SQL 쿼리 분석 및 최적화"""
        # 1. 쿼리 파싱
        tables = self._extract_tables(sql)
        joins = self._extract_joins(sql)
        where_conditions = self._extract_where_conditions(sql)
        order_by_columns = self._extract_order_by_columns(sql)
        
        # 2. 인덱스 힌트 생성
        index_hints = self._generate_index_hints(sql, tables, where_conditions, order_by_columns)
        
        # 3. SQL 최적화
        optimized_sql = self._apply_optimizations(sql, index_hints, optimization_level)
        
        # 4. 성능 개선률 추정
        estimated_improvement = self._estimate_improvement(sql, optimized_sql, index_hints)
        
        return QueryAnalysis(
            original_sql=sql,
            optimized_sql=optimized_sql,
            tables=tables,
            joins=joins,
            where_conditions=where_conditions,
            order_by_columns=order_by_columns,
            index_hints=index_hints,
            optimization_level=optimization_level,
            estimated_improvement=estimated_improvement
        )
    
    def _extract_tables(self, sql: str) -> List[str]:
        """SQL에서 테이블명 추출"""
        # FROM 절에서 테이블 추출
        from_pattern = r"FROM\s+(\w+)"
        from_matches = re.findall(from_pattern, sql, re.IGNORECASE)
        
        # JOIN 절에서 테이블 추출
        join_pattern = r"JOIN\s+(\w+)"
        join_matches = re.findall(join_pattern, sql, re.IGNORECASE)
        
        tables = list(set(from_matches + join_matches))
        return [table for table in tables if table.startswith('t_')]
    
    def _extract_joins(self, sql: str) -> List[Tuple[str, str]]:
        """SQL에서 조인 정보 추출"""
        joins = []
        join_pattern = r"JOIN\s+(\w+)\s+ON\s+(\w+\.\w+)\s*=\s*(\w+\.\w+)"
        matches = re.findall(join_pattern, sql, re.IGNORECASE)
        
        for match in matches:
            table, left_col, right_col = match
            left_table = left_col.split('.')[0]
            right_table = right_col.split('.')[0]
            joins.append((left_table, right_table))
        
        return joins
    
    def _extract_where_conditions(self, sql: str) -> List[str]:
        """SQL에서 WHERE 조건 추출"""
        where_pattern = r"WHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|$)"
        match = re.search(where_pattern, sql, re.IGNORECASE | re.DOTALL)
        
        if match:
            conditions = match.group(1).strip()
            # AND, OR로 분리
            return [cond.strip() for cond in re.split(r'\s+(?:AND|OR)\s+', conditions)]
        
        return []
    
    def _extract_order_by_columns(self, sql: str) -> List[str]:
        """SQL에서 ORDER BY 컬럼 추출"""
        order_pattern = r"ORDER\s+BY\s+(.*?)(?:\s+LIMIT|$)"
        match = re.search(order_pattern, sql, re.IGNORECASE)
        
        if match:
            columns = match.group(1).strip()
            # 쉼표로 분리
            return [col.strip().split()[0] for col in columns.split(',')]
        
        return []
    
    def _generate_index_hints(self, sql: str, tables: List[str], where_conditions: List[str], order_by_columns: List[str]) -> List[IndexHint]:
        """인덱스 힌트 생성"""
        hints = []
        
        for table in tables:
            if table not in self.index_mapping:
                continue
            
            available_indexes = self.index_mapping[table]
            
            # WHERE 조건 기반 인덱스 힌트
            for condition in where_conditions:
                for index in available_indexes:
                    if self._is_index_relevant(condition, index):
                        hints.append(IndexHint(
                            table_name=table,
                            index_name=index,
                            hint_type="USE",
                            reason=f"WHERE condition optimization: {condition}",
                            confidence=0.8
                        ))
            
            # ORDER BY 기반 인덱스 힌트
            for column in order_by_columns:
                for index in available_indexes:
                    if column in index:
                        hints.append(IndexHint(
                            table_name=table,
                            index_name=index,
                            hint_type="USE",
                            reason=f"ORDER BY optimization: {column}",
                            confidence=0.9
                        ))
        
        return hints
    
    def _is_index_relevant(self, condition: str, index: str) -> bool:
        """조건이 인덱스와 관련이 있는지 확인"""
        # 컬럼명이 인덱스명에 포함되어 있는지 확인
        condition_lower = condition.lower()
        index_lower = index.lower()
        
        # 일반적인 컬럼명 매핑
        column_mappings = {
            'ins_datetime': 'datetime',
            'status': 'status',
            'email': 'email',
            'member_no': 'member',
            'creator_no': 'creator',
            'view_count': 'view',
            'like_count': 'like'
        }
        
        for column, pattern in column_mappings.items():
            if pattern in index_lower and column in condition_lower:
                return True
        
        return False
    
    def _apply_optimizations(self, sql: str, index_hints: List[IndexHint], optimization_level: OptimizationLevel) -> str:
        """최적화 적용"""
        optimized_sql = sql
        
        # 1. 인덱스 힌트 추가
        if index_hints:
            optimized_sql = self._add_index_hints(optimized_sql, index_hints)
        
        # 2. LIMIT 절 추가 (대용량 결과 방지)
        if optimization_level in [OptimizationLevel.INTERMEDIATE, OptimizationLevel.ADVANCED]:
            optimized_sql = self._add_limit_clause(optimized_sql)
        
        # 3. 쿼리 구조 최적화
        if optimization_level == OptimizationLevel.ADVANCED:
            optimized_sql = self._optimize_query_structure(optimized_sql)
        
        return optimized_sql
    
    def _add_index_hints(self, sql: str, index_hints: List[IndexHint]) -> str:
        """인덱스 힌트 추가"""
        # 테이블별로 그룹화
        hints_by_table = {}
        for hint in index_hints:
            if hint.table_name not in hints_by_table:
                hints_by_table[hint.table_name] = []
            hints_by_table[hint.table_name].append(hint)
        
        # 각 테이블에 인덱스 힌트 추가
        for table, hints in hints_by_table.items():
            # 가장 높은 신뢰도의 힌트 선택
            best_hint = max(hints, key=lambda h: h.confidence)
            
            # FROM 절에 인덱스 힌트 추가
            from_pattern = f"FROM\\s+{table}"
            hint_clause = f"FROM {table} {best_hint.hint_type} INDEX ({best_hint.index_name})"
            sql = re.sub(from_pattern, hint_clause, sql, flags=re.IGNORECASE)
        
        return sql
    
    def _add_limit_clause(self, sql: str) -> str:
        """LIMIT 절 추가"""
        if not re.search(r"LIMIT\s+\d+", sql, re.IGNORECASE):
            # 기본 LIMIT 1000 추가
            sql += " LIMIT 1000"
        
        return sql
    
    def _optimize_query_structure(self, sql: str) -> str:
        """쿼리 구조 최적화"""
        # 1. 불필요한 SELECT * 최적화
        if "SELECT *" in sql.upper():
            # 실제 필요한 컬럼만 선택하도록 제안
            sql = sql.replace("SELECT *", "SELECT /*+ SPECIFIC_COLUMNS */ *")
        
        # 2. 서브쿼리 최적화
        sql = self._optimize_subqueries(sql)
        
        return sql
    
    def _optimize_subqueries(self, sql: str) -> str:
        """서브쿼리 최적화"""
        # EXISTS를 IN으로 변환하는 등의 최적화
        # 실제 구현에서는 더 복잡한 로직이 필요
        return sql
    
    def _estimate_improvement(self, original_sql: str, optimized_sql: str, index_hints: List[IndexHint]) -> float:
        """성능 개선률 추정"""
        if not index_hints:
            return 0.0
        
        # 인덱스 힌트 개수와 신뢰도 기반으로 개선률 추정
        total_confidence = sum(hint.confidence for hint in index_hints)
        avg_confidence = total_confidence / len(index_hints)
        
        # 기본 개선률 계산
        base_improvement = len(index_hints) * 10  # 힌트당 10% 개선
        confidence_factor = avg_confidence * 0.5  # 신뢰도 기반 가중치
        
        return min(base_improvement + confidence_factor, 80.0)  # 최대 80% 개선
    
    def get_optimization_suggestions(self, analysis: QueryAnalysis) -> List[str]:
        """최적화 제안 목록 반환"""
        suggestions = []
        
        if not analysis.index_hints:
            suggestions.append("인덱스 힌트를 추가하여 쿼리 성능을 개선할 수 있습니다.")
        
        if analysis.estimated_improvement > 50:
            suggestions.append(f"예상 성능 개선률: {analysis.estimated_improvement:.1f}%")
        
        if "SELECT *" in analysis.original_sql.upper():
            suggestions.append("SELECT * 대신 필요한 컬럼만 선택하는 것을 고려해보세요.")
        
        if not re.search(r"LIMIT\s+\d+", analysis.original_sql, re.IGNORECASE):
            suggestions.append("대용량 결과를 방지하기 위해 LIMIT 절을 추가하는 것을 고려해보세요.")
        
        return suggestions


# 테스트 함수들
def test_sql_optimization():
    """SQL 최적화 테스트"""
    optimizer = SQLOptimizer()
    
    test_queries = [
        "SELECT * FROM t_member WHERE status = 'A'",
        "SELECT COUNT(*) FROM t_member_login_log WHERE ins_datetime >= '2024-01-01'",
        "SELECT c.creator_name, COUNT(m.member_no) FROM t_creator c LEFT JOIN t_member m ON c.creator_no = m.creator_no GROUP BY c.creator_no ORDER BY COUNT(m.member_no) DESC",
        "SELECT * FROM t_post WHERE view_count > 100 ORDER BY like_count DESC"
    ]
    
    print("=== SQL 최적화 테스트 ===")
    for sql in test_queries:
        print(f"원본 SQL: {sql}")
        
        analysis = optimizer.analyze_and_optimize(sql)
        
        print(f"  → 최적화된 SQL: {analysis.optimized_sql}")
        print(f"  → 인덱스 힌트: {len(analysis.index_hints)}개")
        for hint in analysis.index_hints:
            print(f"    - {hint.table_name}: {hint.index_name} ({hint.reason})")
        print(f"  → 예상 개선률: {analysis.estimated_improvement:.1f}%")
        
        suggestions = optimizer.get_optimization_suggestions(analysis)
        if suggestions:
            print(f"  → 제안사항: {'; '.join(suggestions)}")
        
        print()


if __name__ == "__main__":
    test_sql_optimization()

