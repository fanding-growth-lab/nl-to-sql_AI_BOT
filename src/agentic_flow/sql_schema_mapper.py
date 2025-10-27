#!/usr/bin/env python3
"""
SQL Schema Mapper
실제 DB 스키마와 생성된 SQL 간 매핑 알고리즘
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class MappingStrategy(Enum):
    """매핑 전략"""
    EXACT_MATCH = "exact_match"
    SIMILARITY_MATCH = "similarity_match"
    SEMANTIC_MATCH = "semantic_match"
    FALLBACK_MATCH = "fallback_match"


@dataclass
class ColumnMapping:
    """컬럼 매핑 정보"""
    original_column: str
    mapped_column: str
    table: str
    strategy: MappingStrategy
    confidence: float
    reason: str


@dataclass
class TableMapping:
    """테이블 매핑 정보"""
    original_table: str
    mapped_table: str
    strategy: MappingStrategy
    confidence: float
    reason: str


@dataclass
class SQLMappingResult:
    """SQL 매핑 결과"""
    original_sql: str
    mapped_sql: str
    column_mappings: List[ColumnMapping]
    table_mappings: List[TableMapping]
    confidence: float
    is_successful: bool
    error_message: Optional[str] = None


class SQLSchemaMapper:
    """실제 DB 스키마와 생성된 SQL 간 매핑 알고리즘"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 컬럼 매핑 규칙 (실제 DB 스키마 기반)
        self.column_mapping_rules = {
            # 날짜 관련 컬럼 매핑
            "created_at": ["ins_datetime", "created_date", "reg_date", "join_date", "signup_date"],
            "updated_at": ["modified_date", "update_date", "last_modified"],
            "deleted_at": ["delete_date", "remove_date", "del_datetime"],
            
            # 상태 관련 컬럼 매핑
            "status": ["fanding_status", "public_status", "state", "condition", "active_status"],
            "is_active": ["active", "enabled", "valid"],
            "is_deleted": ["deleted", "removed", "disabled"],
            
            # 사용자 관련 컬럼 매핑 (실제 DB 스키마 기반)
            "user_id": ["member_no", "creator_no", "no"],
            "email": ["c_email"],
            "name": ["nickname"],
            "phone": ["phone_number", "mobile", "tel"],
            
            # 금액 관련 컬럼 매핑 (실제 DB 스키마 기반)
            "amount": ["remain_price", "price", "heat"],
            "total": ["total_amount", "sum_amount"],
            "count": ["cnt", "num", "quantity"],
            
            # 추가 실제 컬럼들
            "view_count": ["view_count"],
            "like_count": ["like_count"],
            "creator_no": ["creator_no", "seller_creator_no"],
            "member_no": ["member_no"],
            "department_no": ["department_no"],
            "fanding_no": ["fanding_no"],
            "community_no": ["community_no"],
            "content_type": ["content_type"],
            "currency_no": ["currency_no"],
            "pay_datetime": ["pay_datetime"],
            "start_date": ["start_date"],
            "end_date": ["end_date"],
            "edition": ["edition"],
            "login_datetime": ["login_datetime"]
        }
        
        # 테이블 매핑 규칙 (실제 DB 스키마 기반)
        self.table_mapping_rules = {
            "users": ["t_member_info"],
            "members": ["t_member_info"],
            "creators": ["t_creator"],
            "posts": ["t_community"],
            "payments": ["t_payment"],
            "projects": ["t_project"],
            "collections": ["t_collection"],
            "tiers": ["t_tier"],
            "login_logs": ["t_member_info"],
            "fanding": ["t_fanding"],
            "fanding_logs": ["t_fanding_log"],
            "follows": ["t_follow"],
            "replies": ["t_community_reply"],
            "reviews": ["t_review"],
            "departments": ["t_creator_department"],
            "department_mappings": ["t_creator_department_mapping"],
            "surveys": ["t_membership_stop_survey_response"],
            "survey_texts": ["t_membership_stop_survey_response_text"]
        }
    
    def map_sql_to_schema(self, sql_query: str, db_schema: Dict[str, Any], user_query: str = "") -> SQLMappingResult:
        """
        SQL을 실제 DB 스키마에 매핑
        
        Args:
            sql_query: 매핑할 SQL 쿼리
            db_schema: 실제 DB 스키마 정보
            user_query: 원본 사용자 쿼리
            
        Returns:
            SQLMappingResult: 매핑 결과
        """
        try:
            self.logger.info(f"Mapping SQL to schema: {sql_query[:100]}...")
            
            # 1. SQL 파싱
            parsed_sql = self._parse_sql_query(sql_query)
            
            # 2. 테이블 매핑
            table_mappings = self._map_tables(parsed_sql["tables"], db_schema)
            
            # 3. 컬럼 매핑
            column_mappings = self._map_columns(parsed_sql["columns"], table_mappings, db_schema)
            
            # 4. SQL 재생성
            mapped_sql = self._regenerate_sql(sql_query, table_mappings, column_mappings)
            
            # 5. 신뢰도 계산
            confidence = self._calculate_mapping_confidence(table_mappings, column_mappings)
            
            # 6. 성공 여부 판단
            is_successful = confidence > 0.5 and len([m for m in column_mappings if m.confidence > 0.5]) > 0
            
            result = SQLMappingResult(
                original_sql=sql_query,
                mapped_sql=mapped_sql,
                column_mappings=column_mappings,
                table_mappings=table_mappings,
                confidence=confidence,
                is_successful=is_successful
            )
            
            self.logger.info(f"SQL mapping completed: {len(column_mappings)} columns mapped, confidence: {confidence:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error mapping SQL to schema: {str(e)}")
            return SQLMappingResult(
                original_sql=sql_query,
                mapped_sql=sql_query,
                column_mappings=[],
                table_mappings=[],
                confidence=0.0,
                is_successful=False,
                error_message=str(e)
            )
    
    def _parse_sql_query(self, sql_query: str) -> Dict[str, Any]:
        """SQL 쿼리 파싱"""
        sql_upper = sql_query.upper().strip()
        
        parsed = {
            "tables": [],
            "columns": [],
            "functions": [],
            "joins": [],
            "where_conditions": [],
            "group_by": [],
            "order_by": []
        }
        
        # FROM 절에서 테이블 추출
        from_match = re.search(r'FROM\s+(\w+)', sql_upper)
        if from_match:
            parsed["tables"].append(from_match.group(1).lower())
        
        # JOIN 절에서 테이블 추출
        join_matches = re.findall(r'JOIN\s+(\w+)', sql_upper)
        for match in join_matches:
            parsed["tables"].append(match.lower())
            parsed["joins"].append(match.lower())
        
        # SELECT 절에서 컬럼 추출
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_upper, re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            # 컬럼명 추출 (더 정교한 패턴)
            column_matches = re.findall(r'\b(\w+)\b', select_clause)
            for col in column_matches:
                if col.upper() not in ['SELECT', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'DISTINCT', 'AS']:
                    parsed["columns"].append(col.lower())
        
        # WHERE 절에서 조건 추출
        where_match = re.search(r'WHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+HAVING|$)', sql_upper, re.DOTALL)
        if where_match:
            where_clause = where_match.group(1)
            where_columns = re.findall(r'\b(\w+)\b', where_clause)
            for col in where_columns:
                if col.upper() not in ['WHERE', 'AND', 'OR', 'NOT', 'IN', 'LIKE', 'BETWEEN', 'MONTH', 'YEAR', 'DAY']:
                    parsed["where_conditions"].append(col.lower())
        
        return parsed
    
    def _map_tables(self, tables: List[str], db_schema: Dict[str, Any]) -> List[TableMapping]:
        """테이블 매핑"""
        mappings = []
        
        for table in tables:
            # 1. 정확한 매칭
            if table in db_schema:
                mappings.append(TableMapping(
                    original_table=table,
                    mapped_table=table,
                    strategy=MappingStrategy.EXACT_MATCH,
                    confidence=1.0,
                    reason="Exact table name match"
                ))
                continue
            
            # 2. 규칙 기반 매핑
            mapped_table = None
            for rule_table, candidates in self.table_mapping_rules.items():
                if table in candidates:
                    # 후보 중에서 실제 존재하는 테이블 찾기
                    for candidate in candidates:
                        if candidate in db_schema:
                            mapped_table = candidate
                            break
                    if mapped_table:
                        break
            
            if mapped_table:
                mappings.append(TableMapping(
                    original_table=table,
                    mapped_table=mapped_table,
                    strategy=MappingStrategy.SIMILARITY_MATCH,
                    confidence=0.8,
                    reason=f"Mapped via rule: {table} -> {mapped_table}"
                ))
            else:
                # 3. 유사도 기반 매핑
                best_match = self._find_best_table_match(table, db_schema.keys())
                if best_match:
                    mappings.append(TableMapping(
                        original_table=table,
                        mapped_table=best_match,
                        strategy=MappingStrategy.SEMANTIC_MATCH,
                        confidence=0.6,
                        reason=f"Similarity match: {table} -> {best_match}"
                    ))
                else:
                    # 4. 실제 존재하는 테이블 중에서 기본 테이블 선택
                    fallback_table = self._select_fallback_table(db_schema)
                    mappings.append(TableMapping(
                        original_table=table,
                        mapped_table=fallback_table,
                        strategy=MappingStrategy.FALLBACK_MATCH,
                        confidence=0.3,
                        reason=f"Fallback to {fallback_table} table"
                    ))
        
        return mappings
    
    def _map_columns(self, columns: List[str], table_mappings: List[TableMapping], db_schema: Dict[str, Any]) -> List[ColumnMapping]:
        """컬럼 매핑"""
        mappings = []
        
        for column in columns:
            # 각 테이블에 대해 컬럼 매핑 시도
            for table_mapping in table_mappings:
                table_name = table_mapping.mapped_table
                if table_name not in db_schema:
                    continue
                
                # 실제 DB 스키마 구조에 맞게 컬럼명 추출
                table_info = db_schema.get(table_name, {})
                table_columns = list(table_info.get("columns", {}).keys())
                
                # 1. 정확한 매칭
                if column in table_columns:
                    mappings.append(ColumnMapping(
                        original_column=column,
                        mapped_column=column,
                        table=table_name,
                        strategy=MappingStrategy.EXACT_MATCH,
                        confidence=1.0,
                        reason=f"Exact column match in {table_name}"
                    ))
                    break
                
                # 2. 규칙 기반 매핑
                mapped_column = None
                for rule_column, candidates in self.column_mapping_rules.items():
                    if column in candidates:
                        # 후보 중에서 실제 존재하는 컬럼 찾기
                        for candidate in candidates:
                            if candidate in table_columns:
                                mapped_column = candidate
                                break
                        if mapped_column:
                            break
                
                if mapped_column:
                    mappings.append(ColumnMapping(
                        original_column=column,
                        mapped_column=mapped_column,
                        table=table_name,
                        strategy=MappingStrategy.SIMILARITY_MATCH,
                        confidence=0.8,
                        reason=f"Mapped via rule: {column} -> {mapped_column}"
                    ))
                    break
                
                # 3. 유사도 기반 매핑
                best_match = self._find_best_column_match(column, table_columns)
                if best_match:
                    mappings.append(ColumnMapping(
                        original_column=column,
                        mapped_column=best_match,
                        table=table_name,
                        strategy=MappingStrategy.SEMANTIC_MATCH,
                        confidence=0.6,
                        reason=f"Similarity match: {column} -> {best_match}"
                    ))
                    break
        
        # 매핑되지 않은 컬럼에 대해 기본값 제공
        mapped_columns = [m.original_column for m in mappings]
        for column in columns:
            if column not in mapped_columns:
                # 실제 존재하는 테이블과 컬럼으로 폴백
                fallback_table = self._select_fallback_table(db_schema)
                fallback_column = self._select_fallback_column(fallback_table, db_schema)
                mappings.append(ColumnMapping(
                    original_column=column,
                    mapped_column=fallback_column,
                    table=fallback_table,
                    strategy=MappingStrategy.FALLBACK_MATCH,
                    confidence=0.3,
                    reason=f"Fallback to {fallback_column} column in {fallback_table}"
                ))
        
        return mappings
    
    def _find_best_table_match(self, table: str, available_tables: List[str]) -> Optional[str]:
        """최적 테이블 매칭 찾기"""
        best_match = None
        best_score = 0.0
        
        for available_table in available_tables:
            score = self._calculate_similarity(table, available_table)
            if score > best_score and score > 0.5:
                best_score = score
                best_match = available_table
        
        return best_match
    
    def _find_best_column_match(self, column: str, available_columns: List[str]) -> Optional[str]:
        """최적 컬럼 매칭 찾기"""
        best_match = None
        best_score = 0.0
        
        for available_column in available_columns:
            score = self._calculate_similarity(column, available_column)
            if score > best_score and score > 0.5:
                best_score = score
                best_match = available_column
        
        return best_match
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """문자열 유사도 계산 (간단한 Jaccard 유사도)"""
        set1 = set(str1.lower())
        set2 = set(str2.lower())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _regenerate_sql(self, original_sql: str, table_mappings: List[TableMapping], column_mappings: List[ColumnMapping]) -> str:
        """매핑된 정보로 SQL 재생성"""
        mapped_sql = original_sql
        
        # 테이블 매핑 적용
        for table_mapping in table_mappings:
            if table_mapping.original_table != table_mapping.mapped_table:
                mapped_sql = re.sub(
                    rf'\b{table_mapping.original_table}\b',
                    table_mapping.mapped_table,
                    mapped_sql,
                    flags=re.IGNORECASE
                )
        
        # 컬럼 매핑 적용
        for column_mapping in column_mappings:
            if column_mapping.original_column != column_mapping.mapped_column:
                mapped_sql = re.sub(
                    rf'\b{column_mapping.original_column}\b',
                    column_mapping.mapped_column,
                    mapped_sql,
                    flags=re.IGNORECASE
                )
        
        return mapped_sql
    
    def _calculate_mapping_confidence(self, table_mappings: List[TableMapping], column_mappings: List[ColumnMapping]) -> float:
        """매핑 신뢰도 계산"""
        if not table_mappings and not column_mappings:
            return 0.0
        
        # 테이블 매핑 신뢰도
        table_confidence = sum(mapping.confidence for mapping in table_mappings) / len(table_mappings) if table_mappings else 0.0
        
        # 컬럼 매핑 신뢰도
        column_confidence = sum(mapping.confidence for mapping in column_mappings) / len(column_mappings) if column_mappings else 0.0
        
        # 전체 신뢰도 (가중 평균)
        total_mappings = len(table_mappings) + len(column_mappings)
        if total_mappings == 0:
            return 0.0
        
        return (table_confidence * len(table_mappings) + column_confidence * len(column_mappings)) / total_mappings
    
    def _select_fallback_table(self, db_schema: Dict[str, Any]) -> str:
        """폴백 테이블 선택 (실제 존재하는 테이블 중에서)"""
        # 우선순위 순서로 테이블 선택 (PDF 스키마 기반)
        priority_tables = [
            "t_member_info", "t_creator", "t_fanding", "t_fanding_log", 
            "t_payment", "t_community", "t_follow", "t_review"
        ]
        
        for table in priority_tables:
            if table in db_schema:
                return table
        
        # 우선순위 테이블이 없으면 첫 번째 사용 가능한 테이블 사용
        available_tables = list(db_schema.keys())
        if available_tables:
            return available_tables[0]
        
        # 모든 테이블이 없으면 기본값
        return "t_member_info"
    
    def _select_fallback_column(self, table_name: str, db_schema: Dict[str, Any]) -> str:
        """폴백 컬럼 선택 (실제 존재하는 컬럼 중에서)"""
        if table_name not in db_schema:
            return "member_no"
        
        table_info = db_schema[table_name]
        columns = list(table_info.get("columns", {}).keys())
        
        if not columns:
            return "member_no"
        
        # 우선순위 순서로 컬럼 선택 (PDF 스키마 기반)
        priority_columns = ["member_no", "creator_no", "no", "fanding_status", "ins_datetime", "status"]
        
        for col in priority_columns:
            if col in columns:
                return col
        
        # 우선순위 컬럼이 없으면 첫 번째 컬럼 사용
        return columns[0]
