#!/usr/bin/env python3
"""
SQL Schema Analyzer
SQL 스키마 불일치 감지 및 분석 시스템
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from core.db import get_cached_db_schema


class SchemaMismatchType(Enum):
    """스키마 불일치 유형"""
    MISSING_TABLE = "missing_table"
    MISSING_COLUMN = "missing_column"
    INVALID_COLUMN_TYPE = "invalid_column_type"
    INVALID_FUNCTION = "invalid_function"
    INVALID_JOIN = "invalid_join"
    INVALID_WHERE_CONDITION = "invalid_where_condition"
    INVALID_GROUP_BY = "invalid_group_by"
    INVALID_ORDER_BY = "invalid_order_by"


@dataclass
class SchemaMismatch:
    """스키마 불일치 정보"""
    type: SchemaMismatchType
    element: str  # 테이블명, 컬럼명 등
    position: int  # SQL에서의 위치
    severity: str  # "error", "warning", "info"
    message: str
    suggestion: str
    confidence: float


@dataclass
class SchemaAnalysisResult:
    """스키마 분석 결과"""
    is_valid: bool
    mismatches: List[SchemaMismatch]
    valid_tables: List[str]
    valid_columns: List[str]
    confidence: float
    recommendations: List[str]


class SQLSchemaAnalyzer:
    """SQL 스키마 불일치 감지 및 분석 시스템"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def analyze_sql_schema_compatibility(self, sql_query: str, user_query: str = "") -> SchemaAnalysisResult:
        """
        SQL 쿼리의 스키마 호환성 분석
        
        Args:
            sql_query: 분석할 SQL 쿼리
            user_query: 원본 사용자 쿼리 (컨텍스트 제공)
            
        Returns:
            SchemaAnalysisResult: 분석 결과
        """
        try:
            self.logger.info(f"Analyzing SQL schema compatibility: {sql_query[:100]}...")
            
            # 1. 실제 DB 스키마 로드
            db_schema = self._load_database_schema()
            
            # 2. SQL 파싱
            parsed_sql = self._parse_sql_query(sql_query)
            
            # 3. 스키마 불일치 감지
            mismatches = self._detect_schema_mismatches(parsed_sql, db_schema)
            
            # 4. 유효한 테이블/컬럼 추출
            valid_tables, valid_columns = self._extract_valid_elements(parsed_sql, db_schema)
            
            # 5. 신뢰도 계산
            confidence = self._calculate_confidence(mismatches, valid_tables, valid_columns)
            
            # 6. 권장사항 생성
            recommendations = self._generate_recommendations(mismatches, user_query)
            
            # 7. 결과 생성
            result = SchemaAnalysisResult(
                is_valid=len([m for m in mismatches if m.severity == "error"]) == 0,
                mismatches=mismatches,
                valid_tables=valid_tables,
                valid_columns=valid_columns,
                confidence=confidence,
                recommendations=recommendations
            )
            
            self.logger.info(f"Schema analysis completed: {len(mismatches)} mismatches found, confidence: {confidence:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing SQL schema: {str(e)}")
            return SchemaAnalysisResult(
                is_valid=False,
                mismatches=[SchemaMismatch(
                    type=SchemaMismatchType.MISSING_TABLE,
                    element="unknown",
                    position=0,
                    severity="error",
                    message=f"Schema analysis failed: {str(e)}",
                    suggestion="Check database connection and schema",
                    confidence=0.0
                )],
                valid_tables=[],
                valid_columns=[],
                confidence=0.0,
                recommendations=["Fix database connection issues"]
            )
    
    def _load_database_schema(self) -> Dict[str, Any]:
        """실제 DB 스키마 로드 (중앙화된 스키마 사용)"""
        try:
            # 중앙화된 스키마 캐시 사용
            return get_cached_db_schema()
        except Exception as e:
            self.logger.error(f"Error loading database schema: {str(e)}")
            return {}
    
    def _parse_sql_query(self, sql_query: str) -> Dict[str, Any]:
        """SQL 쿼리 파싱"""
        sql_upper = sql_query.upper().strip()
        
        # 기본 SQL 구조 파싱
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
            # 컬럼명 추출 (간단한 패턴)
            column_matches = re.findall(r'\b(\w+)\b', select_clause)
            for col in column_matches:
                if col.upper() not in ['SELECT', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'DISTINCT']:
                    parsed["columns"].append(col.lower())
        
        # WHERE 절에서 조건 추출
        where_match = re.search(r'WHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+HAVING|$)', sql_upper, re.DOTALL)
        if where_match:
            where_clause = where_match.group(1)
            # 컬럼명 추출
            where_columns = re.findall(r'\b(\w+)\b', where_clause)
            for col in where_columns:
                if col.upper() not in ['WHERE', 'AND', 'OR', 'NOT', 'IN', 'LIKE', 'BETWEEN']:
                    parsed["where_conditions"].append(col.lower())
        
        # 함수 추출
        function_matches = re.findall(r'\b(COUNT|SUM|AVG|MAX|MIN|DATE_FORMAT|MONTH|YEAR|DAY)\s*\(', sql_upper)
        parsed["functions"] = list(set(function_matches))
        
        return parsed
    
    def _detect_schema_mismatches(self, parsed_sql: Dict[str, Any], db_schema: Dict[str, Any]) -> List[SchemaMismatch]:
        """스키마 불일치 감지"""
        mismatches = []
        
        # 1. 테이블 존재 여부 확인
        for table in parsed_sql["tables"]:
            if table not in db_schema:
                mismatches.append(SchemaMismatch(
                    type=SchemaMismatchType.MISSING_TABLE,
                    element=table,
                    position=0,
                    severity="error",
                    message=f"테이블 '{table}'이 존재하지 않습니다",
                    suggestion=f"사용 가능한 테이블: {', '.join(db_schema.keys())}",
                    confidence=1.0
                ))
        
        # 2. 컬럼 존재 여부 확인
        for table in parsed_sql["tables"]:
            if table in db_schema:
                # 실제 DB 스키마 구조에 맞게 컬럼명 추출
                table_info = db_schema.get(table, {})
                table_columns = list(table_info.get("columns", {}).keys())
                
                for column in parsed_sql["columns"]:
                    if column not in table_columns:
                        mismatches.append(SchemaMismatch(
                            type=SchemaMismatchType.MISSING_COLUMN,
                            element=f"{table}.{column}",
                            position=0,
                            severity="error",
                            message=f"컬럼 '{column}'이 테이블 '{table}'에 존재하지 않습니다",
                            suggestion=f"사용 가능한 컬럼: {', '.join(table_columns)}",
                            confidence=1.0
                        ))
        
        # 3. WHERE 조건의 컬럼 확인
        for table in parsed_sql["tables"]:
            if table in db_schema:
                # 실제 DB 스키마 구조에 맞게 컬럼명 추출
                table_info = db_schema.get(table, {})
                table_columns = list(table_info.get("columns", {}).keys())
                
                for condition in parsed_sql["where_conditions"]:
                    if condition not in table_columns:
                        mismatches.append(SchemaMismatch(
                            type=SchemaMismatchType.INVALID_WHERE_CONDITION,
                            element=f"{table}.{condition}",
                            position=0,
                            severity="warning",
                            message=f"WHERE 조건의 컬럼 '{condition}'이 테이블 '{table}'에 존재하지 않습니다",
                            suggestion=f"사용 가능한 컬럼: {', '.join(table_columns)}",
                            confidence=0.8
                        ))
        
        # 4. 함수 호환성 확인
        for function in parsed_sql["functions"]:
            if function in ["DATE_FORMAT", "MONTH", "YEAR", "DAY"]:
                # 날짜 함수 사용 시 ins_datetime 컬럼이 있는지 확인
                if any("ins_datetime" in col for col in parsed_sql["columns"]):
                    # ins_datetime이 있으면 정상
                    pass
                elif any("created_at" in col for col in parsed_sql["columns"]):
                    # created_at이 있으면 ins_datetime으로 교체 권장
                    mismatches.append(SchemaMismatch(
                        type=SchemaMismatchType.INVALID_FUNCTION,
                        element=f"{function}(created_at)",
                        position=0,
                        severity="warning",
                        message=f"함수 {function}에서 사용된 'created_at' 컬럼을 'ins_datetime'으로 교체하세요",
                        suggestion="created_at 대신 ins_datetime 컬럼을 사용하세요",
                        confidence=0.8
                    ))
                elif any("reg_date" in col for col in parsed_sql["columns"]):
                    # reg_date가 있으면 ins_datetime으로 교체 권장
                    mismatches.append(SchemaMismatch(
                        type=SchemaMismatchType.INVALID_FUNCTION,
                        element=f"{function}(reg_date)",
                        position=0,
                        severity="warning",
                        message=f"함수 {function}에서 사용된 'reg_date' 컬럼을 'ins_datetime'으로 교체하세요",
                        suggestion="reg_date 대신 ins_datetime 컬럼을 사용하세요",
                        confidence=0.8
                    ))
        
        return mismatches
    
    def _extract_valid_elements(self, parsed_sql: Dict[str, Any], db_schema: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """유효한 테이블과 컬럼 추출"""
        valid_tables = []
        valid_columns = []
        
        for table in parsed_sql["tables"]:
            if table in db_schema:
                valid_tables.append(table)
                # 실제 DB 스키마 구조에 맞게 컬럼명 추출
                table_info = db_schema.get(table, {})
                table_columns = list(table_info.get("columns", {}).keys())
                
                for column in parsed_sql["columns"]:
                    if column in table_columns:
                        valid_columns.append(f"{table}.{column}")
        
        return valid_tables, valid_columns
    
    def _calculate_confidence(self, mismatches: List[SchemaMismatch], valid_tables: List[str], valid_columns: List[str]) -> float:
        """신뢰도 계산"""
        if not mismatches:
            return 1.0
        
        # 에러 개수에 따른 신뢰도 감소
        error_count = len([m for m in mismatches if m.severity == "error"])
        warning_count = len([m for m in mismatches if m.severity == "warning"])
        
        # 기본 신뢰도
        base_confidence = 1.0
        
        # 에러당 0.3 감소, 경고당 0.1 감소
        confidence = base_confidence - (error_count * 0.3) - (warning_count * 0.1)
        
        # 유효한 요소가 있으면 보너스
        if valid_tables:
            confidence += 0.1
        if valid_columns:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_recommendations(self, mismatches: List[SchemaMismatch], user_query: str) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        # 에러가 있으면 기본 권장사항
        errors = [m for m in mismatches if m.severity == "error"]
        if errors:
            recommendations.append("SQL 쿼리를 실제 데이터베이스 스키마에 맞게 수정해야 합니다")
            
            # 구체적인 수정 방안 제시
            for error in errors:
                if error.type == SchemaMismatchType.MISSING_COLUMN:
                    recommendations.append(f"컬럼 '{error.element}'을 실제 존재하는 컬럼으로 교체하세요")
                elif error.type == SchemaMismatchType.MISSING_TABLE:
                    recommendations.append(f"테이블 '{error.element}'을 실제 존재하는 테이블로 교체하세요")
        
        # 사용자 쿼리 기반 권장사항
        if "9월" in user_query and "멤버십" in user_query:
            recommendations.append("9월 멤버십 성과 분석을 위해 t_member 테이블의 status 컬럼을 활용하세요")
            recommendations.append("ins_datetime 컬럼을 사용하여 날짜 기준 분석을 수행하세요")
        elif "크리에이터" in user_query or "creator" in user_query.lower():
            recommendations.append("크리에이터 관련 분석을 위해 t_creator 테이블을 활용하세요")
            recommendations.append("t_creator.member_no와 t_member.no를 조인하여 회원 정보를 연결하세요")
        elif "포스트" in user_query or "post" in user_query.lower():
            recommendations.append("포스트 관련 분석을 위해 t_post 테이블을 활용하세요")
            recommendations.append("view_count, like_count 컬럼을 사용하여 인기도를 분석하세요")
        
        return recommendations
