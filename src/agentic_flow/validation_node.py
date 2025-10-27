#!/usr/bin/env python3
"""
SQL 검증 및 데이터 품질 확인 노드
LangGraph 파이프라인에서 중간 검증 단계를 담당하는 노드
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """검증 상태"""
    APPROVED = "approved"          # 승인됨
    NEEDS_REVIEW = "needs_review"  # 검토 필요
    REJECTED = "rejected"          # 거부됨
    PENDING = "pending"           # 대기 중


class ValidationLevel(Enum):
    """검증 수준"""
    BASIC = "basic"        # 기본 검증
    INTERMEDIATE = "intermediate"  # 중간 검증
    STRICT = "strict"      # 엄격한 검증


@dataclass
class ValidationResult:
    """검증 결과"""
    status: ValidationStatus
    confidence: float
    issues: List[str]
    warnings: List[str]
    suggestions: List[str]
    estimated_row_count: Optional[int] = None
    execution_time_estimate: Optional[float] = None
    data_preview: Optional[List[Dict]] = None
    validation_level: ValidationLevel = ValidationLevel.BASIC


@dataclass
class DataQualityMetrics:
    """데이터 품질 메트릭"""
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    validity_score: float
    overall_score: float


class ValidationNode:
    """
    SQL 검증 및 데이터 품질 확인을 담당하는 LangGraph 노드
    """
    
    def __init__(self, db_manager=None):
        """
        ValidationNode 초기화
        
        Args:
            db_manager: DatabaseManager 인스턴스
        """
        self.db_manager = db_manager
        self.validation_history = []
        self.quality_thresholds = {
            "high_confidence": 0.8,
            "medium_confidence": 0.6,
            "low_confidence": 0.4
        }
        
        logger.info("ValidationNode initialized")
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        SQL 검증 및 데이터 품질 확인 처리
        
        Args:
            state: LangGraph 상태 딕셔너리
            
        Returns:
            Dict[str, Any]: 업데이트된 상태
        """
        try:
            logger.info("Processing ValidationNode")
            
            user_query = state.get("user_query", "")
            sql_query = state.get("sql_query", "")
            context_elements = state.get("context_elements", [])
            
            logger.info(f"ValidationNode - user_query: {user_query}")
            logger.info(f"ValidationNode - sql_query: {sql_query}")
            
            # 1. SQL 구문 검증
            syntax_validation = self._validate_sql_syntax(sql_query)
            
            # 2. 스키마 호환성 검증
            schema_validation = self._validate_schema_compatibility(sql_query)
            
            # 3. 데이터 품질 평가
            quality_metrics = self._assess_data_quality(sql_query, context_elements)
            
            # 4. 실행 안전성 검증
            safety_validation = self._validate_execution_safety(sql_query)
            
            # 5. 성능 예측
            performance_prediction = self._predict_execution_performance(sql_query)
            
            # 6. 종합 검증 결과 생성
            validation_result = self._generate_validation_result(
                syntax_validation,
                schema_validation,
                quality_metrics,
                safety_validation,
                performance_prediction
            )
            
            # 7. 신뢰도 기반 처리 결정
            processing_decision = self._determine_processing_decision(validation_result)
            
            # 상태 업데이트
            state["validation_result"] = validation_result
            state["processing_decision"] = processing_decision
            state["needs_user_review"] = processing_decision["needs_user_review"]
            
            logger.info(f"Validation completed - Status: {validation_result.status.value}")
            logger.info(f"Confidence: {validation_result.confidence:.2f}")
            logger.info(f"Needs user review: {processing_decision['needs_user_review']}")
            
            return state
            
        except Exception as e:
            logger.error(f"ValidationNode processing failed: {str(e)}", exc_info=True)
            
            # 오류 시 안전한 폴백 결과 생성
            fallback_result = ValidationResult(
                status=ValidationStatus.NEEDS_REVIEW,
                confidence=0.0,
                issues=[f"검증 중 오류 발생: {str(e)}"],
                warnings=["시스템 오류로 인해 수동 검토가 필요합니다"],
                suggestions=["쿼리를 다시 작성하거나 관리자에게 문의하세요"]
            )
            
            state["validation_result"] = fallback_result
            state["processing_decision"] = {
                "needs_user_review": True,
                "reason": "system_error",
                "confidence": 0.0
            }
            state["needs_user_review"] = True
            
            return state
    
    def _validate_sql_syntax(self, sql_query: str) -> Dict[str, Any]:
        """SQL 구문 검증"""
        try:
            issues = []
            warnings = []
            suggestions = []
            
            # 기본 SQL 키워드 확인
            if not any(keyword in sql_query.upper() for keyword in ["SELECT", "INSERT", "UPDATE", "DELETE"]):
                issues.append("유효한 SQL 키워드가 없습니다")
            
            # 위험한 키워드 확인
            dangerous_keywords = ["DROP", "TRUNCATE", "ALTER", "CREATE", "GRANT", "REVOKE"]
            for keyword in dangerous_keywords:
                if keyword in sql_query.upper():
                    issues.append(f"위험한 키워드 발견: {keyword}")
                    suggestions.append("읽기 전용 쿼리만 허용됩니다")
            
            # 괄호 균형 확인
            open_parens = sql_query.count('(')
            close_parens = sql_query.count(')')
            if open_parens != close_parens:
                issues.append("괄호가 균형잡히지 않았습니다")
                suggestions.append("괄호를 다시 확인하세요")
            
            # 세미콜론 확인
            if sql_query.strip().endswith(';'):
                warnings.append("세미콜론으로 끝나는 쿼리입니다")
                suggestions.append("세미콜론을 제거하는 것을 권장합니다")
            
            # 기본적인 SELECT 구조 확인
            if "SELECT" in sql_query.upper():
                if "FROM" not in sql_query.upper():
                    issues.append("SELECT 문에 FROM 절이 없습니다")
                
                # 서브쿼리 깊이 확인
                subquery_depth = self._count_subquery_depth(sql_query)
                if subquery_depth > 3:
                    warnings.append(f"서브쿼리 깊이가 깊습니다 (깊이: {subquery_depth})")
                    suggestions.append("쿼리를 단순화하는 것을 고려하세요")
            
            return {
                "is_valid": len(issues) == 0,
                "issues": issues,
                "warnings": warnings,
                "suggestions": suggestions,
                "confidence": max(0.0, 1.0 - len(issues) * 0.2 - len(warnings) * 0.1)
            }
            
        except Exception as e:
            logger.error(f"SQL syntax validation failed: {str(e)}")
            return {
                "is_valid": False,
                "issues": [f"구문 검증 실패: {str(e)}"],
                "warnings": [],
                "suggestions": ["쿼리 구문을 다시 확인하세요"],
                "confidence": 0.0
            }
    
    def _validate_schema_compatibility(self, sql_query: str) -> Dict[str, Any]:
        """스키마 호환성 검증"""
        try:
            issues = []
            warnings = []
            suggestions = []
            
            # 테이블명 추출
            table_names = self._extract_table_names(sql_query)
            
            # 컬럼명 추출
            column_names = self._extract_column_names(sql_query)
            
            # 실제 스키마와 비교
            valid_tables = [
                "t_member", "t_fanding", "t_creator", "t_payment", "t_project", "t_tier",
                "t_post", "t_post_view_log", "t_post_like_log", "t_post_reply",
                "t_collection", "t_community", "t_course", "t_product", "t_review"
            ]
            # 실제 존재하는 컬럼들
            valid_columns = [
                "no", "id", "name", "title", "content", "status", "amount", "type", 
                "email", "phone", "address", "member_no", "creator_no", "post_no",
                "view_count", "like_count", "ins_datetime", "mod_datetime", "del_datetime"
            ]
            
            # 테이블명 검증
            for table in table_names:
                if table not in valid_tables:
                    issues.append(f"존재하지 않는 테이블: {table}")
                    suggestions.append(f"사용 가능한 테이블: {', '.join(valid_tables)}")
            
            # 컬럼명 검증 (간단한 패턴 매칭)
            for column in column_names:
                if not any(valid_col in column.lower() for valid_col in valid_columns):
                    warnings.append(f"의심스러운 컬럼명: {column}")
                    suggestions.append("컬럼명을 다시 확인하세요")
            
            # 조인 조건 검증
            if "JOIN" in sql_query.upper():
                join_issues = self._validate_join_conditions(sql_query)
                issues.extend(join_issues)
            
            return {
                "is_compatible": len(issues) == 0,
                "issues": issues,
                "warnings": warnings,
                "suggestions": suggestions,
                "confidence": max(0.0, 1.0 - len(issues) * 0.3 - len(warnings) * 0.1)
            }
            
        except Exception as e:
            logger.error(f"Schema compatibility validation failed: {str(e)}")
            return {
                "is_compatible": False,
                "issues": [f"스키마 검증 실패: {str(e)}"],
                "warnings": [],
                "suggestions": ["스키마 정보를 확인하세요"],
                "confidence": 0.0
            }
    
    def _assess_data_quality(self, sql_query: str, context_elements: List[Any]) -> DataQualityMetrics:
        """데이터 품질 평가"""
        try:
            # 기본 점수 초기화
            completeness_score = 1.0
            accuracy_score = 1.0
            consistency_score = 1.0
            validity_score = 1.0
            
            # SQL 복잡도에 따른 품질 점수 조정
            if "JOIN" in sql_query.upper():
                consistency_score -= 0.1  # 조인이 많을수록 일관성 위험
            
            if "WHERE" not in sql_query.upper() and "SELECT" in sql_query.upper():
                warnings = ["WHERE 조건이 없는 SELECT 문입니다"]
                completeness_score -= 0.2
            
            if "COUNT(*)" in sql_query.upper():
                accuracy_score += 0.1  # COUNT는 일반적으로 정확
            
            if "GROUP BY" in sql_query.upper():
                validity_score -= 0.1  # 그룹핑은 유효성 검증 필요
            
            # 컨텍스트 요소 기반 품질 평가
            for element in context_elements:
                if hasattr(element, 'context_type'):
                    if element.context_type.value == "business":
                        accuracy_score += 0.05  # 비즈니스 컨텍스트 있으면 정확도 증가
                    elif element.context_type.value == "temporal":
                        validity_score += 0.05  # 시간 컨텍스트 있으면 유효성 증가
            
            # 점수 범위 제한 (0.0 - 1.0)
            completeness_score = max(0.0, min(1.0, completeness_score))
            accuracy_score = max(0.0, min(1.0, accuracy_score))
            consistency_score = max(0.0, min(1.0, consistency_score))
            validity_score = max(0.0, min(1.0, validity_score))
            
            # 전체 점수 계산
            overall_score = (completeness_score + accuracy_score + consistency_score + validity_score) / 4
            
            return DataQualityMetrics(
                completeness_score=completeness_score,
                accuracy_score=accuracy_score,
                consistency_score=consistency_score,
                validity_score=validity_score,
                overall_score=overall_score
            )
            
        except Exception as e:
            logger.error(f"Data quality assessment failed: {str(e)}")
            return DataQualityMetrics(
                completeness_score=0.5,
                accuracy_score=0.5,
                consistency_score=0.5,
                validity_score=0.5,
                overall_score=0.5
            )
    
    def _validate_execution_safety(self, sql_query: str) -> Dict[str, Any]:
        """실행 안전성 검증"""
        try:
            issues = []
            warnings = []
            suggestions = []
            
            # 읽기 전용 쿼리만 허용
            write_operations = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER"]
            for operation in write_operations:
                if operation in sql_query.upper():
                    issues.append(f"쓰기 작업 감지: {operation}")
                    suggestions.append("읽기 전용 쿼리만 허용됩니다")
            
            # 대용량 데이터 조회 위험성 평가
            if "LIMIT" not in sql_query.upper() and "SELECT" in sql_query.upper():
                warnings.append("LIMIT 절이 없어 대용량 데이터 조회 가능성이 있습니다")
                suggestions.append("LIMIT 절을 추가하여 결과 수를 제한하세요")
            
            # 복잡한 집계 함수 확인
            complex_aggregations = ["STDDEV", "VARIANCE", "CORR", "REGR"]
            for agg in complex_aggregations:
                if agg in sql_query.upper():
                    warnings.append(f"복잡한 집계 함수 사용: {agg}")
                    suggestions.append("성능에 영향을 줄 수 있습니다")
            
            # 중첩 쿼리 깊이 확인
            nested_depth = self._count_nested_depth(sql_query)
            if nested_depth > 2:
                warnings.append(f"중첩 쿼리 깊이가 깊습니다 (깊이: {nested_depth})")
                suggestions.append("쿼리를 단순화하는 것을 고려하세요")
            
            return {
                "is_safe": len(issues) == 0,
                "issues": issues,
                "warnings": warnings,
                "suggestions": suggestions,
                "confidence": max(0.0, 1.0 - len(issues) * 0.4 - len(warnings) * 0.1)
            }
            
        except Exception as e:
            logger.error(f"Execution safety validation failed: {str(e)}")
            return {
                "is_safe": False,
                "issues": [f"안전성 검증 실패: {str(e)}"],
                "warnings": [],
                "suggestions": ["쿼리를 다시 검토하세요"],
                "confidence": 0.0
            }
    
    def _predict_execution_performance(self, sql_query: str) -> Dict[str, Any]:
        """실행 성능 예측"""
        try:
            # 기본 성능 지표
            estimated_time = 0.1  # 기본 0.1초
            estimated_rows = 100  # 기본 100행
            
            # 쿼리 복잡도에 따른 성능 조정
            if "JOIN" in sql_query.upper():
                join_count = sql_query.upper().count("JOIN")
                estimated_time += join_count * 0.2
                estimated_rows *= (join_count + 1)
            
            if "GROUP BY" in sql_query.upper():
                estimated_time += 0.3
                estimated_rows /= 2
            
            if "ORDER BY" in sql_query.upper():
                estimated_time += 0.2
            
            if "WHERE" in sql_query.upper():
                estimated_time += 0.1
            
            # 서브쿼리 영향
            subquery_count = self._count_subquery_depth(sql_query)
            estimated_time += subquery_count * 0.5
            estimated_rows *= (subquery_count + 1)
            
            # 성능 등급 결정
            if estimated_time < 1.0:
                performance_grade = "fast"
            elif estimated_time < 5.0:
                performance_grade = "medium"
            else:
                performance_grade = "slow"
            
            return {
                "estimated_time": estimated_time,
                "estimated_rows": estimated_rows,
                "performance_grade": performance_grade,
                "confidence": max(0.0, 1.0 - estimated_time * 0.1)
            }
            
        except Exception as e:
            logger.error(f"Performance prediction failed: {str(e)}")
            return {
                "estimated_time": 1.0,
                "estimated_rows": 1000,
                "performance_grade": "unknown",
                "confidence": 0.5
            }
    
    def _generate_validation_result(self, syntax_validation: Dict, schema_validation: Dict, 
                                  quality_metrics: DataQualityMetrics, safety_validation: Dict,
                                  performance_prediction: Dict) -> ValidationResult:
        """종합 검증 결과 생성"""
        try:
            # 모든 이슈와 경고 수집
            all_issues = []
            all_issues.extend(syntax_validation.get("issues", []))
            all_issues.extend(schema_validation.get("issues", []))
            all_issues.extend(safety_validation.get("issues", []))
            
            all_warnings = []
            all_warnings.extend(syntax_validation.get("warnings", []))
            all_warnings.extend(schema_validation.get("warnings", []))
            all_warnings.extend(safety_validation.get("warnings", []))
            
            all_suggestions = []
            all_suggestions.extend(syntax_validation.get("suggestions", []))
            all_suggestions.extend(schema_validation.get("suggestions", []))
            all_suggestions.extend(safety_validation.get("suggestions", []))
            
            # 종합 신뢰도 계산
            syntax_confidence = syntax_validation.get("confidence", 0.0)
            schema_confidence = schema_validation.get("confidence", 0.0)
            safety_confidence = safety_validation.get("confidence", 0.0)
            performance_confidence = performance_prediction.get("confidence", 0.0)
            
            overall_confidence = (syntax_confidence + schema_confidence + safety_confidence + 
                                performance_confidence + quality_metrics.overall_score) / 5
            
            # 상태 결정 (더 관대한 임계값 적용)
            if len(all_issues) == 0 and overall_confidence >= 0.7:
                status = ValidationStatus.APPROVED
            elif len(all_issues) <= 1 and overall_confidence >= 0.6:  # 경미한 이슈 허용
                status = ValidationStatus.NEEDS_REVIEW
            elif overall_confidence >= 0.8:  # 높은 신뢰도면 이슈가 있어도 검토로
                status = ValidationStatus.NEEDS_REVIEW
            else:
                status = ValidationStatus.REJECTED
            
            return ValidationResult(
                status=status,
                confidence=overall_confidence,
                issues=all_issues,
                warnings=all_warnings,
                suggestions=all_suggestions,
                estimated_row_count=performance_prediction.get("estimated_rows"),
                execution_time_estimate=performance_prediction.get("estimated_time"),
                validation_level=ValidationLevel.BASIC
            )
            
        except Exception as e:
            logger.error(f"Validation result generation failed: {str(e)}")
            return ValidationResult(
                status=ValidationStatus.REJECTED,
                confidence=0.0,
                issues=[f"검증 결과 생성 실패: {str(e)}"],
                warnings=[],
                suggestions=["시스템 오류로 인해 쿼리를 다시 시도하세요"]
            )
    
    def _determine_processing_decision(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """신뢰도 기반 처리 결정"""
        try:
            decision = {
                "needs_user_review": False,
                "reason": "",
                "confidence": validation_result.confidence,
                "auto_approve": False
            }
            
            if validation_result.status == ValidationStatus.APPROVED:
                if validation_result.confidence >= self.quality_thresholds["high_confidence"]:
                    decision["auto_approve"] = True
                    decision["reason"] = "high_confidence"
                else:
                    decision["needs_user_review"] = True
                    decision["reason"] = "medium_confidence"
            
            elif validation_result.status == ValidationStatus.NEEDS_REVIEW:
                decision["needs_user_review"] = True
                decision["reason"] = "validation_issues"
            
            elif validation_result.status == ValidationStatus.REJECTED:
                decision["needs_user_review"] = True
                decision["reason"] = "critical_issues"
            
            return decision
            
        except Exception as e:
            logger.error(f"Processing decision failed: {str(e)}")
            return {
                "needs_user_review": True,
                "reason": "decision_error",
                "confidence": 0.0,
                "auto_approve": False
            }
    
    def _extract_table_names(self, sql_query: str) -> List[str]:
        """SQL에서 테이블명 추출"""
        try:
            # 간단한 정규식으로 테이블명 추출
            table_pattern = r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)'
            tables = re.findall(table_pattern, sql_query, re.IGNORECASE)
            
            # JOIN 절에서도 테이블명 추출
            join_pattern = r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)'
            join_tables = re.findall(join_pattern, sql_query, re.IGNORECASE)
            
            tables.extend(join_tables)
            return list(set(tables))  # 중복 제거
            
        except Exception as e:
            logger.error(f"Table name extraction failed: {str(e)}")
            return []
    
    def _extract_column_names(self, sql_query: str) -> List[str]:
        """SQL에서 컬럼명 추출"""
        try:
            # SELECT 절에서 컬럼명 추출
            select_pattern = r'SELECT\s+(.*?)\s+FROM'
            select_match = re.search(select_pattern, sql_query, re.IGNORECASE | re.DOTALL)
            
            if select_match:
                select_clause = select_match.group(1)
                # 간단한 컬럼명 추출 (복잡한 경우는 제외)
                column_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)'
                columns = re.findall(column_pattern, select_clause)
                return columns
            
            return []
            
        except Exception as e:
            logger.error(f"Column name extraction failed: {str(e)}")
            return []
    
    def _validate_join_conditions(self, sql_query: str) -> List[str]:
        """조인 조건 검증"""
        try:
            issues = []
            
            # ON 절이 있는지 확인
            if "JOIN" in sql_query.upper() and "ON" not in sql_query.upper():
                issues.append("JOIN 절에 ON 조건이 없습니다")
            
            # 등가 조인 확인
            if "=" in sql_query and "JOIN" in sql_query.upper():
                # 간단한 등가 조인 패턴 확인
                pass  # 추가 검증 로직
            
            return issues
            
        except Exception as e:
            logger.error(f"Join condition validation failed: {str(e)}")
            return [f"조인 조건 검증 실패: {str(e)}"]
    
    def _count_subquery_depth(self, sql_query: str) -> int:
        """서브쿼리 깊이 계산"""
        try:
            # 괄호 개수로 서브쿼리 깊이 추정
            depth = 0
            max_depth = 0
            
            for char in sql_query:
                if char == '(':
                    depth += 1
                    max_depth = max(max_depth, depth)
                elif char == ')':
                    depth -= 1
            
            return max(0, max_depth - 1)  # 최상위 SELECT 제외
            
        except Exception as e:
            logger.error(f"Subquery depth counting failed: {str(e)}")
            return 0
    
    def _count_nested_depth(self, sql_query: str) -> int:
        """중첩 깊이 계산"""
        try:
            # SELECT 문 중첩 깊이 계산
            select_count = sql_query.upper().count("SELECT")
            return max(0, select_count - 1)
            
        except Exception as e:
            logger.error(f"Nested depth counting failed: {str(e)}")
            return 0
    
    def get_validation_history(self) -> List[ValidationResult]:
        """검증 히스토리 반환"""
        return self.validation_history.copy()
    
    def clear_validation_history(self):
        """검증 히스토리 정리"""
        self.validation_history.clear()
