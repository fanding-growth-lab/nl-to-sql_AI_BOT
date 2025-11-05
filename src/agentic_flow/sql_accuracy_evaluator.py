"""
SQL 정확도 평가 모듈
생성된 SQL의 정확도를 측정하고 평가하는 시스템
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from core.db import execute_query, get_cached_db_schema
from core.logging import get_logger

logger = get_logger(__name__)


class AccuracyLevel(Enum):
    """정확도 수준"""
    PERFECT = "perfect"  # 완벽 (100%)
    HIGH = "high"  # 높음 (80-99%)
    MEDIUM = "medium"  # 중간 (50-79%)
    LOW = "low"  # 낮음 (0-49%)
    FAILED = "failed"  # 실패 (SQL 실행 불가)


@dataclass
class SQLAccuracyMetrics:
    """SQL 정확도 메트릭"""
    syntax_valid: bool  # 구문 유효성
    execution_success: bool  # 실행 성공 여부
    schema_compatibility: bool  # 스키마 호환성
    result_quality: float  # 결과 품질 (0.0-1.0)
    accuracy_score: float  # 종합 정확도 점수 (0.0-1.0)
    accuracy_level: AccuracyLevel  # 정확도 수준
    execution_error: Optional[str] = None  # 실행 오류 메시지
    row_count: Optional[int] = None  # 반환된 행 수
    execution_time: Optional[float] = None  # 실행 시간
    validation_issues: List[str] = None  # 검증 문제점
    suggestions: List[str] = None  # 개선 제안
    
    def __post_init__(self):
        if self.validation_issues is None:
            self.validation_issues = []
        if self.suggestions is None:
            self.suggestions = []


class SQLAccuracyEvaluator:
    """
    SQL 정확도 평가기
    생성된 SQL의 정확도를 종합적으로 평가
    """
    
    def __init__(self):
        """SQLAccuracyEvaluator 초기화"""
        self.db_schema = None
        self.evaluation_history: List[Dict[str, Any]] = []
        logger.info("SQLAccuracyEvaluator initialized")
    
    def evaluate_sql(
        self,
        sql_query: str,
        user_query: str,
        expected_result: Optional[List[Dict[str, Any]]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
        sql_params: Optional[Dict[str, Any]] = None
    ) -> SQLAccuracyMetrics:
        """
        SQL 정확도 평가
        
        Args:
            sql_query: 평가할 SQL 쿼리
            user_query: 원본 사용자 쿼리
            expected_result: 예상 결과 (선택사항, 수동 평가 시 사용)
            validation_result: 기존 검증 결과 (선택사항)
            
        Returns:
            SQLAccuracyMetrics: 정확도 평가 결과
        """
        logger.info(f"Evaluating SQL accuracy for query: {user_query[:50]}...")
        
        # 1. 구문 유효성 검증
        syntax_valid = self._validate_syntax(sql_query)
        
        # 2. 스키마 호환성 검증
        schema_compatibility = self._validate_schema_compatibility(sql_query)
        
        # 3. SQL 실행 테스트
        execution_result = self._test_execution(sql_query, sql_params=sql_params)
        execution_success = execution_result["success"]
        execution_error = execution_result.get("error")
        row_count = execution_result.get("row_count")
        execution_time = execution_result.get("execution_time")
        query_result = execution_result.get("result", [])
        
        # 4. 결과 품질 평가
        result_quality = self._assess_result_quality(
            query_result=query_result,
            user_query=user_query,
            expected_result=expected_result
        )
        
        # 5. 검증 결과 통합 (기존 validation_result 활용)
        validation_issues = []
        suggestions = []
        if validation_result:
            validation_issues.extend(validation_result.get("issues", []))
            suggestions.extend(validation_result.get("suggestions", []))
        
        # 6. 종합 정확도 점수 계산
        accuracy_score = self._calculate_accuracy_score(
            syntax_valid=syntax_valid,
            execution_success=execution_success,
            schema_compatibility=schema_compatibility,
            result_quality=result_quality
        )
        
        # 7. 정확도 수준 결정
        accuracy_level = self._determine_accuracy_level(accuracy_score)
        
        # 메트릭 생성
        metrics = SQLAccuracyMetrics(
            syntax_valid=syntax_valid,
            execution_success=execution_success,
            schema_compatibility=schema_compatibility,
            result_quality=result_quality,
            accuracy_score=accuracy_score,
            accuracy_level=accuracy_level,
            execution_error=execution_error,
            row_count=row_count,
            execution_time=execution_time,
            validation_issues=validation_issues,
            suggestions=suggestions
        )
        
        # 평가 이력 기록
        self._record_evaluation(
            user_query=user_query,
            sql_query=sql_query,
            metrics=metrics
        )
        
        logger.info(f"SQL accuracy evaluation completed: {accuracy_level.value} ({accuracy_score:.2%})")
        return metrics
    
    def _validate_syntax(self, sql_query: str) -> bool:
        """SQL 구문 유효성 검증"""
        try:
            # 기본 구문 검사
            sql_upper = sql_query.strip().upper()
            if not sql_upper:
                return False
            
            # 기본 SQL 키워드 검증
            valid_start_keywords = ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"]
            if not any(sql_upper.startswith(keyword) for keyword in valid_start_keywords):
                # 주석 제거 후 다시 확인
                lines = [line for line in sql_query.split('\n') if not line.strip().startswith('--')]
                cleaned_sql = ' '.join(lines).strip().upper()
                if not any(cleaned_sql.startswith(keyword) for keyword in valid_start_keywords):
                    return False
            
            # 괄호 매칭 검증
            open_count = sql_query.count('(')
            close_count = sql_query.count(')')
            if open_count != close_count:
                return False
            
            return True
        except Exception as e:
            logger.warning(f"Syntax validation error: {e}")
            return False
    
    def _validate_schema_compatibility(self, sql_query: str) -> bool:
        """스키마 호환성 검증"""
        try:
            if self.db_schema is None:
                self.db_schema = get_cached_db_schema()
            
            # 테이블명 추출
            import re
            table_pattern = r'FROM\s+`?(\w+)`?|JOIN\s+`?(\w+)`?'
            matches = re.findall(table_pattern, sql_query, re.IGNORECASE)
            table_names = [t for match in matches for t in match if t]
            
            # 테이블명 검증
            valid_tables = set(self.db_schema.keys())
            for table in table_names:
                if table and table not in valid_tables:
                    logger.debug(f"Invalid table name found: {table}")
                    return False
            
            return True
        except Exception as e:
            logger.warning(f"Schema compatibility validation error: {e}")
            return True  # 오류 발생 시 통과로 처리 (보수적 접근)
    
    def _test_execution(self, sql_query: str, sql_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """SQL 실행 테스트"""
        try:
            import time
            start_time = time.time()
            
            # SQL 실행 (sql_params 전달)
            result = execute_query(sql_query, params=sql_params, readonly=True)
            execution_time = time.time() - start_time
            
            # 결과 처리
            if isinstance(result, int):
                # 영향받은 행 수 (INSERT, UPDATE, DELETE 등)
                return {
                    "success": True,
                    "row_count": result,
                    "result": [],
                    "execution_time": execution_time
                }
            elif isinstance(result, list):
                # SELECT 쿼리 결과
                return {
                    "success": True,
                    "row_count": len(result),
                    "result": result,
                    "execution_time": execution_time
                }
            else:
                return {
                    "success": True,
                    "row_count": 0,
                    "result": [],
                    "execution_time": execution_time
                }
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"SQL execution test failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "row_count": None,
                "result": [],
                "execution_time": None
            }
    
    def _assess_result_quality(
        self,
        query_result: List[Dict[str, Any]],
        user_query: str,
        expected_result: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """결과 품질 평가"""
        try:
            # 예상 결과가 있으면 비교
            if expected_result is not None:
                return self._compare_results(query_result, expected_result)
            
            # 예상 결과가 없으면 휴리스틱 평가
            # 1. 결과가 비어있지 않은지
            if not query_result:
                return 0.5  # 결과가 없으면 중간 점수
            
            # 2. 결과 행 수가 합리적인지 (너무 많거나 적지 않은지)
            row_count = len(query_result)
            if row_count == 0:
                return 0.3  # 결과 없음
            elif row_count > 10000:
                return 0.7  # 결과가 너무 많음 (잠재적 문제)
            else:
                return 0.9  # 합리적인 결과 수
            
        except Exception as e:
            logger.warning(f"Result quality assessment error: {e}")
            return 0.5
    
    def _compare_results(
        self,
        actual: List[Dict[str, Any]],
        expected: List[Dict[str, Any]]
    ) -> float:
        """실제 결과와 예상 결과 비교"""
        try:
            if len(actual) != len(expected):
                # 행 수가 다르면 부분 점수
                return 0.5
            
            if len(actual) == 0:
                return 1.0 if len(expected) == 0 else 0.0
            
            # 각 행 비교
            matches = 0
            for i, (actual_row, expected_row) in enumerate(zip(actual, expected)):
                if actual_row == expected_row:
                    matches += 1
                else:
                    # 부분 매칭 (키만 비교)
                    if set(actual_row.keys()) == set(expected_row.keys()):
                        matches += 0.5
            
            return matches / len(actual)
        except Exception as e:
            logger.warning(f"Result comparison error: {e}")
            return 0.5
    
    def _calculate_accuracy_score(
        self,
        syntax_valid: bool,
        execution_success: bool,
        schema_compatibility: bool,
        result_quality: float
    ) -> float:
        """종합 정확도 점수 계산"""
        # 가중치:
        # - 구문 유효성: 30%
        # - 실행 성공: 40%
        # - 스키마 호환성: 20%
        # - 결과 품질: 10%
        
        score = 0.0
        
        if syntax_valid:
            score += 0.3
        elif execution_success:
            score += 0.1  # 실행은 성공했지만 구문이 완벽하지 않음
        
        if execution_success:
            score += 0.4
        
        if schema_compatibility:
            score += 0.2
        
        score += result_quality * 0.1
        
        return min(1.0, max(0.0, score))
    
    def _determine_accuracy_level(self, accuracy_score: float) -> AccuracyLevel:
        """정확도 수준 결정"""
        if accuracy_score >= 1.0:
            return AccuracyLevel.PERFECT
        elif accuracy_score >= 0.8:
            return AccuracyLevel.HIGH
        elif accuracy_score >= 0.5:
            return AccuracyLevel.MEDIUM
        elif accuracy_score > 0.0:
            return AccuracyLevel.LOW
        else:
            return AccuracyLevel.FAILED
    
    def _record_evaluation(
        self,
        user_query: str,
        sql_query: str,
        metrics: SQLAccuracyMetrics
    ) -> None:
        """평가 이력 기록"""
        try:
            evaluation_record = {
                "timestamp": datetime.now().isoformat(),
                "user_query": user_query,
                "sql_query": sql_query,
                "metrics": asdict(metrics),
                "accuracy_score": metrics.accuracy_score,
                "accuracy_level": metrics.accuracy_level.value
            }
            
            self.evaluation_history.append(evaluation_record)
            
            # 이력 크기 제한 (최근 1000개만 유지)
            if len(self.evaluation_history) > 1000:
                self.evaluation_history = self.evaluation_history[-1000:]
                
        except Exception as e:
            logger.warning(f"Failed to record evaluation: {e}")
    
    def get_accuracy_statistics(self) -> Dict[str, Any]:
        """정확도 통계 반환"""
        if not self.evaluation_history:
            return {
                "total_evaluations": 0,
                "average_accuracy": 0.0,
                "accuracy_distribution": {},
                "success_rate": 0.0
            }
        
        scores = [record["accuracy_score"] for record in self.evaluation_history]
        levels = [record["accuracy_level"] for record in self.evaluation_history]
        
        # 레벨별 분포
        level_counts = {}
        for level in AccuracyLevel:
            level_counts[level.value] = levels.count(level.value)
        
        # 성공률 (HIGH 이상)
        success_count = sum(1 for level in levels if level in [AccuracyLevel.PERFECT.value, AccuracyLevel.HIGH.value])
        
        return {
            "total_evaluations": len(self.evaluation_history),
            "average_accuracy": sum(scores) / len(scores) if scores else 0.0,
            "accuracy_distribution": level_counts,
            "success_rate": success_count / len(levels) if levels else 0.0,
            "min_accuracy": min(scores) if scores else 0.0,
            "max_accuracy": max(scores) if scores else 0.0
        }


# 싱글톤 인스턴스
_evaluator_instance: Optional[SQLAccuracyEvaluator] = None


def get_accuracy_evaluator() -> SQLAccuracyEvaluator:
    """SQL 정확도 평가기 싱글톤 인스턴스 반환"""
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = SQLAccuracyEvaluator()
    return _evaluator_instance

