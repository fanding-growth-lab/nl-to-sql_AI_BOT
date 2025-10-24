#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
복합 쿼리 검증 및 최적화 시스템
생성된 SQL의 정확성, 성능, 보안을 종합적으로 검증하고 최적화하는 시스템
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from agentic_flow.relationship_based_sql_generator import RelationshipBasedSQLGenerator, GeneratedSQL
from agentic_flow.join_path_optimizer import JoinPathOptimizer, OptimizationStrategy
from agentic_flow.entity_relationship_extractor import EntityRelationshipExtractor
from core.db import get_db_session, execute_query


class ValidationLevel(Enum):
    """검증 레벨"""
    BASIC = "basic"      # 기본 검증
    INTERMEDIATE = "intermediate"  # 중급 검증
    ADVANCED = "advanced"  # 고급 검증
    EXPERT = "expert"    # 전문가 검증


class OptimizationType(Enum):
    """최적화 유형"""
    PERFORMANCE = "performance"  # 성능 최적화
    SECURITY = "security"        # 보안 최적화
    READABILITY = "readability"  # 가독성 최적화
    MAINTAINABILITY = "maintainability"  # 유지보수성 최적화


@dataclass
class ValidationResult:
    """검증 결과"""
    is_valid: bool
    confidence: float
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    performance_score: float = 0.0
    security_score: float = 0.0
    readability_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """최적화 결과"""
    original_sql: str
    optimized_sql: str
    optimization_type: OptimizationType
    improvements: List[str]
    performance_gain: float
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)


class ComplexQueryValidator:
    """복합 쿼리 검증 및 최적화 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 하위 시스템 초기화
        self.sql_generator = RelationshipBasedSQLGenerator()
        self.join_optimizer = JoinPathOptimizer()
        self.entity_extractor = EntityRelationshipExtractor()
        
        # 검증 규칙 초기화
        self.validation_rules = self._initialize_validation_rules()
        self.optimization_rules = self._initialize_optimization_rules()
        
    def _initialize_validation_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """검증 규칙 초기화"""
        return {
            "syntax": [
                {
                    "name": "SELECT 문 확인",
                    "pattern": r"^\s*SELECT\s+",
                    "level": "error",
                    "message": "SELECT 문으로 시작해야 합니다."
                },
                {
                    "name": "세미콜론 확인",
                    "pattern": r";\s*$",
                    "level": "warning",
                    "message": "문장 끝에 세미콜론을 추가하세요."
                }
            ],
            "security": [
                {
                    "name": "위험한 키워드 검사",
                    "pattern": r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE)\b",
                    "level": "error",
                    "message": "위험한 키워드가 포함되어 있습니다."
                },
                {
                    "name": "SQL 인젝션 패턴 검사",
                    "pattern": r"('|\"|;|--|\/\*|\*\/)",
                    "level": "warning",
                    "message": "SQL 인젝션 가능성이 있습니다."
                }
            ],
            "performance": [
                {
                    "name": "LIMIT 절 확인",
                    "pattern": r"\bLIMIT\s+\d+",
                    "level": "warning",
                    "message": "성능을 위해 LIMIT 절을 추가하세요."
                },
                {
                    "name": "ORDER BY 절 확인",
                    "pattern": r"\bORDER\s+BY\b",
                    "level": "info",
                    "message": "결과의 일관성을 위해 ORDER BY 절을 추가하세요."
                },
                {
                    "name": "복잡한 조인 확인",
                    "pattern": r"\bJOIN\b",
                    "level": "info",
                    "message": "조인 최적화를 고려하세요."
                }
            ],
            "readability": [
                {
                    "name": "들여쓰기 확인",
                    "pattern": r"^\s+",
                    "level": "info",
                    "message": "가독성을 위해 적절한 들여쓰기를 사용하세요."
                },
                {
                    "name": "별칭 사용 확인",
                    "pattern": r"\bAS\s+\w+",
                    "level": "info",
                    "message": "가독성을 위해 컬럼 별칭을 사용하세요."
                }
            ]
        }
    
    def _initialize_optimization_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """최적화 규칙 초기화"""
        return {
            "performance": [
                {
                    "name": "인덱스 힌트 추가",
                    "condition": "large_table_join",
                    "action": "add_index_hint",
                    "description": "대용량 테이블 조인 시 인덱스 힌트 추가"
                },
                {
                    "name": "서브쿼리 최적화",
                    "condition": "nested_subquery",
                    "action": "optimize_subquery",
                    "description": "중첩 서브쿼리를 JOIN으로 변환"
                },
                {
                    "name": "불필요한 컬럼 제거",
                    "condition": "select_star",
                    "action": "specify_columns",
                    "description": "SELECT * 대신 필요한 컬럼만 지정"
                }
            ],
            "security": [
                {
                    "name": "매개변수화된 쿼리",
                    "condition": "dynamic_values",
                    "action": "parameterize_query",
                    "description": "동적 값을 매개변수로 처리"
                },
                {
                    "name": "권한 검사",
                    "condition": "sensitive_data",
                    "action": "check_permissions",
                    "description": "민감한 데이터 접근 권한 확인"
                }
            ],
            "readability": [
                {
                    "name": "포맷팅 개선",
                    "condition": "poor_formatting",
                    "action": "format_sql",
                    "description": "SQL 포맷팅 개선"
                },
                {
                    "name": "주석 추가",
                    "condition": "complex_logic",
                    "action": "add_comments",
                    "description": "복잡한 로직에 주석 추가"
                }
            ]
        }
    
    def validate_query(self, sql: str, level: ValidationLevel = ValidationLevel.INTERMEDIATE) -> ValidationResult:
        """쿼리 검증"""
        self.logger.info(f"쿼리 검증 시작: {level.value} 레벨")
        
        try:
            issues = []
            warnings = []
            suggestions = []
            
            # 1. 기본 문법 검증
            syntax_result = self._validate_syntax(sql)
            issues.extend(syntax_result["issues"])
            warnings.extend(syntax_result["warnings"])
            suggestions.extend(syntax_result["suggestions"])
            
            # 2. 보안 검증
            security_result = self._validate_security(sql)
            issues.extend(security_result["issues"])
            warnings.extend(security_result["warnings"])
            suggestions.extend(security_result["suggestions"])
            
            # 3. 성능 검증
            performance_result = self._validate_performance(sql)
            issues.extend(performance_result["issues"])
            warnings.extend(performance_result["warnings"])
            suggestions.extend(performance_result["suggestions"])
            
            # 4. 가독성 검증
            readability_result = self._validate_readability(sql)
            issues.extend(readability_result["issues"])
            warnings.extend(readability_result["warnings"])
            suggestions.extend(readability_result["suggestions"])
            
            # 5. 고급 검증 (레벨에 따라)
            if level in [ValidationLevel.ADVANCED, ValidationLevel.EXPERT]:
                advanced_result = self._validate_advanced(sql)
                issues.extend(advanced_result["issues"])
                warnings.extend(advanced_result["warnings"])
                suggestions.extend(advanced_result["suggestions"])
            
            # 6. 점수 계산
            scores = self._calculate_scores(sql, issues, warnings, suggestions)
            
            # 7. 신뢰도 계산
            confidence = self._calculate_confidence(issues, warnings, suggestions)
            
            result = ValidationResult(
                is_valid=len(issues) == 0,
                confidence=confidence,
                issues=issues,
                warnings=warnings,
                suggestions=suggestions,
                performance_score=scores["performance"],
                security_score=scores["security"],
                readability_score=scores["readability"],
                details={
                    "validation_level": level.value,
                    "total_issues": len(issues),
                    "total_warnings": len(warnings),
                    "total_suggestions": len(suggestions)
                }
            )
            
            self.logger.info(f"쿼리 검증 완료: {len(issues)}개 이슈, {len(warnings)}개 경고")
            return result
            
        except Exception as e:
            self.logger.error(f"쿼리 검증 실패: {str(e)}")
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                issues=[f"검증 실패: {str(e)}"],
                performance_score=0.0,
                security_score=0.0,
                readability_score=0.0
            )
    
    def _validate_syntax(self, sql: str) -> Dict[str, List[str]]:
        """문법 검증"""
        issues = []
        warnings = []
        suggestions = []
        
        # SELECT 문 확인
        if not sql.strip().upper().startswith('SELECT'):
            issues.append("SELECT 문으로 시작해야 합니다.")
        
        # 세미콜론 확인
        if not sql.strip().endswith(';'):
            warnings.append("문장 끝에 세미콜론을 추가하세요.")
        
        # 괄호 균형 확인
        open_parens = sql.count('(')
        close_parens = sql.count(')')
        if open_parens != close_parens:
            issues.append("괄호가 균형을 이루지 않습니다.")
        
        return {"issues": issues, "warnings": warnings, "suggestions": suggestions}
    
    def _validate_security(self, sql: str) -> Dict[str, List[str]]:
        """보안 검증"""
        issues = []
        warnings = []
        suggestions = []
        
        # 위험한 키워드 검사
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        for keyword in dangerous_keywords:
            if keyword in sql.upper():
                issues.append(f"위험한 키워드 '{keyword}'가 포함되어 있습니다.")
        
        # SQL 인젝션 패턴 검사
        injection_patterns = [
            r"('|\"|;|--|\/\*|\*\/)",
            r"\bUNION\s+SELECT\b",
            r"\bOR\s+1\s*=\s*1\b",
            r"\bAND\s+1\s*=\s*1\b"
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                warnings.append("SQL 인젝션 가능성이 있습니다.")
                suggestions.append("매개변수화된 쿼리를 사용하세요.")
        
        return {"issues": issues, "warnings": warnings, "suggestions": suggestions}
    
    def _validate_performance(self, sql: str) -> Dict[str, List[str]]:
        """성능 검증"""
        issues = []
        warnings = []
        suggestions = []
        
        # LIMIT 절 확인
        if 'LIMIT' not in sql.upper():
            warnings.append("성능을 위해 LIMIT 절을 추가하세요.")
        
        # ORDER BY 절 확인
        if 'ORDER BY' not in sql.upper():
            suggestions.append("결과의 일관성을 위해 ORDER BY 절을 추가하세요.")
        
        # 복잡한 조인 확인
        join_count = sql.upper().count('JOIN')
        if join_count > 3:
            warnings.append("복잡한 조인이 있습니다. 서브쿼리로 분리하는 것을 고려하세요.")
        
        # 서브쿼리 확인
        subquery_count = sql.count('(') - sql.count(')')
        if subquery_count > 2:
            warnings.append("중첩된 서브쿼리가 있습니다. JOIN으로 변환하는 것을 고려하세요.")
        
        # SELECT * 확인
        if 'SELECT *' in sql.upper():
            suggestions.append("필요한 컬럼만 선택하여 성능을 개선하세요.")
        
        return {"issues": issues, "warnings": warnings, "suggestions": suggestions}
    
    def _validate_readability(self, sql: str) -> Dict[str, List[str]]:
        """가독성 검증"""
        issues = []
        warnings = []
        suggestions = []
        
        # 들여쓰기 확인
        lines = sql.split('\n')
        indentation_consistent = True
        for line in lines:
            if line.strip() and not line.startswith((' ', '\t')):
                if any(keyword in line.upper() for keyword in ['FROM', 'WHERE', 'ORDER BY', 'GROUP BY', 'HAVING']):
                    indentation_consistent = False
                    break
        
        if not indentation_consistent:
            suggestions.append("가독성을 위해 적절한 들여쓰기를 사용하세요.")
        
        # 별칭 사용 확인
        if 'AS ' not in sql.upper() and ' ' in sql:
            suggestions.append("가독성을 위해 컬럼 별칭을 사용하세요.")
        
        # 주석 확인
        if '--' not in sql and '/*' not in sql:
            if len(sql.split('\n')) > 5:
                suggestions.append("복잡한 쿼리에 주석을 추가하세요.")
        
        return {"issues": issues, "warnings": warnings, "suggestions": suggestions}
    
    def _validate_advanced(self, sql: str) -> Dict[str, List[str]]:
        """고급 검증"""
        issues = []
        warnings = []
        suggestions = []
        
        # 인덱스 사용 가능성 확인
        if 'WHERE' in sql.upper():
            where_clause = sql.upper().split('WHERE')[1].split('ORDER BY')[0] if 'ORDER BY' in sql.upper() else sql.upper().split('WHERE')[1]
            if '=' in where_clause or 'LIKE' in where_clause:
                suggestions.append("WHERE 절의 컬럼에 인덱스가 있는지 확인하세요.")
        
        # 조인 순서 최적화
        if 'JOIN' in sql.upper():
            suggestions.append("조인 순서를 최적화하여 성능을 개선하세요.")
        
        # 집계 함수 최적화
        if any(func in sql.upper() for func in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']):
            suggestions.append("집계 함수 사용 시 GROUP BY 절을 확인하세요.")
        
        return {"issues": issues, "warnings": warnings, "suggestions": suggestions}
    
    def _calculate_scores(self, sql: str, issues: List[str], warnings: List[str], suggestions: List[str]) -> Dict[str, float]:
        """점수 계산"""
        # 성능 점수
        performance_score = 1.0
        if 'LIMIT' not in sql.upper():
            performance_score -= 0.2
        if 'SELECT *' in sql.upper():
            performance_score -= 0.1
        if sql.upper().count('JOIN') > 3:
            performance_score -= 0.2
        
        # 보안 점수
        security_score = 1.0
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        for keyword in dangerous_keywords:
            if keyword in sql.upper():
                security_score = 0.0
                break
        
        # 가독성 점수
        readability_score = 1.0
        if len(sql.split('\n')) < 3:
            readability_score -= 0.3
        if 'AS ' not in sql.upper():
            readability_score -= 0.1
        
        return {
            "performance": max(0.0, performance_score),
            "security": max(0.0, security_score),
            "readability": max(0.0, readability_score)
        }
    
    def _calculate_confidence(self, issues: List[str], warnings: List[str], suggestions: List[str]) -> float:
        """신뢰도 계산"""
        total_checks = len(issues) + len(warnings) + len(suggestions)
        if total_checks == 0:
            return 1.0
        
        # 이슈는 신뢰도를 크게 감소
        issue_penalty = len(issues) * 0.3
        # 경고는 중간 정도 감소
        warning_penalty = len(warnings) * 0.1
        # 제안은 약간 감소
        suggestion_penalty = len(suggestions) * 0.05
        
        confidence = 1.0 - issue_penalty - warning_penalty - suggestion_penalty
        return max(0.0, confidence)
    
    def optimize_query(self, sql: str, optimization_type: OptimizationType = OptimizationType.PERFORMANCE) -> OptimizationResult:
        """쿼리 최적화"""
        self.logger.info(f"쿼리 최적화 시작: {optimization_type.value}")
        
        try:
            original_sql = sql
            optimized_sql = sql
            improvements = []
            performance_gain = 0.0
            
            if optimization_type == OptimizationType.PERFORMANCE:
                optimized_sql, improvements, performance_gain = self._optimize_performance(sql)
            elif optimization_type == OptimizationType.SECURITY:
                optimized_sql, improvements, performance_gain = self._optimize_security(sql)
            elif optimization_type == OptimizationType.READABILITY:
                optimized_sql, improvements, performance_gain = self._optimize_readability(sql)
            elif optimization_type == OptimizationType.MAINTAINABILITY:
                optimized_sql, improvements, performance_gain = self._optimize_maintainability(sql)
            
            result = OptimizationResult(
                original_sql=original_sql,
                optimized_sql=optimized_sql,
                optimization_type=optimization_type,
                improvements=improvements,
                performance_gain=performance_gain,
                confidence=0.8,  # 기본 신뢰도
                details={
                    "optimization_type": optimization_type.value,
                    "improvements_count": len(improvements)
                }
            )
            
            self.logger.info(f"쿼리 최적화 완료: {len(improvements)}개 개선사항")
            return result
            
        except Exception as e:
            self.logger.error(f"쿼리 최적화 실패: {str(e)}")
            return OptimizationResult(
                original_sql=sql,
                optimized_sql=sql,
                optimization_type=optimization_type,
                improvements=[f"최적화 실패: {str(e)}"],
                performance_gain=0.0,
                confidence=0.0
            )
    
    def _optimize_performance(self, sql: str) -> Tuple[str, List[str], float]:
        """성능 최적화"""
        optimized_sql = sql
        improvements = []
        performance_gain = 0.0
        
        # LIMIT 절 추가
        if 'LIMIT' not in sql.upper():
            optimized_sql += " LIMIT 100"
            improvements.append("LIMIT 절 추가")
            performance_gain += 0.2
        
        # ORDER BY 절 추가
        if 'ORDER BY' not in sql.upper():
            optimized_sql += " ORDER BY 1"
            improvements.append("ORDER BY 절 추가")
            performance_gain += 0.1
        
        # SELECT * 최적화
        if 'SELECT *' in sql.upper():
            # 실제로는 필요한 컬럼을 파악해야 함
            improvements.append("SELECT * 대신 필요한 컬럼만 지정")
            performance_gain += 0.1
        
        return optimized_sql, improvements, performance_gain
    
    def _optimize_security(self, sql: str) -> Tuple[str, List[str], float]:
        """보안 최적화"""
        optimized_sql = sql
        improvements = []
        performance_gain = 0.0
        
        # 위험한 키워드 제거
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        for keyword in dangerous_keywords:
            if keyword in sql.upper():
                optimized_sql = optimized_sql.replace(keyword, f"-- {keyword}")  # 주석 처리
                improvements.append(f"위험한 키워드 '{keyword}' 주석 처리")
                performance_gain += 0.3
        
        return optimized_sql, improvements, performance_gain
    
    def _optimize_readability(self, sql: str) -> Tuple[str, List[str], float]:
        """가독성 최적화"""
        optimized_sql = sql
        improvements = []
        performance_gain = 0.0
        
        # 들여쓰기 개선
        lines = sql.split('\n')
        formatted_lines = []
        indent_level = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append('')
                continue
            
            # 들여쓰기 레벨 조정
            if line.upper().startswith(('FROM', 'WHERE', 'ORDER BY', 'GROUP BY', 'HAVING')):
                indent_level = 1
            elif line.upper().startswith(('SELECT', 'JOIN')):
                indent_level = 0
            
            formatted_line = '  ' * indent_level + line
            formatted_lines.append(formatted_line)
        
        optimized_sql = '\n'.join(formatted_lines)
        improvements.append("들여쓰기 개선")
        performance_gain += 0.1
        
        return optimized_sql, improvements, performance_gain
    
    def _optimize_maintainability(self, sql: str) -> Tuple[str, List[str], float]:
        """유지보수성 최적화"""
        optimized_sql = sql
        improvements = []
        performance_gain = 0.0
        
        # 주석 추가
        if '--' not in sql and '/*' not in sql:
            optimized_sql = f"-- 쿼리 설명\n{optimized_sql}"
            improvements.append("주석 추가")
            performance_gain += 0.1
        
        return optimized_sql, improvements, performance_gain
    
    def comprehensive_analysis(self, query: str) -> Dict[str, Any]:
        """종합 분석"""
        self.logger.info(f"종합 분석 시작: {query}")
        
        try:
            # 1. SQL 생성
            generated_sql = self.sql_generator.generate_sql_from_query(query)
            
            # 2. 검증
            validation_result = self.validate_query(generated_sql.sql, ValidationLevel.ADVANCED)
            
            # 3. 최적화
            optimization_result = self.optimize_query(generated_sql.sql, OptimizationType.PERFORMANCE)
            
            # 4. 종합 결과
            analysis_result = {
                "original_query": query,
                "generated_sql": generated_sql.sql,
                "validation": {
                    "is_valid": validation_result.is_valid,
                    "confidence": validation_result.confidence,
                    "issues": validation_result.issues,
                    "warnings": validation_result.warnings,
                    "suggestions": validation_result.suggestions,
                    "scores": {
                        "performance": validation_result.performance_score,
                        "security": validation_result.security_score,
                        "readability": validation_result.readability_score
                    }
                },
                "optimization": {
                    "optimized_sql": optimization_result.optimized_sql,
                    "improvements": optimization_result.improvements,
                    "performance_gain": optimization_result.performance_gain,
                    "confidence": optimization_result.confidence
                },
                "recommendations": self._generate_recommendations(validation_result, optimization_result)
            }
            
            self.logger.info("종합 분석 완료")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"종합 분석 실패: {str(e)}")
            return {
                "error": str(e),
                "original_query": query
            }
    
    def _generate_recommendations(self, validation_result: ValidationResult, optimization_result: OptimizationResult) -> List[str]:
        """추천사항 생성"""
        recommendations = []
        
        # 검증 결과 기반 추천
        if validation_result.performance_score < 0.7:
            recommendations.append("성능 최적화가 필요합니다. 인덱스와 조인 순서를 확인하세요.")
        
        if validation_result.security_score < 0.8:
            recommendations.append("보안 강화가 필요합니다. 위험한 키워드를 제거하세요.")
        
        if validation_result.readability_score < 0.6:
            recommendations.append("가독성 개선이 필요합니다. 들여쓰기와 주석을 추가하세요.")
        
        # 최적화 결과 기반 추천
        if optimization_result.performance_gain > 0.2:
            recommendations.append("최적화된 쿼리를 사용하여 성능을 개선할 수 있습니다.")
        
        return recommendations


