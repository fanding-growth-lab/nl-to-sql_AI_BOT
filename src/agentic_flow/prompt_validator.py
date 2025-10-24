"""
프롬프트 변수 검증 및 디버깅 시스템
ValidationNode에서 발생하는 프롬프트 변수 매핑 오류를 해결합니다.
"""

import re
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """프롬프트 검증 결과"""
    is_valid: bool
    missing_vars: List[str]
    unused_vars: List[str]
    template_vars: List[str]
    provided_vars: List[str]
    suggestions: List[str]

@dataclass
class ExecutionRecord:
    """프롬프트 실행 기록"""
    timestamp: str
    template: str
    variables: Dict[str, Any]
    execution_time: float
    success: bool
    error: Optional[str] = None
    result: Optional[Any] = None

class PromptValidator:
    """프롬프트 변수 검증기"""
    
    def __init__(self):
        self.validation_history = []
        logger.info("PromptValidator initialized")
    
    def validate_prompt_variables(self, template: str, variables: Dict[str, Any]) -> ValidationResult:
        """
        프롬프트 템플릿과 제공된 변수 간의 일치 여부 검증
        
        Args:
            template: 프롬프트 템플릿 문자열
            variables: 제공된 변수 딕셔너리
            
        Returns:
            ValidationResult: 검증 결과
        """
        try:
            # 템플릿에서 변수 추출 (중괄호 패턴)
            template_vars = set(self._extract_template_variables(template))
            provided_vars = set(variables.keys())
            
            # 누락된 변수와 사용되지 않은 변수 식별
            missing_vars = list(template_vars - provided_vars)
            unused_vars = list(provided_vars - template_vars)
            
            # 제안사항 생성
            suggestions = self._generate_suggestions(missing_vars, unused_vars, template)
            
            # 검증 결과 생성
            result = ValidationResult(
                is_valid=len(missing_vars) == 0,
                missing_vars=missing_vars,
                unused_vars=unused_vars,
                template_vars=list(template_vars),
                provided_vars=list(provided_vars),
                suggestions=suggestions
            )
            
            # 검증 기록 저장
            self.validation_history.append({
                "timestamp": datetime.now().isoformat(),
                "template": template[:100] + "..." if len(template) > 100 else template,
                "variables": list(variables.keys()),
                "result": result
            })
            
            logger.debug(f"Prompt validation completed: valid={result.is_valid}, missing={len(missing_vars)}")
            return result
            
        except Exception as e:
            logger.error(f"Prompt validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                missing_vars=[],
                unused_vars=[],
                template_vars=[],
                provided_vars=[],
                suggestions=[f"검증 오류: {str(e)}"]
            )
    
    def _extract_template_variables(self, template: str) -> List[str]:
        """템플릿에서 변수 추출"""
        # 중괄호로 둘러싸인 변수명 추출
        # 예: {name}, {data}, {query} 등
        pattern = r'\{([^{}]+)\}'
        matches = re.findall(pattern, template)
        
        # 변수명 정리 (공백 제거, 중복 제거)
        variables = []
        for match in matches:
            var_name = match.strip()
            if var_name and var_name not in variables:
                variables.append(var_name)
        
        return variables
    
    def _generate_suggestions(self, missing_vars: List[str], unused_vars: List[str], template: str) -> List[str]:
        """검증 결과에 따른 제안사항 생성"""
        suggestions = []
        
        if missing_vars:
            suggestions.append(f"누락된 변수: {', '.join(missing_vars)}")
            suggestions.append("변수 값을 제공하거나 템플릿에서 해당 변수를 제거하세요")
        
        if unused_vars:
            suggestions.append(f"사용되지 않은 변수: {', '.join(unused_vars)}")
            suggestions.append("불필요한 변수를 제거하거나 템플릿에서 활용하세요")
        
        # 중괄호 이스케이프 제안
        if '{' in template and '}' in template:
            suggestions.append("중괄호 이스케이프: {{}} 또는 raw 문자열(r\"\") 사용을 고려하세요")
        
        return suggestions
    
    def get_validation_history(self) -> List[Dict[str, Any]]:
        """검증 기록 반환"""
        return self.validation_history
    
    def clear_history(self):
        """검증 기록 초기화"""
        self.validation_history = []
        logger.info("Validation history cleared")

class PromptDebugger:
    """프롬프트 디버거"""
    
    def __init__(self, log_level: int = logging.INFO):
        self.logger = self._setup_logger(log_level)
        self.execution_history = []
        self.validator = PromptValidator()
        
    def _setup_logger(self, log_level: int) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger("prompt_debugger")
        logger.setLevel(log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def debug_prompt_execution(self, template: str, variables: Dict[str, Any], 
                             execution_func, max_retries: int = 3) -> Optional[Any]:
        """
        프롬프트 실행 과정 디버깅
        
        Args:
            template: 프롬프트 템플릿
            variables: 변수 딕셔너리
            execution_func: 실행 함수
            max_retries: 최대 재시도 횟수
            
        Returns:
            실행 결과 또는 None
        """
        start_time = time.time()
        self.logger.info(f"프롬프트 실행 시작: {template[:50]}...")
        
        # 변수 검증
        validation_result = self.validator.validate_prompt_variables(template, variables)
        
        if not validation_result.is_valid:
            self.logger.warning(f"프롬프트 변수 오류: {validation_result.missing_vars}")
            
            # 자동 변수 보정 시도
            corrected_variables = self._auto_correct_variables(template, variables, validation_result)
            if corrected_variables:
                self.logger.info("변수 자동 보정 완료")
                variables = corrected_variables
            else:
                self.logger.error("변수 자동 보정 실패")
                return None
        
        # 프롬프트 실행 (재시도 포함)
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"실행 시도 {attempt + 1}/{max_retries}")
                result = execution_func(template, variables)
                
                execution_time = time.time() - start_time
                
                # 실행 기록
                execution_record = ExecutionRecord(
                    timestamp=datetime.now().isoformat(),
                    template=template,
                    variables=variables,
                    execution_time=execution_time,
                    success=True,
                    result=result
                )
                self.execution_history.append(execution_record)
                
                self.logger.info(f"프롬프트 실행 완료: {execution_time:.2f}초")
                return result
                
            except Exception as e:
                self.logger.error(f"프롬프트 실행 오류 (시도 {attempt + 1}/{max_retries}): {str(e)}")
                
                if attempt == max_retries - 1:
                    # 최대 재시도 횟수 초과 시 오류 기록
                    execution_record = ExecutionRecord(
                        timestamp=datetime.now().isoformat(),
                        template=template,
                        variables=variables,
                        execution_time=time.time() - start_time,
                        success=False,
                        error=str(e)
                    )
                    self.execution_history.append(execution_record)
                    return None
    
    def _auto_correct_variables(self, template: str, variables: Dict[str, Any], 
                              validation_result: ValidationResult) -> Optional[Dict[str, Any]]:
        """변수 자동 보정"""
        try:
            corrected_variables = variables.copy()
            
            # 누락된 변수에 기본값 할당
            for missing_var in validation_result.missing_vars:
                if missing_var == "query":
                    corrected_variables[missing_var] = "사용자 쿼리"
                elif missing_var == "sql_query":
                    corrected_variables[missing_var] = "SELECT * FROM table"
                elif missing_var == "data_json":
                    corrected_variables[missing_var] = "[]"
                elif missing_var == "insights":
                    corrected_variables[missing_var] = "인사이트 데이터"
                elif missing_var == "new_members":
                    corrected_variables[missing_var] = "신규 회원 데이터"
                else:
                    corrected_variables[missing_var] = f"<{missing_var}_placeholder>"
            
            self.logger.info(f"변수 자동 보정: {validation_result.missing_vars}")
            return corrected_variables
            
        except Exception as e:
            self.logger.error(f"변수 자동 보정 실패: {e}")
            return None
    
    def get_execution_history(self) -> List[ExecutionRecord]:
        """실행 기록 반환"""
        return self.execution_history
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        if not self.execution_history:
            return {"total_executions": 0}
        
        successful_executions = [r for r in self.execution_history if r.success]
        failed_executions = [r for r in self.execution_history if not r.success]
        
        if successful_executions:
            avg_execution_time = sum(r.execution_time for r in successful_executions) / len(successful_executions)
            max_execution_time = max(r.execution_time for r in successful_executions)
            min_execution_time = min(r.execution_time for r in successful_executions)
        else:
            avg_execution_time = max_execution_time = min_execution_time = 0.0
        
        return {
            "total_executions": len(self.execution_history),
            "successful_executions": len(successful_executions),
            "failed_executions": len(failed_executions),
            "success_rate": len(successful_executions) / len(self.execution_history) if self.execution_history else 0,
            "avg_execution_time": avg_execution_time,
            "max_execution_time": max_execution_time,
            "min_execution_time": min_execution_time
        }

class PromptExecutor:
    """프롬프트 실행기 (Fallback 메커니즘 포함)"""
    
    def __init__(self, llm_client=None, debugger=None):
        self.llm_client = llm_client
        self.debugger = debugger or PromptDebugger()
        self.fallback_templates = self._initialize_fallback_templates()
        
    def _initialize_fallback_templates(self) -> Dict[str, str]:
        """대체 프롬프트 템플릿 초기화"""
        return {
            "data_analysis": "다음 데이터를 간단히 분석해주세요: {data}",
            "sql_insight": "SQL 쿼리 결과를 요약해주세요: {sql_query}",
            "general_insight": "다음 정보를 바탕으로 인사이트를 제공해주세요: {query}"
        }
    
    def execute_with_fallback(self, template: str, variables: Dict[str, Any], 
                            max_retries: int = 3) -> Any:
        """
        프롬프트 실행 및 오류 시 대체 로직 실행
        
        Args:
            template: 프롬프트 템플릿
            variables: 변수 딕셔너리
            max_retries: 최대 재시도 횟수
            
        Returns:
            실행 결과
        """
        def execution_func(tmpl, vars):
            if self.llm_client:
                return self.llm_client.generate(tmpl, vars)
            else:
                # 모의 실행 (테스트용)
                return f"응답: {tmpl.format(**vars)}"
        
        # 디버깅을 통한 실행
        result = self.debugger.debug_prompt_execution(
            template, variables, execution_func, max_retries
        )
        
        if result is not None:
            return result
        
        # Fallback 실행
        self.debugger.logger.warning("간소화된 대체 프롬프트 사용")
        return self._execute_fallback_prompt(variables)
    
    def _execute_fallback_prompt(self, variables: Dict[str, Any]) -> str:
        """간소화된 대체 프롬프트 실행"""
        try:
            # 사용 가능한 변수에 따라 적절한 대체 템플릿 선택
            if "data" in variables:
                fallback_template = self.fallback_templates["data_analysis"]
            elif "sql_query" in variables:
                fallback_template = self.fallback_templates["sql_insight"]
            else:
                fallback_template = self.fallback_templates["general_insight"]
            
            # 대체 프롬프트 실행
            if self.llm_client:
                return self.llm_client.generate(fallback_template, variables)
            else:
                return f"대체 응답: {fallback_template.format(**variables)}"
                
        except Exception as e:
            self.debugger.logger.error(f"대체 프롬프트 실행 오류: {str(e)}")
            return "프롬프트 처리 중 오류가 발생했습니다. 다시 시도해주세요."

def fix_template_escaping(template: str) -> str:
    """템플릿 중괄호 이스케이프 수정"""
    # 이중 중괄호로 이스케이프
    escaped_template = template.replace("{", "{{").replace("}", "}}")
    return escaped_template

def create_raw_template(template: str) -> str:
    """Raw 문자열 템플릿 생성"""
    return f"r\"{template}\""

if __name__ == "__main__":
    # 테스트 코드
    print("=== 프롬프트 변수 검증 시스템 테스트 ===")
    
    # 검증기 초기화
    validator = PromptValidator()
    debugger = PromptDebugger()
    executor = PromptExecutor(debugger=debugger)
    
    # 테스트 케이스들
    test_cases = [
        {
            "template": "사용자 {name}의 {month}월 데이터",
            "variables": {"name": "홍길동", "month": "8"},
            "expected_valid": True
        },
        {
            "template": "사용자 {name}의 {month}월 데이터",
            "variables": {"name": "홍길동"},
            "expected_valid": False
        },
        {
            "template": "JSON 예시: {{\"name\": \"{name}\", \"age\": {age}}}",
            "variables": {"name": "홍길동", "age": 30},
            "expected_valid": True
        }
    ]
    
    # 각 테스트 케이스 실행
    for i, test_case in enumerate(test_cases):
        print(f"\n테스트 {i+1}: {test_case['template'][:30]}...")
        
        # 검증 실행
        result = validator.validate_prompt_variables(
            test_case["template"], 
            test_case["variables"]
        )
        
        print(f"  검증 결과: {'유효' if result.is_valid else '무효'}")
        print(f"  누락된 변수: {result.missing_vars}")
        print(f"  사용되지 않은 변수: {result.unused_vars}")
        print(f"  제안사항: {result.suggestions}")
        
        # 예상 결과와 비교
        if result.is_valid == test_case["expected_valid"]:
            print("  [SUCCESS] 예상 결과와 일치")
        else:
            print("  [FAIL] 예상 결과와 불일치")
    
    # 디버깅 테스트
    print(f"\n=== 디버깅 테스트 ===")
    result = executor.execute_with_fallback(
        "사용자 {name}의 {month}월 데이터 분석",
        {"name": "홍길동"}
    )
    print(f"실행 결과: {result}")
    
    # 성능 통계
    stats = debugger.get_performance_stats()
    print(f"\n성능 통계: {stats}")
    
    print("\n테스트 완료")

