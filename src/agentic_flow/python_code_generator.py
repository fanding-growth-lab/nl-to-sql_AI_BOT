"""
Python Code Generator Node for Complex Analysis

This module implements PythonCodeGeneratorNode which generates Python code
for complex data analysis tasks using LLM.
"""

import ast
import re
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .state import GraphState, QueryIntent
from agentic_flow.llm_output_parser import parse_json_response
from langchain.output_parsers.json import SimpleJsonOutputParser
from core.logging import get_logger

logger = get_logger(__name__)


class BaseNode(ABC):
    """Base class for all pipeline nodes (local copy to avoid circular import)."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self._llm_service = None  # Lazy initialization
    
    @abstractmethod
    def process(self, state: GraphState) -> GraphState:
        """Process the current state and return updated state."""
        pass
    
    def _log_processing(self, state: GraphState, component: str):
        """Log processing information."""
        self.logger.info(
            f"Processing {component}",
            user_id=state.get("user_id"),
            channel_id=state.get("channel_id"),
            query=state.get("user_query", "")[:100]
        )
    
    def _get_llm_service(self):
        """Get LLM service instance (lazy initialization)."""
        if self._llm_service is None:
            from agentic_flow.llm_service import get_llm_service
            self._llm_service = get_llm_service()
        return self._llm_service
    
    def _get_intent_llm(self):
        """Get intent classification LLM (lightweight, fast response)."""
        return self._get_llm_service().get_intent_llm()
    
    def _get_sql_llm(self):
        """Get SQL generation LLM (high-performance model)."""
        return self._get_llm_service().get_sql_llm()


@dataclass
class PythonCodeGenerationResult:
    """Python 코드 생성 결과"""
    code: str
    confidence: float
    imports: List[str]
    main_function: Optional[str]
    is_safe: bool
    validation_errors: List[str]


class PythonCodeGeneratorNode(BaseNode):
    """
    LLM 기반 Python 코드 생성 노드
    
    복잡한 분석 작업을 위한 Python 코드를 동적으로 생성합니다.
    Pandas, NumPy 등 데이터 분석 라이브러리를 활용합니다.
    """
    
    # 허용된 모듈 화이트리스트
    ALLOWED_MODULES = {
        'pandas', 'pd',
        'numpy', 'np',
        'math',
        'datetime',
        'collections',
        'json',
        're',
        'itertools',
        'functools',
        'operator'
    }
    
    # 위험한 함수/패턴 블랙리스트
    DANGEROUS_PATTERNS = {
        'open(', 'file(',  # 파일 접근
        'eval(', 'exec(', 'compile(', '__import__',  # 코드 실행
        'subprocess', 'os.system', 'os.popen',  # 시스템 명령
        'socket', 'urllib', 'requests', 'http',  # 네트워크
        'pickle', 'marshal',  # 직렬화 (보안 취약점)
        '__builtins__', '__globals__', '__locals__',  # 내부 접근
        'input(', 'raw_input('  # 사용자 입력
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm = self._get_sql_llm()  # SQL LLM 사용 (고성능 모델)
        self._setup_prompt()
    
    def _setup_prompt(self):
        """Python 코드 및 데이터 수집 SQL 생성 프롬프트 설정 (Phase 2)"""
        self.prompt_template = """다음 자연어 쿼리를 분석하여 두 가지를 생성하세요:

1. **data_gathering_sql**: Python 코드 실행에 필요한 데이터를 가져올 간단하고 안전한 SQL 쿼리
   - SELECT 문만 사용 (INSERT, UPDATE, DELETE 금지)
   - 필요한 컬럼만 조회 (성능 최적화)
   - WHERE 절로 필터링 (필요한 데이터만)

2. **python_code**: 가져온 데이터를 pandas DataFrame(df)으로 받아서 실제 계산을 수행할 Python 코드
   - Pandas와 NumPy를 사용
   - 결과는 `result` 변수에 저장 (dict, list, DataFrame 등 가능)
   - 안전한 코드만 생성 (파일 접근, 네트워크, 시스템 명령 금지)

응답 형식 (JSON):
{{
    "data_gathering_sql": "SELECT ... FROM ... WHERE ...",
    "python_code": "import pandas as pd\\nimport numpy as np\\n\\n# 계산 코드...\\nresult = ..."
}}

예시:
질문: "A 크리에이터 구독자 중 B 크리에이터도 구독하는 비율을 계산하세요"
{{
    "data_gathering_sql": "SELECT creator_no, member_no FROM t_fanding WHERE creator_no IN ('A', 'B')",
    "python_code": "import pandas as pd\\nimport numpy as np\\n\\n# A 크리에이터 구독자 집합\\ncreator_a_subscribers = df[df['creator_no'] == 'A']['member_no'].unique()\\n# B 크리에이터 구독자 집합\\ncreator_b_subscribers = df[df['creator_no'] == 'B']['member_no'].unique()\\n# 교집합 계산\\nintersection = np.intersect1d(creator_a_subscribers, creator_b_subscribers)\\n# 비율 계산\\nratio = len(intersection) / len(creator_a_subscribers) if len(creator_a_subscribers) > 0 else 0\\nresult = {{'ratio': ratio, 'intersection_count': len(intersection), 'total_a': len(creator_a_subscribers)}}"
}}

질문: "월별 신규 회원 수 추이를 계산하세요"
{{
    "data_gathering_sql": "SELECT ins_datetime FROM t_member_info",
    "python_code": "import pandas as pd\\n\\ndf['join_month'] = pd.to_datetime(df['ins_datetime']).dt.to_period('M')\\nmonthly_new_members = df.groupby('join_month').size().reset_index(name='count')\\nresult = monthly_new_members.to_dict('records')"
}}

질문: {query}

JSON 형식으로 응답하세요:"""
    
    def process(self, state: GraphState) -> GraphState:
        """
        상태를 처리하여 Python 코드를 생성합니다.
        
        Args:
            state: 현재 파이프라인 상태
            
        Returns:
            업데이트된 상태
        """
        self._log_processing(state, "PythonCodeGeneratorNode")
        
        # COMPLEX_ANALYSIS 의도만 처리
        intent = state.get("intent")
        if intent != QueryIntent.COMPLEX_ANALYSIS:
            logger.info(f"Skipping Python code generation (intent: {intent})")
            return state
        
        user_query = state.get("user_query", "")
        if not user_query:
            logger.warning("No user query found for Python code generation")
            return state
        
        try:
            # Phase 2: LLM을 사용하여 data_gathering_sql과 python_code 모두 생성
            generation_result = self._generate_python_code_and_sql(user_query, state)
            
            if not generation_result or not generation_result.get("python_code"):
                logger.warning("Failed to generate Python code and SQL")
                state["python_code_error"] = "Python 코드 및 SQL 생성 실패"
                return state
            
            data_gathering_sql = generation_result.get("data_gathering_sql", "")
            generated_code = generation_result.get("python_code", "")
            
            if not data_gathering_sql:
                logger.warning("Failed to generate data_gathering_sql")
                state["python_code_error"] = "데이터 수집 SQL 생성 실패"
                return state
            
            # Python 코드 검증
            validation_result = self._validate_code(generated_code)
            
            if not validation_result.is_safe:
                logger.warning(f"Generated code failed validation: {validation_result.validation_errors}")
                state["python_code_error"] = f"코드 검증 실패: {', '.join(validation_result.validation_errors)}"
                return state
            
            # 상태에 저장 (Phase 2: data_gathering_sql도 함께 저장)
            state["data_gathering_sql"] = data_gathering_sql
            state["python_code"] = generated_code
            state["sql_query"] = data_gathering_sql  # sql_execution 노드가 사용할 수 있도록
            state["python_code_result"] = {
                "code": generated_code,
                "data_gathering_sql": data_gathering_sql,
                "confidence": validation_result.confidence,
                "imports": validation_result.imports,
                "is_safe": validation_result.is_safe,
                "main_function": validation_result.main_function
            }
            state["confidence_scores"]["python_code_generation"] = validation_result.confidence
            
            logger.info(
                f"Python code and data_gathering_sql generated successfully "
                f"(confidence: {validation_result.confidence:.2f})"
            )
            logger.info(f"Data gathering SQL: {data_gathering_sql[:100]}...")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in Python code generation: {str(e)}")
            state["python_code_error"] = f"Python 코드 생성 중 오류: {str(e)}"
            return state
    
    def _generate_python_code_and_sql(self, query: str, state: GraphState) -> Optional[Dict[str, str]]:
        """
        Phase 2: LLM을 사용하여 data_gathering_sql과 python_code 모두 생성
        
        Args:
            query: 사용자 쿼리
            state: 현재 상태 (추가 컨텍스트 사용 가능)
            
        Returns:
            {"data_gathering_sql": "...", "python_code": "..."} 또는 None
        """
        import json
        
        try:
            if not self.llm:
                logger.error("LLM not available for Python code generation")
                return None
            
            # 프롬프트 생성
            prompt = self.prompt_template.format(query=query)
            
            # 스키마 정보 추가 (있는 경우)
            schema_mapping = state.get("schema_mapping")
            rag_schema_context = state.get("rag_schema_context", "")
            
            if rag_schema_context:
                prompt += f"\n\n사용 가능한 데이터베이스 스키마 정보:\n{rag_schema_context}"
            elif schema_mapping:
                schema_info = self._format_schema_info(schema_mapping)
                prompt += f"\n\n사용 가능한 데이터 스키마 정보:\n{schema_info}"
            
            logger.info(f"Generating Python code and data_gathering_sql for query: {query[:100]}...")
            
            # LLM 호출 (메시지 형식으로 변환)
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            
            if not response:
                logger.error("LLM returned None response")
                return None
            
            # LangChain 표준 Output Parser 사용
            json_parser = SimpleJsonOutputParser()
            result = parse_json_response(response, parser=json_parser, fallback_extract=True)
            
            if result and "data_gathering_sql" in result and "python_code" in result:
                logger.debug(
                    f"Generated data_gathering_sql ({len(result['data_gathering_sql'])} chars) "
                    f"and python_code ({len(result['python_code'])} chars)"
                )
                return {
                    "data_gathering_sql": result["data_gathering_sql"].strip(),
                    "python_code": result["python_code"].strip()
                }
            else:
                logger.warning("JSON response missing required fields (data_gathering_sql, python_code)")
                # Fallback: 기존 방식으로 코드만 추출 시도
                code = self._extract_code_from_response(response)
                if code:
                    logger.warning("Falling back to code-only extraction (data_gathering_sql will be empty)")
                    return {
                        "data_gathering_sql": "",  # 빈 SQL은 나중에 처리 필요
                        "python_code": code
                    }
                return None
            
        except Exception as e:
            logger.error(f"Error generating Python code and SQL: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def _generate_python_code(self, query: str, state: GraphState) -> Optional[str]:
        """
        [Deprecated] LLM을 사용하여 Python 코드 생성 (Phase 2 이전 방식)
        Phase 2에서는 _generate_python_code_and_sql을 사용하세요.
        
        Args:
            query: 사용자 쿼리
            state: 현재 상태 (추가 컨텍스트 사용 가능)
            
        Returns:
            생성된 Python 코드 또는 None
        """
        result = self._generate_python_code_and_sql(query, state)
        return result.get("python_code") if result else None
    
    def _extract_code_from_response(self, response: Any) -> Optional[str]:
        """
        LLM 응답에서 Python 코드 추출
        
        Args:
            response: LLM 응답 객체
            
        Returns:
            추출된 Python 코드 또는 None
        """
        try:
            # 응답 내용 가져오기
            if hasattr(response, 'content'):
                content = response.content
            elif isinstance(response, str):
                content = response
            else:
                logger.warning(f"Unexpected response type: {type(response)}")
                return None
            
            # 코드 블록 찾기 (```python ... ```)
            code_block_match = re.search(r'```python\s*\n(.*?)```', content, re.DOTALL)
            if code_block_match:
                code = code_block_match.group(1).strip()
                return code
            
            # 코드 블록 없으면 전체 내용에서 코드 라인만 추출
            lines = content.split('\n')
            code_lines = []
            in_code = False
            
            for line in lines:
                # Python 키워드나 import가 있으면 코드로 간주
                if re.match(r'^\s*(import|from|def|class|if|for|while|try|with)', line):
                    in_code = True
                
                if in_code or line.strip().startswith(('import ', 'from ', 'def ', 'class ', '# ')):
                    code_lines.append(line)
                elif in_code and not line.strip():
                    code_lines.append(line)
            
            if code_lines:
                return '\n'.join(code_lines).strip()
            
            # 마지막 시도: 전체 내용 반환
            return content.strip()
            
        except Exception as e:
            logger.error(f"Error extracting code from response: {str(e)}")
            return None
    
    def _validate_code(self, code: str) -> PythonCodeGenerationResult:
        """
        생성된 Python 코드 검증
        
        Args:
            code: 검증할 Python 코드
            
        Returns:
            검증 결과
        """
        validation_errors = []
        imports = []
        main_function = None
        
        try:
            # 1. AST 파싱으로 구문 검증
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                validation_errors.append(f"구문 오류: {str(e)}")
                return PythonCodeGenerationResult(
                    code=code,
                    confidence=0.0,
                    imports=[],
                    main_function=None,
                    is_safe=False,
                    validation_errors=validation_errors
                )
            
            # 2. 위험한 패턴 검사
            code_lower = code.lower()
            for pattern in self.DANGEROUS_PATTERNS:
                if pattern.lower() in code_lower:
                    validation_errors.append(f"위험한 패턴 감지: {pattern}")
            
            # 3. 모듈 import 검사
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        if module_name not in self.ALLOWED_MODULES:
                            validation_errors.append(f"허용되지 않은 모듈: {module_name}")
                        else:
                            imports.append(module_name)
                
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module.split('.')[0] if node.module else ''
                    if module_name not in self.ALLOWED_MODULES:
                        validation_errors.append(f"허용되지 않은 모듈: {module_name}")
                    else:
                        imports.append(module_name)
                
                elif isinstance(node, ast.FunctionDef):
                    if not main_function:
                        main_function = node.name
            
            # 4. 안전성 점수 계산
            is_safe = len(validation_errors) == 0
            confidence = 0.9 if is_safe else max(0.1, 1.0 - len(validation_errors) * 0.2)
            
            # pandas 또는 numpy 사용 여부 확인
            has_data_library = any(imp in ['pandas', 'numpy'] for imp in imports)
            if has_data_library:
                confidence = min(1.0, confidence + 0.1)
            
            return PythonCodeGenerationResult(
                code=code,
                confidence=confidence,
                imports=list(set(imports)),
                main_function=main_function,
                is_safe=is_safe,
                validation_errors=validation_errors
            )
            
        except Exception as e:
            logger.error(f"Error validating code: {str(e)}")
            validation_errors.append(f"검증 중 오류: {str(e)}")
            return PythonCodeGenerationResult(
                code=code,
                confidence=0.0,
                imports=[],
                main_function=None,
                is_safe=False,
                validation_errors=validation_errors
            )
    
    def _format_schema_info(self, schema_mapping: Any) -> str:
        """
        스키마 매핑 정보를 문자열로 포맷팅
        
        Args:
            schema_mapping: 스키마 매핑 객체
            
        Returns:
            포맷팅된 스키마 정보 문자열
        """
        try:
            if isinstance(schema_mapping, dict):
                tables = schema_mapping.get('tables', [])
                if tables:
                    schema_info = "테이블 및 컬럼:\n"
                    for table in tables[:5]:  # 최대 5개 테이블만
                        table_name = table.get('name', 'unknown')
                        columns = table.get('columns', [])
                        schema_info += f"- {table_name}: {', '.join([c.get('name', '') for c in columns[:10]])}\n"
                    return schema_info
            
            return "스키마 정보 없음"
        except Exception as e:
            logger.warning(f"Error formatting schema info: {str(e)}")
            return "스키마 정보 포맷팅 오류"

