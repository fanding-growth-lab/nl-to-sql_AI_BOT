#!/usr/bin/env python3
"""
Dynamic SQL Generator
LLM을 사용한 동적 SQL 생성기
"""

import re
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser

from .nodes import BaseNode
from agentic_flow.llm_service import get_llm_service
from agentic_flow.llm_output_parser import parse_json_response
from agentic_flow.state import GraphState
# Removed: from .date_utils import DateUtils (deleted module)


@dataclass
class DynamicSQLResult:
    """동적 SQL 생성 결과"""
    sql_query: str
    confidence: float
    reasoning: str
    extracted_params: Dict[str, Any]


class DynamicSQLGenerator(BaseNode):
    """LLM 기반 동적 SQL 생성기"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm_service = get_llm_service()
        self.llm = self.llm_service.get_sql_llm()
        self.json_parser = SimpleJsonOutputParser()
        self._setup_prompt()
    
    def _initialize_llm(self) -> Optional[ChatGoogleGenerativeAI]:
        """LLM 초기화 (deprecated - use llm_service)"""
        try:
            # API 키 확인
            api_key = self.config.get('llm', {}).get('api_key', '')
            if not api_key:
                self.logger.error("Google API key not found in config")
                return None
            
            from pydantic import SecretStr
            # LLM 초기화 (지원되는 모델 사용)
            llm = ChatGoogleGenerativeAI(
                model=self.config.get('llm', {}).get('model', 'gemini-1.5-pro'),  # 지원되는 모델
                api_key=SecretStr(api_key),
                temperature=0.1,
                max_tokens=512,  # 토큰 수 줄임
                timeout=15.0,  # 타임아웃 증가
            )
            
            self.logger.info(f"LLM initialized successfully: {self.config.get('llm', {}).get('model', 'gemini-2.5-pro')}")
            return llm
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM for dynamic SQL generation: {str(e)}")
            return None
    
    def _setup_prompt(self):
        """프롬프트 설정 (개선된 버전)"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """자연어를 SQL로 변환하세요.

테이블: t_member, t_creator, t_member_login_log

JSON 형식으로만 응답:
{{
    "sql_query": "SELECT ...",
    "confidence": 0.9,
    "reasoning": "이유"
}}"""),
            ("human", "쿼리: {query}")
        ])
    
    def generate_dynamic_sql(self, query: str, schema_info: Optional[Dict] = None) -> Optional[DynamicSQLResult]:
        """동적 SQL 생성"""
        try:
            if not self.llm:
                self.logger.error("LLM not available for dynamic SQL generation - check API key and model configuration")
                return None
            
            # 프롬프트 생성
            formatted_prompt = self.prompt.format(query=query)
            
            # LLM 호출
            self.logger.info(f"Calling LLM with query: {query[:100]}...")
            self.logger.info(f"LLM Model: {self.config.get('llm', {}).get('model', 'unknown')}")
            
            try:
                # LangChain 메시지 형식으로 변환 (문자열을 HumanMessage로)
                from langchain_core.messages import HumanMessage
                messages = [HumanMessage(content=formatted_prompt)]
                response = self.llm.invoke(messages)
                self.logger.info(f"LLM invoke completed successfully")
            except Exception as e:
                self.logger.error(f"LLM invoke failed: {str(e)}")
                return None
            
            if not response:
                self.logger.error("LLM returned None response")
                return None
            
            # LangChain 표준 Output Parser 사용
            result_data = parse_json_response(response, parser=self.json_parser, fallback_extract=True)
            
            if not result_data:
                self.logger.warning("No valid data extracted from LLM response")
                return None
            
            try:
                # DynamicSQLResult 생성 시 안전한 타입 변환
                self.logger.info(f"Creating DynamicSQLResult from: {result_data}")
                
                sql_query = str(result_data.get("sql_query", ""))
                confidence = float(result_data.get("confidence", 0.5))
                reasoning = str(result_data.get("reasoning", ""))
                extracted_params = result_data.get("extracted_params", {})
                
                # extracted_params가 딕셔너리가 아닌 경우 처리
                if not isinstance(extracted_params, dict):
                    extracted_params = {}
                
                self.logger.info(f"DynamicSQLResult parameters: sql_query='{sql_query}', confidence={confidence}, reasoning='{reasoning}'")
                
                result = DynamicSQLResult(
                    sql_query=sql_query,
                    confidence=confidence,
                    reasoning=reasoning,
                    extracted_params=extracted_params
                )
                
                self.logger.info(f"DynamicSQLResult created successfully: {result}")
                return result
            except Exception as e:
                self.logger.error(f"Error creating DynamicSQLResult: {str(e)}")
                self.logger.error(f"Result data: {result_data}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error in dynamic SQL generation: {str(e)}")
            self.logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    # Note: JSON 파싱 로직은 llm_output_parser 모듈로 이동됨
    # parse_json_response 함수를 사용하세요
    def _extract_json_from_text_deprecated(self, text: str) -> Optional[Dict[str, Any]]:
        """텍스트에서 JSON 추출 (개선된 버전)"""
        try:
            import json
            import re
            
            if not text or not text.strip():
                self.logger.warning("빈 응답을 받았습니다")
                return self._create_fallback_json("빈 응답")
            
            # 여러 JSON 추출 방법 시도
            methods = [
                self._extract_json_by_blocks,
                self._extract_json_by_regex,
                self._extract_json_by_manual_parsing,
                self._extract_json_from_markdown,
                self._extract_json_with_extra_text
            ]
            
            for method in methods:
                try:
                    self.logger.debug(f"JSON 추출 방법 시도: {method.__name__}")
                    result = method(text)
                    if result:
                        self.logger.info(f"JSON 추출 성공: {method.__name__}")
                        self.logger.info(f"추출된 결과: {result}")
                        return result
                    else:
                        self.logger.debug(f"JSON 추출 방법 {method.__name__}: 결과 없음")
                except Exception as e:
                    self.logger.debug(f"JSON 추출 방법 {method.__name__} 실패: {str(e)}")
                    continue
            
            # 모든 방법 실패 시 fallback
            self.logger.warning("모든 JSON 추출 방법 실패, fallback 사용")
            return self._create_fallback_json(text)
            
        except Exception as e:
            self.logger.warning(f"JSON 추출 중 오류 발생: {str(e)}")
            return self._create_fallback_json(text)
    
    def _extract_json_by_blocks(self, text: str) -> Optional[Dict[str, Any]]:
        """JSON 블록으로 추출 (개선된 버전)"""
        import json
        import re
        
        try:
            # 먼저 ```json ... ``` 블록에서 추출 시도
            json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            match = re.search(json_block_pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                json_str = match.group(1).strip()
                # JSON 문자열 내의 개행 정리
                json_str = re.sub(r'\n\s*', ' ', json_str)
                return json.loads(json_str)
            
            # ```json 블록이 없으면 일반 JSON 객체 찾기
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx+1]
                # JSON 문자열 정리 (개행 정리)
                json_str = re.sub(r'\n\s*', ' ', json_str.strip())
                return json.loads(json_str)
        except Exception as e:
            self.logger.debug(f"JSON 블록 추출 실패: {str(e)}")
        return None
    
    def _extract_json_by_regex(self, text: str) -> Optional[Dict[str, Any]]:
        """정규식으로 JSON 추출"""
        import json
        import re
        
        # JSON 패턴 찾기
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        return None
    
    def _extract_json_by_manual_parsing(self, text: str) -> Optional[Dict[str, Any]]:
        """수동 파싱으로 JSON 추출"""
        try:
            # 기본 JSON 구조 생성
            result = {
                "sql_query": "",
                "confidence": 0.5,
                "reasoning": "Manual parsing fallback",
                "extracted_params": {}
            }
            
            # SQL 쿼리 추출 시도 (개선된 패턴)
            sql_patterns = [
                r'"sql_query":\s*"([^"]*)"',
                r'sql_query["\']?\s*:\s*["\']([^"\']*)["\']',
                r'SELECT\s+.*?(?=\s*$|\s*["\']|,|\})',
                r'SELECT\s+.*',
            ]
            
            for pattern in sql_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    result["sql_query"] = match.group(1) if match.groups() else match.group(0)
                    break
            
            # 신뢰도 추출 시도
            confidence_patterns = [
                r'"confidence":\s*([0-9.]+)',
                r'confidence["\']?\s*:\s*([0-9.]+)',
            ]
            
            for pattern in confidence_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    result["confidence"] = float(match.group(1))
                    break
            
            return result if result["sql_query"] else None
            
        except Exception as e:
            self.logger.warning(f"Manual JSON parsing failed: {str(e)}")
            return None
    
    def _extract_json_from_markdown(self, text: str) -> Optional[Dict[str, Any]]:
        """마크다운 코드 블록에서 JSON 추출"""
        import json
        import re
        
        # ```json ... ``` 패턴 찾기
        json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_block_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _extract_json_with_extra_text(self, text: str) -> Optional[Dict[str, Any]]:
        """추가 텍스트가 포함된 응답에서 JSON 추출"""
        import json
        import re
        
        # JSON 객체 패턴 찾기
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _create_fallback_json(self, text: str) -> Dict[str, Any]:
        """Fallback JSON 생성"""
        import re
        
        # SQL 쿼리 패턴 추출 시도 (개선된 패턴)
        sql_patterns = [
            r'SELECT\s+.*?(?=\s*$|\s*["\']|,|\})',
            r'SELECT\s+.*',
            r'"sql_query":\s*"([^"]*)"',
            r'sql_query["\']?\s*:\s*["\']([^"\']*)["\']',
        ]
        
        sql_query = ""
        for pattern in sql_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                # 그룹이 있는 경우 (캡처 그룹)
                if match.groups():
                    sql_query = match.group(1)
                else:
                    sql_query = match.group(0)
                break
        
        if not sql_query:
            sql_query = "-- SQL 추출 실패"
        
        return {
            "sql_query": sql_query,
            "confidence": 0.3,
            "reasoning": "JSON 파싱에 실패하여 자동으로 생성된 응답입니다.",
            "extracted_params": {},
            "parsing_error": True
        }
    
    def extract_month_from_query(self, query: str) -> Optional[str]:
        """쿼리에서 월 추출 (DateUtils 대체 구현)"""
        # Simple month extraction without DateUtils
        import re
        month_patterns = {
            r'(\d+)월': r'\1',
            r'(\d+)월달': r'\1',
            r'(\d+)월분': r'\1'
        }
        
        for pattern, replacement in month_patterns.items():
            match = re.search(pattern, query)
            if match:
                month = match.group(1)
                if 1 <= int(month) <= 12:
                    return month.zfill(2)  # 01, 02, etc.
        
        return None
    
    def generate_membership_performance_sql(self, query: str) -> str:
        """멤버십 성과 분석 SQL 생성"""
        month = self.extract_month_from_query(query)  # Use our own method
        
        return f"""
        SELECT 
            '{month}' as analysis_month,
            COUNT(*) as total_members,
            COUNT(CASE WHEN status = 'A' THEN 1 END) as active_members,
            COUNT(CASE WHEN status = 'I' THEN 1 END) as inactive_members,
            COUNT(CASE WHEN status = 'D' THEN 1 END) as deleted_members,
            ROUND(COUNT(CASE WHEN status = 'A' THEN 1 END) * 100.0 / COUNT(*), 2) as active_rate_percent,
            ROUND(COUNT(CASE WHEN status = 'I' THEN 1 END) * 100.0 / COUNT(*), 2) as inactive_rate_percent,
            ROUND(COUNT(CASE WHEN status = 'D' THEN 1 END) * 100.0 / COUNT(*), 2) as deletion_rate_percent
        FROM t_member
        """
    
    def process(self, state: GraphState) -> GraphState:
        """동적 SQL 생성 처리"""
        self._log_processing(state, "DynamicSQLGenerator")
        
        try:
            user_query = state.get("user_query")
            if not user_query:
                self.logger.error("user_query is None or empty")
                return state
            
            # 동적 SQL 생성
            result = self.generate_dynamic_sql(user_query)
            
            if result and result.sql_query:
                # 결과를 state에 저장 (GraphState는 TypedDict이므로 dict로 캐스팅)
                state_dict: Dict[str, Any] = state  # type: ignore[assignment]
                
                # dynamic_sql_result에 상세 정보 저장
                state_dict["dynamic_sql_result"] = {
                    "sql_query": result.sql_query,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "extracted_params": result.extracted_params
                }
                
                # SQL이 성공적으로 생성되었으므로 clarification 응답 제거
                # 이전에 NLProcessor에서 설정한 clarification이 있어도 SQL 생성이 성공했으면 무시
                if state_dict.get("conversation_response") and state_dict.get("needs_clarification"):
                    self.logger.info(
                        f"SQL successfully generated (confidence: {result.confidence:.2f}), "
                        f"clearing clarification response to proceed with SQL execution"
                    )
                    state_dict["conversation_response"] = None
                    state_dict["needs_clarification"] = False
                
                # sql_query에도 저장하여 SQLGenerationNode에서 사용 가능하도록 함
                # 단, 기존 sql_query가 템플릿 SQL이고 dynamic_sql의 confidence가 높으면 덮어쓰기
                existing_sql = state_dict.get("sql_query")
                if not existing_sql or (existing_sql and result.confidence >= 0.8):
                    state_dict["sql_query"] = result.sql_query
                    self.logger.info(f"Dynamic SQL saved to state['sql_query'] with confidence: {result.confidence:.2f}")
                else:
                    self.logger.info(f"Keeping existing SQL (template) as DynamicSQL confidence ({result.confidence:.2f}) is lower")
                
                self.logger.info(f"Dynamic SQL generated with confidence: {result.confidence:.2f}")
                self.logger.debug(f"Generated SQL: {result.sql_query}")
            else:
                self.logger.warning("Dynamic SQL generation failed")
                state_dict: Dict[str, Any] = state  # type: ignore[assignment]
                state_dict["dynamic_sql_result"] = None
            
        except Exception as e:
            self.logger.error(f"Error in dynamic SQL generation: {str(e)}")
            state_dict: Dict[str, Any] = state  # type: ignore[assignment]
            state_dict["dynamic_sql_result"] = None
        
        return state
