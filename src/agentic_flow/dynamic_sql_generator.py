#!/usr/bin/env python3
"""
Dynamic SQL Generator
LLM을 사용한 동적 SQL 생성기
"""

import re
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from .nodes import BaseNode
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
        self.llm = self._initialize_llm()
        self._setup_prompt()
    
    def _initialize_llm(self) -> Optional[ChatGoogleGenerativeAI]:
        """LLM 초기화"""
        try:
            # API 키 확인
            api_key = self.config.get('llm', {}).get('api_key', '')
            if not api_key:
                self.logger.error("Google API key not found in config")
                return None
            
            # LLM 초기화 (지원되는 모델 사용)
            llm = ChatGoogleGenerativeAI(
                model=self.config.get('llm', {}).get('model', 'gemini-1.5-pro'),  # 지원되는 모델
                google_api_key=api_key,
                temperature=0.1,
                max_output_tokens=512,  # 토큰 수 줄임
                request_timeout=15.0,  # 타임아웃 증가
                convert_system_message_to_human=True,  # 최신 버전 호환성
                model_kwargs={
                    "response_mime_type": "application/json"
                }
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
                response = self.llm.invoke(formatted_prompt)
                self.logger.info(f"LLM invoke completed successfully")
            except Exception as e:
                self.logger.error(f"LLM invoke failed: {str(e)}")
                return None
            
            # 응답 내용 확인
            if not response:
                self.logger.error("LLM returned None response")
                return None
                
            if not hasattr(response, 'content'):
                self.logger.error("LLM response has no content attribute")
                # 후보/메타 내 텍스트 추출 시도
                alt_text = None
                try:
                    alt_text = getattr(response, 'additional_kwargs', {}).get('text') or getattr(response, 'response_metadata', {}).get('text')
                except Exception:
                    pass
                if alt_text:
                    self.logger.info("Using alternative text from response metadata")
                    response_content = alt_text
                else:
                    return None
                
            else:
                response_content = response.content
                
            if not response_content:
                # 후보/메타 내 텍스트 재시도
                alt_text = None
                try:
                    alt_text = getattr(response, 'additional_kwargs', {}).get('text') or getattr(response, 'response_metadata', {}).get('text')
                except Exception:
                    pass
                if not alt_text:
                    self.logger.warning("LLM returned empty content - may be due to ambiguous query requiring clarification")
                    return None
                self.logger.info("Using alternative text from response metadata (empty content)")
                response_content = alt_text
            
            self.logger.info(f"LLM Response length: {len(response.content)}")
            self.logger.info(f"LLM Response content: {response.content[:200]}...")
            self.logger.info(f"Full LLM Response: {repr(response.content)}")
            
            # JSON 파싱 (디버깅 추가)
            try:
                import json
                # 응답 내용 정리
                cleaned_content = response.content.strip()
                self.logger.info(f"Cleaned content: {cleaned_content[:200]}...")
                
                result_data = json.loads(cleaned_content)
                self.logger.info(f"JSON parsing successful: {result_data}")
            except json.JSONDecodeError as e:
                self.logger.warning(f"JSON parsing failed: {str(e)}")
                self.logger.warning(f"Response content: {response.content}")
                # JSON 파싱 실패 시 텍스트에서 추출
                result_data = self._extract_json_from_text(response.content)
                self.logger.info(f"Extracted from text: {result_data}")
            except Exception as e:
                self.logger.error(f"Unexpected error in JSON parsing: {str(e)}")
                self.logger.error(f"Response content: {response.content}")
                self.logger.error(f"Response content repr: {repr(response.content)}")
                # 예상치 못한 에러 시 텍스트에서 추출
                result_data = self._extract_json_from_text(response.content)
                self.logger.info(f"Extracted from text: {result_data}")
            
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
    
    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
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
        """JSON 블록으로 추출"""
        import json
        
        try:
            # JSON 블록 찾기
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx+1]
                # 문자열 정리
                json_str = json_str.strip()
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
                import re
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
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """동적 SQL 생성 처리"""
        self._log_processing(state, "DynamicSQLGenerator")
        
        try:
            user_query = state.get("user_query")
            if not user_query:
                self.logger.error("user_query is None or empty")
                return state
            
            # 동적 SQL 생성
            result = self.generate_dynamic_sql(user_query)
            
            if result:
                # 결과를 state에 저장
                state["dynamic_sql_result"] = {
                    "sql_query": result.sql_query,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                    "extracted_params": result.extracted_params
                }
                
                self.logger.info(f"Dynamic SQL generated with confidence: {result.confidence:.2f}")
                self.logger.debug(f"Generated SQL: {result.sql_query}")
            else:
                self.logger.warning("Dynamic SQL generation failed")
                state["dynamic_sql_result"] = None
            
        except Exception as e:
            self.logger.error(f"Error in dynamic SQL generation: {str(e)}")
            state["dynamic_sql_result"] = None
        
        return state
