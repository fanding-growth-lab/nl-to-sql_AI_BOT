#!/usr/bin/env python3
"""
LLM-based Intent Classifier
빠른 LLM 모델을 사용한 인텐트 분류 노드
"""

import json
import logging
import time
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
# Removed: JsonOutputParser (not used)

from .nodes import BaseNode, QueryIntent
from .state import GraphState
from agentic_flow.llm_service import get_llm_service
from agentic_flow.intent_classification_stats import get_integrator, QueryInteractionMetrics, get_classification_stats
from typing import Dict, Any, Optional
from core.config import get_settings


@dataclass
class IntentClassificationResult:
    """인텐트 분류 결과"""
    intent: QueryIntent
    confidence: float
    reasoning: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }


class LLMIntentClassifier(BaseNode):
    """LLM 기반 인텐트 분류기"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm_service = get_llm_service()
        self.llm = self.llm_service.get_intent_llm()
        # Removed: JsonOutputParser (not used)
        self._setup_prompt()
    
    def _initialize_llm(self) -> Optional[ChatGoogleGenerativeAI]:
        """빠른 LLM 모델 초기화 (deprecated - use llm_service)"""
        try:
            # 설정에서 인텐트 분류용 LLM 설정 가져오기
            settings = get_settings()
            
            self.logger.info(f"Intent LLM config: model={settings.llm.intent_model}, "
                           f"temperature={settings.llm.intent_temperature}, "
                           f"max_tokens={settings.llm.intent_max_tokens}")
            
            # 인텐트 분류용 빠른 모델 사용
            from pydantic import SecretStr
            return ChatGoogleGenerativeAI(
                model=settings.llm.intent_model,
                api_key=SecretStr(settings.llm.api_key) if settings.llm.api_key else None,
                temperature=settings.llm.intent_temperature,
                max_tokens=settings.llm.intent_max_tokens,
                timeout=10.0,
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM for intent classification: {str(e)}")
            return None
    
    def _setup_prompt(self):
        """프롬프트 설정"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """쿼리 의도를 분류하세요:

1. GREETING: 인사말
2. HELP_REQUEST: 도움 요청  
3. GENERAL_CHAT: 일반 대화
4. DATA_QUERY: 데이터 조회/분석 요청

JSON 응답:
{{
    "intent": "GREETING|HELP_REQUEST|GENERAL_CHAT|DATA_QUERY",
    "confidence": 0.0~1.0,
    "reasoning": "간단한 이유"
}}"""),
            ("human", "쿼리: {query}")
        ])
    
    def process(self, state: GraphState) -> GraphState:
        """인텐트 분류 처리"""
        self._log_processing(state, "LLMIntentClassifier")
        
        # 통계 수집을 위한 시작 시간 기록
        start_time = time.time()
        is_error = False
        classification_result: Optional[IntentClassificationResult] = None
        user_query: Optional[str] = None  # Initialize user_query for finally block
        
        try:
            user_query = state.get("user_query")
            if not user_query:
                self.logger.error("user_query is None or empty")
                state["llm_intent_result"] = None
                is_error = True
                return state
            
            # LLM이 사용 불가능한 경우 None 반환
            if not self.llm:
                self.logger.warning("LLM not available, skipping LLM intent classification")
                state["llm_intent_result"] = None
                is_error = True
                return state
            
            # LLM으로 인텐트 분류
            classification_result = self._classify_intent_with_llm(user_query)
            
            # 결과를 state에 저장
            state["llm_intent_result"] = classification_result.to_dict() if classification_result else None
            
            # 로깅
            if classification_result:
                self.logger.info(f"LLM Intent Classification: {classification_result.intent.value} (confidence: {classification_result.confidence:.2f})")
                self.logger.debug(f"Reasoning: {classification_result.reasoning}")
            else:
                self.logger.warning("LLM intent classification failed")
                is_error = True
            
        except Exception as e:
            self.logger.error(f"Error in LLM intent classification: {str(e)}")
            state["llm_intent_result"] = None
            is_error = True
            classification_result = None
        
        finally:
            # 통계 수집 (user_query가 있을 때만)
            if user_query:
                self._record_classification_stats(user_query, classification_result, start_time, is_error)
        
        return state
    
    def _classify_intent_with_llm(self, query: str) -> Optional[IntentClassificationResult]:
        """LLM을 사용한 인텐트 분류"""
        try:
            if self.llm is None:
                self.logger.warning("LLM not available for intent classification")
                return None
                
            # 프롬프트 생성
            formatted_prompt = self.prompt.format(query=query)
            
            # LLM 호출
            response = self.llm.invoke(formatted_prompt)
            
            # response.content가 str인지 확인
            response_text: str
            if isinstance(response.content, str):
                response_text = response.content
            elif isinstance(response.content, list):
                # 리스트인 경우 첫 번째 요소를 문자열로 변환
                response_text = str(response.content[0]) if response.content else ""
            else:
                response_text = str(response.content)
            
            # JSON 파싱
            try:
                result_data = json.loads(response_text)
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 텍스트에서 추출 시도
                result_data = self._extract_json_from_text(response_text)
            
            if not result_data:
                return None
            
            # QueryIntent로 변환
            intent_str = result_data.get("intent", "").upper()
            try:
                intent = QueryIntent(intent_str)
            except ValueError:
                self.logger.warning(f"Unknown intent: {intent_str}")
                intent = QueryIntent.GENERAL_CHAT
            
            confidence = float(result_data.get("confidence", 0.5))
            reasoning = result_data.get("reasoning", "No reasoning provided")
            
            return IntentClassificationResult(intent, confidence, reasoning)
            
        except Exception as e:
            self.logger.error(f"Error in LLM classification: {str(e)}")
            return None
    
    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """텍스트에서 JSON 추출 시도"""
        try:
            # JSON 블록 찾기
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx:end_idx+1]
                return json.loads(json_str)
        except Exception as e:
            self.logger.warning(f"Failed to extract JSON from text: {str(e)}")
        
        return None
    
    def _record_classification_stats(self, user_query: str, result: Optional[IntentClassificationResult], 
                                   start_time: float, is_error: bool) -> None:
        """통계 수집 메서드"""
        try:
            response_time_ms = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            if result:
                intent = result.intent.value
                confidence = result.confidence
            else:
                intent = "UNKNOWN"
                confidence = 0.0
            
            # 통합 시스템에 기록 (QueryInteractionMetrics 사용)
            try:
                # QueryInteractionMetrics 생성
                metrics = QueryInteractionMetrics(
                    user_query=user_query,  # 사용자 쿼리 전달
                    intent=intent,
                    intent_confidence=confidence,
                    response_time_ms=response_time_ms,
                    timestamp=time.time(),
                    is_error=is_error
                )
                
                # 통합 시스템에 기록
                integrator = get_integrator()
                integrator.record_complete_query_interaction(metrics)
                
            except Exception as e:
                # 통합 시스템을 사용할 수 없는 경우 로그만 기록
                self.logger.warning(f"Intent classification stats integration failed: {e}")
            
        except Exception as e:
            self.logger.warning(f"Failed to record classification stats: {e}")
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """분류 통계 반환 (AutoLearning 통합)"""
        return get_classification_stats()
