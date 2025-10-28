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
from typing import Dict, Any, Optional
from core.config import get_settings
from .intent_classification_stats import get_stats_collector


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
        self.llm = self._initialize_llm()
        # Removed: JsonOutputParser (not used)
        self._setup_prompt()
    
    def _initialize_llm(self) -> Optional[ChatGoogleGenerativeAI]:
        """빠른 LLM 모델 초기화"""
        try:
            # 설정에서 인텐트 분류용 LLM 설정 가져오기
            settings = get_settings()
            
            self.logger.info(f"Intent LLM config: model={settings.llm.intent_model}, "
                           f"temperature={settings.llm.intent_temperature}, "
                           f"max_tokens={settings.llm.intent_max_tokens}")
            
            # 인텐트 분류용 빠른 모델 사용
            return ChatGoogleGenerativeAI(
                model=settings.llm.intent_model,
                google_api_key=settings.llm.api_key,
                temperature=settings.llm.intent_temperature,
                max_output_tokens=settings.llm.intent_max_tokens,
                request_timeout=10.0,
                convert_system_message_to_human=True
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
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """인텐트 분류 처리"""
        self._log_processing(state, "LLMIntentClassifier")
        
        # 통계 수집을 위한 시작 시간 기록
        start_time = time.time()
        is_error = False
        
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
            result = self._classify_intent_with_llm(user_query)
            
            # 결과를 state에 저장
            state["llm_intent_result"] = result.to_dict() if result else None
            
            # 로깅
            if result:
                self.logger.info(f"LLM Intent Classification: {result.intent.value} (confidence: {result.confidence:.2f})")
                self.logger.debug(f"Reasoning: {result.reasoning}")
            else:
                self.logger.warning("LLM intent classification failed")
                is_error = True
            
        except Exception as e:
            self.logger.error(f"Error in LLM intent classification: {str(e)}")
            state["llm_intent_result"] = None
            is_error = True
        
        finally:
            # 통계 수집
            self._record_classification_stats(result, start_time, is_error)
        
        return state
    
    def _classify_intent_with_llm(self, query: str) -> Optional[IntentClassificationResult]:
        """LLM을 사용한 인텐트 분류"""
        try:
            # 프롬프트 생성
            formatted_prompt = self.prompt.format(query=query)
            
            # LLM 호출
            response = self.llm.invoke(formatted_prompt)
            
            # JSON 파싱
            try:
                result_data = json.loads(response.content)
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 텍스트에서 추출 시도
                result_data = self._extract_json_from_text(response.content)
            
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
    
    def _record_classification_stats(self, result: Optional[IntentClassificationResult], 
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
            
            # 통계 수집기 가져오기 및 기록
            stats_collector = get_stats_collector()
            stats_collector.record_classification(
                intent=intent,
                confidence=confidence,
                response_time_ms=response_time_ms,
                is_error=is_error
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to record classification stats: {e}")
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """분류 통계 반환"""
        try:
            stats_collector = get_stats_collector()
            stats = stats_collector.get_stats()
            return stats.to_dict()
        except Exception as e:
            self.logger.error(f"Failed to get classification stats: {e}")
            return {
                "total_classifications": 0,
                "average_confidence": 0.0,
                "intent_distribution": {},
                "error_rate": 0.0,
                "response_times": {"min": 0, "max": 0, "avg": 0},
                "last_updated": 0
            }
