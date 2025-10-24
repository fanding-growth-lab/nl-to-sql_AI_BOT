#!/usr/bin/env python3
"""
LLM-based Intent Classifier
빠른 LLM 모델을 사용한 인텐트 분류 노드
"""

import json
import logging
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .nodes import BaseNode, QueryIntent
from typing import Dict, Any, Optional


class IntentClassificationResult:
    """인텐트 분류 결과"""
    
    def __init__(self, intent: QueryIntent, confidence: float, reasoning: str):
        self.intent = intent
        self.confidence = confidence
        self.reasoning = reasoning
    
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
        self.parser = JsonOutputParser()
        self._setup_prompt()
    
    def _initialize_llm(self) -> Optional[ChatGoogleGenerativeAI]:
        """빠른 LLM 모델 초기화"""
        try:
            # 디버깅을 위한 로그 추가
            llm_config = self.config.get('llm', {})
            api_key = llm_config.get('api_key', '')
            model = llm_config.get('model', 'gemini-2.5-pro')
            
            self.logger.info(f"LLMIntentClassifier config: llm={llm_config}")
            self.logger.info(f"LLMIntentClassifier api_key: {api_key[:20] if api_key else 'EMPTY'}...")
            self.logger.info(f"LLMIntentClassifier model: {model}")
            
            # 빠른 응답을 위한 가벼운 모델 사용
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",  # 더 빠른 모델 사용
                google_api_key=api_key,
                temperature=0.1,  # 일관된 결과를 위한 낮은 temperature
                max_output_tokens=256,  # 더 짧은 응답
                request_timeout=10.0,  # 10초 타임아웃
                convert_system_message_to_human=True  # 최신 버전 호환성
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
        
        try:
            user_query = state.get("user_query")
            if not user_query:
                self.logger.error("user_query is None or empty")
                state["llm_intent_result"] = None
                return state
            
            # LLM이 사용 불가능한 경우 None 반환
            if not self.llm:
                self.logger.warning("LLM not available, skipping LLM intent classification")
                state["llm_intent_result"] = None
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
            
        except Exception as e:
            self.logger.error(f"Error in LLM intent classification: {str(e)}")
            state["llm_intent_result"] = None
        
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
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """분류 통계 반환"""
        # 실제 구현에서는 통계 수집 로직 추가
        return {
            "total_classifications": 0,
            "average_confidence": 0.0,
            "intent_distribution": {},
            "error_rate": 0.0
        }
