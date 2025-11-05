#!/usr/bin/env python3
"""
LLM-based Intent Classifier
빠른 LLM 모델을 사용한 인텐트 분류 노드
"""

import json
import logging
import re
import time
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser

from .nodes import BaseNode, QueryIntent
from .state import GraphState
from agentic_flow.llm_service import get_llm_service
from agentic_flow.intent_classification_stats import get_integrator, QueryInteractionMetrics, get_stats_collector
from agentic_flow.llm_output_parser import parse_json_response
from typing import Dict, Any, Optional
from core.config import get_settings
from langchain_core.messages import SystemMessage, HumanMessage

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
        self.json_parser = SimpleJsonOutputParser()
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
        """프롬프트 설정 (개선된 버전)"""
        
        # 시스템 프롬프트를 더 상세하게 정의 (하이브리드 분류)
        system_prompt = """
    당신은 사용자 쿼리의 의도를 5가지 카테고리로 정확하게 분류하는 전문 AI 분류기입니다.

    [분류 카테고리]
    1. GREETING: "안녕", "반가워" 등 간단한 인사말
    2. HELP_REQUEST: "도와줘", "어떻게 써?", "사용법 알려줘" 등 봇의 기능이나 사용법을 묻는 질문
    3. GENERAL_CHAT: "오늘 날씨 어때?", "고마워", "수고했어" 등 데이터와 관련 없는 일반적인 대화
    4. SIMPLE_AGGREGATION: 간단한 집계나 조회 질문 (SQL로 쉽게 처리 가능)
       - COUNT, SUM, AVG, MIN, MAX 등 단순 집계 함수 사용
       - 단일 테이블 조회 또는 단순 JOIN
       - 단순 WHERE 절 필터링
       - GROUP BY를 사용한 기본적인 그룹화
       예: "회원 수 몇 명이야?", "8월 매출 합계 알려줘", "최근 7일 신규 가입자 보여줘", "월별 매출 합계"
    
    5. COMPLEX_ANALYSIS: 복잡한 분석이나 계산이 필요한 질문 (Python으로 처리 필요)
       - 여러 데이터 집합의 교집합/합집합/차집합 연산
       - 비율 계산 (예: "A와 B의 공통 구독자 비율")
       - 복잡한 시계열 분석 및 트렌드 분석
       - 여러 단계의 데이터 가공/조합이 필요한 경우
       - 통계적 분석 (상관관계, 회귀 분석 등)
       예: "A 크리에이터 구독자 중 B 크리에이터도 구독하는 비율 알려줘"
          "최근 3개월 매출 트렌드 분석해줘"
          "상위 10개 크리에이터의 평균 구독자 증가율 비교해줘"

    [분류 규칙]
    - 데이터 조회가 아닌 경우: GREETING, HELP_REQUEST, GENERAL_CHAT 중 선택
    - 데이터 조회인 경우: 반드시 SIMPLE_AGGREGATION 또는 COMPLEX_ANALYSIS 중 선택
    - 불명확하거나 애매한 경우: SIMPLE_AGGREGATION 선택 (SQL이 더 안전하고 빠름)
    - SIMPLE_AGGREGATION과 COMPLEX_ANALYSIS의 구분 기준:
      * 단순 집계 함수로 끝나는 경우 → SIMPLE_AGGREGATION
      * 여러 데이터셋을 조합하거나 복잡한 계산이 필요한 경우 → COMPLEX_ANALYSIS
      * "비율", "교집합", "트렌드", "분석", "비교" 등의 키워드가 있으면 → COMPLEX_ANALYSIS 고려

    [분류 예시]
    - 쿼리: "하이"
      {"intent": "GREETING", "confidence": 1.0, "reasoning": "간단한 인사말입니다."}
    
    - 쿼리: "뭘 할 수 있어?"
      {"intent": "HELP_REQUEST", "confidence": 0.9, "reasoning": "봇의 기능(할 수 있는 일)에 대해 질문하고 있습니다."}
    
    - 쿼리: "수고 많았어"
      {"intent": "GENERAL_CHAT", "confidence": 0.8, "reasoning": "데이터 요청이 아닌 감사 표현입니다."}
    
    - 쿼리: "8월 신규 가입자 수 알려줘"
      {"intent": "SIMPLE_AGGREGATION", "confidence": 1.0, "reasoning": "단순 COUNT 집계로 SQL로 쉽게 처리 가능합니다."}
    
    - 쿼리: "지난 달 매출 합계 알려줘"
      {"intent": "SIMPLE_AGGREGATION", "confidence": 1.0, "reasoning": "단순 SUM 집계로 SQL로 처리 가능합니다."}
    
    - 쿼리: "A 크리에이터 구독자 중 B 크리에이터도 구독하는 비율은?"
      {"intent": "COMPLEX_ANALYSIS", "confidence": 1.0, "reasoning": "교집합과 비율 계산이 필요하여 Python으로 처리해야 합니다."}
    
    - 쿼리: "최근 3개월 매출 트렌드 분석해줘"
      {"intent": "COMPLEX_ANALYSIS", "confidence": 0.95, "reasoning": "시계열 분석과 트렌드 분석이 필요하여 Python으로 처리해야 합니다."}

    [출력 형식]
    당신은 **반드시** 다음과 같은 JSON 형식으로만 응답해야 합니다. 다른 텍스트를 포함하지 마세요.

    {{
        "intent": "GREETING|HELP_REQUEST|GENERAL_CHAT|SIMPLE_AGGREGATION|COMPLEX_ANALYSIS",
        "confidence": 0.0~1.0,
        "reasoning": "분류에 대한 간단한 근거 (특히 SIMPLE_AGGREGATION vs COMPLEX_ANALYSIS 구분 근거 포함)"
    }}
    """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
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
            
            # 대화 히스토리 가져오기
            conversation_history = state.get("conversation_history", [])
            
            # 이전 대화에서 needs_clarification이 있었는지 확인하고 보완 쿼리인지 판단
            previous_query, is_clarification_followup = self._detect_clarification_followup(user_query, conversation_history)
            
            if is_clarification_followup and previous_query:
                # 보완 쿼리인 경우: 이전 쿼리와 현재 쿼리를 결합하여 전체 쿼리로 재구성
                combined_query = f"{previous_query} ({user_query})"
                self.logger.info(f"Detected clarification followup. Combining queries: '{combined_query}'")
                state["user_query"] = combined_query
                user_query = combined_query
                # needs_clarification 플래그 제거
                state["needs_clarification"] = False
                state["conversation_response"] = None
                state["skip_sql_generation"] = False
            
            # LLM으로 인텐트 분류 (대화 히스토리 포함)
            classification_result = self._classify_intent_with_llm(user_query, conversation_history)
            
            # 결과를 state에 저장
            state["llm_intent_result"] = classification_result.to_dict() if classification_result else None
            
            # 비데이터 의도인 경우 인텐트만 설정 (응답 생성은 data_summarization에서 수행)
            if classification_result:
                intent = classification_result.intent
                if intent in [QueryIntent.GREETING, QueryIntent.HELP_REQUEST, QueryIntent.GENERAL_CHAT]:
                    state["intent"] = intent.value
                    state["skip_sql_generation"] = True
                    # conversation_response는 data_summarization 노드에서 생성 (봇 기능에 맞춘 응답)
                    self.logger.info(f"Non-data intent detected ({intent.value}), will generate response in data_summarization")
            
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
                self._record_classification_stats(user_query, classification_result, start_time, is_error) #추후 돌아와서 개발
        
        return state
    
    def _classify_intent_with_llm(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Optional[IntentClassificationResult]:
        """LLM을 사용한 인텐트 분류"""
        max_retries = 3
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                if self.llm is None:
                    self.logger.warning("LLM not available for intent classification")
                    return None
                    
                # 프롬프트 생성 (대화 히스토리 포함)
                # LangChain의 ChatPromptTemplate을 직접 사용하여 안전하게 처리
                messages = self._build_messages_with_history(query, conversation_history)
                
                # LLM 호출
                response = self.llm.invoke(messages)
                
                # 응답 내용 확인
                response_text = response.content if hasattr(response, 'content') else str(response)
                if not response_text or not response_text.strip():
                    if attempt < max_retries - 1:
                        self.logger.warning(f"LLM returned empty response (attempt {attempt + 1}/{max_retries}), retrying...")
                        import time
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        self.logger.error(f"LLM returned empty response after {max_retries} attempts")
                        # Fallback: 대화 히스토리 기반 간단한 분류 시도
                        return self._fallback_intent_classification(query, conversation_history)
                
                # LangChain 표준 Output Parser 사용
                result_data = parse_json_response(response, parser=self.json_parser, fallback_extract=True)
                
                # result_data 검증
                if not result_data:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Failed to parse LLM response as JSON (attempt {attempt + 1}/{max_retries}), retrying...")
                        import time
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        self.logger.warning(f"Failed to parse LLM response as JSON after {max_retries} attempts. Response text: {response_text[:200]}")
                        # Fallback: 대화 히스토리 기반 간단한 분류 시도
                        return self._fallback_intent_classification(query, conversation_history)
                
                if not isinstance(result_data, dict):
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Parsed result is not a dictionary (attempt {attempt + 1}/{max_retries}), retrying...")
                        import time
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        self.logger.warning(f"Parsed result is not a dictionary. Type: {type(result_data)}, Value: {result_data}")
                        return self._fallback_intent_classification(query, conversation_history)
                
                # intent 필드 추출 및 검증
                intent_value = result_data.get("intent")
                if not intent_value:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"'intent' field not found in response (attempt {attempt + 1}/{max_retries}), retrying...")
                        import time
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        self.logger.warning(f"'intent' field not found in response. Available keys: {list(result_data.keys())}")
                        return self._fallback_intent_classification(query, conversation_history)
                
                if not isinstance(intent_value, str):
                    self.logger.warning(f"'intent' field is not a string. Type: {type(intent_value)}, Value: {intent_value}")
                    # 문자열로 변환 시도
                    intent_str = str(intent_value).upper()
                else:
                    intent_str = intent_value.upper()
                
                # QueryIntent로 변환
                try:
                    intent = QueryIntent(intent_str)
                except ValueError:
                    self.logger.warning(f"Unknown intent: {intent_str}. Available intents: {[e.value for e in QueryIntent]}")
                    intent = QueryIntent.GENERAL_CHAT
                
                # confidence 필드 추출 및 검증
                confidence_value = result_data.get("confidence", 0.5)
                try:
                    confidence = float(confidence_value) if confidence_value is not None else 0.5
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid confidence value: {confidence_value}, using default 0.5")
                    confidence = 0.5
                
                # reasoning 필드 추출
                reasoning = result_data.get("reasoning", "No reasoning provided")
                if not isinstance(reasoning, str):
                    reasoning = str(reasoning) if reasoning else "No reasoning provided"
                
                return IntentClassificationResult(intent, confidence, reasoning)
                
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Error in LLM classification (attempt {attempt + 1}/{max_retries}): {str(e)}, retrying...")
                    import time
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    self.logger.error(f"Error in LLM classification after {max_retries} attempts: {str(e)}", exc_info=True)
                    # Fallback: 대화 히스토리 기반 간단한 분류 시도
                    return self._fallback_intent_classification(query, conversation_history)
        
        # 모든 재시도 실패 시 fallback
        return self._fallback_intent_classification(query, conversation_history)
    
    def _fallback_intent_classification(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Optional[IntentClassificationResult]:
        """
        LLM 호출 실패 시 대화 히스토리 기반 간단한 분류 (Fallback)
        """
        try:
            query_lower = query.lower().strip()
            
            # 간단한 키워드 기반 분류
            greeting_keywords = ["안녕", "하이", "헬로", "반가워", "만나서"]
            help_keywords = ["도와", "도움", "사용법", "어떻게", "뭘 할 수", "기능"]
            
            if any(keyword in query_lower for keyword in greeting_keywords):
                return IntentClassificationResult(
                    QueryIntent.GREETING,
                    0.7,
                    "Fallback: 키워드 기반 분류 (인사말)"
                )
            
            if any(keyword in query_lower for keyword in help_keywords):
                return IntentClassificationResult(
                    QueryIntent.HELP_REQUEST,
                    0.7,
                    "Fallback: 키워드 기반 분류 (도움말 요청)"
                )
            
            # 대화 히스토리 기반 분류
            if conversation_history:
                # 이전 대화에서 이름을 물어본 적이 있으면 GENERAL_CHAT으로 분류
                recent_history = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
                for msg in recent_history:
                    content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                    if "이름" in content or "뭐라고" in content:
                        return IntentClassificationResult(
                            QueryIntent.GENERAL_CHAT,
                            0.8,
                            "Fallback: 대화 히스토리 기반 분류 (일반 대화)"
                        )
            
            # 기본값: GENERAL_CHAT
            return IntentClassificationResult(
                QueryIntent.GENERAL_CHAT,
                0.6,
                "Fallback: 기본 분류 (일반 대화)"
            )
            
        except Exception as e:
            self.logger.error(f"Error in fallback intent classification: {str(e)}")
            return None
    
    def _detect_clarification_followup(
        self, 
        current_query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[Optional[str], bool]:
        """
        이전 대화에서 명확화 요청이 있었는지 확인하고, 현재 쿼리가 보완 정보인지 판단합니다.
        
        Returns:
            (previous_query, is_clarification_followup): 이전 쿼리와 보완 쿼리 여부
        """
        if not conversation_history or len(conversation_history) < 2:
            return None, False
        
        # 마지막 assistant 응답 확인
        last_assistant_msg = None
        prev_user_query = None
        
        # 히스토리를 역순으로 확인하여 마지막 assistant 응답과 그 이전 user 쿼리 찾기
        # 최근 5개 메시지만 확인 (너무 오래된 대화는 무시)
        recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
        
        for i in range(len(recent_history) - 1, -1, -1):
            msg = recent_history[i]
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            
            if role == "assistant" and not last_assistant_msg:
                last_assistant_msg = content
            elif role == "user" and not prev_user_query and last_assistant_msg:
                prev_user_query = content
                break
        
        # 명확화 요청 패턴 확인
        clarification_patterns = [
            "어떤 컬럼으로 그룹핑할까요",
            "어떤 컬럼으로",
            "어떤 컬럼",
            "그룹핑할까요",
            "확인할 수 없습니다",
            "다시 질문해주시면",
            "예:",
            "creator_id",
            "creator_no"
        ]
        
        is_clarification_response = False
        if last_assistant_msg:
            is_clarification_response = any(
                pattern in str(last_assistant_msg) for pattern in clarification_patterns
            )
        
        # 현재 쿼리가 보완 정보인지 확인
        clarification_keywords = [
            "칼럼으로", "컬럼으로", "column", "그룹핑", "group by",
            "creator_id", "creator_no", "member_id", "member_no",
            "식별", "식별해", "으로 식별", "로 식별", "으로 가져", "로 가져",
            "으로 조회", "로 조회", "으로 찾", "로 찾"
        ]
        
        is_followup = any(
            keyword in current_query.lower() for keyword in clarification_keywords
        )
        
        if is_clarification_response and is_followup and prev_user_query:
            self.logger.info(
                f"Detected clarification followup: previous='{prev_user_query[:50]}...', "
                f"current='{current_query}', assistant_response contains clarification"
            )
            return prev_user_query, True
        
        return None, False
    
    def _build_messages_with_history(
        self, 
        query: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List:
        """
        대화 히스토리를 포함한 메시지 리스트 생성 (LangChain 형식)
        
        사용자 쿼리에 중괄호가 포함되어 있을 수 있으므로, 
        프롬프트 템플릿 대신 직접 메시지를 구성하여 안전하게 처리합니다.
        
        Args:
            query: 현재 사용자 쿼리
            conversation_history: 이전 대화 히스토리 [{"role": "user|assistant", "content": "..."}]
            
        Returns:
            LangChain 메시지 리스트
        """
  
        
        # 시스템 프롬프트 구성 (중괄호를 이중으로 작성하여 템플릿 파싱 충돌 방지)
        system_prompt = """당신은 사용자 쿼리의 의도를 5가지 카테고리로 정확하게 분류하는 전문 AI 분류기입니다.

[분류 카테고리]
1. GREETING: "안녕", "반가워" 등 간단한 인사말
2. HELP_REQUEST: "도와줘", "어떻게 써?", "사용법 알려줘" 등 봇의 기능이나 사용법을 묻는 질문
3. GENERAL_CHAT: "오늘 날씨 어때?", "고마워", "수고했어" 등 데이터와 관련 없는 일반적인 대화
4. SIMPLE_AGGREGATION: 간단한 집계나 조회 질문 (SQL로 쉽게 처리 가능)
   - COUNT, SUM, AVG, MIN, MAX 등 단순 집계 함수 사용
   - 단일 테이블 조회 또는 단순 JOIN
   - 단순 WHERE 절 필터링
   - GROUP BY를 사용한 기본적인 그룹화
   예: "회원 수 몇 명이야?", "8월 매출 합계 알려줘", "최근 7일 신규 가입자 보여줘", "월별 매출 합계"

5. COMPLEX_ANALYSIS: 복잡한 분석이나 계산이 필요한 질문 (Python으로 처리 필요)
   - 여러 데이터 집합의 교집합/합집합/차집합 연산
   - 비율 계산 (예: "A와 B의 공통 구독자 비율")
   - 복잡한 시계열 분석 및 트렌드 분석
   - 여러 단계의 데이터 가공/조합이 필요한 경우
   - 통계적 분석 (상관관계, 회귀 분석 등)
   예: "A 크리에이터 구독자 중 B 크리에이터도 구독하는 비율 알려줘"
      "최근 3개월 매출 트렌드 분석해줘"
      "상위 10개 크리에이터의 평균 구독자 증가율 비교해줘"

[분류 규칙]
- 데이터 조회가 아닌 경우: GREETING, HELP_REQUEST, GENERAL_CHAT 중 선택
- 데이터 조회인 경우: 반드시 SIMPLE_AGGREGATION 또는 COMPLEX_ANALYSIS 중 선택
- 불명확하거나 애매한 경우: SIMPLE_AGGREGATION 선택 (SQL이 더 안전하고 빠름)
- SIMPLE_AGGREGATION과 COMPLEX_ANALYSIS의 구분 기준:
  * 단순 집계 함수로 끝나는 경우 → SIMPLE_AGGREGATION
  * 여러 데이터셋을 조합하거나 복잡한 계산이 필요한 경우 → COMPLEX_ANALYSIS
  * "비율", "교집합", "트렌드", "분석", "비교" 등의 키워드가 있으면 → COMPLEX_ANALYSIS 고려

[출력 형식]
반드시 다음 JSON 형식으로만 응답하세요. 다른 텍스트를 포함하지 마세요.

{{
    "intent": "GREETING|HELP_REQUEST|GENERAL_CHAT|SIMPLE_AGGREGATION|COMPLEX_ANALYSIS",
    "confidence": 0.0~1.0,
    "reasoning": "분류에 대한 간단한 근거 (특히 SIMPLE_AGGREGATION vs COMPLEX_ANALYSIS 구분 근거 포함)"
}}"""
        
        # 사용자 쿼리 메시지 구성
        user_query_message = f"쿼리: {query}"
        
        # 히스토리가 있으면 메시지에 추가
        if conversation_history and len(conversation_history) > 0:
            history_content = "\n\n[이전 대화 히스토리]\n"
            for msg in conversation_history[-5:]:  # 최근 5개 메시지만 포함 (토큰 절약)
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    history_content += f"사용자: {content}\n"
                elif role == "assistant":
                    history_content += f"어시스턴트: {content}\n"
            
            history_content += "\n위 대화 히스토리를 고려하여 현재 쿼리의 의도를 분류해주세요."
            user_query_message += history_content
        
        # 메시지 리스트 구성
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query_message)
        ]
        
        return messages
    
    # Note: JSON 파싱 로직은 llm_output_parser 모듈로 이동됨
    # parse_json_response 함수를 사용하세요
    
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
    
    # NOTE: 응답 생성 로직은 data_summarization 노드로 이동됨
    # llm_intent_classifier는 인텐트 분류만 수행하고, 
    # 실제 응답 생성은 data_summarization에서 봇 기능에 맞춘 응답을 생성합니다.
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """분류 통계 반환 (AutoLearning 통합)"""
        stats = get_stats_collector().get_stats()
        return stats.to_dict()
