"""
LangGraph Node Components for NL-to-SQL Pipeline

This module implements the individual nodes that make up the LangGraph pipeline.
"""

import re
import logging
import sqlparse
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from .prompts import GeminiSQLGenerator, SQLPromptTemplate
from .fanding_sql_templates import FandingSQLTemplates
# Removed unused imports: enhanced_rag_mapper, data_insight_analyzer, dynamic_schema_expander

from .state import (
    GraphState, Entity, SchemaMapping, SQLResult, 
    QueryIntent, QueryComplexity,
    set_sql_result, set_rag_mapping_result, set_dynamic_pattern, 
    set_fanding_template, set_conversation_response, clear_sql_generation,
    get_effective_sql, is_sql_generation_skipped
)
from core.config import get_settings
from core.db import get_db_session, get_cached_db_schema, extract_table_names, extract_column_names, validate_sql_syntax
from core.logging import get_logger

logger = get_logger(__name__)


# Constants and Configuration for Agentic Flow
# Intent Classification Patterns
GREETING_PATTERNS = [
    "안녕", "반가워", "hello", "hi", "좋은 아침", "좋은 저녁", 
    "환영", "인사", "만나서 반가워", "반갑습니다", "안녕하세요",
    "안녕하세요", "반갑습니다", "처음 뵙겠습니다", "만나서 반갑습니다",
    "좋은 하루", "좋은 하루 되세요", "좋은 하루 보내세요"
]

HELP_REQUEST_PATTERNS = [
    "도움", "사용법", "어떻게", "help", "명령어", 
    "도와줘", "설명", "가이드", "사용법", "도움말",
    "사용법 알려줘", "어떻게 사용하나요", "기능", "기능이 뭐야",
    "뭐가 있어", "뭘 할 수 있어", "할 수 있는 것", "기능 설명",
    "너가 할 수 있는 일", "뭐야", "뭐지", "뭔가", "뭔데"
]

GENERAL_CHAT_PATTERNS = [
    "어때", "어떠", "좋아", "나쁘", "재미", "재미있", "지루", "피곤",
    "날씨", "오늘", "어제", "내일", "주말", "휴일", "일", "일정",
    "고마워", "감사", "미안", "죄송", "괜찮", "괜찮아", "괜찮습니다",
    "뭐야", "뭐지", "뭔가", "뭔데", "뭔가요", "뭔가요?"
]

FANDING_DATA_KEYWORDS = [
    "멤버십", "성과", "회원", "매출", "방문자", "리텐션", "포스트", 
    "조회수", "인기", "분석", "통계", "리포트", "월간", 
    "일간", "주간", "년간", "크리에이터", "펀딩", "프로젝트",
    "8월", "9월", "10월", "11월", "12월", "1월", "2월", "3월", 
    "4월", "5월", "6월", "7월", "올해", "작년", "지난달", "이번달",
    "신규", "이탈", "활성", "구독", "결제", "수익", "매출액",
    "현황", "상황", "결과", "성과분석", "성과", "분석해줘", "보고서",
    "요약", "정리", "현재", "최근", "지금", "오늘", "어제", "내일"
]

DATA_QUERY_PATTERNS = [
    "조회", "검색", "보여줘", "찾아", "테이블", "쿼리",
    "개수", "수", "합계", "평균", "최대", "최소", "통계",
    "알려줘", "보여줘", "찾아줘", "가져와", "얼마나", "몇 개",
    "몇 명", "얼마", "어느 정도"
]

QUESTION_PATTERNS = [
    "뭐", "무엇", "어떤", "어디", "언제", "왜", "어떻게", "누구",
    "뭔가", "뭔지", "뭔데", "뭐야", "뭐지", "뭔가요", "뭔가요?"
]

GRATITUDE_PATTERNS = [
    "고마워", "감사", "감사합니다", "고마워요", "고맙습니다",
    "수고", "수고하셨", "수고하셨어요", "수고하셨습니다"
]

# Response Templates
GREETING_RESPONSES = [
    "안녕하세요! 👋 Fanding Data Report 봇입니다. 무엇을 도와드릴까요?",
    "안녕하세요! 😊 데이터 분석을 도와드리겠습니다.",
    "반갑습니다! 🤖 멤버십 성과나 회원 데이터를 조회해드릴 수 있어요.",
    "안녕하세요! 📊 Fanding 데이터를 분석해드리겠습니다."
]

GENERAL_CHAT_RESPONSES = [
    "안녕하세요! 😊 데이터 분석에 대해 궁금한 것이 있으시면 언제든 말씀해주세요!",
    "네, 듣고 있어요! 📊 Fanding 데이터를 조회하고 싶으시면 말씀해주세요.",
    "좋은 하루 보내세요! 🤖 멤버십 성과나 회원 데이터가 궁금하시면 언제든 물어보세요.",
    "감사합니다! 😊 데이터 분석을 도와드릴 준비가 되어있어요."
]

# SQL Security Keywords
DANGEROUS_SQL_KEYWORDS = [
    'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 
    'TRUNCATE', 'EXEC', 'EXECUTE', 'UNION', 'SCRIPT',
    'GRANT', 'REVOKE', 'COMMIT', 'ROLLBACK'
]

# Table Name Mapping
TABLE_NAME_MAPPING = {
    # 회원 관련
    "users": "t_member",
    "user": "t_member",
    "members": "t_member",
    "member": "t_member",
    "회원": "t_member",
    "사용자": "t_member",
    
    # 회원 정보 관련
    "user_info": "t_member_info",
    "member_info": "t_member_info",
    "회원정보": "t_member_info",
    
    # 회원 프로필 관련
    "user_profile": "t_member_profile",
    "member_profile": "t_member_profile",
    "profiles": "t_member_profile",
    "프로필": "t_member_profile",
    
    # 크리에이터 관련
    "creators": "t_creator",
    "creator": "t_creator",
    "크리에이터": "t_creator",
    "창작자": "t_creator",
    
    # 펀딩 관련
    "fundings": "t_funding",
    "funding": "t_funding",
    "펀딩": "t_funding",
    "projects": "t_funding",
    "프로젝트": "t_funding",
    
    # 펀딩 참여자
    "funding_members": "t_funding_member",
    "backers": "t_funding_member",
    "supporters": "t_funding_member",
    "후원자": "t_funding_member",
    
    # 팔로우 관계
    "follows": "t_follow",
    "follow": "t_follow",
    "팔로우": "t_follow",
    
    # 주문 관련
    "orders": "t_order",
    "order": "t_order",
    "주문": "t_order",
}

# Korean to English Mappings
KOREAN_MAPPINGS = {
    '보여줘': 'show',
    '찾아줘': 'find',
    '가져와': 'get',
    '개수': 'count',
    '합계': 'sum',
    '평균': 'average',
    '최대': 'max',
    '최소': 'min'
}

# Entity Extraction Keywords
MEMBER_KEYWORDS = ["회원", "멤버", "사용자", "유저", "member", "user", "회원수", "멤버수"]
CREATOR_KEYWORDS = ["크리에이터", "창작자", "작가", "아티스트", "제작자", "creator"]
DATE_KEYWORDS = ["신규", "현황", "월간", "일간", "주간", "년간"]
LOGIN_KEYWORDS = ["로그인", "login", "접속"]
RANKING_KEYWORDS = ["top", "top5", "top10", "상위", "최고", "많은", "적은", "순위"]
STATISTICS_KEYWORDS = ["개수", "수", "합계", "평균", "최대", "최소", "통계", "분석"]

# Confidence Thresholds
LLM_CONFIDENCE_THRESHOLD_HIGH = 0.8
LLM_CONFIDENCE_THRESHOLD_MEDIUM = 0.6
LLM_CONFIDENCE_THRESHOLD_LOW = 0.3
RAG_CONFIDENCE_THRESHOLD = 0.6
SQL_GENERATION_CONFIDENCE_THRESHOLD = 0.7


# Response Generator Functions

def generate_greeting_response(user_query: str) -> str:
    """Generate a random greeting response."""
    return random.choice(GREETING_RESPONSES)

def generate_help_response(user_query: str) -> str:
    """Generate a help response."""
    return """🤖 **Fanding Data Report 봇 사용법**

**📊 데이터 조회 기능:**
• "활성 회원 수 조회해줘" - 활성 회원 수 확인
• "8월 멤버십 성과 분석해줘" - 특정 월 성과 분석
• "전체 회원 수 보여줘" - 전체 회원 수 확인
• "신규 회원 현황 알려줘" - 신규 회원 현황

**💡 사용 팁:**
• 구체적인 질문을 해주세요 (예: "8월 성과", "활성 회원")
• 날짜나 기간을 명시해주세요 (예: "이번 달", "지난 주")
• 멤버십, 회원, 성과 등 키워드를 포함해주세요

**❓ 궁금한 점이 있으시면 언제든 말씀해주세요!**"""

def generate_general_chat_response(user_query: str) -> str:
    """Generate a random general chat response."""
    return random.choice(GENERAL_CHAT_RESPONSES)

def generate_error_response(error: Exception) -> str:
    """Generate user-friendly error response."""
    error_type = type(error).__name__
    
    # 특정 에러 타입별 맞춤형 응답
    if "UnicodeEncodeError" in error_type:
        return """😅 **인코딩 오류가 발생했습니다**

죄송합니다. 특수 문자나 이모지 처리 중 문제가 발생했어요.
다시 시도해주시거나 다른 방식으로 질문해주세요! 🤖"""
    
    elif "ConnectionError" in error_type or "TimeoutError" in error_type:
        return """🌐 **연결 오류가 발생했습니다**

데이터베이스나 외부 서비스 연결에 문제가 있어요.
잠시 후 다시 시도해주세요! 🔄"""
    
    elif "ValueError" in error_type or "TypeError" in error_type:
        return """⚠️ **입력 처리 오류가 발생했습니다**

질문을 이해하는 데 문제가 있었어요.
다른 방식으로 질문해주시면 도와드릴게요! 💡"""
    
    else:
        return """😔 **처리 중 오류가 발생했습니다**

예상치 못한 문제가 발생했어요.
다시 시도해주시거나 기술팀에 문의해주세요! 🛠️

**💡 도움말:** "사용법 알려줘"라고 말씀해주시면 사용법을 안내해드릴게요."""

def generate_clarification_question(user_query: str) -> str:
    """Generate a clarification question for ambiguous queries."""
    q = user_query.lower()
    needs_topk = ("top" in q or "상위" in q or "top5" in q)
    needs_period = any(k in q for k in ["이번", "지난", "이번달", "지난달", "월", "분기", "주", "week", "month", "quarter"])
    needs_metric = any(k in q for k in ["회원수", "신규", "활성", "로그인", "조회수", "매출", "판매"]) 
    
    parts = []
    if needs_period:
        parts.append("기간(예: 2025-08, 지난달)을 알려주세요.")
    if needs_topk:
        parts.append("상위 K 개(예: Top5)는 몇 개를 원하시나요?")
    if needs_metric:
        parts.append("어떤 지표를 기준으로 랭킹을 원하시나요? (예: 신규 회원수)")
    if not parts:
        parts.append("기간/지표/Top-K 중 필요한 정보를 알려주세요.")
    
    return "질의를 정확히 처리하기 위해 다음을 확인해 주세요: " + " ".join(parts)


class BaseNode(ABC):
    """Base class for all pipeline nodes."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
    
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


class NLProcessor(BaseNode):
    """Natural Language Processing node for query analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm = self._initialize_llm()
        self.fanding_templates = FandingSQLTemplates()
        # Removed: EnhancedRAGMapper and DynamicSchemaExpander (deleted modules)
        # self.rag_mapper = EnhancedRAGMapper(config)
        # self.schema_expander = DynamicSchemaExpander(config)
    
    def _initialize_llm(self):
        """Initialize the LLM for natural language processing."""
        settings = get_settings()
        try:
            return ChatGoogleGenerativeAI(
                model=settings.llm.model,
                google_api_key=settings.llm.api_key,
                temperature=settings.llm.temperature,
                max_output_tokens=settings.llm.max_tokens,
                request_timeout=10.0  # 10초 타임아웃 설정
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM: {str(e)}. Using mock LLM.")
            # Try to create a simple LLM instance for testing
            try:
                import os
                return ChatGoogleGenerativeAI(
                    model="gemini-2.5-pro",
                    google_api_key=os.environ.get('GOOGLE_API_KEY', ''),
                    temperature=0.1,
                    max_output_tokens=1024,
                    request_timeout=10.0  # 10초 타임아웃 설정
                )
            except:
                return None
    
    def process(self, state: GraphState) -> GraphState:
        """Process natural language query and extract intent and entities."""
        self._log_processing(state, "NLProcessor")
        
        try:
            # 입력 데이터 검증
            user_query = state.get("user_query")
            if not user_query:
                self.logger.error("user_query is None or empty")
                state["conversation_response"] = "죄송합니다. 질문을 받지 못했어요. 다시 말씀해주세요! 😊"
                state["skip_sql_generation"] = True
                state["success"] = False
                return state
            
            # Normalize query
            normalized_query = self._normalize_query(user_query)
            
            # 정규화된 쿼리 검증
            if not normalized_query or len(normalized_query.strip()) == 0:
                self.logger.error("normalized_query is empty after processing")
                state["conversation_response"] = "죄송합니다. 질문을 이해하지 못했어요. 다시 말씀해주세요! 🤔"
                state["skip_sql_generation"] = True
                state["success"] = False
                return state
            
            # Extract intent and entities (LLM 결과 포함)
            llm_intent_result = state.get("llm_intent_result")
            intent, entities = self._extract_intent_and_entities(normalized_query, llm_intent_result)   # NOTE: confidence가 높다면 LLM의 것을, 낮다면 규칙 기반으로 intent를 분류하고 필요하다면 entity를 추출함
            
            # Update state
            state["normalized_query"] = normalized_query
            state["intent"] = intent
            state["entities"] = entities
            
            # 인사말 처리 (우선순위 1)
            if intent == QueryIntent.GREETING:
                response = self._handle_greeting(user_query)
                set_conversation_response(state, response, skip_sql=True)
                state["success"] = True
                self.logger.info(f"Greeting handled: {user_query}")
                return state
            
            # 도움말 요청 처리 (우선순위 2)
            if intent == QueryIntent.HELP_REQUEST:
                response = self._handle_help_request(user_query)
                set_conversation_response(state, response, skip_sql=True)
                state["success"] = True
                self.logger.info(f"Help request handled: {user_query}")
                return state
            
            # 스키마 정보 요청 처리 (우선순위 3 - SHOW/DESCRIBE 대안)
            schema_info_response = self.fanding_templates.get_schema_info(user_query)
            if schema_info_response:
                state["conversation_response"] = schema_info_response
                state["intent"] = QueryIntent.HELP_REQUEST
                state["skip_sql_generation"] = True
                state["success"] = True
                self.logger.info(f"Schema information request handled: {user_query}")
                return state
            
            # 인텐트별 처리 (개선된 버전)
            if intent == QueryIntent.GENERAL_CHAT:
                # 일반 대화 처리
                response = self._handle_general_chat(user_query)
                set_conversation_response(state, response, skip_sql=True)
                clear_sql_generation(state)
                state["success"] = True
                self.logger.info(f"General chat handled: {intent}")
                
            elif intent == QueryIntent.DATA_QUERY:
                # 데이터 조회 의도 - Fanding 템플릿 매칭 시도
                self.logger.info(f"Data query intent detected: {user_query}")
                self._handle_data_query(state, user_query)
                state["success"] = True
            
            else:
                # 알 수 없는 인텐트 - 일반 대화로 처리
                self.logger.warning(f"Unknown intent: {intent}, treating as general chat")
                state["skip_sql_generation"] = True
                state["conversation_response"] = self._handle_general_chat(user_query)
                state["sql_query"] = None
                state["validated_sql"] = None
                state["success"] = True
                self.logger.info(f"Unknown intent handled as general chat: {intent}")
            
            # Log confidence
            confidence = self._calculate_confidence(normalized_query, intent, entities)
            state["confidence_scores"]["nl_processing"] = confidence
            
            self.logger.info(f"Processed query: {normalized_query}")
            self.logger.info(f"Intent: {intent}, Entities: {len(entities)}")
            
        except Exception as e:
            self.logger.error(f"Error in NLProcessor: {str(e)}", exc_info=True)
            # 사용자 친화적인 에러 메시지 생성
            error_response = self._generate_error_response(e)
            state["conversation_response"] = error_response
            state["skip_sql_generation"] = True
            state["sql_query"] = None
            state["validated_sql"] = None
            state["success"] = False
            state["error_message"] = f"Natural language processing failed: {str(e)}"
        
        return state
    
    def _handle_greeting(self, user_query: str) -> str:
        """인사말 처리 (랜덤 응답)"""
        return generate_greeting_response(user_query)
    
    def _handle_help_request(self, user_query: str) -> str:
        """도움말 요청 처리"""
        return generate_help_response(user_query)
    
    def _handle_general_chat(self, user_query: str) -> str:
        """일반 대화 처리 (랜덤 응답)"""
        return generate_general_chat_response(user_query)
    
    def _generate_error_response(self, error: Exception) -> str:
        """사용자 친화적인 에러 응답 생성"""
        return generate_error_response(error)
    
    def _handle_data_query(self, state: GraphState, user_query: str) -> None:
        """데이터 조회 의도 처리 (RAG + 동적 스키마 확장 통합)"""
        # 애매한 쿼리인지 먼저 확인
        if self.fanding_templates.is_ambiguous_query(user_query):
            self.logger.info(f"🔍 Ambiguous query detected - requesting clarification: {user_query}")
            clarification_question = self.fanding_templates.generate_clarification_question(user_query)
            state["conversation_response"] = clarification_question
            state["skip_sql_generation"] = True
            self.logger.info("✅ Generated clarification question for ambiguous query (this is normal behavior)")
            return
        
        # 1. RAG 매핑 시도 (우선순위 높음) - DISABLED: EnhancedRAGMapper deleted
        # try:
        #     rag_result = self.rag_mapper.map_query_to_schema(user_query, context={"prefer_detailed": True})
        #     if rag_result and rag_result.confidence > LLM_CONFIDENCE_THRESHOLD_HIGH:
        #         self.logger.info(f"RAG mapping successful: {rag_result.source.value} (confidence: {rag_result.confidence:.2f})")
        #         set_rag_mapping_result(state, rag_result)
        #         state["skip_sql_generation"] = False
        #         self.logger.info(f"RAG SQL applied: {rag_result.sql_template[:100]}...")
        #         return
        # except Exception as e:
        #     self.logger.warning(f"RAG mapping failed: {str(e)}")
        
        # Skip RAG mapping and go directly to Fanding templates
        
        # 2. Fanding 템플릿 매칭 시도 (폴백)
        fanding_template = self.fanding_templates.match_query_to_template(user_query)
        if fanding_template:
            self.logger.info(f"Fanding template matched: {fanding_template.name}")
            set_fanding_template(state, fanding_template)
            state["skip_sql_generation"] = False
            self.logger.info(f"SQL template applied: {fanding_template.sql_template}")
        else:
            # 3. 동적 월별 템플릿 생성 시도 (멤버십 성과 관련)
            try:
                dynamic_template = self.fanding_templates.create_dynamic_monthly_template(user_query)
                if dynamic_template:
                    self.logger.info(f"Dynamic monthly template created: {dynamic_template.name}")
                    set_fanding_template(state, dynamic_template)
                    state["skip_sql_generation"] = False
                    self.logger.info(f"Dynamic SQL applied: {dynamic_template.sql_template[:100]}...")
                    return
            except Exception as e:
                self.logger.warning(f"Dynamic monthly template creation failed: {str(e)}")
            
            # 4. 모든 방법 실패 시 일반 SQL 생성으로 진행
            self.logger.info("No template/pattern matched, proceeding with general SQL generation")
            state["skip_sql_generation"] = False

    def _generate_conversation_response(self, intent: QueryIntent, query: str) -> str:
        """인텐트별 대화 응답 생성 (기존 메서드 유지)"""
        if intent == QueryIntent.GREETING:
            return """안녕하세요! 👋 팬딩 데이터 리포트 시스템을 돕는 AI 어시스턴트, PF_bearbot이라고 해요.

저는 크리에이터님의 데이터를 분석해서 궁금한 점들을 바로바로 알려드리는 역할을 하고 있어요.

제가 주로 도와드릴 수 있는 것들이에요.

멤버십 데이터 분석: 회원 수나 신규/이탈 현황이 어떤지 알려드려요.

월간 성과 리포트: 매출이나 방문자, 리텐션 같은 핵심 성과를 정리해 드려요.

콘텐츠 성과 분석: 어떤 포스트가 인기가 많았는지 조회수를 바탕으로 알려드릴 수 있어요.

자동 리포트 생성: 매월 크리에이터님께 꼭 맞는 리포트를 만들어 드려요.

예를 들어, 저에게 이렇게 한번 물어보세요.

"8월 멤버십 성과 어땠어?"

"회원 수 변화 추이 보여줘"

"인기 포스트 TOP5 알려줘"

"리텐션 현황은?"

물론 "월별 매출 성장률"이나 "고객 평균 수명(LTV) 분석" 같은 좀 더 깊이 있는 분석도 가능하답니다.

궁금한 게 생기면 언제든 편하게 저를 찾아주세요! 🤖"""
        
        elif intent == QueryIntent.HELP_REQUEST:
            return """🔍 **PF_bearbot 도움말 - Fanding Data Report**

**🚀 주요 기능:**
• 📊 **멤버십 데이터 분석**: 회원 수, 신규/이탈, 활성도 분석
• 📈 **성과 리포트**: 월간 매출, 방문자, 리텐션 분석
• 🔍 **콘텐츠 성과**: 포스트 조회수, 인기 콘텐츠 분석
• 📋 **자동 리포트**: 크리에이터 맞춤형 월간 리포트 생성

**💡 기본 사용법:**
```
"8월 멤버십 성과" → 월간 성과 리포트
"회원 수 변화 추이" → 회원 증감 분석
"인기 포스트 TOP5" → 콘텐츠 성과 분석
"리텐션 현황" → 회원 유지율 분석
```

**🎯 고급 분석 예시:**
• "월별 매출 성장률 분석"
• "멤버십 구독 기간 분포"
• "고객 평균 수명 분석"
• "포스트 발행과 방문자 상관관계"
• "신규 vs 기존 회원 비율"

**⚡ 빠른 명령어:**
• "도움말" → 이 도움말 표시
• "성과" → 최근 성과 요약
• "분석" → 사용 가능한 분석 목록

더 궁금한 것이 있으시면 언제든 물어보세요! 🤖"""
        
        elif intent == QueryIntent.GENERAL_CHAT:
            return """안녕하세요! 😊

저는 **PF_bearbot**입니다! 크리에이터를 위한 **Fanding Data Report** 시스템을 도와드리는 AI 어시스턴트예요.

**🚀 무엇을 도와드릴까요?**
• 📊 **멤버십 데이터 분석**: 회원 수, 신규/이탈, 활성도 분석
• 📈 **성과 리포트**: 월간 매출, 방문자, 리텐션 분석
• 🔍 **콘텐츠 성과**: 포스트 조회수, 인기 콘텐츠 분석
• 📋 **자동 리포트**: 크리에이터 맞춤형 월간 리포트 생성

**💡 간단한 예시:**
• "8월 멤버십 성과"
• "회원 수 변화 추이"
• "인기 포스트 TOP5"
• "리텐션 현황"

구체적인 질문을 해주시면 정확한 답변을 드릴게요! 🤖"""
        
        else:
            return """안녕하세요! 👋

저는 **PF_bearbot**입니다! 크리에이터를 위한 **Fanding Data Report** 시스템을 도와드리는 AI 어시스턴트예요.

**🚀 주요 기능:**
• 📊 멤버십 데이터 분석
• 📈 성과 리포트 생성
• 🔍 콘텐츠 성과 분석
• 📋 자동 리포트 생성

구체적인 질문을 해주시면 정확한 답변을 드릴게요! 🤖"""
    
    def _normalize_query(self, query: str) -> str:
        """Normalize the user query."""
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', query.strip())
        
        # Convert to lowercase for consistency
        normalized = normalized.lower()
        
        # Handle common Korean database terms (상수에서 가져오기)
        for korean, english in KOREAN_MAPPINGS.items():
            normalized = normalized.replace(korean, english)
        
        return normalized
    
    def _extract_intent_and_entities(self, query: str, llm_intent_result: Optional[Dict] = None) -> Tuple[QueryIntent, List[Entity]]:
        """Extract intent and entities from the query."""
        
        # 1. LLM 분류 결과가 있으면 우선 사용 (MEDIUM 임계값 0.6 사용)
        # MEDIUM 임계값: LLM 분류가 상당히 확실할 때만 사용하여 오분류 방지
        if llm_intent_result and llm_intent_result.get("confidence", 0) >= LLM_CONFIDENCE_THRESHOLD_MEDIUM:
            try:
                llm_intent = QueryIntent(llm_intent_result["intent"])
                self.logger.info(f"Using LLM intent classification: {llm_intent.value} (confidence: {llm_intent_result['confidence']:.2f})")
                # 엔티티도 추출
                entities = self._extract_entities_from_query(query)
                return llm_intent, entities
            except ValueError:
                self.logger.warning(f"Invalid LLM intent: {llm_intent_result.get('intent')}")
        
        # 2. LLM 분류 결과가 있으면 참고 (LOW 임계값 0.3 사용)
        # LOW 임계값: LLM이 불확실해도 규칙 기반보다는 나을 수 있으므로 최소한의 신뢰도로 참고
        if llm_intent_result and llm_intent_result.get("confidence", 0) >= LLM_CONFIDENCE_THRESHOLD_LOW:
            try:
                llm_intent = QueryIntent(llm_intent_result["intent"])
                self.logger.info(f"Using LLM intent as fallback: {llm_intent.value} (confidence: {llm_intent_result['confidence']:.2f})")
                entities = self._extract_entities_from_query(query)
                return llm_intent, entities
            except ValueError:
                pass
        
        # 3. 데이터 조회 의도가 있는 경우 처리 (LLM 실패 시 fallback)
        if self._has_data_query_indicators(query):
            # 데이터 조회 의도가 있으면 DATA_QUERY로 분류 (LLM 실패해도)
            self.logger.info(f"Data query indicators detected, classifying as DATA_QUERY: {query}")
            entities = self._extract_entities_from_query(query)
            return QueryIntent.DATA_QUERY, entities
        
        # 4. 규칙 기반 분류 시도 (데이터 조회 의도가 없는 경우만)
        rule_based_intent = self._classify_intent_by_rules(query)
        
        if rule_based_intent != QueryIntent.UNKNOWN:
            # 규칙 기반으로 분류된 경우 (인사, 일반 대화 등)
            return rule_based_intent, []
        
        # 5. 모든 분류 실패 시 일반 대화로 분류
        return QueryIntent.GENERAL_CHAT, []
    
    def _extract_entities_from_query(self, query: str) -> List[Entity]:
        """쿼리에서 엔티티 추출"""
        entities = []
        
        # 간단한 키워드 기반 엔티티 추출
        query_lower = query.lower()
        
        # 회원 관련 키워드
        if any(keyword in query_lower for keyword in MEMBER_KEYWORDS):
            entities.append(Entity(name="member", type="table", confidence=0.9))
        
        # 크리에이터 관련 키워드
        if any(keyword in query_lower for keyword in CREATOR_KEYWORDS):
            entities.append(Entity(name="creator", type="table", confidence=0.9))
        
        # 날짜 관련 키워드
        if any(keyword in query_lower for keyword in DATE_KEYWORDS):
            entities.append(Entity(name="date", type="column", confidence=0.8))
        
        # 로그인 관련 키워드
        if any(keyword in query_lower for keyword in LOGIN_KEYWORDS):
            entities.append(Entity(name="login", type="table", confidence=0.8))
        
        # Top/순위 관련 키워드
        if any(keyword in query_lower for keyword in RANKING_KEYWORDS):
            entities.append(Entity(name="ranking", type="aggregation", confidence=0.8))
        
        # 통계 관련 키워드
        if any(keyword in query_lower for keyword in STATISTICS_KEYWORDS):
            entities.append(Entity(name="statistics", type="aggregation", confidence=0.8))
        
        return entities
    
    def _classify_intent_by_rules(self, query: str) -> QueryIntent:
        """규칙 기반 인텐트 분류 (개선된 버전)"""
        query_lower = query.lower().strip()
        
        # 1. 먼저 명확한 비데이터 의도 패턴 확인 (우선순위 높음)
        
        # 1-1. 인사말 패턴 (최우선)
        if any(pattern in query_lower for pattern in GREETING_PATTERNS):
            return QueryIntent.GREETING
        
        # 1-2. 도움말 요청 패턴 (우선순위 높음)
        if any(pattern in query_lower for pattern in HELP_REQUEST_PATTERNS):
            return QueryIntent.HELP_REQUEST
        
        # 1-3. 일반 대화 패턴 (우선순위 높음)
        if any(pattern in query_lower for pattern in GENERAL_CHAT_PATTERNS):
            return QueryIntent.GENERAL_CHAT
        
        # 2. Fanding 데이터 조회 키워드 확인 (명확한 데이터 조회 의도만)
        if any(keyword in query_lower for keyword in FANDING_DATA_KEYWORDS):
            return QueryIntent.DATA_QUERY
        
        # 3. 데이터 조회 의도 키워드 (명확한 조회 의도)
        if any(pattern in query_lower for pattern in DATA_QUERY_PATTERNS):
            return QueryIntent.DATA_QUERY
        
        # 4. 질문 패턴 (의문사 기반) - 일반 대화로 분류
        if any(pattern in query_lower for pattern in QUESTION_PATTERNS):
            # 질문이지만 데이터 조회 의도가 없으면 일반 대화로 분류
            return QueryIntent.GENERAL_CHAT
        
        # 5. 감사/인사 표현
        if any(pattern in query_lower for pattern in GRATITUDE_PATTERNS):
            return QueryIntent.GENERAL_CHAT
        
        # 6. 기본값: 일반 대화로 분류 (UNKNOWN 대신)
        return QueryIntent.GENERAL_CHAT
    
    def _has_data_query_indicators(self, query: str) -> bool:
        """데이터 조회 의도가 있는지 확인"""
        query_lower = query.lower()
        
        # 데이터 조회 키워드 (더 구체적으로 수정)
        data_keywords = [
            "조회", "검색", "데이터", "테이블", "쿼리",
            "사용자", "회원", "크리에이터", "펀딩", "프로젝트", "주문",
            "개수", "수", "합계", "평균", "최대", "최소", "통계",
            "멤버십", "성과", "매출", "방문자", "리텐션", "포스트",
            "조회수", "인기", "분석", "리포트", "월간", "일간", "주간", "년간",
            # 추가 키워드
            "뽑아줘", "뽑아", "추출", "선택", "고르", "정렬", "순위",
            "top", "top5", "top10", "상위", "최고", "많은", "적은",
            "회원수", "멤버수", "사용자수", "가입자", "활성", "신규",
            "크리에이터", "창작자", "작가", "아티스트", "제작자"
        ]
        
        # 데이터 조회와 관련된 구체적인 질문 패턴
        data_question_patterns = [
            "얼마나", "몇 개", "몇 명", "몇 건", "몇 개의", "몇 명의",
            "가져와", "찾아줘", "보여줘", "알려줘"  # 데이터 관련 맥락에서만
        ]
        
        # 일반적인 질문은 제외
        general_question_patterns = [
            "뭐야", "뭔가", "뭔지", "어떻게", "왜", "언제", "어디서",
            "할 수 있는", "할 수 있는지", "할 수 있는게", "할 수 있는것"
        ]
        
        # 일반적인 질문 패턴이 포함되어 있으면 데이터 조회가 아님
        if any(pattern in query_lower for pattern in general_question_patterns):
            return False
            
        # 데이터 조회 키워드나 구체적인 질문 패턴이 있는 경우만
        return (any(keyword in query_lower for keyword in data_keywords) or
                any(pattern in query_lower for pattern in data_question_patterns))
    
    def _extract_with_llm(self, query: str) -> Tuple[QueryIntent, List[Entity]]:
        """LLM을 사용한 인텐트 및 엔티티 추출"""
        system_prompt = """
        You are a database query analyzer. Analyze the given natural language query and extract:
        1. Query intent (SELECT, COUNT, AGGREGATE, FILTER, JOIN, UNKNOWN)
        2. Entities (tables, columns, values, conditions)
        
        Return your analysis in JSON format:
        {
            "intent": "SELECT",
            "entities": [
                {"name": "users", "type": "table", "confidence": 0.9},
                {"name": "email", "type": "column", "confidence": 0.8}
            ]
        }
        """
        
        try:
            if self.llm is None:
                # LLM이 없으면 기본 데이터 조회로 분류
                return QueryIntent.SELECT, []
            
            # 최신 LangChain 방식: SystemMessage 대신 HumanMessage에 시스템 프롬프트 포함
            messages = [
                HumanMessage(content=f"{system_prompt}\n\nAnalyze this query: {query}")
            ]
            
            response = self.llm.invoke(messages)
            result = self._parse_llm_response(response.content)
            
            intent = QueryIntent(result.get("intent", "UNKNOWN"))
            entities = [
                Entity(
                    name=entity["name"],
                    type=entity["type"],
                    confidence=entity["confidence"],
                    context=entity.get("context")
                )
                for entity in result.get("entities", [])
            ]
            
            return intent, entities
            
        except Exception as e:
            self.logger.error(f"Error extracting intent and entities: {str(e)}")
            return QueryIntent.UNKNOWN, []
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response and extract JSON."""
        try:
            import json
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"intent": "UNKNOWN", "entities": []}
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {str(e)}")
            return {"intent": "UNKNOWN", "entities": []}
    
    def _calculate_confidence(self, query: str, intent: QueryIntent, entities: List[Entity]) -> float:
        """Calculate confidence score for the processing."""
        base_confidence = 0.8
        
        # Adjust based on intent clarity
        if intent != QueryIntent.UNKNOWN:
            base_confidence += 0.1
        
        # Adjust based on entity extraction
        if entities:
            avg_entity_confidence = sum(e.confidence for e in entities) / len(entities)
            base_confidence = (base_confidence + avg_entity_confidence) / 2
        
        return min(base_confidence, 1.0)


class SchemaMapper(BaseNode):
    """Database schema mapping node."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # 캐싱된 데이터베이스 스키마 사용
        self.db_schema = get_cached_db_schema()
        
        # 테이블 이름 매핑: 일반적인 이름 -> 실제 DB 테이블명
        self.table_name_mapping = TABLE_NAME_MAPPING
    
    def process(self, state: GraphState) -> GraphState:
        """Map entities to database schema."""
        self._log_processing(state, "SchemaMapper")
        
        try:
            entities = state.get("entities", [])
            intent = state.get("intent")
            
            # Map entities to schema
            relevant_tables = self._find_relevant_tables(entities)
            relevant_columns = self._find_relevant_columns(entities, relevant_tables)
            relationships = self._find_relationships(relevant_tables)
            
            # Calculate confidence
            confidence = self._calculate_mapping_confidence(
                entities, relevant_tables, relevant_columns
            )
            
            # Create schema mapping
            schema_mapping = SchemaMapping(
                relevant_tables=relevant_tables,
                relevant_columns=relevant_columns,
                relationships=relationships,
                confidence=confidence
            )
            
            state["schema_mapping"] = schema_mapping
            state["confidence_scores"]["schema_mapping"] = confidence
            
            self.logger.info(f"Mapped to {len(relevant_tables)} tables, {len(relevant_columns)} columns")
            
        except Exception as e:
            self.logger.error(f"Error in SchemaMapper: {str(e)}")
            state["error_message"] = f"Schema mapping failed: {str(e)}"
        
        return state
    
    def _find_relevant_tables(self, entities: List[Entity]) -> List[str]:
        """Find relevant tables based on entities."""
        relevant_tables = []
        
        for entity in entities:
            if entity.type == "table":
                # Direct table mention
                table_name = self._normalize_table_name(entity.name)
                if table_name in self.db_schema:
                    relevant_tables.append(table_name)
            elif entity.type == "column":
                # Find tables containing this column
                for table_name, table_info in self.db_schema.items():
                    if entity.name in table_info.get("columns", {}):
                        relevant_tables.append(table_name)
        
        return list(set(relevant_tables))
    
    def _find_relevant_columns(self, entities: List[Entity], tables: List[str]) -> List[str]:
        """Find relevant columns based on entities and tables."""
        relevant_columns = []
        
        for entity in entities:
            if entity.type == "column":
                relevant_columns.append(entity.name)
        
        # Add columns from relevant tables
        for table in tables:
            if table in self.db_schema:
                table_columns = list(self.db_schema[table].get("columns", {}).keys())
                relevant_columns.extend(table_columns)
        
        return list(set(relevant_columns))
    
    def _find_relationships(self, tables: List[str]) -> List[Dict[str, str]]:
        """Find relationships between tables."""
        relationships = []
        
        # Simple relationship detection based on common patterns
        for table1 in tables:
            for table2 in tables:
                if table1 != table2:
                    # Check for foreign key relationships
                    if self._has_foreign_key_relationship(table1, table2):
                        relationships.append({
                            "from_table": table1,
                            "to_table": table2,
                            "type": "foreign_key"
                        })
        
        return relationships
    
    def _has_foreign_key_relationship(self, table1: str, table2: str) -> bool:
        """Check if there's a foreign key relationship between tables."""
        if table1 not in self.db_schema or table2 not in self.db_schema:
            return False
        
        # Simple heuristic: check if table1 has a column that references table2
        table1_columns = self.db_schema[table1].get("columns", {})
        for column_name, column_info in table1_columns.items():
            if column_info.get("type") == "foreign_key" and table2 in str(column_info):
                return True
        
        return False
    
    def _normalize_table_name(self, name: str) -> str:
        """Normalize table name to match schema."""
        # Remove common prefixes/suffixes
        name = name.lower().strip()
        
        # 먼저 매핑 테이블에서 확인
        if name in self.table_name_mapping:
            return self.table_name_mapping[name]
        
        # Handle common variations
        if name.endswith('s'):
            singular_name = name[:-1]
            if singular_name in self.table_name_mapping:
                return self.table_name_mapping[singular_name]
        
        # Check if it matches any table in the schema
        for table_name in self.db_schema.keys():
            if name in table_name.lower() or table_name.lower() in name:
                return table_name
        
        return name
    
    
    def _calculate_mapping_confidence(self, entities: List[Entity], tables: List[str], columns: List[str]) -> float:
        """Calculate confidence for schema mapping."""
        if not entities:
            return 0.0
        
        # Base confidence from entity extraction
        avg_entity_confidence = sum(e.confidence for e in entities) / len(entities)
        
        # Bonus for finding relevant tables/columns
        mapping_bonus = 0.0
        if tables:
            mapping_bonus += 0.2
        if columns:
            mapping_bonus += 0.1
        
        return min(avg_entity_confidence + mapping_bonus, 1.0)


class SQLGenerationNode(BaseNode):
    """SQL 생성 에이전트 노드 - 자연어 쿼리를 SQL로 변환"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.db_schema = config.get("db_schema", {})
    
        # GeminiSQLGenerator 초기화
        self.sql_generator = self._initialize_sql_generator()
        
        # SQLPromptTemplate 초기화
        self.prompt_template = SQLPromptTemplate(db_schema=self.db_schema)
        
        # FandingSQLTemplates 초기화
        from .fanding_sql_templates import FandingSQLTemplates
        self.fanding_templates = FandingSQLTemplates()
        
    def _initialize_sql_generator(self):
        """SQL 생성기 초기화"""
        try:
            # 환경 변수에서 직접 API 키 가져오기
            import os
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("GOOGLE_API_KEY")
            
            return GeminiSQLGenerator(api_key=api_key)
        except Exception as e:
            self.logger.error(f"Failed to initialize SQL generator: {e}")
            return None
    
    def process(self, state: GraphState) -> GraphState:
        """자연어 쿼리를 SQL로 변환"""
        self._log_processing(state, "SQLGenerationNode")
        
        try:
            # 일반 대화인 경우 SQL 생성 건너뛰기
            skip_flag = state.get("skip_sql_generation", False)
            conversation_response = state.get("conversation_response")
            intent = state.get("intent")
            
            self.logger.info(f"SQLGenerationNode - skip_sql_generation: {skip_flag}")
            self.logger.info(f"SQLGenerationNode - conversation_response: {conversation_response is not None}")
            self.logger.info(f"SQLGenerationNode - intent: {intent}")
            
            # 대화 인텐트인 경우 SQL 생성 건너뛰기
            if (skip_flag or conversation_response or 
                intent in ["GREETING", "GENERAL_CHAT", "HELP_REQUEST"]):
                # 애매한 쿼리로 인한 명확화 질문인지 확인
                if state.get("conversation_response") and "어떤" in str(state.get("conversation_response", "")):
                    self.logger.info("Skipping SQL generation - clarification question for ambiguous query")
                else:
                    self.logger.info("Skipping SQL generation for conversation intent")
                state["sql_query"] = None
                state["validated_sql"] = None
                state["confidence_scores"]["sql_generation"] = 1.0
                return state
            
            user_query = state["user_query"]
            schema_mapping = state.get("schema_mapping")
            
            # 누적 슬롯 병합 (이전 state + 현재 질의)
            prior_slots = state.get("slots") or {}
            new_slots = self._extract_simple_slots(user_query)
            slots = {**prior_slots, **{k: v for k, v in new_slots.items() if v}}
            state["slots"] = slots
            
            # 스키마 매핑 정보를 SQL 생성기에 설정
            if schema_mapping and self.sql_generator:
                # 관련 테이블 정보를 스키마에 추가
                relevant_schema = {}
                for table_name in schema_mapping.relevant_tables:
                    if table_name in self.db_schema:
                        relevant_schema[table_name] = self.db_schema[table_name]
                
                self.sql_generator.set_schema(relevant_schema)
            
            # 1. 동적 SQL 생성 시도 (월별 쿼리 등)
            dynamic_sql_result = state.get("dynamic_sql_result")
            # SQL 생성 임계값(0.7) 사용: 동적 SQL 생성은 높은 정확도가 필요하므로 높은 신뢰도 요구
            if dynamic_sql_result and dynamic_sql_result.get("confidence", 0) >= SQL_GENERATION_CONFIDENCE_THRESHOLD:
                self.logger.info("Using dynamic SQL generation result")
                set_sql_result(state, dynamic_sql_result["sql_query"], dynamic_sql_result["confidence"])
                return state
            
            # 2. RAG 매핑 결과 확인 (최우선) - 신뢰도 임계값 낮춤
            rag_result = state.get("rag_mapping_result")
            # RAG 임계값(0.6) 사용: RAG 매핑 결과는 중간 수준의 신뢰도로도 사용 가능
            if rag_result and rag_result.confidence > RAG_CONFIDENCE_THRESHOLD:
                self.logger.info(f"Using RAG mapping result: {rag_result.source.value} (confidence: {rag_result.confidence:.2f})")
                set_sql_result(state, rag_result.sql_template, rag_result.confidence)
                return state
            
            # 2.5 슬롯 기반 결정적 빌드 (누적 슬롯 사용)
            slots = state.get("slots") or {}
            # intent 추론 보강: metric이 active_members이고 creator/top_k/월이 존재하면 active_members용 intent
            if (slots.get("group_by") == "creator" or ("크리에이터" in user_query)) and slots.get("top_k") and slots.get("month"):
                metric = slots.get("metric") or ("active_members" if ("활성" in user_query) else "new_members")
                # 누적 반영
                slots["metric"] = metric
                state["slots"] = slots
                month = slots.get("month")
                k = int(slots.get("top_k", 5))
                creator_col = self._guess_creator_column()
                if creator_col:
                    if metric == "active_members":
                        sql = (
                            "SELECT {creator_col}, COUNT(*) AS active_members "
                            "FROM t_member WHERE status = 'A' "
                            "GROUP BY {creator_col} ORDER BY active_members DESC LIMIT {k}"
                        ).format(creator_col=creator_col, k=k)
                    else:
                        sql = (
                            "SELECT {creator_col}, COUNT(DISTINCT member_no) AS new_members "
                            "FROM t_member_login_log "
                            "WHERE DATE_FORMAT(ins_datetime, '%Y-%m') = '{month}' "
                            "GROUP BY {creator_col} ORDER BY new_members DESC LIMIT {k}"
                        ).format(creator_col=creator_col, month=month, k=k)
                    state["sql_query"] = sql
                    state["confidence_scores"]["sql_generation"] = 0.8
                    self.logger.info("Built deterministic SQL using accumulated slots")
                    return state
                else:
                    clarification = (
                        "크리에이터 식별 컬럼을 확인할 수 없습니다. 어떤 컬럼으로 그룹핑할까요? 예: creator_id/creator_no"
                    )
                    state["clarification_question"] = clarification
                    state["conversation_response"] = True
                    state["conversation_text"] = clarification
                    state["confidence_scores"]["sql_generation"] = 0.0
                    return state
            
            # 3. 이미 SQL이 설정되어 있는지 확인 (Fanding 템플릿 등)
            existing_sql = state.get("sql_query")
            sql_validation_failed = state.get("sql_validation_failed", False)
            
            if existing_sql and not sql_validation_failed:
                self.logger.info(f"SQL already exists, skipping generation: {existing_sql[:100]}...")
                state["confidence_scores"]["sql_generation"] = 1.0
                return state
            elif existing_sql and sql_validation_failed:
                self.logger.info(f"Previous SQL validation failed, generating new SQL...")
                # SQL 검증 실패 시 새로운 SQL 생성
                state["sql_query"] = None
                state["sql_validation_failed"] = False
            
            # SQL 생성
            if self.sql_generator:
                result = self.sql_generator.generate_sql(user_query)
                
                if result["success"]:
                    state["sql_query"] = result["sql"]
                    state["sql_generation_metadata"] = {
                        "model": result.get("model"),
                        "prompt_length": result.get("prompt_length"),
                        "response_length": result.get("response_length"),
                        "mock": result.get("mock", False)
                    }
                    
                    # 신뢰도 점수 계산
                    confidence = self._calculate_sql_confidence(result, schema_mapping)
                    state["confidence_scores"]["sql_generation"] = confidence
            
                    self.logger.info(f"Generated SQL: {result['sql']}")
                else:
                    # SQL 생성 실패 시 Fanding 템플릿 시도
                    self.logger.warning(f"SQL generation failed: {result.get('error', 'Unknown error')}")
                    self.logger.info("Attempting Fanding template fallback...")
                    
                    fanding_template = self.fanding_templates.match_query_to_template(user_query)
                    if fanding_template:
                        state["sql_query"] = fanding_template.sql_template
                        state["fanding_template"] = fanding_template
                        state["confidence_scores"]["sql_generation"] = 0.8  # 템플릿 사용 시 중간 신뢰도
                        self.logger.info(f"Fanding template fallback successful: {fanding_template.name}")
                    else:
                        # DATA_QUERY인데 생성 실패 시: 명확화 질문 요청
                        clarification = self._build_clarification_question(user_query)
                        state["clarification_question"] = clarification
                        state["conversation_response"] = clarification  # 문자열로 설정
                        state["conversation_text"] = clarification
                        state["confidence_scores"]["sql_generation"] = 0.0
                        self.logger.info("Asking clarification instead of switching to generic conversation")
            else:
                # SQL 생성기 없음: 명확화 질문 요청
                clarification = self._build_clarification_question(user_query)
                state["clarification_question"] = clarification
                state["conversation_response"] = clarification  # 문자열로 설정
                state["conversation_text"] = clarification
                state["confidence_scores"]["sql_generation"] = 0.0
                self.logger.warning("No SQL generator available, asking for clarification")
            
        except Exception as e:
            self.logger.error(f"Error in SQLGenerationNode: {str(e)}")
            state["error_message"] = f"SQL generation failed: {str(e)}"
            state["confidence_scores"]["sql_generation"] = 0.0
        
        return state
    
    def _build_clarification_question(self, user_query: str) -> str:
        """간단한 명확화 질문 생성 (기간/Top-K/지표 우선)"""
        return generate_clarification_question(user_query)
    
    def _calculate_sql_confidence(self, result: Dict[str, Any], schema_mapping) -> float:
        """SQL 생성 신뢰도 계산"""
        base_confidence = 0.8
        
        # 모의 SQL인 경우 신뢰도 감소
        if result.get("mock", False):
            base_confidence *= 0.7
        
        # 스키마 매핑이 있는 경우 보너스
        if schema_mapping and schema_mapping.relevant_tables:
            base_confidence += 0.1
        
        # 응답 길이에 따른 조정
        response_length = result.get("response_length", 0)
        if response_length > 100:
            base_confidence += 0.05
        
        return min(base_confidence, 1.0)
    
    def _generate_fallback_sql(self, query: str) -> str:
        """Fallback SQL 생성 (API 사용 불가 시)"""
        query_lower = query.lower()
        
        if "회원" in query_lower or "사용자" in query_lower:
            if "수" in query_lower or "개수" in query_lower:
                return "SELECT COUNT(*) FROM t_member;"
            else:
                return "SELECT * FROM t_member LIMIT 100;"
        elif "크리에이터" in query_lower:
            return "SELECT nickname, description FROM t_creator LIMIT 100;"
        elif "펀딩" in query_lower or "프로젝트" in query_lower:
            return "SELECT title, goal_amount, current_amount FROM t_funding LIMIT 100;"
        else:
            return "SELECT 1 as placeholder;"
    
    def _extract_simple_slots(self, query: str) -> Dict[str, Any]:
        """간단 슬롯 추출: month, top_k, intent(creator_topk_new_members)"""
        q = query.lower()
        from .date_utils import DateUtils
        month = DateUtils.get_analysis_month(query)
        # top-k
        top_k = 5
        m = re.search(r"top\s*(\d+)|상위\s*(\d+)", q)
        if m:
            top_k = int([g for g in m.groups() if g][0])
        # intent
        intent = None
        if ("크리에이터" in q or "creator" in q) and ("top" in q or "상위" in q) and ("신규" in q or "회원" in q):
            intent = "creator_topk_new_members"
        return {"month": month, "top_k": top_k, "intent": intent}
    
    def _guess_creator_column(self) -> Optional[str]:
        """db_schema에서 가능한 크리에이터 식별 컬럼 추정"""
        # 로그인 로그 테이블 기준으로 탐색
        table = self.db_schema.get("t_member_login_log") or {}
        candidates = ["creator_id", "creator_no", "creator", "channel_id", "influencer_id"]
        for c in candidates:
            if c in table:
                return c
        return None


class SQLValidationNode(BaseNode):
    """SQL 검증 에이전트 노드 - 생성된 SQL의 구문 및 의미 검증"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # 캐싱된 데이터베이스 스키마 사용
        self.db_schema = get_cached_db_schema()
    
        
    def process(self, state: GraphState) -> GraphState:
        """SQL 쿼리 검증"""
        self._log_processing(state, "SQLValidationNode")
        
        try:
            # 대화 응답이 있는 경우 검증 건너뛰기
            conversation_response = state.get("conversation_response")
            intent = state.get("intent")
            
            self.logger.info(f"SQLValidationNode - conversation_response: {conversation_response is not None}")
            self.logger.info(f"SQLValidationNode - intent: {intent}")
            
            if (conversation_response or 
                intent in ["GREETING", "GENERAL_CHAT", "HELP_REQUEST"]):
                self.logger.info("Skipping SQL validation for conversation response")
                state["validation_result"] = {"is_valid": True, "message": "Conversation response - validation skipped"}
                state["is_valid"] = True
                return state
            
            sql_query = state.get("sql_query")
            if not sql_query:
                state["error_message"] = "No SQL query to validate"
                return state
            
            # 기본 구문 검증 (새로운 SQL 파서 사용)
            syntax_validation = validate_sql_syntax(sql_query)
            
            # 스키마 일관성 검증 (실제 검증 수행)
            schema_validation = self._validate_schema_compatibility(sql_query)
            
            # 보안 검증
            security_validation = self._validate_security(sql_query)
            
            # 스키마 불일치가 있으면 자동 수정 시도
            corrected_sql = sql_query
            if not schema_validation["is_valid"] and "corrections" in schema_validation:
                corrected_sql = self._apply_schema_corrections(sql_query, schema_validation["corrections"])
                if corrected_sql != sql_query:
                    self.logger.info(f"SQL auto-corrected: {sql_query[:100]}... -> {corrected_sql[:100]}...")
                    state["sql_query"] = corrected_sql
                    state["sql_corrected"] = True
            
            # 종합 검증 결과
            is_valid = all([
                syntax_validation["is_valid"],
                schema_validation["is_valid"],
                security_validation["is_valid"]
            ])
            
            validation_result = {
                "is_valid": is_valid,
                "syntax": syntax_validation,
                "schema": schema_validation,
                "security": security_validation,
                "suggestions": self._generate_suggestions(sql_query, [
                    syntax_validation,
                    schema_validation,
                    security_validation
                ])
            }
            
            state["sql_validation"] = validation_result
            state["validation_result"] = validation_result
            
            # 신뢰도 계산
            confidence = self._calculate_validation_confidence(validation_result)
            state["confidence_scores"]["sql_validation"] = confidence
            
            if is_valid:
                state["validated_sql"] = sql_query
                state["sql_validation_failed"] = False
                self.logger.info("SQL validation passed")
            else:
                state["sql_validation_failed"] = True
                self.logger.warning(f"SQL validation failed: {validation_result['suggestions']}")
            
        except Exception as e:
            self.logger.error(f"Error in SQLValidationNode: {str(e)}")
            state["error_message"] = f"SQL validation failed: {str(e)}"
            state["confidence_scores"]["sql_validation"] = 0.0
        
        return state
    
    
    def _validate_schema_compatibility(self, sql_query: str) -> Dict[str, Any]:
        """스키마 호환성 검증 및 자동 수정"""
        try:
            issues = []
            corrections = []
            
            # 실제 DB 스키마 확인 (캐싱된 스키마 사용)
            actual_schema = self.db_schema
            
            # 테이블명 검증 (새로운 SQL 파서 사용)
            table_names = extract_table_names(sql_query)
            for table_name in table_names:
                if table_name not in actual_schema:
                    # 유사한 테이블명 찾기
                    similar_table = self._find_similar_table(table_name, actual_schema)
                    if similar_table:
                        issues.append(f"Table '{table_name}' not found, did you mean '{similar_table}'?")
                        corrections.append(f"Replace '{table_name}' with '{similar_table}'")
                    else:
                        issues.append(f"Table '{table_name}' not found in schema")
            
            # 컬럼명 검증 (특히 ins_datetime 문제)
            if 'ins_datetime' in sql_query:
                # t_member_login_log 테이블을 사용하는 경우 ins_datetime 컬럼이 올바름
                if 't_member_login_log' in sql_query:
                    # t_member_login_log 테이블에 ins_datetime 컬럼이 있는지 확인
                    login_log_table = actual_schema.get('t_member_login_log', {})
                    login_log_columns = login_log_table.get('columns', {})
                    if 'ins_datetime' not in login_log_columns:
                        issues.append("Column 'ins_datetime' not found in t_member_login_log table")
                    # t_member_login_log를 사용하는 경우는 유효함
                elif 't_member_info' in sql_query:
                    # t_member_info 테이블에 ins_datetime 컬럼이 있는지 확인
                    member_info_table = actual_schema.get('t_member_info', {})
                    member_info_columns = member_info_table.get('columns', {})
                    if 'ins_datetime' not in member_info_columns:
                        issues.append("Column 'ins_datetime' not found in t_member_info table")
                        corrections.append("Verify t_member_info table schema")
                elif 't_member' in sql_query:
                    # t_member 테이블에 ins_datetime 컬럼이 있는지 확인 (t_member는 ins_datetime이 없음)
                    member_table = actual_schema.get('t_member', {})
                    member_columns = member_table.get('columns', {})
                    if 'ins_datetime' not in member_columns:
                        issues.append("Column 'ins_datetime' not found in t_member table")
                        corrections.append("Use t_member_info table instead of t_member for ins_datetime column")
            
            return {
                "is_valid": len(issues) == 0,
                "issues": issues,
                "corrections": corrections,
                "details": "Schema validation completed"
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "error": "Schema validation error",
                "details": str(e)
            }
    
    
    def _find_similar_table(self, table_name: str, schema: Dict[str, Any]) -> Optional[str]:
        """유사한 테이블명 찾기"""
        table_name_lower = table_name.lower()
        
        # 정확한 매칭
        if table_name in schema:
            return table_name
        
        # 부분 매칭
        for actual_table in schema.keys():
            if table_name_lower in actual_table.lower() or actual_table.lower() in table_name_lower:
                return actual_table
        
        return None
    
    def _apply_schema_corrections(self, sql_query: str, corrections: List[str]) -> str:
        """스키마 수정사항을 SQL에 적용"""
        corrected_sql = sql_query
        
        for correction in corrections:
            if "Replace 'ins_datetime' with" in correction:
                # ins_datetime을 대체 컬럼으로 교체
                alt_col = correction.split("'")[-2]  # 마지막에서 두 번째 따옴표 안의 값
                corrected_sql = re.sub(r'\bins_datetime\b', alt_col, corrected_sql, flags=re.IGNORECASE)
                self.logger.info(f"Replaced 'ins_datetime' with '{alt_col}'")
        
        return corrected_sql
    
    def _validate_security(self, sql_query: str) -> Dict[str, Any]:
        """SQL 보안 검증"""
        try:
            issues = []
            sql_upper = sql_query.upper()
            
            # 위험한 키워드 확인 (단어 경계 고려) - 상수에서 가져오기
            for keyword in DANGEROUS_SQL_KEYWORDS:
                # 단어 경계를 고려한 정확한 매칭
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, sql_upper):
                    issues.append(f"Dangerous keyword detected: {keyword}")
            
            # 주석 확인 (SQL 인젝션 방지) - 템플릿의 정당한 주석은 허용
            # 멀티라인 주석 /* */만 차단 (단일 라인 주석 -- 는 허용)
            if '/*' in sql_query or '*/' in sql_query:
                issues.append("Suspicious multi-line comment detected")
            
            # 단일 라인 주석(--)은 허용 (템플릿에서 정당하게 사용됨)
            
            return {
                "is_valid": len(issues) == 0,
                "issues": issues,
                "details": "Security validation completed"
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "error": "Security validation error",
                "details": str(e)
            }
    
    
    def _generate_suggestions(self, sql_query: str, validations: List[Dict[str, Any]]) -> List[str]:
        """검증 결과를 바탕으로 수정 제안 생성"""
        suggestions = []
        
        for validation in validations:
            if not validation["is_valid"]:
                if "issues" in validation:
                    suggestions.extend(validation["issues"])
                elif "error" in validation:
                    suggestions.append(validation["error"])
        
        return suggestions
    
    def _calculate_validation_confidence(self, validation_result: Dict[str, Any]) -> float:
        """검증 신뢰도 계산"""
        if validation_result["is_valid"]:
            return 1.0
        
        # 각 검증 항목의 가중치
        weights = {
            "syntax": 0.4,
            "schema": 0.4,
            "security": 0.2
        }
        
        confidence = 0.0
        for validation_type, weight in weights.items():
            if validation_result[validation_type]["is_valid"]:
                confidence += weight
        
        return confidence


class DataSummarizationNode(BaseNode):
    """데이터 요약 에이전트 노드 - SQL 실행 결과를 자연어로 요약"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm = self._initialize_llm()
        # Removed: DataInsightAnalyzer (deleted module)
        # self.insight_analyzer = DataInsightAnalyzer(config)
        
    def _initialize_llm(self):
        """LLM 초기화"""
        try:
            # 환경 변수에서 직접 API 키 가져오기
            import os
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("GOOGLE_API_KEY")
            
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                google_api_key=api_key,
                temperature=0.3,
                max_output_tokens=1024,
                request_timeout=10.0  # 10초 타임아웃 설정
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM: {str(e)}")
            return None
    
    def process(self, state: GraphState) -> GraphState:
        """SQL 실행 결과를 자연어로 요약"""
        self._log_processing(state, "DataSummarizationNode")
        
        try:
            # 대화 응답이 있는 경우 요약 건너뛰기
            conversation_response = state.get("conversation_response")
            intent = state.get("intent")
            fanding_template = state.get("fanding_template")
            
            self.logger.info(f"DataSummarizationNode - conversation_response: {conversation_response is not None}")
            self.logger.info(f"DataSummarizationNode - intent: {intent}")
            self.logger.info(f"DataSummarizationNode - fanding_template: {fanding_template is not None}")
            
            if (conversation_response or 
                intent in ["GREETING", "GENERAL_CHAT", "HELP_REQUEST"]):
                self.logger.info("Skipping data summarization for conversation response")
                state["data_summary"] = conversation_response or "대화 응답이 처리되었습니다."
                state["success"] = True
                return state
            
            # Fanding 템플릿이 있는 경우 특별 처리
            if fanding_template:
                query_result = state.get("query_result")
                if query_result:
                    # Fanding 템플릿 결과 포맷팅
                    from .fanding_sql_templates import FandingSQLTemplates
                    templates = FandingSQLTemplates()
                    formatted_result = templates.format_sql_result(fanding_template, query_result)
                    state["data_summary"] = formatted_result
                    state["success"] = True
                    self.logger.info(f"🎯 Fanding template result formatted: {fanding_template.name}")
                    return state
            
            query_result = state.get("query_result")
            user_query = state.get("user_query")
            
            if not query_result:
                state["error_message"] = "No query result to summarize"
                return state
            
            # 결과 데이터 분석
            result_stats = self._analyze_results(query_result)
            
            # 인사이트 분석 수행 - DISABLED: DataInsightAnalyzer deleted
            # try:
            #     sql_query = state.get("sql_query", "")
            #     insight_report = self.insight_analyzer.analyze_data(user_query, query_result, sql_query)
            #     
            #     # 인사이트 리포트를 상태에 저장
            #     state["insight_report"] = insight_report
            #     state["business_insights"] = insight_report.insights
            #     state["insight_summary"] = insight_report.summary
            #     
            #     # 인사이트가 있는 경우 요약에 포함
            #     if insight_report.insights:
            #         insight_text = self.insight_analyzer.format_insight_report(insight_report)
            #         state["insight_report_formatted"] = insight_text
            #     self.logger.info(f"Generated {len(insight_report.insights)} business insights")
            #     
            # except Exception as e:
            #     self.logger.warning(f"Insight analysis failed: {e}")
            #     # 인사이트 분석 실패해도 기본 요약은 계속 진행
            #     state["insight_report"] = None
            #     state["business_insights"] = []
            
            # Set default values since insight analyzer is disabled
            state["insight_report"] = None
            state["business_insights"] = []
            
            # 요약 생성
            if self.llm:
                summary = self._generate_ai_summary(user_query, query_result, result_stats)
            else:
                summary = self._generate_fallback_summary(query_result, result_stats)
            
            state["data_summary"] = summary
            state["result_statistics"] = result_stats
            
            # 신뢰도 계산
            confidence = self._calculate_summary_confidence(summary, result_stats)
            state["confidence_scores"]["data_summarization"] = confidence
            
            self.logger.info(f"Generated summary: {summary[:100]}...")
            
        except Exception as e:
            self.logger.error(f"Error in DataSummarizationNode: {str(e)}")
            state["error_message"] = f"Data summarization failed: {str(e)}"
            state["confidence_scores"]["data_summarization"] = 0.0
        
        return state
    
    def _analyze_results(self, query_result: List[Dict[str, Any]]) -> Dict[str, Any]:
        """결과 데이터 통계 분석"""
        # NoneType 에러 방지
        if not query_result or query_result is None:
            return {"row_count": 0, "columns": [], "data_types": {}}
        
        stats = {
            "row_count": len(query_result),
            "columns": list(query_result[0].keys()) if query_result else [],
            "data_types": {},
            "sample_values": {},
            "null_counts": {}
        }
        
        if query_result:
            # 데이터 타입 분석
            for column in stats["columns"]:
                sample_values = [row.get(column) for row in query_result[:5]]
                stats["sample_values"][column] = sample_values
                
                # NULL 값 개수
                null_count = sum(1 for row in query_result if row.get(column) is None)
                stats["null_counts"][column] = null_count
                
                # 데이터 타입 추론
                non_null_values = [v for v in sample_values if v is not None]
                if non_null_values:
                    first_value = non_null_values[0]
                    if isinstance(first_value, int):
                        stats["data_types"][column] = "integer"
                    elif isinstance(first_value, float):
                        stats["data_types"][column] = "float"
                    elif isinstance(first_value, str):
                        stats["data_types"][column] = "string"
                    else:
                        stats["data_types"][column] = "unknown"
        
        return stats
    
    def _generate_ai_summary(self, user_query: str, query_result: List[Dict[str, Any]], stats: Dict[str, Any]) -> str:
        """AI를 사용한 요약 생성"""
        try:
            # 결과 데이터 포맷팅
            formatted_results = self._format_results(query_result)
            
            # 요약 프롬프트 생성
            summary_prompt = f"""
다음 데이터베이스 쿼리 결과를 분석하여 사용자 친화적인 요약을 생성해주세요.

원본 질문: {user_query}

쿼리 결과 통계:
- 총 행 수: {stats['row_count']}
- 컬럼 수: {len(stats['columns'])}
- 컬럼명: {', '.join(stats['columns'])}

샘플 데이터:
{formatted_results[:500]}...

요구사항:
1. 결과의 주요 내용을 간결하게 설명
2. 데이터의 규모와 특징을 언급
3. 사용자가 이해하기 쉬운 언어 사용
4. 3-5문장으로 요약
5. 한국어로 작성

요약:
"""
            
            # 최신 LangChain 방식: SystemMessage 대신 HumanMessage에 시스템 프롬프트 포함
            messages = [
                HumanMessage(content=f"당신은 데이터 분석 전문가입니다. 쿼리 결과를 사용자 친화적으로 요약해주세요.\n\n{summary_prompt}")
            ]
            
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            self.logger.error(f"AI summary generation failed: {e}")
            return self._generate_fallback_summary(query_result, stats)
    
    def _generate_fallback_summary(self, query_result: List[Dict[str, Any]], stats: Dict[str, Any]) -> str:
        """Fallback 요약 생성"""
        row_count = stats.get("row_count", 0)
        columns = stats.get("columns", [])
        
        if row_count == 0:
            return "쿼리 결과가 없습니다."
        elif row_count == 1:
            return f"총 1개의 결과가 조회되었습니다. 컬럼: {', '.join(columns)}"
        else:
            return f"총 {row_count}개의 결과가 조회되었습니다. 컬럼: {', '.join(columns)}"
    
    def _format_results(self, query_result: List[Dict[str, Any]], max_rows: int = 10) -> str:
        """결과 데이터를 포맷팅"""
        # NoneType 에러 방지
        if not query_result or query_result is None:
            return "결과 없음"
        
        formatted_rows = []
        for i, row in enumerate(query_result[:max_rows]):
            row_str = f"행 {i+1}: {dict(row)}"
            formatted_rows.append(row_str)
        
        result = "\n".join(formatted_rows)
        
        if len(query_result) > max_rows:
            result += f"\n... 및 {len(query_result) - max_rows}개 행 더"
        
        return result
    
    def _calculate_summary_confidence(self, summary: str, stats: Dict[str, Any]) -> float:
        """요약 신뢰도 계산"""
        base_confidence = 0.8
        
        # 요약 길이에 따른 조정
        if len(summary) > 50:
            base_confidence += 0.1
        
        # 통계 정보 활용도에 따른 조정
        if stats.get("row_count", 0) > 0:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)

