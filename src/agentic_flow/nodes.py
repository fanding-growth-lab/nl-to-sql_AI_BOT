"""
LangGraph Node Components for NL-to-SQL Pipeline

This module implements the individual nodes that make up the LangGraph pipeline.
"""

import re
import logging
import sqlparse
import random
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from .prompts import GeminiSQLGenerator, SQLPromptTemplate
from .fanding_sql_templates import FandingSQLTemplates

# Removed unused imports: enhanced_rag_mapper, data_insight_analyzer, dynamic_schema_expander

from .state import (
    GraphState,
    Entity,
    SchemaMapping,
    SQLResult,
    QueryIntent,
    QueryComplexity,
    set_sql_result,
    set_rag_mapping_result,
    set_dynamic_pattern,
    set_fanding_template,
    set_conversation_response,
    clear_sql_generation,
    get_effective_sql,
    is_sql_generation_skipped,
)
from core.config import get_settings
from core.db import (
    get_db_session,
    get_cached_db_schema,
    extract_table_names,
    extract_column_names,
    validate_sql_syntax,
)
from core.logging import get_logger

logger = get_logger(__name__)


# Constants and Configuration for Agentic Flow
# Intent Classification Patterns
GREETING_PATTERNS = [
    "안녕",
    "반가워",
    "hello",
    "hi",
    "좋은 아침",
    "좋은 저녁",
    "환영",
    "인사",
    "만나서 반가워",
    "반갑습니다",
    "안녕하세요",
    "안녕하세요",
    "반갑습니다",
    "처음 뵙겠습니다",
    "만나서 반갑습니다",
    "좋은 하루",
    "좋은 하루 되세요",
    "좋은 하루 보내세요",
]

HELP_REQUEST_PATTERNS = [
    "도움",
    "사용법",
    "어떻게",
    "help",
    "명령어",
    "도와줘",
    "설명",
    "가이드",
    "사용법",
    "도움말",
    "사용법 알려줘",
    "어떻게 사용하나요",
    "기능",
    "기능이 뭐야",
    "뭐가 있어",
    "뭘 할 수 있어",
    "할 수 있는 것",
    "기능 설명",
    "너가 할 수 있는 일",
    "뭐야",
    "뭐지",
    "뭔가",
    "뭔데",
]

GENERAL_CHAT_PATTERNS = [
    "어때",
    "어떠",
    "좋아",
    "나쁘",
    "재미",
    "재미있",
    "지루",
    "피곤",
    "날씨",
    "오늘",
    "어제",
    "내일",
    "주말",
    "휴일",
    "일",
    "일정",
    "고마워",
    "감사",
    "미안",
    "죄송",
    "괜찮",
    "괜찮아",
    "괜찮습니다",
    "뭐야",
    "뭐지",
    "뭔가",
    "뭔데",
    "뭔가요",
    "뭔가요?",
]

FANDING_DATA_KEYWORDS = [
    "멤버십",
    "성과",
    "회원",
    "매출",
    "방문자",
    "리텐션",
    "포스트",
    "조회수",
    "인기",
    "분석",
    "통계",
    "리포트",
    "월간",
    "일간",
    "주간",
    "년간",
    "크리에이터",
    "펀딩",
    "프로젝트",
    "8월",
    "9월",
    "10월",
    "11월",
    "12월",
    "1월",
    "2월",
    "3월",
    "4월",
    "5월",
    "6월",
    "7월",
    "올해",
    "작년",
    "지난달",
    "이번달",
    "신규",
    "이탈",
    "활성",
    "구독",
    "결제",
    "수익",
    "매출액",
    "현황",
    "상황",
    "결과",
    "성과분석",
    "성과",
    "분석해줘",
    "보고서",
    "요약",
    "정리",
    "현재",
    "최근",
    "지금",
    "오늘",
    "어제",
    "내일",
]

DATA_QUERY_PATTERNS = [
    "조회",
    "검색",
    "보여줘",
    "찾아",
    "테이블",
    "쿼리",
    "개수",
    "수",
    "합계",
    "평균",
    "최대",
    "최소",
    "통계",
    "알려줘",
    "보여줘",
    "찾아줘",
    "가져와",
    "얼마나",
    "몇 개",
    "몇 명",
    "얼마",
    "어느 정도",
]

QUESTION_PATTERNS = [
    "뭐",
    "무엇",
    "어떤",
    "어디",
    "언제",
    "왜",
    "어떻게",
    "누구",
    "뭔가",
    "뭔지",
    "뭔데",
    "뭐야",
    "뭐지",
    "뭔가요",
    "뭔가요?",
]

GRATITUDE_PATTERNS = [
    "고마워",
    "감사",
    "감사합니다",
    "고마워요",
    "고맙습니다",
    "수고",
    "수고하셨",
    "수고하셨어요",
    "수고하셨습니다",
]

# Response Templates
GREETING_RESPONSES = [
    "안녕하세요! 👋 Fanding Data Report 봇입니다. 무엇을 도와드릴까요?",
    "안녕하세요! 😊 데이터 분석을 도와드리겠습니다.",
    "반갑습니다! 🤖 멤버십 성과나 회원 데이터를 조회해드릴 수 있어요.",
    "안녕하세요! 📊 Fanding 데이터를 분석해드리겠습니다.",
]

GENERAL_CHAT_RESPONSES = [
    "안녕하세요! 😊 데이터 분석에 대해 궁금한 것이 있으시면 언제든 말씀해주세요!",
    "네, 듣고 있어요! 📊 Fanding 데이터를 조회하고 싶으시면 말씀해주세요.",
    "좋은 하루 보내세요! 🤖 멤버십 성과나 회원 데이터가 궁금하시면 언제든 물어보세요.",
    "감사합니다! 😊 데이터 분석을 도와드릴 준비가 되어있어요.",
]

# SQL Security Keywords
DANGEROUS_SQL_KEYWORDS = [
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "CREATE",
    "ALTER",
    "TRUNCATE",
    "EXEC",
    "EXECUTE",
    "UNION",
    "SCRIPT",
    "GRANT",
    "REVOKE",
    "COMMIT",
    "ROLLBACK",
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
    "보여줘": "show",
    "찾아줘": "find",
    "가져와": "get",
    "개수": "count",
    "합계": "sum",
    "평균": "average",
    "최대": "max",
    "최소": "min",
}

# Entity Extraction Keywords
MEMBER_KEYWORDS = [
    "회원",
    "멤버",
    "사용자",
    "유저",
    "member",
    "user",
    "회원수",
    "멤버수",
]
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
    needs_topk = "top" in q or "상위" in q or "top5" in q
    needs_period = any(
        k in q
        for k in [
            "이번",
            "지난",
            "이번달",
            "지난달",
            "월",
            "분기",
            "주",
            "week",
            "month",
            "quarter",
        ]
    )
    needs_metric = any(
        k in q for k in ["회원수", "신규", "활성", "로그인", "조회수", "매출", "판매"]
    )

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
            query=state.get("user_query", "")[:100],
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


class NLProcessor(BaseNode):
    """Natural Language Processing node for query analysis."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Use centralized LLM service for intent classification
        self.llm = self._get_intent_llm()
        self.fanding_templates = FandingSQLTemplates()
        # Removed: EnhancedRAGMapper and DynamicSchemaExpander (deleted modules)
        # self.rag_mapper = EnhancedRAGMapper(config)
        # self.schema_expander = DynamicSchemaExpander(config)

    def process(self, state: GraphState) -> GraphState:
        """Process natural language query and extract intent and entities."""
        self._log_processing(state, "NLProcessor")

        try:
            # 입력 데이터 검증
            user_query = state.get("user_query")
            if not user_query:
                self.logger.error("user_query is None or empty")
                # 재입력 요청 설정
                state["conversation_response"] = (
                    "죄송합니다. 질문을 받지 못했어요. 😊\n\n"
                    "다시 질문해주시면 처리해드리겠습니다.\n"
                    "예시: '9월 신규 회원 현황 알려줘', '활성 회원 수 조회해줘'"
                )
                state["skip_sql_generation"] = True
                state["needs_clarification"] = True  # 재입력 필요 플래그 설정
                state["success"] = False
                return state

            # Normalize query
            normalized_query = self._normalize_query(user_query)

            # 정규화된 쿼리 검증
            if not normalized_query or len(normalized_query.strip()) == 0:
                self.logger.error("normalized_query is empty after processing")
                # 재입력 요청 설정
                state["conversation_response"] = (
                    "죄송합니다. 질문을 이해하지 못했어요. 🤔\n\n"
                    "다시 질문해주시면 처리해드리겠습니다.\n"
                    "예시: '9월 신규 회원 현황 알려줘', '활성 회원 수 조회해줘'"
                )
                state["skip_sql_generation"] = True
                state["needs_clarification"] = True  # 재입력 필요 플래그 설정
                state["success"] = False
                return state

            # Extract intent and entities (LLM 결과 포함)
            llm_intent_result = state.get("llm_intent_result")
            intent, entities = self._extract_intent_and_entities(
                normalized_query, llm_intent_result
            )  # NOTE: confidence가 높다면 LLM의 것을, 낮다면 규칙 기반으로 intent를 분류하고 필요하다면 entity를 추출함

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
                self.logger.warning(
                    f"Unknown intent: {intent}, treating as general chat"
                )
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
            self.logger.info(
                f"🔍 Ambiguous query detected - requesting clarification: {user_query}"
            )
            clarification_question = (
                self.fanding_templates.generate_clarification_question(user_query)
            )
            state["conversation_response"] = clarification_question
            state["skip_sql_generation"] = True
            state["needs_clarification"] = True  # 재입력 필요 플래그 설정
            state["success"] = True  # 명확화 질문 생성 성공 (정상 처리)
            self.logger.info(
                "✅ Generated clarification question for ambiguous query (this is normal behavior)"
            )
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
            state["success"] = True  # 템플릿 매칭 성공
            self.logger.info(f"SQL template applied: {fanding_template.sql_template}")
        else:
            # 3. 동적 월별 템플릿 생성 시도 (멤버십 성과 관련)
            try:
                dynamic_template = (
                    self.fanding_templates.create_dynamic_monthly_template(user_query)
                )
                if dynamic_template:
                    self.logger.info(
                        f"Dynamic monthly template created: {dynamic_template.name}"
                    )
                    set_fanding_template(state, dynamic_template)
                    state["skip_sql_generation"] = False
                    state["success"] = True  # 동적 템플릿 생성 성공
                    self.logger.info(
                        f"Dynamic SQL applied: {dynamic_template.sql_template[:100]}..."
                    )
                    return
            except Exception as e:
                self.logger.warning(
                    f"Dynamic monthly template creation failed: {str(e)}"
                )

            # 4. 모든 방법 실패 시 일반 SQL 생성으로 진행
            self.logger.info(
                "No template/pattern matched, proceeding with general SQL generation"
            )
            state["skip_sql_generation"] = False
            state["success"] = True  # 일반 SQL 생성으로 진행 (정상 처리)

    def _normalize_query(self, query: str) -> str:
        """Normalize the user query."""
        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", query.strip())

        # Convert to lowercase for consistency
        normalized = normalized.lower()

        # Handle common Korean database terms (상수에서 가져오기)
        for korean, english in KOREAN_MAPPINGS.items():
            normalized = normalized.replace(korean, english)

        return normalized

    def _extract_intent_and_entities(
        self, query: str, llm_intent_result: Optional[Dict] = None
    ) -> Tuple[QueryIntent, List[Entity]]:
        """Extract intent and entities from the query."""

        # 1. LLM 분류 결과가 있으면 우선 사용 (MEDIUM 임계값 0.6 사용)
        # MEDIUM 임계값: LLM 분류가 상당히 확실할 때만 사용하여 오분류 방지
        if (
            llm_intent_result
            and llm_intent_result.get("confidence", 0)
            >= LLM_CONFIDENCE_THRESHOLD_MEDIUM
        ):
            try:
                llm_intent = QueryIntent(llm_intent_result["intent"])
                self.logger.info(
                    f"Using LLM intent classification: {llm_intent.value} (confidence: {llm_intent_result['confidence']:.2f})"
                )
                # 엔티티도 추출
                entities = self._extract_entities_from_query(query)
                return llm_intent, entities
            except ValueError:
                self.logger.warning(
                    f"Invalid LLM intent: {llm_intent_result.get('intent')}"
                )

        # 2. LLM 분류 결과가 있으면 참고 (LOW 임계값 0.3 사용)
        # LOW 임계값: LLM이 불확실해도 규칙 기반보다는 나을 수 있으므로 최소한의 신뢰도로 참고
        if (
            llm_intent_result
            and llm_intent_result.get("confidence", 0) >= LLM_CONFIDENCE_THRESHOLD_LOW
        ):
            try:
                llm_intent = QueryIntent(llm_intent_result["intent"])
                self.logger.info(
                    f"Using LLM intent as fallback: {llm_intent.value} (confidence: {llm_intent_result['confidence']:.2f})"
                )
                entities = self._extract_entities_from_query(query)
                return llm_intent, entities
            except ValueError:
                pass

        # 3. 데이터 조회 의도가 있는 경우 처리 (LLM 실패 시 fallback)
        if self._has_data_query_indicators(query):
            # 데이터 조회 의도가 있으면 DATA_QUERY로 분류 (LLM 실패해도)
            self.logger.info(
                f"Data query indicators detected, classifying as DATA_QUERY: {query}"
            )
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
            entities.append(
                Entity(name="statistics", type="aggregation", confidence=0.8)
            )

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
            "조회",
            "검색",
            "데이터",
            "테이블",
            "쿼리",
            "사용자",
            "회원",
            "크리에이터",
            "펀딩",
            "프로젝트",
            "주문",
            "개수",
            "수",
            "합계",
            "평균",
            "최대",
            "최소",
            "통계",
            "멤버십",
            "성과",
            "매출",
            "방문자",
            "리텐션",
            "포스트",
            "조회수",
            "인기",
            "분석",
            "리포트",
            "월간",
            "일간",
            "주간",
            "년간",
            # 추가 키워드
            "뽑아줘",
            "뽑아",
            "추출",
            "선택",
            "고르",
            "정렬",
            "순위",
            "top",
            "top5",
            "top10",
            "상위",
            "최고",
            "많은",
            "적은",
            "회원수",
            "멤버수",
            "사용자수",
            "가입자",
            "활성",
            "신규",
            "크리에이터",
            "창작자",
            "작가",
            "아티스트",
            "제작자",
        ]

        # 데이터 조회와 관련된 구체적인 질문 패턴
        data_question_patterns = [
            "얼마나",
            "몇 개",
            "몇 명",
            "몇 건",
            "몇 개의",
            "몇 명의",
            "가져와",
            "찾아줘",
            "보여줘",
            "알려줘",  # 데이터 관련 맥락에서만
        ]

        # 일반적인 질문은 제외
        general_question_patterns = [
            "뭐야",
            "뭔가",
            "뭔지",
            "어떻게",
            "왜",
            "언제",
            "어디서",
            "할 수 있는",
            "할 수 있는지",
            "할 수 있는게",
            "할 수 있는것",
        ]

        # 일반적인 질문 패턴이 포함되어 있으면 데이터 조회가 아님
        if any(pattern in query_lower for pattern in general_question_patterns):
            return False

        # 데이터 조회 키워드나 구체적인 질문 패턴이 있는 경우만
        return any(keyword in query_lower for keyword in data_keywords) or any(
            pattern in query_lower for pattern in data_question_patterns
        )

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
            response_content = response.content
            # Handle different response types
            if isinstance(response_content, str):
                response_text = response_content
            elif isinstance(response_content, list):
                # Extract text from list of content blocks
                response_text = " ".join(str(item) for item in response_content)
            else:
                response_text = str(response_content)
            result = self._parse_llm_response(response_text)

            intent = QueryIntent(result.get("intent", "UNKNOWN"))
            entities = [
                Entity(
                    name=entity["name"],
                    type=entity["type"],
                    confidence=entity["confidence"],
                    context=entity.get("context"),
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
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"intent": "UNKNOWN", "entities": []}
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {str(e)}")
            return {"intent": "UNKNOWN", "entities": []}

    def _calculate_confidence(
        self, query: str, intent: QueryIntent, entities: List[Entity]
    ) -> float:
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

    def process(self, state: GraphState) -> GraphState:
        """Map entities to database schema."""
        self._log_processing(state, "SchemaMapper")

        # TODO: Implement schema mapping logic
        pass

        return state


class SQLGenerationNode(BaseNode):
    """SQL 생성 에이전트 노드 - 자연어 쿼리를 SQL로 변환"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.db_schema = config.get("db_schema", {})

        # LLM 서비스에서 SQL LLM 가져오기
        self.llm = self._get_sql_llm()

    def process(self, state: GraphState) -> GraphState:
        """자연어 쿼리를 SQL로 변환"""
        self._log_processing(state, "SQLGenerationNode")

        # TODO: Implement SQL generation logic using LLM and templates
        pass

        return state


class SQLValidationNode(BaseNode):
    """SQL 검증 에이전트 노드 - 생성된 SQL의 구문 및 의미 검증"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # 캐싱된 데이터베이스 스키마 사용
        # self.db_schema = get_cached_db_schema()

    def process(self, state: GraphState) -> GraphState:
        """SQL 쿼리 검증"""
        self._log_processing(state, "SQLValidationNode")

        # TODO: Implement SQL validation logic
        pass

        return state


class DataSummarizationNode(BaseNode):
    """데이터 요약 에이전트 노드 - SQL 실행 결과를 자연어로 요약"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # LLM 서비스에서 SQL LLM 가져오기 (요약용으로도 사용)
        self.llm = self._get_sql_llm()

        # Removed: DataInsightAnalyzer (deleted module)
        # self.insight_analyzer = DataInsightAnalyzer(config)

    def process(self, state: GraphState) -> GraphState:
        """SQL 실행 결과를 자연어로 요약"""
        self._log_processing(state, "DataSummarizationNode")

        try:
            # 대화 응답이 있는 경우 요약 건너뛰기
            conversation_response = state.get("conversation_response")
            intent = state.get("intent")
            fanding_template = state.get("fanding_template")

            self.logger.info(
                f"DataSummarizationNode - conversation_response: {conversation_response is not None}"
            )
            self.logger.info(f"DataSummarizationNode - intent: {intent}")
            self.logger.info(
                f"DataSummarizationNode - fanding_template: {fanding_template is not None}"
            )

            if conversation_response or intent in [
                "GREETING",
                "GENERAL_CHAT",
                "HELP_REQUEST",
            ]:
                self.logger.info(
                    "Skipping data summarization for conversation response"
                )
                state["data_summary"] = (
                    conversation_response or "대화 응답이 처리되었습니다."
                )
                state["success"] = True
                return state

            # Fanding 템플릿이 있는 경우 특별 처리
            if fanding_template:
                query_result = state.get("query_result")
                if query_result:
                    # Fanding 템플릿 결과 포맷팅
                    from .fanding_sql_templates import FandingSQLTemplates

                    templates = FandingSQLTemplates()
                    formatted_result = templates.format_sql_result(
                        fanding_template, query_result
                    )
                    state["data_summary"] = formatted_result
                    state["success"] = True
                    self.logger.info(
                        f"🎯 Fanding template result formatted: {fanding_template.name}"
                    )
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
            state["business_insights"] = None

            # 요약 생성
            if self.llm:
                summary = self._generate_ai_summary(
                    user_query, query_result, result_stats
                )
            else:
                summary = self._generate_fallback_summary(query_result, result_stats)

            state["data_summary"] = summary
            # Ensure result_stats is a dict for result_statistics
            if isinstance(result_stats, dict):
                state["result_statistics"] = result_stats
            else:
                state["result_statistics"] = None

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
            "null_counts": {},
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

    def _generate_ai_summary(
        self, user_query: str, query_result: List[Dict[str, Any]], stats: Dict[str, Any]
    ) -> str:
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
                HumanMessage(
                    content=f"당신은 데이터 분석 전문가입니다. 쿼리 결과를 사용자 친화적으로 요약해주세요.\n\n{summary_prompt}"
                )
            ]

            if not self.llm:
                self.logger.warning("LLM not initialized, returning default summary")
                return "데이터 요약을 생성할 수 없습니다."

            response = self.llm.invoke(messages)
            response_content = response.content
            # Handle different response types
            if isinstance(response_content, str):
                return response_content.strip()
            elif isinstance(response_content, list):
                # Extract text from list of content blocks
                return " ".join(str(item) for item in response_content).strip()
            else:
                return str(response_content).strip()

        except Exception as e:
            self.logger.error(f"AI summary generation failed: {e}")
            return self._generate_fallback_summary(query_result, stats)

    def _generate_fallback_summary(
        self, query_result: List[Dict[str, Any]], stats: Dict[str, Any]
    ) -> str:
        """Fallback 요약 생성"""
        row_count = stats.get("row_count", 0)
        columns = stats.get("columns", [])

        if row_count == 0:
            return "쿼리 결과가 없습니다."
        elif row_count == 1:
            return f"총 1개의 결과가 조회되었습니다. 컬럼: {', '.join(columns)}"
        else:
            return (
                f"총 {row_count}개의 결과가 조회되었습니다. 컬럼: {', '.join(columns)}"
            )

    def _format_results(
        self, query_result: List[Dict[str, Any]], max_rows: int = 10
    ) -> str:
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

    def _calculate_summary_confidence(
        self, summary: str, stats: Dict[str, Any]
    ) -> float:
        """요약 신뢰도 계산"""
        base_confidence = 0.8

        # 요약 길이에 따른 조정
        if len(summary) > 50:
            base_confidence += 0.1

        # 통계 정보 활용도에 따른 조정
        if stats.get("row_count", 0) > 0:
            base_confidence += 0.1

        return min(base_confidence, 1.0)
