"""
LangGraph Node Components for NL-to-SQL Pipeline

This module implements the individual nodes that make up the LangGraph pipeline.
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.output_parsers.json import SimpleJsonOutputParser

from .prompts import (
    SQLPromptTemplate,
    GREETING_PATTERNS, HELP_REQUEST_PATTERNS, GENERAL_CHAT_PATTERNS, GRATITUDE_PATTERNS,
    generate_greeting_response, generate_help_response, 
    generate_general_chat_response, generate_error_response,
    generate_clarification_question
)
from agentic_flow.llm_output_parser import parse_json_response
from agentic_flow.llm_service import get_llm_service
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from .fanding_sql_templates import FandingSQLTemplates
from .date_utils import DateUtils
# Note: PythonCodeGeneratorNode is imported in state_machine.py to avoid circular import

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
# Note: Intent patterns and response templates moved to prompts.py
# Only business logic constants remain here

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

# Note: GRATITUDE_PATTERNS moved to prompts.py
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


# Note: Response generator functions moved to prompts.py
# Import them from prompts module instead


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
            query=state.get("user_query", "")[:100]
        )
    
    def _get_llm_service(self):
        """Get LLM service instance (lazy initialization)."""
        if self._llm_service is None:
            self._llm_service = get_llm_service()
        return self._llm_service
    
    def _get_intent_llm(self):
        """Get intent classification LLM (lightweight, fast response)."""
        return self._get_llm_service().get_intent_llm()
    
    def _get_sql_llm(self):
        """Get SQL generation LLM (high-performance model)."""
        return self._get_llm_service().get_sql_llm()
    
    def _extract_creator_name(self, query: str) -> Optional[str]:
        """
        사용자 쿼리에서 크리에이터 이름 추출 (공통 유틸리티 메서드)
        
        Args:
            query: 사용자 쿼리
            
        Returns:
            추출된 크리에이터 이름 또는 None
        
        예시:
            "세상학 개론의 회원 수" -> "세상학 개론"
            "세상학개론 크리에이터의 10월 신규 회원수" -> "세상학개론"
            "크리에이터 세상학 개론 신규 회원" -> "세상학 개론"
            "상위 5개 크리에이터" -> None (구체적인 이름 없음)
        """
        import re
        
        # 패턴 1: "크리에이터명 크리에이터" 형식 (예: "세상학개론 크리에이터의")
        pattern1 = r'([가-힣a-zA-Z0-9\s]{2,30}?)\s+크리에이터\s*(?:의|이|가|을|를|에|에서|로|으로)?'
        match1 = re.search(pattern1, query, re.IGNORECASE)
        if match1:
            creator_name = match1.group(1).strip()
            # "크리에이터" 키워드 자체는 제외
            if creator_name and creator_name.lower() not in ['크리에이터', 'creator']:
                # 숫자나 날짜 표현이 포함된 경우 제외 (예: "10월", "2024년")
                if not re.search(r'\d+\s*(?:월|년|일|개|명|위)', creator_name):
                    self.logger.debug(f"Extracted creator name (pattern 1): '{creator_name}'")
                    return creator_name
        
        # 패턴 2: "크리에이터명의" 형식 (예: "세상학 개론의 회원 수")
        pattern2 = r'([가-힣a-zA-Z0-9\s]{2,30}?)\s*(?:의|이|가|을|를|에|에서|로|으로)\s+(?:신규|활성|회원|멤버|구독자|팔로워|수|개수|통계|분석|조회|보여|알려|찾아|가져)'
        match2 = re.search(pattern2, query, re.IGNORECASE)
        if match2:
            creator_name = match2.group(1).strip()
            # 키워드 제외 리스트
            exclude_keywords = ['크리에이터', 'creator', '상위', 'top', '인기', 'popular', '최고', '많은', '적은', '전체', '모든']
            if creator_name and len(creator_name) >= 2 and creator_name.lower() not in [kw.lower() for kw in exclude_keywords]:
                # 숫자나 날짜 표현이 포함된 경우 제외
                if not re.search(r'\d+\s*(?:월|년|일|개|명|위)', creator_name):
                    # "상위 5개" 같은 패턴 제외
                    if not re.search(r'(상위|top|인기|최고)\s*\d+', creator_name, re.IGNORECASE):
                        self.logger.debug(f"Extracted creator name (pattern 2): '{creator_name}'")
                        return creator_name
        
        # 패턴 3: "크리에이터 크리에이터명" 형식 (예: "크리에이터 세상학 개론")
        pattern3 = r'크리에이터\s+([가-힣a-zA-Z0-9\s]{2,30}?)(?:\s+(?:의|이|가|을|를|에|에서|로|으로)|신규|활성|회원|멤버|구독자|팔로워|수|개수|통계|분석|조회|보여|알려|찾아|가져|\s+\d+)'
        match3 = re.search(pattern3, query, re.IGNORECASE)
        if match3:
            creator_name = match3.group(1).strip()
            if creator_name:
                # 숫자나 날짜 표현이 포함된 경우 제외
                if not re.search(r'\d+\s*(?:월|년|일|개|명|위)', creator_name):
                    self.logger.debug(f"Extracted creator name (pattern 3): '{creator_name}'")
                    return creator_name
        
        return None


class NLProcessor(BaseNode):
    """Natural Language Processing node for query analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Use centralized LLM service for intent classification
        self.llm = self._get_intent_llm()
        # FandingSQLTemplates를 config에서 공유하거나 새로 생성 (중복 초기화 방지)
        if "fanding_templates" in config:
            self.fanding_templates = config["fanding_templates"]
            self.logger.debug("Using shared FandingSQLTemplates from config")
        else:
            # db_schema는 부모 클래스에서 이미 로드됨 (없으면 자동 로드)
            self.fanding_templates = FandingSQLTemplates(db_schema=getattr(self, 'db_schema', None))
            config["fanding_templates"] = self.fanding_templates  # 다른 노드에서 재사용할 수 있도록 config에 저장
    
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
            
            # needs_clarification이 이미 해결되었는지 확인 (LLMIntentClassifier에서 보완 쿼리 결합)
            if state.get("needs_clarification", False) and not state.get("conversation_response"):
                # needs_clarification이 True였지만 conversation_response가 없다면
                # LLMIntentClassifier에서 보완 쿼리를 결합했을 가능성이 높음
                state["needs_clarification"] = False
                self.logger.info("Cleared needs_clarification flag - clarification followup detected")
            
            # Extract intent and entities (LLM 결과 포함)
            llm_intent_result = state.get("llm_intent_result")
            intent, entities = self._extract_intent_and_entities(normalized_query, llm_intent_result)
            
            # Update state
            state["normalized_query"] = normalized_query
            state["intent"] = intent
            state["entities"] = entities
            
            # 주의: CHAT_PATH (GREETING, HELP_REQUEST, GENERAL_CHAT)는 route_after_intent_classification에서
            # 이미 data_summarization으로 라우팅되므로 nl_processing 노드에는 도달하지 않음
            # 따라서 여기서 도달하는 intent는 데이터 의도(SIMPLE_AGGREGATION, COMPLEX_ANALYSIS)만 처리
            
            # 스키마 정보 요청 처리 (SHOW/DESCRIBE 대안)
            # 이는 데이터 의도이지만 특별한 처리가 필요할 수 있음
            schema_info_response = self.fanding_templates.get_schema_info(user_query)
            if schema_info_response:
                state["conversation_response"] = schema_info_response
                state["intent"] = QueryIntent.HELP_REQUEST
                state["skip_sql_generation"] = True
                state["success"] = True
                self.logger.info(f"Schema information request handled: {user_query}")
                return state
            
            # 데이터 의도 처리 (SIMPLE_AGGREGATION, COMPLEX_ANALYSIS)
            if intent == QueryIntent.SIMPLE_AGGREGATION:
                # 간단한 집계 쿼리 - SQL 경로로 처리
                self.logger.info(f"Simple aggregation intent detected: {user_query}")
                # SQL 경로: fanding_template 매칭 시도
                self._handle_data_query(state, user_query)
                state["success"] = True
            
            elif intent == QueryIntent.COMPLEX_ANALYSIS:
                # 복잡한 분석 쿼리 - Python 경로로 처리
                self.logger.info(f"Complex analysis intent detected: {user_query}")
                # Python 경로 최적화: SQL 템플릿 매칭 건너뛰기 (SQL 생성이 없으므로 불필요)
                # entities는 이미 추출되었으므로 rag_schema_retriever에서 사용 가능
                # _handle_data_query()는 SQL 경로 전용이므로 호출하지 않음
                state["success"] = True
            
            else:
                # 예외 상황: 데이터 의도가 아닌 intent가 도달한 경우
                # 이는 라우팅 로직 오류이거나 intent 분류 오류일 수 있음
                self.logger.error(
                    f"Unexpected intent '{intent}' reached nlp_processing. "
                    f"This should only process SIMPLE_AGGREGATION or COMPLEX_ANALYSIS. "
                    f"Routing may be incorrect."
                )
                # 안전하게 처리: 데이터 조회로 간주하고 진행
                self.logger.warning(f"Treating unexpected intent '{intent}' as SIMPLE_AGGREGATION for safety")
                self._handle_data_query(state, user_query)
                state["success"] = True
            
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
        # 크리에이터 정보가 포함된 쿼리인지 간단한 키워드 체크만 수행
        # (정확한 파싱은 SQLGenerationNode에서 필요할 때만 수행)
        query_lower = user_query.lower()
        has_creator_keyword = (
            '크리에이터' in query_lower or 
            'creator' in query_lower or
            any(keyword in query_lower for keyword in ['작가', '아티스트', '제작자'])
        )
        
        # Fanding 템플릿 매칭 시도
        fanding_template = self.fanding_templates.match_query_to_template(user_query)
        if fanding_template:
            # 크리에이터 정보가 필요한 쿼리인데 템플릿에 크리에이터 필터링이 없는 경우
            # 템플릿을 사용하지 않고 일반 SQL 생성으로 진행
            sql_template = fanding_template.sql_template if hasattr(fanding_template, 'sql_template') else str(fanding_template)
            if has_creator_keyword and 'creator' not in sql_template.lower() and 'creator_no' not in sql_template:
                self.logger.info(f"Fanding template matched but missing creator filter: {fanding_template.name}. Skipping template, will generate SQL with creator filter.")
                # 템플릿을 사용하지 않고 일반 SQL 생성으로 진행
                fanding_template = None
            else:
                self.logger.info(f"Fanding template matched: {fanding_template.name}")
                set_fanding_template(state, fanding_template)
                state["skip_sql_generation"] = False
                state["success"] = True  # 템플릿 매칭 성공
                self.logger.info(f"SQL template applied: {sql_template[:100]}...")
                return  # 템플릿 사용 성공 시 여기서 종료
        
        if not fanding_template:
            # 3. 동적 월별 템플릿 생성 시도 (멤버십 성과 관련)
            try:
                dynamic_template = self.fanding_templates.create_dynamic_monthly_template(user_query)
                if dynamic_template:
                    self.logger.info(f"Dynamic monthly template created: {dynamic_template.name}")
                    set_fanding_template(state, dynamic_template)
                    state["skip_sql_generation"] = False
                    state["success"] = True  # 동적 템플릿 생성 성공
                    self.logger.info(f"Dynamic SQL applied: {dynamic_template.sql_template[:100]}...")
                    return
            except Exception as e:
                self.logger.warning(f"Dynamic monthly template creation failed: {str(e)}")
            
            # 4. 모든 방법 실패 시 일반 SQL 생성으로 진행
            self.logger.info("No template/pattern matched, proceeding with general SQL generation")
            state["skip_sql_generation"] = False
            state["success"] = True  # 일반 SQL 생성으로 진행 (정상 처리)

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
            # 데이터 조회 의도가 있으면 SIMPLE_AGGREGATION으로 분류 (LLM 실패해도)
            # 불명확한 경우 기본값으로 SIMPLE_AGGREGATION 사용 (SQL이 더 안전)
            self.logger.info(f"Data query indicators detected, classifying as SIMPLE_AGGREGATION: {query}")
            entities = self._extract_entities_from_query(query)
            return QueryIntent.SIMPLE_AGGREGATION, entities
        
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
            return QueryIntent.SIMPLE_AGGREGATION
        
        # 3. 데이터 조회 의도 키워드 (명확한 조회 의도)
        if any(pattern in query_lower for pattern in DATA_QUERY_PATTERNS):
            return QueryIntent.SIMPLE_AGGREGATION
        
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
        # db_schema가 없거나 비어있으면 초기화 시점에 한 번만 로드 (성능 최적화)
        self.db_schema = config.get("db_schema") or {}
        if not self.db_schema or len(self.db_schema) == 0:
            from core.db import get_cached_db_schema
            self.db_schema = get_cached_db_schema()
            self.logger.debug("db_schema was empty in config, loaded from cache during initialization")
        
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
        from agentic_flow.utils import calculate_mapping_confidence
        return calculate_mapping_confidence(entities, tables, columns)


class SQLGenerationNode(BaseNode):
    """
    SQL 생성 에이전트 노드 - 자연어 쿼리를 SQL로 변환
    
    - 간단한 쿼리: Few-shot 예제 기반 빠른 경로
    - 복잡한 쿼리: RAG 컨텍스트 기반 정확한 경로
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # db_schema가 없거나 비어있으면 초기화 시점에 한 번만 로드 (성능 최적화)
        self.db_schema = config.get("db_schema") or {}
        if not self.db_schema or len(self.db_schema) == 0:
            self.db_schema = get_cached_db_schema()
            self.logger.debug("db_schema was empty in config, loaded from cache during initialization")
        
        # LLM 서비스에서 SQL LLM 가져오기
        self.llm = self._get_sql_llm()
        
        # SQLPromptTemplate 초기화 (RAG 컨텍스트 기반 복잡한 쿼리용)
        self.prompt_template = SQLPromptTemplate(db_schema=self.db_schema)
        
        # DynamicSQLGenerator 통합: Few-shot 예제 기반 빠른 경로용 프롬프트
        self.simple_prompt = ChatPromptTemplate.from_messages([
            ("system", """자연어를 SQL로 변환하세요.

테이블: t_member, t_creator, t_member_info, t_member_login_log

JSON 형식으로만 응답:
{{
    "sql_query": "SELECT ...",
    "confidence": 0.9,
    "reasoning": "이유"
}}"""),
            ("human", "쿼리: {query}")
        ])
        
        # JSON 파서 초기화 (Few-shot 경로용)
        self.json_parser = SimpleJsonOutputParser()
        
        # FandingSQLTemplates를 config에서 공유하거나 새로 생성 (중복 초기화 방지)
        if "fanding_templates" in config:
            self.fanding_templates = config["fanding_templates"]
            self.logger.debug("Using shared FandingSQLTemplates from config")
        else:
            # db_schema는 부모 클래스에서 이미 로드됨 (없으면 자동 로드)
            self.fanding_templates = FandingSQLTemplates(db_schema=getattr(self, 'db_schema', None))
            config["fanding_templates"] = self.fanding_templates  # 다른 노드에서 재사용할 수 있도록 config에 저장
    
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
            
            # # 대화 인텐트인 경우 SQL 생성 건너뛰기
            # if (skip_flag or conversation_response or 
            #     intent in ["GREETING", "GENERAL_CHAT", "HELP_REQUEST"]):
            #     # 명확화 질문 또는 대화 의도인지 확인
            #     if state.get("conversation_response") and "어떤" in str(state.get("conversation_response", "")):
            #         self.logger.info("Skipping SQL generation - clarification question detected")
            #     else:
            #         self.logger.info("Skipping SQL generation for conversation intent")
            #     state["sql_query"] = None
            #     state["validated_sql"] = None
            #     state["confidence_scores"]["sql_generation"] = 1.0
            #     return state
            
            user_query = state["user_query"]
            schema_mapping = state.get("schema_mapping")
            entities = state.get("entities", [])
            rag_schema_context = state.get("rag_schema_context", "")
            
            # 누적 슬롯 병합 (이전 state + 현재 질의)
            prior_slots = state.get("slots") or {}
            new_slots = self._extract_simple_slots(user_query)
            slots = {**prior_slots, **{k: v for k, v in new_slots.items() if v}}
            state["slots"] = slots
            
            # 쿼리 복잡성 평가 및 조건부 처리
            query_complexity = self._assess_query_complexity(user_query, entities, rag_schema_context)
            self.logger.info(f"Query complexity assessed as: {query_complexity}")
            
            # 1. NLProcessor에서 이미 매칭된 fanding_template 확인 (최우선)
            # NLProcessor에서 템플릿 매칭을 이미 수행했으므로 이를 우선 활용
            # 단, 크리에이터 정보가 필요한 쿼리인데 템플릿에 크리에이터 필터링이 없는 경우는 일반 SQL 생성으로 진행
            fanding_template = state.get("fanding_template")
            if fanding_template:
                # 템플릿 객체에서 SQL 추출
                sql_template = None
                if hasattr(fanding_template, 'sql_template'):
                    sql_template = fanding_template.sql_template
                    template_name = fanding_template.name
                elif isinstance(fanding_template, dict):
                    sql_template = fanding_template.get("sql_template")
                    template_name = fanding_template.get("name", "unknown")
                
                if sql_template:
                    # 크리에이터 정보가 필요한 쿼리인지 확인
                    creator_name = self._extract_creator_name(user_query)
                    if creator_name:
                        # 크리에이터 정보가 필요한데 템플릿에 크리에이터 필터링이 없는 경우
                        if 'creator' not in sql_template.lower() and 'creator_no' not in sql_template:
                            self.logger.info(f"Fanding template '{template_name}' matched but missing creator filter. Extracting creator info and adding to template.")
                            # 크리에이터 정보 추출 및 추가
                            creator_info = self._find_creator_by_name(creator_name)
                            if creator_info:
                                # db_schema를 기반으로 실제 테이블과 컬럼 찾기 (하드코딩 제거)
                                creator_col = self._find_creator_column_in_sql_template(sql_template, state)
                                
                                if creator_col:
                                    # 템플릿 SQL에 크리에이터 필터 추가
                                    # WHERE 절이 있으면 AND로 추가, 없으면 WHERE 추가
                                    if 'WHERE' in sql_template.upper():
                                        # 기존 WHERE 절에 AND 추가
                                        sql_template = sql_template.rstrip(';').rstrip() + f" AND {creator_col} = :creator_no"
                                    else:
                                        # WHERE 절 추가
                                        sql_template = sql_template.rstrip(';').rstrip() + f" WHERE {creator_col} = :creator_no"
                                    
                                    # SQL 파라미터 설정
                                    if "sql_params" not in state:
                                        state["sql_params"] = {}
                                    state["sql_params"]["creator_no"] = creator_info["creator_no"]
                                    
                                    self.logger.info(f"Added creator filter to template: {creator_col} = {creator_info['creator_no']} (creator: '{creator_name}')")
                                else:
                                    self.logger.warning(f"Could not find creator column in SQL template using db_schema, using template as-is")
                            else:
                                self.logger.warning(f"Creator '{creator_name}' not found in database, using template as-is")
                        else:
                            # 템플릿에 이미 크리에이터 필터링이 있는 경우
                            self.logger.info(f"Fanding template '{template_name}' already includes creator filter")
                    
                    self.logger.info(f"Using fanding_template matched by NLProcessor: {template_name}")
                    state["sql_query"] = sql_template
                    state["confidence_scores"]["sql_generation"] = 0.9  # 템플릿 매칭은 높은 신뢰도
                    self.logger.info(f"SQL from NLProcessor-matched template: {sql_template[:100]}...")
                    return state
            
            # 2. 기존 dynamic_sql_result가 있으면 우선 사용 (하위 호환성)
            dynamic_sql_result = state.get("dynamic_sql_result")
            if dynamic_sql_result and isinstance(dynamic_sql_result, dict) and dynamic_sql_result.get("sql_query"):
                dynamic_confidence = dynamic_sql_result.get("confidence", 0.0)
                if dynamic_confidence >= 0.85:
                    self.logger.info(f"Using existing dynamic_sql_result (confidence: {dynamic_confidence:.2f})")
                    state["sql_query"] = dynamic_sql_result["sql_query"]
                    state["confidence_scores"]["sql_generation"] = dynamic_confidence
                return state
            
            # 3. RAG 매핑 결과 확인 - 신뢰도 임계값 낮춤
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
                
                # 크리에이터 이름 추출 (예: "세상학 개론")
                creator_name = self._extract_creator_name(user_query)
                
                # 크리에이터명이 있으면 DB에서 creator_no 검색 (SQL Injection 방지 + 유사도 검색)
                creator_info = None
                if creator_name:
                    creator_info = self._find_creator_by_name(creator_name)
                    if creator_info:
                        self.logger.info(f"Creator found: '{creator_name}' -> creator_no={creator_info['creator_no']}, match_type={creator_info['match_type']}, similarity={creator_info['similarity']:.2f}")
                    else:
                        self.logger.warning(f"Creator not found for name: '{creator_name}'")
                
                creator_col = self._guess_creator_column(state)
                if creator_col:
                    if metric == "active_members":
                        # t_member에는 creator_no가 없으므로 t_fanding을 JOIN해야 함
                        # data_dictionary.md 기준: t_fanding.fanding_status = 'T' (활성 멤버십)
                        if creator_info:
                            # 구체적인 크리에이터명이 있고 매칭된 경우: creator_no를 직접 사용 (SQL Injection 방지)
                            # 파라미터 바인딩을 위해 SQL 쿼리에 파라미터 플레이스홀더 사용
                            sql = (
                                    "SELECT f.{creator_col}, cm.nickname AS creator_name, COUNT(DISTINCT f.member_no) AS active_members "
                                    "FROM t_fanding f "
                                    "INNER JOIN t_member m ON f.member_no = m.no "
                                    "INNER JOIN t_creator c ON f.{creator_col} = c.no "
                                    "INNER JOIN t_member cm ON c.member_no = cm.no "
                                    "WHERE f.fanding_status = 'T' AND m.status = 'A' AND f.{creator_col} = :creator_no "
                                    "GROUP BY f.{creator_col}, cm.nickname ORDER BY active_members DESC LIMIT :limit_k"
                            ).format(creator_col=creator_col)
                            # SQL 파라미터를 state에 저장 (sql_execution 노드에서 사용)
                            if "sql_params" not in state:
                                state["sql_params"] = {}
                            state["sql_params"] = {
                                "creator_no": creator_info["creator_no"],
                                "limit_k": k
                            }
                        else:
                            # 크리에이터명이 없는 경우: 전체 크리에이터별 집계
                            sql = (
                                "SELECT f.{creator_col}, COUNT(DISTINCT f.member_no) AS active_members "
                                "FROM t_fanding f "
                                "INNER JOIN t_member m ON f.member_no = m.no "
                                "WHERE f.fanding_status = 'T' AND m.status = 'A' "
                                "GROUP BY f.{creator_col} ORDER BY active_members DESC LIMIT :limit_k"
                            ).format(creator_col=creator_col)
                            if "sql_params" not in state:
                                state["sql_params"] = {}
                            state["sql_params"] = {"limit_k": k}
                    else:
                        # 신규 회원: t_fanding의 ins_datetime 사용 (최초 팬딩 시작일)
                        if creator_info:
                            # 구체적인 크리에이터명이 있고 매칭된 경우: creator_no를 직접 사용
                            sql = (
                                "SELECT f.{creator_col}, cm.nickname AS creator_name, COUNT(DISTINCT f.member_no) AS new_members "
                                "FROM t_fanding f "
                                "INNER JOIN t_creator c ON f.{creator_col} = c.no "
                                "INNER JOIN t_member cm ON c.member_no = cm.no "
                                "WHERE DATE_FORMAT(f.ins_datetime, '%Y-%m') = :month AND f.{creator_col} = :creator_no "
                                "GROUP BY f.{creator_col}, cm.nickname ORDER BY new_members DESC LIMIT :limit_k"
                            ).format(creator_col=creator_col)
                            if "sql_params" not in state:
                                state["sql_params"] = {}
                            state["sql_params"] = {
                                "creator_no": creator_info["creator_no"],
                                "month": month,
                                "limit_k": k
                            }
                        else:
                            # 크리에이터명이 없는 경우
                            sql = (
                                "SELECT f.{creator_col}, COUNT(DISTINCT f.member_no) AS new_members "
                                "FROM t_fanding f "
                                "WHERE DATE_FORMAT(f.ins_datetime, '%Y-%m') = :month "
                                "GROUP BY f.{creator_col} ORDER BY new_members DESC LIMIT :limit_k"
                            ).format(creator_col=creator_col)
                            if "sql_params" not in state:
                                state["sql_params"] = {}
                            state["sql_params"] = {
                                "month": month,
                                "limit_k": k
                            }
                    state["sql_query"] = sql
                    state["confidence_scores"]["sql_generation"] = 0.8
                    if creator_info:
                        self.logger.info(f"Built deterministic SQL using accumulated slots (creator_name: '{creator_name}' -> creator_no={creator_info['creator_no']}, match_type={creator_info['match_type']})")
                    else:
                        self.logger.info(f"Built deterministic SQL using accumulated slots (no specific creator name)")
                    return state
                else:
                    clarification = (
                        "크리에이터 식별 컬럼을 확인할 수 없습니다. 어떤 컬럼으로 그룹핑할까요? 예: creator_id/creator_no"
                    )
                    state["clarification_question"] = clarification
                    state["conversation_response"] = clarification
                    state["skip_sql_generation"] = True
                    state["needs_clarification"] = True  # 재입력 필요 플래그 설정
                    state["confidence_scores"]["sql_generation"] = 0.0
                    return state
            
            # 복잡성 기반 조건부 SQL 생성
            existing_sql = state.get("sql_query")
            sql_validation_failed = state.get("sql_validation_failed", False)
            
            # 기존 SQL이 있고 검증 실패하지 않았으면 건너뛰기 (Fanding 템플릿 등)
            if existing_sql and not sql_validation_failed and not rag_schema_context:
                self.logger.info(f"SQL already exists, skipping generation: {existing_sql[:100]}...")
                state["confidence_scores"]["sql_generation"] = state.get("confidence_scores", {}).get("sql_generation", 0.8)
                return state
            
            # 검증 실패 시 기존 SQL 무효화
            if sql_validation_failed:
                self.logger.info("Previous SQL validation failed, generating new SQL...")
                state["sql_query"] = None
                state["sql_validation_failed"] = False
                existing_sql = None
            
            # 복잡성에 따라 SQL 생성 경로 선택
            if query_complexity == "simple":
                # 간단한 쿼리: Few-shot 예제 기반 빠른 경로
                self.logger.info("Using simple SQL generation path (Few-shot based)")
                simple_result = self._generate_sql_simple(user_query)
                
                if simple_result and simple_result.get("sql_query"):
                    sql_query = simple_result["sql_query"]
                    confidence = simple_result.get("confidence", 0.7)
                    
                    state["sql_query"] = sql_query
                    state["confidence_scores"]["sql_generation"] = confidence
                    state["dynamic_sql_result"] = {
                        "sql_query": sql_query,
                        "confidence": confidence,
                        "reasoning": simple_result.get("reasoning", "Few-shot 예제 기반 생성")
                    }
                    self.logger.info(f"Simple SQL generated successfully (confidence: {confidence:.2f})")
                else:
                    # 간단한 경로 실패 시 복잡한 경로로 폴백
                    self.logger.warning("Simple SQL generation failed, falling back to complex path")
                    query_complexity = "complex"  # 폴백
            else:
                # 복잡한 쿼리: RAG 컨텍스트 기반 정확한 경로
                self.logger.info("Using complex SQL generation path (RAG context based)")
                complex_sql = self._generate_sql_complex(user_query, rag_schema_context, schema_mapping)
                
                if complex_sql:
                    state["sql_query"] = complex_sql
                    state["sql_generation_metadata"] = {
                        "model": self.llm.model if self.llm else "unknown",
                        "prompt_length": len(str(rag_schema_context)) if rag_schema_context else 0,
                        "response_length": len(complex_sql),
                        "mock": False
                    }
                    confidence = self._calculate_sql_confidence(
                        {"sql": complex_sql, "success": True, "response_length": len(complex_sql)},
                        schema_mapping
                    )
                    state["confidence_scores"]["sql_generation"] = confidence
                    self.logger.info(f"Complex SQL generated successfully (confidence: {confidence:.2f})")
            
            # SQL 생성 실패 시 폴백 처리
            if not state.get("sql_query"):
                self.logger.warning("Both SQL generation paths failed, attempting fallback...")
                
                # NLProcessor에서 이미 매칭을 시도했으므로, 여기서는 다시 시도하지 않음
                # 대신 최종 폴백: 명확화 질문 요청
                clarification = self._build_clarification_question(user_query)
                state["clarification_question"] = clarification
                state["conversation_response"] = clarification
                state["skip_sql_generation"] = True
                state["needs_clarification"] = True
                state["confidence_scores"]["sql_generation"] = 0.0
                self.logger.info("All SQL generation methods failed, asking for clarification")
            
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
        # 공통 유틸리티 함수 사용
        from agentic_flow.utils import calculate_sql_confidence
        return calculate_sql_confidence(result, schema_mapping)
    
    def _extract_simple_slots(self, query: str) -> Dict[str, Any]:
        """간단 슬롯 추출: month, top_k, intent(creator_topk_new_members)"""
        q = query.lower()
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
    
    # _extract_creator_name은 BaseNode로 이동되었으므로 여기서는 제거됨
    
    def _find_creator_by_name(self, creator_name: str) -> Optional[Dict[str, Any]]:
        """
        크리에이터 이름으로 DB에서 크리에이터 검색 (SQL Injection 방지 + 유사도 검색)
        
        Args:
            creator_name: 크리에이터 이름 (예: "세상학 개론")
            
        Returns:
            {
                "creator_no": int,
                "nickname": str,
                "match_type": "exact" | "partial" | "similar",
                "similarity": float
            } 또는 None
        """
        from core.db import execute_query
        from difflib import SequenceMatcher
        
        if not creator_name or len(creator_name.strip()) < 2:
            return None
        
        creator_name = creator_name.strip()
        
        try:
            # 1. 정확한 매칭 시도
            exact_query = """
                SELECT c.no AS creator_no, m.nickname
                FROM t_creator c
                INNER JOIN t_member m ON c.member_no = m.no
                WHERE m.nickname = :creator_name
                LIMIT 1
            """
            exact_results = execute_query(exact_query, {"creator_name": creator_name}, readonly=True)
            
            if exact_results and len(exact_results) > 0:
                self.logger.debug(f"Exact match found for creator name: '{creator_name}' -> creator_no: {exact_results[0]['creator_no']}")
                return {
                    "creator_no": exact_results[0]["creator_no"],
                    "nickname": exact_results[0]["nickname"],
                    "match_type": "exact",
                    "similarity": 1.0
                }
            
            # 2. 부분 매칭 시도 (LIKE 사용, 파라미터 바인딩으로 SQL Injection 방지)
            # 띄어쓰기 문제 해결: 띄어쓰기 있는 버전과 없는 버전 모두 검색
            partial_query = """
                SELECT c.no AS creator_no, m.nickname
                FROM t_creator c
                INNER JOIN t_member m ON c.member_no = m.no
                WHERE m.nickname LIKE :creator_pattern OR m.nickname LIKE :creator_pattern_no_space
                LIMIT 20
            """
            # 부분 매칭: 원본과 띄어쓰기 제거 버전 모두 검색
            creator_pattern = f"%{creator_name}%"
            creator_pattern_no_space = f"%{creator_name.replace(' ', '')}%"  # 띄어쓰기 제거
            partial_results = execute_query(
                partial_query, 
                {
                    "creator_pattern": creator_pattern,
                    "creator_pattern_no_space": creator_pattern_no_space
                }, 
                readonly=True
            )
            
            if partial_results and len(partial_results) > 0:
                # 유사도 계산을 위한 정규화 함수 (띄어쓰기, 특수문자 제거)
                def normalize_for_similarity(text: str) -> str:
                    """유사도 계산을 위해 텍스트 정규화 (띄어쓰기, 하이픈, 특수문자 제거)"""
                    import re
                    # 띄어쓰기, 하이픈, 특수문자 제거 후 소문자 변환
                    normalized = re.sub(r'[\s\-_\-]', '', text.lower())
                    return normalized
                
                # 여러 결과가 있으면 유사도로 정렬
                if len(partial_results) > 1:
                    # 유사도 계산 및 정렬 (정규화된 버전으로 비교)
                    scored_results = []
                    normalized_creator_name = normalize_for_similarity(creator_name)
                    
                    for result in partial_results:
                        normalized_nickname = normalize_for_similarity(result["nickname"])
                        # 정규화된 버전으로 유사도 계산
                        similarity = SequenceMatcher(None, normalized_creator_name, normalized_nickname).ratio()
                        
                        # 추가 점수: 원본에 포함되어 있으면 가산점
                        if creator_name.lower() in result["nickname"].lower() or creator_name.replace(' ', '').lower() in result["nickname"].lower():
                            similarity = max(similarity, 0.7)  # 최소 0.7 보장
                        
                        scored_results.append({
                            **result,
                            "similarity": similarity
                        })
                    scored_results.sort(key=lambda x: x["similarity"], reverse=True)
                    best_match = scored_results[0]
                    
                    # 유사도가 0.5 이상인 경우만 반환 (임계값 낮춤)
                    if best_match["similarity"] >= 0.5:
                        self.logger.debug(f"Similarity match found for creator name: '{creator_name}' -> '{best_match['nickname']}' (similarity: {best_match['similarity']:.2f})")
                        return {
                            "creator_no": best_match["creator_no"],
                            "nickname": best_match["nickname"],
                            "match_type": "similar" if best_match["similarity"] < 0.9 else "partial",
                            "similarity": best_match["similarity"]
                        }
                else:
                    # 단일 결과
                    result = partial_results[0]
                    normalized_creator_name = normalize_for_similarity(creator_name)
                    normalized_nickname = normalize_for_similarity(result["nickname"])
                    similarity = SequenceMatcher(None, normalized_creator_name, normalized_nickname).ratio()
                    
                    # 추가 점수: 원본에 포함되어 있으면 가산점
                    if creator_name.lower() in result["nickname"].lower() or creator_name.replace(' ', '').lower() in result["nickname"].lower():
                        similarity = max(similarity, 0.7)  # 최소 0.7 보장
                    
                    # 유사도 임계값을 0.5로 낮춤
                    if similarity >= 0.5:
                        self.logger.debug(f"Partial match found for creator name: '{creator_name}' -> '{result['nickname']}' (similarity: {similarity:.2f})")
                        return {
                            "creator_no": result["creator_no"],
                            "nickname": result["nickname"],
                            "match_type": "partial",
                            "similarity": similarity
                        }
            
            # 3. 유사도 검색 (모든 크리에이터와 비교, 성능상 제한적)
            # 실제로는 부분 매칭 결과가 없으면 실패로 처리하는 것이 좋음
            self.logger.warning(f"No creator found for name: '{creator_name}'")
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding creator by name '{creator_name}': {e}")
            return None
    
    def _guess_creator_column(self, state: Optional[GraphState] = None) -> Optional[str]:
        """
        db_schema에서 가능한 크리에이터 식별 컬럼 추정
        
        Args:
            state: 현재 상태 (선택적, 관련 테이블 정보를 가져오기 위해 사용)
            
        Returns:
            크리에이터 컬럼명 (예: "creator_no", "seller_creator_no") 또는 None
        
        Note:
            db_schema는 __init__에서 이미 로드되어 있으므로 여기서는 재로드하지 않음
        """
        # 우선순위: creator_no > seller_creator_no > creator_id > creator
        candidates = ["creator_no", "seller_creator_no", "creator_id", "creator"]
        
        # 1. state에서 관련 테이블 정보 확인 (가장 정확)
        if state:
            schema_mapping = state.get("agent_schema_mapping") or state.get("schema_mapping")
            if schema_mapping:
                relevant_tables = []
                if isinstance(schema_mapping, dict):
                    relevant_tables = schema_mapping.get("relevant_tables", [])
                elif hasattr(schema_mapping, "relevant_tables"):
                    relevant_tables = schema_mapping.relevant_tables
                
                # 관련 테이블들에서 크리에이터 컬럼 찾기
                for table_name in relevant_tables:
                    if table_name in self.db_schema:
                        table_cols = self.db_schema[table_name].get("columns", {})
                        for candidate in candidates:
                            if candidate in table_cols:
                                self.logger.debug(f"Found creator column '{candidate}' in table '{table_name}' from schema_mapping")
                                return candidate
        
        # 2. state에서 사용된 SQL 쿼리의 테이블 확인
        if state:
            sql_query = state.get("sql_query")
            if sql_query:
                from core.db import extract_table_names
                used_tables = extract_table_names(sql_query)
                for table_name in used_tables:
                    if table_name in self.db_schema:
                        table_cols = self.db_schema[table_name].get("columns", {})
                        for candidate in candidates:
                            if candidate in table_cols:
                                self.logger.debug(f"Found creator column '{candidate}' in table '{table_name}' from SQL query")
                                return candidate
        
        # 3. 일반적으로 크리에이터 관련 테이블들에서 찾기
        # data_dictionary.md 기준: t_fanding, t_tier, t_creator, t_payment 등
        creator_related_tables = [
            "t_fanding", "t_tier", "t_creator", "t_payment", 
            "t_event", "t_follow", "t_creator_coupon"
        ]
        
        for table_name in creator_related_tables:
            if table_name in self.db_schema:
                table_cols = self.db_schema[table_name].get("columns", {})
                for candidate in candidates:
                    if candidate in table_cols:
                        self.logger.debug(f"Found creator column '{candidate}' in common creator-related table '{table_name}'")
                        return candidate
        
        # 4. db_schema가 비어있으면 다시 로드 시도
        if not self.db_schema or len(self.db_schema) == 0:
            from core.db import get_cached_db_schema
            self.db_schema = get_cached_db_schema()
            self.logger.debug("db_schema was empty, reloaded from cache in _guess_creator_column")
        
        # 5. 모든 테이블에서 크리에이터 컬럼 검색 (fallback)
        for table_name, table_info in self.db_schema.items():
            table_cols = table_info.get("columns", {})
            for candidate in candidates:
                if candidate in table_cols:
                    self.logger.debug(f"Found creator column '{candidate}' in table '{table_name}' (fallback search)")
                    return candidate
        
        self.logger.warning("No creator column found in database schema")
        return None
    
    def _find_creator_column_in_sql_template(self, sql_template: str, state: Optional[GraphState] = None) -> Optional[str]:
        """
        SQL 템플릿에서 사용하는 테이블을 기반으로 db_schema에서 creator 컬럼 찾기
        
        Args:
            sql_template: SQL 템플릿 문자열
            state: GraphState (선택적)
            
        Returns:
            테이블 alias를 포함한 creator 컬럼명 (예: 'f.creator_no') 또는 None
        """
        if not self.db_schema or len(self.db_schema) == 0:
            from core.db import get_cached_db_schema
            self.db_schema = get_cached_db_schema()
            self.logger.debug("db_schema was empty, reloaded from cache in _find_creator_column_in_sql_template")
        
        # SQL 템플릿에서 사용하는 테이블과 alias 추출
        import re
        
        # FROM 절에서 테이블명과 alias 추출 (개선: alias가 없는 경우도 처리)
        # 패턴 1: "FROM t_member_info m" (alias 있음)
        from_pattern_with_alias = r'FROM\s+(\w+)\s+(\w+)'
        # 패턴 2: "FROM t_member_info" (alias 없음)
        from_pattern_no_alias = r'FROM\s+(\w+)(?:\s+WHERE|\s+$)'
        from_matches = re.findall(from_pattern_with_alias, sql_template, re.IGNORECASE)
        from_matches_no_alias = re.findall(from_pattern_no_alias, sql_template, re.IGNORECASE)
        
        # JOIN 절에서도 테이블명과 alias 추출
        join_pattern = r'JOIN\s+(\w+)\s+(\w+)'
        join_matches = re.findall(join_pattern, sql_template, re.IGNORECASE)
        
        all_tables = {}
        for table_name, alias in from_matches + join_matches:
            all_tables[alias] = table_name
        
        # alias가 없는 경우: 테이블명을 alias로 사용
        for table_name in from_matches_no_alias:
            if table_name not in all_tables.values():
                all_tables[table_name] = table_name  # 테이블명을 alias로 사용
        
        # creator 컬럼 후보들
        creator_candidates = ['creator_no', 'creator_id', 'seller_creator_no']
        
        # 각 테이블에서 creator 컬럼 찾기
        for alias, table_name in all_tables.items():
            if table_name in self.db_schema:
                table_cols = self.db_schema[table_name].get("columns", {})
                for candidate in creator_candidates:
                    if candidate in table_cols:
                        self.logger.debug(f"Found creator column '{candidate}' in table '{table_name}' (alias: '{alias}')")
                        return f"{alias}.{candidate}"
        
        # t_fanding이 사용되는 경우 (가장 일반적)
        if 't_fanding' in [t for t in all_tables.values()]:
            fanding_alias = [alias for alias, table in all_tables.items() if table == 't_fanding'][0]
            if 't_fanding' in self.db_schema:
                fanding_cols = self.db_schema['t_fanding'].get("columns", {})
                for candidate in creator_candidates:
                    if candidate in fanding_cols:
                        self.logger.debug(f"Found creator column '{candidate}' in t_fanding (alias: '{fanding_alias}')")
                        return f"{fanding_alias}.{candidate}"
        
        self.logger.warning(f"Could not find creator column in SQL template tables: {list(all_tables.values())}")
        return None
    
    def _assess_query_complexity(self, query: str, entities: List[Entity], rag_context: Optional[str] = None) -> str:
        """
        쿼리 복잡성 평가
        
        Args:
            query: 사용자 쿼리
            entities: 추출된 엔티티 리스트
            rag_context: RAG 검색 컨텍스트 (있는 경우)
            
        Returns:
            "simple" 또는 "complex"
        """
        query_lower = query.lower()
        
        # 복잡한 쿼리 지표
        complexity_indicators = [
            "join", "union", "subquery", "서브쿼리", "교집합", "합집합",
            "group by", "having", "order by", "window", "over",
            "분석", "통계", "트렌드", "패턴", "비교", "상관관계",
            "case when", "if", "nullif", "coalesce"
        ]
        
        # 간단한 쿼리 지표
        simplicity_indicators = [
            "count", "sum", "avg", "max", "min",
            "개수", "합계", "평균", "최대", "최소",
            "몇 명", "얼마나", "몇 개"
        ]
        
        # 복잡도 점수 계산
        complexity_score = 0
        simplicity_score = 0
        
        # 복잡도 지표 확인
        for indicator in complexity_indicators:
            if indicator in query_lower:
                complexity_score += 2
        
        # 간단도 지표 확인
        for indicator in simplicity_indicators:
            if indicator in query_lower:
                simplicity_score += 1
        
        # 엔티티 수 확인 (엔티티가 많을수록 복잡할 가능성)
        if len(entities) > 3:
            complexity_score += 1
        
        # RAG 컨텍스트가 있으면 복잡한 쿼리일 가능성 높음
        if rag_context and len(rag_context) > 500:
            complexity_score += 1
        
        # 최종 결정
        if complexity_score >= 2:
            return "complex"
        elif simplicity_score >= 2 and complexity_score == 0:
            return "simple"
        else:
            # 애매한 경우: 기본적으로 simple로 처리 (빠른 경로 우선)
            return "simple"
    
    def _generate_sql_simple(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Few-shot 예제 기반 빠른 SQL 생성 (DynamicSQLGenerator 방식 통합)
        
        Args:
            query: 사용자 쿼리
            
        Returns:
            {"sql_query": str, "confidence": float, "reasoning": str} 또는 None
        """
        try:
            if not self.llm:
                self.logger.warning("LLM not available for simple SQL generation")
                return None
            
            # Few-shot 프롬프트 생성
            formatted_prompt = self.simple_prompt.format(query=query)
            
            # LangChain 메시지 형식으로 변환
            messages = [HumanMessage(content=formatted_prompt)]
            
            # LLM 호출
            self.logger.debug(f"Calling LLM for simple SQL generation: {query[:50]}...")
            response = self.llm.invoke(messages)
            
            if not response:
                self.logger.warning("LLM returned None response for simple SQL generation")
                return None
            
            # JSON 파싱
            result_data = parse_json_response(response, parser=self.json_parser, fallback_extract=True)
            
            if not result_data:
                self.logger.warning("Failed to parse JSON from simple SQL generation response")
                return None
            
            # 결과 추출
            sql_query = str(result_data.get("sql_query", "")).strip()
            confidence = float(result_data.get("confidence", 0.7))
            reasoning = str(result_data.get("reasoning", "Few-shot 예제 기반 생성"))
            
            if not sql_query:
                self.logger.warning("Empty SQL query from simple SQL generation")
                return None
            
            self.logger.info(f"Simple SQL generated (confidence: {confidence:.2f}): {sql_query[:100]}...")
            
            return {
                "sql_query": sql_query,
                "confidence": confidence,
                "reasoning": reasoning
            }
            
        except Exception as e:
            self.logger.error(f"Error in simple SQL generation: {str(e)}", exc_info=True)
            return None
    
    def _generate_sql_complex(self, query: str, rag_context: Optional[str], schema_mapping: Optional[SchemaMapping]) -> Optional[str]:
        """
        RAG 컨텍스트 기반 정확한 SQL 생성 (기존 SQLGeneration 방식)
        
        Args:
            query: 사용자 쿼리
            rag_context: RAG 검색 컨텍스트
            schema_mapping: 스키마 매핑 정보
            
        Returns:
            생성된 SQL 쿼리 문자열 또는 None
        """
        try:
            if not self.llm:
                self.logger.warning("LLM not available for complex SQL generation")
                return None
            
            # 스키마 매핑 정보를 프롬프트 템플릿에 설정
            if schema_mapping:
                relevant_schema = {}
                for table_name in schema_mapping.relevant_tables:
                    if table_name in self.db_schema:
                        relevant_schema[table_name] = self.db_schema[table_name]
                self.prompt_template.set_schema(relevant_schema)
            
            # 프롬프트 생성 (RAG 컨텍스트 통합)
            prompt = self.prompt_template.create_prompt(
                user_query=query,
                include_relevant_examples=True,
                rag_context=rag_context if rag_context else None,
                max_context_length=4000
            )
            
            if rag_context:
                self.logger.debug(
                    f"Generated complex prompt with RAG context (length: {len(prompt)}, "
                    f"RAG context length: {len(rag_context)})"
                )
            
            # LLM 호출
            response = self.llm.invoke(prompt)
            response_content = response.content
            
            # 응답 처리
            if isinstance(response_content, str):
                sql_query = response_content.strip()
            elif isinstance(response_content, list):
                sql_query = " ".join(str(item) for item in response_content).strip()
            else:
                sql_query = str(response_content).strip()
            
            # SQL 추출 (```sql ... ``` 형태에서 추출)
            if "```sql" in sql_query:
                sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql_query:
                sql_query = sql_query.split("```")[1].split("```")[0].strip()
            
            if sql_query:
                self.logger.info(f"Complex SQL generated: {sql_query[:100]}...")
                return sql_query
            else:
                self.logger.warning("Empty SQL response from complex SQL generation")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in complex SQL generation: {str(e)}", exc_info=True)
        return None


class SQLValidationNode(BaseNode):
    """SQL 검증 에이전트 노드 - 생성된 SQL의 구문 및 의미 검증"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # db_schema가 없거나 비어있으면 초기화 시점에 한 번만 로드 (성능 최적화)
        self.db_schema = config.get("db_schema") or {}
        if not self.db_schema or len(self.db_schema) == 0:
            from core.db import get_cached_db_schema
            self.db_schema = get_cached_db_schema()
            self.logger.debug("db_schema was empty in config, loaded from cache during initialization")
        
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
            
            # SIMPLE_AGGREGATION 쿼리는 간소화된 검증 수행 (하지만 중요한 스키마 검증은 수행)
            if intent == QueryIntent.SIMPLE_AGGREGATION:
                self.logger.info("SIMPLE_AGGREGATION: Performing simplified validation (syntax + security + critical schema checks)")
                
                # 기본 구문 검증
                syntax_validation = validate_sql_syntax(sql_query)
                
                # 보안 검증 (필수)
                security_validation = self._validate_security(sql_query)
                
                # 중요한 스키마 검증만 수행 (t_member + ins_datetime 같은 치명적 오류 방지)
                critical_schema_validation = self._validate_critical_schema_issues(sql_query)
                
                # 종합 검증 결과
                is_valid = (syntax_validation["is_valid"] and 
                           security_validation["is_valid"] and 
                           critical_schema_validation["is_valid"])
                
                validation_result = {
                    "is_valid": is_valid,
                    "confidence": syntax_validation.get("confidence", 0.8) if is_valid else 0.5,
                    "syntax_valid": syntax_validation["is_valid"],
                    "schema_valid": critical_schema_validation["is_valid"],
                    "security_valid": security_validation["is_valid"],
                    "message": "Simplified validation for SIMPLE_AGGREGATION (with critical schema checks)",
                    "simplified": True,
                    "corrections": critical_schema_validation.get("corrections", [])
                }
                
                # 스키마 수정이 필요한 경우 자동 적용
                if not critical_schema_validation["is_valid"] and "corrections" in critical_schema_validation:
                    corrected_sql = self._apply_schema_corrections(sql_query, critical_schema_validation["corrections"])
                    if corrected_sql != sql_query:
                        self.logger.info(f"SQL auto-corrected for SIMPLE_AGGREGATION: {sql_query[:100]}... -> {corrected_sql[:100]}...")
                        state["sql_query"] = corrected_sql
                        state["sql_corrected"] = corrected_sql
                        # 수정 후 다시 검증
                        validation_result["is_valid"] = True
                        validation_result["schema_valid"] = True
                        is_valid = True
                
                state["validation_result"] = validation_result
                state["is_valid"] = is_valid
                
                if not is_valid:
                    errors = []
                    if not syntax_validation["is_valid"]:
                        errors.extend(syntax_validation.get("errors", []))
                    if not security_validation["is_valid"]:
                        errors.extend(security_validation.get("errors", []))
                    if not critical_schema_validation["is_valid"]:
                        errors.extend(critical_schema_validation.get("issues", []))
                    state["error_message"] = "; ".join(errors)
                
                return state
            
            # COMPLEX_ANALYSIS 및 기타 쿼리는 전체 검증 수행
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
                    state["sql_corrected"] = corrected_sql
            
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
    
    def _validate_critical_schema_issues(self, sql_query: str) -> Dict[str, Any]:
        """
        SIMPLE_AGGREGATION을 위한 중요한 스키마 문제만 검증
        (t_member + ins_datetime 같은 치명적 오류 방지)
        """
        issues = []
        corrections = []
        
        # t_member 테이블에 ins_datetime 컬럼이 없는 경우 검증
        if 't_member' in sql_query and 'ins_datetime' in sql_query:
            # t_member 테이블 스키마 확인
            member_table = self.db_schema.get('t_member', {})
            member_columns = member_table.get('columns', {})
            
            if 'ins_datetime' not in member_columns:
                issues.append("Column 'ins_datetime' not found in t_member table")
                corrections.append("Use t_member_info table instead of t_member for ins_datetime column")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "corrections": corrections,
            "message": "Critical schema validation for SIMPLE_AGGREGATION"
        }
    
    def _apply_schema_corrections(self, sql_query: str, corrections: List[str]) -> str:
        """스키마 수정사항을 SQL에 적용"""
        corrected_sql = sql_query
        
        for correction in corrections:
            # t_member 테이블에 ins_datetime이 없는 경우 t_member_info로 테이블 변경
            # 단, t_fanding을 사용하는 경우에는 변경하지 않음 (t_fanding에 ins_datetime과 creator_no가 모두 있음)
            if "Use t_member_info table instead of t_member for ins_datetime column" in correction:
                # t_fanding을 사용하는 경우 변경하지 않음
                if 't_fanding' not in corrected_sql.upper():
                    # t_member를 t_member_info로 교체 (ins_datetime 컬럼 사용 시)
                    if 'ins_datetime' in sql_query:
                        corrected_sql = re.sub(r'\bt_member\b', 't_member_info', corrected_sql, flags=re.IGNORECASE)
                        self.logger.info("Replaced 't_member' with 't_member_info' for ins_datetime column")
                else:
                    self.logger.debug("Skipping t_member -> t_member_info replacement (t_fanding table is being used)")
            elif "Replace 'ins_datetime' with" in correction:
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
        
        # LLM 서비스에서 SQL LLM 가져오기 (요약용으로도 사용)
        self.llm = self._get_sql_llm()
        
    
    def process(self, state: GraphState) -> GraphState:
        """SQL 실행 결과를 자연어로 요약"""
        self._log_processing(state, "DataSummarizationNode")
        
        try:
            # 비데이터 의도 (CHAT_PATH) 처리: 봇 기능에 맞춘 응답 생성
            intent = state.get("intent")
            user_query = state.get("user_query", "")
            conversation_response = state.get("conversation_response")
            
            self.logger.info(f"DataSummarizationNode - intent: {intent}")
            self.logger.info(f"DataSummarizationNode - conversation_response: {conversation_response is not None}")
            
            # 이미 conversation_response가 있으면 사용 (명확화 질문 등)
            if conversation_response:
                self.logger.info("Using existing conversation_response")
                state["data_summary"] = conversation_response
                state["success"] = True
                return state
            
            # 비데이터 의도에 대한 맥락을 고려한 개인화된 응답 생성
            if intent in ["GREETING", "GENERAL_CHAT", "HELP_REQUEST"]:
                try:
                    intent_enum = QueryIntent(intent) if isinstance(intent, str) else intent
                    
                    # 대화 히스토리 가져오기
                    conversation_history = state.get("conversation_history", [])
                    
                    # LLM을 사용하여 맥락을 고려한 개인화된 응답 생성
                    # 히스토리가 있으면 맥락을 활용하고, 없어도 LLM으로 더 자연스러운 응답 생성
                    if self.llm:
                        response = self._generate_contextual_response(
                            intent_enum, user_query, conversation_history
                        )
                    else:
                        # LLM이 없는 경우 템플릿 응답 사용
                        response = self._generate_template_response(intent_enum, user_query)
                    
                    self.logger.info(f"Generated conversation response for {intent}: {response[:50]}...")
                    state["data_summary"] = response
                    state["conversation_response"] = response
                    
                    # Update conversation history in state so LangGraph saves it to checkpointer
                    self._update_conversation_history_in_state(state, user_query, response)
                    
                    state["success"] = True
                    return state
                    
                except Exception as e:
                    self.logger.warning(f"Error generating conversation response: {e}, using fallback")
                    state["data_summary"] = "안녕하세요! 무엇을 도와드릴까요? 📊"
                    state["success"] = True
                    return state
            
            # 데이터 의도인 경우 기존 로직 계속 수행
            fanding_template = state.get("fanding_template")
            
            # Fanding 템플릿이 있는 경우 특별 처리
            if fanding_template:
                query_result = state.get("query_result")
                if query_result:
                    # Fanding 템플릿 결과 포맷팅
                    # db_schema는 config에서 가져오거나 없으면 자동 로드
                    db_schema = state.get("db_schema") or getattr(self, 'db_schema', None)
                    templates = FandingSQLTemplates(db_schema=db_schema)
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
            
            # Set default values (insight analyzer is disabled)
            state["insight_report"] = None
            state["business_insights"] = None
            
            # 요약 생성
            if self.llm:
                summary = self._generate_ai_summary(user_query, query_result, result_stats)
            else:
                summary = self._generate_fallback_summary(query_result, result_stats)
            
            state["data_summary"] = summary
            
            # Update conversation history in state so LangGraph saves it to checkpointer
            self._update_conversation_history_in_state(state, user_query, summary)
            
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
    
    def _generate_contextual_response(
        self, 
        intent: QueryIntent, 
        user_query: str, 
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        대화 히스토리를 활용하여 맥락을 고려한 개인화된 응답 생성
        
        Args:
            intent: 인텐트 타입
            user_query: 현재 사용자 쿼리
            conversation_history: 이전 대화 히스토리 [{"role": "user|assistant", "content": "..."}]
            
        Returns:
            맥락을 고려한 응답 문자열
        """
        try:
            # 히스토리에서 중요한 정보 추출 (이름 등)
            user_name = self._extract_user_name(conversation_history)
            
            # 프롬프트 구성
            history_context = ""
            if conversation_history:
                history_context = "\n\n[이전 대화 히스토리]\n"
                for msg in conversation_history[-5:]:  # 최근 5개 메시지
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "user":
                        history_context += f"사용자: {content}\n"
                    elif role == "assistant":
                        history_context += f"봇: {content}\n"
            
            # 인텐트별 시스템 프롬프트
            if intent == QueryIntent.GREETING:
                system_prompt = """당신은 Fanding 데이터 조회 봇입니다.
사용자가 인사를 건넸을 때, 이전 대화 히스토리를 참고하여 자연스럽고 친근하게 응답하세요.
"""
                if history_context:
                    system_prompt += history_context
                if user_name:
                    system_prompt += f"\n중요: 이전 대화에서 사용자의 이름이 '{user_name}'으로 언급되었습니다. 이 이름을 사용하여 개인화된 인사를 하세요."
                else:
                    system_prompt += "\n친근하고 자연스러운 인사말로 응답하세요."
                
                user_prompt = f"""현재 사용자 메시지: {user_query}

위 히스토리를 참고하여 자연스럽고 친근한 인사 응답을 생성하세요.
이름이 언급되었다면 반드시 이름을 사용하세요."""
            
            elif intent == QueryIntent.GENERAL_CHAT:
                system_prompt = """당신은 Fanding 데이터 조회 봇입니다.
사용자와의 일반적인 대화에서, 이전 대화 히스토리를 참고하여 맥락에 맞는 응답을 하세요.
특히 사용자가 이전에 언급한 정보(이름, 선호사항 등)를 기억하고 활용하세요.
"""
                if history_context:
                    system_prompt += history_context
                if user_name:
                    system_prompt += f"\n중요: 사용자의 이름은 '{user_name}'입니다. 질문에 이름이 포함되면 이를 활용하세요."
                
                user_prompt = f"""현재 사용자 메시지: {user_query}

위 히스토리를 참고하여 맥락에 맞는 응답을 생성하세요.
사용자가 이름을 물어보면 이전에 언급된 이름을 알려주세요.
"""
            
            else:  # HELP_REQUEST
                system_prompt = """당신은 Fanding 데이터 조회 봇입니다.
사용자에게 봇의 기능과 사용법을 설명하세요.
"""
                if history_context:
                    system_prompt += history_context
                user_prompt = f"""현재 사용자 메시지: {user_query}

봇의 기능과 사용법을 친절하게 설명하세요."""
            
            # LLM 호출
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # 응답 추출
            if hasattr(response, 'content'):
                response_text = response.content
                if isinstance(response_text, str):
                    return response_text.strip()
                elif isinstance(response_text, list):
                    return " ".join(str(item) for item in response_text).strip()
                else:
                    return str(response_text).strip()
            else:
                return str(response).strip()
                
        except Exception as e:
            self.logger.warning(f"Error generating contextual response: {e}, using template")
            return self._generate_template_response(intent, user_query)
    
    def _update_conversation_history_in_state(
        self, 
        state: GraphState, 
        user_query: str, 
        assistant_response: str
    ) -> None:
        """
        Update conversation history in state.
        
        This ensures that LangGraph automatically saves the updated history to checkpointer.
        
        Args:
            state: Current graph state
            user_query: Current user query
            assistant_response: Assistant's response
        """
        try:
            # Get existing history or initialize empty list
            history = state.get("conversation_history", [])
            if not isinstance(history, list):
                history = []
            
            # Add current user query
            history.append({
                "role": "user",
                "content": user_query
            })
            
            # Add assistant response
            history.append({
                "role": "assistant",
                "content": assistant_response
            })
            
            # Limit history size to prevent token overflow
            max_history = 20  # Keep last 20 messages (10 user + 10 assistant)
            if len(history) > max_history:
                history = history[-max_history:]
            
            # Update state with updated history
            state["conversation_history"] = history
            
            self.logger.debug(f"Updated conversation history: {len(history)} messages total")
            
        except Exception as e:
            self.logger.warning(f"Failed to update conversation history in state: {str(e)}")
            # Non-critical error, continue execution
    
    def _extract_user_name(self, conversation_history: List[Dict[str, str]]) -> Optional[str]:
        """
        대화 히스토리에서 사용자 이름 추출
        
        Args:
            conversation_history: 대화 히스토리
            
        Returns:
            추출된 이름 또는 None
        """
        # 이름 언급 패턴 (한국어/영어)
        name_patterns = [
            r"내 이름은\s+([가-힣a-zA-Z]+)",
            r"내 이름이\s+([가-힣a-zA-Z]+)",
            r"제 이름은\s+([가-힣a-zA-Z]+)",
            r"제 이름이\s+([가-힣a-zA-Z]+)",
            r"나는\s+([가-힣a-zA-Z]+)",
            r"저는\s+([가-힣a-zA-Z]+)",
            r"([가-힣]{2,4})라고\s*(?:해|합니다)",
            r"([가-힣]{2,4})라고\s*(?:불러|부르)",
        ]
        
        # 최근 메시지부터 역순으로 검색
        for msg in reversed(conversation_history):
            if msg.get("role") != "user":
                continue
                
            content = msg.get("content", "")
            if not content:
                continue
            
            for pattern in name_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    name = match.group(1).strip()
                    # 너무 짧거나 긴 이름 제외
                    if 2 <= len(name) <= 20:
                        self.logger.info(f"Extracted user name from history: {name}")
                        return name
        
        return None
    
    def _generate_template_response(self, intent: QueryIntent, user_query: str) -> str:
        """
        템플릿 기반 응답 생성 (fallback)
        
        Args:
            intent: 인텐트 타입
            user_query: 사용자 쿼리
            
        Returns:
            템플릿 응답 문자열
        """
        if intent == QueryIntent.GREETING:
            return generate_greeting_response(user_query)
        elif intent == QueryIntent.HELP_REQUEST:
            return generate_help_response(user_query)
        elif intent == QueryIntent.GENERAL_CHAT:
            return generate_general_chat_response(user_query)
        else:
            return "무엇을 도와드릴까요? 📊"
    
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

