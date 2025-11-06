"""
Text-to-SQL 프롬프팅 템플릿 모듈

자연어 쿼리를 SQL로 변환하기 위한 프롬프트 템플릿과 few-shot 예제 관리 기능을 제공합니다.
또한 대화형 응답을 위한 템플릿과 패턴을 제공합니다.
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SQLExample:
    """SQL 변환 예제를 나타내는 데이터 클래스"""
    question: str
    sql: str
    description: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[str] = "medium"  # easy, medium, hard
    tags: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "question": self.question,
            "sql": self.sql,
            "description": self.description,
            "category": self.category,
            "difficulty": self.difficulty,
            "tags": self.tags or []
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SQLExample':
        """딕셔너리에서 생성"""
        return cls(
            question=data["question"],
            sql=data["sql"],
            description=data.get("description"),
            category=data.get("category"),
            difficulty=data.get("difficulty", "medium"),
            tags=data.get("tags", [])
        )


class SQLPromptTemplate:
    """
    Text-to-SQL 변환을 위한 프롬프트 템플릿 클래스
    
    데이터베이스 스키마 정보와 few-shot 예제를 활용하여
    자연어 쿼리를 SQL로 변환하기 위한 프롬프트를 생성합니다.
    """
    
    def __init__(self, 
                 db_schema: Optional[Dict[str, Any]] = None,
                 examples: Optional[List[SQLExample]] = None,
                 max_examples: int = 5,
                 fanding_templates: Optional[Any] = None):
        """
        SQLPromptTemplate 초기화
        
        Args:
            db_schema: 데이터베이스 스키마 정보
            examples: few-shot 예제 리스트
            max_examples: 프롬프트에 포함할 최대 예제 수
            fanding_templates: FandingSQLTemplates 인스턴스 (템플릿을 예제로 변환하기 위해)
        """
        self.db_schema = db_schema or {}
        self.examples = examples or []
        self.max_examples = max_examples
        self.logger = get_logger(self.__class__.__name__)
        
        # 예제 로드 (Fanding 템플릿 우선, 없으면 기본 예제 사용)
        if not self.examples:
            self._load_examples(fanding_templates)
    
    def _load_examples(self, fanding_templates: Optional[Any] = None):
        """
        예제 로드 (통합 메서드)
        
        우선순위:
        1. Fanding 템플릿 예제 (실제 사용되는 템플릿)
        2. 기본 예제 (Fanding 템플릿이 없거나 부족한 경우 fallback)
        
        Args:
            fanding_templates: FandingSQLTemplates 인스턴스 (선택사항)
        """
        loaded_count = 0
        
        # 1. Fanding 템플릿 예제 우선 로드
        if fanding_templates:
            try:
                # fanding_templates에서 예제 변환 (최대 max_examples만큼)
                template_examples = fanding_templates.to_sql_examples(max_examples=self.max_examples)
                
                # 딕셔너리를 SQLExample로 변환
                for example_dict in template_examples:
                    try:
                        sql_example = SQLExample(
                            question=example_dict.get("question", ""),
                            sql=example_dict.get("sql", ""),
                            description=example_dict.get("description"),
                            category=example_dict.get("category"),
                            difficulty=example_dict.get("difficulty", "medium"),
                            tags=example_dict.get("tags", [])
                        )
                        self.examples.append(sql_example)
                        loaded_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to convert template example to SQLExample: {e}")
                        continue
                
                if loaded_count > 0:
                    self.logger.info(f"Loaded {loaded_count} Fanding template examples")
            except Exception as e:
                self.logger.warning(f"Failed to load Fanding template examples: {e}")
        
        # 2. Fanding 템플릿 예제가 없거나 부족한 경우 기본 예제 로드 (fallback)
        if loaded_count < self.max_examples:
            default_examples = self._get_default_examples()
            # 필요한 만큼만 추가
            remaining = self.max_examples - loaded_count
            self.examples.extend(default_examples[:remaining])
            self.logger.info(f"Loaded {min(len(default_examples), remaining)} default examples as fallback")
    
    def _get_default_examples(self) -> List[SQLExample]:
        """
        기본 few-shot 예제 반환 (Fanding 템플릿이 없을 때 fallback)
        
        Returns:
            기본 SQLExample 리스트
        """
        return [
            SQLExample(
                question="모든 회원의 이메일과 닉네임을 보여줘",
                sql="SELECT email, nickname FROM t_member_info;",
                category="basic_select",
                difficulty="easy",
                tags=["select", "member"]
            ),
            SQLExample(
                question="활성 상태인 회원 수는 몇 명인가?",
                sql="SELECT COUNT(*) FROM t_member_info WHERE status = 'A';",
                category="aggregation",
                difficulty="medium",
                tags=["count", "where", "status"]
            ),
            SQLExample(
                question="각 크리에이터별 팬딩 멤버 수를 보여줘",
                sql="SELECT c.no, COUNT(DISTINCT f.member_no) as member_count FROM t_creator c LEFT JOIN t_fanding f ON c.no = f.creator_no GROUP BY c.no;",
                category="join_aggregation",
                difficulty="hard",
                tags=["join", "group_by", "count"]
            ),
            SQLExample(
                question="이번 달 신규 회원 수를 알려줘",
                sql="SELECT COUNT(*) FROM t_member_info WHERE DATE_FORMAT(ins_datetime, '%Y-%m') = DATE_FORMAT(NOW(), '%Y-%m');",
                category="date_filter",
                difficulty="medium",
                tags=["date", "where", "member"]
            ),
            SQLExample(
                question="활성 멤버십이 많은 크리에이터 TOP 5를 보여줘",
                sql="SELECT f.creator_no, COUNT(DISTINCT f.member_no) as active_count FROM t_fanding f WHERE f.fanding_status = 'T' GROUP BY f.creator_no ORDER BY active_count DESC LIMIT 5;",
                category="subquery",
                difficulty="hard",
                tags=["group_by", "order_by", "limit"]
            )
        ]
    
    def set_schema(self, schema: Dict[str, Any]):
        """
        데이터베이스 스키마 설정
        
        Args:
            schema: 데이터베이스 스키마 정보
        """
        self.db_schema = schema
        self.logger.info(f"Schema updated with {len(schema)} tables")
    
    def get_relevant_examples(
        self, 
        query: str, 
        limit: Optional[int] = None,
        relevant_tables: Optional[List[str]] = None
    ) -> List[SQLExample]:
        """
        사용자 쿼리와 관련성이 높은 예제들을 선택
        
        Args:
            query: 사용자 쿼리
            limit: 반환할 최대 예제 수
            relevant_tables: RAG로 검색된 관련 테이블 목록 (선택사항)
            
        Returns:
            관련성이 높은 예제 리스트
        """
        if limit is None:
            limit = self.max_examples
        
        # 간단한 키워드 기반 매칭
        query_lower = query.lower()
        
        scored_examples = []
        for example in self.examples:
            score = 0
            
            # 태그 매칭 점수
            if example.tags:
                for tag in example.tags:
                    if tag.lower() in query_lower:
                        score += 2
            
            # 질문 내용 매칭 점수
            question_words = set(re.findall(r'\w+', example.question.lower()))
            query_words = set(re.findall(r'\w+', query_lower))
            common_words = question_words.intersection(query_words)
            score += len(common_words)
            
            # SQL 키워드 매칭 점수
            sql_lower = example.sql.lower()
            if 'select' in query_lower and 'select' in sql_lower:
                score += 1
            if 'count' in query_lower and 'count' in sql_lower:
                score += 1
            if 'join' in query_lower and 'join' in sql_lower:
                score += 1
            
            # RAG로 검색된 관련 테이블과 예제의 SQL에 사용된 테이블 매칭 (추가 점수)
            if relevant_tables:
                # 예제 SQL에서 테이블명 추출
                sql_tables = re.findall(r'FROM\s+(?:`)?(t_\w+)(?:`)?', sql_lower, re.IGNORECASE)
                sql_tables.extend(re.findall(r'JOIN\s+(?:`)?(t_\w+)(?:`)?', sql_lower, re.IGNORECASE))
                
                # 관련 테이블과 일치하는 경우 점수 추가
                for table in relevant_tables:
                    if table.lower() in [t.lower() for t in sql_tables]:
                        score += 3  # 테이블 매칭은 높은 가중치
                        self.logger.debug(f"Example matched relevant table: {table}")
            
            scored_examples.append((score, example))
        
        # 점수 순으로 정렬하고 상위 예제들 반환
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        selected_examples = [example for score, example in scored_examples[:limit]]
        
        if relevant_tables:
            self.logger.debug(
                f"Selected {len(selected_examples)} examples based on query and relevant tables: {relevant_tables}"
            )
        
        return selected_examples
    
    def format_schema(self) -> str:
        """
        데이터베이스 스키마를 프롬프트에 적합한 형식으로 변환
        
        Returns:
            포맷된 스키마 문자열
        """
        if not self.db_schema:
            return "데이터베이스 스키마 정보가 없습니다."
        
        schema_text = "📊 데이터베이스 스키마:\n\n"
        
        for table_name, table_info in self.db_schema.items():
            if isinstance(table_info, dict):
                schema_text += f"🔹 테이블: {table_name}\n"
                
                if 'description' in table_info:
                    schema_text += f"   설명: {table_info['description']}\n"
                
                if 'columns' in table_info:
                    schema_text += "   컬럼:\n"
                    for col_info in table_info['columns']:
                        if isinstance(col_info, dict):
                            col_name = col_info.get('name', 'unknown')
                            col_type = col_info.get('type', 'unknown')
                            col_desc = col_info.get('description', '')
                            col_key = col_info.get('key', '')
                            
                            key_info = f" ({col_key})" if col_key else ""
                            desc_info = f" - {col_desc}" if col_desc else ""
                            
                            schema_text += f"     • {col_name}: {col_type}{key_info}{desc_info}\n"
                
                schema_text += "\n"
        
        return schema_text
    
    def format_examples(self, examples: Optional[List[SQLExample]] = None) -> str:
        """
        few-shot 예제를 프롬프트에 적합한 형식으로 변환
        
        Args:
            examples: 포맷할 예제 리스트 (None이면 전체 예제 사용)
            
        Returns:
            포맷된 예제 문자열
        """
        if examples is None:
            examples = self.examples
        
        if not examples:
            return "예제가 없습니다."
        
        examples_text = "📚 예제:\n\n"
        
        for i, example in enumerate(examples[:self.max_examples], 1):
            examples_text += f"예제 {i}:\n"
            examples_text += f"❓ 자연어: {example.question}\n"
            examples_text += f"💻 SQL: {example.sql}\n"
            
            if example.description:
                examples_text += f"📝 설명: {example.description}\n"
            
            examples_text += "\n"
        
        return examples_text
    
    def create_prompt(
        self, 
        user_query: str, 
        include_relevant_examples: bool = True,
        rag_context: Optional[str] = None,
        max_context_length: int = 8000
    ) -> str:
        """
        사용자 쿼리에 대한 최종 프롬프트 생성
        
        Args:
            user_query: 사용자의 자연어 쿼리
            include_relevant_examples: 관련성 높은 예제만 포함할지 여부
            rag_context: RAG로 검색된 관련 스키마 컨텍스트 (선택사항)
            max_context_length: 최대 컨텍스트 길이 (토큰 수 제한 대략적 추정)
            
        Returns:
            완성된 프롬프트 문자열
        """
        # 현재 날짜 정보 가져오기
        from datetime import datetime
        current_date = datetime.now()
        current_year = current_date.year
        current_month = current_date.month
        current_day = current_date.day
        
        # 시스템 프롬프트
        system_prompt = f"""당신은 자연어를 SQL 쿼리로 변환하는 전문가입니다.
다음 데이터베이스 스키마와 예제를 참고하여 사용자의 질문을 정확한 SQL 쿼리로 변환해주세요.

⚠️ 중요 사항:
- 반드시 읽기 전용(SELECT) 쿼리만 생성하세요
- 테이블명과 컬럼명은 정확히 사용하세요
- 복잡한 쿼리의 경우 JOIN과 서브쿼리를 적절히 활용하세요
- 결과는 SQL 쿼리만 반환하세요 (설명 없이)

📅 현재 날짜 정보:
- 현재 연도: {current_year}년
- 현재 월: {current_month}월
- 현재 일: {current_day}일
- 현재 날짜: {current_year}-{current_month:02d}-{current_day:02d}

⚠️ 날짜 처리 규칙:
- 연도가 명시되지 않은 날짜는 **반드시 현재 연도({current_year}년)를 사용**하세요
- 예: "11월 1일" → "{current_year}-11-01"
- 예: "10월" → "{current_year}-10"
- 예: "지난달" → 현재 날짜 기준 이전 달
- 사용자가 명시적으로 다른 연도를 지정한 경우에만 해당 연도를 사용하세요

"""
        
        # RAG 컨텍스트가 있으면 우선 사용 (더 정확한 관련 스키마 정보)
        if rag_context:
            # RAG 컨텍스트를 주요 스키마 섹션으로 사용
            schema_section = f"📊 관련 스키마 정보 (RAG 검색 결과):\n\n{rag_context}\n"
            self.logger.debug("Using RAG context as primary schema source")
        else:
            # 전체 스키마 정보 사용
            schema_section = self.format_schema()
            self.logger.debug("Using full database schema")
        
        # 예제 정보 추가 (RAG 컨텍스트의 테이블 정보를 활용하여 관련 예제 선택)
        if include_relevant_examples:
            # RAG 컨텍스트에서 테이블명 추출하여 예제 선택 개선
            relevant_tables = []
            if rag_context:
                import re
                table_matches = re.findall(r'## (t_\w+)', rag_context)
                relevant_tables = list(set(table_matches))
            
            relevant_examples = self.get_relevant_examples(
                user_query, 
                relevant_tables=relevant_tables
            )
            examples_section = self.format_examples(relevant_examples)
        else:
            examples_section = self.format_examples()
        
        # 사용자 쿼리 섹션
        user_section = f"❓ 사용자 질문: {user_query}\n\n💻 SQL:"
        
        # 최종 프롬프트 조합
        prompt = f"{system_prompt}{schema_section}\n{examples_section}\n{user_section}"
        
        # 컨텍스트 길이 확인 및 경고 (대략적 추정: 한글 1자 ≈ 1-2 토큰)
        estimated_length = len(prompt)
        if estimated_length > max_context_length:
            self.logger.warning(
                f"Prompt length ({estimated_length}) exceeds recommended limit ({max_context_length}). "
                f"Consider reducing context size."
            )
        
        self.logger.debug(f"Generated prompt for query: {user_query[:50]}... (length: {estimated_length})")
        return prompt


# ============================================================================
# Conversation Response Templates and Patterns
# ============================================================================

# Intent Classification Patterns (for LLM prompt context)
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


def generate_greeting_response(user_query: str) -> str:
    """
    인사말에 대한 랜덤 응답 생성
    
    Args:
        user_query: 사용자 쿼리 (현재는 사용되지 않지만 향후 개인화 가능)
        
    Returns:
        인사 응답 문자열
    """
    import random
    return random.choice(GREETING_RESPONSES)


def generate_help_response(user_query: str) -> str:
    """
    도움말 요청에 대한 응답 생성
    
    Args:
        user_query: 사용자 쿼리 (현재는 사용되지 않지만 향후 개인화 가능)
        
    Returns:
        도움말 응답 문자열
    """
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
    """
    일반 대화에 대한 랜덤 응답 생성
    
    Args:
        user_query: 사용자 쿼리 (현재는 사용되지 않지만 향후 개인화 가능)
        
    Returns:
        일반 대화 응답 문자열
    """
    import random
    return random.choice(GENERAL_CHAT_RESPONSES)


def generate_error_response(error: Exception) -> str:
    """
    에러에 대한 사용자 친화적인 응답 생성
    
    Args:
        error: 발생한 예외 객체
        
    Returns:
        에러 응답 문자열
    """
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
    """
    애매한 쿼리에 대한 명확화 질문 생성
    
    Args:
        user_query: 사용자 쿼리
        
    Returns:
        명확화 질문 문자열
    """
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
