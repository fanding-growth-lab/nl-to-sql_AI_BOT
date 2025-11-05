"""
Text-to-SQL 프롬프팅 템플릿 모듈

Gemini-2.5-pro 모델을 활용한 자연어를 SQL로 변환하기 위한 
프롬프트 템플릿과 few-shot 예제 관리 기능을 제공합니다.
"""

import json
import re
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

try:
    import google.genai as genai
    from google.genai import Client
    from google.genai.types import HarmCategory, HarmBlockThreshold
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    Client = None

from core.logging import get_logger
from core.config import get_settings

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


@dataclass
class ColumnInfo:
    """데이터베이스 컬럼 정보를 나타내는 데이터 클래스"""
    name: str
    type: str
    description: Optional[str] = None
    nullable: bool = True
    key: Optional[str] = None  # PRI, MUL, UNI
    default: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "nullable": self.nullable,
            "key": self.key,
            "default": self.default
        }


@dataclass
class TableInfo:
    """데이터베이스 테이블 정보를 나타내는 데이터 클래스"""
    name: str
    columns: List[ColumnInfo]
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "name": self.name,
            "columns": [col.to_dict() for col in self.columns],
            "description": self.description
        }


class SQLPromptTemplate:
    """
    Text-to-SQL 변환을 위한 프롬프트 템플릿 클래스
    
    데이터베이스 스키마 정보와 few-shot 예제를 활용하여
    자연어 쿼리를 SQL로 변환하기 위한 프롬프트를 생성합니다.
    """
    
    def __init__(self, 
                 db_schema: Optional[Dict[str, Any]] = None,
                 examples: Optional[List[SQLExample]] = None,
                 max_examples: int = 5):
        """
        SQLPromptTemplate 초기화
        
        Args:
            db_schema: 데이터베이스 스키마 정보
            examples: few-shot 예제 리스트
            max_examples: 프롬프트에 포함할 최대 예제 수
        """
        self.db_schema = db_schema or {}
        self.examples = examples or []
        self.max_examples = max_examples
        self.logger = get_logger(self.__class__.__name__)
        
        # 기본 예제 로드
        if not self.examples:
            self._load_default_examples()
    
    def _load_default_examples(self):
        """기본 few-shot 예제 로드"""
        default_examples = [
            SQLExample(
                question="모든 회원의 이메일과 닉네임을 보여줘",
                sql="SELECT email, nickname FROM t_member;",
                category="basic_select",
                difficulty="easy",
                tags=["select", "member"]
            ),
            SQLExample(
                question="활성 상태인 회원 수는 몇 명인가?",
                sql="SELECT COUNT(*) FROM t_member WHERE status = 'ACTIVE';",
                category="aggregation",
                difficulty="medium",
                tags=["count", "where", "status"]
            ),
            SQLExample(
                question="각 크리에이터별 프로젝트 수를 보여줘",
                sql="SELECT c.nickname, COUNT(p.id) as project_count FROM t_creator c LEFT JOIN t_project p ON c.id = p.creator_id GROUP BY c.id, c.nickname;",
                category="join_aggregation",
                difficulty="hard",
                tags=["join", "group_by", "count"]
            ),
            SQLExample(
                question="지난 주에 생성된 펀딩 프로젝트의 제목과 목표 금액을 보여줘",
                sql="SELECT title, goal_amount FROM t_funding WHERE ins_datetime >= DATE_SUB(NOW(), INTERVAL 1 WEEK);",
                category="date_filter",
                difficulty="medium",
                tags=["date", "where", "funding"]
            ),
            SQLExample(
                question="평균 펀딩 금액보다 높은 프로젝트들의 정보를 보여줘",
                sql="SELECT * FROM t_funding WHERE goal_amount > (SELECT AVG(goal_amount) FROM t_funding);",
                category="subquery",
                difficulty="hard",
                tags=["subquery", "avg", "funding"]
            )
        ]
        self.examples.extend(default_examples)
        self.logger.info(f"Loaded {len(default_examples)} default examples")
    
    def set_schema(self, schema: Dict[str, Any]):
        """
        데이터베이스 스키마 설정
        
        Args:
            schema: 데이터베이스 스키마 정보
        """
        self.db_schema = schema
        self.logger.info(f"Schema updated with {len(schema)} tables")
    
    def add_example(self, example: SQLExample):
        """
        예제 추가
        
        Args:
            example: 추가할 SQL 예제
        """
        self.examples.append(example)
        self.logger.debug(f"Added example: {example.question[:50]}...")
    
    def remove_example(self, index: int):
        """
        예제 제거
        
        Args:
            index: 제거할 예제의 인덱스
        """
        if 0 <= index < len(self.examples):
            removed = self.examples.pop(index)
            self.logger.debug(f"Removed example: {removed.question[:50]}...")
    
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
        max_context_length: int = 4000
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
        # 시스템 프롬프트
        system_prompt = """당신은 자연어를 SQL 쿼리로 변환하는 전문가입니다.
다음 데이터베이스 스키마와 예제를 참고하여 사용자의 질문을 정확한 SQL 쿼리로 변환해주세요.

⚠️ 중요 사항:
- 반드시 읽기 전용(SELECT) 쿼리만 생성하세요
- 테이블명과 컬럼명은 정확히 사용하세요
- 복잡한 쿼리의 경우 JOIN과 서브쿼리를 적절히 활용하세요
- 결과는 SQL 쿼리만 반환하세요 (설명 없이)

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
    
    def save_examples_to_file(self, file_path: Union[str, Path]):
        """
        예제들을 JSON 파일로 저장
        
        Args:
            file_path: 저장할 파일 경로
        """
        file_path = Path(file_path)
        examples_data = [example.to_dict() for example in self.examples]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(examples_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved {len(self.examples)} examples to {file_path}")
    
    def load_examples_from_file(self, file_path: Union[str, Path]):
        """
        JSON 파일에서 예제들을 로드
        
        Args:
            file_path: 로드할 파일 경로
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            self.logger.warning(f"Examples file not found: {file_path}")
            return
        
        with open(file_path, 'r', encoding='utf-8') as f:
            examples_data = json.load(f)
        
        self.examples = [SQLExample.from_dict(data) for data in examples_data]
        self.logger.info(f"Loaded {len(self.examples)} examples from {file_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        템플릿 통계 정보 반환
        
        Returns:
            통계 정보 딕셔너리
        """
        if not self.examples:
            return {"total_examples": 0}
        
        categories = {}
        difficulties = {}
        tags = {}
        
        for example in self.examples:
            # 카테고리 통계
            if example.category:
                categories[example.category] = categories.get(example.category, 0) + 1
            
            # 난이도 통계
            difficulties[example.difficulty] = difficulties.get(example.difficulty, 0) + 1
            
            # 태그 통계
            if example.tags:
                for tag in example.tags:
                    tags[tag] = tags.get(tag, 0) + 1
        
        return {
            "total_examples": len(self.examples),
            "total_tables": len(self.db_schema),
            "categories": categories,
            "difficulties": difficulties,
            "top_tags": dict(sorted(tags.items(), key=lambda x: x[1], reverse=True)[:10])
        }


class GeminiSQLGenerator:
    """
    Gemini-2.5-pro 모델을 활용한 SQL 생성기
    
    자연어 쿼리를 SQL로 변환하며, 데이터베이스 스키마와 
    few-shot 예제를 활용하여 정확한 SQL을 생성합니다.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-2.5-pro",
                 temperature: float = 0.1,
                 max_output_tokens: int = 2048):
        """
        GeminiSQLGenerator 초기화
        
        Args:
            api_key: Google AI API 키 (None이면 설정에서 가져옴)
            model_name: 사용할 모델 이름
            temperature: 생성 온도 (0.0-1.0)
            max_output_tokens: 최대 출력 토큰 수
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.logger = get_logger(self.__class__.__name__)
        
        # API 키 설정
        if api_key:
            self.api_key = api_key
            self.logger.debug(f"API key provided directly: {api_key[:10]}...")
        else:
            # 캐시 클리어 후 설정 가져오기
            get_settings.cache_clear()
            settings = get_settings()
            self.api_key = settings.llm.api_key
            self.logger.debug(f"API key from settings: {self.api_key[:10] if self.api_key else 'None'}...")
        
        # 모델 초기화
        self.model = None
        self.prompt_template = SQLPromptTemplate()
        
        if GOOGLE_AI_AVAILABLE and self.api_key:
            try:
                self._initialize_model()
            except Exception as e:
                self.logger.error(f"Failed to initialize Gemini model: {e}")
                self.model = None
        else:
            if not GOOGLE_AI_AVAILABLE:
                self.logger.warning("Google AI library not available")
            if not self.api_key:
                self.logger.warning(f"No API key provided. API key value: '{self.api_key}'")
    
    def _initialize_model(self):
        """Gemini 모델 초기화"""
        try:
            # 최신 Client API 사용
            self.client = genai.Client(api_key=self.api_key)
            
            # 모델 설정 저장
            self.generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "top_p": 0.8,
                "top_k": 40
            }
            self.safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            self.logger.info(f"Initialized Gemini client: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    def set_schema(self, schema: Dict[str, Any]):
        """
        데이터베이스 스키마 설정
        
        Args:
            schema: 데이터베이스 스키마 정보
        """
        self.prompt_template.set_schema(schema)
        self.logger.info(f"Schema updated for {len(schema)} tables")
    
    def add_example(self, example: SQLExample):
        """
        SQL 예제 추가
        
        Args:
            example: 추가할 SQL 예제
        """
        self.prompt_template.add_example(example)
    
    def load_examples_from_file(self, file_path: Union[str, Path]):
        """
        파일에서 예제들 로드
        
        Args:
            file_path: 예제 파일 경로
        """
        self.prompt_template.load_examples_from_file(file_path)
    
    async def generate_sql_async(self, query: str) -> Dict[str, Any]:
        """
        자연어 쿼리를 SQL로 변환 (비동기)
        
        Args:
            query: 자연어 쿼리
            
        Returns:
            변환 결과 딕셔너리
        """
        if not self.client:
            return self._generate_mock_sql(query)
        
        try:
            # 프롬프트 생성
            prompt = self.prompt_template.create_prompt(query)
            
            # Gemini API 호출 (최신 Client API)
            response = await self.client.models.generate_content_async(
                model=self.model_name,
                contents=prompt
            )
            
            # SQL 추출
            sql = self._extract_sql(response.text)
            
            return {
                "success": True,
                "sql": sql,
                "original_query": query,
                "model": self.model_name,
                "prompt_length": len(prompt),
                "response_length": len(response.text)
            }
            
        except Exception as e:
            self.logger.error(f"SQL generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "sql": None,
                "original_query": query,
                "model": self.model_name
            }
    
    def generate_sql(self, query: str) -> Dict[str, Any]:
        """
        자연어 쿼리를 SQL로 변환 (동기)
        
        Args:
            query: 자연어 쿼리
            
        Returns:
            변환 결과 딕셔너리
        """
        if not self.client:
            return self._generate_mock_sql(query)
        
        try:
            # 프롬프트 생성
            prompt = self.prompt_template.create_prompt(query)
            
            # Gemini API 호출 (동기, 최신 Client API)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            # SQL 추출
            sql = self._extract_sql(response.text)
            
            return {
                "success": True,
                "sql": sql,
                "original_query": query,
                "model": self.model_name,
                "prompt_length": len(prompt),
                "response_length": len(response.text)
            }
            
        except Exception as e:
            self.logger.error(f"SQL generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "sql": None,
                "original_query": query,
                "model": self.model_name
            }
    
    def _generate_mock_sql(self, query: str) -> Dict[str, Any]:
        """
        모의 SQL 생성 (API가 사용 불가능한 경우)
        
        Args:
            query: 자연어 쿼리
            
        Returns:
            모의 변환 결과
        """
        query_lower = query.lower()
        
        # 간단한 키워드 기반 SQL 생성
        if "회원" in query_lower or "사용자" in query_lower:
            if "수" in query_lower or "개수" in query_lower:
                sql = "SELECT COUNT(*) FROM t_member;"
            elif "목록" in query_lower or "리스트" in query_lower:
                sql = "SELECT email, nickname FROM t_member LIMIT 100;"
            else:
                sql = "SELECT * FROM t_member LIMIT 100;"
        elif "크리에이터" in query_lower:
            sql = "SELECT nickname, description FROM t_creator LIMIT 100;"
        elif "펀딩" in query_lower or "프로젝트" in query_lower:
            sql = "SELECT title, goal_amount, current_amount FROM t_funding LIMIT 100;"
        elif "커뮤니티" in query_lower:
            sql = "SELECT title, content FROM t_community LIMIT 100;"
        else:
            sql = "SELECT 1 as placeholder;"
        
        return {
            "success": True,
            "sql": sql,
            "original_query": query,
            "model": "mock",
            "prompt_length": 0,
            "response_length": 0,
            "mock": True
        }
    
    def _extract_sql(self, response_text: str) -> str:
        """
        응답 텍스트에서 SQL 쿼리 추출
        
        Args:
            response_text: 모델 응답 텍스트
            
        Returns:
            추출된 SQL 쿼리
        """
        if not response_text:
            return "SELECT 1 as placeholder;"
        
        # 코드 블록에서 SQL 추출
        code_block_pattern = r'```(?:sql)?\s*(.*?)\s*```'
        code_match = re.search(code_block_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        if code_match:
            sql = code_match.group(1).strip()
        else:
            # SQL 키워드로 시작하는 라인 찾기
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if re.match(r'^\s*(SELECT|WITH|INSERT|UPDATE|DELETE)', line, re.IGNORECASE):
                    sql = line
                    break
            else:
                # 첫 번째 라인 사용
                sql = lines[0].strip() if lines else response_text.strip()
        
        # SQL 정리
        sql = sql.strip()
        if not sql.endswith(';'):
            sql += ';'
        
        return sql
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 반환
        
        Returns:
            모델 정보 딕셔너리
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "model_available": self.model is not None,
            "api_key_configured": bool(self.api_key),
            "google_ai_available": GOOGLE_AI_AVAILABLE,
            "examples_count": len(self.prompt_template.examples),
            "schema_tables": len(self.prompt_template.db_schema)
        }
    
    def validate_sql(self, sql: str) -> Dict[str, Any]:
        """
        SQL 쿼리 유효성 검증
        
        Args:
            sql: 검증할 SQL 쿼리
            
        Returns:
            검증 결과
        """
        issues = []
        
        # 읽기 전용 검증
        sql_upper = sql.strip().upper()
        dangerous_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE']
        
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                issues.append(f"위험한 키워드 발견: {keyword}")
        
        # 기본 SQL 구조 검증
        if not sql_upper.startswith('SELECT') and not sql_upper.startswith('WITH'):
            issues.append("SELECT 또는 WITH로 시작해야 합니다")
        
        # 세미콜론 검증
        if not sql.endswith(';'):
            issues.append("세미콜론(;)으로 끝나야 합니다")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "sql": sql
        }


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
