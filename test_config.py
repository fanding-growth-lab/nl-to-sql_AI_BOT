import os
import re
import json
import pandas as pd
import io # Add io import
from dotenv import load_dotenv
load_dotenv()


# === SQL CONFIG ===
from sqlalchemy import create_engine, text

DB_USER = os.getenv("DB_USERNAME")
DB_PASS = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")      # 또는 IP 주소
DB_PORT = 3306                      # MariaDB 기본 포트
DB_NAME = os.getenv("DB_DATABASE")
DB_CHARSET = 'utf8mb4'
SQL_LIMIT = 1000

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset={DB_CHARSET}",
    echo=False  # SQL 로그를 보고 싶으면 True
)
with engine.connect() as conn:
    result = conn.execute(text("SELECT 1"))
    assert result.scalar()


# === GEMINI CONFIG ===
from langchain_google_genai import ChatGoogleGenerativeAI

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LangChain 모델 초기화
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)


# === GLOBAL FUNCTIONS ===
def save_result(result, output_filename: str, verbose=False):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, output_filename)
    if file_path.endswith(".json"):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(result)
    if verbose:
        print(f"{output_dir}/{output_filename}에 아래 내용 저장.")
        print(result)
    return file_path

import sys
from io import StringIO

def run_dynamic_code(code: str, context: dict = None):
    # Create proper globals context for exec to allow imports
    if context is None:
        context = {'__builtins__': __builtins__}
    else:
        # Ensure __builtins__ is always available
        if '__builtins__' not in context:
            context['__builtins__'] = __builtins__
    
    local_env = {}
    
    # Redirect stdout to capture print statements
    old_stdout = sys.stdout
    redirected_output = StringIO()
    sys.stdout = redirected_output
    
    error = None
    captured_output = None
    try:
        exec(code, context, local_env)
        captured_output = redirected_output.getvalue()
    except Exception as e:
        error = e
    finally:
        sys.stdout = old_stdout # Restore stdout
    
    return {"local_env": local_env, "captured_output": captured_output, "error": error}


# === GLOBAL VARIABLES ===
# Business rules are now handled by rule_rag.py


PROMPT_SEARCH_RELATIVE_TABLES = """
주어진 사용자 질문에 답하기 위해 반드시 필요한 데이터 테이블 목록과 이유를 반환합니다.
관련 테이블의 컬럼은 누락 없이 모두 반환합니다.
출력 형식 외의 다른 설명이나 주석, 텍스트를 붙이지 않습니다.

사용자 질문:
{user_query}

사용할 수 있는 데이터 테이블 목록:
{rag_schema_context}

출력 형식:
```json
[
    {{ "table": <테이블명>, "schema": [{{"column": <컬럼명>, "type": <자료형>, "description": <설명>}}, ...], "reason": <테이블 선택 이유> }},
    ...
]
```
"""

PROMPT_GENERATE_SQL_QUERY = f"""
사용자 질문과 관련 테이블 정보를 바탕으로 필요한 데이터를 추출할 수 있는 SQL 쿼리문을 작성합니다.
출력 형식 외의 다른 설명이나 주석, 텍스트를 붙이지 않습니다.

사용자 질문:
{{user_query}}

관련 테이블 정보:
{{relative_tables}}

비즈니스 규칙:
{{business_rules}}

{{sql_feedback}}

규칙:
- MariaDB 문법을 따른다.
- 사용자 질문에 날짜, 이름, ID 등의 조건이 있다면 WHERE 절에 포함시킨다.
- alias(AS 문법) 사용 시 반드시 백틱(`)으로 감싸 예약어와 혼동되지 않게한다.
- 날짜 조건이 주어지면 앞에 1개월 여유를 두고 조회한다. (예: 8월 신규 가입 여부를 알기 위해선 7월 데이터도 필요) 
- `비즈니스 규칙`을 따른다.

[매우 중요 - Raw Data 추출 원칙]
1. **절대 SQL에서 집계하지 마세요.**
   - 사용자가 "평균", "합계", "수(Count)"를 물어보더라도, SQL에서는 `AVG`, `SUM`, `COUNT`, `GROUP BY`를 사용하지 마세요.
   - 대신, 파이썬이 계산할 수 있도록 **필요한 모든 로우 데이터(Raw Data)**를 `SELECT` 하세요.
   - 예: "월 평균 방문자 수" -> `SELECT ins_datetime, member_no FROM t_post_view_log ...` (O)
   - 예: "월 평균 방문자 수" -> `SELECT AVG(COUNT(*)) ...` (X)

2. **쿼리를 통합하세요 (효율성).**
   - 동일한 테이블(`t_post_view_log` 등)이 여러 지표 계산에 필요하다면, **단 하나의 쿼리**로 통합하여 조회하세요.
   - `WHERE` 절을 포괄적으로 설정하여 한 번에 가져오세요.
   - 예: "10월 방문자 수"와 "10월 상위 포스트"를 물어보면 -> `t_post_view_log`를 10월 전체로 한 번만 조회하세요.

[필수 체크리스트] - 아래 항목을 위반하면 절대 안됩니다.
1. 쿼리에 GROUP BY, SUM, COUNT, AVG, MIN, MAX 등의 집계/통계 함수가 포함되어 있지 않은가? (반드시 예, 로우 데이터만 추출해야 함)
2. 신규 회원/이탈 회원 등을 SQL에서 판단하려고 하지 않았는가? (반드시 예, SQL은 판단 로직 없이 모든 로그를 가져와야 함)
3. 동일한 테이블을 여러 번 조회하는 비효율적인 쿼리를 작성하지 않았는가? (반드시 예, 가능한 한 통합해야 함)

출력 형식:
```json
[ {{{{ "query_name": "<SQL 쿼리명>", "sql": "<SQL 문장>" }}}}, ... ]
```  
"""

PROMPT_VALIDATE_SQL_QUERY = """
생성된 SQL 쿼리문들이 아래의 규칙들을 잘 지키는지 검증하고 종합하여 최종 판단을 내립니다.
출력 형식 외의 다른 설명이나 주석, 텍스트를 붙이지 않습니다.

사용자 질문:
{user_query}

관련 테이블 정보:
{relative_tables}

비즈니스 규칙:
{business_rules}

검증할 SQL 쿼리문:
{sql_queries}

규칙:
1. (비즈니스 규칙 검증) 비즈니스 규칙을 잘 따르고 있는가? (반드시 예)
2. (정합성 검증) 관련 테이블 목록에 명시된 테이블이나 컬럼만 사용했는가? (반드시 예)
3. (정합성 검증) MariaDB에서 정상 동작하는 문법으로 작성했는가? (반드시 예)
4. (날짜 조건 검증) 쿼리 목적에 맞는 올바른 날짜 범위를 사용했는가? (반드시 예)
   - **Block Logic 적용 쿼리** (신규/갱신/이탈 판단용): 1개월 여유를 두고 조회 (예: 8월 분석 시 7월부터)
   - **스냅샷 쿼리** (특정 시점 활성 회원 수): 해당 시점 기준 조건 (예: 월말 23:59:59 기준 활성 상태)
   - **일반 조회 쿼리** (포스트 조회 로그 등): 분석 대상 기간만 조회
5. (로우 데이터 추출 여부 검증) GROUP BY, SUM, COUNT, AVG 등 집계 및 통계 함수를 사용하지 않고 로우 데이터만 추출하였는가? (반드시 예)
   - **중요**: 사용자가 "평균", "합계"를 물어봤더라도 SQL에는 집계 함수가 없어야 합니다.
6. (최적화 검증) 여러 쿼리로 나누지 않고 가능한 하나의 쿼리로 작성하였는가? (반드시 예)
   - 특히 `t_post_view_log`와 같이 무거운 테이블을 여러 번 조회하는 것은 **매우 비효율적**이므로, 한 번의 조회로 필요한 데이터를 모두 가져와야 합니다.
   - 예: "방문자 수"와 "상위 포스트"를 위해 `t_post_view_log`를 각각 조회했다면 **False**입니다.

출력 형식:
```json
{{"is_valid": <true 또는 false>, "feedback": <true인 경우 빈 문자열, false인 경우 검증 결과를 바탕으로 쿼리문을 개선하기 위해 필요한 내용>}}
```
"""

PROMPT_PLAN_PYTHON_ANALYSIS = """
사용자 질문을 해결하기 위한 파이썬 데이터 분석 단계를 절차적으로 계획합니다.
각 단계는 명확하고 실행 가능한 작업 단위여야 합니다.

사용자 질문:
{user_query}

관련 테이블 정보:
{relative_tables}

비즈니스 규칙:
{business_rules}

SQL 쿼리문:
{sql_queries}

**중요 원칙 (반드시 준수)**:
- **한 단계는 하나의 명확한 목적만**: 여러 테이블 조회, 여러 데이터 처리를 하나의 단계에 넣지 마세요
- 각 단계는 독립적으로 실행 가능해야 합니다
- 단계별로 생성된 변수는 다음 단계에서 재사용됩니다
- 에러 발생 시 해당 단계만 재실행되므로, 단계를 작게 나누는 것이 중요합니다

가이드라인:
1. **데이터 로딩은 테이블별로 분리**: 
   - 작가/크리에이터 정보 조회 (1단계)
   - Payment 데이터 조회 (2단계)
   - Membership 데이터 조회 (3단계)
2. **전처리 및 파생변수 생성**: 날짜 변환, 결측치 처리, 필요한 파생 변수(예: Block ID) 생성을 별도 단계로 분리하세요.
   - **중요**: 멤버십 분석 시 "Membership Block Logic" 적용 단계를 반드시 포함하세요.
3. **필터링 및 추출**: 분석 대상 기간이나 조건에 맞는 데이터 필터링 단계를 포함하세요.
4. **집계 및 계산**: GroupBy, Pivot 등을 이용한 집계 단계를 포함하세요.
5. **최종 결과 도출**: 최종적인 숫자나 표를 도출하는 단계를 포함하세요.

출력 형식 예시 (단계를 세분화):
```json
[
    "1. 작가 정보 조회 (nickname으로 creator_no 식별)",
    "2. Payment 데이터 조회 및 날짜 변환",
    "3. Membership(t_fanding_log) 데이터 조회 및 날짜 변환",
    "4. Membership Block Logic 적용하여 Block 생성",
    "5. 월별 활성 회원 필터링 및 신규/기존 구분",
    "6. 월별 매출액 집계",
    "7. 최종 결과 병합 및 출력"
]
```
"""

PROMPT_GENERATE_PYTHON_STEP = f"""
현재 계획된 분석 단계에 맞춰 파이썬 코드를 작성합니다.
이전 단계에서 생성된 변수들을 사용할 수 있습니다.

사용자 질문:
{{user_query}}

비즈니스 규칙:
{{business_rules}}

Python 규칙:
{{python_rules}}

전체 계획:
{{python_plan}}

**현재 진행할 단계**:
{{current_step}}

**이전 단계까지의 실행 컨텍스트 (변수 목록)**:
{{python_context}}

{{step_feedback}}

**❗ 중요: 테이블 스키마 준수 (반드시 확인)**
- python_context에 '_table_schemas' 키가 있습니다. 여기에 모든 테이블의 스키마 정보가 포함되어 있습니다.
- SQL 쿼리 작성 시 **반드시** 해당 스키마에 존재하는 컬럼만 사용하세요.
- 스키마에 없는 컬럼을 사용하면 에러가 발생합니다.
- 예: t_fanding_log에 'member_no' 컬럼이 없다면 절대 사용하지 마세요.

**중요 원칙 (반드시 준수)**:
1. **첫 단계 필수 작업**: 
   - 만약 현재 단계가 **첫 번째 단계(1단계)**라면, 반드시 `sqlalchemy.create_engine`을 사용하여 DB 연결을 설정하세요.
   - `engine` 변수에 할당해야 다음 단계에서 재사용 가능합니다.
   - DB URL: `mysql+pymysql://readonly_user_business_data:Fanding!Data!@epic-readonly.ceind7azkfjy.ap-northeast-2.rds.amazonaws.com:3306/fanding?charset=utf8mb4`
2. **SQL 쿼리 사용**:
   - `python_context['sql_queries']`에 SQL 쿼리들이 리스트로 저장되어 있습니다.
   - 이를 활용하거나, 필요시 직접 쿼리를 작성하세요.
3. **현재 단계만 처리** - 이 단계의 목표만 달성하세요. 다음 단계 작업은 절대 포함하지 마세요.
4. **이전 변수 재사용 (매우 중요)**:
   - `python_context`에 이미 존재하는 변수(DataFrame 등)를 **반드시** 그대로 사용하세요.
   - **절대** 모의 데이터(Mock Data)를 생성하지 마세요. 이전 단계에서 생성된 데이터가 있다고 가정하세요.
   - 변수명을 정확히 확인하세요. 오타(예: `_existing_existing_`)를 주의하세요.
5. **라이브러리 import** - 필요한 라이브러리는 단계 내에서 import 하세요 (pandas as pd, numpy as np 등).
6. **결과 출력** - 최종 결과는 반드시 `print()`로 출력하세요. 중간 과정도 필요시 출력하세요.
7. **Code Example 참고** - 복잡한 로직(Block Logic 등)은 비즈니스/Python 규칙의 [Code Example]을 정확히 따르세요.

출력 형식:
```python
# 코드 작성
```
"""

PROMPT_VALIDATE_PYTHON_STEP = """
방금 실행한 파이썬 코드 단계가 올바르게 수행되었는지 검증합니다.

사용자 질문:
{user_query}

비즈니스 규칙:
{business_rules}

Python 규칙:
{python_rules}

현재 단계:
{current_step}

실행된 코드:
{step_code}

실행 결과:
{step_result}

**검증 기준 (반드시 확인)**:
1. **실행 성공** - 코드가 에러 없이 정상 실행되었는가?
2. **단계 목표 달성** - 현재 단계에서 요구한 작업이 완료되었는가?
3. **비즈니스 규칙 준수** - 비즈니스 규칙(Block Logic, 메트릭 정의 등)을 정확히 따랐는가?
4. **데이터 정합성** - 생성/변환된 데이터가 의도대로 올바르게 처리되었는가? (결과값 확인)

출력 형식:
```json
{{"is_valid": <true 또는 false>, "feedback": <true인 경우 빈 문자열, false인 경우 수정이 필요한 구체적인 피드백>}}
```
"""

PROMPT_GENERATE_FINAL_RESULT = """
분석된 결과를 바탕으로 사용자에게 전달할 최종 답변을 작성합니다.

분석 결과:
{python_execution_result}

오류 메시지 (있는 경우):
{error_message}

작성 가이드:
1. 분석 결과가 성공적이라면, 주요 수치와 인사이트을 요약하여 친절하게 설명하세요.
2. 오류가 있다면, 어떤 문제가 있었는지 간략히 언급하세요.
3. 결과는 마크다운 형식으로 깔끔하게 정리하세요.
"""

STATE = {
    "user_query": "25년 11월 '강환국 작가', '고래돈공부' 크리에이터의 월 성과 및 팬덤 만족도를 분석하고 비교해줘.",
    "user_id": None,
    "channel_id": None,
    "session_id": "ebe64650-26e5-4d0d-bf9a-21b80d0133e2",
    "context": {
        "user_id": None,
        "channel_id": None
    },
    "normalized_query": "25년 11월 '강환국 작가', '고래돈공부' 크리에이터의 월 성과 및 팬덤 만족도를 분석하고 비교해줘.",
    "intent": "COMPLEX_ANALYSIS",
    "llm_intent_result": {
        "intent": "COMPLEX_ANALYSIS",
        "confidence": 0.95,
        "reasoning": "전체 멤버십 가입자 수는 단순 집계이지만, 특정 크리에이터들의 '월 성과 및 팬덤 만족도를 분석하고 비교'"
    },
    "entities": [
        # NOTE: 이전 단계에서 사용할 테이블까지도 뽑아줌 -> 실패 시 테이블 선택 단계까지 돌아가야.
        # NOTE: 기능 테스트를 위해 모두 다 넣음
        {"name": "creator", "type": "table", "confidence": 0.9, "context": None},
        {"name": "creator_coupon", "type": "table", "confidence": 0.9, "context": None},
        {"name": "creator_coupon_member", "type": "table", "confidence": 0.9, "context": None},
        {"name": "creator_department", "type": "table", "confidence": 0.9, "context": None},
        {"name": "creator_department_mapping", "type": "table", "confidence": 0.9, "context": None},
        {"name": "event", "type": "table", "confidence": 0.9, "context": None},
        {"name": "event_member", "type": "table", "confidence": 0.9, "context": None},
        {"name": "fanding", "type": "table", "confidence": 0.9, "context": None},
        {"name": "fanding_log", "type": "table", "confidence": 0.9, "context": None},
        {"name": "fanding_reserve_log", "type": "table", "confidence": 0.9, "context": None},
        {"name": "follow", "type": "table", "confidence": 0.9, "context": None},
        {"name": "member", "type": "table", "confidence": 0.9, "context": None},
        {"name": "member_join_phone_number", "type": "table", "confidence": 0.9, "context": None},
        {"name": "payment", "type": "table", "confidence": 0.9, "context": None},
        {"name": "post", "type": "table", "confidence": 0.9, "context": None},
        {"name": "post_like_log", "type": "table", "confidence": 0.9, "context": None},
        {"name": "post_reply_like_log", "type": "table", "confidence": 0.9, "context": None},
        {"name": "post_view_log", "type": "table", "confidence": 0.9, "context": None},
        {"name": "tier", "type": "table", "confidence": 0.9, "context": None},
        {"name": "statistics", "type": "aggregation", "confidence": 0.8, "context": None}
    ],
    "agent_schema_mapping": None,
    "sql_query": None,
    "validated_sql": None,
    "query_result": [],
    "data_summary": None,
    "skip_sql_generation": False,
    "conversation_response": None,
    "needs_clarification": None,
    "fanding_template": None,
    "validation_result": None,
    "processing_decision": None,
    "is_valid": True,
    "error_message": None,
    "retry_count": 0,
    "max_retries": 3,
    "current_node": "python_code_generation",
    "execution_status": "pending",
    "node_results": [],
    "processing_time": 0.0,
    "execution_time": 0.0,
    "confidence_scores": {
        "nl_processing": 0.8833333333333333,
        "schema_mapping": 1.0
    },
    "debug_info": {
        "node_performance": {"...": "..."},
        "routing_decisions": {"...": "..."}
    },
    "rag_schema_chunks": [
        {"...": "..."},
        {"...": "..."}
    ],
    "rag_schema_context": (
        "## t_creator - 크리에이터 정보 테이블\n\n"
        "**설명:** [여기에 테이블에 대한 설명을 작성해주세요]\n\n"
        "| 컬럼명 | 데이터 타입 | 설명 |\n"
        "| --- | --- | --- |\n"
        "| `no` | `int(10) unsigned` | [크리에이터를 고유하게 식별하는 번호 (PK)] |\n"
        "| `member_no` | `int(10) unsigned` | [해당 크리에이터의 member_no. (즉, 크리에이터도 멤버의 한 종류)] |\n"
        "| `launching_datetime` | `datetime` | [크리에이터 서비스 런칭일] |\n"
        "| `is_active` | `char(1)` | [크리에이터 활성화 여부] |\n"
        "---\n\n"
        "## t_creator_coupon - 크리에이터 쿠폰 테이블\n\n"
        "**설명:** [여기에 테이블에 대한 설명을 작성해주세요]\n\n"
        "| 컬럼명 | 데이터 타입 | 설명 |\n"
        "| --- | --- | --- |\n"
        "| `no` | `int(10) unsigned` | [쿠폰 고유 식별 번호 (PK)] |\n"
        "| `creator_no` | `int(10) unsigned` | [해당 쿠폰을 발행한 크리에이터의 번호 (FK)] |\n"
        "| `name` | `varchar(50)` | [쿠폰 이름] |\n"
        "| `code` | `varchar(17)` | [쿠폰 활성화 코드] |\n"
        "| `duration` | `int(10) unsigned` | [쿠폰 혜택 기간] |\n"
        "| `expiry_end_date` | `date` | [쿠폰 등록 만료일] |\n"
        "| `ins_datetime` | `datetime` | [쿠폰 등록 시작일] |\n"
        "---\n\n"
        "## t_creator_coupon_member\n\n"
        "**설명:** [여기에 테이블에 대한 설명을 작성해주세요]\n\n"
        "| 컬럼명 | 데이터 타입 | 설명 |\n"
        "| --- | --- | --- |\n"
        "| `no` | `int(10) unsigned` | [쿠폰 사용 로그 번호] |\n"
        "| `coupon_no` | `int(10) unsigned` | [쿠폰 번호 (FK to t_creator_coupon.no)] |\n"
        "| `member_no` | `int(10) unsigned` | [쿠폰을 사용한 회원의 멤버 번호 (FK to t_member.no)] |\n"
        "| `ins_datetime` | `datetime` | [쿠폰 사용일] |\n"
        "---\n\n"
        "## t_creator_department\n\n"
        "**설명:** [여기에 테이블에 대한 설명을 작성해주세요]\n\n"
        "| 컬럼명 | 데이터 타입 | 설명 |\n"
        "| --- | --- | --- |\n"
        "| `no` | `int(11) unsigned` | [부서 카테고리를 고유하게 식별하는 번호 (PK)] |\n"
        "| `name` | `varchar(15)` | [부서 카테고리 이름 (한국어, 예: 프로페셔널, 셀러브리티, 보이스 등)] |\n"
        "| `name_eng` | `varchar(30)` | [부서 카테고리 이름 (영어)] |\n"
        "---\n\n"
        "## t_creator_department_mapping\n\n"
        "**설명:** [여기에 테이블에 대한 설명을 작성해주세요]\n\n"
        "| 컬럼명 | 데이터 타입 | 설명 |\n"
        "| --- | --- | --- |\n"
        "| `creator_no` | `int(11) unsigned` | [t_creator 테이블의 no (크리에이터 고유 ID)를 참조 (FK)] |\n"
        "| `department_no` | `int(11) unsigned` | [t_creator_department 테이블의 no (부서 카테고리 ID)를 참조 (FK)] |\n"
        "---\n\n"
        "## t_event\n\n"
        "**설명:** [여기에 테이블에 대한 설명을 작성해주세요]\n\n"
        "| 컬럼명 | 데이터 타입 | 설명 |\n"
        "| --- | --- | --- |\n"
        "| `no` | `int(11) unsigned` | [이벤트 고유 번호 (PK)] |\n"
        "| `creator_no` | `int(11) unsigned` | [이벤트를 생성한 크리에이터의 번호 (FK to t_creator.no)] |\n"
        "| `title` | `varchar(255)` | [이벤트 이름] |\n"
        "| `is_offline` | `char(1)` | [오프라인 이벤트 여부 (t/f)] |\n"
        "| `online_url` | `varchar(300)` | [이벤트 url] |\n"
        "| `address` | `varchar(300)` | [이벤트 개최 주소] |\n"
        "| `detail_address` | `varchar(300)` | [이벤트 개최 상세 주소] |\n"
        "| `latitude` | `decimal(10,8)` | [이벤트 주소 위도] |\n"
        "| `longitude` | `decimal(11,8)` | [이벤트 주소 경도] |\n"
        "| `event_start_time` | `datetime` | [이벤트 시작일] |\n"
        "| `event_end_time` | `datetime` | [이벤트 종료일] |\n"
        "| `selling_start_time` | `datetime` | [이벤트 판매 시작일] |\n"
        "| `selling_end_time` | `datetime` | [이벤트 판매 종료일] |\n"
        "| `is_private` | `char(1)` | [이벤트 공개/비공개 여부] |\n"
        "---\n\n"
        "## t_event_member\n\n"
        "**설명:** [여기에 테이블에 대한 설명을 작성해주세요]\n\n"
        "| 컬럼명 | 데이터 타입 | 설명 |\n"
        "| --- | --- | --- |\n"
        "| `no` | `int(11) unsigned` | [이벤트 신청 로그 번호 (PK)] |\n"
        "| `event_no` | `int(11) unsigned` | [이벤트 번호 (FK to t_event.no)] |\n"
        "| `ticket_no` | `int(11) unsigned` | [이벤트 참석 티켓 번호] |\n"
        "| `member_no` | `int(11) unsigned` | [이벤트 신청자 회원 번호 (FK to t_member.no)] |\n"
        "| `attendance` | `char(1)` | [참석 여부] |\n"
        "| `status` | `char(1)` | [상태] |\n"
        "| `attend_datetime` | `datetime` | [참석 날짜] |\n"
        "| `ins_datetime` | `datetime` | [신청 로그 생성 날짜] |\n"
        "---\n\n"
        "## t_fanding\n\n"
        "**설명:** [멤버십 활성화 정보]\n\n"
        "| 컬럼명 | 데이터 타입 | 설명 |\n"
        "| --- | --- | --- |\n"
        "| `no` | `int(11) unsigned` | [팬딩(멤버십 가입 건) 자체를 고유하게 식별하는 번호 (PK)] |\n"
        "| `current_tier_no` | `int(10) unsigned` | [현재 이용중인 멤버십 번호 (FK to t_tier.no)] |\n"
        "| `current_fanding_log_no` | `int(11) unsigned` | [현재 해당하는 팬딩로그 번호 (FK to t_fanding_log.no)] |\n"
        "| `member_no` | `int(11) unsigned` | [해당 멤버십에 가입한 멤버의 member_no (FK to t_member.no)] |\n"
        "| `creator_no` | `int(11) unsigned` | [해당 멤버십을 제공하는 크리에이터의 creator_no (FK to t_creator.no)] |\n"
        "| `fanding_status` | `char(1)` | [현재 시점의 멤버십 상태 ('T': 가입 중/활성, 'F': 이탈/비활성)] |\n"
        "| `ins_datetime` | `datetime` | [해당 멤버가 이 크리에이터의 멤버십에 최초로 가입한 날짜 및 시간] |\n"
        "---\n\n"
        "## t_fanding_log\n\n"
        "**설명:** [여기에 테이블에 대한 설명을 작성해주세요]\n\n"
        "| 컬럼명 | 데이터 타입 | 설명 |\n"
        "| --- | --- | --- |\n"
        "| `no` | `int(11) unsigned` | [로그 레코드 자체의 고유 ID (PK)] |\n"
        "| `fanding_no` | `int(11) unsigned` | [t_fanding 테이블의 No를 참조 (FK)] |\n"
        "| `edition` | `smallint(5) unsigned` | [멤버십을 몇 번째 구매하고 있는지 나타내는 횟수] |\n"
        "| `period` | `smallint(5) unsigned` | [사용중인 멤버십 상품의 기간 (개월수)] |\n"
        "| `tier_log_no` | `int(10) unsigned` | [사용중인 멤버십 정보 로그 번호 (FK to t_tier_log.no)] |\n"
        "| `currency_no` | `tinyint(3) unsigned` | [통화 구분 (1: 원화, 2: 달러 등)] |\n"
        "| `price` | `decimal(9,2) unsigned` | [해당 멤버십 기간의 가격] |\n"
        "| `heat` | `int(10) unsigned` | [사용된 히트(서비스 내 재화)] |\n"
        "| `coupon_member_no` | `int(10) unsigned` | [사용한 쿠폰 로그 번호 (FK to t_creator_coupon_member.no)] |\n"
        "| `start_date` | `date` | [해당 멤버십 기간의 시작일] |\n"
        "| `end_date` | `date` | [해당 멤버십 기간의 종료일] |\n"
        "---\n\n"
        "## t_fanding_reserve_log\n\n"
        "**설명:** [여기에 테이블에 대한 설명을 작성해주세요]\n\n"
        "| 컬럼명 | 데이터 타입 | 설명 |\n"
        "| --- | --- | --- |\n"
        "| `no` | `int(11) unsigned` | [멤버십 갱신 중단 설정 로그 번호 (PK)] |\n"
        "| `fanding_no` | `int(11) unsigned` | [멤버십 정보 번호] |\n"
        "| `status` | `char(1)` | [갱신 설정 상태 (t=갱신 활성화,f=갱신 비활성화)] |\n"
        "| `tier_no` | `int(10) unsigned` | [이용중인 멤버십 번호 (FK to t_tier.no)] |\n"
        "| `is_complete` | `char(1)` | [갱신 중단 실행 여부] |\n"
        "| `ins_datetime` | `datetime` | [갱신 중단 설정 날짜] |\n"
        "---\n\n"
        "## t_follow\n\n"
        "**설명:** [여기에 테이블에 대한 설명을 작성해주세요]\n\n"
        "| 컬럼명 | 데이터 타입 | 설명 |\n"
        "| --- | --- | --- |\n"
        "| `no` | `int(10) unsigned` | [팔로우 액션의 고유 ID (PK)] |\n"
        "| `creator_no` | `int(10) unsigned` | [팔로우를 받은 크리에이터의 creator_no (FK to t_creator.no)] |\n"
        "| `member_no` | `int(10) unsigned` | [팔로우를 한 멤버의 member_no (FK to t_member_info.member_no)] |\n"
        "| `ins_datetime` | `datetime` | [팔로우 발생 시각] |\n"
        "---\n\n"
        "## t_member\n\n"
        "**설명:** [여기에 테이블에 대한 설명을 작성해주세요]\n\n"
        "| 컬럼명 | 데이터 타입 | 설명 |\n"
        "| --- | --- | --- |\n"
        "| `no` | `int(11) unsigned` | [회원의 고유 번호 (PK)] |\n"
        "| `email` | `varchar(200)` | [회원 이메일] |\n"
        "| `nickname` | `varchar(100)` | [회원 닉네임] |\n"
        "| `status` | `char(1)` | [가입 상태 (A=가입/인증 완료, J=가입완료)] |\n"
        "| `is_admin` | `char(1)` | [플랫폼 어드민 권한 여부] |\n"
        "---\n\n"
        "## t_member_join_phone_number\n\n"
        "**설명:** [여기에 테이블에 대한 설명을 작성해주세요]\n\n"
        "| 컬럼명 | 데이터 타입 | 설명 |\n"
        "| --- | --- | --- |\n"
        "| `no` | `int(11) unsigned` | [회원 전화번호 로그 번호 (PK)] |\n"
        "| `phone_country_no` | `int(11) unsigned` | [전화번호 지역번호] |\n"
        "| `member_no` | `int(11) unsigned` | [회원의 회원 번호 (FK to t_member.no)] |\n"
        "| `phone_number` | `varchar(20)` | [회원의 전화번호] |\n"
        "---\n\n"
        "## t_payment\n\n"
        "**설명:** [여기에 테이블에 대한 설명을 작성해주세요]\n\n"
        "| 컬럼명 | 데이터 타입 | 설명 |\n"
        "| --- | --- | --- |\n"
        "| `no` | `int(10) unsigned` | [결제 고유 ID (PK)] |\n"
        "| `member_no` | `int(11) unsigned` | [결제를 한 멤버의 member_no (FK)] |\n"
        "| `seller_creator_no` | `int(10) unsigned` | [매출 발생 크리에이터의 creator_no (FK)] |\n"
        "| `tier_no` | `int(10) unsigned` | [구매한 멤버십 상품 번호 (FK to t_tier.no)] |\n"
        "| `item` | `varchar(20)` | [결제 상품 구분 (F: 멤버십, C:컨텐츠 등)] |\n"
        "| `order_name` | `varchar(300)` | [구매 상품 이름] |\n"
        "| `currency_no` | `tinyint(3) unsigned` | [통화 구분 (1:원화,2:달러)] |\n"
        "| `heat` | `int(10) unsigned` | [결제 히트] |\n"
        "| `remain_heat` | `int(10) unsigned` | [실제 사용된 히트] |\n"
        "| `price` | `decimal(10,2) unsigned` | [결제 금액] |\n"
        "| `remain_price` | `decimal(10,2) unsigned` | [실제 결제 금액 (통화 적용 전)] |\n"
        "| `is_tax_free` | `char(1)` | [면세 여부] |\n"
        "| `status` | `char(1)` | [결제 상태 ('T','P' = 결제완료)] |\n"
        "| `ins_datetime` | `datetime` | [결제 요청일] |\n"
        "| `pay_datetime` | `datetime` | [결제 완료일] |\n"
        "---\n\n"
        "## t_post\n\n"
        "**설명:** [여기에 테이블에 대한 설명을 작성해주세요]\n\n"
        "| 컬럼명 | 데이터 타입 | 설명 |\n"
        "| --- | --- | --- |\n"
        "| `no` | `int(11) unsigned` | [포스트 고유 번호 (PK)] |\n"
        "| `member_no` | `int(11) unsigned` | [작성자 회원 번호 (FK)] |\n"
        "| `title` | `varchar(210)` | [포스트 제목] |\n"
        "| `content` | `mediumtext` | [포스트 내용] |\n"
        "| `status` | `varchar(10)` | [포스트 상태 (public=발행완료)] |\n"
        "| `public_range` | `char(1)` | [공개 범위 (A:전체,F:회원,C:유료,T:멤버십 지정)] |\n"
        "| `content_type` | `char(1)` | [컨텐츠 유형 (M,I,A,복합)] |\n"
        "| `is_fix_home` | `char(1)` | [홈화면 고정 여부] |\n"
        "| `is_fix_top` | `char(1)` | [상단 고정 여부] |\n"
        "| `view_count` | `int(11) unsigned` | [조회 수] |\n"
        "| `like_count` | `int(11) unsigned` | [좋아요 수] |\n"
        "| `ins_datetime` | `datetime` | [업로드 날짜] |\n"
        "| `mod_datetime` | `datetime` | [수정 날짜] |\n"
        "| `del_datetime` | `datetime` | [삭제 날짜] |\n"
        "---\n\n"
        "## t_post_like_log\n\n"
        "**설명:** [여기에 테이블에 대한 설명을 작성해주세요]\n\n"
        "| 컬럼명 | 데이터 타입 | 설명 |\n"
        "| --- | --- | --- |\n"
        "| `no` | `int(10) unsigned` | [포스트 좋아요 로그 번호 (PK)] |\n"
        "| `post_no` | `int(10) unsigned` | [좋아요를 클릭한 포스트 번호 (FK)] |\n"
        "| `member_no` | `int(10) unsigned` | [좋아요를 누른 회원 번호 (FK)] |\n"
        "| `ins_datetime` | `datetime` | [좋아요 날짜] |\n"
        "---\n\n"
        "## t_post_reply_like_log\n\n"
        "**설명:** [여기에 테이블에 대한 설명을 작성해주세요]\n\n"
        "| 컬럼명 | 데이터 타입 | 설명 |\n"
        "| --- | --- | --- |\n"
        "| `no` | `int(10) unsigned` | [댓글 좋아요 로그 번호 (PK)] |\n"
        "| `reply_no` | `int(11) unsigned` | [좋아요를 클릭한 댓글 번호 (FK)] |\n"
        "| `member_no` | `int(11) unsigned` | [좋아요를 누른 회원 번호 (FK)] |\n"
        "| `ins_datetime` | `datetime` | [좋아요 날짜] |\n"
        "---\n\n"
        "## t_post_view_log\n\n"
        "**설명:** [여기에 테이블에 대한 설명을 작성해주세요]\n\n"
        "| 컬럼명 | 데이터 타입 | 설명 |\n"
        "| --- | --- | --- |\n"
        "| `no` | `int(10) unsigned` | [포스트 조회 로그 번호 (PK)] |\n"
        "| `post_no` | `int(10) unsigned` | [조회한 포스트 번호 (FK)] |\n"
        "| `member_no` | `int(10) unsigned` | [조회한 회원 번호 (FK)] |\n"
        "| `is_auth` | `char(1)` | [인증 여부] |\n"
        "| `ins_datetime` | `datetime` | [조회 날짜] |\n"
        "---\n\n"
        "## t_tier\n\n"
        "**설명:** [여기에 테이블에 대한 설명을 작성해주세요]\n\n"
        "| 컬럼명 | 데이터 타입 | 설명 |\n"
        "| --- | --- | --- |\n"
        "| `no` | `int(10) unsigned` | [멤버십 상품 고유 ID (PK)] |\n"
        "| `creator_no` | `int(10) unsigned` | [멤버십 제공 크리에이터 ID (FK)] |\n"
        "| `public_status` | `varchar(10)` | [공개 상태] |\n"
        "| `is_renewable` | `char(1)` | [갱신 가능 여부] |\n"
        "| `end_criteria` | `varchar(10)` | [종료 기준] |\n"
        "| `name` | `varchar(60)` | [멤버십 이름 (예: '눈팅족')] |\n"
        "| `regular_price` | `int(10) unsigned` | [정가] |\n"
        "| `price` | `int(10) unsigned` | [판매가] |\n"
        "| `regular_heat` | `int(10) unsigned` | [정가 히트] |\n"
        "| `heat` | `int(10) unsigned` | [판매 히트] |\n"
        "| `sponsor_limit` | `int(11)` | [스폰서 제한] |\n"
        "| `is_private` | `char(1)` | [비공개 여부] |\n"
        "| `is_approval_required` | `char(1)` | [승인 필요 여부] |\n"
        "| `is_monthly_pass_allowed` | `char(1)` | [월간패스 허용 여부] |\n"
        "| `period` | `tinyint(3) unsigned` | [기간 (개월)] |\n"
        "| `end_date` | `date` | [종료일] |\n"
        "| `join_start_date` | `date` | [가입 시작일] |\n"
        "| `join_end_date` | `date` | [가입 종료일] |\n"
        "---"
    ),
    "conversation_history": [],
    "query_result_cache": None,
    "resolved_context": None,
    "review_status": None,
    "review_result": None,
    "final_sql": None,
    "explanation": None,
    "success": True,

    # === 새로 추가될 states ===
    "data_gathering_sql": "<생성된 SQL 문자열>",
    "python_code": "<생성된 Python 코드>",
    "sql_query": "<생성된 SQL 문자열>",  # sql_execution 노드용 alias
    "python_code_result": {
        "code": "<생성된 Python 코드>",
        "data_gathering_sql": "<SQL>",
        "confidence": "<float>",
        "imports": ["pandas", "matplotlib", ...],  # 검증 결과에서 추출
        "is_safe": True,
        "main_function": "<main 함수명>"
    },
    "confidence_scores": {
        "nl_processing": 0.8833,
        "schema_mapping": 1.0,
        "python_code_generation": "<float>"
    },
}
