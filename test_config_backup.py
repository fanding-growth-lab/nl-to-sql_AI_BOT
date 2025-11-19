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
from google import genai

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)
chat = client.chats.create(model='gemini-2.5-flash')

# === GLOBAL FUNCTIONS ===
def generate(contents):
    ipt = contents
    response = chat.send_message(contents)
    opt = response.text
    clean_opt = re.sub(r"```(?:json|sql|python)?\s*([\s\S]*?)\s*```", r"\1", opt).strip()

    return clean_opt

def get_data_gathering_sql(user_query, rag_schema_context) -> list:
    return json.loads(
        generate(
            contents=PROMPT_DATA_GATHERING.format(user_query=user_query, rag_schema_context=rag_schema_context, business_rules=BUSINESS_RULES)
        )
    )

def save_sql_queries_to_json(sqls: list, session_id: str):
    output_dir = "sql_queries"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{session_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sqls, f, ensure_ascii=False, indent=2)
    return file_path

def save_sql_results_to_json(results: dict, session_id: str):
    output_dir = "sql_query_results"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{session_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return file_path

def get_python_code(user_query, results_head, results_file_path, sql_queries_file_path, error_feedback=None):
    prompt_content = PROMPT_GENERATE_PYTHON_CODE.format(
        user_query=user_query, 
        results_head=results_head, 
        results_file_path=results_file_path,
        sql_queries_file_path=sql_queries_file_path
    )
    if error_feedback:
        prompt_content += f"\n\n## Previous Error Feedback\n{error_feedback}\n\n### 🎯 Your Task: Correct the code based on the feedback."

    return generate(contents=prompt_content)

def execute_sql(query: str):
    df = pd.read_sql(text(query), engine)    
    json_data = df.to_json(orient="records", force_ascii=False)
    return json_data

def run_data_gathering(state):
    sqls = get_data_gathering_sql(state["user_query"], state["rag_schema_context"])
    
    # Save SQL queries to a JSON file
    sql_queries_file_path = save_sql_queries_to_json(sqls, state["session_id"])

    full_results = {
        x["table"]: execute_sql(x["sql"]) 
        for x in sqls
    }
    
    # Save full results to a JSON file
    results_file_path = save_sql_results_to_json(full_results, state["session_id"])
    
    return full_results, results_file_path, sql_queries_file_path

def run_generate_python_code(state, full_sql_results, results_file_path, sql_queries_file_path, max_retries=3):
    # Extract head(10) for each table result
    results_head = {}
    for table_name, json_data in full_sql_results.items():
        df = pd.read_json(io.StringIO(json_data)) # Use io.StringIO to suppress FutureWarning
        results_head[table_name] = df.head(10).to_json(orient="records", force_ascii=False)
    
    python_code = ""
    for retry_count in range(max_retries):
        error_feedback = state.get("error_message") if retry_count > 0 else None
        
        python_code = get_python_code(state["user_query"], results_head, results_file_path, sql_queries_file_path, error_feedback)

        # save python code
        output_dir = "python_codes"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{state['session_id']}_{retry_count}.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(python_code)
        
        try:
            # Attempt to execute the generated Python code
            local_env = {}
            exec(python_code, {}, local_env)
            print("Generated Python code executed successfully (simulated).")
            state["error_message"] = None # Clear error message on success
            return python_code
        except Exception as e:
            error_message = f"Python code execution failed: {e}"
            print(error_message)
            state["error_message"] = error_message
            if retry_count < max_retries - 1:
                print(f"Retrying Python code generation... (Attempt {retry_count + 1}/{max_retries})")
            else:
                print("Max retries reached. Could not generate executable Python code.")
                return python_code # Return the last generated code even if it failed
    return python_code

def run_dynamic_code(code: str, context: dict = None):
    local_env = {}
    exec(code, context or {}, local_env)
    return local_env

# === GLOBAL VARIABLES ===
BUSINESS_RULES = """
* Data Relationship Summary
- Creator's name can be found in `t_member.nickname`.
- `t_member.no` joins with `t_creator.member_no`.
- `t_creator.no` joins with `t_payment.seller_creator_no`.
- `t_fanding_log.coupon_member_no` joins with `t_creator_coupon_member.no` to check for coupon usage.

* Aggregation Timeframes:
- Daily: 00:00:00 to 23:59:59. Snapshot at 23:59:59.
- Weekly: Monday 00:00:00 to Sunday 23:59:59. Snapshot at Sunday 23:59:59.
- Monthly: First day of the month 00:00:00 to last day of the month 23:59:59. Snapshot at last day 23:59:59.

* Payment Data Rules
- Completed Payment: `status` is NOT 'W' (Waiting) or 'F' (Failed), and `pay_datetime` is not NULL.
- Refund: `status` is 'R' (Full Refund) or 'P' (Partial Refund).
  - 'R' (Full Refund): The member is considered to have no experience, and the payment is excluded from the installment count.
  - 'P' (Partial Refund): The member has some experience, and the payment is included in the installment count.
- Actual sales amount must be calculated using `remain_price`. The `price` column should NOT be used.
- When analyzing sales, you must include statuses 'T' (Approved) and 'P' (Partially Refunded).
- Currency Conversion:
  - For KRW (currency_no = 1): use `remain_price`.
  - For USD (currency_no = 2): use `remain_price` * 1360.
  - For HEAT (currency_no is NULL): use `remain_heat` * 110.

* Member and Membership Aggregation Rules
- Active Member & Churn Grace Period: This is a special rule that applies ONLY IF the aggregation date is within 3 days of the CURRENT DATE.
  - Rule: A member is considered 'Active' (not churned) even if their membership has ended, as long as the end date is within 3 days of the aggregation date.
  - Example: If today is Oct 2nd and we are aggregating data for Oct 1st, a member whose subscription ended on Oct 1st is still considered active. If we are aggregating data for Sep 15th, this rule does not apply.
- New Subscriber: A member who starts a membership within a given period and has no prior membership history.
- Churner: A member whose membership ends within a given period and does not restart within 3 days (subject to the grace period rule above).
- Cancellation Booker: A member who has a cancellation scheduled (`중단예약=T`) as of the aggregation snapshot time.
- Re-subscriber after Churn: A member who starts a new membership within a given period, and had a previous membership that ended more than 3 days before the new start date.

* Weekly Active Member Calculation (Snapshot-based):
- To count weekly active members, the query must first generate a series of dates representing the snapshot time for each week (Sunday at 23:59:59) within the requested period.
- A recursive CTE is the required method for generating this date series.
- For each snapshot date, the query must count the number of distinct members whose continuous membership 'Block' (calculated as per the rules below) was active on that date.
- A member is considered active on a snapshot date if the snapshot date is between the `start_date` and `end_date` of their membership block (inclusive).
- The final output should be the snapshot date (or week identifier) and the corresponding count of active members.

* Membership Block, Edition, and Month Count Rules
- **Core Concept**: When analyzing user retention or continuous membership, individual `t_fanding_log` records must be grouped into continuous 'Blocks'. A simple date range check on individual logs is incorrect as it can misinterpret short breaks as churn.
- Continuous Subscription: A new membership log in `t_fanding_log` is considered part of the same block if it starts within 3 days of the previous log's end date for the same `fanding_no`. A gap of 4 days or more signifies a new block.
- **SQL Implementation Hint**: To correctly group logs into blocks, a multi-step process using CTEs is required:
  1.  **Order Logs**: Use `LAG(end_date, 1) OVER (PARTITION BY fanding_no ORDER BY start_date)` to get the previous log's end date (`prev_end_date`). Note: MariaDB `LAG` function should be used with the `LAG(expression, offset)` syntax.
  2.  **Flag New Blocks**: Create a flag (`new_block_flag`) using a `CASE` statement. A new block starts if `prev_end_date` is NULL or if `start_date >= DATE_ADD(prev_end_date, INTERVAL 4 DAY)`.
  3.  **Group Blocks**: Use a cumulative `SUM(new_block_flag)` over the same window function to assign a unique `block_no` to each block.
  4.  **Finalize Blocks**: `GROUP BY fanding_no, block_no` and find the `MIN(start_date)` and `MAX(end_date)` to get the final start and end date for each block.
- **Final Analysis**: All retention, active user counts, and churn analysis must be performed on these calculated blocks, not on the raw `t_fanding_log` entries.
- Free Coupon Usage: If `t_fanding_log.coupon_member_no` is not NULL, it indicates the membership was started with a coupon. This is considered the beginning of a new block and the first Edition (회차 0).
- Edition (회차): The number of payments made within a single block. The count starts from 0 for a coupon-based start.
- Month Count (개월차): The number of months a block has been active, calculated from the block's start date.
"""

PROMPT_DATA_GATHERING = """
You are an expert SQL generator for the Fanding platform.
You will receive two inputs:
1. A user's natural language query (`user_query`)
2. The database schema (`rag_schema_context`)

---

## User Query
{user_query}

---

## Database Schema
{rag_schema_context}

---

## Business Rules
{business_rules}

### 🎯 Your Task
Generate **raw-level SQL queries** for each relevant table in the database
based on the user's request.

Each query must follow these strict rules:
1. **JOIN은 원칙적으로 사용하지 않는다.**  
   - 한 쿼리에서는 하나의 테이블만 조회한다.  
   - 단, 다음과 같은 경우에만 최소한의 JOIN을 허용한다:  
     - 사용자가 이름, 닉네임 등으로 데이터를 요청했는데  
       그 정보가 현재 테이블이 아닌 다른 테이블에 있을 때.  
     - 예: 크리에이터 정보(`t_creator`)를 닉네임(`t_member.nickname`)으로 조회해야 하는 경우.  
       → `t_creator c JOIN t_member m ON c.member_no = m.no`  

2. **집계, 요약, 통계 금지**  
   - `SUM`, `COUNT`, `AVG`, `MAX`, `MIN`, `GROUP BY` 등의 함수는 사용하지 않는다.  
   - 오직 개별 레코드(원본 행)만 조회한다. 

3. **필요한 조건만 WHERE로 제한**  
   - 사용자 요청에 날짜, 이름, ID 등의 조건이 있다면 WHERE 절에 포함시킨다.  
   - 예: `"8월 매출"` → `WHERE pay_datetime BETWEEN '2025-08-01' AND '2025-08-31'`  
   - 예: `"강환국 작가"` → `JOIN t_member` 후 `m.nickname LIKE '%강환국%'`  

4. **SELECT * 사용**  
   - 가능하면 `SELECT *`를 사용하되, JOIN을 사용하는 경우엔 `테이블별 alias.*` 형식 사용  
     (예: `SELECT c.* FROM t_creator c JOIN t_member m ...`)  

5. **출력 형식은 JSON 배열로 반환**  
   - 각 객체는 다음 형태를 따른다:  
     ```json
     {{
       "table": "<테이블 이름>",
       "sql": "<SQL 문장>"
     }}
     ```  
   - 여러 테이블이 관련 있을 경우, JSON 리스트로 여러 쿼리를 포함시킨다.  
   - **설명, 주석, 마크다운, 텍스트를 추가하지 않는다.**  
   - 결과는 유효한 JSON이어야 한다. (파싱 가능한 구조)

6. **반드시 `rag_schema_context`를 근거로 쿼리문을 작성한다.**
   - 스키마에 정의되지 않은 컬럼이나 테이블은 사용하지 않는다.
---

### 🧾 Example Behavior

**Example**
User query:  
> "A 크리에이터의 8월 매출 데이터를 보여줘."

Expected Output:
```json
[
  {{
    "table": "t_payment",
    "sql": "SELECT * FROM t_payment WHERE pay_datetime BETWEEN '2025-08-01' AND '2025-08-31';"
  }},
  {{
    "table": "t_creator",
    "sql": "SELECT c.* FROM t_creator c JOIN t_member m ON c.member_no = m.no WHERE m.nickname LIKE '%A%';"
  }}
]
```
"""

PROMPT_GENERATE_PYTHON_CODE = """
You are an expert data analyst and Python developer.
You are working with data extracted from the Fanding platform database.
You will be given:
1. The user's natural language request (`user_query`)
2. The retrieved SQL results as JSON data from multiple tables (`results`)

Your goal is to write Python code that performs analysis on this data
and produces outputs that fully answer the user's request.

---

## User Query
{user_query}

---

## SQL Query Results (by table) - Head(10)
{results_head}

## Full SQL Query Results File Path
{results_file_path}

## SQL Queries File Path
{sql_queries_file_path}

---

### 🎯 Your Task
Write **executable Python code** that analyzes or compares the data according to the user's request.

Follow these rules carefully:

1. **데이터 불러오기**
   - `results_file_path`는 전체 SQL 쿼리 결과가 저장된 JSON 파일의 경로이다.
   - `results_file_path`를 사용하여 전체 데이터를 로드하고, 각 테이블 데이터를 `pandas.DataFrame`으로 변환해야 한다.
     ```python
     import json
     import pandas as pd

     with open(results_file_path, 'r', encoding='utf-8') as f:
         full_results = json.load(f)
     
     df_payment = pd.DataFrame(json.loads(full_results["t_payment"]))
     df_creator = pd.DataFrame(json.loads(full_results["t_creator"]))
     ```
   - 테이블 이름에 따라 자동으로 DataFrame 변수를 생성하라.

2. **데이터 병합 및 가공**
   - `user_query`의 요구사항에 맞게 필요한 테이블을 병합(merge)하거나 필터링한다.
   - JOIN 조건은 스키마(`rag_schema_context`)를 기반으로 합리적으로 설정한다.
     예: `t_payment.seller_creator_no = t_creator.no`, `t_fanding.member_no = t_member.no`
   - 기간, 이름, 크리에이터, 멤버 등과 관련된 필터 조건을 적용한다.

3. **집계 / 분석 / 비교**
   - 사용자 요청이 “분석”, “비교”, “성과” 등과 관련될 경우,
     단순 집계(예: `groupby`, `value_counts`, `mean`)를 수행한다.
   - 예를 들어, “8월 매출 비교”라면 크리에이터별 합계(price)를 계산한다.
   - 단, LLM이 임의로 수치를 만들면 안 되며, DataFrame 내 데이터를 기준으로만 계산한다.

4. **시각화 (선택)**
   - 시각적 비교나 트렌드가 필요한 경우 matplotlib 또는 seaborn을 활용한다.
   - 그래프 출력은 선택적이며, `plt.show()`로 끝내야 한다.

5. **출력 형식**
   - 코드 내에서 `print()`를 통해 주요 결과를 명시적으로 출력하라.
   - 함수 정의, 변수명, 주석을 포함한 **완전한 실행 가능한 코드**를 작성한다.
   - 설명 문장이나 해설은 출력하지 말고, 코드만 반환한다.

6. **보안 및 안전성**
   - 외부 API 호출, 파일 저장, 시스템 명령어 사용 등은 금지한다.
   - pandas, matplotlib, numpy 등 기본 라이브러리만 사용 가능하다.

---

### 🧾 Example Behavior

**Example 1**  
User query:  
> "A 크리에이터와 B 크리에이터의 8월 매출을 비교해줘."

Expected Output:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Convert JSON results to DataFrames
df_payment = pd.DataFrame(results["t_payment"])
df_creator = pd.DataFrame(results["t_creator"])
df_member = pd.DataFrame(results["t_member"])

# Merge payment with creator info
merged = df_payment.merge(df_creator, left_on="seller_creator_no", right_on="no", how="left")

# Filter for August 2025
merged["pay_datetime"] = pd.to_datetime(merged["pay_datetime"])
august = merged[
    (merged["pay_datetime"].dt.month == 8) & (merged["pay_datetime"].dt.year == 2025)
]

# Filter creators
target = august[august["name"].isin(["A", "B"])]

# Aggregate sales
summary = target.groupby("name")["price"].sum().reset_index()

print(summary)

# Optional plot
plt.bar(summary["name"], summary["price"])
plt.title("8월 크리에이터별 매출 비교")
plt.xlabel("크리에이터")
plt.ylabel("매출 금액")
plt.show()
```

---

### ⚙️ Output Format

- Return **only executable Python code**, no markdown, no commentary.

- Do not include explanation, quotes, or code fences.

- Use only `results`, `pandas`, and `matplotlib` (optional).

---

**Now generate Python code that performs the data analysis for the given user query and results.**

---

## ✅ 프롬프트 구조 요약

| 섹션 | 설명 |
|------|------|
| **입력** | `{{user_query}}`, `{{results}}` |
| **핵심 작업** | `pandas`로 JSON 로드 → 병합 → 분석 → 출력 |
| **규칙** | JOIN은 DataFrame merge로 수행, 외부 API 금지 |
| **출력** | 완전한 Python 코드만, markdown 금지 |
| **예시** | 8월 매출 비교 케이스 포함 |

---

## ✅ (선택) — 자동 포맷팅용 파이썬 함수 예시

```python
def make_python_generation_prompt(state, results):
    return PROMPT_PYTHON_GENERATION.format(
        user_query=state["user_query"],
        results=json.dumps(results, ensure_ascii=False, indent=2)
    )
```
"""

STATE = {
    "user_query": "25년 8월 전체 멤버십 가입자 수와 '강환국 작가', '고래돈공부' 크리에이터의 월 성과를 분석하고 비교해줘.",
    "user_id": None,
    "channel_id": None,
    "session_id": "ebe64650-26e5-4d0d-bf9a-21b80d0133e2",
    "context": {
        "user_id": None,
        "channel_id": None
    },
    "normalized_query": "25년 8월 전체 멤버십 가입자 수와 '강환국 작가', '고래돈공부' 크리에이터의 월 성과를 분석하고 비교해줘.",
    "intent": "COMPLEX_ANALYSIS",
    "llm_intent_result": {
        "intent": "COMPLEX_ANALYSIS",
        "confidence": 0.95,
        "reasoning": "전체 멤버십 가입자 수는 단순 집계이지만, 특정 크리에이터들의 '월 성과를 분석하고 비교'"
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
