import json
import re
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from test_config import (
    llm,
    STATE,
    save_result,
    run_dynamic_code,
    PROMPT_SEARCH_RELATIVE_TABLES,
    PROMPT_GENERATE_SQL_QUERY,
    PROMPT_VALIDATE_SQL_QUERY,
    PROMPT_GENERATE_FINAL_RESULT,
    PROMPT_PLAN_PYTHON_ANALYSIS,
    PROMPT_GENERATE_PYTHON_STEP,
    PROMPT_VALIDATE_PYTHON_STEP,
)
from rule_rag import retrieve_relevant_rules


def summarize_context_for_llm(context: Dict[str, Any]) -> str:
    """Create a meaningful summary of python context for the LLM."""
    summary_lines = []
    
    for var_name, value in context.items():
        if var_name == "__builtins__":
            continue
        
        # 테이블 스키마 정보 특별 처리
        if var_name == "_table_schemas":
            summary_lines.append("\n=== 📋 테이블 스키마 정보 (SQL 작성 시 반드시 참조) ===")
            for table_info in value:
                table_name = table_info.get("table", "unknown")
                schema = table_info.get("schema", [])
                summary_lines.append(f"  • {table_name}:")
                for col in schema:
                    col_name = col.get("column", "unknown")
                    col_type = col.get("type", "")
                    summary_lines.append(f"    - {col_name} ({col_type})")
            summary_lines.append("=== (스키마에 없는 컬럼 사용 금지!) ===\n")
            continue
            
        type_name = type(value).__name__
        
        # DataFrame인 경우
        if hasattr(value, 'shape') and hasattr(value, 'columns'):
            columns_list = list(value.columns)
            columns_preview = columns_list[:10]  # 처음 10개 컬럼만
            if len(columns_list) > 10:
                columns_preview_str = f"{columns_preview}... (총 {len(columns_list)}개 컬럼)"
            else:
                columns_preview_str = str(columns_list)
            summary_lines.append(
                f"- `{var_name}`: DataFrame, shape={value.shape}, columns={columns_preview_str}"
            )
        # List/Tuple인 경우
        elif isinstance(value, (list, tuple)):
            if len(value) > 0:
                first_item_type = type(value[0]).__name__
                summary_lines.append(
                    f"- `{var_name}`: {type_name}, length={len(value)}, first_item_type={first_item_type}"
                )
            else:
                summary_lines.append(
                    f"- `{var_name}`: {type_name}, length=0 (empty)"
                )
        # Dict인 경우
        elif isinstance(value, dict):
            keys_preview = list(value.keys())[:5]
            if len(value) > 5:
                summary_lines.append(
                    f"- `{var_name}`: {type_name}, keys={keys_preview}... (총 {len(value)}개)"
                )
            else:
                summary_lines.append(
                    f"- `{var_name}`: {type_name}, keys={list(value.keys())}"
                )
        # String인 경우
        elif isinstance(value, str):
            preview = value[:100] + "..." if len(value) > 100 else value
            summary_lines.append(
                f"- `{var_name}`: {type_name}, value='{preview}'"
            )
        # 숫자/기본 타입인 경우
        elif isinstance(value, (int, float, bool)):
            summary_lines.append(
                f"- `{var_name}`: {type_name}, value={value}"
            )
        # Module인 경우
        elif type_name == 'module':
            module_name = getattr(value, '__name__', 'unknown')
            summary_lines.append(
                f"- `{var_name}`: imported module '{module_name}'"
            )
        # 기타
        else:
            summary_lines.append(
                f"- `{var_name}`: {type_name}"
            )
    
    return "\n".join(summary_lines) if summary_lines else "No variables in context yet."


class AgentState(TypedDict):
    user_query: str
    rag_schema_context: str
    relative_tables: List[Dict[str, Any]]
    sql_queries: List[Dict[str, str]]
    sql_validation: Dict[str, Any]
    python_code: str
    python_execution_result: str
    python_validation: Dict[str, Any] # 추가
    final_result: str
    error: str
    retry_count: int
    max_retries: int
    sql_feedback: str
    python_error_feedback: str
    python_validation_feedback: str # 추가
    # Iterative Python Execution State
    python_plan: List[str]
    current_step_index: int
    python_context: Dict[str, Any]
    step_code: str
    step_result: str
    step_validation: Dict[str, Any]
    step_retry_count: int  # 현재 단계 재시도 횟수
    max_step_retries: int  # 단계별 최대 재시도 횟수


def search_relative_tables_node(state: AgentState):
    print("---노드: 관련 테이블 검색---")
    prompt = PromptTemplate(
        template=PROMPT_SEARCH_RELATIVE_TABLES,
        input_variables=["user_query", "rag_schema_context"],
    )
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({
        "user_query": state["user_query"],
        "rag_schema_context": state["rag_schema_context"]
    })
    save_result(result, "relative_tables.json", True)
    return {"relative_tables": result}


def generate_sql_queries_node(state: AgentState):
    print("---노드: SQL 쿼리 생성---")
    prompt = PromptTemplate(
        template=PROMPT_GENERATE_SQL_QUERY,
        input_variables=["user_query", "relative_tables", "business_rules"],
    )
    chain = prompt | llm | JsonOutputParser()
    feedback = state.get("sql_feedback", "")
    if feedback:
        feedback = f"이전에 생성한 결과와 피드백:\n{state.get('sql_queries')}\n{feedback}"
    # Retrieve relevant business rules for SQL
    business_rules = retrieve_relevant_rules(state["user_query"], category="sql", rule_type="business")
    
    result = chain.invoke({
        "user_query": state["user_query"],
        "relative_tables": state["relative_tables"],
        "business_rules": business_rules,
        "sql_feedback": feedback,
    })
    save_result(result, "sql_query.json", True)
    return {"sql_queries": result, "sql_feedback": ""}  # 피드백 사용 후 초기화


def validate_sql_query_node(state: AgentState):
    print("---노드: SQL 쿼리 검증---")
    prompt = PromptTemplate(
        template=PROMPT_VALIDATE_SQL_QUERY,
        input_variables=["user_query", "sql_queries", "relative_tables", "business_rules"],
    )
    chain = prompt | llm | JsonOutputParser()
    # Retrieve relevant business rules for SQL validation
    business_rules = retrieve_relevant_rules(state["user_query"], category="sql", rule_type="business")

    result = chain.invoke({
        "user_query": state["user_query"],
        "relative_tables": state["relative_tables"],
        "business_rules": business_rules,
        "sql_queries": state["sql_queries"],
    })
    save_result(result, "sql_validation.json", False)   # 검증 실패 시 아래에서 피드백 print()
    return {"sql_validation": result}


def decide_sql_revalidation(state: AgentState):
    print("---엣지: SQL 재검증 결정---")
    if state["sql_validation"]["is_valid"]:
        return "plan_python_analysis"
    else:
        return "handle_sql_feedback"


def handle_sql_feedback_node(state: AgentState):
    print("---노드: SQL 피드백 처리---")
    retry_count = state.get("retry_count", 0) + 1
    feedback = state["sql_validation"]["feedback"]
    print(f"SQL 검증 실패 피드백: {feedback}, 재시도 횟수: {retry_count}")
    return {"retry_count": retry_count, "sql_feedback": feedback}


def plan_python_analysis_node(state: AgentState):
    print("---노드: Python 분석 계획 수립---")
    prompt = PromptTemplate(
        template=PROMPT_PLAN_PYTHON_ANALYSIS,
        input_variables=["user_query", "relative_tables", "business_rules", "sql_queries"],
    )
    chain = prompt | llm | StrOutputParser()
    
    # 계획 단계에서는 비즈니스 규칙만 필요 (무엇을 해야 하는지)
    business_rules = retrieve_relevant_rules(state["user_query"], category="common", rule_type="business")
    
    raw_result = chain.invoke({
        "user_query": state["user_query"],
        "relative_tables": state["relative_tables"],
        "business_rules": business_rules,
        "sql_queries": state["sql_queries"]
    })
    
    # JSON 파싱 시도 (Markdown 코드 블록 제거 및 리스트 추출)
    try:
        # ```json ... ``` 또는 [...] 패턴 찾기
        json_match = re.search(r'\[.*\]', raw_result, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)
        else:
            # JSON 패턴을 못 찾으면 전체 텍스트를 줄바꿈으로 분리하여 리스트로 변환 (fallback)
            print("⚠️ 계획 파싱 경고: JSON 리스트를 찾을 수 없어 텍스트를 줄 단위로 분리합니다.")
            result = [line.strip() for line in raw_result.split('\n') if line.strip() and not line.strip().startswith('```')]
            
    except json.JSONDecodeError as e:
        print(f"⚠️ 계획 파싱 에러: {e}. 텍스트 기반으로 처리합니다.")
        result = [line.strip() for line in raw_result.split('\n') if line.strip() and not line.strip().startswith('```')]

    save_result(result, "python_plan.json", True)
    
    # 테이블 스키마를 python_context에 포함 (코드 생성 시 참조용)
    schema_info = {
        "_table_schemas": state["relative_tables"],
        "sql_queries": state["sql_queries"],  # SQL 쿼리 리스트 추가
        "__builtins__": __builtins__
    }
    
    return {
        "python_plan": result,
        "current_step_index": 0,
        "python_context": schema_info,
        "python_code": "",
        "python_execution_result": "",
        "step_retry_count": 0,  # 초기 retry count
        "max_step_retries": 3   # 단계별 최대 3회 재시도
    }


def generate_python_step_code_node(state: AgentState):
    """Generate code for the current step."""
    current_index = state["current_step_index"]
    plan = state["python_plan"]
    current_step = plan[current_index]
    
    print(f"---노드: Python 단계 코드 생성 ({current_index + 1}/{len(plan)}) : {current_step}---")
    
    prompt = PromptTemplate(
        template=PROMPT_GENERATE_PYTHON_STEP,
        input_variables=["user_query", "business_rules", "python_rules", "python_plan", "current_step", "python_context", "step_feedback"],
    )
    chain = prompt | llm | StrOutputParser()
    
    # 현재 단계 context를 포함하여 더 정확한 규칙 검색
    combined_query = f"{state['user_query']} {current_step}"
    
    # 비즈니스 규칙 (메트릭 정의, 데이터 관계 등)
    business_rules = retrieve_relevant_rules(combined_query, category="common", rule_type="business")
    # Python 규칙 (Block Logic 구현, 코드 작성 가이드라인 등)
    python_rules = retrieve_relevant_rules(combined_query, category="python", rule_type="python")
    
    # ✅ 개선된 컨텍스트 요약 - DataFrame 구조, 변수 값 등 상세 정보 제공
    context_summary = summarize_context_for_llm(state["python_context"])
    
    # Get feedback if retry
    step_feedback = state.get("step_validation", {}).get("feedback", "")
    if step_feedback:
        step_feedback = f"이전 시도 피드백:\n{step_feedback}"
    
    raw_output = chain.invoke({
        "user_query": state["user_query"],
        "business_rules": business_rules,
        "python_rules": python_rules,
        "python_plan": plan,
        "current_step": current_step,
        "python_context": context_summary,
        "step_feedback": step_feedback
    })
    
    # Retry count 증가
    current_retry = state.get("step_retry_count", 0)
    
    # Parse JSON output with CoT reasoning
    clean_code = ""
    reasoning = ""
    approach = ""
    expected_output = ""
    potential_issues = ""
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*"code"[\s\S]*\}', raw_output)  # ← raw_output은 chain.invoke() 결과
        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)
            
            # Extract fields
            reasoning = result.get("reasoning", "")
            approach = result.get("approach", "")
            expected_output = result.get("expected_output", "")
            potential_issues = result.get("potential_issues", "")
            clean_code = result.get("code", "")
            
            # Log CoT reasoning
            print(f"\n🧠 추론: {reasoning}")
            print(f"📐 접근법: {approach}")
            if potential_issues:
                print(f"⚠️  예상 문제: {potential_issues}\n")
        else:
            # Fallback
            print("⚠️  JSON 파싱 실패, 일반 코드로 처리")
            clean_code = re.sub(r"```(?:python)?\s*([\s\S]*?)\s*```", r"\1", raw_output).strip()
            
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON 에러: {e}, 폴백 처리")
        clean_code = re.sub(r"```(?:python)?\s*([\s\S]*?)\s*```", r"\1", raw_output).strip()
    print(f"\n생성된 코드:\n{clean_code}")
    
    # 프롬프트 및 결과 로깅
    log_data = {
        "step": f"{current_index + 1}/{len(plan)}",
        "step_description": current_step,
        "retry_count": current_retry,
        "cot_reasoning": {  # ← 새로 추가
            "reasoning": reasoning,
            "approach": approach,
            "expected_output": expected_output,
            "potential_issues": potential_issues
        },
        "prompt_inputs": {
            "user_query": state["user_query"],
            "current_step": current_step,
            "business_rules": business_rules[:200] + "..." if len(business_rules) > 200 else business_rules,
            "python_rules": python_rules[:200] + "..." if len(python_rules) > 200 else python_rules,
            "context_summary": context_summary[:300] + "..." if len(context_summary) > 300 else context_summary,
            "step_feedback": step_feedback[:200] + "..." if step_feedback and len(step_feedback) > 200 else step_feedback
        },
        "generated_code": clean_code
    }
    save_result(log_data, f"step_{current_index + 1}_code_gen_retry_{current_retry}.json", False)
    
    return {
        "step_code": clean_code,
        "step_validation": {},  # Clear validation to prevent stale feedback
        "step_retry_count": current_retry + 1  # 재시도 카운트 증가
    }


def execute_python_step_node(state: AgentState):
    """Execute the generated code for the current step."""
    current_index = state["current_step_index"]
    plan = state["python_plan"]
    current_step = plan[current_index]
    
    print(f"---노드: Python 단계 실행 ({current_index + 1}/{len(plan)}) : {current_step}---")
    
    # Execute the code
    execution_output = run_dynamic_code(state["step_code"], context=state["python_context"])
    
    # Merge new locals into context
    if execution_output["local_env"]:
        state["python_context"].update(execution_output["local_env"])
        
    step_result = execution_output["captured_output"] or ""
    error = execution_output["error"]
    
    if error:
        step_result += f"\nError: {str(error)}"
        print(f"실행 오류: {error}")
    else:
        print(f"실행 결과:\n{step_result}")

    # Accumulate code
    new_accumulated_code = state["python_code"] + "\n\n" + f"# Step: {current_step}\n" + state["step_code"]
        
    return {
        "step_result": step_result,
        "python_code": new_accumulated_code,
        "python_context": state["python_context"]
    }


def validate_python_step_node(state: AgentState):
    print("---노드: Python 단계 검증---")
    current_index = state["current_step_index"]
    plan = state["python_plan"]
    current_step = plan[current_index]
    
    prompt = PromptTemplate(
        template=PROMPT_VALIDATE_PYTHON_STEP,
        input_variables=["user_query", "business_rules", "python_rules", "current_step", "step_code", "step_result"],
    )
    chain = prompt | llm | JsonOutputParser()
    
    # 현재 단계 context를 포함하여 더 정확한 규칙 검색
    combined_query = f"{state['user_query']} {current_step}"
    
    # 비즈니스 규칙 (메트릭 정의, 데이터 관계 등)
    business_rules = retrieve_relevant_rules(combined_query, category="common", rule_type="business")
    # Python 규칙 (Block Logic 구현, 코드 작성 가이드라인 등)
    python_rules = retrieve_relevant_rules(combined_query, category="python", rule_type="python")
    
    result = chain.invoke({
        "user_query": state["user_query"],
        "business_rules": business_rules,
        "python_rules": python_rules,
        "current_step": current_step,
        "step_code": state["step_code"],
        "step_result": state["step_result"]
    })
    
    print(f"검증 결과: {result}")
    return {"step_validation": result}


def check_step_result(state: AgentState):
    print("---엣지: 단계 결과 확인---")
    validation = state["step_validation"]
    
    if validation["is_valid"]:
        next_index = state["current_step_index"] + 1
        state["python_execution_result"] += "\n\n" + state["step_result"]
        if next_index < len(state["python_plan"]):
            print(f"다음 단계로 이동: {next_index + 1}/{len(state['python_plan'])}")
            return "next_step"
        else:
            print("모든 단계 완료, 최종 결과 생성")
            return "finalize"
    else:
        # Check retry limit
        retry_count = state.get("step_retry_count", 0)
        max_retries = state.get("max_step_retries", 3)
        
        if retry_count >= max_retries:
            print(f"⚠️  단계 재시도 한계 초과 ({retry_count}/{max_retries}). 최종 결과로 이동.")
            return "finalize"  # 재시도 한계 초과 시 강제로 종료
        else:
            print(f"현재 단계 재시도: {retry_count + 1}/{max_retries} (단계: {state['current_step_index'] + 1}/{len(state['python_plan'])})")
            return "retry_step"


def increment_step_index_node(state: AgentState):
    """Move to next step by incrementing the index."""
    next_index = state["current_step_index"] + 1
    print(f"Step index incremented: {state['current_step_index']} -> {next_index}")
    return {
        "current_step_index": next_index,
        "step_retry_count": 0  # 새 단계로 이동 시 retry count 초기화
    }


# TODO
def generate_final_result_node(state: AgentState):
    print("---노드: 최종 결과 생성---")
    prompt = PromptTemplate(
        template=PROMPT_GENERATE_FINAL_RESULT,
        input_variables=["python_execution_result", "error_message"],
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "python_execution_result": state.get("python_execution_result", ""),
        "error_message": state.get("error", "")
    })
    save_result(result, "final_result.txt", True)
    return {"final_result": result}


# === LangGraph 워크플로우 정의 ===

# 그래프 초기화
workflow = StateGraph(AgentState)

# --- 노드 정의 ---
workflow.add_node("search_relative_tables", search_relative_tables_node)
workflow.add_node("generate_sql_queries", generate_sql_queries_node)
workflow.add_node("validate_sql_query", validate_sql_query_node)
workflow.add_node("handle_sql_feedback", handle_sql_feedback_node)
# Iterative Python Execution Nodes
workflow.add_node("plan_python_analysis", plan_python_analysis_node)
workflow.add_node("generate_python_step_code", generate_python_step_code_node)
workflow.add_node("execute_python_step", execute_python_step_node)
workflow.add_node("validate_python_step", validate_python_step_node)
workflow.add_node("increment_step_index", increment_step_index_node)
workflow.add_node("generate_final_result", generate_final_result_node)

# --- 엣지 조건부 함수 정의 ---
def decide_sql_retry(state: AgentState):
    print("---엣지: SQL 재시도 결정---")
    if state["retry_count"] > state["max_retries"]:
        return "end_with_error"
    else:
        return "generate_sql_queries"

# --- 엣지 추가 ---
workflow.set_entry_point("search_relative_tables")
workflow.add_edge("search_relative_tables", "generate_sql_queries")
workflow.add_edge("generate_sql_queries", "validate_sql_query")

# SQL 검증 결과에 따른 조건부 엣지
workflow.add_conditional_edges(
    "validate_sql_query",
    decide_sql_revalidation,
    {
        "plan_python_analysis": "plan_python_analysis",  # Use iterative approach
        "handle_sql_feedback": "handle_sql_feedback",
    },
)

# SQL 피드백 처리 후 재시도 여부 결정 엣지
workflow.add_conditional_edges(
    "handle_sql_feedback",
    decide_sql_retry,
    {
        "generate_sql_queries": "generate_sql_queries", # 재시도
        "end_with_error": "end_with_error",             # 재시도 횟수 초과 시
    },
)

# Iterative Python Execution Workflow
workflow.add_edge("plan_python_analysis", "generate_python_step_code")
workflow.add_edge("generate_python_step_code", "execute_python_step")
workflow.add_edge("execute_python_step", "validate_python_step")

workflow.add_conditional_edges(
    "validate_python_step",
    check_step_result,
    {
        "next_step": "increment_step_index",  # Increment and move to next step
        "retry_step": "generate_python_step_code",  # Retry: regenerate code
        "finalize": "generate_final_result",  # All steps complete
    },
)

workflow.add_edge("increment_step_index", "generate_python_step_code")

# 최종 결과 생성 및 오류 종료 엣지
workflow.add_edge("generate_final_result", END)
workflow.add_node("end_with_error", generate_final_result_node) # 최종 오류 처리 노드
workflow.add_edge("end_with_error", END)

# 그래프 컴파일
app = workflow.compile()


if __name__ == "__main__":
    user_query = input("무엇을 도와드릴까요? ")
    
    # 디버그: 입력값 확인
    # print(f"\n[DEBUG] 받은 질문: {user_query}")
    # print(f"[DEBUG] 질문 길이: {len(user_query)}자")
    # print(f"[DEBUG] 질문 repr: {repr(user_query)}\n")
    
    initial_state = {
        "user_query": user_query,
        "rag_schema_context": STATE["rag_schema_context"],
        "retry_count": 0,
        "max_retries": 3,
        "sql_feedback": "",
        "python_error_feedback": ""
    }
    final_state = app.invoke(initial_state, config={"recursion_limit": 150})
    print("\n--- 최종 결과 ---")
    print(final_state.get("final_result", "오류로 인해 최종 결과를 생성하지 못했습니다."))
