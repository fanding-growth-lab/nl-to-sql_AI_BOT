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
    PROMPT_GENERATE_PYTHON_CODE,
    PROMPT_GENERATE_FINAL_RESULT,
    PROMPT_PLAN_PYTHON_ANALYSIS,
    PROMPT_GENERATE_PYTHON_STEP,
    PROMPT_VALIDATE_PYTHON_STEP,
    PROMPT_VALIDATE_PYTHON_EXECUTION,
)
from rule_rag import retrieve_relevant_rules


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
    business_rules = retrieve_relevant_rules(state["user_query"], category="sql")
    
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
    business_rules = retrieve_relevant_rules(state["user_query"], category="sql")

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
        return "generate_python_code"
    else:
        return "handle_sql_feedback"


def handle_sql_feedback_node(state: AgentState):
    print("---노드: SQL 피드백 처리---")
    retry_count = state.get("retry_count", 0) + 1
    feedback = state["sql_validation"]["feedback"]
    print(f"SQL 검증 실패 피드백: {feedback}, 재시도 횟수: {retry_count}")
    return {"retry_count": retry_count, "sql_feedback": feedback}


def generate_python_code_node(state: AgentState):
    print("---노드: Python 코드 생성---")
    prompt = PromptTemplate(
        template=PROMPT_GENERATE_PYTHON_CODE,
        input_variables=["user_query", "relative_tables", "sql_queries", "business_rules"],
    )
    chain = prompt | llm | StrOutputParser()
    feedback = state.get("python_error_feedback", "") or state.get("python_validation_feedback", "")
    if feedback:
        feedback = f"이전에 생성한 결과와 피드백:\n{state.get('python_code')}\n{feedback}"
    # Retrieve relevant business rules for Python
    business_rules = retrieve_relevant_rules(state["user_query"], category="python")

    result = chain.invoke({
        "user_query": state["user_query"],
        "relative_tables": state["relative_tables"],
        "business_rules": business_rules,
        "sql_queries": state["sql_queries"],
        "python_feedback": feedback,
    })
    clean_code = re.sub(r"```(?:python)?\s*([\s\S]*?)\s*```", r"\1", result).strip()
    save_result(clean_code, "python_code.py", True)
    return {"python_code": clean_code, "python_error_feedback": "", "python_validation_feedback": ""} # 피드백 사용 후 초기화


def execute_python_code_node(state: AgentState):
    print("---노드: Python 코드 실행---")
    try:
        execution_output = run_dynamic_code(state["python_code"])
        error = execution_output["error"]
        if error:
            raise
        result = execution_output["captured_output"]
        save_result(result, "python_result.txt", True)
        return {"python_execution_result": result}
    except Exception:
        error_message = f"Python 코드 실행에 실패하였습니다. 에러 발생 라인: {error.__traceback__.tb_lineno}, 에러 타입: {type(error).__name__}, 에러 메시지: {str(error)}"
        save_result(error_message, "python_error.txt", True)
        return {"error": error_message}


def validate_python_execution_node(state: AgentState):
    print("---노드: Python 코드 실행 결과 검증---")
    prompt = PromptTemplate(
        template=PROMPT_VALIDATE_PYTHON_EXECUTION,
        input_variables=["user_query", "python_execution_result", "business_rules"],
    )
    chain = prompt | llm | JsonOutputParser()
    # Retrieve relevant business rules for Python validation
    business_rules = retrieve_relevant_rules(state["user_query"], category="python")

    result = chain.invoke({
        "user_query": state["user_query"],
        "business_rules": business_rules,
        "python_code": state["python_code"],
        "python_execution_result": state["python_execution_result"],
    })
    save_result(result, "python_validation.json", True)
    return {"python_validation": result}


def decide_python_reexecution(state: AgentState):
    print("---엣지: Python 재실행 결정---")
    if state.get("error"):
        return "handle_python_error"
    else:
        return "validate_python_execution" # Python 코드 실행 성공 시 검증 노드로 이동


def handle_python_error_node(state: AgentState):
    print("---노드: Python 오류 처리---")
    retry_count = state.get("retry_count", 0) + 1
    error_message = state["error"]
    print(f"Python 코드 실행 실패 피드백: {error_message}, 재시도 횟수: {retry_count}")
    return {"retry_count": retry_count, "python_error_feedback": error_message, "error": ""} # 오류 메시지 사용 후 초기화


def decide_python_validation(state: AgentState):
    print("---엣지: Python 검증 결과 결정---")
    if state["python_validation"]["is_valid"]:
        return "generate_final_result"
    else:
        return "handle_python_validation_feedback"


def handle_python_validation_feedback_node(state: AgentState):
    print("---노드: Python 검증 피드백 처리---")
    retry_count = state.get("retry_count", 0) + 1
    feedback = state["python_validation"]["feedback"]
    print(f"Python 검증 실패 피드백: {feedback}, 재시도 횟수: {retry_count}")
    return {"retry_count": retry_count, "python_validation_feedback": feedback}


def plan_python_analysis_node(state: AgentState):
    print("---노드: Python 분석 계획 수립---")
    prompt = PromptTemplate(
        template=PROMPT_PLAN_PYTHON_ANALYSIS,
        input_variables=["user_query", "relative_tables", "business_rules", "sql_queries"],
    )
    chain = prompt | llm | JsonOutputParser()
    business_rules = retrieve_relevant_rules(state["user_query"], category="python")
    
    result = chain.invoke({
        "user_query": state["user_query"],
        "relative_tables": state["relative_tables"],
        "business_rules": business_rules,
        "sql_queries": state["sql_queries"]
    })
    
    print(f"수립된 계획: {result}")
    save_result(result, "python_plan.json", True)
    
    return {
        "python_plan": result,
        "current_step_index": 0,
        "python_context": {"__builtins__": __builtins__}, # 초기 컨텍스트
        "python_code": "" # 전체 코드 누적용
    }


def generate_python_step_code_node(state: AgentState):
    """Generate code for the current step."""
    current_index = state["current_step_index"]
    plan = state["python_plan"]
    current_step = plan[current_index]
    
    print(f"---노드: Python 단계 코드 생성 ({current_index + 1}/{len(plan)}) : {current_step}---")
    
    prompt = PromptTemplate(
        template=PROMPT_GENERATE_PYTHON_STEP,
        input_variables=["user_query", "business_rules", "python_plan", "current_step", "python_context", "step_feedback"],
    )
    chain = prompt | llm | StrOutputParser()
    business_rules = retrieve_relevant_rules(state["user_query"], category="python")
    
    # Context summary for prompt
    context_summary = {k: type(v).__name__ for k, v in state["python_context"].items() if k != "__builtins__"}
    
    # Get feedback if retry
    step_feedback = state.get("step_validation", {}).get("feedback", "")
    if step_feedback:
        step_feedback = f"이전 시도 피드백:\n{step_feedback}"
    
    code = chain.invoke({
        "user_query": state["user_query"],
        "business_rules": business_rules,
        "python_plan": plan,
        "current_step": current_step,
        "python_context": context_summary,
        "step_feedback": step_feedback
    })
    
    clean_code = re.sub(r"```(?:python)?\s*([\s\S]*?)\s*```", r"\1", code).strip()
    print(f"생성된 코드:\n{clean_code}")
    
    return {
        "step_code": clean_code
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
        input_variables=["user_query", "business_rules", "current_step", "step_code", "step_result"],
    )
    chain = prompt | llm | JsonOutputParser()
    business_rules = retrieve_relevant_rules(state["user_query"], category="python")
    
    result = chain.invoke({
        "user_query": state["user_query"],
        "business_rules": business_rules,
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
        if next_index < len(state["python_plan"]):
            print(f"다음 단계로 이동: {next_index + 1}/{len(state['python_plan'])}")
            return "next_step"
        else:
            print("모든 단계 완료, 최종 결과 생성")
            return "finalize"
    else:
        print(f"현재 단계 재시도: {state['current_step_index'] + 1}/{len(state['python_plan'])}")
        return "retry_step"


def increment_step_index_node(state: AgentState):
    """Move to next step by incrementing the index."""
    next_index = state["current_step_index"] + 1
    print(f"Step index incremented: {state['current_step_index']} -> {next_index}")
    return {"current_step_index": next_index}


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
# Legacy nodes (kept for compatibility if needed)
workflow.add_node("generate_python_code", generate_python_code_node)
workflow.add_node("execute_python_code", execute_python_code_node)
workflow.add_node("validate_python_execution", validate_python_execution_node) # 추가
workflow.add_node("handle_python_error", handle_python_error_node)
workflow.add_node("handle_python_validation_feedback", handle_python_validation_feedback_node) # 추가
workflow.add_node("generate_final_result", generate_final_result_node)

# --- 엣지 조건부 함수 정의 ---
def decide_sql_retry(state: AgentState):
    print("---엣지: SQL 재시도 결정---")
    if state["retry_count"] > state["max_retries"]:
        return "end_with_error"
    else:
        return "generate_sql_queries"

def decide_python_retry(state: AgentState):
    print("---엣지: Python 재시도 결정---")
    if state["retry_count"] > state["max_retries"]:
        return "end_with_error"
    else:
        return "generate_python_code"

def decide_python_validation_retry(state: AgentState):
    print("---엣지: Python 검증 재시도 결정---")
    if state["retry_count"] > state["max_retries"]:
        return "end_with_error"
    else:
        return "generate_python_code" # 검증 실패 시 Python 코드 재생성으로 이동

# --- 엣지 추가 ---
workflow.set_entry_point("search_relative_tables")
workflow.add_edge("search_relative_tables", "generate_sql_queries")
workflow.add_edge("generate_sql_queries", "validate_sql_query")

# SQL 검증 결과에 따른 조건부 엣지
workflow.add_conditional_edges(
    "validate_sql_query",
    decide_sql_revalidation,
    {
        "generate_python_code": "plan_python_analysis",  # Use iterative approach
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

# Legacy Python code generation workflow (kept for reference, but not used)
workflow.add_edge("generate_python_code", "execute_python_code")

# Python 코드 실행 결과에 따른 조건부 엣지 (수정)
workflow.add_conditional_edges(
    "execute_python_code",
    decide_python_reexecution,
    {
        "validate_python_execution": "validate_python_execution", # 실행 성공 시 검증 노드로
        "handle_python_error": "handle_python_error",
    },
)

# Python 오류 처리 후 재시도 여부 결정 엣지
workflow.add_conditional_edges(
    "handle_python_error",
    decide_python_retry,
    {
        "generate_python_code": "generate_python_code", # 재시도
        "end_with_error": "end_with_error",             # 재시도 횟수 초과 시
    },
)

# Python 검증 결과에 따른 조건부 엣지 (추가)
workflow.add_conditional_edges(
    "validate_python_execution",
    decide_python_validation,
    {
        "generate_final_result": "generate_final_result",
        "handle_python_validation_feedback": "handle_python_validation_feedback",
    },
)

# Python 검증 피드백 처리 후 재시도 여부 결정 엣지 (추가)
workflow.add_conditional_edges(
    "handle_python_validation_feedback",
    decide_python_validation_retry,
    {
        "generate_python_code": "generate_python_code", # 재시도
        "end_with_error": "end_with_error",             # 재시도 횟수 초과 시
    },
)

# 최종 결과 생성 및 오류 종료 엣지
workflow.add_edge("generate_final_result", END)
workflow.add_node("end_with_error", generate_final_result_node) # 최종 오류 처리 노드
workflow.add_edge("end_with_error", END)

# 그래프 컴파일
app = workflow.compile()


if __name__ == "__main__":
    user_query = input("무엇을 도와드릴까요? ")
    
    # 디버그: 입력값 확인
    print(f"\n[DEBUG] 받은 질문: {user_query}")
    print(f"[DEBUG] 질문 길이: {len(user_query)}자")
    print(f"[DEBUG] 질문 repr: {repr(user_query)}\n")
    
    initial_state = {
        "user_query": user_query,
        "rag_schema_context": STATE["rag_schema_context"],
        "retry_count": 0,
        "max_retries": 3,
        "sql_feedback": "",
        "python_error_feedback": ""
    }
    final_state = app.invoke(initial_state)
    print("\n--- 최종 결과 ---")
    print(final_state.get("final_result", "오류로 인해 최종 결과를 생성하지 못했습니다."))
