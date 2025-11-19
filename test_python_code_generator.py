import re
import json
from test_config import (
    STATE, 
    chat, save_result, run_dynamic_code, 
    BUSINESS_RULES_FOR_SQL_GENERATION,
    BUSINESS_RULES_FOR_PYTHON_GENERATION,
    PROMPT_SEARCH_RELATIVE_TABLES,
    PROMPT_GENERATE_SQL_QUERY,
    PROMPT_VALIDATE_SQL_QUERY,
    PROMPT_GENERATE_PYTHON_CODE,    # 따로 검수 X, 에러 발생 시 핸들링
    PROMPT_GENERATE_FINAL_RESULT,
)

def _get_info(state: dict) -> dict:
    assert state.get("user_query") and state.get("entities") and state.get("rag_schema_context")
    return {
        "user_query": state["user_query"],
        "entities": state["entities"],
        "rag_schema_context": state["rag_schema_context"],
    }

def search_relative_tables(user_query: str, rag_schema_context: str):
    try:
        content = PROMPT_SEARCH_RELATIVE_TABLES.format(user_query=user_query, rag_schema_context=rag_schema_context)
        response = chat.send_message(content)
        clean_response = re.sub(r"```(?:json|sql|python)?\s*([\s\S]*?)\s*```", r"\1", response.text).strip()
        result = json.loads(clean_response)
    except Exception:
        result = None
    return result

def generate_sql_queries(business_rules: str):
    try:
        content = PROMPT_GENERATE_SQL_QUERY.format(business_rules=business_rules)
        response = chat.send_message(content)
        clean_response = re.sub(r"```(?:json|sql|python)?\s*([\s\S]*?)\s*```", r"\1", response.text).strip()
        result = json.loads(clean_response)
    except Exception:
        result = None
    return result

def validate_sql_query():
    result = {"is_valid": True, "feedback": ""}
    try:
        content = PROMPT_VALIDATE_SQL_QUERY
        response = chat.send_message(content)
        clean_response = re.sub(r"```(?:json|sql|python)?\s*([\s\S]*?)\s*```", r"\1", response.text).strip()
        result = json.loads(clean_response)
    except Exception:
        result = {"is_valid": False, "feedback": "답변 생성에 실패하였습니다. 다시 시도해주세요."}
    return result

def response_feedback(feedback: str, stage: str):
    result = {"is_valid": True, "feedback": ""}
    try:
        content = f"{feedback}\n\n이 내용을 참고하여 이전 {stage} 작업을 다시 진행해주세요."
        response = chat.send_message(content)
        try:    # NOTE 이 부분 마지막으로 수정
            clean_response = re.sub(r"```(?:json|sql|python)?\s*([\s\S]*?)\s*```", r"\1", response.text).strip()
            save_result(json.loads(clean_response), f"feedback_regeneration.json", True)
        except Exception:
            pass
        content = f"이에 대한 검증 작업을 다시 진행해주세요. 출력 형식에 주의합니다."
        response = chat.send_message(content)
        clean_response = re.sub(r"```(?:json|sql|python)?\s*([\s\S]*?)\s*```", r"\1", response.text).strip()
        result = json.loads(clean_response)
        try:
            save_result(result, f"feedback_validation.json", True)
        except Exception:
            pass
    except Exception:
        result = {"is_valid": False, "feedback": "답변 생성에 실패하였습니다. 다시 시도해주세요."}
    return result

def generate_python_code(business_rules) -> str:
    try:
        content = PROMPT_GENERATE_PYTHON_CODE.format(business_rules=business_rules)
        response = chat.send_message(content)
        clean_response = re.sub(r"```(?:json|sql|python)?\s*([\s\S]*?)\s*```", r"\1", response.text).strip()
        result = clean_response
    except Exception:
        result = ""
    return result

def generate_final_result(python_execution_result: str) -> str:
    try:
        content = PROMPT_GENERATE_FINAL_RESULT.format(python_execution_result=python_execution_result)
        response = chat.send_message(content)
        clean_response = re.sub(r"```(?:json|sql|python)?\s*([\s\S]*?)\s*```", r"\1", response.text).strip()
        result = clean_response
    except Exception:
        result = ""
    return result

def process(state: dict) -> dict:
    info = _get_info(state)

    max_retry = 3

    # --- 관련 있는 테이블 파악 ---

    num_retry = 0
    relative_tables = None
    while relative_tables is None:
        assert num_retry < max_retry
        relative_tables = search_relative_tables(info["user_query"], info["rag_schema_context"])
        num_retry += 1
    save_result(relative_tables, "relative_tables.json", True)

    # --- SQL 쿼리문 작성 ---

    num_retry = 0
    sql_query = None
    while sql_query is None:
        assert num_retry < max_retry
        sql_query = generate_sql_queries(BUSINESS_RULES_FOR_SQL_GENERATION)
        num_retry += 1
    save_result(sql_query, f"sql_query.json", True)

    num_retry = 0
    sql_validation = None
    while sql_validation is None:
        assert num_retry < max_retry
        sql_validation = validate_sql_query()
        num_retry += 1
    save_result(sql_validation, f"sql_validation.json", True)

    if not sql_validation["is_valid"]:
        feedback = sql_validation["feedback"]
        for _ in range(3):
            retry = response_feedback(feedback, "SQL 쿼리문 작성")
            if retry["is_valid"]:
                break
            feedback = retry["feedback"]

    # --- 파이썬 코드 작성 ---

    num_retry = 0
    num_appendix = 1
    
    python_code = ""
    while python_code == "":
        assert num_retry < max_retry
        python_code = generate_python_code(BUSINESS_RULES_FOR_PYTHON_GENERATION)
        num_retry += 1
    save_result(python_code, f"python_code.py", True)

    while True:

        python_execution_result = None
        for _ in range(3):
            try:
                execution_output = run_dynamic_code(python_code)
                python_execution_result = execution_output["captured_output"]
                break
            except Exception as e:
                error_message = f"Python code execution failed: {e}"
                print(error_message)
                feedback = error_message
                content = f"{feedback}\n\n이 내용을 참고하여 파이썬 코드 작성 작업을 다시 진행해주세요. 출력 형식에 주의합니다."
                response = chat.send_message(content)
                python_code = re.sub(r"```(?:json|sql|python)?\s*([\s\S]*?)\s*```", r"\1", response.text).strip()
                save_result(python_code, f"regenerated_python_code.py", True)
        
        STATE["python_execution_result"] = python_execution_result
        
        # --- 최종 응답 생성 ---

        num_retry = 0
        final_result = ""
        while final_result == "":
            assert num_retry < max_retry
            final_result = generate_final_result(STATE["python_execution_result"])
            num_retry += 1
        save_result(final_result, f"final_result.txt", True)

        # --- 추가 질문에 대해 파이썬 코드 생성 ---

        user_input = input()
        if user_input == 'exit': break

        num_retry = 0
        python_code = ""
        while python_code == "":
            assert num_retry < max_retry
            try:
                content = f"""
`추가질문`이 있습니다. 다음 내용을 바탕으로 `1. SQL 쿼리문 작성`, `2. 파이썬 코드 작성` 작업을 진행하고 완성된 형태의 실행가능한 파이썬 코드를 작성합니다.

추가질문:
{user_input}

규칙: 
- 별도의 설명이나 주석, 텍스트를 붙이지 않고 파이썬 코드만 작성합니다.
"""
                response = chat.send_message(content)
                python_code = re.sub(r"```(?:json|sql|python)?\s*([\s\S]*?)\s*```", r"\1", response.text).strip()
            except Exception as e:
                print(f"문제가 발생: {e}")
            num_retry += 1

        if python_code == "":
            break

        save_result(python_code, f"python_code_{num_appendix}.py", True)
        num_appendix += 1

    return final_result



if __name__ == "__main__":
    STATE['user_query'] = input("무엇을 도와드릴까요? ")
    process(STATE)
