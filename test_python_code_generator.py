from test_config import STATE, run_data_gathering, run_generate_python_code

def process(state: dict) -> dict:
    full_results, results_file_path, sql_queries_file_path = run_data_gathering(state)
    print(f"Full SQL Results saved to: {results_file_path}")
    print(f"SQL Queries saved to: {sql_queries_file_path}")
    # import json
    # import pandas as pd

    # results_file_path = "sql_query_results/ebe64650-26e5-4d0d-bf9a-21b80d0133e2.json"
    # with open(results_file_path, 'r', encoding='utf-8') as f:
    #     full_results = json.load(f)
    code = run_generate_python_code(state, full_results, results_file_path, sql_queries_file_path)
    print(code)


if __name__ == "__main__":
    process(STATE)
