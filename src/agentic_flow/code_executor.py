"""
Code Executor Node for Safe Python Code Execution

This module implements CodeExecutorNode which safely executes Python code
in a sandboxed environment with resource limits and security restrictions.
"""

import sys
import threading
import queue
import traceback
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Windows에서는 signal 모듈 제한적으로 사용
try:
    import signal
    SIGNAL_AVAILABLE = True
except ImportError:
    SIGNAL_AVAILABLE = False

# resource 모듈은 Unix/Linux 전용
try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from .state import GraphState, QueryIntent
from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CodeExecutionResult:
    """Python 코드 실행 결과"""
    success: bool
    result: Any
    error_message: Optional[str]
    execution_time: float
    memory_used_mb: Optional[float]
    output: Optional[str]
    stdout: Optional[str]
    stderr: Optional[str]


class CodeExecutorNode:
    """
    안전한 Python 코드 실행 노드 (샌드박스 환경)
    
    생성된 Python 코드를 안전하게 실행하며, 리소스 제한과 보안 메커니즘을 제공합니다.
    """
    
    # 리소스 제한 설정
    MAX_EXECUTION_TIME = 30.0  # 최대 실행 시간 (초)
    MAX_MEMORY_MB = 512  # 최대 메모리 사용량 (MB)
    MAX_OUTPUT_SIZE = 10 * 1024 * 1024  # 최대 출력 크기 (10MB)
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        CodeExecutorNode 초기화
        
        Args:
            config: 설정 딕셔너리
                - max_execution_time: 최대 실행 시간 (초)
                - max_memory_mb: 최대 메모리 사용량 (MB)
        """
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)
        
        # 리소스 제한 설정
        self.max_execution_time = self.config.get("max_execution_time", self.MAX_EXECUTION_TIME)
        self.max_memory_mb = self.config.get("max_memory_mb", self.MAX_MEMORY_MB)
        
        # 허용된 모듈 (PythonCodeGeneratorNode와 일치)
        self.allowed_modules = {
            'pandas', 'pd', 'numpy', 'np', 'math', 'datetime',
            'collections', 'json', 're', 'itertools', 'functools', 'operator'
        }
        
        # Windows에서는 resource 모듈이 제한적으로 동작하므로 플랫폼 확인
        self.is_windows = sys.platform == 'win32'
        
        if not PANDAS_AVAILABLE:
            self.logger.warning("Pandas/NumPy not available. Python code execution may be limited.")
    
    def process(self, state: GraphState) -> GraphState:
        """
        상태를 처리하여 Python 코드를 실행합니다.
        
        Args:
            state: 현재 파이프라인 상태
            
        Returns:
            업데이트된 상태
        """
        self.logger.info(
            f"Processing CodeExecutorNode",
            user_id=state.get("user_id"),
            channel_id=state.get("channel_id"),
            query=state.get("user_query", "")[:100]
        )
        
        # COMPLEX_ANALYSIS 의도이고 Python 코드가 있는 경우만 실행
        intent = state.get("intent")
        python_code = state.get("python_code")
        
        if intent != QueryIntent.COMPLEX_ANALYSIS:
            self.logger.info(f"Skipping code execution (intent: {intent})")
            return state
        
        if not python_code:
            self.logger.warning("No Python code found for execution")
            state["python_execution_error"] = "실행할 Python 코드가 없습니다"
            return state
        
        # Python 코드 실행
        try:
            execution_result = self.execute_code(python_code, state)
            
            if execution_result.success:
                # 실행 성공: 결과를 상태에 저장
                state["python_execution_result"] = {
                    "success": True,
                    "result": self._serialize_result(execution_result.result),
                    "execution_time": execution_result.execution_time,
                    "memory_used_mb": execution_result.memory_used_mb,
                    "output": execution_result.output
                }
                state["query_result"] = self._convert_to_query_result(execution_result.result)
                state["confidence_scores"]["python_execution"] = 0.9
                
                self.logger.info(
                    f"Python code executed successfully "
                    f"(time: {execution_result.execution_time:.3f}s, "
                    f"memory: {execution_result.memory_used_mb or 'N/A'}MB)"
                )
            else:
                # 실행 실패: 에러 정보 저장
                state["python_execution_error"] = execution_result.error_message
                state["python_execution_result"] = {
                    "success": False,
                    "error_message": execution_result.error_message,
                    "execution_time": execution_result.execution_time
                }
                state["confidence_scores"]["python_execution"] = 0.0
                
                self.logger.warning(f"Python code execution failed: {execution_result.error_message}")
            
            return state
            
        except Exception as e:
            error_msg = f"Code execution error: {str(e)}"
            self.logger.error(error_msg)
            state["python_execution_error"] = error_msg
            state["python_execution_result"] = {
                "success": False,
                "error_message": error_msg,
                "execution_time": 0.0
            }
            return state
    
    def execute_code(self, code: str, state: GraphState) -> CodeExecutionResult:
        """
        Python 코드를 안전하게 실행합니다.
        
        Args:
            code: 실행할 Python 코드
            state: 현재 상태 (데이터 컨텍스트 포함)
            
        Returns:
            코드 실행 결과
        """
        import time
        import io
        import contextlib
        
        start_time = time.time()
        
        # DB 쿼리 결과를 DataFrame으로 변환
        df = self._prepare_dataframe(state)
        
        # 실행 환경 준비
        execution_globals = self._create_safe_globals(df)
        execution_locals = {}
        
        # stdout/stderr 캡처
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # 타임아웃을 사용한 코드 실행
            with contextlib.redirect_stdout(stdout_capture):
                with contextlib.redirect_stderr(stderr_capture):
                    # Windows에서는 signal.alarm이 작동하지 않으므로 threading 사용
                    if self.is_windows:
                        result = self._execute_with_threading_timeout(
                            code, execution_globals, execution_locals
                        )
                    else:
                        result = self._execute_with_signal_timeout(
                            code, execution_globals, execution_locals
                        )
            
            execution_time = time.time() - start_time
            
            # 결과 확인
            if isinstance(result, Exception):
                error_message = str(result)
                if isinstance(result, TimeoutError):
                    error_message = f"코드 실행 시간 초과 (최대 {self.max_execution_time}초)"
                
                return CodeExecutionResult(
                    success=False,
                    result=None,
                    error_message=error_message,
                    execution_time=execution_time,
                    memory_used_mb=None,
                    output=None,
                    stdout=stdout_capture.getvalue(),
                    stderr=stderr_capture.getvalue()
                )
            
            # 'result' 변수 확인
            if 'result' not in execution_locals and 'result' not in execution_globals:
                # result가 없으면 마지막 표현식의 결과 사용 시도
                return CodeExecutionResult(
                    success=False,
                    result=None,
                    error_message="코드에서 'result' 변수를 생성하지 않았습니다",
                    execution_time=execution_time,
                    memory_used_mb=None,
                    output=None,
                    stdout=stdout_capture.getvalue(),
                    stderr=stderr_capture.getvalue()
                )
            
            result_value = execution_locals.get('result') or execution_globals.get('result')
            
            # 메모리 사용량 측정 (가능한 경우)
            memory_mb = None
            try:
                if not self.is_windows:
                    # Unix/Linux에서만 정확한 메모리 측정 가능
                    import psutil
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
            except Exception:
                pass
            
            # 출력 크기 제한 확인
            output = stdout_capture.getvalue()
            if len(output) > self.MAX_OUTPUT_SIZE:
                output = output[:self.MAX_OUTPUT_SIZE] + "... (truncated)"
            
            return CodeExecutionResult(
                success=True,
                result=result_value,
                error_message=None,
                execution_time=execution_time,
                memory_used_mb=memory_mb,
                output=output,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = f"코드 실행 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
            
            return CodeExecutionResult(
                success=False,
                result=None,
                error_message=error_message,
                execution_time=execution_time,
                memory_used_mb=None,
                output=None,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue()
            )
    
    def _execute_with_threading_timeout(
        self, code: str, globals_dict: Dict, locals_dict: Dict
    ) -> Any:
        """
        Windows 환경에서 threading을 사용한 타임아웃 실행
        
        Args:
            code: 실행할 코드
            globals_dict: 전역 네임스페이스
            locals_dict: 지역 네임스페이스
            
        Returns:
            실행 결과 또는 Exception
        """
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def execute():
            try:
                exec(compile(code, '<string>', 'exec'), globals_dict, locals_dict)
                result_queue.put(None)  # 성공 표시
            except Exception as e:
                exception_queue.put(e)
        
        thread = threading.Thread(target=execute, daemon=True)
        thread.start()
        thread.join(timeout=self.max_execution_time)
        
        if thread.is_alive():
            # 타임아웃 발생
            return TimeoutError("Execution timeout")
        
        if not exception_queue.empty():
            return exception_queue.get()
        
        return None
    
    def _execute_with_signal_timeout(
        self, code: str, globals_dict: Dict, locals_dict: Dict
    ) -> Any:
        """
        Unix/Linux 환경에서 signal을 사용한 타임아웃 실행
        
        Args:
            code: 실행할 코드
            globals_dict: 전역 네임스페이스
            locals_dict: 지역 네임스페이스
            
        Returns:
            실행 결과 또는 Exception
        """
        if not SIGNAL_AVAILABLE:
            # signal이 없으면 threading 방식 사용
            return self._execute_with_threading_timeout(code, globals_dict, locals_dict)
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Execution timeout")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(self.max_execution_time))
        
        try:
            exec(compile(code, '<string>', 'exec'), globals_dict, locals_dict)
            result = None
        except TimeoutError:
            result = TimeoutError("Execution timeout")
        except Exception as e:
            result = e
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        
        return result
    
    def _create_safe_globals(self, df: Optional[Any]) -> Dict[str, Any]:
        """
        안전한 전역 네임스페이스 생성
        
        Args:
            df: pandas DataFrame (없을 수 있음)
            
        Returns:
            안전한 전역 네임스페이스 딕셔너리
        """
        # __builtins__를 제한된 버전으로 설정 (import 문 지원)
        safe_builtins = self._create_safe_builtins()
        # __import__를 제한적으로 허용 (화이트리스트 기반)
        original_import = safe_builtins.get('__import__')
        
        def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
            """제한된 import 함수 (허용된 모듈만)"""
            if name in self.allowed_modules:
                if original_import:
                    return original_import(name, globals, locals, fromlist, level)
                else:
                    return __import__(name, globals, locals, fromlist, level)
            else:
                raise ImportError(f"Module '{name}' is not allowed")
        
        safe_builtins['__import__'] = restricted_import
        
        safe_globals = {
            '__builtins__': safe_builtins,
        }
        
        # 허용된 모듈만 import
        if PANDAS_AVAILABLE and df is not None:
            safe_globals['pd'] = pd
            safe_globals['pandas'] = pd
            safe_globals['df'] = df
            safe_globals['np'] = np
            safe_globals['numpy'] = np
        else:
            # pandas가 없으면 빈 DataFrame 생성
            if PANDAS_AVAILABLE:
                safe_globals['pd'] = pd
                safe_globals['pandas'] = pd
                safe_globals['df'] = pd.DataFrame()
            if 'np' not in safe_globals:
                try:
                    import numpy as np
                    safe_globals['np'] = np
                    safe_globals['numpy'] = np
                except ImportError:
                    pass
        
        # 기타 허용된 모듈
        import math
        import datetime
        import json
        import re
        import itertools
        import functools
        import operator
        from collections import defaultdict, Counter
        
        safe_globals.update({
            'math': math,
            'datetime': datetime,
            'json': json,
            're': re,
            'itertools': itertools,
            'functools': functools,
            'operator': operator,
            'defaultdict': defaultdict,
            'Counter': Counter
        })
        
        return safe_globals
    
    def _create_safe_builtins(self) -> Dict[str, Any]:
        """
        안전한 __builtins__ 딕셔너리 생성 (위험한 함수 제거)
        
        Returns:
            안전한 builtins 딕셔너리
        """
        import builtins
        
        # 허용된 builtins만 포함
        # __import__는 제한적으로 필요하지만 직접 호출은 차단 (import 문으로만 사용)
        safe_builtins = {
            'abs', 'all', 'any', 'bool', 'chr', 'dict', 'dir', 'divmod',
            'enumerate', 'filter', 'float', 'format', 'frozenset', 'getattr',
            'hasattr', 'hash', 'hex', 'int', 'isinstance', 'issubclass',
            'iter', 'len', 'list', 'map', 'max', 'min', 'next', 'oct',
            'ord', 'pow', 'print', 'range', 'repr', 'reversed', 'round',
            'set', 'slice', 'sorted', 'str', 'sum', 'tuple', 'type', 'zip',
            'True', 'False', 'None', 'Ellipsis', 'NotImplemented',
            '__import__'  # import 문에서 필요 (하지만 직접 호출은 안전하게 처리)
        }
        
        safe_dict = {}
        for name in safe_builtins:
            if hasattr(builtins, name):
                safe_dict[name] = getattr(builtins, name)
        
        return safe_dict
    
    def _prepare_dataframe(self, state: GraphState) -> Optional[Any]:
        """
        상태에서 데이터를 가져와 DataFrame으로 변환
        
        Args:
            state: 현재 상태
            
        Returns:
            pandas DataFrame 또는 None
        """
        if not PANDAS_AVAILABLE:
            return None
        
        # 기존 쿼리 결과가 있으면 DataFrame으로 변환
        query_result = state.get("query_result", [])
        if query_result:
            try:
                df = pd.DataFrame(query_result)
                self.logger.info(f"Converted {len(df)} rows to DataFrame")
                return df
            except Exception as e:
                self.logger.warning(f"Failed to convert query result to DataFrame: {e}")
        
        # SQL 쿼리 결과가 있으면 실행하여 DataFrame 생성
        sql_query = state.get("sql_query")
        if sql_query:
            try:
                from core.db import execute_query
                query_result = execute_query(sql_query, readonly=True)
                if isinstance(query_result, list) and query_result:
                    df = pd.DataFrame(query_result)
                    self.logger.info(f"Executed SQL and converted {len(df)} rows to DataFrame")
                    return df
            except Exception as e:
                self.logger.warning(f"Failed to execute SQL for DataFrame: {e}")
        
        return None
    
    def _serialize_result(self, result: Any) -> Any:
        """
        실행 결과를 직렬화 가능한 형태로 변환
        
        Args:
            result: 실행 결과 (DataFrame, dict, list 등)
            
        Returns:
            직렬화 가능한 결과
        """
        if result is None:
            return None
        
        try:
            # pandas DataFrame/Series
            if PANDAS_AVAILABLE:
                if isinstance(result, pd.DataFrame):
                    return result.to_dict('records')
                elif isinstance(result, pd.Series):
                    return result.to_dict()
            
            # numpy 배열
            if isinstance(result, (np.ndarray, np.generic)):
                return result.tolist()
            
            # 기본 타입 (dict, list, str, int, float, bool)
            if isinstance(result, (dict, list, str, int, float, bool)):
                return result
            
            # 그 외는 문자열로 변환
            return str(result)
            
        except Exception as e:
            self.logger.warning(f"Failed to serialize result: {e}")
            return str(result)
    
    def _convert_to_query_result(self, result: Any) -> List[Dict[str, Any]]:
        """
        실행 결과를 query_result 형식으로 변환
        
        Args:
            result: 실행 결과
            
        Returns:
            query_result 형식의 리스트
        """
        serialized = self._serialize_result(result)
        
        if isinstance(serialized, list):
            return serialized
        elif isinstance(serialized, dict):
            return [serialized]
        else:
            return [{"result": serialized}]

