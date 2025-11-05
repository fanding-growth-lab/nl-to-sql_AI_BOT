"""
Result Integrator Node for Hybrid Query Processing

This module implements ResultIntegratorNode which integrates results from
both SQL and Python processing paths.
"""

from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .state import GraphState, QueryIntent
from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class IntegrationResult:
    """결과 통합 결과"""
    success: bool
    integrated_result: Any
    source: str  # "sql", "python", "hybrid"
    confidence: float
    execution_metadata: Dict[str, Any]


class ResultIntegratorNode:
    """
    SQL과 Python 경로의 결과를 통합하는 노드
    
    두 경로의 결과를 수집하고, 형식을 정규화하며, 필요시 충돌을 해결합니다.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        ResultIntegratorNode 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)
    
    def process(self, state: GraphState) -> GraphState:
        """
        상태를 처리하여 SQL과 Python 결과를 통합합니다.
        
        Args:
            state: 현재 파이프라인 상태
            
        Returns:
            업데이트된 상태
        """
        self.logger.info(
            f"Processing ResultIntegratorNode",
            user_id=state.get("user_id"),
            channel_id=state.get("channel_id"),
            query=state.get("user_query", "")[:100]
        )
        
        intent = state.get("intent")
        
        # SIMPLE_AGGREGATION: SQL 결과만 사용
        if intent == QueryIntent.SIMPLE_AGGREGATION:
            sql_result = state.get("query_result", [])
            if sql_result:
                state["integrated_result"] = {
                    "success": True,
                    "result": sql_result,
                    "source": "sql",
                    "confidence": state.get("confidence_scores", {}).get("sql_execution", 0.8),
                    "execution_metadata": {
                        "sql_used": state.get("final_sql"),
                        "execution_time": state.get("processing_time", 0.0)
                    }
                }
                self.logger.info("Integrated SQL result for SIMPLE_AGGREGATION")
                return state
        
        # COMPLEX_ANALYSIS: Python 결과 우선, 없으면 SQL 결과 사용
        if intent == QueryIntent.COMPLEX_ANALYSIS:
            python_result = state.get("python_execution_result")
            sql_result = state.get("query_result", [])
            
            if python_result and python_result.get("success"):
                # Python 실행 성공: Python 결과 사용
                python_data = python_result.get("result")
                state["integrated_result"] = {
                    "success": True,
                    "result": python_data,
                    "source": "python",
                    "confidence": state.get("confidence_scores", {}).get("python_execution", 0.9),
                    "execution_metadata": {
                        "python_code_used": True,
                        "execution_time": python_result.get("execution_time", 0.0),
                        "memory_used_mb": python_result.get("memory_used_mb"),
                        "sql_fallback_available": len(sql_result) > 0
                    }
                }
                self.logger.info("Integrated Python result for COMPLEX_ANALYSIS")
            elif sql_result:
                # Python 실행 실패 시 SQL 결과로 폴백
                state["integrated_result"] = {
                    "success": True,
                    "result": sql_result,
                    "source": "sql_fallback",
                    "confidence": state.get("confidence_scores", {}).get("sql_execution", 0.7) * 0.8,  # 폴백이므로 신뢰도 낮춤
                    "execution_metadata": {
                        "python_execution_failed": True,
                        "python_error": state.get("python_execution_error"),
                        "sql_fallback_used": True,
                        "sql_used": state.get("final_sql")
                    }
                }
                self.logger.warning("Python execution failed, using SQL fallback result")
            else:
                # 둘 다 실패
                state["integrated_result"] = {
                    "success": False,
                    "result": None,
                    "source": "none",
                    "confidence": 0.0,
                    "execution_metadata": {
                        "python_execution_failed": True,
                        "python_error": state.get("python_execution_error"),
                        "sql_result_available": False
                    }
                }
                self.logger.error("Both Python and SQL execution failed")
            
            return state
        
        # 기타 의도: SQL 결과 사용
        sql_result = state.get("query_result", [])
        if sql_result:
            state["integrated_result"] = {
                "success": True,
                "result": sql_result,
                "source": "sql",
                "confidence": state.get("confidence_scores", {}).get("sql_execution", 0.8),
                "execution_metadata": {
                    "sql_used": state.get("final_sql"),
                    "execution_time": state.get("processing_time", 0.0)
                }
            }
        else:
            state["integrated_result"] = {
                "success": False,
                "result": None,
                "source": "none",
                "confidence": 0.0,
                "execution_metadata": {}
            }
        
        return state
    
    def _normalize_result(self, result: Any) -> List[Dict[str, Any]]:
        """
        결과를 정규화된 형식으로 변환
        
        Args:
            result: 원본 결과 (DataFrame, dict, list 등)
            
        Returns:
            정규화된 결과 리스트
        """
        if result is None:
            return []
        
        # 이미 리스트인 경우
        if isinstance(result, list):
            # 리스트 내 항목이 dict인지 확인
            if result and isinstance(result[0], dict):
                return result
            else:
                # 다른 타입이면 dict로 변환
                return [{"value": item} for item in result]
        
        # dict인 경우
        if isinstance(result, dict):
            return [result]
        
        # 그 외는 문자열로 변환
        return [{"result": str(result)}]
    
    def _resolve_conflict(
        self, sql_result: Any, python_result: Any, confidence_sql: float, confidence_python: float
    ) -> tuple[Any, str, float]:
        """
        SQL과 Python 결과 간 충돌 해결
        
        Args:
            sql_result: SQL 실행 결과
            python_result: Python 실행 결과
            confidence_sql: SQL 결과 신뢰도
            confidence_python: Python 결과 신뢰도
            
        Returns:
            (선택된 결과, 소스, 신뢰도) 튜플
        """
        # 신뢰도가 높은 결과 선택
        if confidence_python > confidence_sql:
            return python_result, "python", confidence_python
        else:
            return sql_result, "sql", confidence_sql

