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
                # 쿼리 결과를 캐시에 저장 (하이브리드 접근)
                self._save_query_result_to_cache(state)
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
                # Python 결과를 캐시에 저장 (하이브리드 접근)
                self._save_query_result_to_cache(state)
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
        
        # 쿼리 결과를 캐시에 저장 (하이브리드 접근)
        if state.get("integrated_result", {}).get("success"):
            self._save_query_result_to_cache(state)
        
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
    
    def _save_query_result_to_cache(self, state: GraphState) -> None:
        """
        쿼리 결과를 구조화된 캐시에 저장 (하이브리드 접근)
        
        Args:
            state: 현재 파이프라인 상태
        """
        try:
            import hashlib
            import json
            from datetime import datetime
            
            # 캐시 초기화
            if state.get("query_result_cache") is None:
                state["query_result_cache"] = {}
            
            # 쿼리 패턴 기반 키 생성
            user_query = state.get("user_query", "")
            intent_raw = state.get("intent", "")
            # QueryIntent enum을 문자열로 변환
            intent = intent_raw.value if hasattr(intent_raw, 'value') else str(intent_raw)
            sql_query = state.get("final_sql") or state.get("sql_query")
            python_code = state.get("python_code")
            
            # 키 생성: 쿼리 패턴 + 파라미터 조합
            cache_key = self._generate_cache_key(user_query, intent, state)
            
            # 캐시 데이터 구성
            integrated_result = state.get("integrated_result", {})
            cache_data = {
                "query": user_query,
                "intent": intent,
                "result": integrated_result.get("result", []),
                "sql": sql_query,
                "python_code": python_code,
                "params": state.get("sql_params", {}),
                "timestamp": datetime.now().isoformat(),
                "data_summary": state.get("data_summary"),  # data_summarization 노드에서 생성된 요약
                "source": integrated_result.get("source", "unknown")
            }
            
            # 캐시에 저장 (최대 10개까지만 유지)
            cache = state["query_result_cache"]
            if len(cache) >= 10:
                # 가장 오래된 항목 제거 (timestamp 기준)
                oldest_key = min(cache.keys(), key=lambda k: cache[k].get("timestamp", ""))
                del cache[oldest_key]
                self.logger.debug(f"Removed oldest cache entry: {oldest_key}")
            
            cache[cache_key] = cache_data
            self.logger.info(f"Saved query result to cache with key: {cache_key}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save query result to cache: {e}")
    
    def _generate_cache_key(self, user_query: str, intent: str, state: GraphState) -> str:
        """
        쿼리 패턴 기반 캐시 키 생성
        
        Args:
            user_query: 사용자 쿼리
            intent: 인텐트
            state: 현재 상태
            
        Returns:
            캐시 키 (문자열)
        """
        import re
        import hashlib
        
        # 쿼리에서 패턴 추출 (날짜, TOP N, 크리에이터 등)
        query_lower = user_query.lower()
        
        # 패턴 추출
        patterns = []
        
        # 월/년 추출
        month_match = re.search(r'(\d+)\s*월', query_lower)
        if month_match:
            patterns.append(f"month_{month_match.group(1)}")
        
        year_match = re.search(r'(\d+)\s*년', query_lower)
        if year_match:
            patterns.append(f"year_{year_match.group(1)}")
        
        # TOP N 추출
        top_match = re.search(r'top\s*(\d+)', query_lower) or re.search(r'(\d+)\s*위', query_lower)
        if top_match:
            patterns.append(f"top_{top_match.group(1)}")
        
        # 키워드 추출 (매출, 신규 회원, 정산금액, etc.)
        keywords = []
        if any(kw in query_lower for kw in ["매출", "sales", "정산", "정산금액", "결제", "수익", "매출액", "금액", "수익금", "revenue", "payment", "settlement"]):
            keywords.append("sales")
        if "신규" in query_lower or "new" in query_lower:
            keywords.append("new_members")
        if "크리에이터" in query_lower or "creator" in query_lower:
            keywords.append("creator")
        
        # 파라미터 기반 키 (creator_no 등)
        sql_params = state.get("sql_params", {})
        param_keys = []
        if "creator_no" in sql_params:
            param_keys.append(f"creator_{sql_params['creator_no']}")
        
        # 키 조합
        key_parts = [intent] + patterns + keywords + param_keys
        
        # 키가 너무 길면 해시 사용
        key_string = "_".join(key_parts) if key_parts else user_query[:50]
        if len(key_string) > 100:
            key_hash = hashlib.md5(key_string.encode()).hexdigest()[:8]
            return f"{intent}_{key_hash}"
        
        return key_string

