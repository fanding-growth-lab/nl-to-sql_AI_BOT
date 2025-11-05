"""
Hybrid Query Processor for Routing Between SQL and Python Paths

This module implements HybridQueryProcessor which manages routing between
SQL and Python processing paths based on query intent classification.
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass

from .state import QueryIntent
from core.logging import get_logger

logger = get_logger(__name__)


class ProcessingPath(Enum):
    """처리 경로 타입"""
    SQL = "sql"
    PYTHON = "python"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


@dataclass
class RoutingDecision:
    """라우팅 결정 결과"""
    path: ProcessingPath
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]


class HybridQueryProcessor:
    """
    SQL 경로와 Python 경로 간의 라우팅을 관리하는 프로세서
    
    QueryIntentClassifier 결과에 따라 적절한 처리 경로를 선택하고,
    처리 상태를 추적하며 오류 발생 시 대체 경로를 시도합니다.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        HybridQueryProcessor 초기화
        
        Args:
            config: 설정 딕셔너리
                - sql_confidence_threshold: SQL 경로 선택 최소 신뢰도
                - python_confidence_threshold: Python 경로 선택 최소 신뢰도
                - enable_fallback: 폴백 활성화 여부
        """
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)
        
        self.sql_confidence_threshold = self.config.get("sql_confidence_threshold", 0.7)
        self.python_confidence_threshold = self.config.get("python_confidence_threshold", 0.8)
        self.enable_fallback = self.config.get("enable_fallback", True)
        
        # 처리 상태 추적
        self._processing_states: Dict[str, Dict[str, Any]] = {}
    
    def route_query(self, intent: QueryIntent, intent_confidence: float) -> RoutingDecision:
        """
        쿼리를 적절한 처리 경로로 라우팅
        
        Args:
            intent: 분류된 의도
            intent_confidence: 의도 분류 신뢰도
            
        Returns:
            라우팅 결정 결과
        """
        # SIMPLE_AGGREGATION: SQL 경로
        if intent == QueryIntent.SIMPLE_AGGREGATION:
            return RoutingDecision(
                path=ProcessingPath.SQL,
                confidence=intent_confidence,
                reasoning=f"SIMPLE_AGGREGATION 의도: SQL 경로가 최적",
                metadata={
                    "intent": intent.value,
                    "intent_confidence": intent_confidence,
                    "path_selected": "sql"
                }
            )
        
        # COMPLEX_ANALYSIS: Python 경로
        if intent == QueryIntent.COMPLEX_ANALYSIS:
            return RoutingDecision(
                path=ProcessingPath.PYTHON,
                confidence=intent_confidence,
                reasoning=f"COMPLEX_ANALYSIS 의도: Python 경로가 최적",
                metadata={
                    "intent": intent.value,
                    "intent_confidence": intent_confidence,
                    "path_selected": "python"
                }
            )
        
        # 기타 의도: 기본적으로 SQL 경로 (하위 호환성)
        return RoutingDecision(
            path=ProcessingPath.SQL,
            confidence=0.5,
            reasoning=f"Unknown intent ({intent.value}): Defaulting to SQL path",
            metadata={
                "intent": intent.value,
                "intent_confidence": intent_confidence,
                "path_selected": "sql",
                "default_route": True
            }
        )
    
    def should_try_fallback(
        self, 
        primary_path: ProcessingPath, 
        primary_error: Optional[str],
        intent: QueryIntent
    ) -> bool:
        """
        폴백 경로 시도 여부 결정
        
        Args:
            primary_path: 주요 처리 경로
            primary_error: 주요 경로에서 발생한 오류
            intent: 쿼리 의도
            
        Returns:
            폴백 시도 여부
        """
        if not self.enable_fallback:
            return False
        
        if not primary_error:
            return False
        
        # SIMPLE_AGGREGATION은 SQL만 사용 (폴백 없음)
        if intent == QueryIntent.SIMPLE_AGGREGATION:
            return False
        
        # COMPLEX_ANALYSIS는 Python 실패 시 SQL 폴백 가능
        if intent == QueryIntent.COMPLEX_ANALYSIS and primary_path == ProcessingPath.PYTHON:
            return True
        
        return False
    
    def get_fallback_path(self, primary_path: ProcessingPath) -> Optional[ProcessingPath]:
        """
        폴백 경로 가져오기
        
        Args:
            primary_path: 주요 처리 경로
            
        Returns:
            폴백 경로 또는 None
        """
        if primary_path == ProcessingPath.PYTHON:
            return ProcessingPath.SQL
        elif primary_path == ProcessingPath.SQL:
            return ProcessingPath.PYTHON
        else:
            return None
    
    def track_processing_state(self, session_id: str, state_info: Dict[str, Any]) -> None:
        """
        처리 상태 추적
        
        Args:
            session_id: 세션 ID
            state_info: 상태 정보
        """
        self._processing_states[session_id] = {
            **state_info,
            "timestamp": self._get_timestamp()
        }
    
    def get_processing_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        처리 상태 가져오기
        
        Args:
            session_id: 세션 ID
            
        Returns:
            상태 정보 또는 None
        """
        return self._processing_states.get(session_id)
    
    def clear_processing_state(self, session_id: str) -> None:
        """
        처리 상태 정리
        
        Args:
            session_id: 세션 ID
        """
        if session_id in self._processing_states:
            del self._processing_states[session_id]
    
    def _get_timestamp(self) -> float:
        """현재 타임스탬프 가져오기"""
        import time
        return time.time()

