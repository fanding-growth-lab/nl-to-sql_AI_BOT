"""
공통 유틸리티 함수 모듈

중복 코드를 제거하고 공통 유틸리티 함수를 제공합니다.
"""

from typing import Union, Optional, List, Dict, Any
from .state import QueryIntent, Entity


def normalize_intent(intent: Union[str, QueryIntent, None]) -> Optional[str]:
    """
    Intent를 문자열로 정규화
    
    AgentState의 intent 필드는 Optional[str]이지만,
    실제로는 QueryIntent enum이 저장될 수도 있습니다.
    이 함수는 두 경우를 모두 처리합니다.
    
    Args:
        intent: QueryIntent enum, 문자열, 또는 None
        
    Returns:
        정규화된 intent 문자열 또는 None
    """
    if intent is None:
        return None
    
    if isinstance(intent, str):
        return intent
    
    if isinstance(intent, QueryIntent):
        return intent.value
    
    if hasattr(intent, 'value'):
        return str(intent.value)
    
    return str(intent)


def is_intent_equal(intent: Union[str, QueryIntent, None], target: QueryIntent) -> bool:
    """
    Intent가 특정 QueryIntent와 일치하는지 확인
    
    Args:
        intent: 비교할 intent (enum, 문자열, 또는 None)
        target: 비교 대상 QueryIntent enum
        
    Returns:
        일치 여부
    """
    normalized = normalize_intent(intent)
    if normalized is None:
        return False
    
    return normalized == target.value


def is_intent_in(intent: Union[str, QueryIntent, None], targets: list[QueryIntent]) -> bool:
    """
    Intent가 여러 QueryIntent 중 하나와 일치하는지 확인
    
    Args:
        intent: 비교할 intent (enum, 문자열, 또는 None)
        targets: 비교 대상 QueryIntent enum 리스트
        
    Returns:
        일치 여부
    """
    normalized = normalize_intent(intent)
    if normalized is None:
        return False
    
    return any(normalized == target.value for target in targets)


def calculate_mapping_confidence(
    entities: List[Entity],
    tables: Optional[List[str]] = None,
    columns: Optional[List[str]] = None
) -> float:
    """
    스키마 매핑 신뢰도 계산
    
    여러 노드에서 중복된 confidence 계산 로직을 통합합니다.
    
    Args:
        entities: 추출된 엔티티 리스트
        tables: 관련 테이블 리스트 (선택)
        columns: 관련 컬럼 리스트 (선택)
        
    Returns:
        신뢰도 점수 (0.0-1.0)
    """
    if not entities:
        return 0.0
    
    # 엔티티 추출 신뢰도의 평균
    avg_entity_confidence = sum(e.confidence for e in entities) / len(entities)
    
    # 관련 테이블/컬럼 발견 보너스
    mapping_bonus = 0.0
    if tables:
        mapping_bonus += 0.2
    if columns:
        mapping_bonus += 0.1
    
    return min(avg_entity_confidence + mapping_bonus, 1.0)


def calculate_sql_confidence(
    result: Dict[str, Any],
    schema_mapping: Optional[Any] = None,
    has_mock: bool = False
) -> float:
    """
    SQL 생성 신뢰도 계산
    
    Args:
        result: SQL 생성 결과 딕셔너리
        schema_mapping: 스키마 매핑 객체 (선택)
        has_mock: 모의 SQL 여부
        
    Returns:
        신뢰도 점수 (0.0-1.0)
    """
    base_confidence = 0.8
    
    # 모의 SQL인 경우 신뢰도 감소
    if has_mock or result.get("mock", False):
        base_confidence *= 0.7
    
    # 스키마 매핑이 있는 경우 보너스
    if schema_mapping:
        # schema_mapping 객체의 relevant_tables 속성 확인
        if hasattr(schema_mapping, 'relevant_tables') and schema_mapping.relevant_tables:
            base_confidence += 0.1
        elif isinstance(schema_mapping, dict) and schema_mapping.get("relevant_tables"):
            base_confidence += 0.1
    
    # 응답 길이에 따른 조정
    response_length = result.get("response_length", 0)
    if response_length > 100:
        base_confidence += 0.05
    
    return min(base_confidence, 1.0)

