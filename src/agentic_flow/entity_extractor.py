#!/usr/bin/env python3
"""
엔티티 추출 시스템
자연어 쿼리에서 데이터베이스 관련 엔티티를 추출하는 고급 시스템
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """엔티티 타입 열거형"""
    TABLE = "table"
    COLUMN = "column"
    VALUE = "value"
    OPERATION = "operation"
    CONDITION = "condition"
    AGGREGATION = "aggregation"


@dataclass
class ExtractedEntity:
    """추출된 엔티티 정보"""
    text: str
    type: EntityType
    confidence: float
    start_pos: int
    end_pos: int
    context: str = ""
    alternatives: List[str] = None


class EntityExtractor:
    """
    자연어 쿼리에서 데이터베이스 관련 엔티티를 추출하는 시스템
    """
    
    def __init__(self):
        """EntityExtractor 초기화"""
        self.table_patterns = self._build_table_patterns()
        self.column_patterns = self._build_column_patterns()
        self.value_patterns = self._build_value_patterns()
        self.operation_patterns = self._build_operation_patterns()
        self.aggregation_patterns = self._build_aggregation_patterns()
        
        logger.info("EntityExtractor initialized")
    
    def _build_table_patterns(self) -> List[Tuple[str, EntityType, float]]:
        """테이블명 추출 패턴 구축"""
        return [
            # 직접적인 테이블명 언급
            (r'\b(t_\w+)\b', EntityType.TABLE, 0.9),
            (r'\b(users?|members?|customers?)\b', EntityType.TABLE, 0.8),
            (r'\b(orders?|payments?|transactions?)\b', EntityType.TABLE, 0.8),
            (r'\b(products?|items?|goods?)\b', EntityType.TABLE, 0.8),
            (r'\b(posts?|articles?|contents?)\b', EntityType.TABLE, 0.8),
            (r'\b(communities?|forums?|boards?)\b', EntityType.TABLE, 0.8),
            (r'\b(events?|campaigns?|promotions?)\b', EntityType.TABLE, 0.8),
            
            # 복합 테이블명
            (r'\b(user\s+profiles?|member\s+profiles?)\b', EntityType.TABLE, 0.7),
            (r'\b(order\s+items?|payment\s+methods?)\b', EntityType.TABLE, 0.7),
            (r'\b(community\s+posts?|forum\s+threads?)\b', EntityType.TABLE, 0.7),
        ]
    
    def _build_column_patterns(self) -> List[Tuple[str, EntityType, float]]:
        """컬럼명 추출 패턴 구축"""
        return [
            # 기본 컬럼명
            (r'\b(id|no|num|number)\b', EntityType.COLUMN, 0.9),
            (r'\b(name|title|subject)\b', EntityType.COLUMN, 0.8),
            (r'\b(email|mail|e-mail)\b', EntityType.COLUMN, 0.9),
            (r'\b(phone|tel|telephone)\b', EntityType.COLUMN, 0.8),
            (r'\b(address|addr|location)\b', EntityType.COLUMN, 0.8),
            (r'\b(date|time|created|updated|modified)\b', EntityType.COLUMN, 0.8),
            (r'\b(status|state|condition)\b', EntityType.COLUMN, 0.8),
            (r'\b(price|cost|amount|total|sum)\b', EntityType.COLUMN, 0.8),
            (r'\b(count|quantity|qty|amount)\b', EntityType.COLUMN, 0.8),
            (r'\b(description|desc|content|text)\b', EntityType.COLUMN, 0.8),
            
            # 복합 컬럼명
            (r'\b(created_at|updated_at|deleted_at)\b', EntityType.COLUMN, 0.9),
            (r'\b(user_id|member_id|customer_id)\b', EntityType.COLUMN, 0.9),
            (r'\b(order_id|payment_id|transaction_id)\b', EntityType.COLUMN, 0.9),
            (r'\b(product_id|item_id|goods_id)\b', EntityType.COLUMN, 0.9),
            (r'\b(post_id|article_id|content_id)\b', EntityType.COLUMN, 0.9),
            (r'\b(community_id|forum_id|board_id)\b', EntityType.COLUMN, 0.9),
            (r'\b(event_id|campaign_id|promotion_id)\b', EntityType.COLUMN, 0.9),
            
            # 특수 컬럼명
            (r'\b(c_email|c_phone|c_name)\b', EntityType.COLUMN, 0.9),
            (r'\b(is_\w+|has_\w+|can_\w+)\b', EntityType.COLUMN, 0.8),
        ]
    
    def _build_value_patterns(self) -> List[Tuple[str, EntityType, float]]:
        """값 추출 패턴 구축"""
        return [
            # 날짜/시간 값
            (r'\b(\d{4}년|\d{1,2}월|\d{1,2}일)\b', EntityType.VALUE, 0.9),
            (r'\b(\d{4}-\d{2}-\d{2})\b', EntityType.VALUE, 0.9),
            (r'\b(오늘|어제|내일|이번주|지난주|이번달|지난달)\b', EntityType.VALUE, 0.8),
            (r'\b(최근|최신|최근에|요즘)\b', EntityType.VALUE, 0.7),
            
            # 상태 값
            (r'\b(활성|active|활성화)\b', EntityType.VALUE, 0.9),
            (r'\b(비활성|inactive|비활성화)\b', EntityType.VALUE, 0.9),
            (r'\b(완료|completed|완료됨)\b', EntityType.VALUE, 0.9),
            (r'\b(진행중|processing|진행중인)\b', EntityType.VALUE, 0.9),
            (r'\b(대기|pending|대기중)\b', EntityType.VALUE, 0.9),
            (r'\b(취소|cancelled|취소됨)\b', EntityType.VALUE, 0.9),
            
            # 숫자 값
            (r'\b(\d+)\b', EntityType.VALUE, 0.6),
            (r'\b(\d+\.\d+)\b', EntityType.VALUE, 0.7),
            (r'\b(0|1|true|false|yes|no)\b', EntityType.VALUE, 0.8),
        ]
    
    def _build_operation_patterns(self) -> List[Tuple[str, EntityType, float]]:
        """연산 추출 패턴 구축"""
        return [
            # 비교 연산
            (r'\b(같은|동일한|일치하는)\b', EntityType.OPERATION, 0.8),
            (r'\b(다른|다른|차이가\s+있는)\b', EntityType.OPERATION, 0.8),
            (r'\b(보다\s+큰|초과하는|넘는)\b', EntityType.OPERATION, 0.8),
            (r'\b(보다\s+작은|미만인|안\s+넘는)\b', EntityType.OPERATION, 0.8),
            (r'\b(이상인|이상|최소)\b', EntityType.OPERATION, 0.8),
            (r'\b(이하인|이하|최대)\b', EntityType.OPERATION, 0.8),
            (r'\b(포함하는|포함된|들어있는)\b', EntityType.OPERATION, 0.8),
            (r'\b(시작하는|로\s+시작하는)\b', EntityType.OPERATION, 0.8),
            (r'\b(끝나는|로\s+끝나는)\b', EntityType.OPERATION, 0.8),
            
            # 논리 연산
            (r'\b(그리고|and|그리고도)\b', EntityType.OPERATION, 0.9),
            (r'\b(또는|or|혹은)\b', EntityType.OPERATION, 0.9),
            (r'\b(아닌|not|아니다)\b', EntityType.OPERATION, 0.9),
        ]
    
    def _build_aggregation_patterns(self) -> List[Tuple[str, EntityType, float]]:
        """집계 함수 추출 패턴 구축"""
        return [
            # 집계 함수
            (r'\b(개수|수|count|총\s*개수)\b', EntityType.AGGREGATION, 0.9),
            (r'\b(합계|합|sum|총합)\b', EntityType.AGGREGATION, 0.9),
            (r'\b(평균|avg|average|평균값)\b', EntityType.AGGREGATION, 0.9),
            (r'\b(최대|max|maximum|최댓값)\b', EntityType.AGGREGATION, 0.9),
            (r'\b(최소|min|minimum|최솟값)\b', EntityType.AGGREGATION, 0.9),
            (r'\b(총|전체|all|모든)\b', EntityType.AGGREGATION, 0.7),
        ]
    
    def extract_entities(self, query: str) -> List[ExtractedEntity]:
        """
        자연어 쿼리에서 엔티티 추출
        
        Args:
            query: 자연어 쿼리
            
        Returns:
            List[ExtractedEntity]: 추출된 엔티티 목록
        """
        try:
            entities = []
            query_lower = query.lower()
            
            # 각 패턴 타입별로 엔티티 추출
            entities.extend(self._extract_by_patterns(query, query_lower, self.table_patterns))
            entities.extend(self._extract_by_patterns(query, query_lower, self.column_patterns))
            entities.extend(self._extract_by_patterns(query, query_lower, self.value_patterns))
            entities.extend(self._extract_by_patterns(query, query_lower, self.operation_patterns))
            entities.extend(self._extract_by_patterns(query, query_lower, self.aggregation_patterns))
            
            # 중복 제거 및 정렬
            entities = self._deduplicate_entities(entities)
            entities = self._sort_entities_by_confidence(entities)
            
            logger.debug(f"Extracted {len(entities)} entities from query: {query}")
            return entities
            
        except Exception as e:
            logger.error(f"Failed to extract entities from query '{query}': {str(e)}")
            return []
    
    def _extract_by_patterns(self, query: str, query_lower: str, 
                           patterns: List[Tuple[str, EntityType, float]]) -> List[ExtractedEntity]:
        """패턴을 사용하여 엔티티 추출"""
        entities = []
        
        for pattern, entity_type, base_confidence in patterns:
            try:
                matches = re.finditer(pattern, query_lower, re.IGNORECASE)
                
                for match in matches:
                    text = match.group(1) if match.groups() else match.group(0)
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # 컨텍스트 추출
                    context = self._extract_context(query, start_pos, end_pos)
                    
                    # 신뢰도 조정
                    confidence = self._adjust_confidence(text, entity_type, base_confidence, context)
                    
                    entity = ExtractedEntity(
                        text=text,
                        type=entity_type,
                        confidence=confidence,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        context=context
                    )
                    
                    entities.append(entity)
                    
            except Exception as e:
                logger.warning(f"Error processing pattern '{pattern}': {str(e)}")
                continue
        
        return entities
    
    def _extract_context(self, query: str, start_pos: int, end_pos: int, 
                        context_window: int = 20) -> str:
        """엔티티 주변 컨텍스트 추출"""
        context_start = max(0, start_pos - context_window)
        context_end = min(len(query), end_pos + context_window)
        return query[context_start:context_end]
    
    def _adjust_confidence(self, text: str, entity_type: EntityType, 
                         base_confidence: float, context: str) -> float:
        """컨텍스트를 고려한 신뢰도 조정"""
        confidence = base_confidence
        
        # 컨텍스트 기반 조정
        if entity_type == EntityType.TABLE:
            # 테이블 관련 키워드가 있으면 신뢰도 증가
            if any(keyword in context.lower() for keyword in ['테이블', 'table', '데이터', 'data']):
                confidence += 0.1
        elif entity_type == EntityType.COLUMN:
            # 컬럼 관련 키워드가 있으면 신뢰도 증가
            if any(keyword in context.lower() for keyword in ['컬럼', 'column', '필드', 'field']):
                confidence += 0.1
        elif entity_type == EntityType.VALUE:
            # 값 관련 키워드가 있으면 신뢰도 증가
            if any(keyword in context.lower() for keyword in ['값', 'value', '데이터', 'data']):
                confidence += 0.1
        
        # 길이 기반 조정
        if len(text) > 10:  # 긴 텍스트는 더 신뢰할 만함
            confidence += 0.05
        
        # 최종 신뢰도 범위 제한
        return max(0.0, min(1.0, confidence))
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """중복 엔티티 제거"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            # 위치와 텍스트로 중복 판단
            key = (entity.start_pos, entity.end_pos, entity.text.lower())
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _sort_entities_by_confidence(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """신뢰도 순으로 엔티티 정렬"""
        return sorted(entities, key=lambda x: x.confidence, reverse=True)
    
    def get_entities_by_type(self, entities: List[ExtractedEntity], 
                           entity_type: EntityType) -> List[ExtractedEntity]:
        """특정 타입의 엔티티만 필터링"""
        return [entity for entity in entities if entity.type == entity_type]
    
    def get_high_confidence_entities(self, entities: List[ExtractedEntity], 
                                    threshold: float = 0.7) -> List[ExtractedEntity]:
        """높은 신뢰도의 엔티티만 필터링"""
        return [entity for entity in entities if entity.confidence >= threshold]
    
    def suggest_alternatives(self, entity: ExtractedEntity) -> List[str]:
        """엔티티에 대한 대안 제안"""
        alternatives = []
        
        if entity.type == EntityType.TABLE:
            # 테이블명 대안 제안
            if 'user' in entity.text.lower():
                alternatives.extend(['users', 't_user', 't_member'])
            elif 'member' in entity.text.lower():
                alternatives.extend(['members', 't_member', 't_user'])
            elif 'order' in entity.text.lower():
                alternatives.extend(['orders', 't_order', 't_payment'])
        
        elif entity.type == EntityType.COLUMN:
            # 컬럼명 대안 제안
            if 'email' in entity.text.lower():
                alternatives.extend(['email', 'c_email', 'user_email'])
            elif 'name' in entity.text.lower():
                alternatives.extend(['name', 'c_name', 'user_name'])
            elif 'id' in entity.text.lower():
                alternatives.extend(['id', 'no', 'num', 'number'])
        
        return list(set(alternatives))  # 중복 제거



