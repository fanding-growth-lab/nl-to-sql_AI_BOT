#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
자연어 쿼리에서 엔티티 관계 추출 시스템
사용자의 자연어 쿼리에서 테이블, 컬럼, 관계 등을 추출하는 고급 시스템
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
# import jieba  # 한국어 형태소 분석기 (선택사항)
from core.db import get_db_session, execute_query


class EntityType(Enum):
    """엔티티 유형"""
    TABLE = "table"
    COLUMN = "column"
    VALUE = "value"
    RELATIONSHIP = "relationship"
    AGGREGATION = "aggregation"
    CONDITION = "condition"
    TIME_PERIOD = "time_period"


class RelationshipType(Enum):
    """관계 유형"""
    DIRECT = "direct"  # 직접 관계
    INDIRECT = "indirect"  # 간접 관계
    TEMPORAL = "temporal"  # 시간적 관계
    HIERARCHICAL = "hierarchical"  # 계층적 관계
    ASSOCIATIVE = "associative"  # 연관 관계


@dataclass
class Entity:
    """엔티티 정보"""
    name: str
    entity_type: EntityType
    confidence: float
    original_text: str
    position: int
    synonyms: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    """관계 정보"""
    source_entity: Entity
    target_entity: Entity
    relationship_type: RelationshipType
    confidence: float
    description: str
    join_condition: Optional[str] = None


@dataclass
class QueryIntent:
    """쿼리 의도"""
    intent_type: str
    entities: List[Entity]
    relationships: List[Relationship]
    conditions: List[str]
    aggregations: List[str]
    time_filters: List[str]
    confidence: float


class EntityRelationshipExtractor:
    """엔티티 관계 추출기"""
    
    def __init__(self, database_schema: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.database_schema = database_schema or {}
        self.table_mappings = self._build_table_mappings()
        self.column_mappings = self._build_column_mappings()
        self.relationship_patterns = self._build_relationship_patterns()
        self.temporal_patterns = self._build_temporal_patterns()
        self.aggregation_patterns = self._build_aggregation_patterns()
        
    def _build_table_mappings(self) -> Dict[str, List[str]]:
        """테이블 매핑 구축"""
        return {
            "회원": ["t_member", "member", "사용자", "유저"],
            "크리에이터": ["t_creator", "creator", "제작자", "작가"],
            "포스트": ["t_post", "post", "게시물", "글", "아티클"],
            "커뮤니티": ["t_community", "community", "그룹", "모임"],
            "댓글": ["t_post_reply", "reply", "답글", "댓글"],
            "좋아요": ["t_post_like_log", "like", "좋아요", "추천"],
            "펀딩": ["t_funding", "funding", "펀딩", "후원"],
            "상품": ["t_product", "product", "상품", "아이템"],
            "주문": ["t_product_order", "order", "주문", "구매"],
            "결제": ["t_payment", "payment", "결제", "지불"],
            "쿠폰": ["t_coupon", "coupon", "쿠폰", "할인"],
            "이벤트": ["t_event", "event", "이벤트", "행사"],
            "설문": ["t_survey", "survey", "설문", "조사"],
            "알림": ["t_notification", "notification", "알림", "공지"],
            "메시지": ["t_message", "message", "메시지", "쪽지"]
        }
    
    def _build_column_mappings(self) -> Dict[str, List[str]]:
        """컬럼 매핑 구축"""
        return {
            "이름": ["name", "title", "이름", "제목"],
            "번호": ["no", "id", "번호", "아이디"],
            "생성일": ["created_at", "ins_datetime", "생성일", "등록일"],
            "수정일": ["updated_at", "upd_datetime", "수정일", "변경일"],
            "상태": ["status", "state", "상태", "상황"],
            "타입": ["type", "category", "타입", "종류"],
            "내용": ["content", "description", "내용", "설명"],
            "수량": ["count", "amount", "수량", "개수"],
            "가격": ["price", "cost", "가격", "비용"],
            "날짜": ["date", "datetime", "날짜", "시간"],
            "이메일": ["email", "mail", "이메일", "메일"],
            "전화번호": ["phone", "tel", "전화번호", "연락처"],
            "주소": ["address", "location", "주소", "위치"],
            "URL": ["url", "link", "링크", "주소"],
            "이미지": ["image", "img", "이미지", "사진"],
            "파일": ["file", "attachment", "파일", "첨부"]
        }
    
    def _build_relationship_patterns(self) -> List[Dict[str, Any]]:
        """관계 패턴 구축"""
        return [
            {
                "pattern": r"(\w+)의\s+(\w+)",
                "type": RelationshipType.DIRECT,
                "description": "소유 관계"
            },
            {
                "pattern": r"(\w+)와\s+(\w+)",
                "type": RelationshipType.ASSOCIATIVE,
                "description": "연관 관계"
            },
            {
                "pattern": r"(\w+)에서\s+(\w+)",
                "type": RelationshipType.HIERARCHICAL,
                "description": "계층 관계"
            },
            {
                "pattern": r"(\w+)에\s+(\w+)",
                "type": RelationshipType.INDIRECT,
                "description": "간접 관계"
            },
            {
                "pattern": r"(\w+)와\s+(\w+)의\s+(\w+)",
                "type": RelationshipType.ASSOCIATIVE,
                "description": "복합 관계"
            }
        ]
    
    def _build_temporal_patterns(self) -> List[Dict[str, Any]]:
        """시간 패턴 구축"""
        return [
            {
                "pattern": r"(\d{4})년",
                "type": "year",
                "description": "년도"
            },
            {
                "pattern": r"(\d{1,2})월",
                "type": "month",
                "description": "월"
            },
            {
                "pattern": r"(\d{1,2})일",
                "type": "day",
                "description": "일"
            },
            {
                "pattern": r"최근\s+(\d+)일",
                "type": "recent_days",
                "description": "최근 N일"
            },
            {
                "pattern": r"지난\s+(\d+)주",
                "type": "past_weeks",
                "description": "지난 N주"
            },
            {
                "pattern": r"이번\s+(\w+)",
                "type": "this_period",
                "description": "이번 기간"
            },
            {
                "pattern": r"작년",
                "type": "last_year",
                "description": "작년"
            },
            {
                "pattern": r"올해",
                "type": "this_year",
                "description": "올해"
            }
        ]
    
    def _build_aggregation_patterns(self) -> List[Dict[str, Any]]:
        """집계 패턴 구축"""
        return [
            {
                "pattern": r"총\s+(\w+)",
                "function": "COUNT",
                "description": "총 개수"
            },
            {
                "pattern": r"평균\s+(\w+)",
                "function": "AVG",
                "description": "평균"
            },
            {
                "pattern": r"최대\s+(\w+)",
                "function": "MAX",
                "description": "최대값"
            },
            {
                "pattern": r"최소\s+(\w+)",
                "function": "MIN",
                "description": "최소값"
            },
            {
                "pattern": r"합계\s+(\w+)",
                "function": "SUM",
                "description": "합계"
            },
            {
                "pattern": r"(\w+)\s+개수",
                "function": "COUNT",
                "description": "개수"
            },
            {
                "pattern": r"(\w+)\s+수",
                "function": "COUNT",
                "description": "수"
            }
        ]
    
    def extract_entities_and_relationships(self, query: str) -> QueryIntent:
        """엔티티와 관계 추출"""
        self.logger.info(f"엔티티 관계 추출 시작: {query}")
        
        try:
            # 1. 엔티티 추출
            entities = self._extract_entities(query)
            
            # 2. 관계 추출
            relationships = self._extract_relationships(query, entities)
            
            # 3. 조건 추출
            conditions = self._extract_conditions(query)
            
            # 4. 집계 추출
            aggregations = self._extract_aggregations(query)
            
            # 5. 시간 필터 추출
            time_filters = self._extract_time_filters(query)
            
            # 6. 의도 분류
            intent_type = self._classify_intent(query, entities, relationships)
            
            # 7. 신뢰도 계산
            confidence = self._calculate_confidence(entities, relationships, conditions)
            
            query_intent = QueryIntent(
                intent_type=intent_type,
                entities=entities,
                relationships=relationships,
                conditions=conditions,
                aggregations=aggregations,
                time_filters=time_filters,
                confidence=confidence
            )
            
            self.logger.info(f"엔티티 관계 추출 완료: {len(entities)}개 엔티티, {len(relationships)}개 관계")
            return query_intent
            
        except Exception as e:
            self.logger.error(f"엔티티 관계 추출 실패: {str(e)}")
            return QueryIntent(
                intent_type="unknown",
                entities=[],
                relationships=[],
                conditions=[],
                aggregations=[],
                time_filters=[],
                confidence=0.0
            )
    
    def _extract_entities(self, query: str) -> List[Entity]:
        """엔티티 추출"""
        entities = []
        
        # 테이블 엔티티 추출
        table_entities = self._extract_table_entities(query)
        entities.extend(table_entities)
        
        # 컬럼 엔티티 추출
        column_entities = self._extract_column_entities(query)
        entities.extend(column_entities)
        
        # 값 엔티티 추출
        value_entities = self._extract_value_entities(query)
        entities.extend(value_entities)
        
        return entities
    
    def _extract_table_entities(self, query: str) -> List[Entity]:
        """테이블 엔티티 추출"""
        entities = []
        
        for table_name, synonyms in self.table_mappings.items():
            for synonym in synonyms:
                if synonym in query:
                    # 매칭된 텍스트의 위치 찾기
                    position = query.find(synonym)
                    
                    entity = Entity(
                        name=table_name,
                        entity_type=EntityType.TABLE,
                        confidence=self._calculate_entity_confidence(synonym, query),
                        original_text=synonym,
                        position=position,
                        synonyms=synonyms,
                        attributes={"table_name": table_name}
                    )
                    entities.append(entity)
        
        return entities
    
    def _extract_column_entities(self, query: str) -> List[Entity]:
        """컬럼 엔티티 추출"""
        entities = []
        
        for column_name, synonyms in self.column_mappings.items():
            for synonym in synonyms:
                if synonym in query:
                    position = query.find(synonym)
                    
                    entity = Entity(
                        name=column_name,
                        entity_type=EntityType.COLUMN,
                        confidence=self._calculate_entity_confidence(synonym, query),
                        original_text=synonym,
                        position=position,
                        synonyms=synonyms,
                        attributes={"column_name": column_name}
                    )
                    entities.append(entity)
        
        return entities
    
    def _extract_value_entities(self, query: str) -> List[Entity]:
        """값 엔티티 추출"""
        entities = []
        
        # 숫자 값 추출
        number_pattern = r'\d+'
        for match in re.finditer(number_pattern, query):
            entity = Entity(
                name=match.group(),
                entity_type=EntityType.VALUE,
                confidence=0.8,
                original_text=match.group(),
                position=match.start(),
                attributes={"value_type": "number"}
            )
            entities.append(entity)
        
        # 문자열 값 추출 (따옴표로 둘러싸인)
        string_pattern = r'["\']([^"\']+)["\']'
        for match in re.finditer(string_pattern, query):
            entity = Entity(
                name=match.group(1),
                entity_type=EntityType.VALUE,
                confidence=0.9,
                original_text=match.group(),
                position=match.start(),
                attributes={"value_type": "string"}
            )
            entities.append(entity)
        
        return entities
    
    def _extract_relationships(self, query: str, entities: List[Entity]) -> List[Relationship]:
        """관계 추출"""
        relationships = []
        
        # 패턴 기반 관계 추출
        for pattern_info in self.relationship_patterns:
            pattern = pattern_info["pattern"]
            for match in re.finditer(pattern, query):
                source_text = match.group(1)
                target_text = match.group(2)
                
                # 매칭되는 엔티티 찾기
                source_entity = self._find_entity_by_text(source_text, entities)
                target_entity = self._find_entity_by_text(target_text, entities)
                
                if source_entity and target_entity:
                    relationship = Relationship(
                        source_entity=source_entity,
                        target_entity=target_entity,
                        relationship_type=pattern_info["type"],
                        confidence=0.7,
                        description=pattern_info["description"]
                    )
                    relationships.append(relationship)
        
        # 엔티티 간 거리 기반 관계 추출
        distance_relationships = self._extract_distance_based_relationships(entities, query)
        relationships.extend(distance_relationships)
        
        return relationships
    
    def _extract_distance_based_relationships(self, entities: List[Entity], query: str) -> List[Relationship]:
        """거리 기반 관계 추출"""
        relationships = []
        
        # 엔티티들을 위치 순으로 정렬
        sorted_entities = sorted(entities, key=lambda e: e.position)
        
        # 인접한 엔티티들 간의 관계 추출
        for i in range(len(sorted_entities) - 1):
            current_entity = sorted_entities[i]
            next_entity = sorted_entities[i + 1]
            
            # 거리 계산
            distance = next_entity.position - current_entity.position
            
            # 거리가 가까우면 관계로 간주
            if distance < 20:  # 20자 이내
                relationship = Relationship(
                    source_entity=current_entity,
                    target_entity=next_entity,
                    relationship_type=RelationshipType.ASSOCIATIVE,
                    confidence=0.6,
                    description="인접 관계"
                )
                relationships.append(relationship)
        
        return relationships
    
    def _extract_conditions(self, query: str) -> List[str]:
        """조건 추출"""
        conditions = []
        
        # 조건 패턴들
        condition_patterns = [
            r"(\w+)\s*=\s*(\w+)",
            r"(\w+)\s*>\s*(\w+)",
            r"(\w+)\s*<\s*(\w+)",
            r"(\w+)\s*like\s*(\w+)",
            r"(\w+)\s*in\s*\(([^)]+)\)",
            r"(\w+)\s*between\s*(\w+)\s*and\s*(\w+)"
        ]
        
        for pattern in condition_patterns:
            for match in re.finditer(pattern, query, re.IGNORECASE):
                condition = match.group(0)
                conditions.append(condition)
        
        return conditions
    
    def _extract_aggregations(self, query: str) -> List[str]:
        """집계 추출"""
        aggregations = []
        
        for pattern_info in self.aggregation_patterns:
            pattern = pattern_info["pattern"]
            for match in re.finditer(pattern, query):
                aggregation = {
                    "function": pattern_info["function"],
                    "column": match.group(1) if match.groups() else "all",
                    "description": pattern_info["description"]
                }
                aggregations.append(aggregation)
        
        return aggregations
    
    def _extract_time_filters(self, query: str) -> List[str]:
        """시간 필터 추출"""
        time_filters = []
        
        for pattern_info in self.temporal_patterns:
            pattern = pattern_info["pattern"]
            for match in re.finditer(pattern, query):
                time_filter = {
                    "type": pattern_info["type"],
                    "value": match.group(1) if match.groups() else match.group(0),
                    "description": pattern_info["description"]
                }
                time_filters.append(time_filter)
        
        return time_filters
    
    def _classify_intent(self, query: str, entities: List[Entity], relationships: List[Relationship]) -> str:
        """의도 분류"""
        # 키워드 기반 의도 분류
        intent_keywords = {
            "조회": ["조회", "찾기", "검색", "보기", "확인"],
            "통계": ["통계", "분석", "집계", "합계", "평균", "최대", "최소"],
            "비교": ["비교", "대비", "차이", "vs"],
            "트렌드": ["트렌드", "추이", "변화", "증가", "감소"],
            "랭킹": ["랭킹", "순위", "top", "최고", "최다"],
            "필터링": ["필터", "조건", "where", "and", "or"]
        }
        
        for intent, keywords in intent_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    return intent
        
        # 기본 의도
        return "조회"
    
    def _calculate_entity_confidence(self, text: str, query: str) -> float:
        """엔티티 신뢰도 계산"""
        # 기본 신뢰도
        confidence = 0.5
        
        # 텍스트 길이에 따른 보정
        if len(text) > 3:
            confidence += 0.2
        
        # 쿼리 내에서의 위치에 따른 보정
        if query.startswith(text):
            confidence += 0.1
        
        # 대소문자 일치에 따른 보정
        if text in query:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _find_entity_by_text(self, text: str, entities: List[Entity]) -> Optional[Entity]:
        """텍스트로 엔티티 찾기"""
        for entity in entities:
            if text in entity.original_text or text in entity.synonyms:
                return entity
        return None
    
    def _calculate_confidence(self, entities: List[Entity], relationships: List[Relationship], conditions: List[str]) -> float:
        """전체 신뢰도 계산"""
        if not entities:
            return 0.0
        
        # 엔티티 신뢰도의 평균
        entity_confidence = sum(e.confidence for e in entities) / len(entities)
        
        # 관계 신뢰도의 평균
        relationship_confidence = 0.0
        if relationships:
            relationship_confidence = sum(r.confidence for r in relationships) / len(relationships)
        
        # 조건이 있으면 보너스
        condition_bonus = 0.1 if conditions else 0.0
        
        # 가중 평균
        total_confidence = (entity_confidence * 0.6 + relationship_confidence * 0.3 + condition_bonus)
        
        return min(total_confidence, 1.0)
    
    def generate_sql_suggestions(self, query_intent: QueryIntent) -> List[Dict[str, Any]]:
        """SQL 제안 생성"""
        suggestions = []
        
        if not query_intent.entities:
            return suggestions
        
        # 기본 SELECT 쿼리 제안
        if query_intent.intent_type == "조회":
            suggestion = self._generate_select_suggestion(query_intent)
            if suggestion:
                suggestions.append(suggestion)
        
        # 집계 쿼리 제안
        if query_intent.aggregations:
            suggestion = self._generate_aggregation_suggestion(query_intent)
            if suggestion:
                suggestions.append(suggestion)
        
        # 통계 쿼리 제안
        if query_intent.intent_type == "통계":
            suggestion = self._generate_statistics_suggestion(query_intent)
            if suggestion:
                suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_select_suggestion(self, query_intent: QueryIntent) -> Dict[str, Any]:
        """SELECT 쿼리 제안 생성"""
        tables = [e.attributes.get("table_name", e.name) for e in query_intent.entities if e.entity_type == EntityType.TABLE]
        
        if not tables:
            return None
        
        # 기본 SELECT 쿼리
        sql = f"SELECT * FROM {tables[0]}"
        
        # 조인 추가
        if len(tables) > 1:
            for table in tables[1:]:
                sql += f" JOIN {table} ON ..."  # 실제로는 조인 조건을 계산해야 함
        
        # WHERE 조건 추가
        if query_intent.conditions:
            sql += f" WHERE {' AND '.join(query_intent.conditions)}"
        
        return {
            "type": "SELECT",
            "sql": sql,
            "description": "기본 조회 쿼리",
            "confidence": query_intent.confidence
        }
    
    def _generate_aggregation_suggestion(self, query_intent: QueryIntent) -> Dict[str, Any]:
        """집계 쿼리 제안 생성"""
        tables = [e.attributes.get("table_name", e.name) for e in query_intent.entities if e.entity_type == EntityType.TABLE]
        
        if not tables or not query_intent.aggregations:
            return None
        
        # 집계 함수들
        agg_functions = [agg["function"] for agg in query_intent.aggregations]
        agg_columns = [agg["column"] for agg in query_intent.aggregations]
        
        # SELECT 절 구성
        select_parts = []
        for func, col in zip(agg_functions, agg_columns):
            select_parts.append(f"{func}({col})")
        
        sql = f"SELECT {', '.join(select_parts)} FROM {tables[0]}"
        
        return {
            "type": "AGGREGATION",
            "sql": sql,
            "description": "집계 쿼리",
            "confidence": query_intent.confidence
        }
    
    def _generate_statistics_suggestion(self, query_intent: QueryIntent) -> Dict[str, Any]:
        """통계 쿼리 제안 생성"""
        tables = [e.attributes.get("table_name", e.name) for e in query_intent.entities if e.entity_type == EntityType.TABLE]
        
        if not tables:
            return None
        
        # 기본 통계 쿼리
        sql = f"""
        SELECT 
            COUNT(*) as total_count,
            AVG(created_at) as avg_created,
            MAX(created_at) as latest_created,
            MIN(created_at) as earliest_created
        FROM {tables[0]}
        """
        
        return {
            "type": "STATISTICS",
            "sql": sql,
            "description": "통계 분석 쿼리",
            "confidence": query_intent.confidence
        }
