"""
복합 엔티티 패턴 인식 및 추출 시스템

시간-지표, 제품-속성 등 여러 엔티티가 결합된 복합 엔티티를 인식하고 추출하는 시스템을 개발합니다.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

@dataclass
class CompositeEntity:
    """복합 엔티티"""
    value: str
    entity_type: str
    components: List[str]
    confidence: float
    start_pos: int
    end_pos: int
    pattern_matched: str
    extraction_func: str

@dataclass
class PatternMatch:
    """패턴 매칭 결과"""
    pattern: str
    entity_type: str
    matched_text: str
    groups: List[str]
    confidence: float
    start_pos: int
    end_pos: int

class CompositeEntityExtractor:
    """복합 엔티티 추출기"""
    
    def __init__(self):
        """복합 엔티티 추출기 초기화"""
        self.patterns = []
        self.extraction_functions = {}
        self.logger = logging.getLogger(__name__)
        
        # 패턴 및 추출 함수 초기화
        self._initialize_patterns()
        self._initialize_extraction_functions()
    
    def _initialize_patterns(self):
        """복합 엔티티 패턴 초기화"""
        self.patterns = [
            # 시간-지표 패턴
            {
                "pattern": r"(\d+년|\d+월|\d+일|지난주|이번달|올해|작년|이번주|지난주|내일|어제|오늘)[\s]*(매출|가입자|주문|방문자|클릭수|회원|사용자|고객|매상|수익|이익|비용|ROI|전환율|성장률)",
                "entity_type": "time_metric",
                "extraction_func": "extract_time_metric",
                "priority": 1
            },
            
            # 제품-속성 패턴
            {
                "pattern": r"([가-힣]+색상|[가-힣]+사이즈|[가-힣]+타입|[가-힣]+브랜드)[\s]*([가-힣\w]+)",
                "entity_type": "product_attribute",
                "extraction_func": "extract_product_attribute",
                "priority": 2
            },
            
            # 지역-지표 패턴
            {
                "pattern": r"(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)[\s]*(매출|가입자|주문|방문자|회원|사용자|고객)",
                "entity_type": "location_metric",
                "extraction_func": "extract_location_metric",
                "priority": 3
            },
            
            # 순위-지표 패턴
            {
                "pattern": r"(상위|top|탑|제일 많은|가장 많은|최고|최대)[\s]*(\d+)[\s]*(크리에이터|회원|사용자|고객|상품|제품|지역|도시)",
                "entity_type": "ranking_metric",
                "extraction_func": "extract_ranking_metric",
                "priority": 4
            },
            
            # 기간-지표 패턴
            {
                "pattern": r"(분기별|월별|일별|년별|주별)[\s]*(매출|가입자|주문|방문자|회원|사용자|고객|매상|수익|이익|비용)",
                "entity_type": "period_metric",
                "extraction_func": "extract_period_metric",
                "priority": 5
            },
            
            # 비교-지표 패턴
            {
                "pattern": r"(지난달|이번달|작년|올해|전년|이전)[\s]*(대비|비교|대조)[\s]*(매출|가입자|주문|방문자|회원|사용자|고객|매상|수익|이익|비용)",
                "entity_type": "comparison_metric",
                "extraction_func": "extract_comparison_metric",
                "priority": 6
            },
            
            # 카테고리-속성 패턴
            {
                "pattern": r"([가-힣]+카테고리|[가-힣]+분류|[가-힣]+종류)[\s]*([가-힣\w]+)",
                "entity_type": "category_attribute",
                "extraction_func": "extract_category_attribute",
                "priority": 7
            }
        ]
    
    def _initialize_extraction_functions(self):
        """추출 함수 초기화"""
        self.extraction_functions = {
            "extract_time_metric": self._extract_time_metric,
            "extract_product_attribute": self._extract_product_attribute,
            "extract_location_metric": self._extract_location_metric,
            "extract_ranking_metric": self._extract_ranking_metric,
            "extract_period_metric": self._extract_period_metric,
            "extract_comparison_metric": self._extract_comparison_metric,
            "extract_category_attribute": self._extract_category_attribute
        }
    
    def extract_composite_entities(self, text: str) -> List[CompositeEntity]:
        """
        텍스트에서 복합 엔티티 추출
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            추출된 복합 엔티티 리스트
        """
        extracted_entities = []
        
        # 패턴별로 매칭 시도
        for pattern_info in self.patterns:
            pattern = pattern_info["pattern"]
            entity_type = pattern_info["entity_type"]
            extraction_func = pattern_info["extraction_func"]
            priority = pattern_info["priority"]
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                try:
                    # 추출 함수 호출
                    if extraction_func in self.extraction_functions:
                        entity = self.extraction_functions[extraction_func](match, text)
                        if entity:
                            entity.pattern_matched = pattern
                            entity.extraction_func = extraction_func
                            extracted_entities.append(entity)
                except Exception as e:
                    self.logger.error(f"복합 엔티티 추출 오류: {e}")
                    continue
        
        # 중복 제거 및 정렬
        extracted_entities = self._remove_duplicates(extracted_entities)
        extracted_entities.sort(key=lambda x: (x.start_pos, -x.confidence))
        
        return extracted_entities
    
    def _extract_time_metric(self, match, text: str) -> Optional[CompositeEntity]:
        """시간-지표 패턴 추출"""
        groups = match.groups()
        if len(groups) >= 2:
            time_part = groups[0].strip()
            metric_part = groups[1].strip()
            
            return CompositeEntity(
                value=f"{time_part} {metric_part}",
                entity_type="time_metric",
                components=[time_part, metric_part],
                confidence=0.9,
                start_pos=match.start(),
                end_pos=match.end(),
                pattern_matched="",
                extraction_func=""
            )
        return None
    
    def _extract_product_attribute(self, match, text: str) -> Optional[CompositeEntity]:
        """제품-속성 패턴 추출"""
        groups = match.groups()
        if len(groups) >= 2:
            attribute_part = groups[0].strip()
            product_part = groups[1].strip()
            
            return CompositeEntity(
                value=f"{attribute_part} {product_part}",
                entity_type="product_attribute",
                components=[attribute_part, product_part],
                confidence=0.8,
                start_pos=match.start(),
                end_pos=match.end(),
                pattern_matched="",
                extraction_func=""
            )
        return None
    
    def _extract_location_metric(self, match, text: str) -> Optional[CompositeEntity]:
        """지역-지표 패턴 추출"""
        groups = match.groups()
        if len(groups) >= 2:
            location_part = groups[0].strip()
            metric_part = groups[1].strip()
            
            return CompositeEntity(
                value=f"{location_part} {metric_part}",
                entity_type="location_metric",
                components=[location_part, metric_part],
                confidence=0.85,
                start_pos=match.start(),
                end_pos=match.end(),
                pattern_matched="",
                extraction_func=""
            )
        return None
    
    def _extract_ranking_metric(self, match, text: str) -> Optional[CompositeEntity]:
        """순위-지표 패턴 추출"""
        groups = match.groups()
        if len(groups) >= 3:
            ranking_part = groups[0].strip()
            number_part = groups[1].strip()
            metric_part = groups[2].strip()
            
            return CompositeEntity(
                value=f"{ranking_part} {number_part} {metric_part}",
                entity_type="ranking_metric",
                components=[ranking_part, number_part, metric_part],
                confidence=0.95,
                start_pos=match.start(),
                end_pos=match.end(),
                pattern_matched="",
                extraction_func=""
            )
        return None
    
    def _extract_period_metric(self, match, text: str) -> Optional[CompositeEntity]:
        """기간-지표 패턴 추출"""
        groups = match.groups()
        if len(groups) >= 2:
            period_part = groups[0].strip()
            metric_part = groups[1].strip()
            
            return CompositeEntity(
                value=f"{period_part} {metric_part}",
                entity_type="period_metric",
                components=[period_part, metric_part],
                confidence=0.9,
                start_pos=match.start(),
                end_pos=match.end(),
                pattern_matched="",
                extraction_func=""
            )
        return None
    
    def _extract_comparison_metric(self, match, text: str) -> Optional[CompositeEntity]:
        """비교-지표 패턴 추출"""
        groups = match.groups()
        if len(groups) >= 3:
            time_part = groups[0].strip()
            comparison_part = groups[1].strip()
            metric_part = groups[2].strip()
            
            return CompositeEntity(
                value=f"{time_part} {comparison_part} {metric_part}",
                entity_type="comparison_metric",
                components=[time_part, comparison_part, metric_part],
                confidence=0.85,
                start_pos=match.start(),
                end_pos=match.end(),
                pattern_matched="",
                extraction_func=""
            )
        return None
    
    def _extract_category_attribute(self, match, text: str) -> Optional[CompositeEntity]:
        """카테고리-속성 패턴 추출"""
        groups = match.groups()
        if len(groups) >= 2:
            category_part = groups[0].strip()
            attribute_part = groups[1].strip()
            
            return CompositeEntity(
                value=f"{category_part} {attribute_part}",
                entity_type="category_attribute",
                components=[category_part, attribute_part],
                confidence=0.8,
                start_pos=match.start(),
                end_pos=match.end(),
                pattern_matched="",
                extraction_func=""
            )
        return None
    
    def _remove_duplicates(self, entities: List[CompositeEntity]) -> List[CompositeEntity]:
        """중복 엔티티 제거"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            # 위치와 내용을 기준으로 중복 판단
            key = (entity.start_pos, entity.end_pos, entity.value)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def get_entity_components(self, entity: CompositeEntity) -> Dict[str, Any]:
        """복합 엔티티의 구성 요소 분석"""
        components = {
            "original": entity.value,
            "type": entity.entity_type,
            "components": entity.components,
            "confidence": entity.confidence,
            "position": {
                "start": entity.start_pos,
                "end": entity.end_pos
            }
        }
        
        # 엔티티 타입별 특수 처리
        if entity.entity_type == "time_metric":
            components["time_component"] = entity.components[0]
            components["metric_component"] = entity.components[1]
        elif entity.entity_type == "product_attribute":
            components["attribute_component"] = entity.components[0]
            components["product_component"] = entity.components[1]
        elif entity.entity_type == "ranking_metric":
            components["ranking_component"] = entity.components[0]
            components["number_component"] = entity.components[1]
            components["metric_component"] = entity.components[2]
        
        return components
    
    def analyze_entity_relationships(self, entities: List[CompositeEntity]) -> Dict[str, Any]:
        """엔티티 간 관계 분석"""
        relationships = {
            "total_entities": len(entities),
            "entity_types": {},
            "component_overlaps": [],
            "temporal_relationships": [],
            "spatial_relationships": []
        }
        
        # 엔티티 타입별 분류
        for entity in entities:
            entity_type = entity.entity_type
            if entity_type not in relationships["entity_types"]:
                relationships["entity_types"][entity_type] = 0
            relationships["entity_types"][entity_type] += 1
        
        # 구성 요소 중복 분석
        all_components = []
        for entity in entities:
            all_components.extend(entity.components)
        
        component_counts = {}
        for component in all_components:
            component_counts[component] = component_counts.get(component, 0) + 1
        
        relationships["component_overlaps"] = [
            {"component": comp, "count": count}
            for comp, count in component_counts.items()
            if count > 1
        ]
        
        return relationships
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """추출 통계 정보"""
        return {
            "total_patterns": len(self.patterns),
            "extraction_functions": list(self.extraction_functions.keys()),
            "pattern_types": list(set(p["entity_type"] for p in self.patterns))
        }

def test_composite_entity_extractor():
    """복합 엔티티 추출기 테스트"""
    print("=== 복합 엔티티 추출기 테스트 ===")
    
    # 추출기 초기화
    extractor = CompositeEntityExtractor()
    
    # 테스트 쿼리들
    test_queries = [
        "8월 매출 보고서 작성해줘",
        "빨간색 티셔츠 재고 현황은?",
        "서울 지역 신규 가입자 수",
        "상위 5 크리에이터들을 뽑아줘",
        "분기별 매출 분석",
        "지난달 대비 이번달 매출 증가율",
        "전자제품 카테고리별 판매량"
    ]
    
    for query in test_queries:
        print(f"\n쿼리: {query}")
        entities = extractor.extract_composite_entities(query)
        
        if entities:
            print(f"  추출된 복합 엔티티 수: {len(entities)}")
            for i, entity in enumerate(entities, 1):
                print(f"    {i}. {entity.value} (타입: {entity.entity_type})")
                print(f"       구성요소: {entity.components}")
                print(f"       신뢰도: {entity.confidence}")
                print(f"       위치: {entity.start_pos}-{entity.end_pos}")
        else:
            print("  추출된 복합 엔티티 없음")
    
    # 통계 정보
    print(f"\n=== 추출 통계 ===")
    stats = extractor.get_extraction_statistics()
    print(f"총 패턴 수: {stats['total_patterns']}")
    print(f"추출 함수 수: {len(stats['extraction_functions'])}")
    print(f"패턴 타입: {stats['pattern_types']}")

if __name__ == "__main__":
    test_composite_entity_extractor()

