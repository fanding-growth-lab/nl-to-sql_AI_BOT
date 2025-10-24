"""
향상된 엔티티 추출 파이프라인

키워드 확장, 복합 엔티티 인식, 중요도 평가를 통합한 엔티티 추출 시스템을 구현합니다.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json

# 로컬 모듈 임포트
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from keyword_expander import KeywordExpander, ExpansionResult
from composite_entity_extractor import CompositeEntityExtractor, CompositeEntity
from entity_importance_evaluator import EntityImportanceEvaluator, EntityImportance

logger = logging.getLogger(__name__)

@dataclass
class EnhancedEntity:
    """향상된 엔티티"""
    value: str
    entity_type: str
    confidence: float
    importance_score: float
    source: str  # 'basic', 'expanded', 'composite'
    components: List[str]
    synonyms: List[str]
    position: Tuple[int, int]
    metadata: Dict[str, Any]

@dataclass
class ExtractionResult:
    """추출 결과"""
    entities: List[EnhancedEntity]
    total_entities: int
    extraction_time: float
    confidence_avg: float
    importance_avg: float
    source_distribution: Dict[str, int]
    type_distribution: Dict[str, int]

class EnhancedEntityExtractor:
    """향상된 엔티티 추출기"""
    
    def __init__(self, synonym_dict_path: Optional[str] = None):
        """
        향상된 엔티티 추출기 초기화
        
        Args:
            synonym_dict_path: 동의어 사전 파일 경로
        """
        self.logger = logging.getLogger(__name__)
        
        # 하위 시스템 초기화
        self.keyword_expander = KeywordExpander(synonym_dict_path)
        self.composite_extractor = CompositeEntityExtractor()
        self.importance_evaluator = EntityImportanceEvaluator()
        
        # 기본 엔티티 추출 패턴
        self.basic_patterns = {
            'user': r'\b(회원|유저|사용자|고객|멤버|가입자|회원가입자|로그인|가입|탈퇴|활성|비활성)\b',
            'product': r'\b(상품|제품|아이템|물품|굿즈|카테고리|분류|종류|유형|재고|가격|할인)\b',
            'time': r'\b(이번달|지난달|올해|작년|이번주|지난주|내일|어제|오늘|분기별|월별|일별|년별|주별|매월|매일|매년)\b',
            'business': r'\b(매출|이익|비용|ROI|전환율|성장률|수익|판매액|매상|판매 실적|순이익|영업이익|투자수익률|수익률|증가율|성장률|증가비율|클라이언트|구매자|소비자|주문|방문자|클릭수)\b',
            'location': r'\b(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)\b',
            'action': r'\b(분석|검토|조사|검증|작성|생성|추출|계산|비교|대조|평가|측정|모니터링|추적|작성|생성|추출|계산|비교|대조|평가|측정|모니터링|추적)\b'
        }
    
    def extract_entities(self, query: str, user_history: Optional[Dict] = None) -> ExtractionResult:
        """
        향상된 엔티티 추출 파이프라인 실행
        
        Args:
            query: 분석할 쿼리
            user_history: 사용자 이력 (선택사항)
            
        Returns:
            ExtractionResult: 추출 결과
        """
        start_time = datetime.now()
        
        # 1. 기본 엔티티 추출
        basic_entities = self._extract_basic_entities(query)
        self.logger.info(f"기본 엔티티 추출 완료: {len(basic_entities)}개")
        
        # 2. 키워드 확장
        expanded_entities = self._expand_entities(basic_entities, query)
        self.logger.info(f"키워드 확장 완료: {len(expanded_entities)}개")
        
        # 3. 복합 엔티티 추출
        composite_entities = self.composite_extractor.extract_composite_entities(query)
        self.logger.info(f"복합 엔티티 추출 완료: {len(composite_entities)}개")
        
        # 4. 모든 엔티티 통합
        all_entities = self._merge_entities(basic_entities, expanded_entities, composite_entities)
        self.logger.info(f"엔티티 통합 완료: {len(all_entities)}개")
        
        # 5. 중요도 평가
        importance_results = self.importance_evaluator.evaluate_entities(
            [entity['value'] for entity in all_entities], 
            query, 
            user_history
        )
        
        # 6. 향상된 엔티티 객체 생성
        enhanced_entities = self._create_enhanced_entities(
            all_entities, 
            importance_results, 
            query
        )
        
        # 7. 결과 통계 계산
        extraction_time = (datetime.now() - start_time).total_seconds()
        result = self._calculate_statistics(enhanced_entities, extraction_time)
        
        return result
    
    def _extract_basic_entities(self, query: str) -> List[Dict[str, Any]]:
        """기본 엔티티 추출"""
        entities = []
        
        for entity_type, pattern in self.basic_patterns.items():
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                entity = {
                    'value': match.group(),
                    'type': entity_type,
                    'confidence': 0.8,
                    'source': 'basic',
                    'position': (match.start(), match.end()),
                    'metadata': {}
                }
                entities.append(entity)
        
        return entities
    
    def _expand_entities(self, entities: List[Dict], query: str) -> List[Dict[str, Any]]:
        """엔티티 키워드 확장"""
        expanded_entities = []
        
        for entity in entities:
            # 원본 엔티티 추가
            expanded_entities.append(entity)
            
            # 키워드 확장
            expansion_result = self.keyword_expander.expand_keyword(entity['value'])
            
            # 확장된 용어들을 새로운 엔티티로 추가
            for expanded_term in expansion_result.expanded_terms:
                if expanded_term != entity['value']:  # 원본과 다른 경우만
                    expanded_entity = entity.copy()
                    expanded_entity['value'] = expanded_term
                    expanded_entity['source'] = 'expanded'
                    expanded_entity['original_term'] = entity['value']
                    expanded_entity['expansion_confidence'] = expansion_result.confidence
                    expanded_entity['metadata']['expansion_type'] = expansion_result.expansion_type
                    expanded_entities.append(expanded_entity)
        
        return expanded_entities
    
    def _merge_entities(self, basic_entities: List[Dict], expanded_entities: List[Dict], 
                       composite_entities: List[CompositeEntity]) -> List[Dict[str, Any]]:
        """엔티티 통합 및 중복 제거"""
        all_entities = []
        
        # 기본 및 확장 엔티티 추가
        for entity in basic_entities + expanded_entities:
            all_entities.append(entity)
        
        # 복합 엔티티 추가
        for composite_entity in composite_entities:
            entity = {
                'value': composite_entity.value,
                'type': composite_entity.entity_type,
                'confidence': composite_entity.confidence,
                'source': 'composite',
                'position': (composite_entity.start_pos, composite_entity.end_pos),
                'components': composite_entity.components,
                'metadata': {
                    'pattern_matched': composite_entity.pattern_matched,
                    'extraction_func': composite_entity.extraction_func
                }
            }
            all_entities.append(entity)
        
        # 중복 제거
        unique_entities = self._remove_duplicates(all_entities)
        
        return unique_entities
    
    def _remove_duplicates(self, entities: List[Dict]) -> List[Dict[str, Any]]:
        """중복 엔티티 제거"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            # 위치와 내용을 기준으로 중복 판단
            key = (entity['position'][0], entity['position'][1], entity['value'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _create_enhanced_entities(self, entities: List[Dict], importance_results: List[EntityImportance], 
                                 query: str) -> List[EnhancedEntity]:
        """향상된 엔티티 객체 생성"""
        enhanced_entities = []
        
        # 중요도 결과를 딕셔너리로 변환
        importance_dict = {result.entity: result for result in importance_results}
        
        for entity in entities:
            importance = importance_dict.get(entity['value'])
            
            enhanced_entity = EnhancedEntity(
                value=entity['value'],
                entity_type=entity['type'],
                confidence=entity['confidence'],
                importance_score=importance.final_score if importance else 0.0,
                source=entity['source'],
                components=entity.get('components', []),
                synonyms=self._get_synonyms(entity['value']),
                position=entity['position'],
                metadata=entity.get('metadata', {})
            )
            
            enhanced_entities.append(enhanced_entity)
        
        # 중요도 순으로 정렬
        enhanced_entities.sort(key=lambda x: x.importance_score, reverse=True)
        
        return enhanced_entities
    
    def _get_synonyms(self, term: str) -> List[str]:
        """용어의 동의어 조회"""
        expansion_result = self.keyword_expander.expand_keyword(term)
        return [t for t in expansion_result.expanded_terms if t != term]
    
    def _calculate_statistics(self, entities: List[EnhancedEntity], extraction_time: float) -> ExtractionResult:
        """통계 정보 계산"""
        total_entities = len(entities)
        
        if total_entities == 0:
            return ExtractionResult(
                entities=entities,
                total_entities=0,
                extraction_time=extraction_time,
                confidence_avg=0.0,
                importance_avg=0.0,
                source_distribution={},
                type_distribution={}
            )
        
        # 평균 신뢰도
        confidence_avg = sum(entity.confidence for entity in entities) / total_entities
        
        # 평균 중요도
        importance_avg = sum(entity.importance_score for entity in entities) / total_entities
        
        # 소스별 분포
        source_distribution = {}
        for entity in entities:
            source = entity.source
            source_distribution[source] = source_distribution.get(source, 0) + 1
        
        # 타입별 분포
        type_distribution = {}
        for entity in entities:
            entity_type = entity.entity_type
            type_distribution[entity_type] = type_distribution.get(entity_type, 0) + 1
        
        return ExtractionResult(
            entities=entities,
            total_entities=total_entities,
            extraction_time=extraction_time,
            confidence_avg=confidence_avg,
            importance_avg=importance_avg,
            source_distribution=source_distribution,
            type_distribution=type_distribution
        )
    
    def get_top_entities(self, query: str, top_k: int = 5, 
                        user_history: Optional[Dict] = None) -> List[EnhancedEntity]:
        """상위 K개 엔티티 반환"""
        result = self.extract_entities(query, user_history)
        return result.entities[:top_k]
    
    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """쿼리 복잡도 분석"""
        result = self.extract_entities(query)
        
        complexity_score = 0.0
        
        # 엔티티 수 기반 복잡도
        entity_count_score = min(result.total_entities / 10.0, 1.0)
        complexity_score += entity_count_score * 0.3
        
        # 엔티티 타입 다양성 기반 복잡도
        type_diversity_score = len(result.type_distribution) / 6.0  # 최대 6개 타입
        complexity_score += type_diversity_score * 0.2
        
        # 복합 엔티티 비율 기반 복잡도
        composite_ratio = result.source_distribution.get('composite', 0) / max(result.total_entities, 1)
        complexity_score += composite_ratio * 0.3
        
        # 평균 중요도 기반 복잡도
        importance_score = result.importance_avg
        complexity_score += importance_score * 0.2
        
        return {
            'complexity_score': complexity_score,
            'entity_count': result.total_entities,
            'type_diversity': len(result.type_distribution),
            'composite_ratio': composite_ratio,
            'avg_importance': result.importance_avg,
            'extraction_time': result.extraction_time
        }
    
    def get_entity_relationships(self, query: str) -> Dict[str, Any]:
        """엔티티 간 관계 분석"""
        result = self.extract_entities(query)
        
        relationships = {
            'total_entities': result.total_entities,
            'entity_types': result.type_distribution,
            'source_distribution': result.source_distribution,
            'top_entities': result.entities[:3],
            'complexity_analysis': self.analyze_query_complexity(query)
        }
        
        return relationships
    
    def update_user_history(self, user_id: str, query: str, entities: List[str]):
        """사용자 이력 업데이트"""
        self.importance_evaluator.update_user_history(user_id, entities, query)
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """사용자 선호도 조회"""
        return self.importance_evaluator.get_user_preferences(user_id)

def test_enhanced_entity_extractor():
    """향상된 엔티티 추출기 테스트"""
    print("=== 향상된 엔티티 추출기 테스트 ===")
    
    # 추출기 초기화
    extractor = EnhancedEntityExtractor()
    
    # 테스트 쿼리들
    test_queries = [
        "8월 신규 가입자 수 알려줘",
        "지난달 대비 이번달 매출 증가율은?",
        "서울 지역 빨간색 티셔츠 판매량 추이",
        "2023년 분기별 제품 카테고리별 매출 분석",
        "상위 5 크리에이터들을 뽑아줘"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n테스트 쿼리 {i}: {query}")
        
        # 엔티티 추출
        result = extractor.extract_entities(query)
        
        print(f"  총 엔티티 수: {result.total_entities}")
        print(f"  추출 시간: {result.extraction_time:.3f}초")
        print(f"  평균 신뢰도: {result.confidence_avg:.3f}")
        print(f"  평균 중요도: {result.importance_avg:.3f}")
        
        print(f"  소스별 분포: {result.source_distribution}")
        print(f"  타입별 분포: {result.type_distribution}")
        
        print("  상위 엔티티:")
        for j, entity in enumerate(result.entities[:5], 1):
            print(f"    {j}. {entity.value} (타입: {entity.entity_type}, 중요도: {entity.importance_score:.3f})")
            if entity.components:
                print(f"       구성요소: {entity.components}")
            if entity.synonyms:
                print(f"       동의어: {entity.synonyms[:3]}...")
        
        # 복잡도 분석
        complexity = extractor.analyze_query_complexity(query)
        print(f"  복잡도 점수: {complexity['complexity_score']:.3f}")
    
    # 사용자 이력 테스트
    print(f"\n=== 사용자 이력 테스트 ===")
    user_id = "test_user"
    
    # 사용자 이력 업데이트
    extractor.update_user_history(user_id, "매출 분석 리포트", ["매출", "분석", "리포트"])
    extractor.update_user_history(user_id, "사용자 지역 분석", ["사용자", "지역", "분석"])
    extractor.update_user_history(user_id, "매출 이익 비용 분석", ["매출", "이익", "비용"])
    
    # 사용자 선호도 조회
    preferences = extractor.get_user_preferences(user_id)
    print(f"사용자 선호도: {preferences}")
    
    # 관계 분석
    print(f"\n=== 엔티티 관계 분석 ===")
    relationships = extractor.get_entity_relationships("8월 매출 분석 리포트 작성해줘")
    print(f"관계 분석 결과: {relationships}")

if __name__ == "__main__":
    test_enhanced_entity_extractor()
