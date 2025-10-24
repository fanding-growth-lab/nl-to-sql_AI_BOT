"""
컨텍스트 기반 엔티티 중요도 평가 알고리즘

쿼리 컨텍스트에 따라 추출된 엔티티의 중요도를 평가하고 우선순위를 부여하는 알고리즘을 구현합니다.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

@dataclass
class EntityImportance:
    """엔티티 중요도 정보"""
    entity: str
    importance_score: float
    position_score: float
    frequency_score: float
    domain_score: float
    context_score: float
    user_history_score: float
    final_score: float
    ranking: int

@dataclass
class ContextAnalysis:
    """컨텍스트 분석 결과"""
    query_length: int
    business_terms_count: int
    temporal_indicators: List[str]
    spatial_indicators: List[str]
    action_verbs: List[str]
    question_words: List[str]

class EntityImportanceEvaluator:
    """엔티티 중요도 평가기"""
    
    def __init__(self):
        """중요도 평가기 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 비즈니스 핵심 용어 정의
        self.business_terms = {
            '매출', '이익', '비용', 'ROI', '전환율', '성장률', '수익', '판매액', 
            '매상', '판매 실적', '순이익', '영업이익', '투자수익률', '수익률',
            '증가율', '성장률', '증가비율', '클라이언트', '구매자', '소비자'
        }
        
        # 시간 지시어
        self.temporal_indicators = {
            '이번달', '지난달', '올해', '작년', '이번주', '지난주', '내일', '어제', '오늘',
            '분기별', '월별', '일별', '년별', '주별', '매월', '매일', '매년'
        }
        
        # 공간 지시어
        self.spatial_indicators = {
            '서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종',
            '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주'
        }
        
        # 행동 동사
        self.action_verbs = {
            '분석', '검토', '조사', '검증', '작성', '생성', '추출', '계산',
            '비교', '대조', '평가', '측정', '모니터링', '추적'
        }
        
        # 질문어
        self.question_words = {
            '어떤', '무엇', '언제', '어디', '누구', '왜', '어떻게', '얼마나',
            '몇', '어느', '어떤', '무슨'
        }
        
        # 도메인별 가중치
        self.domain_weights = {
            'business': 1.5,
            'time': 1.2,
            'location': 1.1,
            'user': 1.0,
            'product': 0.9,
            'general': 0.7
        }
        
        # 사용자 이력 (실제 구현에서는 데이터베이스에서 로드)
        self.user_history = {}
    
    def calculate_entity_importance(self, entity: str, query_context: str, 
                                 user_history: Optional[Dict] = None) -> EntityImportance:
        """
        엔티티 중요도 계산
        
        Args:
            entity: 평가할 엔티티
            query_context: 쿼리 컨텍스트
            user_history: 사용자 이력 (선택사항)
            
        Returns:
            EntityImportance: 중요도 정보
        """
        # 컨텍스트 분석
        context_analysis = self._analyze_context(query_context)
        
        # 위치 기반 중요도
        position_score = self._calculate_position_score(entity, query_context)
        
        # 빈도 기반 중요도
        frequency_score = self._calculate_frequency_score(entity, query_context)
        
        # 도메인 기반 중요도
        domain_score = self._calculate_domain_score(entity)
        
        # 컨텍스트 기반 중요도
        context_score = self._calculate_context_score(entity, context_analysis)
        
        # 사용자 이력 기반 중요도
        user_history_score = self._calculate_user_history_score(entity, user_history)
        
        # 최종 점수 계산 (가중 평균)
        final_score = (
            position_score * 0.25 +
            frequency_score * 0.20 +
            domain_score * 0.25 +
            context_score * 0.20 +
            user_history_score * 0.10
        )
        
        return EntityImportance(
            entity=entity,
            importance_score=final_score,
            position_score=position_score,
            frequency_score=frequency_score,
            domain_score=domain_score,
            context_score=context_score,
            user_history_score=user_history_score,
            final_score=final_score,
            ranking=0  # 나중에 설정
        )
    
    def _analyze_context(self, query: str) -> ContextAnalysis:
        """쿼리 컨텍스트 분석"""
        query_lower = query.lower()
        
        # 비즈니스 용어 개수
        business_terms_count = sum(1 for term in self.business_terms if term in query)
        
        # 시간 지시어 추출
        temporal_indicators = [term for term in self.temporal_indicators if term in query]
        
        # 공간 지시어 추출
        spatial_indicators = [term for term in self.spatial_indicators if term in query]
        
        # 행동 동사 추출
        action_verbs = [term for term in self.action_verbs if term in query]
        
        # 질문어 추출
        question_words = [term for term in self.question_words if term in query]
        
        return ContextAnalysis(
            query_length=len(query),
            business_terms_count=business_terms_count,
            temporal_indicators=temporal_indicators,
            spatial_indicators=spatial_indicators,
            action_verbs=action_verbs,
            question_words=question_words
        )
    
    def _calculate_position_score(self, entity: str, query: str) -> float:
        """위치 기반 중요도 계산"""
        entity_pos = query.find(entity)
        if entity_pos == -1:
            return 0.0
        
        # 쿼리 시작 부분에 가까울수록 높은 점수
        position_ratio = entity_pos / len(query)
        return max(0.0, 1.0 - position_ratio)
    
    def _calculate_frequency_score(self, entity: str, query: str) -> float:
        """빈도 기반 중요도 계산"""
        frequency = query.count(entity)
        # 빈도가 높을수록 높은 점수 (최대 0.4)
        return min(frequency * 0.2, 0.4)
    
    def _calculate_domain_score(self, entity: str) -> float:
        """도메인 기반 중요도 계산"""
        # 비즈니스 용어인지 확인
        if entity in self.business_terms:
            return self.domain_weights['business']
        
        # 시간 지시어인지 확인
        if entity in self.temporal_indicators:
            return self.domain_weights['time']
        
        # 공간 지시어인지 확인
        if entity in self.spatial_indicators:
            return self.domain_weights['location']
        
        # 사용자 관련 용어인지 확인
        user_terms = {'회원', '사용자', '고객', '멤버', '가입자', '로그인', '가입', '탈퇴'}
        if entity in user_terms:
            return self.domain_weights['user']
        
        # 제품 관련 용어인지 확인
        product_terms = {'상품', '제품', '아이템', '물품', '굿즈', '카테고리', '재고', '가격'}
        if entity in product_terms:
            return self.domain_weights['product']
        
        # 기본 점수
        return self.domain_weights['general']
    
    def _calculate_context_score(self, entity: str, context: ContextAnalysis) -> float:
        """컨텍스트 기반 중요도 계산"""
        score = 0.0
        
        # 비즈니스 용어가 많은 쿼리에서 비즈니스 관련 엔티티는 높은 점수
        if context.business_terms_count > 0 and entity in self.business_terms:
            score += 0.3
        
        # 시간 지시어가 있는 쿼리에서 시간 관련 엔티티는 높은 점수
        if context.temporal_indicators and entity in self.temporal_indicators:
            score += 0.2
        
        # 공간 지시어가 있는 쿼리에서 지역 관련 엔티티는 높은 점수
        if context.spatial_indicators and entity in self.spatial_indicators:
            score += 0.2
        
        # 행동 동사가 있는 쿼리에서 관련 엔티티는 높은 점수
        if context.action_verbs:
            score += 0.1
        
        # 질문어가 있는 쿼리에서 핵심 엔티티는 높은 점수
        if context.question_words:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_user_history_score(self, entity: str, user_history: Optional[Dict]) -> float:
        """사용자 이력 기반 중요도 계산"""
        if not user_history:
            return 0.0
        
        # 사용자가 자주 사용하는 용어는 높은 점수
        entity_frequency = user_history.get('entity_frequency', {}).get(entity, 0)
        max_frequency = max(user_history.get('entity_frequency', {}).values()) if user_history.get('entity_frequency') else 1
        
        return (entity_frequency / max_frequency) * 0.5 if max_frequency > 0 else 0.0
    
    def evaluate_entities(self, entities: List[str], query_context: str, 
                        user_history: Optional[Dict] = None) -> List[EntityImportance]:
        """
        엔티티 리스트의 중요도 평가
        
        Args:
            entities: 평가할 엔티티 리스트
            query_context: 쿼리 컨텍스트
            user_history: 사용자 이력 (선택사항)
            
        Returns:
            중요도가 평가된 엔티티 리스트
        """
        importance_results = []
        
        for entity in entities:
            importance = self.calculate_entity_importance(entity, query_context, user_history)
            importance_results.append(importance)
        
        # 중요도 순으로 정렬
        importance_results.sort(key=lambda x: x.final_score, reverse=True)
        
        # 순위 설정
        for i, result in enumerate(importance_results, 1):
            result.ranking = i
        
        return importance_results
    
    def get_top_entities(self, entities: List[str], query_context: str, 
                        top_k: int = 5, user_history: Optional[Dict] = None) -> List[EntityImportance]:
        """상위 K개 엔티티 반환"""
        importance_results = self.evaluate_entities(entities, query_context, user_history)
        return importance_results[:top_k]
    
    def analyze_entity_relationships(self, entities: List[str], query_context: str) -> Dict[str, Any]:
        """엔티티 간 관계 분석"""
        importance_results = self.evaluate_entities(entities, query_context)
        
        # 엔티티 타입별 분류
        entity_types = {
            'business': [],
            'time': [],
            'location': [],
            'user': [],
            'product': [],
            'general': []
        }
        
        for result in importance_results:
            entity = result.entity
            if entity in self.business_terms:
                entity_types['business'].append(result)
            elif entity in self.temporal_indicators:
                entity_types['time'].append(result)
            elif entity in self.spatial_indicators:
                entity_types['location'].append(result)
            elif entity in {'회원', '사용자', '고객', '멤버', '가입자'}:
                entity_types['user'].append(result)
            elif entity in {'상품', '제품', '아이템', '물품', '굿즈'}:
                entity_types['product'].append(result)
            else:
                entity_types['general'].append(result)
        
        return {
            'total_entities': len(entities),
            'entity_types': {k: len(v) for k, v in entity_types.items()},
            'top_entities': importance_results[:3],
            'importance_distribution': {
                'high': len([r for r in importance_results if r.final_score > 0.7]),
                'medium': len([r for r in importance_results if 0.4 <= r.final_score <= 0.7]),
                'low': len([r for r in importance_results if r.final_score < 0.4])
            }
        }
    
    def update_user_history(self, user_id: str, entities: List[str], query: str):
        """사용자 이력 업데이트"""
        if user_id not in self.user_history:
            self.user_history[user_id] = {
                'entity_frequency': {},
                'query_history': [],
                'preferred_entities': []
            }
        
        # 엔티티 빈도 업데이트
        for entity in entities:
            if entity not in self.user_history[user_id]['entity_frequency']:
                self.user_history[user_id]['entity_frequency'][entity] = 0
            self.user_history[user_id]['entity_frequency'][entity] += 1
        
        # 쿼리 이력 추가
        self.user_history[user_id]['query_history'].append({
            'query': query,
            'entities': entities,
            'timestamp': datetime.now().isoformat()
        })
        
        # 최근 100개 쿼리만 유지
        if len(self.user_history[user_id]['query_history']) > 100:
            self.user_history[user_id]['query_history'] = self.user_history[user_id]['query_history'][-100:]
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """사용자 선호도 분석"""
        if user_id not in self.user_history:
            return {}
        
        user_data = self.user_history[user_id]
        
        # 가장 자주 사용하는 엔티티
        top_entities = sorted(
            user_data['entity_frequency'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # 도메인별 선호도
        domain_preferences = {}
        for entity, frequency in user_data['entity_frequency'].items():
            if entity in self.business_terms:
                domain_preferences['business'] = domain_preferences.get('business', 0) + frequency
            elif entity in self.temporal_indicators:
                domain_preferences['time'] = domain_preferences.get('time', 0) + frequency
            elif entity in self.spatial_indicators:
                domain_preferences['location'] = domain_preferences.get('location', 0) + frequency
        
        return {
            'top_entities': top_entities,
            'domain_preferences': domain_preferences,
            'total_queries': len(user_data['query_history']),
            'unique_entities': len(user_data['entity_frequency'])
        }

def test_entity_importance_evaluator():
    """엔티티 중요도 평가기 테스트"""
    print("=== 엔티티 중요도 평가기 테스트 ===")
    
    # 평가기 초기화
    evaluator = EntityImportanceEvaluator()
    
    # 테스트 쿼리와 엔티티
    test_cases = [
        {
            "query": "8월 매출 분석 리포트 작성해줘",
            "entities": ["8월", "매출", "분석", "리포트", "작성"]
        },
        {
            "query": "지난달 신규 사용자 수가 가장 많은 지역은?",
            "entities": ["지난달", "신규", "사용자", "가장", "많은", "지역"]
        },
        {
            "query": "서울 지역 빨간색 티셔츠 판매량 추이",
            "entities": ["서울", "지역", "빨간색", "티셔츠", "판매량", "추이"]
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n테스트 케이스 {i}: {case['query']}")
        print(f"엔티티: {case['entities']}")
        
        # 중요도 평가
        importance_results = evaluator.evaluate_entities(
            case['entities'], 
            case['query']
        )
        
        print("중요도 평가 결과:")
        for result in importance_results:
            print(f"  {result.ranking}. {result.entity}")
            print(f"     최종 점수: {result.final_score:.3f}")
            print(f"     위치 점수: {result.position_score:.3f}")
            print(f"     빈도 점수: {result.frequency_score:.3f}")
            print(f"     도메인 점수: {result.domain_score:.3f}")
            print(f"     컨텍스트 점수: {result.context_score:.3f}")
        
        # 관계 분석
        relationships = evaluator.analyze_entity_relationships(
            case['entities'], 
            case['query']
        )
        print(f"\n관계 분석:")
        print(f"  총 엔티티 수: {relationships['total_entities']}")
        print(f"  엔티티 타입별 분포: {relationships['entity_types']}")
        print(f"  중요도 분포: {relationships['importance_distribution']}")
    
    # 사용자 이력 테스트
    print(f"\n=== 사용자 이력 테스트 ===")
    user_id = "test_user"
    
    # 사용자 이력 업데이트
    evaluator.update_user_history(user_id, ["매출", "분석", "리포트"], "매출 분석 리포트")
    evaluator.update_user_history(user_id, ["사용자", "지역", "분석"], "사용자 지역 분석")
    evaluator.update_user_history(user_id, ["매출", "이익", "비용"], "매출 이익 비용 분석")
    
    # 사용자 선호도 조회
    preferences = evaluator.get_user_preferences(user_id)
    print(f"사용자 선호도: {preferences}")

if __name__ == "__main__":
    test_entity_importance_evaluator()

