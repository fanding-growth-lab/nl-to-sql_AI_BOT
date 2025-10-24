"""
키워드 확장 및 동의어 사전 시스템

자연어 쿼리에서 엔티티 추출 정확도를 향상시키기 위한 키워드 확장 기능을 제공합니다.
도메인별 동의어 및 유사어 사전을 구축하고 키워드 확장 알고리즘을 구현합니다.
"""

import json
import logging
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import re

logger = logging.getLogger(__name__)

@dataclass
class SynonymEntry:
    """동의어 항목"""
    term: str
    synonyms: List[str]
    domain: str
    confidence: float
    relationship_type: str  # 'exact', 'partial', 'context_dependent'

@dataclass
class ExpansionResult:
    """키워드 확장 결과"""
    original_term: str
    expanded_terms: List[str]
    domain: str
    confidence: float
    expansion_type: str  # 'synonym', 'domain_specific', 'general'

class KeywordExpander:
    """키워드 확장 시스템"""
    
    def __init__(self, synonym_dict_path: Optional[str] = None):
        """
        키워드 확장기 초기화
        
        Args:
            synonym_dict_path: 동의어 사전 파일 경로
        """
        self.synonym_dict = {}
        self.domain_weights = {}
        self.expansion_cache = {}
        self.logger = logging.getLogger(__name__)
        
        # 기본 도메인 가중치 설정
        self._initialize_domain_weights()
        
        # 동의어 사전 로드
        if synonym_dict_path and Path(synonym_dict_path).exists():
            self.load_synonym_dictionary(synonym_dict_path)
        else:
            self._build_default_synonym_dictionary()
    
    def _initialize_domain_weights(self):
        """도메인별 가중치 초기화"""
        self.domain_weights = {
            'user': 1.0,      # 사용자 관련 용어
            'product': 0.9,   # 제품 관련 용어
            'time': 0.8,      # 시간 관련 용어
            'business': 1.2,  # 비즈니스 용어 (높은 가중치)
            'location': 0.7,  # 지역 관련 용어
            'general': 0.5    # 일반 용어
        }
    
    def _build_default_synonym_dictionary(self):
        """기본 동의어 사전 구축"""
        self.synonym_dict = {
            # 사용자 관련 용어
            'user': {
                '회원': ['유저', '사용자', '고객', '멤버', '가입자', '회원가입자'],
                '가입': ['등록', '생성', '신규', '온보딩', '가입자'],
                '로그인': ['접속', '로그온', '인증', '로그인'],
                '탈퇴': ['삭제', '제거', '해지', '탈퇴'],
                '활성': ['활성화', '활성상태', '정상', '활성'],
                '비활성': ['비활성화', '비활성상태', '정지', '비활성']
            },
            
            # 제품 관련 용어
            'product': {
                '상품': ['제품', '아이템', '물품', '굿즈', '상품'],
                '카테고리': ['분류', '종류', '유형', '카테고리'],
                '재고': ['재고량', '재고수량', '재고', '재고현황'],
                '가격': ['금액', '비용', '요금', '가격'],
                '할인': ['할인율', '할인가', '할인', '세일']
            },
            
            # 시간 관련 용어
            'time': {
                '월별': ['매월', '월간', '월 단위', '월마다', '월별'],
                '일별': ['매일', '일간', '일 단위', '일마다', '일별'],
                '년별': ['매년', '년간', '년 단위', '년마다', '년별'],
                '분기별': ['분기간', '분기 단위', '분기마다', '분기별'],
                '주별': ['매주', '주간', '주 단위', '주마다', '주별'],
                '이번달': ['이번월', '현재월', '이번달'],
                '지난달': ['전월', '이전월', '지난달'],
                '올해': ['이번년', '현재년', '올해'],
                '작년': ['전년', '이전년', '작년']
            },
            
            # 비즈니스 용어
            'business': {
                '매출': ['수익', '판매액', '수입', '매상', '판매 실적', '매출액'],
                '이익': ['수익', '이익', '순이익', '영업이익'],
                '비용': ['지출', '비용', '경비', '운영비'],
                'ROI': ['투자수익률', 'ROI', '수익률'],
                '전환율': ['전환율', '컨버전율', '전환비율'],
                '성장률': ['증가율', '성장률', '증가비율'],
                '고객': ['클라이언트', '고객', '구매자', '소비자'],
                '주문': ['주문', '구매', '결제', '주문건수'],
                '방문자': ['방문자', '방문객', '사용자', '방문자수'],
                '클릭수': ['클릭', '클릭수', '클릭횟수', '클릭량']
            },
            
            # 지역 관련 용어
            'location': {
                '서울': ['서울시', '서울', '서울특별시'],
                '부산': ['부산시', '부산', '부산광역시'],
                '대구': ['대구시', '대구', '대구광역시'],
                '인천': ['인천시', '인천', '인천광역시'],
                '광주': ['광주시', '광주', '광주광역시'],
                '대전': ['대전시', '대전', '대전광역시'],
                '울산': ['울산시', '울산', '울산광역시'],
                '세종': ['세종시', '세종', '세종특별자치시']
            },
            
            # 일반 용어
            'general': {
                '분석': ['분석', '검토', '조사', '검증'],
                '보고서': ['리포트', '보고서', '문서', '자료'],
                '현황': ['상황', '현황', '상태', '현재상태'],
                '추이': ['트렌드', '추이', '변화', '흐름'],
                '비교': ['대비', '비교', '대조', '대비분석'],
                '증가': ['상승', '증가', '증대', '늘어남'],
                '감소': ['하락', '감소', '감소', '줄어듦'],
                '최고': ['최대', '최고', '최상', '최대값'],
                '최저': ['최소', '최저', '최하', '최소값'],
                '평균': ['평균', '평균값', '평균치', '평균']
            }
        }
    
    def load_synonym_dictionary(self, file_path: str):
        """동의어 사전 파일 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.synonym_dict = json.load(f)
            self.logger.info(f"동의어 사전 로드 완료: {file_path}")
        except Exception as e:
            self.logger.error(f"동의어 사전 로드 실패: {e}")
            self._build_default_synonym_dictionary()
    
    def save_synonym_dictionary(self, file_path: str):
        """동의어 사전 파일 저장"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.synonym_dict, f, ensure_ascii=False, indent=2)
            self.logger.info(f"동의어 사전 저장 완료: {file_path}")
        except Exception as e:
            self.logger.error(f"동의어 사전 저장 실패: {e}")
    
    def expand_keyword(self, keyword: str, domain: Optional[str] = None) -> ExpansionResult:
        """
        키워드를 동의어로 확장
        
        Args:
            keyword: 확장할 키워드
            domain: 도메인 (선택사항)
            
        Returns:
            ExpansionResult: 확장 결과
        """
        # 캐시 확인
        cache_key = f"{keyword}_{domain or 'all'}"
        if cache_key in self.expansion_cache:
            return self.expansion_cache[cache_key]
        
        expanded_terms = [keyword]  # 원본 키워드 포함
        confidence = 1.0
        expansion_type = "exact"
        
        # 도메인별 확장
        if domain and domain in self.synonym_dict:
            domain_synonyms = self._get_domain_synonyms(keyword, domain)
            if domain_synonyms:
                expanded_terms.extend(domain_synonyms)
                confidence = max(confidence, 0.9)
                expansion_type = "domain_specific"
        
        # 모든 도메인에서 검색
        all_synonyms = self._get_all_synonyms(keyword)
        if all_synonyms:
            expanded_terms.extend(all_synonyms)
            confidence = max(confidence, 0.8)
            expansion_type = "synonym"
        
        # 중복 제거 및 정렬
        expanded_terms = list(set(expanded_terms))
        expanded_terms.sort()
        
        result = ExpansionResult(
            original_term=keyword,
            expanded_terms=expanded_terms,
            domain=domain or "general",
            confidence=confidence,
            expansion_type=expansion_type
        )
        
        # 캐시에 저장
        self.expansion_cache[cache_key] = result
        
        return result
    
    def _get_domain_synonyms(self, keyword: str, domain: str) -> List[str]:
        """특정 도메인에서 동의어 검색"""
        synonyms = []
        if domain in self.synonym_dict:
            for term, term_synonyms in self.synonym_dict[domain].items():
                if keyword == term or keyword in term_synonyms:
                    synonyms.extend(term_synonyms)
                    synonyms.append(term)
        return synonyms
    
    def _get_all_synonyms(self, keyword: str) -> List[str]:
        """모든 도메인에서 동의어 검색"""
        synonyms = []
        for domain, domain_dict in self.synonym_dict.items():
            for term, term_synonyms in domain_dict.items():
                if keyword == term or keyword in term_synonyms:
                    synonyms.extend(term_synonyms)
                    synonyms.append(term)
        return synonyms
    
    def expand_entities(self, entities: List[Dict], domain: Optional[str] = None) -> List[Dict]:
        """
        엔티티 리스트의 키워드 확장
        
        Args:
            entities: 엔티티 리스트
            domain: 도메인 (선택사항)
            
        Returns:
            확장된 엔티티 리스트
        """
        expanded_entities = []
        
        for entity in entities:
            # 원본 엔티티 추가
            expanded_entities.append(entity)
            
            # 키워드 확장
            if 'value' in entity:
                expansion_result = self.expand_keyword(entity['value'], domain)
                
                # 확장된 용어들을 새로운 엔티티로 추가
                for expanded_term in expansion_result.expanded_terms:
                    if expanded_term != entity['value']:  # 원본과 다른 경우만
                        expanded_entity = entity.copy()
                        expanded_entity['value'] = expanded_term
                        expanded_entity['original_term'] = entity['value']
                        expanded_entity['expansion_confidence'] = expansion_result.confidence
                        expanded_entity['expansion_type'] = expansion_result.expansion_type
                        expanded_entities.append(expanded_entity)
        
        return expanded_entities
    
    def get_domain_importance(self, term: str, domain: str) -> float:
        """도메인별 용어 중요도 계산"""
        base_importance = self.domain_weights.get(domain, 0.5)
        
        # 비즈니스 용어는 높은 중요도
        if domain == 'business':
            business_terms = ['매출', '이익', '비용', 'ROI', '전환율', '성장률']
            if any(business_term in term for business_term in business_terms):
                return base_importance * 1.5
        
        return base_importance
    
    def add_synonym(self, term: str, synonym: str, domain: str, confidence: float = 0.8):
        """동의어 추가"""
        if domain not in self.synonym_dict:
            self.synonym_dict[domain] = {}
        
        if term not in self.synonym_dict[domain]:
            self.synonym_dict[domain][term] = []
        
        if synonym not in self.synonym_dict[domain][term]:
            self.synonym_dict[domain][term].append(synonym)
            self.logger.info(f"동의어 추가: {term} -> {synonym} (도메인: {domain})")
    
    def remove_synonym(self, term: str, synonym: str, domain: str):
        """동의어 제거"""
        if domain in self.synonym_dict and term in self.synonym_dict[domain]:
            if synonym in self.synonym_dict[domain][term]:
                self.synonym_dict[domain][term].remove(synonym)
                self.logger.info(f"동의어 제거: {term} -> {synonym} (도메인: {domain})")
    
    def get_statistics(self) -> Dict:
        """동의어 사전 통계 정보"""
        total_terms = 0
        total_synonyms = 0
        domain_stats = {}
        
        for domain, domain_dict in self.synonym_dict.items():
            domain_terms = len(domain_dict)
            domain_synonyms = sum(len(synonyms) for synonyms in domain_dict.values())
            
            domain_stats[domain] = {
                'terms': domain_terms,
                'synonyms': domain_synonyms
            }
            
            total_terms += domain_terms
            total_synonyms += domain_synonyms
        
        return {
            'total_domains': len(self.synonym_dict),
            'total_terms': total_terms,
            'total_synonyms': total_synonyms,
            'domain_stats': domain_stats,
            'cache_size': len(self.expansion_cache)
        }

def test_keyword_expander():
    """키워드 확장기 테스트"""
    print("=== 키워드 확장기 테스트 ===")
    
    # 키워드 확장기 초기화
    expander = KeywordExpander()
    
    # 테스트 케이스
    test_cases = [
        ("회원", "user"),
        ("매출", "business"),
        ("상품", "product"),
        ("8월", "time"),
        ("서울", "location")
    ]
    
    for keyword, domain in test_cases:
        print(f"\n키워드: {keyword} (도메인: {domain})")
        result = expander.expand_keyword(keyword, domain)
        print(f"  확장 결과: {result.expanded_terms}")
        print(f"  신뢰도: {result.confidence}")
        print(f"  확장 타입: {result.expansion_type}")
    
    # 통계 정보
    print(f"\n=== 동의어 사전 통계 ===")
    stats = expander.get_statistics()
    print(f"총 도메인 수: {stats['total_domains']}")
    print(f"총 용어 수: {stats['total_terms']}")
    print(f"총 동의어 수: {stats['total_synonyms']}")
    print(f"캐시 크기: {stats['cache_size']}")
    
    # 도메인별 통계
    for domain, domain_stats in stats['domain_stats'].items():
        print(f"  {domain}: {domain_stats['terms']}개 용어, {domain_stats['synonyms']}개 동의어")

if __name__ == "__main__":
    test_keyword_expander()

