#!/usr/bin/env python3
"""
쿼리 컨텍스트 분석 및 활용 메커니즘
사용자 쿼리의 컨텍스트를 분석하고 스키마 매핑에 활용하는 고급 시스템
"""

import logging
import re
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ContextType(Enum):
    """컨텍스트 타입"""
    TEMPORAL = "temporal"  # 시간 관련
    SPATIAL = "spatial"    # 공간 관련
    BUSINESS = "business"  # 비즈니스 도메인
    TECHNICAL = "technical"  # 기술적
    USER_PREFERENCE = "user_preference"  # 사용자 선호도
    QUERY_PATTERN = "query_pattern"  # 쿼리 패턴
    DATA_SEMANTICS = "data_semantics"  # 데이터 의미론


class ContextConfidence(Enum):
    """컨텍스트 신뢰도"""
    HIGH = "high"      # 0.8-1.0
    MEDIUM = "medium"  # 0.5-0.8
    LOW = "low"        # 0.0-0.5


@dataclass
class ContextElement:
    """컨텍스트 요소"""
    context_type: ContextType
    value: Any
    confidence: float
    source: str  # 컨텍스트 출처
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryContext:
    """쿼리 컨텍스트"""
    query_id: str
    user_query: str
    context_elements: List[ContextElement]
    inferred_intent: str
    domain_context: str
    temporal_context: Optional[str] = None
    spatial_context: Optional[str] = None
    business_context: Optional[str] = None
    confidence_score: float = 0.0
    creation_time: datetime = field(default_factory=datetime.now)


@dataclass
class ContextInference:
    """컨텍스트 추론 결과"""
    inferred_context: str
    confidence: float
    reasoning: str
    supporting_evidence: List[str]
    alternative_interpretations: List[str] = field(default_factory=list)


class ContextAnalyzer:
    """
    쿼리 컨텍스트 분석 및 활용 메커니즘
    """
    
    def __init__(self, entity_extractor=None, schema_mapper=None):
        """
        ContextAnalyzer 초기화
        
        Args:
            entity_extractor: EntityExtractor 인스턴스
            schema_mapper: SchemaMapper 인스턴스
        """
        self.entity_extractor = entity_extractor
        self.schema_mapper = schema_mapper
        self.context_cache = {}
        self.user_history = defaultdict(list)
        self.domain_patterns = self._initialize_domain_patterns()
        self.temporal_patterns = self._initialize_temporal_patterns()
        self.spatial_patterns = self._initialize_spatial_patterns()
        self.business_patterns = self._initialize_business_patterns()
        self.context_weights = self._initialize_context_weights()
        
        # 컨텍스트 분석 통계
        self.analysis_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "context_types_detected": defaultdict(int),
            "average_confidence": 0.0,
            "cache_hit_rate": 0.0
        }
        
        logger.info("ContextAnalyzer initialized")
    
    def analyze_query_context(self, user_query: str, user_id: str = None, 
                            session_id: str = None, additional_context: Dict[str, Any] = None) -> QueryContext:
        """
        쿼리 컨텍스트 분석
        
        Args:
            user_query: 사용자 쿼리
            user_id: 사용자 ID
            session_id: 세션 ID
            additional_context: 추가 컨텍스트 정보
            
        Returns:
            QueryContext: 분석된 컨텍스트
        """
        try:
            logger.info(f"Analyzing context for query: {user_query[:100]}...")
            
            # 쿼리 ID 생성
            query_id = self._generate_query_id(user_query, user_id, session_id)
            
            # 캐시 확인
            cache_key = self._generate_cache_key(user_query, user_id)
            if cache_key in self.context_cache:
                cached_context = self.context_cache[cache_key]
                if self._is_cache_valid(cached_context):
                    logger.debug("Using cached context analysis")
                    self.analysis_stats["cache_hit_rate"] += 1
                    return cached_context
            
            # 컨텍스트 요소들 분석
            context_elements = []
            
            # 시간적 컨텍스트 분석
            temporal_context = self._analyze_temporal_context(user_query)
            if temporal_context:
                context_elements.append(temporal_context)
            
            # 공간적 컨텍스트 분석
            spatial_context = self._analyze_spatial_context(user_query)
            if spatial_context:
                context_elements.append(spatial_context)
            
            # 비즈니스 도메인 컨텍스트 분석
            business_context = self._analyze_business_context(user_query)
            if business_context:
                context_elements.append(business_context)
            
            # 기술적 컨텍스트 분석
            technical_context = self._analyze_technical_context(user_query)
            if technical_context:
                context_elements.append(technical_context)
            
            # 사용자 선호도 컨텍스트 분석
            user_preference_context = self._analyze_user_preference_context(user_query, user_id)
            if user_preference_context:
                context_elements.append(user_preference_context)
            
            # 쿼리 패턴 컨텍스트 분석
            query_pattern_context = self._analyze_query_pattern_context(user_query)
            if query_pattern_context:
                context_elements.append(query_pattern_context)
            
            # 데이터 의미론 컨텍스트 분석
            data_semantics_context = self._analyze_data_semantics_context(user_query)
            if data_semantics_context:
                context_elements.append(data_semantics_context)
            
            # 의도 추론
            inferred_intent = self._infer_query_intent(user_query, context_elements)
            
            # 도메인 컨텍스트 결정
            domain_context = self._determine_domain_context(context_elements)
            
            # 전체 신뢰도 계산
            confidence_score = self._calculate_overall_confidence(context_elements)
            
            # QueryContext 생성
            query_context = QueryContext(
                query_id=query_id,
                user_query=user_query,
                context_elements=context_elements,
                inferred_intent=inferred_intent,
                domain_context=domain_context,
                temporal_context=temporal_context.value if temporal_context else None,
                spatial_context=spatial_context.value if spatial_context else None,
                business_context=business_context.value if business_context else None,
                confidence_score=confidence_score
            )
            
            # 캐시에 저장
            self.context_cache[cache_key] = query_context
            
            # 사용자 히스토리에 추가
            if user_id:
                self.user_history[user_id].append(query_context)
                # 히스토리 크기 제한
                if len(self.user_history[user_id]) > 100:
                    self.user_history[user_id] = self.user_history[user_id][-100:]
            
            # 통계 업데이트
            self._update_analysis_stats(query_context)
            
            logger.info(f"Context analysis completed: {len(context_elements)} elements, confidence: {confidence_score:.2f}")
            
            return query_context
            
        except Exception as e:
            logger.error(f"Failed to analyze query context: {str(e)}", exc_info=True)
            return self._create_fallback_context(user_query)
    
    def _analyze_temporal_context(self, user_query: str) -> Optional[ContextElement]:
        """시간적 컨텍스트 분석"""
        try:
            temporal_indicators = []
            confidence = 0.0
            
            # 날짜 패턴 검색
            date_patterns = [
                r'(\d{4})년',
                r'(\d{1,2})월',
                r'(\d{1,2})일',
                r'(\d{4})-(\d{1,2})-(\d{1,2})',
                r'최근',
                r'지난',
                r'이번',
                r'올해',
                r'작년',
                r'내년',
                r'분기',
                r'주간',
                r'일간',
                r'시간별'
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, user_query)
                if matches:
                    temporal_indicators.extend(matches)
                    confidence += 0.4  # 0.2 -> 0.4로 증가
            
            # 시간 관련 키워드
            time_keywords = ['분석', '트렌드', '변화', '증가', '감소', '성과', '실적']
            for keyword in time_keywords:
                if keyword in user_query:
                    confidence += 0.2  # 0.1 -> 0.2로 증가
            
            if temporal_indicators or confidence > 0.3:
                temporal_value = {
                    "indicators": temporal_indicators,
                    "keywords": [kw for kw in time_keywords if kw in user_query],
                    "granularity": self._determine_temporal_granularity(user_query)
                }
                
                return ContextElement(
                    context_type=ContextType.TEMPORAL,
                    value=temporal_value,
                    confidence=min(confidence, 1.0),
                    source="pattern_matching",
                    metadata={"patterns_found": len(temporal_indicators)}
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to analyze temporal context: {str(e)}")
            return None
    
    def _analyze_spatial_context(self, user_query: str) -> Optional[ContextElement]:
        """공간적 컨텍스트 분석"""
        try:
            spatial_indicators = []
            confidence = 0.0
            
            # 지역/국가 패턴
            spatial_patterns = [
                r'([가-힣]+시)',
                r'([가-힣]+구)',
                r'([가-힣]+동)',
                r'([가-힣]+국)',
                r'전국',
                r'지역별',
                r'도시별',
                r'구별'
            ]
            
            for pattern in spatial_patterns:
                matches = re.findall(pattern, user_query)
                if matches:
                    spatial_indicators.extend(matches)
                    confidence += 0.5  # 0.3 -> 0.5로 증가
            
            # 공간 관련 키워드
            spatial_keywords = ['지역', '위치', '거리', '근처', '주변']
            for keyword in spatial_keywords:
                if keyword in user_query:
                    confidence += 0.2  # 0.1 -> 0.2로 증가
            
            if spatial_indicators or confidence > 0.3:
                spatial_value = {
                    "indicators": spatial_indicators,
                    "keywords": [kw for kw in spatial_keywords if kw in user_query],
                    "granularity": self._determine_spatial_granularity(user_query)
                }
                
                return ContextElement(
                    context_type=ContextType.SPATIAL,
                    value=spatial_value,
                    confidence=min(confidence, 1.0),
                    source="pattern_matching",
                    metadata={"patterns_found": len(spatial_indicators)}
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to analyze spatial context: {str(e)}")
            return None
    
    def _analyze_business_context(self, user_query: str) -> Optional[ContextElement]:
        """비즈니스 도메인 컨텍스트 분석"""
        try:
            business_domains = []
            confidence = 0.0
            
            # 비즈니스 도메인 키워드 매핑
            domain_keywords = {
                "funding": ["펀딩", "투자", "후원", "모금", "자금"],
                "member": ["회원", "사용자", "고객", "가입자"],
                "creator": ["창작자", "제작자", "개발자", "아티스트"],
                "payment": ["결제", "충전", "포인트", "금액", "가격"],
                "project": ["프로젝트", "캠페인", "기획", "계획"],
                "analytics": ["분석", "통계", "데이터", "리포트", "성과"],
                "marketing": ["마케팅", "홍보", "광고", "프로모션"]
            }
            
            for domain, keywords in domain_keywords.items():
                matches = [kw for kw in keywords if kw in user_query]
                if matches:
                    business_domains.append(domain)
                    confidence += 0.4 * len(matches)  # 0.2 -> 0.4로 증가
            
            if business_domains or confidence > 0.3:
                business_value = {
                    "domains": business_domains,
                    "primary_domain": business_domains[0] if business_domains else "general",
                    "domain_confidence": confidence
                }
                
                return ContextElement(
                    context_type=ContextType.BUSINESS,
                    value=business_value,
                    confidence=min(confidence, 1.0),
                    source="keyword_matching",
                    metadata={"domains_found": len(business_domains)}
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to analyze business context: {str(e)}")
            return None
    
    def _analyze_technical_context(self, user_query: str) -> Optional[ContextElement]:
        """기술적 컨텍스트 분석"""
        try:
            technical_indicators = []
            confidence = 0.0
            
            # 기술적 키워드
            tech_keywords = {
                "database": ["테이블", "컬럼", "데이터베이스", "스키마"],
                "query": ["쿼리", "조회", "검색", "필터"],
                "performance": ["성능", "속도", "최적화", "효율"],
                "api": ["API", "인터페이스", "엔드포인트"],
                "security": ["보안", "권한", "인증", "암호화"]
            }
            
            for category, keywords in tech_keywords.items():
                matches = [kw for kw in keywords if kw in user_query]
                if matches:
                    technical_indicators.append({
                        "category": category,
                        "keywords": matches
                    })
                    confidence += 0.3 * len(matches)  # 0.15 -> 0.3으로 증가
            
            # SQL 관련 키워드
            sql_keywords = ["SELECT", "WHERE", "JOIN", "GROUP BY", "ORDER BY", "COUNT", "SUM", "AVG"]
            sql_matches = [kw for kw in sql_keywords if kw.upper() in user_query.upper()]
            if sql_matches:
                confidence += 0.2 * len(sql_matches)  # 0.1 -> 0.2로 증가
            
            if technical_indicators or confidence > 0.3:
                technical_value = {
                    "indicators": technical_indicators,
                    "sql_keywords": sql_matches,
                    "complexity_level": self._assess_query_complexity(user_query)
                }
                
                return ContextElement(
                    context_type=ContextType.TECHNICAL,
                    value=technical_value,
                    confidence=min(confidence, 1.0),
                    source="keyword_analysis",
                    metadata={"categories_found": len(technical_indicators)}
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to analyze technical context: {str(e)}")
            return None
    
    def _analyze_user_preference_context(self, user_query: str, user_id: str = None) -> Optional[ContextElement]:
        """사용자 선호도 컨텍스트 분석"""
        try:
            if not user_id:
                return None
            
            user_history = self.user_history.get(user_id, [])
            if not user_history:
                return None
            
            # 사용자 히스토리 분석
            preferred_domains = defaultdict(int)
            preferred_context_types = defaultdict(int)
            query_patterns = []
            
            for context in user_history[-10:]:  # 최근 10개 쿼리만 분석
                if context.business_context:
                    preferred_domains[context.business_context] += 1
                
                for element in context.context_elements:
                    # ContextType을 문자열로 변환하여 해시 가능하게 만듦
                    preferred_context_types[element.context_type.value] += 1
                
                query_patterns.append(context.user_query)
            
            # 선호도 점수 계산
            most_preferred_domain = max(preferred_domains.items(), key=lambda x: x[1])[0] if preferred_domains else None
            most_preferred_context_type = max(preferred_context_types.items(), key=lambda x: x[1])[0] if preferred_context_types else None
            
            confidence = min(len(user_history) / 10.0, 1.0)  # 히스토리 길이에 비례
            
            if confidence > 0.3:
                preference_value = {
                    "preferred_domain": most_preferred_domain,
                    "preferred_context_type": most_preferred_context_type,
                    "domain_frequency": dict(preferred_domains),
                    "context_type_frequency": dict(preferred_context_types),
                    "history_length": len(user_history)
                }
                
                return ContextElement(
                    context_type=ContextType.USER_PREFERENCE,
                    value=preference_value,
                    confidence=confidence,
                    source="user_history",
                    metadata={"history_analyzed": len(user_history)}
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to analyze user preference context: {str(e)}")
            return None
    
    def _analyze_query_pattern_context(self, user_query: str) -> Optional[ContextElement]:
        """쿼리 패턴 컨텍스트 분석"""
        try:
            pattern_indicators = []
            confidence = 0.0
            
            # 쿼리 패턴 분류
            query_patterns = {
                "aggregation": ["개수", "합계", "평균", "최대", "최소", "COUNT", "SUM", "AVG", "MAX", "MIN"],
                "filtering": ["조건", "필터", "WHERE", "특정", "선택"],
                "grouping": ["그룹", "분류", "GROUP BY", "별로"],
                "sorting": ["정렬", "순서", "ORDER BY", "오름차순", "내림차순"],
                "joining": ["연결", "조인", "JOIN", "관계", "매칭"],
                "comparison": ["비교", "대비", "차이", "증가", "감소"],
                "trend": ["트렌드", "추이", "변화", "경향", "패턴"]
            }
            
            for pattern_type, keywords in query_patterns.items():
                matches = [kw for kw in keywords if kw in user_query]
                if matches:
                    pattern_indicators.append({
                        "pattern_type": pattern_type,
                        "matches": matches,
                        "confidence": len(matches) * 0.2  # 0.1 -> 0.2로 증가
                    })
                    confidence += len(matches) * 0.2  # 0.1 -> 0.2로 증가
            
            # 쿼리 복잡도 평가
            complexity_score = self._assess_query_complexity(user_query)
            
            if pattern_indicators or confidence > 0.3:
                pattern_value = {
                    "patterns": pattern_indicators,
                    "complexity_score": complexity_score,
                    "query_type": self._classify_query_type(user_query)
                }
                
                return ContextElement(
                    context_type=ContextType.QUERY_PATTERN,
                    value=pattern_value,
                    confidence=min(confidence, 1.0),
                    source="pattern_analysis",
                    metadata={"patterns_found": len(pattern_indicators)}
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to analyze query pattern context: {str(e)}")
            return None
    
    def _analyze_data_semantics_context(self, user_query: str) -> Optional[ContextElement]:
        """데이터 의미론 컨텍스트 분석"""
        try:
            semantic_indicators = []
            confidence = 0.0
            
            # 데이터 의미론 키워드
            semantic_keywords = {
                "temporal": ["시간", "날짜", "기간", "시점", "순서"],
                "quantitative": ["수량", "금액", "크기", "규모", "양"],
                "qualitative": ["상태", "품질", "등급", "수준", "특성"],
                "categorical": ["카테고리", "분류", "유형", "종류", "그룹"],
                "relational": ["관계", "연관", "상호작용", "연결", "매칭"],
                "comparative": ["비교", "대조", "차이", "유사", "동일"]
            }
            
            for semantic_type, keywords in semantic_keywords.items():
                matches = [kw for kw in keywords if kw in user_query]
                if matches:
                    semantic_indicators.append({
                        "semantic_type": semantic_type,
                        "matches": matches,
                        "strength": len(matches)
                    })
                    confidence += len(matches) * 0.2  # 0.1 -> 0.2로 증가
            
            # 데이터 타입 추론
            data_types = self._infer_data_types(user_query)
            
            if semantic_indicators or data_types:
                semantic_value = {
                    "indicators": semantic_indicators,
                    "inferred_data_types": data_types,
                    "semantic_complexity": len(semantic_indicators)
                }
                
                return ContextElement(
                    context_type=ContextType.DATA_SEMANTICS,
                    value=semantic_value,
                    confidence=min(confidence, 1.0),
                    source="semantic_analysis",
                    metadata={"semantic_types_found": len(semantic_indicators)}
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to analyze data semantics context: {str(e)}")
            return None
    
    def _infer_query_intent(self, user_query: str, context_elements: List[ContextElement]) -> str:
        """쿼리 의도 추론"""
        try:
            intent_scores = defaultdict(float)
            
            # 기본 의도 키워드
            intent_keywords = {
                "analyze": ["분석", "검토", "확인", "살펴보기"],
                "report": ["리포트", "보고서", "결과", "요약"],
                "compare": ["비교", "대조", "차이", "대비"],
                "trend": ["트렌드", "추이", "변화", "경향"],
                "filter": ["필터", "조건", "특정", "선택"],
                "aggregate": ["집계", "합계", "총계", "평균"],
                "search": ["검색", "찾기", "조회", "확인"]
            }
            
            for intent, keywords in intent_keywords.items():
                for keyword in keywords:
                    if keyword in user_query:
                        intent_scores[intent] += 0.3  # 0.2 -> 0.3으로 증가
            
            # 컨텍스트 요소 기반 의도 강화
            for element in context_elements:
                if element.context_type == ContextType.QUERY_PATTERN:
                    pattern_value = element.value
                    if "aggregation" in pattern_value.get("patterns", {}):
                        intent_scores["aggregate"] += 0.5  # 0.3 -> 0.5로 증가
                    if "trend" in pattern_value.get("patterns", {}):
                        intent_scores["trend"] += 0.5  # 0.3 -> 0.5로 증가
                    if "comparison" in pattern_value.get("patterns", {}):
                        intent_scores["compare"] += 0.5  # 0.3 -> 0.5로 증가
            
            # 가장 높은 점수의 의도 반환
            if intent_scores:
                return max(intent_scores.items(), key=lambda x: x[1])[0]
            else:
                return "general_query"
                
        except Exception as e:
            logger.error(f"Failed to infer query intent: {str(e)}")
            return "unknown"
    
    def _determine_domain_context(self, context_elements: List[ContextElement]) -> str:
        """도메인 컨텍스트 결정"""
        try:
            domain_scores = defaultdict(float)
            
            for element in context_elements:
                if element.context_type == ContextType.BUSINESS:
                    business_value = element.value
                    primary_domain = business_value.get("primary_domain", "general")
                    domain_scores[primary_domain] += element.confidence
                
                elif element.context_type == ContextType.USER_PREFERENCE:
                    preference_value = element.value
                    preferred_domain = preference_value.get("preferred_domain")
                    if preferred_domain:
                        domain_scores[preferred_domain] += element.confidence * 0.5
            
            if domain_scores:
                return max(domain_scores.items(), key=lambda x: x[1])[0]
            else:
                return "general"
                
        except Exception as e:
            logger.error(f"Failed to determine domain context: {str(e)}")
            return "general"
    
    def _calculate_overall_confidence(self, context_elements: List[ContextElement]) -> float:
        """전체 신뢰도 계산"""
        try:
            if not context_elements:
                return 0.0
            
            # 가중 평균 계산
            total_weight = 0.0
            weighted_confidence = 0.0
            
            for element in context_elements:
                weight = self.context_weights.get(element.context_type, 1.0)
                weighted_confidence += element.confidence * weight
                total_weight += weight
            
            base_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0
            
            # 컨텍스트 요소 개수에 따른 보너스 점수
            element_bonus = min(len(context_elements) * 0.05, 0.2)  # 최대 0.2 보너스
            
            # 높은 신뢰도 요소가 많을 때 추가 보너스
            high_confidence_count = sum(1 for e in context_elements if e.confidence > 0.6)
            high_confidence_bonus = min(high_confidence_count * 0.1, 0.3)  # 최대 0.3 보너스
            
            # 최종 신뢰도 계산 (최대 1.0)
            final_confidence = min(base_confidence + element_bonus + high_confidence_bonus, 1.0)
            
            return final_confidence
            
        except Exception as e:
            logger.error(f"Failed to calculate overall confidence: {str(e)}")
            return 0.0
    
    def get_context_recommendations(self, query_context: QueryContext) -> List[str]:
        """컨텍스트 기반 권장사항 생성"""
        try:
            recommendations = []
            
            # 컨텍스트 요소별 권장사항
            for element in query_context.context_elements:
                if element.context_type == ContextType.TEMPORAL:
                    if element.confidence > 0.7:
                        recommendations.append("시간 범위를 명확히 지정하여 더 정확한 결과를 얻을 수 있습니다")
                
                elif element.context_type == ContextType.BUSINESS:
                    if element.confidence > 0.6:
                        recommendations.append(f"{element.value.get('primary_domain', '해당')} 도메인에 특화된 쿼리를 사용하세요")
                
                elif element.context_type == ContextType.TECHNICAL:
                    complexity = element.value.get("complexity_level", "medium")
                    if complexity == "high":
                        recommendations.append("복잡한 쿼리는 단계별로 나누어 실행하는 것을 권장합니다")
                
                elif element.context_type == ContextType.QUERY_PATTERN:
                    patterns = element.value.get("patterns", [])
                    if len(patterns) > 3:
                        recommendations.append("여러 패턴이 포함된 쿼리는 성능에 영향을 줄 수 있습니다")
            
            # 전체 신뢰도 기반 권장사항
            if query_context.confidence_score < 0.5:
                recommendations.append("쿼리를 더 구체적으로 작성하면 더 정확한 결과를 얻을 수 있습니다")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get context recommendations: {str(e)}")
            return ["일반적인 쿼리 최적화를 권장합니다"]
    
    def _generate_query_id(self, user_query: str, user_id: str = None, session_id: str = None) -> str:
        """쿼리 ID 생성"""
        try:
            import hashlib
            
            query_string = f"{user_query}_{user_id or 'anonymous'}_{session_id or 'default'}"
            query_hash = hashlib.md5(query_string.encode()).hexdigest()[:8]
            
            return f"query_{query_hash}_{int(datetime.now().timestamp())}"
            
        except Exception as e:
            logger.error(f"Failed to generate query ID: {str(e)}")
            return f"query_{int(datetime.now().timestamp())}"
    
    def _generate_cache_key(self, user_query: str, user_id: str = None) -> str:
        """캐시 키 생성"""
        try:
            import hashlib
            
            cache_string = f"{user_query}_{user_id or 'anonymous'}"
            return hashlib.md5(cache_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to generate cache key: {str(e)}")
            return user_query
    
    def _is_cache_valid(self, cached_context: QueryContext) -> bool:
        """캐시 유효성 확인"""
        try:
            # 1시간 캐시 유효 기간
            cache_duration = timedelta(hours=1)
            return datetime.now() - cached_context.creation_time < cache_duration
            
        except Exception as e:
            logger.error(f"Failed to check cache validity: {str(e)}")
            return False
    
    def _create_fallback_context(self, user_query: str) -> QueryContext:
        """폴백 컨텍스트 생성"""
        try:
            return QueryContext(
                query_id=self._generate_query_id(user_query),
                user_query=user_query,
                context_elements=[],
                inferred_intent="general_query",
                domain_context="general",
                confidence_score=0.1
            )
            
        except Exception as e:
            logger.error(f"Failed to create fallback context: {str(e)}")
            return QueryContext(
                query_id="fallback",
                user_query=user_query,
                context_elements=[],
                inferred_intent="unknown",
                domain_context="general",
                confidence_score=0.0
            )
    
    def _initialize_domain_patterns(self) -> Dict[str, List[str]]:
        """도메인 패턴 초기화"""
        return {
            "funding": ["펀딩", "투자", "후원", "모금", "자금", "fanding"],
            "member": ["회원", "사용자", "고객", "가입자", "member"],
            "creator": ["창작자", "제작자", "개발자", "아티스트", "creator"],
            "payment": ["결제", "충전", "포인트", "금액", "가격", "payment"]
        }
    
    def _initialize_temporal_patterns(self) -> Dict[str, List[str]]:
        """시간적 패턴 초기화"""
        return {
            "absolute": ["2024년", "2023년", "2022년"],
            "relative": ["최근", "지난", "이번", "올해", "작년"],
            "granularity": ["분기", "월별", "주간", "일간", "시간별"]
        }
    
    def _initialize_spatial_patterns(self) -> Dict[str, List[str]]:
        """공간적 패턴 초기화"""
        return {
            "country": ["한국", "대한민국", "전국"],
            "city": ["서울", "부산", "대구", "인천"],
            "district": ["강남구", "서초구", "마포구"],
            "granularity": ["지역별", "도시별", "구별", "동별"]
        }
    
    def _initialize_business_patterns(self) -> Dict[str, List[str]]:
        """비즈니스 패턴 초기화"""
        return {
            "analytics": ["분석", "통계", "데이터", "리포트", "성과"],
            "marketing": ["마케팅", "홍보", "광고", "프로모션"],
            "finance": ["재무", "회계", "예산", "수익", "비용"],
            "operations": ["운영", "관리", "처리", "업무", "프로세스"]
        }
    
    def _initialize_context_weights(self) -> Dict[ContextType, float]:
        """컨텍스트 가중치 초기화"""
        return {
            ContextType.TEMPORAL: 1.5,  # 1.2 -> 1.5로 증가
            ContextType.SPATIAL: 1.3,   # 1.0 -> 1.3으로 증가
            ContextType.BUSINESS: 2.0,  # 1.5 -> 2.0으로 증가
            ContextType.TECHNICAL: 1.4, # 1.1 -> 1.4로 증가
            ContextType.USER_PREFERENCE: 1.2,  # 0.8 -> 1.2로 증가
            ContextType.QUERY_PATTERN: 1.6,    # 1.3 -> 1.6으로 증가
            ContextType.DATA_SEMANTICS: 1.3    # 1.0 -> 1.3으로 증가
        }
    
    def _determine_temporal_granularity(self, user_query: str) -> str:
        """시간적 세분성 결정"""
        granularity_keywords = {
            "yearly": ["년", "연도"],
            "monthly": ["월", "월별"],
            "weekly": ["주", "주간"],
            "daily": ["일", "일간"],
            "hourly": ["시간", "시간별"]
        }
        
        for granularity, keywords in granularity_keywords.items():
            if any(keyword in user_query for keyword in keywords):
                return granularity
        
        return "general"
    
    def _determine_spatial_granularity(self, user_query: str) -> str:
        """공간적 세분성 결정"""
        granularity_keywords = {
            "country": ["전국", "국가"],
            "city": ["시", "도시"],
            "district": ["구", "군"],
            "neighborhood": ["동", "읍", "면"]
        }
        
        for granularity, keywords in granularity_keywords.items():
            if any(keyword in user_query for keyword in keywords):
                return granularity
        
        return "general"
    
    def _assess_query_complexity(self, user_query: str) -> str:
        """쿼리 복잡도 평가"""
        complexity_indicators = {
            "low": ["단순", "간단", "기본"],
            "medium": ["복합", "다중", "여러"],
            "high": ["복잡", "고급", "정교", "다단계"]
        }
        
        for level, keywords in complexity_indicators.items():
            if any(keyword in user_query for keyword in keywords):
                return level
        
        # 키워드 개수 기반 복잡도 평가
        keyword_count = len(re.findall(r'[\w가-힣]+', user_query))
        if keyword_count < 5:
            return "low"
        elif keyword_count < 10:
            return "medium"
        else:
            return "high"
    
    def _classify_query_type(self, user_query: str) -> str:
        """쿼리 타입 분류"""
        query_type_patterns = {
            "count": ["개수", "수량", "COUNT"],
            "sum": ["합계", "총계", "SUM"],
            "average": ["평균", "AVG"],
            "max": ["최대", "MAX"],
            "min": ["최소", "MIN"],
            "list": ["목록", "리스트", "조회"],
            "detail": ["상세", "세부", "정보"]
        }
        
        for query_type, keywords in query_type_patterns.items():
            if any(keyword in user_query for keyword in keywords):
                return query_type
        
        return "general"
    
    def _infer_data_types(self, user_query: str) -> List[str]:
        """데이터 타입 추론"""
        data_types = []
        
        type_indicators = {
            "numeric": ["숫자", "수치", "금액", "가격", "포인트"],
            "text": ["텍스트", "문자", "이름", "제목", "설명"],
            "date": ["날짜", "시간", "일시", "기간"],
            "boolean": ["상태", "여부", "유무", "활성"],
            "categorical": ["카테고리", "분류", "유형", "등급"]
        }
        
        for data_type, keywords in type_indicators.items():
            if any(keyword in user_query for keyword in keywords):
                data_types.append(data_type)
        
        return data_types
    
    def _update_analysis_stats(self, query_context: QueryContext):
        """분석 통계 업데이트"""
        try:
            self.analysis_stats["total_analyses"] += 1
            self.analysis_stats["successful_analyses"] += 1
            
            # 컨텍스트 타입별 통계
            for element in query_context.context_elements:
                self.analysis_stats["context_types_detected"][element.context_type] += 1
            
            # 평균 신뢰도 업데이트
            total_analyses = self.analysis_stats["total_analyses"]
            current_avg = self.analysis_stats["average_confidence"]
            new_avg = (current_avg * (total_analyses - 1) + query_context.confidence_score) / total_analyses
            self.analysis_stats["average_confidence"] = new_avg
            
        except Exception as e:
            logger.error(f"Failed to update analysis stats: {str(e)}")
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """분석 통계 반환"""
        try:
            stats = self.analysis_stats.copy()
            
            # 성공률 계산
            if stats["total_analyses"] > 0:
                stats["success_rate"] = stats["successful_analyses"] / stats["total_analyses"]
            else:
                stats["success_rate"] = 0.0
            
            # 캐시 히트율 계산
            if stats["total_analyses"] > 0:
                stats["cache_hit_rate"] = stats["cache_hit_rate"] / stats["total_analyses"]
            else:
                stats["cache_hit_rate"] = 0.0
            
            # 컨텍스트 타입별 통계를 딕셔너리로 변환
            stats["context_types_detected"] = {
                k.value: v for k, v in stats["context_types_detected"].items()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get analysis statistics: {str(e)}")
            return {"error": str(e)}
    
    def clear_cache(self):
        """캐시 정리"""
        try:
            self.context_cache.clear()
            logger.info("Context analyzer cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
    
    def clear_user_history(self, user_id: str = None):
        """사용자 히스토리 정리"""
        try:
            if user_id:
                if user_id in self.user_history:
                    del self.user_history[user_id]
                    logger.info(f"User history cleared for user: {user_id}")
            else:
                self.user_history.clear()
                logger.info("All user history cleared")
        except Exception as e:
            logger.error(f"Failed to clear user history: {str(e)}")
