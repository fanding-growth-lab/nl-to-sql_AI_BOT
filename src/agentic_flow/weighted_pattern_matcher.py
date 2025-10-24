"""
가중치 기반 패턴 매칭 시스템

이 모듈은 다양한 패턴에 가중치를 부여하여 매칭 정확도를 향상시키는 시스템을 제공합니다.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PatternMatch:
    """패턴 매칭 결과"""
    pattern_id: str
    confidence: float
    matched_text: str
    extracted_data: Dict[str, Any]
    context: Optional[str] = None


class WeightedPatternMatcher:
    """가중치 기반 패턴 매칭 클래스"""
    
    def __init__(self):
        self.patterns = self._create_weighted_patterns()
        self.context_weights = self._create_context_weights()
        self.logger = logging.getLogger(__name__)
    
    def _create_weighted_patterns(self) -> Dict[str, Dict[str, Any]]:
        """가중치 기반 패턴 정의"""
        return {
            "monthly_members": {
                "patterns": [
                    r"(\d{1,2})월\s*(신규|새로운|가입)\s*(회원|맴버|멤버)",
                    r"(신규|새로운|가입)\s*(회원|맴버|멤버)\s*(\d{1,2})월",
                    r"(\d{1,2})월\s*(회원|맴버|멤버)\s*(수|명|명수)",
                    r"(january|february|march|april|may|june|july|august|september|october|november|december)\s*(new|신규)\s*(members|회원)"
                ],
                "weight": 1.0,
                "category": "membership",
                "extractors": {
                    "month": r"(\d{1,2})|(january|february|march|april|may|june|july|august|september|october|november|december)",
                    "type": r"(신규|새로운|가입|new)",
                    "entity": r"(회원|맴버|멤버|members)"
                }
            },
            "top_creators": {
                "patterns": [
                    r"상위\s*(\d+)\s*(크리에이터|creator)",
                    r"탑\s*(\d+)\s*(크리에이터|creator)",
                    r"top\s*(\d+)\s*(크리에이터|creator)",
                    r"(\d+)\s*위\s*(크리에이터|creator)",
                    r"인기\s*(\d+)\s*(크리에이터|creator)"
                ],
                "weight": 1.2,
                "category": "ranking",
                "extractors": {
                    "top_n": r"(\d+)",
                    "entity": r"(크리에이터|creator)"
                }
            },
            "revenue_analysis": {
                "patterns": [
                    r"(\d{1,2})월\s*(매출|수익|revenue)",
                    r"(매출|수익|revenue)\s*(\d{1,2})월",
                    r"(\d{4})년\s*(\d{1,2})월\s*(매출|수익|revenue)",
                    r"(monthly|월간)\s*(revenue|매출|수익)"
                ],
                "weight": 1.1,
                "category": "revenue",
                "extractors": {
                    "month": r"(\d{1,2})",
                    "year": r"(\d{4})",
                    "metric": r"(매출|수익|revenue)"
                }
            },
            "performance_metrics": {
                "patterns": [
                    r"(\d{1,2})월\s*(성과|실적|performance)",
                    r"(성과|실적|performance)\s*(\d{1,2})월",
                    r"월별\s*(성과|실적|performance)",
                    r"(monthly|월간)\s*(performance|성과|실적)"
                ],
                "weight": 1.0,
                "category": "performance",
                "extractors": {
                    "month": r"(\d{1,2})",
                    "metric": r"(성과|실적|performance)"
                }
            },
            "content_analysis": {
                "patterns": [
                    r"(\d{1,2})월\s*(게시글|포스트|post)",
                    r"(게시글|포스트|post)\s*(\d{1,2})월",
                    r"인기\s*(\d+)\s*(게시글|포스트|post)",
                    r"top\s*(\d+)\s*(posts|게시글|포스트)"
                ],
                "weight": 0.9,
                "category": "content",
                "extractors": {
                    "month": r"(\d{1,2})",
                    "top_n": r"(\d+)",
                    "entity": r"(게시글|포스트|post)"
                }
            }
        }
    
    def _create_context_weights(self) -> Dict[str, float]:
        """컨텍스트별 가중치 정의"""
        return {
            "membership": 1.0,
            "ranking": 1.2,
            "revenue": 1.1,
            "performance": 1.0,
            "content": 0.9,
            "activity": 0.8
        }
    
    def match_patterns(self, query: str) -> List[PatternMatch]:
        """쿼리에 대한 패턴 매칭 수행"""
        query_lower = query.lower().strip()
        matches = []
        
        for pattern_id, pattern_info in self.patterns.items():
            for pattern in pattern_info["patterns"]:
                match = re.search(pattern, query_lower, re.IGNORECASE)
                if match:
                    confidence = self._calculate_confidence(
                        pattern_info, match, query_lower
                    )
                    
                    extracted_data = self._extract_data(
                        pattern_info, match, query_lower
                    )
                    
                    context = self._determine_context(
                        pattern_info, extracted_data, query_lower
                    )
                    
                    matches.append(PatternMatch(
                        pattern_id=pattern_id,
                        confidence=confidence,
                        matched_text=match.group(0),
                        extracted_data=extracted_data,
                        context=context
                    ))
        
        # 신뢰도 기준으로 정렬
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches
    
    def _calculate_confidence(self, pattern_info: Dict, match: re.Match, query: str) -> float:
        """패턴 매칭 신뢰도 계산"""
        base_weight = pattern_info["weight"]
        
        # 패턴 길이 기반 가중치
        pattern_length = len(match.group(0))
        query_length = len(query)
        length_ratio = pattern_length / query_length if query_length > 0 else 0
        
        # 컨텍스트 가중치
        context_weight = self._get_context_weight(pattern_info.get("category", ""))
        
        # 최종 신뢰도 계산
        confidence = base_weight * context_weight * (0.5 + length_ratio * 0.5)
        
        return min(confidence, 1.0)  # 최대 1.0으로 제한
    
    def _extract_data(self, pattern_info: Dict, match: re.Match, query: str) -> Dict[str, Any]:
        """패턴에서 데이터 추출"""
        extracted = {}
        
        if "extractors" in pattern_info:
            for key, extractor_pattern in pattern_info["extractors"].items():
                extractor_match = re.search(extractor_pattern, query, re.IGNORECASE)
                if extractor_match:
                    extracted[key] = extractor_match.group(1)
        
        return extracted
    
    def _determine_context(self, pattern_info: Dict, extracted_data: Dict, query: str) -> str:
        """컨텍스트 결정"""
        # 패턴 카테고리 우선
        if "category" in pattern_info:
            return pattern_info["category"]
        
        # 추출된 데이터 기반 컨텍스트 결정
        if "entity" in extracted_data:
            entity = extracted_data["entity"].lower()
            if any(keyword in entity for keyword in ["회원", "맴버", "멤버", "member"]):
                return "membership"
            elif any(keyword in entity for keyword in ["크리에이터", "creator"]):
                return "ranking"
            elif any(keyword in entity for keyword in ["게시글", "포스트", "post"]):
                return "content"
        
        return "general"
    
    def _get_context_weight(self, context: str) -> float:
        """컨텍스트별 가중치 반환"""
        return self.context_weights.get(context, 1.0)
    
    def get_best_match(self, query: str) -> Optional[PatternMatch]:
        """최고 신뢰도 매칭 결과 반환"""
        matches = self.match_patterns(query)
        return matches[0] if matches else None
    
    def get_all_matches(self, query: str, min_confidence: float = 0.5) -> List[PatternMatch]:
        """최소 신뢰도 이상의 모든 매칭 결과 반환"""
        matches = self.match_patterns(query)
        return [match for match in matches if match.confidence >= min_confidence]
    
    def generate_sql_from_match(self, match: PatternMatch) -> str:
        """매칭 결과를 SQL로 변환"""
        if not match:
            return "SELECT 1"
        
        # 패턴별 SQL 템플릿
        sql_templates = {
            "monthly_members": self._generate_monthly_members_sql,
            "top_creators": self._generate_top_creators_sql,
            "revenue_analysis": self._generate_revenue_sql,
            "performance_metrics": self._generate_performance_sql,
            "content_analysis": self._generate_content_sql
        }
        
        generator = sql_templates.get(match.pattern_id)
        if generator:
            return generator(match.extracted_data)
        
        return "SELECT 1"
    
    def _generate_monthly_members_sql(self, data: Dict[str, Any]) -> str:
        """월별 회원 SQL 생성"""
        month = data.get("month", "8")
        year = datetime.now().year
        
        return f"""
        SELECT COUNT(DISTINCT member_no) as new_members
        FROM t_member_login_log
        WHERE EXTRACT(YEAR FROM ins_datetime) = {year}
        AND EXTRACT(MONTH FROM ins_datetime) = {month}
        """
    
    def _generate_top_creators_sql(self, data: Dict[str, Any]) -> str:
        """상위 크리에이터 SQL 생성"""
        top_n = data.get("top_n", "5")
        
        return f"""
        SELECT creator_no, creator_name, COUNT(*) as member_count
        FROM t_creator c
        LEFT JOIN t_member m ON c.creator_no = m.creator_no
        GROUP BY creator_no, creator_name
        ORDER BY member_count DESC
        LIMIT {top_n}
        """
    
    def _generate_revenue_sql(self, data: Dict[str, Any]) -> str:
        """매출 분석 SQL 생성"""
        month = data.get("month", "8")
        year = data.get("year", datetime.now().year)
        
        return f"""
        SELECT SUM(price) as total_revenue, COUNT(*) as transaction_count
        FROM t_payment
        WHERE EXTRACT(YEAR FROM ins_datetime) = {year}
        AND EXTRACT(MONTH FROM ins_datetime) = {month}
        AND status = 'completed'
        """
    
    def _generate_performance_sql(self, data: Dict[str, Any]) -> str:
        """성과 분석 SQL 생성"""
        month = data.get("month", "8")
        year = datetime.now().year
        
        return f"""
        SELECT 
            COUNT(*) as total_members,
            COUNT(CASE WHEN status = 'A' THEN 1 END) as active_members,
            ROUND(COUNT(CASE WHEN status = 'A' THEN 1 END) * 100.0 / COUNT(*), 2) as active_rate
        FROM t_member
        WHERE EXTRACT(YEAR FROM ins_datetime) = {year}
        AND EXTRACT(MONTH FROM ins_datetime) = {month}
        """
    
    def _generate_content_sql(self, data: Dict[str, Any]) -> str:
        """콘텐츠 분석 SQL 생성"""
        top_n = data.get("top_n", "10")
        
        return f"""
        SELECT post_no, title, view_count, like_count
        FROM t_post
        WHERE status != 'D'
        ORDER BY view_count DESC
        LIMIT {top_n}
        """


# 테스트 함수들
def test_weighted_pattern_matching():
    """가중치 기반 패턴 매칭 테스트"""
    matcher = WeightedPatternMatcher()
    
    test_cases = [
        "8월 신규 회원수",
        "회원수가 제일 많은 Top5 크리에이터들을 뽑아줘",
        "9월 매출 현황",
        "상위 10 크리에이터",
        "이번달 성과 분석",
        "인기 게시글 20개",
        "일반적인 질문"
    ]
    
    print("=== 가중치 기반 패턴 매칭 테스트 ===")
    for query in test_cases:
        print(f"쿼리: {query}")
        
        # 최고 매칭 결과
        best_match = matcher.get_best_match(query)
        if best_match:
            print(f"  → 최고 매칭: {best_match.pattern_id}")
            print(f"  → 신뢰도: {best_match.confidence:.2f}")
            print(f"  → 매칭된 텍스트: {best_match.matched_text}")
            print(f"  → 추출된 데이터: {best_match.extracted_data}")
            print(f"  → 컨텍스트: {best_match.context}")
            
            # SQL 생성
            sql = matcher.generate_sql_from_match(best_match)
            print(f"  → 생성된 SQL: {sql.strip()}")
        else:
            print("  → 매칭 결과 없음")
        
        print()


if __name__ == "__main__":
    test_weighted_pattern_matching()

