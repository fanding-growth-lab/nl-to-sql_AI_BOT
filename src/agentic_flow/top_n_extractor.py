"""
Top N 패턴 인식 및 추출 시스템

이 모듈은 자연어 쿼리에서 'Top 5', '상위 10' 등의 패턴을 인식하고 처리하는 기능을 제공합니다.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TopNInfo:
    """추출된 Top N 정보"""
    n: int
    pattern_type: str  # 'top', 'best', 'highest', 'most', etc.
    confidence: float
    context: Optional[str] = None  # 'creators', 'members', 'revenue', etc.


class TopNExtractor:
    """Top N 패턴 추출 클래스"""
    
    def __init__(self):
        self.patterns = self._create_top_n_patterns()
        self.context_keywords = self._create_context_keywords()
        self.logger = logging.getLogger(__name__)
    
    def _create_top_n_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Top N 패턴 정의"""
        return {
            "korean_top": {
                "patterns": [
                    r"상위\s*(\d+)",
                    r"탑\s*(\d+)",
                    r"최고\s*(\d+)",
                    r"최상위\s*(\d+)",
                    r"인기\s*(\d+)",
                    r"인기순\s*(\d+)",
                    r"랭킹\s*(\d+)",
                    r"순위\s*(\d+)",
                    r"(\d+)\s*위",
                    r"(\d+)\s*등"
                ],
                "weight": 1.0
            },
            "english_top": {
                "patterns": [
                    r"top\s*(\d+)",
                    r"best\s*(\d+)",
                    r"highest\s*(\d+)",
                    r"most\s*(\d+)",
                    r"top\s*(\d+)\s*of",
                    r"(\d+)\s*top",
                    r"(\d+)\s*best"
                ],
                "weight": 1.0
            },
            "korean_ranking": {
                "patterns": [
                    r"(\d+)\s*개",
                    r"(\d+)\s*명",
                    r"(\d+)\s*건",
                    r"(\d+)\s*개씩",
                    r"(\d+)\s*명씩"
                ],
                "weight": 0.8
            },
            "english_ranking": {
                "patterns": [
                    r"(\d+)\s*items",
                    r"(\d+)\s*records",
                    r"(\d+)\s*entries",
                    r"(\d+)\s*results"
                ],
                "weight": 0.8
            }
        }
    
    def _create_context_keywords(self) -> Dict[str, List[str]]:
        """컨텍스트 키워드 정의"""
        return {
            "creators": [
                "크리에이터", "creator", "작가", "아티스트", "artist", "창작자",
                "크리에이터", "크리에이션", "creation"
            ],
            "members": [
                "회원", "member", "사용자", "user", "맴버", "멤버",
                "가입자", "subscriber", "구독자"
            ],
            "revenue": [
                "매출", "revenue", "수익", "income", "판매", "sales",
                "금액", "amount", "가격", "price"
            ],
            "content": [
                "콘텐츠", "content", "게시글", "post", "글", "article",
                "포스트", "게시물", "작품", "work"
            ],
            "performance": [
                "성과", "performance", "실적", "achievement", "결과", "result",
                "지표", "metric", "통계", "statistics"
            ],
            "activity": [
                "활동", "activity", "참여", "participation", "참여도", "engagement",
                "방문", "visit", "조회", "view"
            ]
        }
    
    def extract_top_n_info(self, query: str) -> Optional[TopNInfo]:
        """쿼리에서 Top N 정보 추출"""
        query_lower = query.lower().strip()
        
        # 1. 패턴 매칭으로 Top N 추출
        top_n_match = self._match_top_n_patterns(query_lower)
        if not top_n_match:
            return None
        
        n, pattern_type, confidence = top_n_match
        
        # 2. 컨텍스트 추출
        context = self._extract_context(query_lower)
        
        # 3. 유효성 검증
        if not self._validate_top_n(n):
            return None
        
        return TopNInfo(
            n=n,
            pattern_type=pattern_type,
            confidence=confidence,
            context=context
        )
    
    def _match_top_n_patterns(self, query: str) -> Optional[Tuple[int, str, float]]:
        """Top N 패턴 매칭"""
        best_match = None
        best_confidence = 0.0
        
        for pattern_type, pattern_info in self.patterns.items():
            for pattern in pattern_info["patterns"]:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    try:
                        n = int(match.group(1))
                        confidence = pattern_info["weight"]
                        
                        if confidence > best_confidence:
                            best_match = (n, pattern_type, confidence)
                            best_confidence = confidence
                    except (ValueError, IndexError):
                        continue
        
        return best_match
    
    def _extract_context(self, query: str) -> Optional[str]:
        """컨텍스트 추출"""
        for context_type, keywords in self.context_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    return context_type
        
        return None
    
    def _validate_top_n(self, n: int) -> bool:
        """Top N 값 유효성 검증"""
        # 최소값: 1, 최대값: 100 (시스템 부하 방지)
        return 1 <= n <= 100
    
    def get_sql_limit_clause(self, top_n_info: TopNInfo) -> str:
        """Top N 정보를 SQL LIMIT 절로 변환"""
        return f"LIMIT {top_n_info.n}"
    
    def get_sql_order_clause(self, top_n_info: TopNInfo, context: str = None) -> str:
        """컨텍스트에 따른 ORDER BY 절 생성"""
        if not context:
            context = top_n_info.context
        
        order_mapping = {
            "creators": "ORDER BY creator_score DESC",
            "members": "ORDER BY member_count DESC",
            "revenue": "ORDER BY revenue DESC",
            "content": "ORDER BY view_count DESC",
            "performance": "ORDER BY performance_score DESC",
            "activity": "ORDER BY activity_count DESC"
        }
        
        return order_mapping.get(context, "ORDER BY score DESC")
    
    def get_human_readable_top_n(self, top_n_info: TopNInfo) -> str:
        """Top N 정보를 사람이 읽기 쉬운 형태로 변환"""
        context_map = {
            "creators": "크리에이터",
            "members": "회원",
            "revenue": "매출",
            "content": "콘텐츠",
            "performance": "성과",
            "activity": "활동"
        }
        
        context_name = context_map.get(top_n_info.context, "항목")
        return f"상위 {top_n_info.n}개 {context_name}"
    
    def is_top_n_query(self, query: str) -> bool:
        """Top N 쿼리인지 판단"""
        top_n_info = self.extract_top_n_info(query)
        return top_n_info is not None and top_n_info.confidence > 0.5


# 테스트 함수들
def test_top_n_extraction():
    """Top N 추출 테스트"""
    extractor = TopNExtractor()
    
    test_cases = [
        "상위 5 크리에이터",
        "top 10 members",
        "최고 3 매출 상품",
        "인기 20 게시글",
        "탑 15 크리에이터들을 뽑아줘",
        "회원수가 제일 많은 Top5 크리에이터들을 뽑아줘",
        "상위 100개 결과",
        "일반 쿼리 (Top N 없음)"
    ]
    
    print("=== Top N 추출 테스트 ===")
    for query in test_cases:
        top_n_info = extractor.extract_top_n_info(query)
        print(f"쿼리: {query}")
        if top_n_info:
            print(f"  → Top N: {top_n_info.n}")
            print(f"  → 패턴 타입: {top_n_info.pattern_type}")
            print(f"  → 컨텍스트: {top_n_info.context}")
            print(f"  → 신뢰도: {top_n_info.confidence}")
            print(f"  → SQL LIMIT: {extractor.get_sql_limit_clause(top_n_info)}")
            print(f"  → SQL ORDER: {extractor.get_sql_order_clause(top_n_info)}")
            print(f"  → 사람이 읽기 쉬운 형태: {extractor.get_human_readable_top_n(top_n_info)}")
        else:
            print("  → Top N 패턴 없음")
        print()


if __name__ == "__main__":
    test_top_n_extraction()

