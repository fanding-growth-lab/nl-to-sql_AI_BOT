"""
통합 RAG 개선 시스템

이 모듈은 개선된 날짜/월 매칭, Top N 패턴 인식, 가중치 기반 매칭을 통합하여
RAG 매핑 정확도를 향상시키는 시스템을 제공합니다.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime

from date_month_extractor import DateMonthExtractor, DateInfo
from top_n_extractor import TopNExtractor, TopNInfo
from weighted_pattern_matcher import WeightedPatternMatcher, PatternMatch

logger = logging.getLogger(__name__)


@dataclass
class ImprovedQueryAnalysis:
    """개선된 쿼리 분석 결과"""
    original_query: str
    date_info: Optional[DateInfo]
    top_n_info: Optional[TopNInfo]
    pattern_match: Optional[PatternMatch]
    confidence: float
    sql_template: str
    clarification_needed: bool = False
    clarification_question: Optional[str] = None


class IntegratedRAGImprover:
    """통합 RAG 개선 시스템"""
    
    def __init__(self):
        self.date_extractor = DateMonthExtractor()
        self.top_n_extractor = TopNExtractor()
        self.pattern_matcher = WeightedPatternMatcher()
        self.logger = logging.getLogger(__name__)
    
    def analyze_query(self, query: str) -> ImprovedQueryAnalysis:
        """쿼리 종합 분석"""
        # 1. 날짜/월 정보 추출
        date_info = self.date_extractor.extract_date_info(query)
        
        # 2. Top N 정보 추출
        top_n_info = self.top_n_extractor.extract_top_n_info(query)
        
        # 3. 패턴 매칭
        pattern_match = self.pattern_matcher.get_best_match(query)
        
        # 4. 신뢰도 계산
        confidence = self._calculate_overall_confidence(date_info, top_n_info, pattern_match)
        
        # 5. SQL 템플릿 생성
        sql_template = self._generate_sql_template(date_info, top_n_info, pattern_match, query)
        
        # 6. 명확화 필요 여부 판단
        clarification_needed, clarification_question = self._check_clarification_needed(
            date_info, top_n_info, pattern_match, query
        )
        
        return ImprovedQueryAnalysis(
            original_query=query,
            date_info=date_info,
            top_n_info=top_n_info,
            pattern_match=pattern_match,
            confidence=confidence,
            sql_template=sql_template,
            clarification_needed=clarification_needed,
            clarification_question=clarification_question
        )
    
    def _calculate_overall_confidence(self, date_info: Optional[DateInfo], 
                                    top_n_info: Optional[TopNInfo], 
                                    pattern_match: Optional[PatternMatch]) -> float:
        """전체 신뢰도 계산"""
        confidences = []
        
        if date_info and date_info.confidence > 0:
            confidences.append(date_info.confidence)
        
        if top_n_info and top_n_info.confidence > 0:
            confidences.append(top_n_info.confidence)
        
        if pattern_match and pattern_match.confidence > 0:
            confidences.append(pattern_match.confidence)
        
        if not confidences:
            return 0.0
        
        # 가중 평균 계산 (패턴 매칭에 더 높은 가중치)
        weights = [0.3, 0.2, 0.5]  # date, top_n, pattern 순서
        weighted_sum = sum(conf * weight for conf, weight in zip(confidences, weights[:len(confidences)]))
        weight_sum = sum(weights[:len(confidences)])
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    def _generate_sql_template(self, date_info: Optional[DateInfo], 
                             top_n_info: Optional[TopNInfo], 
                             pattern_match: Optional[PatternMatch], 
                             query: str) -> str:
        """SQL 템플릿 생성"""
        # 패턴 매칭이 있으면 우선 사용
        if pattern_match and pattern_match.confidence > 0.7:
            return self.pattern_matcher.generate_sql_from_match(pattern_match)
        
        # 기본 SQL 템플릿 생성
        base_sql = "SELECT * FROM t_member WHERE 1=1"
        
        # 날짜 조건 추가
        if date_info and date_info.month:
            date_condition = self.date_extractor.get_sql_date_condition(date_info)
            if date_condition != "1=1":
                base_sql += f" AND {date_condition}"
        
        # Top N 조건 추가
        if top_n_info:
            base_sql += f" {self.top_n_extractor.get_sql_limit_clause(top_n_info)}"
        
        return base_sql
    
    def _check_clarification_needed(self, date_info: Optional[DateInfo], 
                                  top_n_info: Optional[TopNInfo], 
                                  pattern_match: Optional[PatternMatch], 
                                  query: str) -> Tuple[bool, Optional[str]]:
        """명확화 필요 여부 판단"""
        # 신뢰도가 낮은 경우
        if not any([
            date_info and date_info.confidence > 0.5,
            top_n_info and top_n_info.confidence > 0.5,
            pattern_match and pattern_match.confidence > 0.5
        ]):
            return True, "구체적인 정보가 필요합니다. 예를 들어, '8월 신규 회원수' 또는 '상위 5 크리에이터'와 같이 명확히 말씀해주세요."
        
        # 날짜 정보가 애매한 경우
        if not date_info or date_info.confidence < 0.3:
            if any(keyword in query.lower() for keyword in ["회원", "멤버", "맴버", "매출", "수익"]):
                return True, "어떤 기간의 데이터를 원하시나요? (예: 8월, 9월, 이번달, 지난달 등)"
        
        # Top N 정보가 애매한 경우
        if "크리에이터" in query or "creator" in query.lower():
            if not top_n_info or top_n_info.confidence < 0.3:
                return True, "몇 개의 크리에이터를 보고 싶으신가요? (예: 상위 5개, Top 10개 등)"
        
        return False, None
    
    def get_improved_sql_templates(self) -> Dict[str, str]:
        """개선된 SQL 템플릿 반환"""
        return {
            "monthly_new_members": """
            SELECT COUNT(DISTINCT member_no) as new_members
            FROM t_member_login_log
            WHERE {date_condition}
            """,
            "top_creators_by_members": """
            SELECT c.creator_no, c.creator_name, COUNT(m.member_no) as member_count
            FROM t_creator c
            LEFT JOIN t_member m ON c.creator_no = m.creator_no
            WHERE {date_condition}
            GROUP BY c.creator_no, c.creator_name
            ORDER BY member_count DESC
            LIMIT {top_n}
            """,
            "monthly_revenue": """
            SELECT SUM(price) as total_revenue, COUNT(*) as transaction_count
            FROM t_payment
            WHERE {date_condition}
            AND status = 'completed'
            """,
            "top_content": """
            SELECT post_no, title, view_count, like_count
            FROM t_post
            WHERE {date_condition}
            AND status != 'D'
            ORDER BY view_count DESC
            LIMIT {top_n}
            """
        }
    
    def update_rag_templates(self, query: str, analysis: ImprovedQueryAnalysis) -> Dict[str, Any]:
        """RAG 템플릿 업데이트"""
        # 기존 템플릿에 개선된 정보 반영
        updated_template = {
            "query": query,
            "date_condition": self.date_extractor.get_sql_date_condition(analysis.date_info) if analysis.date_info else "1=1",
            "top_n": analysis.top_n_info.n if analysis.top_n_info else 10,
            "confidence": analysis.confidence,
            "pattern_id": analysis.pattern_match.pattern_id if analysis.pattern_match else None,
            "context": analysis.pattern_match.context if analysis.pattern_match else None
        }
        
        return updated_template


# 통합 테스트 함수
def test_integrated_rag_improvement():
    """통합 RAG 개선 테스트"""
    improver = IntegratedRAGImprover()
    
    test_cases = [
        "8월 신규 회원수",
        "회원수가 제일 많은 Top5 크리에이터들을 뽑아줘",
        "9월 매출 현황",
        "상위 10 크리에이터",
        "이번달 성과 분석",
        "지난달 활성 사용자",
        "August new members",
        "top 3 creators",
        "애매한 질문"
    ]
    
    print("=== 통합 RAG 개선 테스트 ===")
    for query in test_cases:
        print(f"쿼리: {query}")
        
        analysis = improver.analyze_query(query)
        
        print(f"  → 전체 신뢰도: {analysis.confidence:.2f}")
        
        if analysis.date_info:
            print(f"  → 날짜 정보: {improver.date_extractor.get_human_readable_date(analysis.date_info)}")
        
        if analysis.top_n_info:
            print(f"  → Top N 정보: {improver.top_n_extractor.get_human_readable_top_n(analysis.top_n_info)}")
        
        if analysis.pattern_match:
            print(f"  → 패턴 매칭: {analysis.pattern_match.pattern_id} (신뢰도: {analysis.pattern_match.confidence:.2f})")
        
        if analysis.clarification_needed:
            print(f"  → 명확화 필요: {analysis.clarification_question}")
        else:
            print(f"  → 생성된 SQL: {analysis.sql_template.strip()}")
        
        print()


if __name__ == "__main__":
    test_integrated_rag_improvement()

