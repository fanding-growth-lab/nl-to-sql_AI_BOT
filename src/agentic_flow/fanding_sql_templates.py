"""
Fanding Data Report System SQL Templates

This module contains SQL templates for various Fanding Data Report analysis features.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
import re

from core.db import get_cached_db_schema

logger = logging.getLogger(__name__)


class FandingAnalysisType(Enum):
    """Fanding Data Report analysis types"""
    MEMBERSHIP_DATA = "membership_data"
    PERFORMANCE_REPORT = "performance_report"
    CONTENT_PERFORMANCE = "content_performance"
    ADVANCED_ANALYSIS = "advanced_analysis"


@dataclass
class SQLTemplate:
    """SQL template structure"""
    name: str
    description: str
    sql_template: str
    parameters: List[str]
    analysis_type: FandingAnalysisType
    keywords: Optional[List[str]] = None  # 키워드 기반 매칭을 위한 필드 추가


class FandingSQLTemplates:
    """Fanding Data Report SQL Templates"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.logger = logging.getLogger(__name__)
        # 중앙화된 스키마 정보 로드
        self.db_schema = get_cached_db_schema()
        
        # 템플릿 검증 실행 (중요: 스키마 동기화 확인)
        self._validate_templates()
    
    def _initialize_templates(self) -> Dict[str, SQLTemplate]:
        """Initialize all Fanding SQL templates"""
        templates = {}
        
        # 멤버십 데이터 분석 템플릿
        templates.update(self._get_membership_templates())
        
        # 성과 리포트 템플릿
        templates.update(self._get_performance_templates())
        
        # 콘텐츠 성과 분석 템플릿
        templates.update(self._get_content_templates())
        
        # 고급 분석 템플릿
        templates.update(self._get_advanced_templates())
        
        return templates
    
    def _get_membership_templates(self) -> Dict[str, SQLTemplate]:
        """멤버십 데이터 분석 템플릿"""
        return {
            "member_count": SQLTemplate(
                name="회원 수 조회",
                description="전체 회원 수 조회 (상태별 필터링 가능)",
                sql_template="""
                SELECT COUNT(*) as total_members
                FROM t_member_info
                WHERE 1=1
                """,
                parameters=[],
                analysis_type=FandingAnalysisType.MEMBERSHIP_DATA,
                keywords=["회원", "멤버", "맴버", "회원수", "멤버수", "맴버수", "전체", "모든", "사용자", "가입자"]
                   ),
                   "active_member_count": SQLTemplate(
                       name="활성 회원 수",
                       description="활성 상태 회원 수 조회 (최근 로그인 기준)",
                       sql_template="""
                       SELECT COUNT(*) as active_members 
                       FROM t_member_info 
                       WHERE login_datetime >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                       """,
                       parameters=[],
                       analysis_type=FandingAnalysisType.MEMBERSHIP_DATA,
                       keywords=["활성", "회원", "멤버", "맴버", "활성회원", "활성멤버", "로그인", "최근"]
                   ),
                   "new_members_this_month": SQLTemplate(
                       name="이번 달 신규 회원",
                       description="이번 달 신규 가입 회원 수",
                       sql_template="""
                       SELECT COUNT(*) as new_members_this_month 
                       FROM t_member_info 
                       WHERE DATE_FORMAT(ins_datetime, '%Y-%m') = DATE_FORMAT(NOW(), '%Y-%m')
                       """,
                       parameters=[],
                       analysis_type=FandingAnalysisType.MEMBERSHIP_DATA,
                       keywords=["신규", "회원", "멤버", "맴버", "신규회원", "신규멤버", "가입", "현황", "이번달", "이번"]
                   ),
                   
                   "new_members_specific_month": SQLTemplate(
                       name="{month}월 신규 회원",
                       description="{month}월 신규 가입 회원 수",
                       sql_template="""
                       SELECT COUNT(*) as new_members_{month}month 
                       FROM t_member_info 
                       WHERE DATE_FORMAT(ins_datetime, '%Y-%m') = CONCAT(YEAR(NOW()), '-{month:02d}')
                       """,
                       parameters=["month"],
                       analysis_type=FandingAnalysisType.MEMBERSHIP_DATA,
                       keywords=["신규", "회원", "멤버", "맴버", "신규회원", "신규멤버", "가입", "현황", "1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월"]
                   ),
            
            "monthly_member_trend": SQLTemplate(
                name="월별 회원 수 추이",
                description="월별 회원 수 변화 추이",
                sql_template="""
                SELECT 
                    DATE_FORMAT(ins_datetime, '%Y-%m') as month,
                    COUNT(*) as total_members,
                    COUNT(CASE WHEN login_datetime >= DATE_SUB(NOW(), INTERVAL 30 DAY) THEN 1 END) as active_members,
                    ROUND(COUNT(CASE WHEN login_datetime >= DATE_SUB(NOW(), INTERVAL 30 DAY) THEN 1 END) * 100.0 / COUNT(*), 2) as active_rate_percent
                FROM t_member_info
                WHERE ins_datetime >= DATE_SUB(NOW(), INTERVAL 12 MONTH)
                GROUP BY DATE_FORMAT(ins_datetime, '%Y-%m')
                ORDER BY month
                """,
                parameters=[],
                analysis_type=FandingAnalysisType.MEMBERSHIP_DATA
            ),
            
            "member_retention": SQLTemplate(
                name="회원 리텐션 분석",
                description="회원 유지율 분석 (팬딩 멤버십 기반)",
                sql_template="""
                SELECT 
                    DATE_FORMAT(f.ins_datetime, '%Y-%m') as cohort_month,
                    COUNT(DISTINCT f.member_no) as cohort_size,
                    COUNT(DISTINCT CASE WHEN f.fanding_status = 'T' THEN f.member_no END) as active_members,
                    COUNT(DISTINCT CASE WHEN f.fanding_status = 'F' THEN f.member_no END) as inactive_members,
                    ROUND(COUNT(DISTINCT CASE WHEN f.fanding_status = 'T' THEN f.member_no END) * 100.0 / COUNT(DISTINCT f.member_no), 2) as retention_rate
                FROM t_fanding f
                WHERE f.ins_datetime >= DATE_SUB(NOW(), INTERVAL 12 MONTH)
                GROUP BY DATE_FORMAT(f.ins_datetime, '%Y-%m')
                ORDER BY cohort_month
                """,
                parameters=[],
                analysis_type=FandingAnalysisType.MEMBERSHIP_DATA
            ),
            
            
            "subscription_duration_distribution": SQLTemplate(
                name="멤버십 구독 기간 분포",
                description="멤버십 구독 기간별 분포 (팬딩 로그 기반)",
                sql_template="""
                SELECT 
                    CASE 
                        WHEN DATEDIFF(end_date, start_date) <= 30 THEN '1개월 이하'
                        WHEN DATEDIFF(end_date, start_date) <= 90 THEN '1-3개월'
                        WHEN DATEDIFF(end_date, start_date) <= 180 THEN '3-6개월'
                        WHEN DATEDIFF(end_date, start_date) <= 365 THEN '6-12개월'
                        ELSE '12개월 이상'
                    END as duration_range,
                    COUNT(*) as subscription_count,
                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM t_fanding_log), 2) as percentage
                FROM t_fanding_log
                WHERE status = 'T'
                GROUP BY duration_range
                ORDER BY subscription_count DESC
                """,
                parameters=[],
                analysis_type=FandingAnalysisType.MEMBERSHIP_DATA
            )
        }
    
    def _get_performance_templates(self) -> Dict[str, SQLTemplate]:
        """성과 리포트 템플릿"""
        return {
            "monthly_revenue": SQLTemplate(
                name="월간 매출 분석",
                description="월간 매출 및 성장률 분석 (결제 정보 기반)",
                sql_template="""
                SELECT 
                    DATE_FORMAT(pay_datetime, '%Y-%m') as month,
                    SUM(CASE WHEN currency_no = 1 THEN remain_price ELSE remain_price * 1360 END) as total_revenue_krw,
                    SUM(CASE WHEN currency_no = 2 THEN remain_price ELSE remain_price / 1360 END) as total_revenue_usd,
                    COUNT(*) as transaction_count,
                    AVG(CASE WHEN currency_no = 1 THEN remain_price ELSE remain_price * 1360 END) as avg_transaction_value_krw,
                    0 as prev_month_revenue,
                    0 as growth_rate_percent
                FROM t_payment
                WHERE status IN ('T', 'P')
                GROUP BY DATE_FORMAT(pay_datetime, '%Y-%m')
                ORDER BY month
                """,
                parameters=[],
                analysis_type=FandingAnalysisType.PERFORMANCE_REPORT,
                keywords=["월간", "매출", "수익", "revenue", "income", "월별", "매출분석", "수익분석", "결제", "payment"]
            ),
            
            "visitor_trend": SQLTemplate(
                name="방문자 수 추이",
                description="일별 방문자 수 추이 분석 (로그인 기준)",
                sql_template="""
                SELECT 
                    DATE(login_datetime) as date,
                    COUNT(DISTINCT member_no) as unique_visitors,
                    COUNT(*) as total_logins,
                    AVG(1) as avg_session_duration
                FROM t_member_info
                WHERE login_datetime >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                AND login_datetime IS NOT NULL
                GROUP BY DATE(login_datetime)
                ORDER BY date
                """,
                parameters=[],
                analysis_type=FandingAnalysisType.PERFORMANCE_REPORT,
                keywords=["방문자", "visitor", "추이", "trend", "일별", "daily", "로그인", "login", "방문", "visit"]
            ),
            
            "revenue_growth_analysis": SQLTemplate(
                name="매출 성장률 분석",
                description="전월 대비 매출 성장률 분석 (결제 정보 기반)",
                sql_template="""
                SELECT 
                    DATE_FORMAT(pay_datetime, '%Y-%m') as month,
                    SUM(CASE WHEN currency_no = 1 THEN remain_price ELSE remain_price * 1360 END) as revenue_krw,
                    0 as prev_month_revenue,
                    0 as growth_rate_percent
                FROM t_payment
                WHERE pay_datetime >= DATE_SUB(NOW(), INTERVAL 6 MONTH)
                AND status IN ('T', 'P')
                GROUP BY DATE_FORMAT(pay_datetime, '%Y-%m')
                ORDER BY month
                """,
                parameters=[],
                analysis_type=FandingAnalysisType.PERFORMANCE_REPORT,
                keywords=["매출", "성장률", "growth", "rate", "증가", "increase", "전월", "previous", "비교", "comparison"]
            )
        }
    
    def _get_content_templates(self) -> Dict[str, SQLTemplate]:
        """콘텐츠 성과 분석 템플릿"""
        return {
            "top_posts": SQLTemplate(
                name="인기 포스트 TOP{top_k}",
                description="조회수 기준 인기 포스트 상위 {top_k}개",
                sql_template="""
                SELECT 
                    c.no as post_id,
                    c.content_type,
                    c.ins_datetime as post_date,
                    c.status,
                    COUNT(cr.no) as reply_count,
                    c.creator_no,
                    c.member_no
                FROM t_community c
                LEFT JOIN t_community_reply cr ON c.no = cr.community_no
                WHERE c.ins_datetime >= DATE_SUB(NOW(), INTERVAL {days} DAY)
                AND c.status = 'public'
                GROUP BY c.no, c.content_type, c.ins_datetime, c.status, c.creator_no, c.member_no
                ORDER BY reply_count DESC
                LIMIT {top_k}
                """,
                parameters=["top_k", "days"],
                analysis_type=FandingAnalysisType.CONTENT_PERFORMANCE,
                keywords=["인기", "포스트", "post", "top", "상위", "조회수", "view", "댓글", "reply", "커뮤니티", "community"]
            ),
            
            "content_engagement_analysis": SQLTemplate(
                name="콘텐츠 참여도 분석",
                description="포스트별 참여도 지표 분석 (커뮤니티 기반)",
                sql_template="""
                SELECT 
                    DATE_FORMAT(c.ins_datetime, '%Y-%m-%d') as post_date,
                    COUNT(c.no) as posts_published,
                    COUNT(cr.no) as total_replies,
                    COUNT(DISTINCT c.creator_no) as active_creators,
                    COUNT(DISTINCT c.member_no) as active_members,
                    ROUND(COUNT(cr.no) / NULLIF(COUNT(c.no), 0), 2) as avg_replies_per_post
                FROM t_community c
                LEFT JOIN t_community_reply cr ON c.no = cr.community_no
                WHERE c.ins_datetime >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                AND c.status = 'public'
                GROUP BY DATE_FORMAT(c.ins_datetime, '%Y-%m-%d')
                ORDER BY post_date
                """,
                parameters=[],
                analysis_type=FandingAnalysisType.CONTENT_PERFORMANCE,
                keywords=["콘텐츠", "content", "참여도", "engagement", "포스트", "post", "댓글", "reply", "크리에이터", "creator"]
            ),
            
            "post_visitor_correlation": SQLTemplate(
                name="포스트 발행과 방문자 상관관계",
                description="포스트 발행과 방문자 수 상관관계 분석",
                sql_template="""
                WITH daily_metrics AS (
                    SELECT 
                        DATE(c.ins_datetime) as date,
                        COUNT(c.no) as posts_count,
                        COUNT(DISTINCT mi.member_no) as unique_visitors
                    FROM t_community c
                    LEFT JOIN t_member_info mi ON DATE(c.ins_datetime) = DATE(mi.login_datetime)
                    WHERE c.ins_datetime >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                    AND c.status = 'public'
                    GROUP BY DATE(c.ins_datetime)
                )
                SELECT 
                    date,
                    posts_count,
                    unique_visitors,
                    ROUND(unique_visitors / NULLIF(posts_count, 0), 2) as visitors_per_post
                FROM daily_metrics
                ORDER BY date
                """,
                parameters=[],
                analysis_type=FandingAnalysisType.CONTENT_PERFORMANCE,
                keywords=["포스트", "post", "발행", "publish", "방문자", "visitor", "상관관계", "correlation", "관계", "relation"]
            )
        }
    
    def _get_advanced_templates(self) -> Dict[str, SQLTemplate]:
        """고급 분석 템플릿"""
        return {
            "customer_lifetime_analysis": SQLTemplate(
                name="고객 평균 수명 분석",
                description="고객 평균 수명 및 가치 분석 (팬딩 멤버십 기반)",
                sql_template="""
                SELECT 
                    AVG(DATEDIFF(NOW(), mi.ins_datetime)) as avg_customer_age_days,
                    AVG(DATEDIFF(fl.end_date, fl.start_date)) as avg_subscription_duration_days,
                    AVG(fl.price) as avg_subscription_value,
                    COUNT(DISTINCT CASE WHEN f.fanding_status = 'T' THEN f.member_no END) as active_subscribers,
                    COUNT(DISTINCT CASE WHEN f.fanding_status = 'F' THEN f.member_no END) as cancelled_subscribers,
                    ROUND(
                        COUNT(DISTINCT CASE WHEN f.fanding_status = 'T' THEN f.member_no END) * 100.0 / 
                        COUNT(DISTINCT f.member_no), 2
                    ) as retention_rate
                FROM t_fanding f
                LEFT JOIN t_member_info mi ON f.member_no = mi.member_no
                LEFT JOIN t_fanding_log fl ON f.No = fl.fanding_no
                WHERE 1=1
                """,
                parameters=[],
                analysis_type=FandingAnalysisType.ADVANCED_ANALYSIS,
                keywords=["고객", "customer", "평균", "average", "수명", "lifetime", "ltv", "가치", "value", "분석", "analysis", "멤버십", "membership"]
            ),
            
            "cancellation_analysis": SQLTemplate(
                name="멤버십 중단 예약 비율",
                description="멤버십 중단 예약 현황 분석 (팬딩 상태 기반)",
                sql_template="""
                SELECT 
                    '전체' as subscription_plan,
                    COUNT(DISTINCT f.member_no) as total_subscribers,
                    COUNT(DISTINCT CASE WHEN f.fanding_status = 'F' THEN f.member_no END) as scheduled_cancellations,
                    ROUND(
                        COUNT(DISTINCT CASE WHEN f.fanding_status = 'F' THEN f.member_no END) * 100.0 / 
                        COUNT(DISTINCT f.member_no), 2
                    ) as cancellation_rate_percent
                FROM t_fanding f
                WHERE f.fanding_status = 'T'
                GROUP BY '전체'
                ORDER BY cancellation_rate_percent DESC
                """,
                parameters=[],
                analysis_type=FandingAnalysisType.ADVANCED_ANALYSIS,
                keywords=["멤버십", "membership", "중단", "cancellation", "예약", "scheduled", "비율", "rate", "취소", "cancel"]
            ),
            
            "monthly_performance_comparison": SQLTemplate(
                name="월별 성과 비교",
                description="월별 성과 지표 비교 분석 (멤버십 및 매출 기반)",
                sql_template="""
                SELECT 
                    DATE_FORMAT(f.ins_datetime, '%Y-%m') as month,
                    COUNT(DISTINCT f.member_no) as new_members,
                    COUNT(DISTINCT CASE WHEN f.fanding_status = 'T' THEN f.member_no END) as active_members,
                    SUM(CASE WHEN p.currency_no = 1 THEN p.remain_price ELSE p.remain_price * 1360 END) as total_revenue_krw,
                    0 as prev_new_members,
                    0 as prev_active_members,
                    0 as prev_revenue,
                    0 as member_growth_rate,
                    0 as revenue_growth_rate
                FROM t_fanding f
                LEFT JOIN t_payment p ON f.member_no = p.member_no 
                    AND DATE_FORMAT(f.ins_datetime, '%Y-%m') = DATE_FORMAT(p.pay_datetime, '%Y-%m')
                WHERE f.ins_datetime >= DATE_SUB(NOW(), INTERVAL 12 MONTH)
                GROUP BY DATE_FORMAT(f.ins_datetime, '%Y-%m')
                ORDER BY month
                """,
                parameters=[],
                analysis_type=FandingAnalysisType.ADVANCED_ANALYSIS,
                keywords=["월별", "monthly", "성과", "performance", "비교", "comparison", "지표", "metrics", "분석", "analysis"]
            ),
            
            "creator_department_analysis": SQLTemplate(
                name="크리에이터 부서별 분석",
                description="크리에이터 부서별 성과 분석",
                sql_template="""
                SELECT 
                    cd.name as department_name,
                    COUNT(DISTINCT c.no) as total_creators,
                    COUNT(DISTINCT f.member_no) as total_subscribers,
                    COUNT(DISTINCT CASE WHEN f.fanding_status = 'T' THEN f.member_no END) as active_subscribers,
                    SUM(CASE WHEN p.currency_no = 1 THEN p.remain_price ELSE p.remain_price * 1360 END) as total_revenue_krw
                FROM t_creator_department cd
                LEFT JOIN t_creator_department_mapping cdm ON cd.no = cdm.department_no
                LEFT JOIN t_creator c ON cdm.creator_no = c.no
                LEFT JOIN t_fanding f ON c.no = f.creator_no
                LEFT JOIN t_payment p ON f.member_no = p.member_no
                WHERE cd.no BETWEEN 3 AND 8  -- 엔터 그룹
                GROUP BY cd.name
                ORDER BY total_revenue_krw DESC
                """,
                parameters=[],
                analysis_type=FandingAnalysisType.ADVANCED_ANALYSIS,
                keywords=["크리에이터", "creator", "부서", "department", "분석", "analysis", "성과", "performance", "부서별"]
            ),
            
            "follow_analysis": SQLTemplate(
                name="팔로우 분석",
                description="크리에이터 팔로우 현황 분석",
                sql_template="""
                SELECT 
                    c.no as creator_no,
                    COUNT(DISTINCT f.member_no) as total_followers,
                    COUNT(DISTINCT fan.member_no) as paying_subscribers,
                    ROUND(COUNT(DISTINCT fan.member_no) * 100.0 / COUNT(DISTINCT f.member_no), 2) as conversion_rate
                FROM t_creator c
                LEFT JOIN t_follow f ON c.no = f.creator_no
                LEFT JOIN t_fanding fan ON c.no = fan.creator_no AND fan.fanding_status = 'T'
                GROUP BY c.no
                ORDER BY total_followers DESC
                LIMIT 10
                """,
                parameters=[],
                analysis_type=FandingAnalysisType.ADVANCED_ANALYSIS,
                keywords=["팔로우", "follow", "분석", "analysis", "크리에이터", "creator", "현황", "status", "구독자", "subscriber"]
            ),
            
            "review_analysis": SQLTemplate(
                name="리뷰 분석",
                description="크리에이터 리뷰 현황 분석",
                sql_template="""
                SELECT 
                    r.creator_no,
                    COUNT(r.no) as total_reviews,
                    COUNT(DISTINCT r.member_no) as unique_reviewers,
                    COUNT(DISTINCT r.fanding_log_no) as reviewed_subscriptions
                FROM t_review r
                WHERE r.ins_datetime >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                GROUP BY r.creator_no
                ORDER BY total_reviews DESC
                LIMIT 10
                """,
                parameters=[],
                analysis_type=FandingAnalysisType.ADVANCED_ANALYSIS,
                keywords=["리뷰", "review", "분석", "analysis", "크리에이터", "creator", "현황", "status", "평가", "rating"]
            ),
            
            "cancellation_survey_analysis": SQLTemplate(
                name="멤버십 취소 설문 분석",
                description="멤버십 취소 사유 분석",
                sql_template="""
                SELECT 
                    CASE ssr.stop_survey_no
                        WHEN 1 THEN '크리에이터 활동 부족'
                        WHEN 2 THEN '리워드 불만족'
                        WHEN 3 THEN '목표 달성'
                        WHEN 4 THEN '가격 부담'
                        WHEN 5 THEN '서비스 불만족'
                        ELSE '기타'
                    END as cancellation_reason,
                    COUNT(*) as response_count,
                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM t_membership_stop_survey_response), 2) as percentage
                FROM t_membership_stop_survey_response ssr
                WHERE ssr.del_datetime IS NULL
                GROUP BY ssr.stop_survey_no
                ORDER BY response_count DESC
                """,
                parameters=[],
                analysis_type=FandingAnalysisType.ADVANCED_ANALYSIS,
                keywords=["멤버십", "membership", "취소", "cancellation", "설문", "survey", "분석", "analysis", "사유", "reason"]
            )
        }
    
    def get_template(self, template_name: str) -> Optional[SQLTemplate]:
        """Get SQL template by name"""
        return self.templates.get(template_name)
    
    def get_parameterized_template(self, template_name: str, parameters: Dict[str, Any]) -> Optional[SQLTemplate]:
        """Get SQL template with parameters applied"""
        template = self.get_template(template_name)
        if not template:
            return None
        
        # 파라미터가 없으면 원본 템플릿 반환
        if not template.parameters:
            return template
        
        try:
            # 기본값 설정
            default_params = {
                "top_k": 5,
                "days": 30,
                "months": 12
            }
            
            # 사용자 파라미터와 기본값 병합
            final_params = {**default_params, **parameters}
            
            # 템플릿 복사 및 파라미터 적용
            import copy
            param_template = copy.deepcopy(template)
            
            # SQL 템플릿에 파라미터 적용
            # {month:02d} 같은 복잡한 포맷 문자열은 직접 처리
            sql_with_params = template.sql_template
            if 'month' in final_params:
                month_val = final_params['month']
                sql_with_params = sql_with_params.replace('{month:02d}', f"{month_val:02d}")
                sql_with_params = sql_with_params.replace('{month}', str(month_val))
            
            # 나머지 파라미터는 일반 format으로 처리
            other_params = {k: v for k, v in final_params.items() if k != 'month'}
            if other_params:
                sql_with_params = sql_with_params.format(**other_params)
            
            param_template.sql_template = sql_with_params
            
            # name과 description 포맷팅 (month 파라미터 포함)
            name_with_params = template.name
            desc_with_params = template.description
            if 'month' in final_params:
                month_val = final_params['month']
                name_with_params = name_with_params.replace('{month}', str(month_val))
                desc_with_params = desc_with_params.replace('{month}', str(month_val))
            
            other_params = {k: v for k, v in final_params.items() if k != 'month'}
            if other_params:
                name_with_params = name_with_params.format(**other_params)
                desc_with_params = desc_with_params.format(**other_params)
            
            param_template.name = name_with_params
            param_template.description = desc_with_params
            
            return param_template
            
        except KeyError as e:
            self.logger.error(f"Missing parameter {e} for template {template_name}")
            return template
        except Exception as e:
            self.logger.error(f"Error applying parameters to template {template_name}: {e}")
            return template
    
    def get_templates_by_type(self, analysis_type: FandingAnalysisType) -> List[SQLTemplate]:
        """Get all templates for a specific analysis type"""
        return [
            template for template in self.templates.values()
            if template.analysis_type == analysis_type
        ]
    
    def get_all_templates(self) -> Dict[str, SQLTemplate]:
        """Get all available templates"""
        return self.templates
    
    def is_ambiguous_query(self, query: str) -> bool:
        """애매한 쿼리인지 판단 (개선된 버전)"""
        query_lower = query.lower()
        
        # 애매한 키워드들
        ambiguous_keywords = [
            "회원 수", "회원수", "멤버 수", "맴버 수", "사용자 수", "가입자 수",
            "데이터", "정보", "통계", "분석", "결과", "현황"
        ]
        
        # 구체적인 키워드들 (확장됨)
        specific_keywords = [
            # 월 표현 (한국어)
            "1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월",
            "일월", "이월", "삼월", "사월", "오월", "육월", "칠월", "팔월", "구월", "십월", "십일월", "십이월",
            # 영어 월 표현
            "january", "february", "march", "april", "may", "june", "july", "august", 
            "september", "october", "november", "december",
            "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
            # 상대적 시간 표현
            "전체", "활성", "신규", "이탈", "월별", "일별", "주별", "년별",
            "올해", "작년", "지난달", "이번달", "이번주", "지난주", "어제", "오늘", "내일",
            "last month", "this month", "last year", "this year",
            # 성과 및 분석 키워드
            "성과", "실적", "추이", "변화", "증감", "성장률",
            # Top N 패턴
            "top5", "top10", "top3", "top1", "top2", "top4", "top6", "top7", "top8", "top9",
            "상위", "탑", "최고", "인기", "랭킹", "순위", "크리에이터", "creator",
            # 숫자 패턴
            "1위", "2위", "3위", "4위", "5위", "6위", "7위", "8위", "9위", "10위",
            "1등", "2등", "3등", "4등", "5등", "6등", "7등", "8등", "9등", "10등"
        ]
        
        # 애매한 키워드가 있지만 구체적인 키워드가 없는 경우
        has_ambiguous = any(keyword in query_lower for keyword in ambiguous_keywords)
        has_specific = any(keyword in query_lower for keyword in specific_keywords)
        
        return has_ambiguous and not has_specific

    def generate_clarification_question(self, query: str) -> str:
        """애매한 쿼리에 대한 구체적인 질문 생성"""
        query_lower = query.lower()
        
        if "회원" in query_lower or "멤버" in query_lower or "맴버" in query_lower:
            return """🤔 **어떤 회원 수를 원하시나요?**

다음 중에서 선택해주세요:

📊 **기본 회원 수**
• "전체 회원 수" - 모든 회원 (탈퇴 포함)
• "활성 회원 수" - 현재 활성 상태인 회원만

📈 **시간별 회원 수**
• "이번 달 신규 회원" - 10월 신규 가입
• "8월 신규 회원" - 특정 월 신규 가입
• "월별 회원 수 추이" - 월별 변화 추이

🎯 **성과 분석**
• "8월 멤버십 성과" - 월별 멤버십 성과
• "회원 리텐션 현황" - 회원 유지율 분석

어떤 정보가 필요하신지 말씀해주세요! 😊"""
        
        elif "데이터" in query_lower or "정보" in query_lower:
            return """🤔 **어떤 데이터를 원하시나요?**

다음 중에서 선택해주세요:

👥 **회원 관련**
• "전체 회원 수", "활성 회원 수"
• "월별 회원 수 추이", "회원 리텐션"

💰 **성과 관련**  
• "8월 멤버십 성과", "월간 매출 현황"
• "크리에이터 성과 분석"

📝 **콘텐츠 관련**
• "인기 포스트 TOP5", "포스트 참여도 분석"

어떤 데이터가 필요하신지 구체적으로 말씀해주세요! 😊"""
        
        else:
            return """🤔 **더 구체적으로 말씀해주세요!**

다음과 같은 형태로 질문해주시면 정확한 답변을 드릴 수 있습니다:

📊 **회원 관련**
• "전체 회원 수", "활성 회원 수"
• "8월 신규 회원", "월별 회원 수 추이"

💰 **성과 관련**
• "8월 멤버십 성과", "월간 매출 현황"

📝 **콘텐츠 관련**
• "인기 포스트 TOP5", "포스트 조회수 분석"

어떤 정보가 필요하신지 구체적으로 말씀해주세요! 😊"""

    def match_query_to_template(self, query: str) -> Optional[SQLTemplate]:
        """
        자연어 쿼리를 적절한 SQL 템플릿에 매칭 (키워드 기반 점수 매칭)
        
        Args:
            query: 사용자 쿼리
            
        Returns:
            매칭된 SQLTemplate 또는 None
        """
        query_lower = query.lower()
        
        # 파라미터 추출
        extracted_params = self._extract_parameters_from_query(query)
        
        # 키워드 기반 점수 매칭
        best_template = self._find_best_template_by_keywords(query_lower, extracted_params)
        
        return best_template
    
    def _find_best_template_by_keywords(self, query_lower: str, extracted_params: Dict[str, Any]) -> Optional[SQLTemplate]:
        """
        키워드 기반 점수 매칭으로 최적의 템플릿 찾기
        
        Args:
            query_lower: 소문자로 변환된 쿼리
            extracted_params: 추출된 파라미터
            
        Returns:
            최고 점수를 받은 템플릿 또는 None
        """
        template_scores = []
        
        for template_name, template in self.templates.items():
            if not template.keywords:
                continue
                
            # 키워드 매칭 점수 계산
            score = self._calculate_keyword_score(query_lower, template.keywords)
            
            if score > 0:
                template_scores.append((template, score, template_name))
        
        if not template_scores:
            return None
        
        # 점수순으로 정렬하여 최고 점수 템플릿 선택
        template_scores.sort(key=lambda x: x[1], reverse=True)
        best_template, best_score, best_name = template_scores[0]
        
        # 최소 임계점 이상인 경우만 반환
        if best_score >= 0.3:  # 30% 이상 매칭
            self.logger.debug(f"키워드 매칭: '{best_name}' (점수: {best_score:.2f})")
            
            # 파라미터가 있으면 적용
            if extracted_params:
                return self.get_parameterized_template(best_name, extracted_params)
        else:
                return best_template
        
        return None
    
    def _calculate_keyword_score(self, query_lower: str, template_keywords: Optional[List[str]]) -> float:
        """
        쿼리와 템플릿 키워드 간의 매칭 점수 계산 (Jaccard 유사도 기반, 개선됨)
        
        Args:
            query_lower: 소문자로 변환된 쿼리
            template_keywords: 템플릿의 키워드 리스트
            
        Returns:
            매칭 점수 (0.0 ~ 1.0)
        """
        if not template_keywords:
            return 0.0
        
        # 쿼리에서 키워드 추출 (단어별 분리 + 부분 매칭)
        import re
        query_words = set(query_lower.split())
        
        # 부분 매칭: "9월" 같은 키워드는 "9월신규"에서도 매칭되어야 함
        query_text = query_lower
        
        # 템플릿 키워드를 소문자로 변환
        template_words = set([kw.lower() for kw in template_keywords])
        
        # 교집합 계산 (단어 단위)
        word_intersection = query_words.intersection(template_words)
        
        # 부분 매칭 계산 (문자열 포함 여부)
        partial_matches = []
        for kw in template_keywords:
            kw_lower = kw.lower()
            if kw_lower in query_text:
                partial_matches.append(kw_lower)
        
        # 전체 매칭 키워드 집합 (단어 매칭 + 부분 매칭 중복 제거)
        all_matches = set(list(word_intersection) + partial_matches)
        
        # 월 키워드 가중치 (특정 월 템플릿 우선)
        month_keywords = ['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월']
        has_month_in_query = any(month in query_lower for month in month_keywords)
        has_month_in_template = any(month in template_keywords for month in month_keywords)
        
        # 월 매칭 보너스: 쿼리와 템플릿 모두 월이 있으면 높은 점수
        month_bonus = 0.0
        if has_month_in_query and has_month_in_template:
            # 쿼리에서 월 추출
            query_month = None
            for month in month_keywords:
                if month in query_lower:
                    query_month = month
                    break
            
            # 템플릿에서 월 추출
            template_month = None
            for month in month_keywords:
                if month in template_keywords:
                    template_month = month
                    break
            
            # 동일한 월이면 높은 보너스
            if query_month and template_month and query_month == template_month:
                month_bonus = 0.5
            elif query_month and template_month:
                month_bonus = 0.3  # 다른 월이지만 월 키워드가 둘 다 있음
        
        # 합집합 계산
        union = query_words.union(template_words)
        
        if not union:
            return 0.0
        
        # Jaccard 유사도 계산
        jaccard_score = len(all_matches) / len(union) if union else 0
        
        # 정확한 매칭 비율
        exact_matches = len(all_matches)
        total_template_keywords = len(template_keywords)
        exact_match_ratio = exact_matches / total_template_keywords if total_template_keywords > 0 else 0
        
        # 최종 점수: Jaccard 유사도, 정확한 매칭 비율, 월 보너스의 가중 평균
        base_score = (jaccard_score * 0.5) + (exact_match_ratio * 0.3) + (month_bonus)
        
        return min(base_score, 1.0)  # 최대 1.0으로 제한
    
    def _apply_dynamic_year_to_template(self, query: str, template: SQLTemplate) -> SQLTemplate:
        """템플릿에 동적 연도 처리 적용"""
        try:
            from .date_utils import DateUtils
            
            # 쿼리에서 연도 추출
            extracted_year = DateUtils.extract_year_from_query(query)
            if not extracted_year:
                return template
            
            # SQL 템플릿에서 연도 부분을 동적으로 교체
            sql_template = template.sql_template
            
            # 현재 연도를 추출된 연도로 교체
            if "CONCAT(YEAR(NOW()), " in sql_template:
                # CONCAT(YEAR(NOW()), '-09') 형태를 CONCAT('2024', '-09') 형태로 교체
                sql_template = sql_template.replace("CONCAT(YEAR(NOW()), ", f"CONCAT('{extracted_year}', ")
            elif "YEAR(NOW())" in sql_template:
                # YEAR(NOW())를 '2024'로 교체
                sql_template = sql_template.replace("YEAR(NOW())", f"'{extracted_year}'")
            
            # 새로운 SQLTemplate 생성
            return SQLTemplate(
                name=template.name,
                description=f"{template.description} ({extracted_year}년 데이터)",
                sql_template=sql_template,
                parameters=template.parameters,
                analysis_type=template.analysis_type
            )
        except Exception as e:
            # 동적 연도 처리 실패 시 원본 템플릿 반환
            return template
    
    def format_sql_result(self, template: SQLTemplate, result: List[Dict]) -> str:
        """Format SQL result for user-friendly display"""
        if not result:
            return f"📊 **{template.name}**\n\n데이터가 없습니다."
        
        # 기본 포맷팅
        formatted_result = f"📊 **{template.name}**\n\n"
        
        # 결과 데이터 포맷팅
        if len(result) == 1:
            # 단일 결과
            row = result[0]
            for key, value in row.items():
                formatted_result += f"• **{key}**: {value}\n"
        else:
            # 다중 결과 - 테이블 형태
            if result:
                headers = list(result[0].keys())
                formatted_result += "| " + " | ".join(headers) + " |\n"
                formatted_result += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                
                for row in result[:10]:  # 최대 10개 행만 표시
                    values = [str(row.get(header, "")) for header in headers]
                    formatted_result += "| " + " | ".join(values) + " |\n"
                
                if len(result) > 10:
                    formatted_result += f"\n*총 {len(result)}개 결과 중 상위 10개만 표시*"
        
        return formatted_result
    
    def create_dynamic_monthly_template(self, query: str) -> Optional[SQLTemplate]:
        """
        동적으로 월별 멤버십 성과 템플릿 생성 (개선된 날짜 처리)
        
        Args:
            query: 사용자 쿼리
            
        Returns:
            생성된 SQLTemplate 또는 None
        """
        try:
            from .date_utils import DateUtils
            
            # 쿼리에서 월 추출 (개선된 날짜 처리)
            month_info = DateUtils.extract_month_with_year_from_query(query)
            if not month_info:
                return None
                
            year, month = month_info
            
            # 월을 두 자리 숫자로 변환
            month_num = f"{month:02d}"
            
            # 정확한 YYYY-MM 형식 생성
            yyyy_mm = f"{year}-{month_num}"
            
            # 동적 SQL 템플릿 생성 (개선된 날짜 필터링)
            sql_template = f"""
            SELECT 
                '{month_num}월' as analysis_month,
                COUNT(DISTINCT m.no) as total_members,
                COUNT(DISTINCT CASE WHEN m.status = 'A' THEN m.no END) as active_members,
                COUNT(DISTINCT CASE WHEN m.status = 'I' THEN m.no END) as inactive_members,
                COUNT(DISTINCT CASE WHEN m.status = 'D' THEN m.no END) as deleted_members,
                ROUND(COUNT(DISTINCT CASE WHEN m.status = 'A' THEN m.no END) * 100.0 / COUNT(DISTINCT m.no), 2) as active_rate_percent,
                ROUND(COUNT(DISTINCT CASE WHEN m.status = 'I' THEN m.no END) * 100.0 / COUNT(DISTINCT m.no), 2) as inactive_rate_percent,
                ROUND(COUNT(DISTINCT CASE WHEN m.status = 'D' THEN m.no END) * 100.0 / COUNT(DISTINCT m.no), 2) as deletion_rate_percent
            FROM t_member m
            LEFT JOIN t_member_login_log l ON m.no = l.member_no
            WHERE DATE_FORMAT(l.ins_datetime, '%Y-%m') = '{yyyy_mm}'
            """
            
            return SQLTemplate(
                name=f"{month_num}월 멤버십 성과 분석",
                description=f"{month_num}월 멤버십 성과 상세 분석 ({year}년 데이터)",
                sql_template=sql_template,
                parameters=[],
                analysis_type=FandingAnalysisType.MEMBERSHIP_DATA,
                keywords=["멤버십", "맴버쉽", "성과", "실적", "분석", f"{month}월", f"{month_num}월"]
            )
        except Exception as e:
            self.logger.error(f"동적 월별 템플릿 생성 실패: {str(e)}")
            return None

    def get_schema_info(self, query: str) -> Optional[str]:
        """스키마 정보 조회 (SHOW/DESCRIBE 대안)"""
        query_lower = query.lower().strip()
        
        # 너무 짧은 쿼리는 스키마 정보 요청이 아님
        if len(query_lower) < 3:
            return None
            
        # 명확한 스키마 관련 키워드가 있어야 함
        schema_keywords = ['테이블', 'table', '어떤', '목록', '리스트', '구조', 'structure', '스키마', 'schema', '컬럼', 'column']
        has_schema_keyword = any(keyword in query_lower for keyword in schema_keywords)
        
        # 테이블 목록 조회
        if has_schema_keyword and any(keyword in query_lower for keyword in ['테이블', 'table', '어떤', '목록', '리스트']):
            return self._get_table_list()
        
        # 특정 테이블 구조 조회 (더 엄격한 조건)
        if has_schema_keyword:
            for table_name, description in self._get_table_descriptions().items():
                # description이 None인 경우 처리
                description_safe = description or ""
                # 테이블명이 쿼리에 명시적으로 포함되어야 함 (부분 매칭 방지)
                if (table_name.lower() in query_lower and 
                    len(table_name) > 3 and  # 너무 짧은 테이블명 제외
                    query_lower.count(table_name.lower()) == 1):  # 정확히 한 번만 매칭
                    return self._get_table_structure(table_name, description)
        
        return None
    
    def _get_table_list(self) -> str:
        """접근 가능한 테이블 목록 반환"""
        tables = self._get_table_descriptions()
        
        result = "📋 **접근 가능한 테이블 목록**\n\n"
        for table_name, description in tables.items():
            result += f"• **{table_name}**: {description}\n"
        
        result += f"\n총 {len(tables)}개의 테이블에 접근 가능합니다."
        return result
    
    def _get_table_descriptions(self) -> Dict[str, str]:
        """테이블별 설명 반환 (동적 스키마 정보 사용)"""
        descriptions = {}
        for table_name, table_info in self.db_schema.items():
            descriptions[table_name] = table_info.get("description", f"{table_name} table")
        return descriptions
    
    def _get_table_structure(self, table_name: str, description: str) -> str:
        """특정 테이블의 구조 정보 반환"""
        # 실제 컬럼 정보 (하드코딩된 스키마 정보 활용)
        column_info = self._get_table_columns(table_name)
        
        result = f"📊 **{table_name} 테이블 구조**\n\n"
        result += f"**설명**: {description}\n\n"
        result += "**주요 컬럼**:\n"
        
        for column, col_type in column_info.items():
            result += f"• **{column}**: {col_type}\n"
        
        return result
    
    def _get_table_columns(self, table_name: str) -> Dict[str, str]:
        """테이블별 주요 컬럼 정보 (동적 스키마 정보 사용)"""
        if table_name not in self.db_schema:
            return {}
        
        table_info = self.db_schema[table_name]
        columns = table_info.get("columns", {})
        
        # 컬럼 정보를 사용자 친화적인 형태로 변환
        column_descriptions = {}
        for column_name, column_info in columns.items():
            col_type = column_info.get("type", "")
            col_desc = column_info.get("description", "")
            nullable = column_info.get("nullable", True)
            
            # 컬럼 설명 생성
            description_parts = []
            if col_desc:
                description_parts.append(col_desc)
            if col_type:
                description_parts.append(f"({col_type})")
            if not nullable:
                description_parts.append("NOT NULL")
            
            column_descriptions[column_name] = " ".join(description_parts) if description_parts else f"{column_name} ({col_type})"
        
        return column_descriptions
    
    def _extract_parameters_from_query(self, query: str) -> Dict[str, Any]:
        """쿼리에서 파라미터 추출"""
        import re
        
        params = {}
        query_lower = query.lower()
        
        # TOP N 패턴 추출 (top5, top10, 상위 3개 등)
        top_patterns = [
            r'top\s*(\d+)',
            r'상위\s*(\d+)',
            r'탑\s*(\d+)',
            r'(\d+)\s*위',
            r'(\d+)\s*등'
        ]
        
        for pattern in top_patterns:
            match = re.search(pattern, query_lower)
            if match:
                params['top_k'] = int(match.group(1))
                break
        
        # 기간 패턴 추출 (최근 7일, 지난 30일 등)
        period_patterns = [
            r'최근\s*(\d+)\s*일',
            r'지난\s*(\d+)\s*일',
            r'(\d+)\s*일간',
            r'(\d+)\s*일\s*동안'
        ]
        
        for pattern in period_patterns:
            match = re.search(pattern, query_lower)
            if match:
                params['days'] = int(match.group(1))
                break
        
        # 월 패턴 추출 (최근 3개월, 지난 6개월 등)
        month_patterns = [
            r'최근\s*(\d+)\s*개?월',
            r'지난\s*(\d+)\s*개?월',
            r'(\d+)\s*개?월간',
            r'(\d+)\s*개?월\s*동안'
        ]
        
        for pattern in month_patterns:
            match = re.search(pattern, query_lower)
            if match:
                params['months'] = int(match.group(1))
                break
        
        # 단일 월 파라미터 추출 (예: "9월", "9월 신규 회원")
        single_month_pattern = r'(\d+)\s*월'
        month_match = re.search(single_month_pattern, query_lower)
        if month_match:
            month_num = int(month_match.group(1))
            if 1 <= month_num <= 12:
                params['month'] = month_num
        
        return params

    def _validate_templates(self) -> None:
        """
        템플릿과 실제 DB 스키마 간의 동기화 검증
        
        모든 템플릿의 SQL에서 사용된 테이블과 컬럼명이 실제 DB 스키마에 존재하는지 확인합니다.
        존재하지 않는 테이블/컬럼이 발견되면 심각한 오류 로그를 기록합니다.
        """
        validation_errors = []
        validation_warnings = []
        
        self.logger.info("템플릿 스키마 검증을 시작합니다...")
        
        for template_name, template in self.templates.items():
            try:
                # SQL 템플릿에서 테이블명과 컬럼명 추출
                sql_content = template.sql_template.lower()
                
                # 테이블명 추출 (FROM, JOIN 절에서)
                table_names = self._extract_table_names_from_sql(sql_content)
                
                # 각 테이블에 대해 검증
                for table_name in table_names:
                    if table_name not in self.db_schema:
                        error_msg = f"Template '{template_name}' uses invalid table: '{table_name}'"
                        validation_errors.append(error_msg)
                        self.logger.error(error_msg)
                        continue
                    
                    # 테이블이 존재하면 컬럼명 검증
                    table_info = self.db_schema[table_name]
                    table_columns = set(table_info.get("columns", {}).keys())
                    
                    # SQL에서 사용된 컬럼명 추출
                    column_names = self._extract_column_names_from_sql(sql_content, table_name)
                    
                    for column_name in column_names:
                        if column_name not in table_columns:
                            error_msg = f"Template '{template_name}' uses invalid column '{column_name}' in table '{table_name}'"
                            validation_errors.append(error_msg)
                            self.logger.error(error_msg)
                
                # 템플릿 파라미터 검증
                if template.parameters:
                    for param in template.parameters:
                        if f"{{{param}}}" not in template.sql_template:
                            warning_msg = f"Template '{template_name}' declares parameter '{param}' but doesn't use it in SQL"
                            validation_warnings.append(warning_msg)
                            self.logger.warning(warning_msg)
                
            except Exception as e:
                error_msg = f"Template '{template_name}' validation failed: {str(e)}"
                validation_errors.append(error_msg)
                self.logger.error(error_msg)
        
        # 검증 결과 요약
        if validation_errors:
            self.logger.error(f"템플릿 검증 실패: {len(validation_errors)}개 오류 발견")
            self.logger.error("발견된 오류들:")
            for error in validation_errors:
                self.logger.error(f"  - {error}")
        else:
            self.logger.info(f"템플릿 검증 성공: {len(self.templates)}개 템플릿 모두 유효")
        
        if validation_warnings:
            self.logger.warning(f"템플릿 검증 경고: {len(validation_warnings)}개 경고 발견")
            for warning in validation_warnings:
                self.logger.warning(f"  - {warning}")
    
    def _extract_table_names_from_sql(self, sql_content: str) -> List[str]:
        """
        SQL 쿼리에서 테이블명 추출 (CTE 제외)
        
        Args:
            sql_content: SQL 쿼리 문자열 (소문자)
            
        Returns:
            추출된 테이블명 리스트 (CTE 제외)
        """
        import re
        
        table_names = []
        
        # CTE 이름들을 먼저 추출하여 제외
        cte_names = set()
        cte_pattern = r'with\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\('
        cte_matches = re.findall(cte_pattern, sql_content)
        cte_names.update(cte_matches)
        
        # FROM 절에서 테이블명 추출
        from_pattern = r'from\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        from_matches = re.findall(from_pattern, sql_content)
        table_names.extend(from_matches)
        
        # JOIN 절에서 테이블명 추출
        join_pattern = r'join\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        join_matches = re.findall(join_pattern, sql_content)
        table_names.extend(join_matches)
        
        # LEFT JOIN, RIGHT JOIN, INNER JOIN 등도 처리
        left_join_pattern = r'left\s+join\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        left_join_matches = re.findall(left_join_pattern, sql_content)
        table_names.extend(left_join_matches)
        
        right_join_pattern = r'right\s+join\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        right_join_matches = re.findall(right_join_pattern, sql_content)
        table_names.extend(right_join_matches)
        
        inner_join_pattern = r'inner\s+join\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        inner_join_matches = re.findall(inner_join_pattern, sql_content)
        table_names.extend(inner_join_matches)
        
        # CTE 이름들을 제외하고 중복 제거
        filtered_names = [name for name in table_names if name not in cte_names]
        return list(set(filtered_names))
    
    def _extract_column_names_from_sql(self, sql_content: str, table_name: str) -> List[str]:
        """
        SQL 쿼리에서 특정 테이블의 컬럼명 추출
        
        Args:
            sql_content: SQL 쿼리 문자열 (소문자)
            table_name: 검증할 테이블명
            
        Returns:
            추출된 컬럼명 리스트
        """
        import re
        
        column_names = []
        
        # 테이블명.컬럼명 패턴 추출
        table_column_pattern = rf'{table_name}\.([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(table_column_pattern, sql_content)
        column_names.extend(matches)
        
        # WHERE 절에서 직접 사용된 컬럼명 추출 (테이블명 없이)
        # 단, 이는 더 복잡한 로직이 필요하므로 일단 테이블명.컬럼명 패턴만 처리
        
        return list(set(column_names))
