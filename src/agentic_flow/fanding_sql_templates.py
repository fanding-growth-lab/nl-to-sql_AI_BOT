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
    
    def __init__(self, db_schema: Optional[Dict[str, Any]] = None):
        self.templates = self._initialize_templates()
        self.logger = logging.getLogger(__name__)
        # db_schema가 제공되지 않으면 초기화 시점에 한 번만 로드 (성능 최적화)
        if db_schema is None or len(db_schema) == 0:
            self.db_schema = get_cached_db_schema()
            self.logger.debug("db_schema was not provided, loaded from cache during initialization")
        else:
            self.db_schema = db_schema
        
        # 스키마가 fallback인지 확인 (DB 연결 실패 시)
        self._is_fallback_schema = self._check_if_fallback_schema()
        
        # 템플릿 검증 실행 (중요: 스키마 동기화 확인)
        # 단, fallback 스키마일 때는 경고만 표시하고 오류로 처리하지 않음
        self._validate_templates()
    
    def _initialize_templates(self) -> Dict[str, SQLTemplate]:
        """
        Initialize all Fanding SQL templates
        
        주의: db_schema는 __init__에서 로드되므로, 여기서는 아직 사용할 수 없습니다.
        템플릿은 정적으로 정의하고, 실제 사용 시 db_schema를 기반으로 검증/수정됩니다.
        """
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
    
    def _get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        db_schema에서 테이블 정보 가져오기 (헬퍼 메서드)
        
        Args:
            table_name: 테이블명
            
        Returns:
            테이블 정보 딕셔너리 또는 None
        """
        if not self.db_schema or len(self.db_schema) == 0:
            return None
        return self.db_schema.get(table_name)
    
    def _get_table_columns(self, table_name: str) -> Dict[str, Any]:
        """
        db_schema에서 테이블의 컬럼 정보 가져오기 (헬퍼 메서드)
        
        Args:
            table_name: 테이블명
            
        Returns:
            컬럼 딕셔너리 (빈 딕셔너리 반환 가능)
        """
        table_info = self._get_table_info(table_name)
        if not table_info:
            return {}
        return table_info.get("columns", {})
    
    def _find_pk_column(self, table_name: str) -> Optional[str]:
        """
        테이블의 Primary Key 컬럼 찾기
        
        Args:
            table_name: 테이블명
            
        Returns:
            PK 컬럼명 또는 None
        """
        columns = self._get_table_columns(table_name)
        # 일반적인 PK 컬럼명 후보
        pk_candidates = ['no', 'id', f'{table_name.replace("t_", "")}_no', 'member_no']
        for candidate in pk_candidates:
            if candidate in columns:
                return candidate
        return None
    
    def _has_column(self, table_name: str, column_name: str) -> bool:
        """
        테이블에 특정 컬럼이 있는지 확인
        
        Args:
            table_name: 테이블명
            column_name: 컬럼명
            
        Returns:
            컬럼 존재 여부
        """
        columns = self._get_table_columns(table_name)
        return column_name in columns
    
    def _validate_and_fix_template_sql(self, sql_template: str) -> str:
        """
        템플릿 SQL을 db_schema 기반으로 검증하고 수정
        
        Args:
            sql_template: 원본 SQL 템플릿
            
        Returns:
            검증/수정된 SQL 템플릿
        """
        if not self.db_schema or len(self.db_schema) == 0:
            return sql_template
        
        import re
        fixed_sql = sql_template
        
        # t_member와 t_member_info 관련 자동 수정
        # t_member_info에 ins_datetime이 있는데 t_member를 사용하는 경우
        if 't_member' in fixed_sql.lower() and 'ins_datetime' in fixed_sql.lower():
            # t_fanding을 사용하는 경우는 변경하지 않음
            if 't_fanding' not in fixed_sql.lower():
                # t_member_info에 ins_datetime이 있는지 확인
                if self._has_column('t_member_info', 'ins_datetime'):
                    # t_member를 t_member_info로 변경 (단, 이미 t_member_info가 있으면 안 함)
                    if 't_member_info' not in fixed_sql.lower():
                        fixed_sql = re.sub(r'\bt_member\b', 't_member_info', fixed_sql, flags=re.IGNORECASE)
                        self.logger.debug("자동 수정: t_member -> t_member_info (ins_datetime 사용)")
        
        # 컬럼명 자동 수정
        # t_member_info에서 member_no를 사용해야 하는데 no를 사용하는 경우
        if 't_member_info' in fixed_sql.lower():
            # t_member_info에는 'no'가 없고 'member_no'가 있음
            if not self._has_column('t_member_info', 'no') and self._has_column('t_member_info', 'member_no'):
                # m.no 패턴을 m.member_no로 변경 (단, 이미 member_no가 있으면 안 함)
                fixed_sql = re.sub(r'\bm\.no\b', 'm.member_no', fixed_sql, flags=re.IGNORECASE)
                self.logger.debug("자동 수정: m.no -> m.member_no (t_member_info 사용)")
        
        return fixed_sql
    
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
                       keywords=["활성", "회원", "멤버", "맴버", "활성회원", "활성멤버", "로그인", "최근", "크리에이터", "creator"]
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
                       keywords=["신규", "회원", "멤버", "맴버", "신규회원", "신규멤버", "가입", "현황", "이번달", "이번", "크리에이터", "creator"]
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
                       keywords=["신규", "회원", "멤버", "맴버", "신규회원", "신규멤버", "가입", "현황", "1월", "2월", "3월", "4월", "5월", "6월", "7월", "8월", "9월", "10월", "11월", "12월", "크리에이터", "creator"]
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
        """
        Get SQL template by name
        
        템플릿을 반환하기 전에 db_schema를 기반으로 SQL을 검증/수정합니다.
        """
        template = self.templates.get(template_name)
        if not template:
            return None
        
        # db_schema 기반으로 SQL 검증 및 자동 수정
        if self.db_schema and len(self.db_schema) > 0:
            fixed_sql = self._validate_and_fix_template_sql(template.sql_template)
            if fixed_sql != template.sql_template:
                # 수정된 SQL로 새 템플릿 생성
                from copy import deepcopy
                fixed_template = deepcopy(template)
                fixed_template.sql_template = fixed_sql
                self.logger.debug(f"템플릿 '{template_name}' SQL이 db_schema 기반으로 자동 수정되었습니다")
                return fixed_template
        
        return template
    
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
            
            # db_schema 기반으로 SQL 검증 및 자동 수정 (파라미터 적용 후)
            if self.db_schema and len(self.db_schema) > 0:
                fixed_sql = self._validate_and_fix_template_sql(sql_with_params)
                if fixed_sql != sql_with_params:
                    param_template.sql_template = fixed_sql
                    self.logger.debug(f"파라미터 적용된 템플릿 '{template_name}' SQL이 db_schema 기반으로 자동 수정되었습니다")
            
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
    
    def match_query_to_template(self, query: str) -> Optional[SQLTemplate]:
        """
        자연어 쿼리를 적절한 SQL 템플릿에 매칭 (키워드 기반 점수 매칭)
        
        쿼리에서 파라미터(예: 월 정보)를 추출하여 매칭된 템플릿의 파라미터를 자동으로 채웁니다.
        
        주의: 크리에이터 키워드가 있는 쿼리는 매칭된 템플릿에 크리에이터 필터가 있는지 확인합니다.
        크리에이터 필터가 없으면 None을 반환하여 동적 템플릿 생성을 유도합니다.
        
        Args:
            query: 사용자 쿼리
            
        Returns:
            매칭된 SQLTemplate (파라미터 적용됨) 또는 None
        """
        query_lower = query.lower()
        
        # 크리에이터 정보가 필요한 쿼리인지 확인
        has_creator_keyword = (
            '크리에이터' in query_lower or 
            'creator' in query_lower or
            any(keyword in query_lower for keyword in ['작가', '아티스트', '제작자'])
        )
        
        # 파라미터 추출 (월 정보 등)
        extracted_params = self._extract_parameters_from_query(query)
        
        if extracted_params:
            self.logger.debug(f"쿼리에서 추출된 파라미터: {extracted_params}")
        
        # 키워드 기반 점수 매칭
        best_template = self._find_best_template_by_keywords(query_lower, extracted_params)
        
        # 매칭된 템플릿이 있고 파라미터가 있으면 적용
        if best_template and extracted_params:
            # 파라미터가 필요한 템플릿인지 확인
            if best_template.parameters:
                param_template = self.get_parameterized_template(
                    self._get_template_name(best_template), 
                    extracted_params
                )
                if param_template:
                    best_template = param_template
                    self.logger.info(f"매칭된 템플릿 '{best_template.name}'에 파라미터 적용: {extracted_params}")
        
        # 크리에이터 키워드가 있는 경우, 템플릿에 크리에이터 필터가 있는지 확인
        if best_template and has_creator_keyword:
            sql_template = best_template.sql_template if hasattr(best_template, 'sql_template') else str(best_template)
            has_creator_filter = 'creator' in sql_template.lower() or 'creator_no' in sql_template.lower()
            uses_t_fanding = 't_fanding' in sql_template.lower() or ('f.' in sql_template.lower() and 'FROM t_fanding' in sql_template.upper())
            
            if not has_creator_filter and not uses_t_fanding:
                # 크리에이터 필터가 없고 t_fanding도 사용하지 않으면 템플릿 스킵
                self.logger.info(
                    f"템플릿 '{best_template.name}'이 매칭되었지만 크리에이터 필터가 없고 t_fanding을 사용하지 않습니다. "
                    f"동적 템플릿 생성을 위해 None 반환"
                )
                return None
        
        return best_template
    
    def _get_template_name(self, template: SQLTemplate) -> str:
        """템플릿 객체에서 템플릿 이름 찾기 (키 조회용)"""
        for name, tpl in self.templates.items():
            if tpl == template or (hasattr(tpl, 'name') and tpl.name == template.name):
                return name
        return ""
    
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
            
            # 파라미터가 있으면 적용 (매칭된 템플릿의 파라미터를 동적으로 채움)
            if extracted_params and best_template.parameters:
                param_template = self.get_parameterized_template(best_name, extracted_params)
                # get_parameterized_template 내부에서 이미 db_schema 기반 검증/수정이 수행됨
                if param_template:
                    self.logger.debug(f"템플릿 '{best_name}' 파라미터 적용 완료: {extracted_params}")
                    return param_template
                else:
                    # 파라미터 적용 실패 시 원본 템플릿 반환
                    self.logger.warning(f"템플릿 '{best_name}' 파라미터 적용 실패, 원본 템플릿 반환")
                    return best_template
            else:
                # 파라미터가 없거나 템플릿에 파라미터가 필요 없으면 원본 템플릿 반환
                # db_schema 기반 검증/수정은 match_query_to_template에서 처리됨
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
        동적으로 월별 템플릿 생성 (매칭된 템플릿이 없을 때만 사용)
        
        주의: 이 메서드는 match_query_to_template에서 템플릿이 매칭되지 않았을 때만 호출되어야 합니다.
        매칭된 템플릿이 있으면 그 템플릿의 파라미터를 동적으로 채우는 것이 우선입니다.
        
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
            
            # 쿼리 의도 분석: 신규 회원수 vs 멤버십 성과 분석
            query_lower = query.lower()
            has_new_member_keywords = any(keyword in query_lower for keyword in [
                '신규', '새로운', '가입', 'new', '신규회원', '신규멤버', '회원수', '회원 수'
            ])
            has_membership_performance_keywords = any(keyword in query_lower for keyword in [
                '멤버십', '맴버쉽', '성과', '실적', 'performance', 'membership', '분석'
            ])
            
            # 크리에이터 정보가 필요한 쿼리인지 확인 (중요: 크리에이터 필터가 필요하면 t_fanding 사용)
            has_creator_keyword = (
                '크리에이터' in query_lower or 
                'creator' in query_lower or
                any(keyword in query_lower for keyword in ['작가', '아티스트', '제작자'])
            )
                
            year, month = month_info
            
            # 월을 두 자리 숫자로 변환
            month_num = f"{month:02d}"
            
            # 정확한 YYYY-MM 형식 생성
            yyyy_mm = f"{year}-{month_num}"
            
            # db_schema에서 실제 테이블과 컬럼 정보 확인
            if not self.db_schema or len(self.db_schema) == 0:
                self.logger.warning("db_schema가 비어있어 동적 템플릿 생성 실패")
                return None
            
            # t_fanding 테이블 확인
            if 't_fanding' not in self.db_schema:
                self.logger.warning("t_fanding 테이블이 db_schema에 없어 동적 템플릿 생성 실패")
                return None
            
            fanding_schema = self.db_schema['t_fanding']
            fanding_columns = fanding_schema.get('columns', {})
            
            # 필요한 컬럼 확인
            if 'member_no' not in fanding_columns or 'ins_datetime' not in fanding_columns:
                self.logger.warning("t_fanding에 필요한 컬럼(member_no, ins_datetime)이 없어 동적 템플릿 생성 실패")
                return None
            
            # t_member 테이블 확인 (JOIN용)
            member_table = None
            member_pk = None
            member_status_col = None
            
            if 't_member' in self.db_schema:
                member_table = 't_member'
                member_schema = self.db_schema['t_member']
                member_columns = member_schema.get('columns', {})
                # t_member의 PK 찾기 (일반적으로 'no')
                for col_name in ['no', 'id', 'member_no']:
                    if col_name in member_columns:
                        member_pk = col_name
                        break
                # status 컬럼 확인
                if 'status' in member_columns:
                    member_status_col = 'status'
            elif 't_member_info' in self.db_schema:
                member_table = 't_member_info'
                member_schema = self.db_schema['t_member_info']
                member_columns = member_schema.get('columns', {})
                # t_member_info의 PK 찾기
                for col_name in ['member_no', 'no', 'id']:
                    if col_name in member_columns:
                        member_pk = col_name
                        break
                # status 컬럼 확인
                if 'status' in member_columns:
                    member_status_col = 'status'
            
            if not member_table or not member_pk:
                self.logger.warning("t_member 또는 t_member_info 테이블을 찾을 수 없어 동적 템플릿 생성 실패")
                return None
            
            # JOIN 조건 구성 (동적)
            join_condition = f"f.member_no = m.{member_pk}"
            
            # 쿼리 의도에 따라 다른 템플릿 생성
            if has_new_member_keywords and not has_membership_performance_keywords:
                # 신규 회원수 템플릿 생성
                # 크리에이터 필터가 필요한 경우 t_fanding 사용 (creator_no 컬럼 필요)
                if has_creator_keyword:
                    # 크리에이터 필터가 필요한 경우: t_fanding 사용 (creator_no 컬럼 포함)
                    if 't_fanding' in self.db_schema and self._has_column('t_fanding', 'ins_datetime') and self._has_column('t_fanding', 'creator_no'):
                        # t_fanding 사용 (멤버십 가입 기준, 크리에이터 필터 가능)
                        sql_template = f"""
            SELECT COUNT(DISTINCT f.member_no) as new_members_{month_num}month 
            FROM t_fanding f
            WHERE DATE_FORMAT(f.ins_datetime, '%Y-%m') = '{yyyy_mm}'
            """
                        self.logger.info(f"크리에이터 필터가 필요한 쿼리로 t_fanding 테이블 사용 (creator_no 필터는 SQLGenerationNode에서 추가됨)")
                    else:
                        self.logger.warning("크리에이터 필터가 필요한데 t_fanding 테이블 또는 creator_no 컬럼이 없습니다")
                        return None
                else:
                    # 크리에이터 필터가 필요 없는 경우: t_member_info 또는 t_fanding 사용
                    if 't_member_info' in self.db_schema and self._has_column('t_member_info', 'ins_datetime'):
                        # t_member_info 사용 (더 정확한 신규 회원수)
                        sql_template = f"""
            SELECT COUNT(*) as new_members_{month_num}month 
            FROM t_member_info 
            WHERE DATE_FORMAT(ins_datetime, '%Y-%m') = '{yyyy_mm}'
            """
                    elif 't_fanding' in self.db_schema and self._has_column('t_fanding', 'ins_datetime'):
                        # t_fanding 사용 (멤버십 가입 기준)
                        sql_template = f"""
            SELECT COUNT(DISTINCT f.member_no) as new_members_{month_num}month 
            FROM t_fanding f
            WHERE DATE_FORMAT(f.ins_datetime, '%Y-%m') = '{yyyy_mm}'
            """
                    else:
                        # fallback: t_fanding 사용
                        sql_template = f"""
            SELECT COUNT(DISTINCT f.member_no) as new_members_{month_num}month 
            FROM t_fanding f
            WHERE DATE_FORMAT(f.ins_datetime, '%Y-%m') = '{yyyy_mm}'
            """
                
                self.logger.info(f"동적 신규 회원 템플릿 생성 완료: {month_num}월")
                return SQLTemplate(
                    name=f"{month_num}월 신규 회원",
                    description=f"{month_num}월 신규 가입 회원 수 ({year}년 데이터, db_schema 기반 동적 생성)",
                    sql_template=sql_template,
                    parameters=[],
                    analysis_type=FandingAnalysisType.MEMBERSHIP_DATA,
                    keywords=["신규", "회원", "멤버", "맴버", "신규회원", "신규멤버", "가입", "현황", f"{month}월", f"{month_num}월"]
                )
            else:
                # 멤버십 성과 분석 템플릿 생성 (기본값)
                # status 컬럼이 있으면 status 기반 집계, 없으면 기본 집계만
                if member_status_col:
                    sql_template = f"""
            SELECT 
                '{month_num}월' as analysis_month,
                COUNT(DISTINCT f.member_no) as total_members,
                COUNT(DISTINCT CASE WHEN m.{member_status_col} = 'A' THEN f.member_no END) as active_members,
                COUNT(DISTINCT CASE WHEN m.{member_status_col} = 'I' THEN f.member_no END) as inactive_members,
                COUNT(DISTINCT CASE WHEN m.{member_status_col} = 'D' THEN f.member_no END) as deleted_members,
                ROUND(COUNT(DISTINCT CASE WHEN m.{member_status_col} = 'A' THEN f.member_no END) * 100.0 / COUNT(DISTINCT f.member_no), 2) as active_rate_percent,
                ROUND(COUNT(DISTINCT CASE WHEN m.{member_status_col} = 'I' THEN f.member_no END) * 100.0 / COUNT(DISTINCT f.member_no), 2) as inactive_rate_percent,
                ROUND(COUNT(DISTINCT CASE WHEN m.{member_status_col} = 'D' THEN f.member_no END) * 100.0 / COUNT(DISTINCT f.member_no), 2) as deletion_rate_percent
            FROM t_fanding f
            INNER JOIN {member_table} m ON {join_condition}
            WHERE DATE_FORMAT(f.ins_datetime, '%Y-%m') = '{yyyy_mm}'
            """
                else:
                    # status 컬럼이 없으면 기본 집계만
                    sql_template = f"""
            SELECT 
                '{month_num}월' as analysis_month,
                COUNT(DISTINCT f.member_no) as total_members
            FROM t_fanding f
            INNER JOIN {member_table} m ON {join_condition}
            WHERE DATE_FORMAT(f.ins_datetime, '%Y-%m') = '{yyyy_mm}'
            """
                
                self.logger.info(f"동적 멤버십 성과 템플릿 생성 완료: {member_table} 사용, PK={member_pk}, status_col={member_status_col}")
                
                return SQLTemplate(
                    name=f"{month_num}월 멤버십 성과 분석",
                    description=f"{month_num}월 멤버십 성과 상세 분석 ({year}년 데이터, db_schema 기반 동적 생성)",
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
        
        # 단일 월 파라미터 추출 (예: "9월", "9월 신규 회원", "10월 신규 회원수")
        # 다양한 패턴 지원: "10월", "10월달", "10월분", "10월 신규", "10월의" 등
        single_month_patterns = [
            r'(\d+)\s*월\s*(?:달|분|의|에|로|으로|에서)?',  # "10월", "10월달", "10월의" 등
            r'(\d+)\s*월달',  # "10월달"
            r'(\d+)\s*월분',  # "10월분"
        ]
        
        for pattern in single_month_patterns:
            month_match = re.search(pattern, query_lower)
            if month_match:
                month_num = int(month_match.group(1))
                if 1 <= month_num <= 12:
                    params['month'] = month_num
                    self.logger.debug(f"쿼리에서 월 파라미터 추출: {month_num}월")
                    break  # 첫 번째 매칭만 사용
        
        return params

    def _check_if_fallback_schema(self) -> bool:
        """
        현재 스키마가 fallback 스키마인지 확인
        
        Returns:
            bool: fallback 스키마이면 True
        """
        # fallback 스키마는 일반적으로 제한된 테이블만 포함
        # 실제 DB 스키마는 더 많은 테이블을 포함할 것으로 예상
        fallback_tables = {"t_member", "t_creator", "t_funding"}
        current_tables = set(self.db_schema.keys())
        
        # fallback 테이블과 정확히 일치하거나, 매우 적은 테이블만 있으면 fallback으로 간주
        if len(current_tables) <= 3 and current_tables.issubset(fallback_tables):
            return True
        
        # 또는 실제 DB에서 로드된 테이블이 매우 적으면 (DB 연결 실패 가능성)
        if len(current_tables) <= 5:
            self.logger.warning(f"스키마에 테이블이 {len(current_tables)}개만 있습니다. DB 연결 실패 가능성이 있습니다.")
            return True
        
        return False
    
    def _validate_templates(self) -> None:
        """
        템플릿과 실제 DB 스키마 간의 동기화 검증
        
        모든 템플릿의 SQL에서 사용된 테이블과 컬럼명이 실제 DB 스키마에 존재하는지 확인합니다.
        존재하지 않는 테이블/컬럼이 발견되면 심각한 오류 로그를 기록합니다.
        단, fallback 스키마일 때는 경고로 처리합니다.
        """
        validation_errors = []
        validation_warnings = []
        
        self.logger.info("템플릿 스키마 검증을 시작합니다...")
        
        # fallback 스키마일 때는 검증을 건너뛰고 경고만 표시
        if self._is_fallback_schema:
            self.logger.warning(
                f"⚠️  스키마가 fallback 모드입니다 (DB 연결 실패 가능성). "
                f"템플릿 검증을 건너뜁니다. 현재 스키마 테이블 수: {len(self.db_schema)}"
            )
            return
        
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

            # fallback 스키마가 아닌 경우에만 오류로 처리
            if self._is_fallback_schema:
                # fallback 스키마일 때는 경고로만 표시
                self.logger.warning(
                    f"템플릿 검증 경고 (fallback 스키마): {len(validation_errors)}개 템플릿이 현재 스키마와 일치하지 않습니다. "
                    f"DB 연결이 정상화되면 자동으로 해결됩니다."
                )
            else:
                # 실제 스키마일 때는 오류로 표시
                self.logger.error(f"템플릿 검증 실패: {len(validation_errors)}개 오류 발견")
                self.logger.error("발견된 오류들:")
                for error in validation_errors[:10]:  # 처음 10개만 표시 (너무 많으면 로그가 과도함)
                    self.logger.error(f"  - {error}")
                if len(validation_errors) > 10:
                    self.logger.error(f"  ... 외 {len(validation_errors) - 10}개 오류 더 있음")
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
