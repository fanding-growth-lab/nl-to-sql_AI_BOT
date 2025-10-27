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


class FandingSQLTemplates:
    """Fanding Data Report SQL Templates"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.logger = logging.getLogger(__name__)
        # 중앙화된 스키마 정보 로드
        self.db_schema = get_cached_db_schema()
    
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
                analysis_type=FandingAnalysisType.MEMBERSHIP_DATA
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
                       analysis_type=FandingAnalysisType.MEMBERSHIP_DATA
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
                       analysis_type=FandingAnalysisType.MEMBERSHIP_DATA
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
                       analysis_type=FandingAnalysisType.MEMBERSHIP_DATA
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
                analysis_type=FandingAnalysisType.PERFORMANCE_REPORT
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
                analysis_type=FandingAnalysisType.PERFORMANCE_REPORT
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
                analysis_type=FandingAnalysisType.PERFORMANCE_REPORT
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
                analysis_type=FandingAnalysisType.CONTENT_PERFORMANCE
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
                analysis_type=FandingAnalysisType.CONTENT_PERFORMANCE
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
                analysis_type=FandingAnalysisType.CONTENT_PERFORMANCE
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
                analysis_type=FandingAnalysisType.ADVANCED_ANALYSIS
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
                analysis_type=FandingAnalysisType.ADVANCED_ANALYSIS
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
                analysis_type=FandingAnalysisType.ADVANCED_ANALYSIS
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
                analysis_type=FandingAnalysisType.ADVANCED_ANALYSIS
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
                analysis_type=FandingAnalysisType.ADVANCED_ANALYSIS
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
                analysis_type=FandingAnalysisType.ADVANCED_ANALYSIS
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
                analysis_type=FandingAnalysisType.ADVANCED_ANALYSIS
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
            param_template.sql_template = template.sql_template.format(**final_params)
            param_template.name = template.name.format(**final_params)
            param_template.description = template.description.format(**final_params)
            
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
        """Match natural language query to appropriate SQL template with parameter extraction"""
        query_lower = query.lower()
        
        # 파라미터 추출
        extracted_params = self._extract_parameters_from_query(query)
        
        # 1단계: 가장 간단한 쿼리부터 처리
        simple_queries = [
            ('전체 회원 수', 'member_count'),
            ('전체 회원', 'member_count'),
            ('모든 회원', 'member_count'),
            ('활성 회원 수', 'active_member_count'),
            ('활성 회원', 'active_member_count'),
            ('활성회원수', 'active_member_count'),  # 띄어쓰기 없는 버전
            ('활성회원', 'active_member_count'),   # 띄어쓰기 없는 버전
            ('이번 달 신규 회원', 'new_members_this_month'),
            ('이번 달 신규', 'new_members_this_month'),
            ('회원 수', 'member_count'),  # 기본값
            ('멤버 수', 'member_count'),
            ('맴버 수', 'member_count'),
            ('사용자 수', 'member_count'),
            ('가입자 수', 'member_count')
        ]
        
        for keyword, template_name in simple_queries:
            if keyword in query_lower:
                return self.get_template(template_name)
        
        # 멤버십 데이터 관련 키워드
        membership_keywords = [
            '회원', '멤버', '맴버', '멤버십', '맴버쉽', '가입자', '사용자', '리텐션', '유지율', 
            '구독', '기간', '분포', '추이', '증감'
        ]
        
        # 성과 리포트 관련 키워드
        performance_keywords = [
            '매출', '수익', '매출액', '성장률', '방문자', '방문', 
            '수익률', '성과', '실적', '전월', '대비'
        ]
        
        # 콘텐츠 성과 관련 키워드
        content_keywords = [
            '포스트', '게시글', '콘텐츠', '조회', '조회수', '인기', 
            '상위', '순위', '참여', '댓글', '좋아요'
        ]
        
        # 고급 분석 관련 키워드
        advanced_keywords = [
            '수명', '생애', '가치', '중단', '취소', '예약', 
            '비율', '분포', '트렌드', '상관관계', '부서', '엔터',
            '팔로우', '팔로워', '리뷰', '설문', '사유'
        ]
        
        # 키워드 매칭을 통한 템플릿 선택
        if any(keyword in query_lower for keyword in membership_keywords):
            # 신규 회원 관련 특별 처리 (월별 키워드 포함)
            if '신규' in query_lower or '새로운' in query_lower:
                # 특정 월 신규 회원 조회 (4월, 5월 등)
                month_match = re.search(r'(\d+)월', query_lower)
                if month_match:
                    month = int(month_match.group(1))
                    if 1 <= month <= 12:
                        return self.get_parameterized_template("new_members_specific_month", {"month": month})
                # 이번 달 신규 회원
                elif '이번 달' in query_lower or '이번달' in query_lower or '이번 월' in query_lower:
                    return self.get_template("new_members_this_month")
                # 기본 신규 회원 (이번 달)
                else:
                    return self.get_template("new_members_this_month")
            
            # 월별 멤버십 성과 관련 처리 (동적 처리)
            elif any(month in query_lower for month in ['1월', '2월', '3월', '4월', '5월', '6월', 
                                                     '7월', '8월', '9월', '10월', '11월', '12월']):
                # 동적 월별 템플릿 생성
                dynamic_template = self.create_dynamic_monthly_template(query)
                if dynamic_template:
                    return dynamic_template
            # 멤버십 성과 관련 특별 처리 (월별 키워드 포함)
            elif ('멤버십' in query_lower or '맴버쉽' in query_lower) and ('성과' in query_lower or '실적' in query_lower or '분석' in query_lower):
                return self.get_template("monthly_member_trend")  # 기본 월별 멤버십 성과
            elif '회원 수' in query_lower or '전체 회원' in query_lower:
                return self.get_template("member_count")
            elif '월별' in query_lower or '추이' in query_lower:
                return self.get_template("monthly_member_trend")
            elif '리텐션' in query_lower or '유지율' in query_lower:
                return self.get_template("member_retention")
            elif '구독 기간' in query_lower or '분포' in query_lower:
                return self.get_template("subscription_duration_distribution")
        
        elif any(keyword in query_lower for keyword in performance_keywords):
            if '매출' in query_lower and '현황' in query_lower:
                return self.get_template("monthly_revenue")
            elif '매출' in query_lower and '월간' in query_lower:
                return self.get_template("monthly_revenue")
            elif '방문자' in query_lower or '방문' in query_lower:
                return self.get_template("visitor_trend")
            elif '성장률' in query_lower or '증감' in query_lower:
                return self.get_template("revenue_growth_analysis")
        
        elif any(keyword in query_lower for keyword in content_keywords):
            if '인기' in query_lower or 'top' in query_lower or '상위' in query_lower:
                return self.get_template("top_posts")
            elif '참여' in query_lower or '댓글' in query_lower or '좋아요' in query_lower:
                return self.get_template("content_engagement_analysis")
            elif '상관관계' in query_lower or '관계' in query_lower:
                return self.get_template("post_visitor_correlation")
        
        elif any(keyword in query_lower for keyword in advanced_keywords):
            if '수명' in query_lower or '생애' in query_lower or '가치' in query_lower:
                return self.get_template("customer_lifetime_analysis")
            elif '중단' in query_lower or '취소' in query_lower:
                if '설문' in query_lower or '사유' in query_lower:
                    return self.get_template("cancellation_survey_analysis")
                else:
                    return self.get_template("cancellation_analysis")
            elif '부서' in query_lower or '엔터' in query_lower:
                return self.get_template("creator_department_analysis")
            elif '팔로우' in query_lower or '팔로워' in query_lower:
                return self.get_template("follow_analysis")
            elif '리뷰' in query_lower:
                return self.get_template("review_analysis")
            elif '월별' in query_lower and '비교' in query_lower:
                return self.get_template("monthly_performance_comparison")
            elif '성과' in query_lower and '비교' in query_lower:
                return self.get_template("monthly_performance_comparison")
            elif '최근' in query_lower and '성과' in query_lower:
                return self.get_template("monthly_performance_comparison")
            elif '기간' in query_lower and '분포' in query_lower:
                return self.get_template("subscription_duration_distribution")
        
        # 추가 매칭 로직
        if '취소' in query_lower or '중단' in query_lower:
            return self.get_template("cancellation_analysis")
        elif '월별' in query_lower and '비교' in query_lower:
            template = self.get_template("monthly_performance_comparison")
        elif '최근' in query_lower and ('성과' in query_lower or '비교' in query_lower):
            template = self.get_template("monthly_performance_comparison")
        else:
            template = None
        
        # 동적 연도 처리 적용
        if template:
            template = self._apply_dynamic_year_to_template(query, template)
            
            # 파라미터화된 템플릿 적용
            if extracted_params:
                template = self.get_parameterized_template(template.name, extracted_params)
        
        return template
    
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
        """동적으로 월별 멤버십 성과 템플릿 생성"""
        try:
            from .date_utils import DateUtils
            
            # 쿼리에서 월 추출
            month = DateUtils.extract_month_from_query(query)
            if not month:
                return None
                
            # 월을 두 자리 숫자로 변환 (예: "9월" -> "09")
            month_num = month.split('-')[1] if '-' in month else month
            if len(month_num) == 1:
                month_num = f"0{month_num}"
            
            # 동적 SQL 템플릿 생성 (조인을 활용한 월별 필터링)
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
            WHERE DATE_FORMAT(l.ins_datetime, '%Y-%m') = CONCAT(YEAR(NOW()), '-{month_num}')
            """
            
            return SQLTemplate(
                name=f"{month_num}월 멤버십 성과 분석",
                description=f"{month_num}월 멤버십 성과 상세 분석",
                sql_template=sql_template,
                parameters=[],
                analysis_type=FandingAnalysisType.MEMBERSHIP_DATA
            )
        except Exception as e:
            print(f"동적 월별 템플릿 생성 실패: {str(e)}")
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
        
        return params
