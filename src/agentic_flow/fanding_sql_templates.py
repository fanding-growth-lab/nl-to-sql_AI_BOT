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
from .date_utils import DateUtils

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
            "total_members": SQLTemplate(
                name="전체 회원 수",
                description="크리에이터이름과 날짜를 인자로 받아 전체 회원 수를 추출합니다.",
                sql_template="""
                        SELECT
                            COUNT(DISTINCT f.member_no) AS subscriber_count
                        FROM
                            t_fanding_log AS fl
                        JOIN
                            t_fanding AS f ON fl.fanding_no = f.no
                        JOIN
                            t_creator AS c ON f.creator_no = c.no
                        JOIN
                            t_member AS m ON c.member_no = m.no
                        WHERE
                            m.nickname = '{creator_name}'
                            AND CAST('{target_date}' AS DATE) BETWEEN fl.start_date AND fl.end_date;

                            """,
                parameters=["creator_name", "target_date"],
                analysis_type=FandingAnalysisType.MEMBERSHIP_DATA,
                keywords=["전체 회원 수", "구독자 수", "멤버 수", "크리에이터", "날짜"]
            ),

            "new_monthly_members": SQLTemplate(
                name="월간 신규 회원 수",
                description="크리에이터이름과 날짜를 인자로 받아 신규 회원 수를 추출합니다.",
                sql_template="""
                        SELECT
                            COUNT(DISTINCT f.member_no) AS new_subscriber_count
                        FROM
                            t_fanding_log AS fl
                        JOIN
                            t_fanding AS f ON fl.fanding_no = f.no
                        JOIN
                            t_creator AS c ON f.creator_no = c.no
                        JOIN
                            t_member AS m ON c.member_no = m.no
                        WHERE
                            m.nickname = '{creator_name}'
                            -- 1. 구독 시작일이 해당 월에 포함되는 조건
                            AND DATE_FORMAT(fl.start_date, '%Y-%m') = '{target_month}'
                            -- 2. 시작일로부터 3일 이내에 종료되지 않은 조건
                            AND (fl.end_date IS NULL OR DATEDIFF(fl.end_date, fl.start_date) > 3);
                            """,
                parameters=["creator_name", "target_month"],
                analysis_type=FandingAnalysisType.MEMBERSHIP_DATA,
                keywords=["신규 회원 수", "월간"]
            ),

            "new_weekly_members": SQLTemplate(
                name="주간 신규 회원 수",
                description="크리에이터이름과 날짜를 인자로 받아 신규 회원 수를 추출합니다.",
                sql_template="""
                            SELECT
                                COUNT(DISTINCT f.member_no) AS new_subscriber_count
                            FROM
                                t_fanding_log AS fl
                            JOIN
                                t_fanding AS f ON fl.fanding_no = f.no
                            JOIN
                                t_creator AS c ON f.creator_no = c.no
                            JOIN
                                t_member AS m ON c.member_no = m.no
                            WHERE
                                m.nickname = '{creator_name}'
                                -- 1. 해당 주의 목요일이 조회하려는 연도와 월에 속하는지 확인
                                AND YEAR(fl.start_date + INTERVAL (4 - DAYOFWEEK(fl.start_date + INTERVAL (7 - 2) DAY)) DAY) = {target_year}
                                AND MONTH(fl.start_date + INTERVAL (4 - DAYOFWEEK(fl.start_date + INTERVAL (7 - 2) DAY)) DAY) = {target_month}
                                -- 2. 해당 월의 몇 번째 주인지 계산
                                AND (
                                    WEEK(fl.start_date, 3) - 
                                    WEEK(DATE_FORMAT(fl.start_date, '%Y-%m-01'), 3) + 1
                                ) = {target_week_of_month}
                                -- 3. 시작일로부터 3일 이내에 종료되지 않은 조건
                                AND (fl.end_date IS NULL OR DATEDIFF(fl.end_date, fl.start_date) > 3);
                            """,
                parameters=["creator_name", "target_year", "target_month", "target_week_of_month"],
                analysis_type=FandingAnalysisType.MEMBERSHIP_DATA,
                keywords=["신규 회원 수", "주간"]
            ),

            "new_daily_members": SQLTemplate(
                name="일간 신규 회원 수",
                description="크리에이터이름과 날짜를 인자로 받아 신규 회원 수를 추출합니다.",
                sql_template="""
                            SELECT
                                COUNT(DISTINCT f.member_no) AS new_daily_subscriber_count
                            FROM
                                t_fanding_log AS fl
                            JOIN
                                t_fanding AS f ON fl.fanding_no = f.no
                            JOIN
                                t_creator AS c ON f.creator_no = c.no
                            JOIN
                                t_member AS m ON c.member_no = m.no
                            WHERE
                                m.nickname = '{creator_name}'
                                -- 1. 구독 시작일이 지정된 날짜('target_date')에 포함되는 조건
                                AND DATE(fl.start_date) = '{target_date}'
                                -- 2. 해당 구독이 3일 이내에 종료되지 않은 조건 (여전히 활성이거나, 3일 넘게 지속)
                                AND (fl.end_date IS NULL OR DATEDIFF(fl.end_date, fl.start_date) >= 3);
                            """,
                parameters=["creator_name", "target_date"],
                analysis_type=FandingAnalysisType.MEMBERSHIP_DATA,
                keywords=["신규 회원 수", "일간"]
            ),

            "churn_monthly_members": SQLTemplate(
                name="월간 이탈 회원 수",
                description="크리에이터이름과 날짜를 인자로 받아 이탈 회원 수를 추출합니다.",
                sql_template="""
                        SELECT
                            COUNT(DISTINCT f.member_no) AS monthly_churned_user_count
                        FROM
                            t_fanding_log AS fl_churn
                        JOIN
                            t_fanding AS f ON fl_churn.fanding_no = f.no
                        JOIN
                            t_creator AS c ON f.creator_no = c.no
                        JOIN
                            t_member AS m ON c.member_no = m.no
                        WHERE
                            m.nickname = '{creator_name}'
                            -- 1. 이탈일(end_date)이 대상 월에 속하는지 확인
                            AND DATE_FORMAT(fl_churn.end_date, '%Y-%m') = '{target_month}'
                            -- 2. 이탈 로직: 종료일 이후 3일 이내에 재시작하지 않음
                            AND NOT EXISTS (
                                SELECT 1
                                FROM
                                    t_fanding_log AS fl_restart
                                JOIN
                                    t_fanding AS f_restart ON fl_restart.fanding_no = f_restart.no
                                WHERE
                                    f_restart.member_no = f.member_no
                                    AND f_restart.creator_no = f.creator_no
                                    AND fl_restart.start_date > fl_churn.end_date
                                    AND fl_restart.start_date <= DATE_ADD(fl_churn.end_date, INTERVAL 3 DAY)
                            );
                            """,
                parameters=["creator_name", "target_month"],
                analysis_type=FandingAnalysisType.MEMBERSHIP_DATA,
                keywords=["이탈 회원 수", "월간", "이탈자"]
            ),
            "churn_weekly_members": SQLTemplate(
                name="주간 이탈 회원 수",
                description="크리에이터이름과 날짜를 인자로 받아 이탈 회원 수를 추출합니다.",
                sql_template="""
                            SELECT
                                COUNT(DISTINCT f.member_no) AS weekly_churned_user_count
                            FROM
                                t_fanding_log AS fl_churn
                            JOIN
                                t_fanding AS f ON fl_churn.fanding_no = f.no
                            JOIN
                                t_creator AS c ON f.creator_no = c.no
                            JOIN
                                t_member AS m ON c.member_no = m.no
                            WHERE
                                m.nickname = '{creator_name}'
                                
                                -- 1. 이탈일(end_date)이 대상 주(Week)에 속하는지 확인 (제공된 주간 로직 적용)
                                
                                -- 1-1. 이탈일이 속한 주의 목요일이 조회하려는 연도와 월에 속하는지 확인
                                AND YEAR(fl_churn.end_date + INTERVAL (4 - DAYOFWEEK(fl_churn.end_date + INTERVAL (7 - 2) DAY)) DAY) = {target_year}
                                AND MONTH(fl_churn.end_date + INTERVAL (4 - DAYOFWEEK(fl_churn.end_date + INTERVAL (7 - 2) DAY)) DAY) = {target_month}
                                
                                -- 1-2. 이탈일이 해당 월의 몇 번째 주인지 계산 (WEEK 모드 3 기준)
                                AND (
                                    WEEK(fl_churn.end_date, 3) - 
                                    WEEK(DATE_FORMAT(fl_churn.end_date, '%Y-%m-01'), 3) + 1
                                ) = {target_week_of_month}

                                -- 2. 이탈 로직: 종료일 이후 3일 이내에 재시작하지 않음
                                AND NOT EXISTS (
                                    SELECT 1
                                    FROM
                                        t_fanding_log AS fl_restart
                                    JOIN
                                        t_fanding AS f_restart ON fl_restart.fanding_no = f_restart.no
                                    WHERE
                                        -- 동일한 멤버
                                        f_restart.member_no = f.member_no
                                        -- 동일한 크리에이터
                                        AND f_restart.creator_no = f.creator_no
                                        -- 종료일 이후에 시작
                                        AND fl_restart.start_date > fl_churn.end_date
                                        -- 종료일로부터 3일 이내에 시작
                                        AND fl_restart.start_date <= DATE_ADD(fl_churn.end_date, INTERVAL 3 DAY)
                                );
                            """,
                parameters=["creator_name", "target_year", "target_month", "target_week_of_month"],
                analysis_type=FandingAnalysisType.MEMBERSHIP_DATA,
                keywords=["이탈 회원 수", "주간", "이탈자"]
            ),
            "churn_daily_members": SQLTemplate(
                name="일간 이탈 회원 수",
                description="크리에이터이름과 날짜를 인자로 받아 일간 이탈 회원 수를 추출합니다.",
                sql_template="""
                            SELECT
                                COUNT(DISTINCT f.member_no) AS daily_churned_user_count
                            FROM
                                t_fanding_log AS fl_churn
                            JOIN
                                t_fanding AS f ON fl_churn.fanding_no = f.no
                            JOIN
                                t_creator AS c ON f.creator_no = c.no
                            JOIN
                                t_member AS m ON c.member_no = m.no
                            WHERE
                                m.nickname = '{creator_name}'
                                -- 1. 이탈일(end_date)이 지정된 날짜('target_date')와 일치하는지 확인
                                AND DATE(fl_churn.end_date) = '{target_date}'
                                -- 2. 이탈 로직: 종료일 이후 3일 이내에 재시작하지 않음
                                AND NOT EXISTS (
                                    SELECT 1
                                    FROM
                                        t_fanding_log AS fl_restart
                                    JOIN
                                        t_fanding AS f_restart ON fl_restart.fanding_no = f_restart.no
                                    WHERE
                                        f_restart.member_no = f.member_no
                                        AND f_restart.creator_no = f.creator_no
                                        AND fl_restart.start_date > fl_churn.end_date
                                        AND fl_restart.start_date <= DATE_ADD(fl_churn.end_date, INTERVAL 3 DAY)
                                );
                            """,
                parameters=["creator_name", "target_date"],
                analysis_type=FandingAnalysisType.MEMBERSHIP_DATA,
                keywords=["이탈 회원 수", "일간", "이탈자"]
            ),
            "suspension_monthly_members": SQLTemplate(
                name="월간 중단 예약자 수",
                description="특정 월에 구독 종료가 예약된 회원 수를 집계합니다.",
                sql_template="""
                        WITH filtered_reserves AS (
                          SELECT
                            fl.end_date,
                            f.member_no
                          FROM t_fanding_reserve_log fr
                          JOIN t_fanding f ON fr.fanding_no = f.no
                          JOIN t_fanding_log fl ON f.current_fanding_log_no = fl.no
                          JOIN t_creator c ON f.creator_no = c.no
                          JOIN t_member m ON c.member_no = m.no
                          JOIN t_creator_department_mapping cdm ON f.creator_no = cdm.creator_no
                          JOIN t_creator_department cd ON cdm.department_no = cd.no
                          WHERE
                            fr.status = 'F'
                            AND fr.is_complete = 'F'
                            AND cd.name_eng = 'professional'
                            AND m.nickname = '{creator_name}'
                        )
                        SELECT
                          COUNT(DISTINCT fr.member_no) AS reserved_member_count
                        FROM filtered_reserves fr
                        WHERE DATE_FORMAT(fr.end_date, '%Y-%m') = '{target_month}';
                        """,
                parameters=["creator_name", "target_month"],
                analysis_type=FandingAnalysisType.MEMBERSHIP_DATA,
                keywords=["중단 예약자 수", "월간", "구독 중단"]
            ),
            "suspension_weekly_members": SQLTemplate(
                name="주간 중단 예약자 수",
                description="특정 주에 구독 종료가 예약된 회원 수를 집계합니다.",
                sql_template="""
                        WITH filtered_reserves AS (
                          SELECT
                            fl.end_date,
                            f.member_no
                          FROM t_fanding_reserve_log fr
                          JOIN t_fanding f ON fr.fanding_no = f.no
                          JOIN t_fanding_log fl ON f.current_fanding_log_no = fl.no
                          JOIN t_creator c ON f.creator_no = c.no
                          JOIN t_member m ON c.member_no = m.no
                          JOIN t_creator_department_mapping cdm ON f.creator_no = cdm.creator_no
                          JOIN t_creator_department cd ON cdm.department_no = cd.no
                          WHERE
                            fr.status = 'F'
                            AND fr.is_complete = 'F'
                            AND cd.name_eng = 'professional'
                            AND m.nickname = '{creator_name}'
                        )
                        SELECT
                          COUNT(DISTINCT fr.member_no) AS reserved_member_count
                        FROM filtered_reserves fr
                        WHERE
                            YEAR(fr.end_date + INTERVAL (4 - DAYOFWEEK(fr.end_date + INTERVAL (7 - 2) DAY)) DAY) = {target_year}
                            AND MONTH(fr.end_date + INTERVAL (4 - DAYOFWEEK(fr.end_date + INTERVAL (7 - 2) DAY)) DAY) = {target_month}
                            AND (
                                WEEK(fr.end_date, 3) - 
                                WEEK(DATE_FORMAT(fr.end_date, '%Y-%m-01'), 3) + 1
                            ) = {target_week_of_month};
                        """,
                parameters=["creator_name", "target_year", "target_month", "target_week_of_month"],
                analysis_type=FandingAnalysisType.MEMBERSHIP_DATA,
                keywords=["중단 예약자 수", "주간", "구독 중단"]
            ),
            "suspension_daily_members": SQLTemplate(
                name="일간 중단 예약자 수",
                description="특정 일에 구독 종료가 예약된 회원 수를 집계합니다.",
                sql_template="""
                        WITH filtered_reserves AS (
                          SELECT
                            fl.end_date,
                            f.member_no
                          FROM t_fanding_reserve_log fr
                          JOIN t_fanding f ON fr.fanding_no = f.no
                          JOIN t_fanding_log fl ON f.current_fanding_log_no = fl.no
                          JOIN t_creator c ON f.creator_no = c.no
                          JOIN t_member m ON c.member_no = m.no
                          JOIN t_creator_department_mapping cdm ON f.creator_no = cdm.creator_no
                          JOIN t_creator_department cd ON cdm.department_no = cd.no
                          WHERE
                            fr.status = 'F'
                            AND fr.is_complete = 'F'
                            AND cd.name_eng = 'professional'
                            AND m.nickname = '{creator_name}'
                        )
                        SELECT
                          COUNT(DISTINCT fr.member_no) AS reserved_member_count
                        FROM filtered_reserves fr
                        WHERE DATE(fr.end_date) = '{target_date}';
                        """,
                parameters=["creator_name", "target_date"],
                analysis_type=FandingAnalysisType.MEMBERSHIP_DATA,
                keywords=["중단 예약자 수", "일간", "구독 중단"]
            ),
        }

    
    def _get_performance_templates(self) -> Dict[str, SQLTemplate]:
        """성과 리포트 템플릿"""
        return {
            "monthly_sales": SQLTemplate(
                name="월간 매출 집계",
                description="특정 크리에이터의 월간 매출을 집계합니다.",
                sql_template="""
                        SELECT
                            DATE_FORMAT(v.sales_date, '%Y-%m') AS sales_month,
                            SUM(v.converted_net_price_sum) AS total_sales
                        FROM v_creator_daily_net_sales v
                        JOIN t_creator c ON v.seller_creator_no = c.no
                        JOIN t_member m ON c.member_no = m.no
                        JOIN t_creator_department_mapping cdm ON c.no = cdm.creator_no
                        JOIN t_creator_department cd ON cdm.department_no = cd.no
                        WHERE
                            cd.name_eng = 'professional'
                            AND m.nickname = '{creator_name}'
                            AND DATE_FORMAT(v.sales_date, '%Y-%m') = '{target_month}'
                        GROUP BY sales_month;
                        """,
                parameters=["creator_name", "target_month"],
                analysis_type=FandingAnalysisType.PERFORMANCE_REPORT,
                keywords=["매출", "월간"]
            ),
            "weekly_sales": SQLTemplate(
                name="주간 매출 집계",
                description="특정 크리에이터의 주간 매출을 집계합니다.",
                sql_template="""
                        SELECT
                            YEARWEEK(v.sales_date, 1) AS sales_week,
                            SUM(v.converted_net_price_sum) AS total_sales
                        FROM v_creator_daily_net_sales v
                        JOIN t_creator c ON v.seller_creator_no = c.no
                        JOIN t_member m ON c.member_no = m.no
                        JOIN t_creator_department_mapping cdm ON c.no = cdm.creator_no
                        JOIN t_creator_department cd ON cdm.department_no = cd.no
                        WHERE
                            cd.name_eng = 'professional'
                            AND m.nickname = '{creator_name}'
                            AND YEAR(v.sales_date) = {target_year}
                            AND WEEK(v.sales_date, 1) = {target_week_of_year}
                        GROUP BY sales_week;
                        """,
                parameters=["creator_name", "target_year", "target_week_of_year"],
                analysis_type=FandingAnalysisType.PERFORMANCE_REPORT,
                keywords=["매출", "주간"]
            ),
            "daily_sales": SQLTemplate(
                name="일간 매출 집계",
                description="특정 크리에이터의 일간 매출을 집계합니다.",
                sql_template="""
                        SELECT
                            v.sales_date,
                            SUM(v.converted_net_price_sum) AS total_sales
                        FROM v_creator_daily_net_sales v
                        JOIN t_creator c ON v.seller_creator_no = c.no
                        JOIN t_member m ON c.member_no = m.no
                        JOIN t_creator_department_mapping cdm ON c.no = cdm.creator_no
                        JOIN t_creator_department cd ON cdm.department_no = cd.no
                        WHERE
                            cd.name_eng = 'professional'
                            AND m.nickname = '{creator_name}'
                            AND v.sales_date = '{target_date}'
                        GROUP BY v.sales_date;
                        """,
                parameters=["creator_name", "target_date"],
                analysis_type=FandingAnalysisType.PERFORMANCE_REPORT,
                keywords=["매출", "일간"]
            ),
        }

    def _get_content_templates(self) -> Dict[str, SQLTemplate]:
        """콘텐츠 성과 분석 템플릿"""
        return {}
    
    def _get_advanced_templates(self) -> Dict[str, SQLTemplate]:
        """고급 분석 템플릿"""
        return {}
    
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

    def get_template_by_period(self, creator_name: str, date_query: str) -> Optional[SQLTemplate]:
        """
        자연어 날짜 쿼리를 분석하여 월간/주간/일간 템플릿을 선택하고 파라미터를 채워 반환합니다.
        """
        query_lower = date_query.lower()
        params = {"creator_name": creator_name}
        
        # '이탈', '중단', '예약', '매출' 키워드 확인
        is_churn_query = "이탈" in query_lower
        is_suspension_query = "중단" in query_lower or "예약" in query_lower
        is_sales_query = "매출" in query_lower

        # 기간 단위 및 템플릿 이름 결정
        period_type = None
        if "주간" in query_lower or "주차" in query_lower:
            period_type = "weekly"
            if is_sales_query:
                template_name = "weekly_sales"
            elif is_suspension_query:
                template_name = "suspension_weekly_members"
            else:
                template_name = "churn_weekly_members" if is_churn_query else "new_weekly_members"
        elif "월간" in query_lower or "월" in query_lower:
            period_type = "monthly"
            if is_sales_query:
                template_name = "monthly_sales"
            elif is_suspension_query:
                template_name = "suspension_monthly_members"
            else:
                template_name = "churn_monthly_members" if is_churn_query else "new_monthly_members"
        elif "일간" in query_lower or "일" in query_lower:
            period_type = "daily"
            if is_sales_query:
                template_name = "daily_sales"
            elif is_suspension_query:
                template_name = "suspension_daily_members"
            else:
                template_name = "churn_daily_members" if is_churn_query else "new_daily_members"
        else:
            self.logger.info("날짜 쿼리에서 월간/주간/일간 키워드를 찾을 수 없습니다.")
            return None

        # 날짜 파라미터 추출
        if period_type == "weekly":
            date_info = DateUtils.extract_month_with_year_from_query(date_query)
            if not date_info:
                self.logger.warning("주간 쿼리에서 연/월 정보를 추출할 수 없습니다.")
                return None
            params["target_year"], params["target_month"] = date_info
            week_match = re.search(r'(\d+)\s*주차', query_lower)
            if not week_match:
                self.logger.warning("주간 쿼리에서 'N주차' 정보를 찾을 수 없습니다.")
                return None
            params["target_week_of_month"] = int(week_match.group(1))
        elif period_type == "monthly":
            date_info = DateUtils.extract_month_with_year_from_query(date_query)
            if not date_info:
                self.logger.warning("월간 쿼리에서 연/월 정보를 추출할 수 없습니다.")
                return None
            params["target_month"] = DateUtils.format_date_for_sql(date_info[0], date_info[1])
        elif period_type == "daily":
            # `extract_date_from_query`가 `YYYY-MM-DD` 형식의 문자열을 반환한다고 가정합니다.
            # 이 함수는 date_utils.py에 추가해야 합니다.
            target_date = DateUtils.extract_date_from_query(date_query)
            if not target_date:
                 self.logger.warning("일간 쿼리에서 날짜 정보를 추출할 수 없습니다.")
                 return None
            params["target_date"] = target_date

        # 템플릿 가져오기 및 파라미터 적용
        template = self.get_template(template_name)
        if not template:
            self.logger.error(f"템플릿 '{template_name}'을 찾을 수 없습니다.")
            return None

        try:
            sql_with_params = template.sql_template.format(**params)
            import copy
            param_template = copy.deepcopy(template)
            param_template.sql_template = sql_with_params
            # 템플릿 이름과 설명도 동적으로 포맷팅 (필요 시)
            # param_template.name = param_template.name.format(**params)
            # param_template.description = param_template.description.format(**params)
            return param_template
        except KeyError as e:
            self.logger.error(f"템플릿 '{template_name}'에 필요한 파라미터가 누락되었습니다: {e}")
            return None
    
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
        자연어 쿼리를 적절한 SQL 템플릿에 매칭 (기간 우선, 그 후 키워드 기반)
        """
        query_lower = query.lower()

        # 1. 기간 기반 템플릿 매칭 시도
        if any(keyword in query_lower for keyword in ["월간", "주간", "일간", "주차", "월"]):
            creator_name = self._extract_creator_name_from_query(query)
            if creator_name:
                period_template = self.get_template_by_period(creator_name, query)
                if period_template:
                    self.logger.info(f"Period-based template matched: {period_template.name}")
                    return period_template

        # 2. 기간 기반 매칭 실패 시, 기존 키워드 기반 매칭 시도
        self.logger.info("Falling back to keyword-based template matching.")
        extracted_params = self._extract_parameters_from_query(query)
        best_template = self._find_best_template_by_keywords(query_lower, extracted_params)
        
        return best_template

    def _extract_creator_name_from_query(self, query: str) -> Optional[str]:
        """
        쿼리에서 크리에이터 이름을 추출하는 간단한 헬퍼 함수.
        (예: "'팬딩'의 8월 3주차 신규 회원 수" -> "팬딩")
        """
        # 따옴표 안의 내용을 추출하는 것을 우선으로 함
        match = re.search(r"['\"](.+?)['\"]", query)
        if match:
            return match.group(1)
        
        # "의" 앞에 오는 단어를 크리에이터 이름으로 간주 (간단한 휴리스틱)
        match = re.search(r"(.+?)\s*의", query)
        if match:
            return match.group(1).strip()

        return None
    
    def _find_best_template_by_keywords(self, query_lower: str, extracted_params: Dict[str, Any]) -> Optional[SQLTemplate]:
        """
        키워드 기반 점수 매칭으로 최적의 템플릿 찾기
        """
        template_scores = []
        
        for template_name, template in self.templates.items():
            # 수정된 점수 계산 함수 호출
            score = self._calculate_keyword_score(query_lower, template)
            
            if score > 0:
                template_scores.append((template, score, template_name))
        
        if not template_scores:
            return None
        
        template_scores.sort(key=lambda x: x[1], reverse=True)
        best_template, best_score, best_name = template_scores[0]
        
        # **매칭 임계값을 더 높여서 엄격하게 판단**
        if best_score >= 0.7:
            self.logger.info(f"Template matched with high confidence: '{best_name}' (Score: {best_score:.2f})")
            if extracted_params:
                return self.get_parameterized_template(best_name, extracted_params)
            return best_template
        
        self.logger.info(f"No template matched with high confidence. Best score: {best_score:.2f} for '{best_name}'. Proceeding to general SQL generation.")
        return None
    
    def _calculate_keyword_score(self, query_lower: str, template: SQLTemplate) -> float:
        """
        쿼리와 템플릿 키워드 간의 매칭 점수 계산 (핵심 키워드 조합 강화)
        """
        template_keywords = template.keywords
        if not template_keywords:
            return 0.0

        # **핵심 키워드 정의 (템플릿별로 다르게 설정 가능)**
        # 예: 'new_members_specific_month' 템플릿은 '신규'와 '회원'이 모두 있어야 함
        required_keywords = []
        if template.name == "{month}월 신규 회원":
            required_keywords = ["신규", "회원"]
        elif template.name == "월간 매출 분석":
            required_keywords = ["월간", "매출"]
        
        # **1. 핵심 키워드 검사 (가장 중요)**
        if required_keywords:
            if not all(keyword in query_lower for keyword in required_keywords):
                return 0.0  # 핵심 키워드가 하나라도 없으면 매칭 실패

        # 2. 전체 키워드 매칭 점수 계산 (기존 로직 활용)
        query_words = set(query_lower.split())
        template_words = set([kw.lower() for kw in template_keywords])
        
        intersection = query_words.intersection(template_words)
        union = query_words.union(template_words)

        if not union:
            return 0.0
            
        jaccard_score = len(intersection) / len(union)

        # 3. 핵심 키워드 포함 시 보너스 점수
        bonus = 0.0
        if required_keywords:
            bonus = 0.5 # 핵심 키워드가 모두 존재하면 높은 보너스

        final_score = jaccard_score + bonus
        return min(final_score, 1.0)
    
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
