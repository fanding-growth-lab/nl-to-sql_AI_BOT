"""
Fanding Data Report System SQL Templates

This module contains SQL templates for various Fanding Data Report analysis features.
"""

from typing import Dict, List, Optional, Any, Tuple
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
    match_score: Optional[float] = None  # 템플릿 매칭 점수 (0.0 ~ 1.0, None이면 미매칭)


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
    
    def _has_column(self, table_name: str, column_name: str) -> bool:
        """
        테이블에 특정 컬럼이 있는지 확인
        
        Args:
            table_name: 테이블명
            column_name: 컬럼명
            
        Returns:
            컬럼 존재 여부
        """
        if not self.db_schema or table_name not in self.db_schema:
            return False
        table_info = self.db_schema[table_name]
        columns = table_info.get("columns", {})
        return column_name in columns
    
    def _validate_and_fix_template_sql(self, sql_template: str) -> str:
        """
        템플릿 SQL을 db_schema 기반으로 검증하고 필요시 수정
        
        주의: 템플릿은 이미 올바른 테이블명을 사용하도록 작성되어 있으므로,
        이 메서드는 특수한 경우에만 컬럼명 수정을 수행합니다.
        
        Args:
            sql_template: 원본 SQL 템플릿
            
        Returns:
            검증/수정된 SQL 템플릿
        """
        if not self.db_schema or len(self.db_schema) == 0:
            return sql_template
        
        import re
        fixed_sql = sql_template
        
        # t_member_info 테이블 사용 시 컬럼명 자동 수정 (m.no -> m.member_no)
        # t_member_info에는 'no' 컬럼이 없고 'member_no'가 있음
        if 't_member_info' in fixed_sql.lower():
            if not self._has_column('t_member_info', 'no') and self._has_column('t_member_info', 'member_no'):
                fixed_sql = re.sub(r'\bm\.no\b', 'm.member_no', fixed_sql, flags=re.IGNORECASE)
                if fixed_sql != sql_template:
                    self.logger.debug("자동 수정: m.no -> m.member_no (t_member_info 사용)")
            
            # status 컬럼이 없으면 status 관련 컬럼을 제거
            if not self._has_column('t_member_info', 'status') and 'm.status' in fixed_sql.lower():
                # status 관련 CASE 문 제거
                fixed_sql = re.sub(
                    r',\s*COUNT\(DISTINCT CASE WHEN m\.status\s*=\s*[\'"]?[AID][\'"]?\s*THEN f\.member_no END\)\s*as\s*\w+',
                    '',
                    fixed_sql,
                    flags=re.IGNORECASE
                )
                # status 관련 ROUND 계산 제거
                fixed_sql = re.sub(
                    r',\s*ROUND\([^)]*m\.status[^)]*,\s*\d+\)\s*as\s*\w+',
                    '',
                    fixed_sql,
                    flags=re.IGNORECASE
                )
                # status 관련 컬럼명 제거 (active_members, inactive_members 등)
                fixed_sql = re.sub(
                    r',\s*(active_members|inactive_members|deleted_members|active_rate_percent|inactive_rate_percent|deletion_rate_percent)',
                    '',
                    fixed_sql,
                    flags=re.IGNORECASE
                )
                if fixed_sql != sql_template:
                    self.logger.warning("t_member_info에 status 컬럼이 없어 status 관련 컬럼을 제거했습니다.")
        
        return fixed_sql
    
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
                keywords=["매출", "월간", "정산", "정산금액", "결제", "수익", "매출액", "금액", "수익금", "sales", "revenue", "payment", "settlement"]
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
                keywords=["매출", "주간", "정산", "정산금액", "결제", "수익", "매출액", "금액", "수익금", "sales", "revenue", "payment", "settlement"]
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
                keywords=["매출", "일간", "정산", "정산금액", "결제", "수익", "매출액", "금액", "수익금", "sales", "revenue", "payment", "settlement"]
            ),
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
            # 주의: 템플릿에는 '{param}' 형태로 파라미터가 정의되어 있음
            sql_with_params = template.sql_template
            
            # month 파라미터 처리
            if 'month' in final_params:
                month_val = final_params['month']
                sql_with_params = sql_with_params.replace('{month:02d}', f"{month_val:02d}")
                sql_with_params = sql_with_params.replace('{month}', str(month_val))
            
            # 나머지 파라미터는 '{param}' 형태로 템플릿에 정의되어 있으므로 replace로 처리
            other_params = {k: v for k, v in final_params.items() if k != 'month'}
            for param_name, param_value in other_params.items():
                # '{param_name}' 형태를 '{value}'로 치환 (따옴표 포함)
                # 예: '{creator_name}' -> '강환국'
                old_pattern_quoted = "'{" + param_name + "}'"
                new_value_quoted = f"'{param_value}'" if isinstance(param_value, str) else f"'{param_value}'"
                sql_with_params = sql_with_params.replace(old_pattern_quoted, new_value_quoted)
                
                # {param_name} 형태도 처리 (따옴표 없는 경우)
                # 예: {target_year} -> 2025
                old_pattern_unquoted = "{" + param_name + "}"
                new_value_unquoted = str(param_value)
                sql_with_params = sql_with_params.replace(old_pattern_unquoted, new_value_unquoted)
            
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
    
    def get_template_by_period(self, creator_name: str, date_query: str, original_query: Optional[str] = None) -> Optional[SQLTemplate]:
        """
        자연어 날짜 쿼리를 분석하여 월간/주간/일간 템플릿을 선택하고 파라미터를 채워 반환합니다.
        
        Args:
            creator_name: 크리에이터 이름
            date_query: 날짜 관련 쿼리
            original_query: 원본 쿼리 (크리에이터 이름 검증 시 사용)
            
        Returns:
            파라미터가 적용된 SQLTemplate 또는 None
        """
        query_lower = date_query.lower()
        params = {"creator_name": creator_name}
        
        # '이탈', '중단', '예약', '매출', '정산' 키워드 확인
        is_churn_query = any(keyword in query_lower for keyword in ['이탈', '탈퇴', 'churn', '취소', '해지'])
        is_suspension_query = "중단" in query_lower or "예약" in query_lower
        # 매출 관련 키워드: 매출, 정산, 정산금액, 결제, 수익 등
        is_sales_query = any(keyword in query_lower for keyword in [
            '매출', '정산', '정산금액', '결제', '수익', '매출액', 'revenue', 'payment', 'pay', 
            'remain_price', '금액', '수익금', 'sales', 'settlement'
        ])

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
        elif "일간" in query_lower or ("일" in query_lower and re.search(r'\d+\s*일', query_lower)):
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
            
            # 주간 매출 템플릿은 target_week_of_year를 사용
            if is_sales_query:
                # 연도와 주차 정보 추출 (예: "2025년 3주차")
                year_match = re.search(r'(\d{4})\s*년', query_lower)
                if year_match:
                    params["target_year"] = int(year_match.group(1))
                
                week_match = re.search(r'(\d+)\s*주차', query_lower)
                if week_match:
                    params["target_week_of_year"] = int(week_match.group(1))
                else:
                    # 월의 몇 번째 주인지 계산
                    from datetime import datetime
                    month = params["target_month"]
                    # 해당 월의 첫 번째 날짜를 기준으로 주차 계산
                    first_day = datetime(params["target_year"], month, 1)
                    week_num = first_day.isocalendar()[1]
                    params["target_week_of_year"] = week_num
            else:
                # 주간 회원 템플릿은 target_week_of_month를 사용
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
            date_info = DateUtils.extract_date_from_query(date_query)
            if not date_info:
                self.logger.warning("일간 쿼리에서 날짜 정보를 추출할 수 없습니다.")
                return None
            year, month, day = date_info
            if day is None:
                self.logger.warning("일간 쿼리에서 일 정보를 추출할 수 없습니다.")
                return None
            params["target_date"] = f"{year}-{month:02d}-{day:02d}"

        # 크리에이터 이름 DB 검증 (SQL Injection 방지 + 유사도 검색)
        # 중요: 검증 실패 시 템플릿을 반환하지 않음 (존재하지 않는 크리에이터에 대한 쿼리 방지)
        validated_creator_name = self._validate_and_find_creator_name(creator_name, original_query=original_query or date_query)
        if not validated_creator_name:
            self.logger.warning(
                f"Creator name '{creator_name}' not found in database or validation failed. "
                f"Template will not be returned to prevent querying non-existent creator."
            )
            # 검증 실패 시 원본 이름 사용하지 않고 None 반환 (안전)
            return None
        
        # 검증된 정확한 nickname 사용
        params["creator_name"] = validated_creator_name
        self.logger.info(f"Validated creator name: '{creator_name}' -> '{validated_creator_name}'")

        # 템플릿 가져오기 및 파라미터 적용
        template = self.get_template(template_name)
        if not template:
            self.logger.error(f"템플릿 '{template_name}'을 찾을 수 없습니다.")
            return None

        try:
            # 템플릿에는 '{param}' 형태로 파라미터가 정의되어 있으므로 replace로 처리
            sql_with_params = template.sql_template
            for param_name, param_value in params.items():
                # '{param_name}' 형태를 '{value}'로 치환 (따옴표 포함)
                # 예: '{creator_name}' -> '세상학 개론'
                # f-string에서 중괄호를 리터럴로 사용하려면 {{ 를 사용해야 하지만,
                # 여기서는 동적 변수이므로 문자열 연결 사용
                old_pattern_quoted = "'{" + param_name + "}'"
                new_value_quoted = f"'{param_value}'" if isinstance(param_value, str) else f"'{param_value}'"
                sql_with_params = sql_with_params.replace(old_pattern_quoted, new_value_quoted)
                
                # {param_name} 형태도 처리 (따옴표 없는 경우)
                # 예: {target_year} -> 2025
                old_pattern_unquoted = "{" + param_name + "}"
                new_value_unquoted = str(param_value)
                sql_with_params = sql_with_params.replace(old_pattern_unquoted, new_value_unquoted)
            
            import copy
            param_template = copy.deepcopy(template)
            param_template.sql_template = sql_with_params
            self.logger.debug(f"Applied parameters to template '{template_name}': {params}")
            return param_template
        except KeyError as e:
            self.logger.error(f"템플릿 '{template_name}'에 필요한 파라미터가 누락되었습니다: {e}")
            return None
    
    def _extract_creator_name_from_query(self, query: str) -> Optional[str]:
        """
        쿼리에서 크리에이터 이름을 추출하는 간단한 헬퍼 함수.
        (예: "'팬딩'의 8월 3주차 신규 회원 수" -> "팬딩")
        
        Args:
            query: 사용자 쿼리
            
        Returns:
            추출된 크리에이터 이름 또는 None
        """
        # 따옴표 안의 내용을 추출하는 것을 우선으로 함
        match = re.search(r"['\"](.+?)['\"]", query)
        if match:
            creator_name = match.group(1)
            # "크리에이터" 단어 제거
            creator_name = re.sub(r'\s*크리에이터\s*', '', creator_name, flags=re.IGNORECASE).strip()
            return creator_name if creator_name else None
        
        # "작가의" 패턴으로 추출 (예: "강환국작가의" -> "강환국")
        match = re.search(r"(.+?)작가\s*의", query, re.IGNORECASE)
        if match:
            creator_name = match.group(1).strip()
            # "작가" 단어 제거
            creator_name = re.sub(r'\s*작가\s*', '', creator_name, flags=re.IGNORECASE).strip()
            if creator_name:
                return creator_name
        
        # "크리에이터의" 또는 "의" 패턴으로 추출
        # "세상학개론 크리에이터의" -> "세상학개론"
        match = re.search(r"(.+?)\s*크리에이터\s*의", query, re.IGNORECASE)
        if match:
            creator_name = match.group(1).strip()
            # 추가로 "크리에이터" 단어가 포함되어 있으면 제거
            creator_name = re.sub(r'\s*크리에이터\s*', '', creator_name, flags=re.IGNORECASE).strip()
            return creator_name if creator_name else None
        
        # "의" 앞에 오는 단어를 크리에이터 이름으로 간주 (간단한 휴리스틱)
        # 다만 "10월", "2024년" 같은 날짜 패턴은 제외
        match = re.search(r"(.+?)\s*의", query)
        if match:
            creator_name = match.group(1).strip()
            
            # 숫자+월 패턴이 앞에 있으면 제거 (예: "10월 세상학 개론" -> "세상학 개론")
            creator_name = re.sub(r'^\d+\s*월\s*', '', creator_name).strip()
            creator_name = re.sub(r'^\d+\s*년\s*', '', creator_name).strip()
            
            # "크리에이터" 단어 제거
            creator_name = re.sub(r'\s*크리에이터\s*', '', creator_name, flags=re.IGNORECASE).strip()
            
            # 회사 이름 제외: "팬딩", "fanding"은 회사 이름이므로 크리에이터 이름이 아님
            # 회사 전체 데이터를 조회하는 쿼리로 처리해야 함
            company_names = ['팬딩', 'fanding', '펜딩', '펀딩']
            if creator_name.lower() in [name.lower() for name in company_names]:
                # 회사 이름이면 크리에이터 이름으로 추출하지 않음
                return None
            
            # 숫자만 있는 경우 제외 (예: "10월의" -> "10월"이 추출되는 경우)
            if creator_name and not re.match(r'^\d+\s*(?:월|년|일|개|명|위)$', creator_name):
                return creator_name

        return None
    
    def _validate_and_find_creator_name(self, creator_name: str, original_query: Optional[str] = None) -> Optional[str]:
        """
        크리에이터 이름을 DB에서 검증하고 정확한 nickname을 반환 (SQL Injection 방지)
        
        Args:
            creator_name: 사용자 입력 크리에이터 이름 (예: "세상학 개론", "세상학개론")
            original_query: 원본 쿼리 (부분 매칭 우선순위 결정에 사용)
            
        Returns:
            DB에서 찾은 정확한 nickname 또는 None (검증 실패 시)
        """
        from core.db import execute_query
        from difflib import SequenceMatcher
        
        if not creator_name or len(creator_name.strip()) < 2:
            return None
        
        creator_name = creator_name.strip()
        
        try:
            # 1. 정확한 매칭 시도
            exact_query = """
                SELECT m.nickname
                FROM t_creator c
                INNER JOIN t_member m ON c.member_no = m.no
                WHERE m.nickname = :creator_name
                LIMIT 1
            """
            exact_results = execute_query(exact_query, {"creator_name": creator_name}, readonly=True)
            
            # 결과 검증 강화
            if exact_results is None:
                self.logger.warning(f"Query execution returned None for creator name: '{creator_name}'")
            elif not isinstance(exact_results, list):
                self.logger.warning(f"Query execution returned unexpected type: {type(exact_results)} for creator name: '{creator_name}'")
            elif len(exact_results) > 0:
                validated_nickname = exact_results[0].get("nickname")
                if validated_nickname:
                    # 정확한 매칭이 있더라도, 원본 쿼리에 "작가"가 포함되어 있고
                    # 매칭된 이름에 "작가"가 없으면 부분 매칭도 시도 (더 정확한 이름 찾기)
                    # 예: "강환국작가" → "강환국" (정확 매칭) vs "강환국 작가" (부분 매칭, 더 정확)
                    should_try_partial = False
                    if original_query and '작가' in original_query and '작가' not in validated_nickname:
                        # 원본 쿼리에 "작가"가 있고, 매칭된 이름에 "작가"가 없으면 부분 매칭 시도
                        should_try_partial = True
                        self.logger.debug(
                            f"Exact match found '{validated_nickname}' but original query contains '작가' and matched name doesn't. "
                            f"Will also try partial match to find more specific name like '{creator_name} 작가'"
                        )
                    
                    if not should_try_partial:
                        self.logger.info(f"Exact match found for creator name: '{creator_name}' -> '{validated_nickname}'")
                        return validated_nickname
                    # should_try_partial이 True면 부분 매칭도 시도하기 위해 계속 진행
                else:
                    self.logger.warning(f"Query result missing 'nickname' field for creator name: '{creator_name}'")
            else:
                self.logger.debug(f"No exact match found for creator name: '{creator_name}' (0 results)")
            
            # 2. 부분 매칭 시도 (LIKE 사용, 파라미터 바인딩으로 SQL Injection 방지)
            # 띄어쓰기 문제 해결: 띄어쓰기 있는 버전과 없는 버전 모두 검색
            partial_query = """
                SELECT m.nickname
                FROM t_creator c
                INNER JOIN t_member m ON c.member_no = m.no
                WHERE m.nickname LIKE :creator_pattern OR m.nickname LIKE :creator_pattern_no_space
                LIMIT 20
            """
            # 부분 매칭: 원본과 띄어쓰기 제거 버전 모두 검색
            creator_pattern = f"%{creator_name}%"
            creator_pattern_no_space = f"%{creator_name.replace(' ', '')}%"  # 띄어쓰기 제거
            partial_results = execute_query(
                partial_query, 
                {
                    "creator_pattern": creator_pattern,
                    "creator_pattern_no_space": creator_pattern_no_space
                }, 
                readonly=True
            )
            
            # 결과 검증 강화
            if partial_results is None:
                self.logger.warning(f"Partial query execution returned None for creator name: '{creator_name}'")
            elif not isinstance(partial_results, list):
                self.logger.warning(f"Partial query execution returned unexpected type: {type(partial_results)} for creator name: '{creator_name}'")
            elif len(partial_results) > 0:
                # 유사도 계산을 위한 정규화 함수
                def normalize_for_similarity(text: str) -> str:
                    """유사도 계산을 위해 텍스트 정규화 (띄어쓰기, 하이픈, 특수문자 제거)"""
                    import re
                    normalized = re.sub(r'[\s\-_\-]', '', text.lower())
                    return normalized
                
                # 여러 결과가 있으면 유사도로 정렬
                if len(partial_results) > 1:
                    scored_results = []
                    normalized_creator_name = normalize_for_similarity(creator_name)
                    
                    self.logger.debug(f"Found {len(partial_results)} partial matches for '{creator_name}': {[r.get('nickname', 'N/A') for r in partial_results]}")
                    
                    for result in partial_results:
                        normalized_nickname = normalize_for_similarity(result["nickname"])
                        similarity = SequenceMatcher(None, normalized_creator_name, normalized_nickname).ratio()
                        
                        # 원본에 포함되어 있으면 가산점 (가장 중요한 조건)
                        # 예: "세상학개론"이 "세상학개론 - 한정수 작가"에 포함되어 있으면 similarity >= 0.7
                        if creator_name.lower() in result["nickname"].lower() or creator_name.replace(' ', '').lower() in result["nickname"].lower():
                            similarity = max(similarity, 0.7)
                            self.logger.debug(
                                f"Creator name '{creator_name}' is contained in '{result['nickname']}', "
                                f"similarity boosted to {similarity:.2f}"
                            )
                        
                        scored_results.append({
                            "nickname": result["nickname"],
                            "similarity": similarity
                        })
                    
                    # 유사도로 정렬
                    scored_results.sort(key=lambda x: x["similarity"], reverse=True)
                    best_match = scored_results[0]
                    
                    # 상위 3개 결과 로깅 (디버깅용)
                    self.logger.debug(f"Top 3 matches for '{creator_name}':")
                    for i, match in enumerate(scored_results[:3], 1):
                        self.logger.debug(f"  {i}. '{match['nickname']}' (similarity: {match['similarity']:.2f})")
                    
                    # 원본 쿼리에 "작가"가 있으면 "작가"가 포함된 이름을 우선 선택
                    if original_query and '작가' in original_query:
                        # "작가"가 포함된 결과 중 가장 높은 유사도 선택
                        작가_matches = [m for m in scored_results if '작가' in m['nickname']]
                        if 작가_matches:
                            best_match = 작가_matches[0]
                            self.logger.debug(
                                f"Original query contains '작가', prioritizing match with '작가': "
                                f"'{best_match['nickname']}' (similarity: {best_match['similarity']:.2f})"
                            )
                    
                    # 유사도 임계값 0.7 (더 엄격하게 변경)
                    if best_match["similarity"] >= 0.7:
                        validated_nickname = best_match["nickname"]
                        self.logger.info(f"Partial match found for creator name: '{creator_name}' -> '{validated_nickname}' (similarity: {best_match['similarity']:.2f})")
                        return validated_nickname
                    else:
                        self.logger.warning(
                            f"Partial match found but similarity too low for creator name: '{creator_name}' -> "
                            f"'{best_match['nickname']}' (similarity: {best_match['similarity']:.2f}, threshold: 0.7)"
                        )
                else:
                    # 단일 결과
                    result = partial_results[0]
                    self.logger.debug(f"Found single partial match for '{creator_name}': '{result.get('nickname', 'N/A')}'")
                    
                    normalized_creator_name = normalize_for_similarity(creator_name)
                    normalized_nickname = normalize_for_similarity(result["nickname"])
                    similarity = SequenceMatcher(None, normalized_creator_name, normalized_nickname).ratio()
                    
                    # 원본에 포함되어 있으면 가산점 (가장 중요한 조건)
                    # 예: "세상학개론"이 "세상학개론 - 한정수 작가"에 포함되어 있으면 similarity >= 0.7
                    if creator_name.lower() in result["nickname"].lower() or creator_name.replace(' ', '').lower() in result["nickname"].lower():
                        similarity = max(similarity, 0.7)
                        self.logger.debug(
                            f"Creator name '{creator_name}' is contained in '{result['nickname']}', "
                            f"similarity boosted to {similarity:.2f}"
                        )
                    
                    # 원본 쿼리에 "작가"가 있으면 "작가"가 포함된 이름에 추가 가산점
                    if original_query and '작가' in original_query and '작가' in result["nickname"]:
                        similarity = max(similarity, 0.8)  # "작가" 포함 시 더 높은 우선순위
                        self.logger.debug(
                            f"Original query contains '작가' and matched name also contains '작가', "
                            f"similarity boosted to {similarity:.2f}"
                        )
                    
                    # 유사도 임계값 0.7 (더 엄격하게 변경)
                    if similarity >= 0.7:
                        validated_nickname = result["nickname"]
                        self.logger.info(f"Partial match found for creator name: '{creator_name}' -> '{validated_nickname}' (similarity: {similarity:.2f})")
                        return validated_nickname
                    else:
                        self.logger.warning(
                            f"Partial match found but similarity too low for creator name: '{creator_name}' -> "
                            f"'{result['nickname']}' (similarity: {similarity:.2f}, threshold: 0.7)"
                        )
            else:
                self.logger.debug(f"No partial match found for creator name: '{creator_name}'")
            
            self.logger.warning(f"No creator found for name: '{creator_name}' (exact and partial match both failed)")
            return None
                
        except Exception as e:
            self.logger.error(f"Error validating creator name '{creator_name}': {e}")
            return None
    
    def match_query_to_template(self, query: str) -> Optional[SQLTemplate]:
        """
        자연어 쿼리를 적절한 SQL 템플릿에 매칭 (기간 기반 우선, 그 후 키워드 기반)
        
        쿼리에서 파라미터(예: 월 정보)를 추출하여 매칭된 템플릿의 파라미터를 자동으로 채웁니다.
        
        Args:
            query: 사용자 쿼리
            
        Returns:
            매칭된 SQLTemplate (파라미터 적용됨) 또는 None
        """
        query_lower = query.lower()

        # 1. 기간 기반 템플릿 매칭 시도 (새 로직, 우선순위 높음)
        # 일간 매칭을 먼저 확인 (더 구체적) - "월 일일" 또는 "월 일" 패턴 확인
        if "일간" in query_lower or (re.search(r'\d+\s*월\s*\d+\s*일', query_lower) or re.search(r'\d+\s*일', query_lower)):
            creator_name = self._extract_creator_name_from_query(query)
            if creator_name:
                period_template = self.get_template_by_period(creator_name, query, original_query=query)
                if period_template:
                    self.logger.info(f"Period-based template matched: {period_template.name}")
                    return period_template
        # 주간 매칭
        elif "주간" in query_lower or "주차" in query_lower:
            creator_name = self._extract_creator_name_from_query(query)
            if creator_name:
                period_template = self.get_template_by_period(creator_name, query, original_query=query)
                if period_template:
                    self.logger.info(f"Period-based template matched: {period_template.name}")
                    return period_template
        # 월간 매칭 (일간 패턴이 아닌 경우만)
        elif "월간" in query_lower or ("월" in query_lower and not re.search(r'\d+\s*일', query_lower)):
            # 매출/정산 관련 쿼리인지 먼저 확인
            is_sales_query = any(keyword in query_lower for keyword in [
                '매출', '정산', '정산금액', '결제', '수익', '매출액', 'revenue', 'payment', 'pay', 
                'remain_price', '금액', '수익금', 'sales', 'settlement'
            ])
            
            # 매출/정산 쿼리인데 크리에이터 이름이 없으면 회사 전체 데이터 조회
            creator_name = self._extract_creator_name_from_query(query)
            if not creator_name and is_sales_query:
                # 회사 전체 정산금액 조회 - 크리에이터 필터 없이 전체 데이터 조회
                self.logger.info("회사 전체 정산금액 조회 쿼리로 인식, 크리에이터 필터 없이 처리")
                # 템플릿 매칭을 건너뛰고 LLM 기반 SQL 생성으로 진행
                return None
            
            if creator_name:
                period_template = self.get_template_by_period(creator_name, query, original_query=query)
                if period_template:
                    self.logger.info(f"Period-based template matched: {period_template.name}")
                    return period_template

        # 2. 기간 기반 매칭 실패 시, 기존 키워드 기반 매칭 시도
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
        best_template, best_template_name = self._find_best_template_by_keywords(query_lower, extracted_params)
        
        # 크리에이터 이름이 필요한 템플릿인 경우 검증
        if best_template and has_creator_keyword:
            # 크리에이터 이름 추출 및 검증
            creator_name = self._extract_creator_name_from_query(query)
            if creator_name:
                validated_creator_name = self._validate_and_find_creator_name(creator_name, original_query=query)
                if not validated_creator_name:
                    self.logger.warning(f"Creator name '{creator_name}' validation failed for template '{best_template_name}', skipping template")
                    return None
                # 검증된 크리에이터 이름을 파라미터에 추가
                if not extracted_params:
                    extracted_params = {}
                extracted_params["creator_name"] = validated_creator_name
        
        # 매칭된 템플릿이 있고 파라미터가 있으면 적용
        if best_template and extracted_params:
            # 파라미터가 필요한 템플릿인지 확인
            if best_template.parameters:
                # 원본 템플릿의 매칭 점수 저장 (파라미터 적용 후에도 유지하기 위해)
                original_match_score = getattr(best_template, 'match_score', None)
                param_template = self.get_parameterized_template(
                    best_template_name, 
                    extracted_params
                )
                if param_template:
                    # 파라미터 적용된 템플릿에도 원본 매칭 점수 유지
                    if original_match_score is not None:
                        param_template.match_score = original_match_score
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
    
    def _find_best_template_by_keywords(self, query_lower: str, extracted_params: Dict[str, Any]) -> Tuple[Optional[SQLTemplate], str]:
        """
        키워드 기반 점수 매칭으로 최적의 템플릿 찾기
        
        Args:
            query_lower: 소문자로 변환된 쿼리
            extracted_params: 추출된 파라미터
            
        Returns:
            (최고 점수를 받은 템플릿 또는 None, 템플릿 이름)
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
            return None, ""
        
        # 점수순으로 정렬하여 최고 점수 템플릿 선택
        template_scores.sort(key=lambda x: x[1], reverse=True)
        best_template, best_score, best_name = template_scores[0]
        
        # 최소 임계점 이상인 경우만 반환
        if best_score >= 0.3:  # 30% 이상 매칭
            self.logger.debug(f"키워드 매칭: '{best_name}' (점수: {best_score:.2f})")
            
            # 템플릿 복사 및 매칭 점수 설정
            from copy import deepcopy
            template_with_score = deepcopy(best_template)
            template_with_score.match_score = best_score
            
            # 파라미터가 있으면 적용 (매칭된 템플릿의 파라미터를 동적으로 채움)
            if extracted_params and best_template.parameters:
                param_template = self.get_parameterized_template(best_name, extracted_params)
                # get_parameterized_template 내부에서 이미 db_schema 기반 검증/수정이 수행됨
                if param_template:
                    # 파라미터 적용된 템플릿에도 점수 설정
                    param_template.match_score = best_score
                    self.logger.debug(f"템플릿 '{best_name}' 파라미터 적용 완료: {extracted_params}")
                    return param_template, best_name
                else:
                    # 파라미터 적용 실패 시 점수가 설정된 원본 템플릿 반환
                    self.logger.warning(f"템플릿 '{best_name}' 파라미터 적용 실패, 원본 템플릿 반환")
                    return template_with_score, best_name
            else:
                # 파라미터가 없거나 템플릿에 파라미터가 필요 없으면 점수가 설정된 원본 템플릿 반환
                # db_schema 기반 검증/수정은 match_query_to_template에서 처리됨
                return template_with_score, best_name
        
        return None, ""
    
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
            # 쿼리에서 월 정보 추출
            month_info = self._extract_month_info(query)
            if not month_info:
                return None
            
            # 쿼리 의도 분석
            intent = self._analyze_template_intent(query)
            
            # 스키마 검증
            if not self._validate_schema_for_dynamic_template():
                return None
            
            # 멤버 테이블 정보 찾기
            member_info = self._find_member_table_info()
            if not member_info:
                return None
            
            year, month = month_info
            month_num = f"{month:02d}"
            yyyy_mm = f"{year}-{month_num}"
            
            # 쿼리 의도에 따라 템플릿 생성
            intent_type = intent.get('type')
            if intent_type == "sales":
                # 매출 관련 쿼리는 동적 템플릿 생성하지 않음 (기존 템플릿 사용)
                self.logger.info("매출 관련 쿼리는 동적 템플릿으로 생성하지 않습니다. 기존 템플릿을 사용하세요.")
                return None
            elif intent_type == "churn_members":
                # 이탈 회원 쿼리는 동적 템플릿 생성하지 않음 (기존 템플릿 사용)
                self.logger.info("이탈 회원 관련 쿼리는 동적 템플릿으로 생성하지 않습니다. 기존 템플릿을 사용하세요.")
                return None
            elif intent_type == "new_members":
                return self._create_new_members_template(
                    month_num, yyyy_mm, year, month, 
                    intent.get('has_creator_keyword', False)
                )
            else:
                return self._create_membership_performance_template(
                    month_num, yyyy_mm, year, month, member_info
                )
        except Exception as e:
            self.logger.error(f"동적 월별 템플릿 생성 실패: {str(e)}")
            return None
    
    def _extract_month_info(self, query: str) -> Optional[Tuple[int, int]]:
        """쿼리에서 월 정보 추출"""
        from .date_utils import DateUtils
        return DateUtils.extract_month_with_year_from_query(query)
    
    def _analyze_template_intent(self, query: str) -> Dict[str, Any]:
        """쿼리 의도 분석: 신규 회원수 vs 이탈 회원수 vs 멤버십 성과 분석 vs 매출"""
        query_lower = query.lower()
        
        # 매출/결제 키워드 확인 (최우선)
        has_payment_keywords = any(keyword in query_lower for keyword in [
            '매출', '결제', '수익', '매출액', 'revenue', 'payment', 'pay', 'remain_price', '금액', '수익금', 'sales'
        ])
        if has_payment_keywords:
            # 매출 관련 쿼리는 동적 템플릿으로 생성하지 않음 (기존 템플릿 사용)
            return {"type": "sales", "has_creator_keyword": False}
        
        # 이탈/탈퇴 키워드 확인 (신규 회원보다 우선)
        has_churn_keywords = any(keyword in query_lower for keyword in [
            '이탈', '탈퇴', 'churn', '취소', '중단', '해지', '이탈자', '탈퇴자', '이탈 회원', '탈퇴 회원'
        ])
        if has_churn_keywords:
            has_creator_keyword = (
                '크리에이터' in query_lower or 
                'creator' in query_lower or
                any(keyword in query_lower for keyword in ['작가', '아티스트', '제작자'])
            )
            return {"type": "churn_members", "has_creator_keyword": has_creator_keyword}
        
        has_new_member_keywords = any(keyword in query_lower for keyword in [
            '신규', '새로운', '가입', 'new', '신규회원', '신규멤버', '회원수', '회원 수'
        ])
        has_membership_performance_keywords = any(keyword in query_lower for keyword in [
            '멤버십', '맴버쉽', '성과', '실적', 'performance', 'membership', '분석'
        ])
        has_creator_keyword = (
            '크리에이터' in query_lower or 
            'creator' in query_lower or
            any(keyword in query_lower for keyword in ['작가', '아티스트', '제작자'])
        )
        
        if has_new_member_keywords and not has_membership_performance_keywords:
            return {"type": "new_members", "has_creator_keyword": has_creator_keyword}
        else:
            return {"type": "membership_performance", "has_creator_keyword": has_creator_keyword}
    
    def _validate_schema_for_dynamic_template(self) -> bool:
        """동적 템플릿 생성을 위한 스키마 검증"""
        if not self.db_schema or len(self.db_schema) == 0:
            self.logger.warning("db_schema가 비어있어 동적 템플릿 생성 실패")
            return False
        
        if 't_fanding' not in self.db_schema:
            self.logger.warning("t_fanding 테이블이 db_schema에 없어 동적 템플릿 생성 실패")
            return False
        
        fanding_schema = self.db_schema['t_fanding']
        fanding_columns = fanding_schema.get('columns', {})
        
        if 'member_no' not in fanding_columns or 'ins_datetime' not in fanding_columns:
            self.logger.warning("t_fanding에 필요한 컬럼(member_no, ins_datetime)이 없어 동적 템플릿 생성 실패")
            return False
        
        return True
    
    def _find_member_table_info(self) -> Optional[Dict[str, str]]:
        """멤버 테이블 정보 찾기 (t_member 또는 t_member_info)"""
        member_table = None
        member_pk = None
        member_status_col = None
        
        if 't_member' in self.db_schema:
            member_table = 't_member'
            member_schema = self.db_schema['t_member']
            member_columns = member_schema.get('columns', {})
            for col_name in ['no', 'id', 'member_no']:
                if col_name in member_columns:
                    member_pk = col_name
                    break
            if 'status' in member_columns:
                member_status_col = 'status'
        elif 't_member_info' in self.db_schema:
            member_table = 't_member_info'
            member_schema = self.db_schema['t_member_info']
            member_columns = member_schema.get('columns', {})
            for col_name in ['member_no', 'no', 'id']:
                if col_name in member_columns:
                    member_pk = col_name
                    break
            if 'status' in member_columns:
                member_status_col = 'status'
        
        if not member_table or not member_pk:
            self.logger.warning("t_member 또는 t_member_info 테이블을 찾을 수 없어 동적 템플릿 생성 실패")
            return None
        
        return {
            'table': member_table,
            'pk': member_pk,
            'status_col': member_status_col,
            'join_condition': f"f.member_no = m.{member_pk}"
        }
    
    def _create_new_members_template(
        self, month_num: str, yyyy_mm: str, year: int, month: int, has_creator_keyword: bool
    ) -> Optional[SQLTemplate]:
        """신규 회원수 템플릿 생성"""
        if has_creator_keyword:
            # 크리에이터 필터가 필요한 경우: t_fanding 사용
            if not (self._has_column('t_fanding', 'ins_datetime') and 
                    self._has_column('t_fanding', 'creator_no')):
                self.logger.warning("크리에이터 필터가 필요한데 t_fanding 테이블 또는 creator_no 컬럼이 없습니다")
                return None
            
            sql_template = f"""
            SELECT COUNT(DISTINCT f.member_no) as new_members_{month_num}month 
            FROM t_fanding f
            WHERE DATE_FORMAT(f.ins_datetime, '%Y-%m') = '{yyyy_mm}'
            """
            self.logger.info(f"크리에이터 필터가 필요한 쿼리로 t_fanding 테이블 사용 (creator_no 필터는 SQLGenerationNode에서 추가됨)")
        else:
            # 크리에이터 필터가 필요 없는 경우
            if 't_member_info' in self.db_schema and self._has_column('t_member_info', 'ins_datetime'):
                sql_template = f"""
            SELECT COUNT(*) as new_members_{month_num}month 
            FROM t_member_info 
            WHERE DATE_FORMAT(ins_datetime, '%Y-%m') = '{yyyy_mm}'
            """
            elif 't_fanding' in self.db_schema and self._has_column('t_fanding', 'ins_datetime'):
                sql_template = f"""
            SELECT COUNT(DISTINCT f.member_no) as new_members_{month_num}month 
            FROM t_fanding f
            WHERE DATE_FORMAT(f.ins_datetime, '%Y-%m') = '{yyyy_mm}'
            """
            else:
                # fallback
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
    
    def _create_membership_performance_template(
        self, month_num: str, yyyy_mm: str, year: int, month: int, member_info: Dict[str, str]
    ) -> SQLTemplate:
        """멤버십 성과 분석 템플릿 생성"""
        member_table = member_info['table']
        member_status_col = member_info['status_col']
        join_condition = member_info['join_condition']
        
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
            sql_template = f"""
            SELECT 
                '{month_num}월' as analysis_month,
                COUNT(DISTINCT f.member_no) as total_members
            FROM t_fanding f
            INNER JOIN {member_table} m ON {join_condition}
            WHERE DATE_FORMAT(f.ins_datetime, '%Y-%m') = '{yyyy_mm}'
            """
        
        self.logger.info(
            f"동적 멤버십 성과 템플릿 생성 완료: {member_table} 사용, "
            f"PK={member_info['pk']}, status_col={member_status_col}"
        )
        
        return SQLTemplate(
            name=f"{month_num}월 멤버십 성과 분석",
            description=f"{month_num}월 멤버십 성과 상세 분석 ({year}년 데이터, db_schema 기반 동적 생성)",
            sql_template=sql_template,
            parameters=[],
            analysis_type=FandingAnalysisType.MEMBERSHIP_DATA,
            keywords=["멤버십", "맴버쉽", "성과", "실적", "분석", f"{month}월", f"{month_num}월"]
        )

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
        # 실제 컬럼 정보 (동적 스키마 정보 활용)
        column_info = self._get_table_column_descriptions(table_name)
        
        result = f"📊 **{table_name} 테이블 구조**\n\n"
        result += f"**설명**: {description}\n\n"
        result += "**주요 컬럼**:\n"
        
        for column, col_type in column_info.items():
            result += f"• **{column}**: {col_type}\n"
        
        return result
    
    def _get_table_column_descriptions(self, table_name: str) -> Dict[str, str]:
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
        
        주의: fallback 스키마일 때는 검증을 건너뜁니다.
        실제 스키마일 때만 검증을 수행하여 초기화 시간을 최적화합니다.
        """
        # fallback 스키마일 때는 검증을 건너뛰고 경고만 표시
        if self._is_fallback_schema:
            self.logger.warning(
                f"⚠️  스키마가 fallback 모드입니다 (DB 연결 실패 가능성). "
                f"템플릿 검증을 건너뜁니다. 현재 스키마 테이블 수: {len(self.db_schema)}"
            )
            return
        
        validation_errors = []
        validation_warnings = []
        
        self.logger.info("템플릿 스키마 검증을 시작합니다...")
        
        for template_name, template in self.templates.items():
            try:
                sql_content = template.sql_template.lower()
                
                # 테이블명 추출 및 검증
                table_names = self._extract_table_names_from_sql(sql_content)
                for table_name in table_names:
                    if table_name not in self.db_schema:
                        validation_errors.append(f"Template '{template_name}' uses invalid table: '{table_name}'")
                        continue
                    
                    # 컬럼명 검증 (테이블이 존재하는 경우에만)
                    table_columns = set(self.db_schema[table_name].get("columns", {}).keys())
                    column_names = self._extract_column_names_from_sql(sql_content, table_name)
                    for column_name in column_names:
                        if column_name not in table_columns:
                            validation_errors.append(f"Template '{template_name}' uses invalid column '{column_name}' in table '{table_name}'")
                
                # 템플릿 파라미터 검증
                # 주의: 파라미터는 '{param}' 또는 {param} 형태로 사용될 수 있음
                if template.parameters:
                    for param in template.parameters:
                        # 파라미터가 SQL에 포함되어 있는지 확인
                        # 패턴 1: '{param}' (작은따옴표 + 중괄호) - 문자열 리터럴
                        # 패턴 2: {param} (중괄호만) - 숫자나 계산식
                        param_pattern1 = "'{" + param + "}'"  # 따옴표 포함
                        param_pattern2 = "{" + param + "}"    # 따옴표 없음
                        
                        # SQL 템플릿에서 파라미터 사용 여부 확인
                        if param_pattern1 not in template.sql_template and param_pattern2 not in template.sql_template:
                            # 파라미터가 SQL에 직접 사용되지 않는 경우만 경고
                            # (동적으로 추가되는 경우도 있으므로 경고 수준)
                            validation_warnings.append(f"Template '{template_name}' declares parameter '{param}' but doesn't use it in SQL")
                
            except Exception as e:
                validation_errors.append(f"Template '{template_name}' validation failed: {str(e)}")
        
        # 검증 결과 요약 (로그는 한 번에 출력하여 성능 최적화)
        if validation_errors:
            self.logger.error(f"템플릿 검증 실패: {len(validation_errors)}개 오류 발견")
            for error in validation_errors[:10]:  # 처음 10개만 표시
                self.logger.error(f"  - {error}")
            if len(validation_errors) > 10:
                self.logger.error(f"  ... 외 {len(validation_errors) - 10}개 오류 더 있음")
        else:
            self.logger.info(f"템플릿 검증 성공: {len(self.templates)}개 템플릿 모두 유효")
        
        if validation_warnings:
            self.logger.warning(f"템플릿 검증 경고: {len(validation_warnings)}개 경고 발견")
            for warning in validation_warnings[:5]:  # 처음 5개만 표시
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
    
    def to_sql_examples(self, max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fanding SQL 템플릿을 prompts.py의 SQLExample 형식으로 변환
        
        이 메서드는 fanding_sql_templates의 실제 사용되는 템플릿을
        LLM 학습용 예제로 변환하여 prompts.py에서 활용할 수 있도록 합니다.
        
        Args:
            max_examples: 변환할 최대 예제 수 (None이면 전체)
            
        Returns:
            SQLExample 형식의 딕셔너리 리스트
        """
        examples = []
        
        # 템플릿을 순회하며 SQLExample 형식으로 변환
        for template_name, template in self.templates.items():
            # 파라미터가 있는 템플릿은 기본값으로 파라미터화하여 예제 생성
            if template.parameters:
                # 파라미터 기본값 설정
                default_params = {}
                if 'month' in template.parameters:
                    from datetime import datetime
                    default_params['month'] = datetime.now().month
                if 'top_k' in template.parameters:
                    default_params['top_k'] = 5
                if 'days' in template.parameters:
                    default_params['days'] = 30
                # 필수 파라미터가 있지만 기본값이 없는 경우 플레이스홀더 사용
                if 'creator_name' in template.parameters:
                    default_params['creator_name'] = '크리에이터명'
                if 'target_date' in template.parameters:
                    default_params['target_date'] = '2025-01-01'
                if 'target_month' in template.parameters:
                    default_params['target_month'] = '2025-01'
                if 'target_year' in template.parameters:
                    default_params['target_year'] = 2025
                if 'target_week_of_year' in template.parameters:
                    default_params['target_week_of_year'] = 1
                if 'target_week_of_month' in template.parameters:
                    default_params['target_week_of_month'] = 1
                
                # 파라미터 적용된 템플릿 가져오기
                param_template = self.get_parameterized_template(template_name, default_params)
                if param_template:
                    sql = param_template.sql_template
                    name = param_template.name
                else:
                    sql = template.sql_template
                    name = template.name
            else:
                sql = template.sql_template
                name = template.name
            
            # SQL 템플릿에서 실제 SQL 추출 (정리)
            sql = sql.strip()
            
            # 템플릿 이름과 설명을 기반으로 자연어 질문 생성
            # 예: "이번 달 신규 회원" → "이번 달 신규 회원 수를 알려줘"
            question = self._generate_question_from_template(name, template.description, template)
            
            # 키워드를 태그로 변환
            tags = []
            if template.keywords:
                tags = template.keywords[:5]  # 최대 5개 태그
            
            # 분석 타입을 카테고리로 사용
            category = template.analysis_type.value if template.analysis_type else "unknown"
            
            # SQL 복잡도에 따라 difficulty 설정
            difficulty = self._estimate_difficulty(sql)
            
            example = {
                "question": question,
                "sql": sql,
                "description": template.description,
                "category": category,
                "difficulty": difficulty,
                "tags": tags,
                "source": f"fanding_template:{template_name}"  # 출처 표시
            }
            
            examples.append(example)
        
        # max_examples가 지정된 경우 제한
        if max_examples and len(examples) > max_examples:
            # 중요도가 높은 템플릿 우선 선택 (멤버십 데이터 > 성과 리포트 > 콘텐츠 > 고급)
            priority_order = [
                FandingAnalysisType.MEMBERSHIP_DATA,
                FandingAnalysisType.PERFORMANCE_REPORT,
                FandingAnalysisType.CONTENT_PERFORMANCE,
                FandingAnalysisType.ADVANCED_ANALYSIS
            ]
            
            sorted_examples = []
            for priority in priority_order:
                for example in examples:
                    if example["category"] == priority.value:
                        sorted_examples.append(example)
                        if len(sorted_examples) >= max_examples:
                            break
                if len(sorted_examples) >= max_examples:
                    break
            
            # 아직 부족하면 나머지 추가
            for example in examples:
                if example not in sorted_examples:
                    sorted_examples.append(example)
                    if len(sorted_examples) >= max_examples:
                        break
            
            examples = sorted_examples[:max_examples]
        
        self.logger.info(f"Converted {len(examples)} Fanding templates to SQL examples")
        return examples
    
    def _generate_question_from_template(self, name: str, description: str, template: SQLTemplate) -> str:
        """
        템플릿 이름과 설명을 기반으로 자연어 질문 생성
        
        Args:
            name: 템플릿 이름
            description: 템플릿 설명
            template: SQLTemplate 객체
            
        Returns:
            자연어 질문 문자열
        """
        # 템플릿 이름을 기반으로 질문 생성
        question_templates = [
            "{name}를 알려줘",
            "{name}을 조회해줘",
            "{name}을 보여줘",
            "{name} 현황을 알려줘",
            "{name} 수를 알려줘"
        ]
        
        import random
        
        # 키워드가 있으면 더 구체적인 질문 생성
        if template.keywords:
            # 키워드 중 가장 관련성 높은 것을 선택
            main_keyword = template.keywords[0] if template.keywords else ""
            
            # 특정 패턴 매칭
            if "신규" in name or "신규" in main_keyword:
                if "월" in name:
                    return f"{name}을 알려줘"
                return f"{name} 수를 알려줘"
            elif "활성" in name or "활성" in main_keyword:
                return f"{name}을 조회해줘"
            elif "매출" in name or "매출" in main_keyword or "revenue" in main_keyword.lower():
                return f"{name}을 분석해줘"
            elif "회원" in name or "member" in main_keyword.lower():
                return f"{name}을 보여줘"
        
        # 기본 템플릿 사용
        template_str = random.choice(question_templates)
        return template_str.format(name=name)
    
    def _estimate_difficulty(self, sql: str) -> str:
        """
        SQL 복잡도를 기반으로 difficulty 추정
        
        Args:
            sql: SQL 쿼리 문자열
            
        Returns:
            "easy", "medium", "hard"
        """
        sql_lower = sql.lower()
        
        # 복잡도 지표
        has_join = "join" in sql_lower
        has_subquery = "select" in sql_lower and sql_lower.count("select") > 1
        has_group_by = "group by" in sql_lower
        has_aggregation = any(func in sql_lower for func in ["count", "sum", "avg", "max", "min"])
        has_case = "case when" in sql_lower or "case" in sql_lower
        has_cte = "with" in sql_lower
        
        # 복잡도 계산
        complexity_score = 0
        if has_join:
            complexity_score += 1
        if has_subquery:
            complexity_score += 2
        if has_group_by:
            complexity_score += 1
        if has_aggregation:
            complexity_score += 1
        if has_case:
            complexity_score += 1
        if has_cte:
            complexity_score += 2
        
        # difficulty 결정
        if complexity_score <= 1:
            return "easy"
        elif complexity_score <= 3:
            return "medium"
        else:
            return "hard"
