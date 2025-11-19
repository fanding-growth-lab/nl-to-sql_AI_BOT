import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np

def get_creator_monthly_stats():
    """
    특정 크리에이터의 월별 회원수(신규, 기존)와 매출액 추이를 조회하여 출력합니다.
    """
    # 데이터베이스 연결 설정
    engine = create_engine(
        f"mysql+pymysql://readonly_user_business_data:Fanding!Data!@epic-readonly.ceind7azkfjy.ap-northeast-2.rds.amazonaws.com:3306/fanding?charset=utf8mb4",
        echo=False
    )

    # SQL 쿼리 정의
    query_sales = """
        SELECT
            p.seller_creator_no AS `creator_no`,
            p.currency_no,
            p.remain_price,
            p.remain_heat,
            p.pay_datetime
        FROM t_payment AS p
        INNER JOIN t_creator AS c ON p.seller_creator_no = c.no
        INNER JOIN t_member AS m ON c.member_no = m.no
        WHERE m.nickname = '강환국 작가'
        AND p.status IN ('T', 'P')
        AND p.pay_datetime BETWEEN '2025-03-01 00:00:00' AND '2025-10-31 23:59:59'
    """
    
    query_memberships = """
        SELECT
            f.creator_no,
            f.member_no,
            f.ins_datetime AS `first_subscription_datetime`,
            fl.start_date,
            fl.end_date
        FROM t_fanding AS f
        INNER JOIN t_fanding_log AS fl ON f.no = fl.fanding_no
        INNER JOIN t_creator AS c ON f.creator_no = c.no
        INNER JOIN t_member AS m ON c.member_no = m.no
        WHERE m.nickname = '강환국 작가'
        AND fl.start_date <= '2025-10-31'
        AND fl.end_date >= '2025-03-01'
    """

    # 데이터베이스에서 데이터 로드
    df_sales = pd.read_sql(text(query_sales), engine)
    df_memberships = pd.read_sql(text(query_memberships), engine)

    # 분석 기간 설정
    target_months = pd.period_range(start='2025-03', end='2025-10', freq='M')

    # 1. 월 매출액 계산
    monthly_sales = pd.Series(0, index=target_months, dtype=float)
    if not df_sales.empty:
        df_sales['pay_datetime'] = pd.to_datetime(df_sales['pay_datetime'])
        
        # 비즈니스 규칙에 따른 매출액 계산
        conditions = [
            df_sales['currency_no'] == 1, # KRW
            df_sales['currency_no'] == 2, # USD
            df_sales['currency_no'].isnull() # HEAT
        ]
        choices = [
            df_sales['remain_price'],
            df_sales['remain_price'] * 1360,
            df_sales['remain_heat'] * 110
        ]
        df_sales['revenue'] = np.select(conditions, choices, default=0.0)
        
        # 월별 매출액 집계
        df_sales['YearMonth'] = df_sales['pay_datetime'].dt.to_period('M')
        monthly_sales_agg = df_sales.groupby('YearMonth')['revenue'].sum()
        monthly_sales.update(monthly_sales_agg)


    # 2. 월별 회원수 계산 (신규, 기존)
    new_members_monthly = pd.Series(0, index=target_months, dtype=int)
    existing_members_monthly = pd.Series(0, index=target_months, dtype=int)

    if not df_memberships.empty:
        df_memberships['first_subscription_datetime'] = pd.to_datetime(df_memberships['first_subscription_datetime'])
        df_memberships['start_date'] = pd.to_datetime(df_memberships['start_date'])
        df_memberships['end_date'] = pd.to_datetime(df_memberships['end_date'])
        
        # 월별 신규 회원수 집계
        df_memberships['NewMemberMonth'] = df_memberships['first_subscription_datetime'].dt.to_period('M')
        new_members_agg = df_memberships.groupby('NewMemberMonth')['member_no'].nunique()
        new_members_monthly.update(new_members_agg)
        
        # 월별 총 활성 회원 및 기존 회원수 계산
        total_active_members = {}
        for month in target_months:
            month_start = month.start_time
            month_end = month.end_time
            
            # 해당 월에 활성화된 멤버십 필터링
            active_mask = (df_memberships['start_date'] <= month_end) & (df_memberships['end_date'] >= month_start)
            active_members_in_month = df_memberships.loc[active_mask, 'member_no'].unique()
            
            total_active_members[month] = len(active_members_in_month)

        total_active_series = pd.Series(total_active_members)
        existing_members_monthly = total_active_series - new_members_monthly


    # 최종 결과 DataFrame 생성
    result_df = pd.DataFrame({
        '신규_회원수': new_members_monthly,
        '기존_회원수': existing_members_monthly,
        '월_매출액': monthly_sales
    })
    
    result_df.index.name = '월'
    result_df = result_df.astype({'신규_회원수': int, '기존_회원수': int, '월_매출액': int})
    
    print(result_df)


if __name__ == '__main__':
    get_creator_monthly_stats()