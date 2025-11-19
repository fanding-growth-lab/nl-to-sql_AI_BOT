import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import json
import matplotlib.pyplot as plt

# Database connection details (provided)
engine = create_engine(
    f"mysql+pymysql://readonly_user_business_data:Fanding!Data!@epic-readonly.ceind7azkfjy.ap-northeast-2.rds.amazonaws.com:3306/fanding?charset=utf8mb4",
    echo=False
)

# SQL 쿼리문
query = """
SELECT
    'NEW_SUBSCRIPTION_START' AS `record_type`,
    tf.member_no AS `member_id`,
    NULL AS `payment_raw_price`,
    NULL AS `payment_currency_no`,
    NULL AS `payment_raw_heat`,
    NULL AS `fanding_start_date_raw`,
    NULL AS `fanding_end_date_raw`,
    tfl.start_date AS `subscription_event_date_raw`, -- 신규 구독 시작 날짜 (복귀 포함)
    NULL AS `pay_datetime_raw`
FROM
    t_fanding tf
JOIN
    t_fanding_log tfl ON tf.no = tfl.fanding_no
WHERE
    tf.creator_no = (
        SELECT tc.no
        FROM t_member tm
        JOIN t_creator tc ON tm.no = tc.member_no
        WHERE tm.nickname = '강환국 작가'
    )
    AND tfl.start_date >= '2025-03-01' -- 구독 시작일 기준 필터링
    AND tfl.start_date <= '2025-10-31'

UNION ALL

SELECT
    'EXISTING_MEMBER_PERIOD' AS `record_type`,
    tf.member_no AS `member_id`,
    NULL AS `payment_raw_price`,
    NULL AS `payment_currency_no`,
    NULL AS `payment_raw_heat`,
    tfl.start_date AS `fanding_start_date_raw`, -- 멤버십 기간 시작일
    tfl.end_date AS `fanding_end_date_raw`,   -- 멤버십 기간 종료일
    NULL AS `subscription_event_date_raw`,
    NULL AS `pay_datetime_raw`
FROM
    t_fanding tf
JOIN
    t_fanding_log tfl ON tf.no = tfl.fanding_no
WHERE
    tf.creator_no = (
        SELECT tc.no
        FROM t_member tm
        JOIN t_creator tc ON tm.no = tc.member_no
        WHERE tm.nickname = '강환국 작가'
    )
    AND tfl.start_date <= '2025-10-31' -- 분석 기간 내에 시작하거나 종료되는 모든 멤버십 기간 포함
    AND tfl.end_date >= '2025-03-01'

UNION ALL

SELECT
    'REVENUE' AS `record_type`,
    NULL AS `member_id`,
    tp.remain_price AS `payment_raw_price`,
    tp.currency_no AS `payment_currency_no`,
    tp.remain_heat AS `payment_raw_heat`,
    NULL AS `fanding_start_date_raw`,
    NULL AS `fanding_end_date_raw`,
    NULL AS `subscription_event_date_raw`,
    tp.pay_datetime AS `pay_datetime_raw` -- 매출 발생 전체 날짜
FROM
    t_payment tp
WHERE
    tp.seller_creator_no = (
        SELECT tc.no
        FROM t_member tm
        JOIN t_creator tc ON tm.no = tc.member_no
        WHERE tm.nickname = '강환국 작가'
    )
    AND tp.pay_datetime >= '2025-03-01 00:00:00'
    AND tp.pay_datetime <= '2025-10-31 23:59:59'
    AND tp.status IN ('T', 'P');
"""

# SQL 쿼리 실행 및 데이터 로드
df = pd.read_sql(text(query), engine)

# 모든 관련 날짜/시간 컬럼을 datetime 객체로 변환 (NaT 포함 가능)
df['fanding_start_date_raw'] = pd.to_datetime(df['fanding_start_date_raw'])
df['fanding_end_date_raw'] = pd.to_datetime(df['fanding_end_date_raw'])
df['subscription_event_date_raw'] = pd.to_datetime(df['subscription_event_date_raw'])
df['pay_datetime_raw'] = pd.to_datetime(df['pay_datetime_raw'])

# 대상 기간 설정
start_date_range = pd.to_datetime('2025-03-01')
end_date_range = pd.to_datetime('2025-10-31')
all_months = pd.period_range(start=start_date_range, end=end_date_range, freq='M')

# 1. 신규 회원수 계산 (복귀 회원 포함)
df_new_subscriptions = df[df['record_type'] == 'NEW_SUBSCRIPTION_START'].copy()
df_new_subscriptions['month_key'] = df_new_subscriptions['subscription_event_date_raw'].dt.strftime('%Y-%m')

new_members_monthly = df_new_subscriptions \
    .groupby('month_key')['member_id'].nunique().reset_index()
new_members_monthly.rename(columns={'member_id': 'new_members'}, inplace=True)

# 2. 기존 회원수 계산 (해당 월의 마지막 날짜 23:59:59에 구독 중인 회원)
df_fanding_periods = df[df['record_type'] == 'EXISTING_MEMBER_PERIOD'].copy()

existing_members_data = []
for month_period in all_months:
    month_end_snapshot = month_period.end_time # YYYY-MM-DD 23:59:59

    # 해당 월의 마지막 날짜에 구독이 활성화된 회원 필터링
    # 즉, 구독 시작일이 마지막 날짜 이전에 시작했고, 구독 종료일이 마지막 날짜 이후에 종료되는 경우
    active_at_month_end = df_fanding_periods[
        (df_fanding_periods['fanding_start_date_raw'].dt.date <= month_end_snapshot.date()) &
        (df_fanding_periods['fanding_end_date_raw'].dt.date >= month_end_snapshot.date())
    ]
    # 해당 월의 마지막 날짜 기준 고유 회원 ID 수
    num_existing_members = active_at_month_end['member_id'].nunique()
    existing_members_data.append({
        'month_key': month_period.strftime('%Y-%m'),
        'existing_members': num_existing_members
    })
existing_members_monthly = pd.DataFrame(existing_members_data)

# 3. 월 매출액 계산
df_revenue = df[df['record_type'] == 'REVENUE'].copy()
df_revenue['calculated_revenue'] = 0.0

# 매출액 계산 로직 (벡터화된 연산 사용)
# KRW (currency_no = 1)
krw_mask = (df_revenue['payment_currency_no'] == 1)
df_revenue.loc[krw_mask, 'calculated_revenue'] = df_revenue.loc[krw_mask, 'payment_raw_price']

# USD (currency_no = 2)
usd_mask = (df_revenue['payment_currency_no'] == 2)
df_revenue.loc[usd_mask, 'calculated_revenue'] = df_revenue.loc[usd_mask, 'payment_raw_price'] * 1360

# HEAT (currency_no is NULL)
heat_mask = (pd.isna(df_revenue['payment_currency_no']))
df_revenue.loc[heat_mask, 'calculated_revenue'] = df_revenue.loc[heat_mask, 'payment_raw_heat'] * 110

# pay_datetime_raw를 사용하여 월 키 생성
df_revenue['month_key'] = df_revenue['pay_datetime_raw'].dt.strftime('%Y-%m')
revenue_monthly = df_revenue.groupby('month_key')['calculated_revenue'].sum().reset_index()
revenue_monthly.rename(columns={'calculated_revenue': 'monthly_revenue'}, inplace=True)

# 4. 최종 결과 병합
# 모든 월을 포함하는 기본 DataFrame 생성
all_month_keys_df = pd.DataFrame({'month_key': [p.strftime('%Y-%m') for p in all_months]})

final_df = all_month_keys_df.merge(new_members_monthly, on='month_key', how='left')
final_df = final_df.merge(existing_members_monthly, on='month_key', how='left')
final_df = final_df.merge(revenue_monthly, on='month_key', how='left')

# NaN 값을 0으로 채우고 데이터 타입 변환
final_df['new_members'] = final_df['new_members'].fillna(0).astype(int)
final_df['existing_members'] = final_df['existing_members'].fillna(0).astype(int)
final_df['monthly_revenue'] = final_df['monthly_revenue'].fillna(0.0)

# 5. JSON 형식으로 출력
result_list = final_df.to_dict(orient='records')
print(json.dumps(result_list, ensure_ascii=False, indent=4))

# (선택적) 시각화: 월별 추이 그래프
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

# 월별 신규 회원수
axes[0].bar(final_df['month_key'], final_df['new_members'], color='skyblue')
axes[0].set_title('월별 신규 회원수 추이')
axes[0].set_ylabel('신규 회원수')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# 월별 기존 회원수
axes[1].bar(final_df['month_key'], final_df['existing_members'], color='lightgreen')
axes[1].set_title('월별 기존 회원수 추이')
axes[1].set_ylabel('기존 회원수')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

# 월별 매출액
axes[2].bar(final_df['month_key'], final_df['monthly_revenue'], color='lightcoral')
axes[2].set_title('월별 매출액 추이')
axes[2].set_xlabel('월')
axes[2].set_ylabel('매출액')
axes[2].tick_params(axis='x', rotation=45)
axes[2].grid(axis='y', linestyle='--', alpha=0.7)
# y축 눈금에 과학적 표기법 방지
from matplotlib.ticker import ScalarFormatter
for ax in axes:
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
    ax.ticklabel_format(style='plain', axis='y')

plt.tight_layout()
plt.show()