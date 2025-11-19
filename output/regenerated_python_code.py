import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np
import matplotlib.pyplot as plt

# Initialize member_first_join_map early to ensure it's always defined
member_first_join_map = {}

# Database connection
engine = create_engine(
    f"mysql+pymysql://readonly_user_business_data:Fanding!Data!@epic-readonly.ceind7azkfjy.ap-northeast-2.rds.amazonaws.com:3306/fanding?charset=utf8mb4",
    echo=False
)

# SQL query
query = """
SELECT
    'FIRST_JOIN' AS `record_type`,
    tf.creator_no AS `creator_id`,
    tf.member_no AS `member_id`,
    tf.ins_datetime AS `event_datetime`,
    NULL AS `start_date_specific`,
    NULL AS `end_date_specific`,
    NULL AS `amount_price`,
    NULL AS `amount_heat`,
    NULL AS `currency_id`,
    NULL AS `transaction_status`
FROM
    t_fanding tf
WHERE
    tf.creator_no = (SELECT tc.no FROM t_member tm JOIN t_creator tc ON tm.no = tc.member_no WHERE tm.nickname = '강환국 작가')

UNION ALL

SELECT
    'TIER_PERIOD' AS `record_type`,
    tf.creator_no AS `creator_id`,
    tf.member_no AS `member_id`,
    tfl.start_date AS `event_datetime`,
    tfl.start_date AS `start_date_specific`,
    tfl.end_date AS `end_date_specific`,
    NULL AS `amount_price`,
    NULL AS `amount_heat`,
    NULL AS `currency_id`,
    NULL AS `transaction_status`
FROM
    t_fanding tf
JOIN
    t_fanding_log tfl ON tf.no = tfl.fanding_no
WHERE
    tf.creator_no = (SELECT tc.no FROM t_member tm JOIN t_creator tc ON tm.no = tc.member_no WHERE tm.nickname = '강환국 작가')
    AND (
        tfl.start_date <= '2025-10-31'
        AND tfl.end_date >= '2025-03-01'
    )

UNION ALL

SELECT
    'PAYMENT' AS `record_type`,
    tp.seller_creator_no AS `creator_id`,
    tp.member_no AS `member_id`,
    tp.pay_datetime AS `event_datetime`,
    NULL AS `start_date_specific`,
    NULL AS `end_date_specific`,
    tp.remain_price AS `amount_price`,
    tp.remain_heat AS `amount_heat`,
    tp.currency_no AS `currency_id`,
    tp.status AS `transaction_status`
FROM
    t_payment tp
WHERE
    tp.seller_creator_no = (SELECT tc.no FROM t_member tm JOIN t_creator tc ON tm.no = tc.member_no WHERE tm.nickname = '강환국 작가')
    AND tp.pay_datetime >= '2025-03-01 00:00:00'
    AND tp.pay_datetime <= '2025-10-31 23:59:59'
    AND tp.status IN ('T', 'P')
ORDER BY
    `event_datetime`, `record_type`, `member_id`;
"""
df = pd.read_sql(text(query), engine)

# Convert datetime columns
df['event_datetime'] = pd.to_datetime(df['event_datetime'])
df['start_date_specific'] = pd.to_datetime(df['start_date_specific'])
df['end_date_specific'] = pd.to_datetime(df['end_date_specific'])

# Define query period
start_date_overall = pd.Timestamp('2025-03-01')
end_date_overall = pd.Timestamp('2025-10-31')

# --- Member Analysis Preparation ---
# Get first join dates for all members of this creator
df_first_join_all = df[df['record_type'] == 'FIRST_JOIN'].copy()

# Populate member_first_join_map if df_first_join_all is not empty
# member_first_join_map is already initialized as {} at the top of the script
if not df_first_join_all.empty:
    df_first_join_all['first_join_month'] = df_first_join_all['event_datetime'].dt.to_period('M')
    member_first_join_map = df_first_join_all.set_index('member_id')['first_join_month'].to_dict()

# --- Monthly Aggregation Loop ---
monthly_results = []
current_month_iter = start_date_overall
while current_month_iter <= end_date_overall:
    month_start = current_month_iter.to_period('M').start_time
    month_end = current_month_iter.to_period('M').end_time

    # New Members: Members whose first join month is the current iterating month
    new_members_this_month_ids = df_first_join_all[
        df_first_join_all['first_join_month'] == current_month_iter.to_period('M')
    ]['member_id'].unique() if not df_first_join_all.empty else np.array([])
    new_member_count = len(new_members_this_month_ids)

    # Active Members (for the current iterating month): Members with an overlapping tier period
    df_tier_period_current_month = df[(df['record_type'] == 'TIER_PERIOD') &
                                      (df['start_date_specific'] <= month_end) &
                                      (df['end_date_specific'] >= month_start)].copy()
    active_members_this_month_ids = df_tier_period_current_month['member_id'].unique()
    
    # Existing Members: Active members whose first join date was before the current iterating month
    # Filter active members to only include those present in member_first_join_map
    existing_members_this_month_ids = [
        member_id for member_id in active_members_this_month_ids
        if member_id in member_first_join_map and member_first_join_map[member_id] < current_month_iter.to_period('M')
    ]
    existing_member_count = len(existing_members_this_month_ids)
    
    # --- Sales Analysis for the current iterating month ---
    df_payment_current_month = df[(df['record_type'] == 'PAYMENT') &
                                  (df['event_datetime'] >= month_start) &
                                  (df['event_datetime'] <= month_end) &
                                  (df['transaction_status'].isin(['T', 'P']))].copy()

    # Calculate actual sales amount based on currency conversion rules
    df_payment_current_month['actual_sales'] = np.where(
        df_payment_current_month['currency_id'] == 1,
        df_payment_current_month['amount_price'],
        np.where(
            df_payment_current_month['currency_id'] == 2,
            df_payment_current_month['amount_price'] * 1360,
            df_payment_current_month['amount_heat'] * 110 # currency_id is NULL for HEAT
        )
    )
    monthly_sales = df_payment_current_month['actual_sales'].sum() if not df_payment_current_month.empty else 0

    monthly_results.append({
        "month": current_month_iter.strftime('%Y-%m'),
        "new_members": int(new_member_count),
        "existing_members": int(existing_member_count),
        "monthly_revenue": float(monthly_sales)
    })

    # Move to next month
    current_month_iter += pd.DateOffset(months=1)

# Format output as JSON
final_output = []
for result in monthly_results:
    final_output.append({
        "월": result["month"],
        "신규회원수": result["new_members"],
        "기존회원수": result["existing_members"],
        "월매출액": round(result["monthly_revenue"], 2)
    })

# Print the final JSON output
print(final_output)

# Optional Visualization
df_plot = pd.DataFrame(final_output)
df_plot['월'] = pd.to_datetime(df_plot['월'])
df_plot = df_plot.set_index('월')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Member Counts
ax1.plot(df_plot.index, df_plot['신규회원수'], marker='o', label='신규 회원수', color='skyblue')
ax1.plot(df_plot.index, df_plot['기존회원수'], marker='o', label='기존 회원수', color='lightcoral')
ax1.set_xlabel('월')
ax1.set_ylabel('회원수', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.legend(loc='upper left')
ax1.set_title('강환국 작가 월별 회원수 및 매출액 추이 (25년 3월 ~ 10월)')

# Plot Monthly Revenue
ax2 = ax1.twinx()
ax2.plot(df_plot.index, df_plot['월매출액'], marker='x', linestyle='--', color='green', label='월 매출액')
ax2.set_ylabel('월 매출액', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.legend(loc='upper right')

fig.tight_layout()
plt.show()