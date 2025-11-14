import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load tables into DataFrames
# The `results` variable is provided as a dictionary containing JSON strings.
df_fanding_log = pd.DataFrame(json.loads(results["t_fanding_log"]))
df_fanding = pd.DataFrame(json.loads(results["t_fanding"]))
df_payment = pd.DataFrame(json.loads(results["t_payment"]))

# --- Creator Nickname Mapping (Inferred due to missing t_creator, t_member in results) ---
# The SQL query for t_payment filters by creator nickname but only returns t_payment columns.
# We need to re-associate seller_creator_no with creator nicknames for analysis.
# We will infer this mapping from the 'order_name' column in df_payment.

creator_name_map = {}
unique_seller_creator_nos = df_payment['seller_creator_no'].unique()

# Attempt to map creator_no to nickname using order_name heuristic
for creator_no in unique_seller_creator_nos:
    # Get relevant order_names for this creator_no
    relevant_orders = df_payment[df_payment['seller_creator_no'] == creator_no]['order_name'].astype(str).str.lower().unique()

    # Heuristic to find creator names
    if any('강환국' in name for name in relevant_orders):
        creator_name_map[creator_no] = '강환국 작가'
    elif any('고래돈공부' in name for name in relevant_orders): # Assuming '고래돈공부' might also appear in order_name
        creator_name_map[creator_no] = '고래돈공부'
    # For this specific request, we only care about '강환국 작가' and '고래돈공부'.
    # Any other creator_no not explicitly matching these names won't be considered.

# Create a DataFrame for inferred creator names
# This is a temporary structure to simulate having a t_creator_name table
inferred_creator_names_df = pd.DataFrame(list(creator_name_map.items()), columns=['seller_creator_no', 'nickname_inferred'])


# --- Analysis Part 1: 2025년 8월 전체 멤버십 가입자 수 ---

# Convert timestamp columns in df_fanding_log to datetime objects
df_fanding_log['start_date'] = pd.to_datetime(df_fanding_log['start_date'], unit='ms')
df_fanding_log['end_date'] = pd.to_datetime(df_fanding_log['end_date'], unit='ms')

# Merge with df_fanding to get `member_no` and `fanding_status` associated with `fanding_no`
merged_fanding_for_subscribers = df_fanding_log.merge(
    df_fanding[['no', 'member_no', 'fanding_status']],
    left_on='fanding_no',
    right_on='no',
    how='inner'
)

# Filter for active memberships ('T': 가입 중/활성)
# The SQL query for `t_fanding_log` already filtered for log periods overlapping August.
# The `fanding_status = 'T'` in df_fanding ensures we only count currently active memberships.
active_in_august_fanding_logs = merged_fanding_for_subscribers[
    (merged_fanding_for_subscribers['fanding_status'] == 'T')
]

# Count distinct `member_no` to get the total number of unique subscribers active in August 2025
total_august_subscribers = active_in_august_fanding_logs['member_no'].nunique()

# --- Analysis Part 2: '강환국 작가', '고래돈공부' 크리에이터의 2025년 8월 월 성과 분석 ---

# Convert timestamp column in df_payment to datetime objects
df_payment['pay_datetime'] = pd.to_datetime(df_payment['pay_datetime'], unit='ms')

# Filter payments for August 2025. The SQL already applied this filter.
# However, to be explicit about the timeframe for the Python analysis part:
start_date_august = pd.Timestamp('2025-08-01 00:00:00')
end_date_august = pd.Timestamp('2025-08-31 23:59:59')
df_payment_august = df_payment[
    (df_payment['pay_datetime'] >= start_date_august) &
    (df_payment['pay_datetime'] <= end_date_august)
].copy()

# Function to calculate actual sales based on currency and heat
def calculate_actual_sales(row):
    # Ensure currency_no is treated as numeric
    currency_no = pd.to_numeric(row['currency_no'], errors='coerce')

    if currency_no == 1: # KRW
        return row['remain_price']
    elif currency_no == 2: # USD
        return row['remain_price'] * 1360
    elif pd.isna(currency_no): # HEAT (currency_no is NULL)
        return row['remain_heat'] * 110
    return 0 # Default for other cases or incomplete data

# Apply the sales calculation to create a new 'actual_sales' column
df_payment_august['actual_sales'] = df_payment_august.apply(calculate_actual_sales, axis=1)

# Merge with the inferred creator names
df_payment_with_creator_info = df_payment_august.merge(
    inferred_creator_names_df,
    on='seller_creator_no',
    how='left'
)

# Filter for the specific creators as per the user query
target_creators = ['강환국 작가', '고래돈공부']
filtered_creator_payments = df_payment_with_creator_info[
    df_payment_with_creator_info['nickname_inferred'].isin(target_creators)
].copy() # Ensure operating on a copy to avoid SettingWithCopyWarning

# Aggregate actual sales by creator nickname for August 2025
creator_performance = filtered_creator_payments.groupby('nickname_inferred')['actual_sales'].sum().reset_index()
creator_performance = creator_performance.rename(columns={'nickname_inferred': 'nickname'})


# Ensure both target creators are in the result, even if one had 0 sales
all_creators_df_for_merge = pd.DataFrame({'nickname': target_creators})
creator_performance = pd.merge(all_creators_df_for_merge, creator_performance, on='nickname', how='left')
creator_performance['actual_sales'] = creator_performance['actual_sales'].fillna(0)


# --- Output Results ---
print(f"2025년 8월 전체 멤버십 가입자 수: {total_august_subscribers}명")
print("\n'강환국 작가', '고래돈공부' 크리에이터의 2025년 8월 월 성과:")
for index, row in creator_performance.iterrows():
    print(f"- {row['nickname']}: {row['actual_sales']:,.0f}원")

# --- Comparison ---
print("\n--- 월 성과 비교 ---")
# Safely get sales for each creator, handling cases where they might not be in the performance data
kanghwanguk_sales = creator_performance[creator_performance['nickname'] == '강환국 작가']['actual_sales'].iloc[0] if '강환국 작가' in creator_performance['nickname'].values else 0
goraedon_sales = creator_performance[creator_performance['nickname'] == '고래돈공부']['actual_sales'].iloc[0] if '고래돈공부' in creator_performance['nickname'].values else 0

if kanghwanguk_sales > goraedon_sales:
    print(f"'강환국 작가'의 성과가 '고래돈공부'보다 {kanghwanguk_sales - goraedon_sales:,.0f}원 더 높습니다.")
elif goraedon_sales > kanghwanguk_sales:
    print(f"'고래돈공부'의 성과가 '강환국 작가'보다 {goraedon_sales - kanghwanguk_sales:,.0f}원 더 높습니다.")
else:
    print("두 크리에이터의 2025년 8월 월 성과가 동일합니다.")

# --- 시각화 (Optional) ---
if not creator_performance.empty and not creator_performance['actual_sales'].isnull().all():
    plt.figure(figsize=(10, 6))
    sns.barplot(x='nickname', y='actual_sales', data=creator_performance, palette='viridis')
    plt.title('2025년 8월 \'강환국 작가\' vs \'고래돈공부\' 크리에이터 월 성과', fontsize=15)
    plt.xlabel('크리에이터', fontsize=12)
    plt.ylabel('매출 금액 (원)', fontsize=12)
    plt.xticks(rotation=0, ha='center')
    plt.ticklabel_format(style='plain', axis='y')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()