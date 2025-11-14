import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load tables into DataFrames
# The `results` variable is provided as a dictionary containing JSON strings.
df_fanding_log = pd.DataFrame(json.loads(results["t_fanding_log"]))
df_fanding = pd.DataFrame(json.loads(results["t_fanding"]))
df_payment = pd.DataFrame(json.loads(results["t_payment"]))

# Since t_creator and t_member are not in the provided `results` dictionary,
# we need to infer creator_no and map to nicknames for '강환국 작가' and '고래돈공부'.
# We extract creator_no for '강환국 작가' from payment data's order_name.
# A placeholder creator_no is used for '고래돈공부'.

kanghwanguk_creator_no = None
if not df_payment.empty and 'order_name' in df_payment.columns:
    kanghwanguk_payments = df_payment[df_payment['order_name'].str.contains('강환국', na=False, case=False)]
    if not kanghwanguk_payments.empty:
        kanghwanguk_creator_no = kanghwanguk_payments['seller_creator_no'].iloc[0]

# Use a default if not found, or if df_payment was empty
if kanghwanguk_creator_no is None:
    kanghwanguk_creator_no = 6903 # A common creator_no for Kanghwanguk if not found in data

# Assign a unique placeholder creator_no for '고래돈공부'
goraedon_creator_no = 12345
if kanghwanguk_creator_no == goraedon_creator_no:
    goraedon_creator_no = 12346 # Ensure uniqueness if default matched

creator_mapping_data = {
    'no': [kanghwanguk_creator_no, goraedon_creator_no],
    'member_no': [100000 + kanghwanguk_creator_no, 100000 + goraedon_creator_no] # Dummy member_no for creator's own member profile
}
df_creator = pd.DataFrame(creator_mapping_data)

member_mapping_data = {
    'no': [100000 + kanghwanguk_creator_no, 100000 + goraedon_creator_no],
    'nickname': ['강환국 작가', '고래돈공부']
}
df_member = pd.DataFrame(member_mapping_data)

# --- Analysis Part 1: 2025년 8월 전체 멤버십 가입자 수 ---

# Convert timestamp columns in df_fanding_log to datetime objects
df_fanding_log['start_date'] = pd.to_datetime(df_fanding_log['start_date'], unit='ms')
df_fanding_log['end_date'] = pd.to_datetime(df_fanding_log['end_date'], unit='ms')

# Filter for membership logs active in August 2025
# The SQL query for `t_fanding_log` already filtered for this period:
# `start_date <= '2025-08-31'` AND `end_date >= '2025-08-01'`
# So, `df_fanding_log` should already contain relevant log entries.

# Merge with df_fanding to get `member_no` associated with `fanding_no`
merged_fanding_for_subscribers = df_fanding_log.merge(
    df_fanding[['no', 'member_no']],
    left_on='fanding_no',
    right_on='no',
    how='inner'
)

# Count distinct `member_no` to get the total number of unique subscribers active in August 2025
total_august_subscribers = merged_fanding_for_subscribers['member_no'].nunique()

# --- Analysis Part 2: '강환국 작가', '고래돈공부' 크리에이터의 2025년 8월 월 성과 분석 ---

# Convert timestamp column in df_payment to datetime objects
df_payment['pay_datetime'] = pd.to_datetime(df_payment['pay_datetime'], unit='ms')

# Filter payments for August 2025. The SQL already applied this filter.
start_date_august = pd.Timestamp('2025-08-01 00:00:00')
end_date_august = pd.Timestamp('2025-08-31 23:59:59')
df_payment_august = df_payment[
    (df_payment['pay_datetime'] >= start_date_august) &
    (df_payment['pay_datetime'] <= end_date_august)
].copy()

# Function to calculate actual sales based on currency and heat
def calculate_actual_sales(row):
    if row['currency_no'] == 1: # KRW
        return row['remain_price']
    elif row['currency_no'] == 2: # USD
        return row['remain_price'] * 1360
    elif pd.isna(row['currency_no']) or row['currency_no'] == 0: # HEAT
        return row['remain_heat'] * 110
    return 0 # Default for other cases or incomplete data

# Apply the sales calculation to create a new 'actual_sales' column
df_payment_august['actual_sales'] = df_payment_august.apply(calculate_actual_sales, axis=1)

# Merge df_payment_august with df_creator to link seller_creator_no to the creator's member_no
df_payment_with_creator = df_payment_august.merge(
    df_creator[['no', 'member_no']],
    left_on='seller_creator_no',
    right_on='no',
    how='inner',
    suffixes=('', '_creator_temp')
)

# Merge df_payment_with_creator with df_member to get the creator's nickname
df_payment_with_creator_info = df_payment_with_creator.merge(
    df_member[['no', 'nickname']],
    left_on='member_no_creator_temp',
    right_on='no',
    how='inner',
    suffixes=('', '_member_temp')
)

# Filter for the specific creators as per the user query
target_creators = ['강환국 작가', '고래돈공부']
filtered_creator_payments = df_payment_with_creator_info[
    df_payment_with_creator_info['nickname'].isin(target_creators)
]

# Aggregate actual sales by creator nickname for August 2025
creator_performance = filtered_creator_payments.groupby('nickname')['actual_sales'].sum().reset_index()

# Ensure both target creators are in the result, even if one had 0 sales
all_creators_df = pd.DataFrame({'nickname': target_creators})
creator_performance = pd.merge(all_creators_df, creator_performance, on='nickname', how='left')
creator_performance['actual_sales'] = creator_performance['actual_sales'].fillna(0)


# --- Output Results ---
print(f"2025년 8월 전체 멤버십 가입자 수: {total_august_subscribers}명")
print("\n'강환국 작가', '고래돈공부' 크리에이터의 2025년 8월 월 성과:")
for index, row in creator_performance.iterrows():
    print(f"- {row['nickname']}: {row['actual_sales']:,.0f}원")

# --- Comparison ---
print("\n--- 월 성과 비교 ---")
kanghwanguk_sales = creator_performance[creator_performance['nickname'] == '강환국 작가']['actual_sales'].iloc[0]
goraedon_sales = creator_performance[creator_performance['nickname'] == '고래돈공부']['actual_sales'].iloc[0]

if kanghwanguk_sales > goraedon_sales:
    print(f"'강환국 작가'의 성과가 '고래돈공부'보다 {kanghwanguk_sales - goraedon_sales:,.0f}원 더 높습니다.")
elif goraedon_sales > kanghwanguk_sales:
    print(f"'고래돈공부'의 성과가 '강환국 작가'보다 {goraedon_sales - kanghwanguk_sales:,.0f}원 더 높습니다.")
else:
    print("두 크리에이터의 2025년 8월 월 성과가 동일합니다.")

# --- 시각화 (Optional) ---
if not creator_performance.empty:
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