import json
import pandas as pd

results_file_path = "sql_query_results\\ebe64650-26e5-4d0d-bf9a-21b80d0133e2.json"

with open(results_file_path, 'r', encoding='utf-8') as f:
    full_results = json.load(f)

# Load dataframes from the JSON results
df_fanding_log = pd.DataFrame(json.loads(full_results["t_fanding_log"]))
df_payment = pd.DataFrame(json.loads(full_results["t_payment"]))

# --- 1. 25년 8월 전체 멤버십 가입자 수 분석 ---
# Convert 'start_date' from milliseconds timestamp to datetime objects
df_fanding_log['start_date'] = pd.to_datetime(df_fanding_log['start_date'], unit='ms')

# Filter for membership activations (start_date) in August 2025
august_2025_fanding_log = df_fanding_log[
    (df_fanding_log['start_date'].dt.year == 2025) &
    (df_fanding_log['start_date'].dt.month == 8)
].copy()

# Count distinct 'fanding_no' to get the number of unique membership activation instances.
# Since member_no is not available in the t_fanding_log output, we count unique fanding_no as subscribers.
total_membership_subscribers = august_2025_fanding_log['fanding_no'].nunique()

# --- 2. '강환국 작가', '고래돈공부' 크리에이터의 25년 8월 월 성과 분석 ---
# Convert 'pay_datetime' from milliseconds timestamp to datetime objects
df_payment['pay_datetime'] = pd.to_datetime(df_payment['pay_datetime'], unit='ms')

# Filter payments for August 2025 and completed/partially refunded status ('T', 'P')
# Business Rule: When analyzing sales, include statuses 'T' (Approved) and 'P' (Partially Refunded).
august_2025_creator_payments = df_payment[
    (df_payment['pay_datetime'].dt.year == 2025) &
    (df_payment['pay_datetime'].dt.month == 8) &
    (df_payment['status'].isin(['T', 'P']))
].copy()

# Identify creators from 'order_name' as 't_payment' results do not include 'nickname' directly
def get_creator_name_from_order(order_name):
    if '강환국' in order_name:
        return '강환국 작가'
    elif '고래돈공부' in order_name:
        return '고래돈공부'
    return 'Other' # In case there are other creators, though SQL filtered for target ones

august_2025_creator_payments['creator_name'] = august_2025_creator_payments['order_name'].apply(get_creator_name_from_order)

# Apply currency conversion rules for actual sales calculation
# Business Rule: Actual sales amount must be calculated using `remain_price`.
# Currency Conversion:
#   - For KRW (currency_no = 1): use `remain_price`.
#   - For USD (currency_no = 2): use `remain_price` * 1360.
#   - For HEAT (currency_no is NULL): use `remain_heat` * 110.
def calculate_actual_sales_amount(row):
    if row['currency_no'] == 1: # KRW
        return row['remain_price']
    elif row['currency_no'] == 2: # USD
        return row['remain_price'] * 1360
    elif pd.isna(row['currency_no']): # HEAT
        return row['remain_heat'] * 110
    return 0 # Default for other or unexpected currency_no

august_2025_creator_payments['actual_sales_krw'] = august_2025_creator_payments.apply(calculate_actual_sales_amount, axis=1)

# Group by creator name and sum the actual sales
creator_performance_summary = august_2025_creator_payments.groupby('creator_name')['actual_sales_krw'].sum().reset_index()

# Filter out 'Other' if any and ensure only target creators are considered
creator_performance_summary = creator_performance_summary[
    creator_performance_summary['creator_name'].isin(['강환국 작가', '고래돈공부'])
]

# --- 3. 분석 및 비교 결과 출력 ---
print(f"### 2025년 8월 멤버십 성과 분석 ###")
print(f"\n1. 전체 멤버십 가입자 수 (2025년 8월): {total_membership_subscribers} 건")

print(f"\n2. 특정 크리에이터 ('강환국 작가', '고래돈공부')의 월별 성과 (2025년 8월):")
if not creator_performance_summary.empty:
    for index, row in creator_performance_summary.iterrows():
        print(f"- {row['creator_name']}: {row['actual_sales_krw']:,.0f} KRW")
else:
    print("  해당 크리에이터들의 8월 매출 데이터가 없습니다.")

print("\n3. 분석 및 비교:")
print(f"2025년 8월 한 달 동안 Fanding 플랫폼에서 총 {total_membership_subscribers}건의 멤버십 가입(활성화)이 이루어졌습니다.")

if len(creator_performance_summary) == 2:
    creator_a = creator_performance_summary[creator_performance_summary['creator_name'] == '강환국 작가'].iloc[0]
    creator_b = creator_performance_summary[creator_performance_summary['creator_name'] == '고래돈공부'].iloc[0]

    print(f"\n'강환국 작가'의 8월 매출은 {creator_a['actual_sales_krw']:,.0f} KRW이며,")
    print(f"'고래돈공부' 크리에이터의 8월 매출은 {creator_b['actual_sales_krw']:,.0f} KRW입니다.")

    if creator_a['actual_sales_krw'] > creator_b['actual_sales_krw']:
        diff = creator_a['actual_sales_krw'] - creator_b['actual_sales_krw']
        print(f"강환국 작가가 고래돈공부 크리에이터보다 약 {diff:,.0f}원 더 높은 매출을 기록했습니다.")
    elif creator_b['actual_sales_krw'] > creator_a['actual_sales_krw']:
        diff = creator_b['actual_sales_krw'] - creator_a['actual_sales_krw']
        print(f"고래돈공부 크리에이터가 강환국 작가보다 약 {diff:,.0f}원 더 높은 매출을 기록했습니다.")
    else:
        print("두 크리에이터의 8월 매출은 동일합니다.")
elif len(creator_performance_summary) == 1:
    only_creator = creator_performance_summary.iloc[0]
    print(f"{only_creator['creator_name']} 크리에이터의 8월 매출은 {only_creator['actual_sales_krw']:,.0f} KRW입니다. 다른 지정 크리에이터의 데이터는 없습니다.")
else:
    print("지정된 두 크리에이터의 8월 매출 데이터를 비교할 수 없습니다.")