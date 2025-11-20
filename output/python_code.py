import pandas as pd
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import numpy as np

# Matplotlib 한글 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터베이스 연결
engine = create_engine(
    f"mysql+pymysql://readonly_user_business_data:Fanding!Data!@epic-readonly.ceind7azkfjy.ap-northeast-2.rds.amazonaws.com:3306/fanding?charset=utf8mb4",
    echo=False
)

# 1. 25년 10월 '강환국 작가'의 포스트 조회 로그 (작성자 본인 조회 제외, 삭제된 포스트 제외)
query_creator_post_views = """
SELECT
    `tpvl`.`ins_datetime` AS `view_datetime`,
    `tpvl`.`member_no` AS `viewer_member_no`,
    `tp`.`no` AS `post_no`,
    `tp`.`title` AS `post_title`,
    `tp`.`ins_datetime` AS `post_upload_datetime`,
    `tp`.`member_no` AS `creator_member_no`
FROM
    `t_post_view_log` AS `tpvl`
JOIN
    `t_post` AS `tp` ON `tpvl`.`post_no` = `tp`.`no`
JOIN
    `t_member` AS `tm_creator` ON `tp`.`member_no` = `tm_creator`.`no`
WHERE
    `tm_creator`.`nickname` = '강환국 작가'
    AND `tp`.`del_datetime` IS NULL
    AND `tpvl`.`member_no` != `tp`.`member_no`
    AND `tpvl`.`ins_datetime` >= '2025-10-01 00:00:00'
    AND `tpvl`.`ins_datetime` <= '2025-10-31 23:59:59';
"""

# 2. 25년 10월 '강환국 작가'의 포스트 업로드 로그 (삭제된 포스트 제외)
query_creator_posts_uploaded = """
SELECT
    `tp`.`no` AS `post_no`,
    `tp`.`title` AS `post_title`,
    `tp`.`ins_datetime` AS `upload_datetime`,
    `tp`.`member_no` AS `creator_member_no`
FROM
    `t_post` AS `tp`
JOIN
    `t_member` AS `tm_creator` ON `tp`.`member_no` = `tm_creator`.`no`
WHERE
    `tm_creator`.`nickname` = '강환국 작가'
    AND `tp`.`del_datetime` IS NULL
    AND `tp`.`ins_datetime` >= '2025-10-01 00:00:00'
    AND `tp`.`ins_datetime` <= '2025-10-31 23:59:59';
"""

# 3. 25년 10월 말 '강환국 작가'의 활성 멤버십 회원 (10월 31일 기준)
query_creator_active_members = """
SELECT
    `tf`.`member_no` AS `active_member_no`,
    `tf`.`creator_no`,
    `tfl`.`start_date`,
    `tfl`.`end_date`
FROM
    `t_fanding` AS `tf`
JOIN
    `t_creator` AS `tc` ON `tf`.`creator_no` = `tc`.`no`
JOIN
    `t_member` AS `tm_creator` ON `tc`.`member_no` = `tm_creator`.`no`
JOIN
    `t_fanding_log` AS `tfl` ON `tf`.`current_fanding_log_no` = `tfl`.`no`
WHERE
    `tm_creator`.`nickname` = '강환국 작가'
    AND `tf`.`fanding_status` = 'T'
    AND `tfl`.`start_date` <= '2025-10-31'
    AND `tfl`.`end_date` >= '2025-10-31';
"""

# 데이터 로드
df_views = pd.read_sql(text(query_creator_post_views), engine)
df_uploads = pd.read_sql(text(query_creator_posts_uploaded), engine)
df_active_members = pd.read_sql(text(query_creator_active_members), engine)

# datetime 컬럼 변환
df_views['view_datetime'] = pd.to_datetime(df_views['view_datetime'])
df_views['post_upload_datetime'] = pd.to_datetime(df_views['post_upload_datetime'])
df_uploads['upload_datetime'] = pd.to_datetime(df_uploads['upload_datetime'])
df_active_members['start_date'] = pd.to_datetime(df_active_members['start_date'])
df_active_members['end_date'] = pd.to_datetime(df_active_members['end_date'])


# 1. 25년 10월 '강환국 작가'의 월 평균 포스트 방문자 수
# 일별 고유 방문자 수 (DAU) 계산
daily_unique_visitors = df_views.groupby(df_views['view_datetime'].dt.date)['viewer_member_no'].nunique()
# 월 평균 일별 방문자 수 계산
monthly_avg_daily_visitors = daily_unique_visitors.mean()

print(f"25년 10월 '강환국 작가'의 월 평균 포스트 방문자 수: {monthly_avg_daily_visitors:.2f}명")


# 2. 25년 10월 '강환국 작가'의 해당 월 회원수 대비 방문자 수 비율
# 해당 월 포스트를 조회한 고유 방문자
unique_viewers_in_month = df_views['viewer_member_no'].unique()
# 해당 월 말 기준 활성 회원 (멤버십 가입자)
unique_active_members = df_active_members['active_member_no'].unique()
total_active_members_count = len(unique_active_members)

# 활성 회원 중 포스트를 조회한 회원 (교집합)
viewing_active_members_count = len(np.intersect1d(unique_viewers_in_month, unique_active_members))

# 비율 계산
if total_active_members_count > 0:
    visitor_to_member_ratio = (viewing_active_members_count / total_active_members_count) * 100
    print(f"25년 10월 '강환국 작가'의 해당 월 회원수 대비 방문자 수 비율: {visitor_to_member_ratio:.2f}%")
else:
    print("25년 10월 '강환국 작가'의 활성 회원수가 없어 방문자 수 비율을 계산할 수 없습니다.")


# 3. 10월 내 포스트 업로드 및 방문자수 추이
# 일별 포스트 업로드 수
daily_uploads_count = df_uploads.groupby(df_uploads['upload_datetime'].dt.date)['post_no'].count().rename('post_count')
# 일별 방문자 수 (위에서 계산한 daily_unique_visitors 사용)
daily_visitors_count = daily_unique_visitors.rename('visitor_count')

# 10월 전체 날짜 범위 생성
october_dates = pd.date_range(start='2025-10-01', end='2025-10-31', freq='D').date
trend_df = pd.DataFrame(index=october_dates)

# 데이터 병합 및 결측치 0으로 채우기
trend_df = trend_df.merge(daily_uploads_count, left_index=True, right_index=True, how='left')
trend_df = trend_df.merge(daily_visitors_count, left_index=True, right_index=True, how='left')
trend_df = trend_df.fillna(0)

# 시각화
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(trend_df.index, trend_df['post_count'], color='blue', marker='o', linestyle='-', label='포스트 업로드 수')
ax1.set_xlabel('날짜')
ax1.set_ylabel('포스트 업로드 수', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_title('25년 10월 강환국 작가 포스트 업로드 및 방문자 수 추이')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(trend_df.index, trend_df['visitor_count'], color='red', marker='x', linestyle='--', label='방문자 수')
ax2.set_ylabel('방문자 수', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.legend(loc='upper right')

fig.autofmt_xdate()
plt.show()


# 4. 방문자 수 상위 5개 포스트 (10월에 업로드된 포스트 중)
# 10월에 업로드된 포스트만 필터링
posts_uploaded_in_october_views = df_views[
    (df_views['post_upload_datetime'].dt.year == 2025) &
    (df_views['post_upload_datetime'].dt.month == 10)
]

# 포스트별 고유 방문자 수 집계
top_posts_by_visitors = posts_uploaded_in_october_views.groupby(['post_no', 'post_title'])['viewer_member_no'].nunique().reset_index()
top_posts_by_visitors = top_posts_by_visitors.rename(columns={'viewer_member_no': 'unique_visitors'})

# 방문자 수 기준으로 내림차순 정렬 후 상위 5개 추출
top_5_posts = top_posts_by_visitors.sort_values(by='unique_visitors', ascending=False).head(5)

print("\n25년 10월 방문자 수 상위 5개 포스트:")
if not top_5_posts.empty:
    for index, row in top_5_posts.iterrows():
        print(f"- 제목: {row['post_title']}, 방문자 수: {row['unique_visitors']}명")
else:
    print("25년 10월에 업로드된 포스트가 없거나 조회 기록이 없습니다.")