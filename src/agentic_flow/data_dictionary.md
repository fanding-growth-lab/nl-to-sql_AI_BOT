# fanding 데이터베이스 데이터 사전

이 문서는 fanding 데이터베이스의 주요 테이블과 컬럼에 대한 설명입니다. AI는 이 정보를 바탕으로 사용자의 질문에 더 정확한 SQL 쿼리를 생성할 수 있습니다.

---

## `t_creator` - 크리에이터 정보 테이블

**설명:** [여기에 테이블에 대한 설명을 작성해주세요]

| 컬럼명 | 데이터 타입 | 설명 |
| --- | --- | --- |
| `no` | `int(10) unsigned` | [크리에이터를 고유하게 식별하는 번호 (PK)] |
| `member_no` | `int(10) unsigned` | [해당 크리에이터의 member_no. (즉, 크리에이터도 멤버의 한 종류)] |
| `launching_datetime` | `datetime` | [크리에이터 서비스 런칭일] |
| `is_active` | `char(1)` | [크리에이터 활성화 여부] |

---

## `t_creator_coupon` - 크리에이터 쿠폰 테이블

**설명:** [여기에 테이블에 대한 설명을 작성해주세요]

| 컬럼명 | 데이터 타입 | 설명 |
| --- | --- | --- |
| `no` | `int(10) unsigned` | [쿠폰 고유 식별 번호 (PK)] |
| `creator_no` | `int(10) unsigned` | [해당 쿠폰을 발행한 크리에이터의 번호 (FK)] |
| `name` | `varchar(50)` | [쿠폰 이름] |
| `code` | `varchar(17)` | [쿠폰 활성화 코드] |
| `duration` | `int(10) unsigned` | [쿠폰 혜택 기간] |
| `expiry_end_date` | `date` | [쿠폰 등록 만료일] |
| `ins_datetime` | `datetime` | [쿠폰 등록 시작일] |

---

## `t_creator_coupon_member`

**설명:** [여기에 테이블에 대한 설명을 작성해주세요]

| 컬럼명 | 데이터 타입 | 설명 |
| --- | --- | --- |
| `no` | `int(10) unsigned` | [쿠폰 사용 로그 번호] |
| `coupon_no` | `int(10) unsigned` | [쿠폰 번호 (FK to t_creator_coupon.no)] |
| `member_no` | `int(10) unsigned` | [쿠폰을 사용한 회원의 멤버 번호 (FK to t_member.no)] |
| `ins_datetime` | `datetime` | [쿠폰 사용일] |

---

## `t_creator_department`

**설명:** [여기에 테이블에 대한 설명을 작성해주세요]

| 컬럼명 | 데이터 타입 | 설명 |
| --- | --- | --- |
| `no` | `int(11) unsigned` | [부서 카테고리를 고유하게 식별하는 번호 (PK)] |
| `name` | `varchar(15)` | [부서 카테고리 이름 (한국어, 예: 프로페셔널, 셀러브리티, 보이스 등)] |
| `name_eng` | `varchar(30)` | [부서 카테고리 이름 (영어)] |

---

## `t_creator_department_mapping`

**설명:** [여기에 테이블에 대한 설명을 작성해주세요]

| 컬럼명 | 데이터 타입 | 설명 |
| --- | --- | --- |
| `creator_no` | `int(11) unsigned` | [t_creator 테이블의 no (크리에이터 고유 ID)를 참조 (FK)] |
| `department_no` | `int(11) unsigned` | [t_creator_department 테이블의 no (부서 카테고리 ID)를 참조 (FK)] |

---

## `t_event`

**설명:** [여기에 테이블에 대한 설명을 작성해주세요]

| 컬럼명 | 데이터 타입 | 설명 |
| --- | --- | --- |
| `no` | `int(11) unsigned` | [이벤트 고유 번호 (PK)] |
| `creator_no` | `int(11) unsigned` | [이벤트를 생성한 크리에이터의 번호 (FK to t_creator.no)] |
| `title` | `varchar(255)` | [이벤트 이름] |
| `is_offline` | `char(1)` | [오프라인 이벤트 여부 (t/f)] |
| `online_url` | `varchar(300)` | [이벤트 url] |
| `address` | `varchar(300)` | [이벤트 개최 주소] |
| `detail_address` | `varchar(300)` | [이벤트 개최 상세 주소] |
| `latitude` | `decimal(10,8)` | [이벤트 주소 위도] |
| `longitude` | `decimal(11,8)` | [이벤트 주소 경도] |
| `event_start_time` | `datetime` | [이벤트 시작일] |
| `event_end_time` | `datetime` | [이벤트 종료일] |
| `selling_start_time` | `datetime` | [이벤트 판매 시작일] |
| `selling_end_time` | `datetime` | [이벤트 판매 종료일] |
| `is_private` | `char(1)` | [이벤트 공개/비공개 여부] |

---

## `t_event_member`

**설명:** [여기에 테이블에 대한 설명을 작성해주세요]

| 컬럼명 | 데이터 타입 | 설명 |
| --- | --- | --- |
| `no` | `int(11) unsigned` | [이벤트 신청 로그 번호 (PK)] |
| `event_no` | `int(11) unsigned` | [이벤트 번호 (FK to t_event.no)] |
| `ticket_no` | `int(11) unsigned` | [이베늩 참석 티켓 번호] |
| `member_no` | `int(11) unsigned` | [이벤트 신청자 회원 번호 (FK to t_member.no)] |
| `attendance` | `char(1)` | [참석 여부] |
| `status` | `char(1)` | [상태] |
| `attend_datetime` | `datetime` | [참석 날짜] |
| `ins_datetime` | `datetime` | [신청 로그 생성 날짜] |

---

## `t_fanding`

**설명:** [멤버십 활성화 정보]

| 컬럼명 | 데이터 타입 | 설명 |
| --- | --- | --- |
| `no` | `int(11) unsigned` | [팬딩(멤버십 가입 건) 자체를 고유하게 식별하는 번호 (PK)] |
| `current_tier_no` | `int(10) unsigned` | [현재 이용중인 멤버십 번호 (FK to t_tier.no)] |
| `current_fanding_log_no` | `int(11) unsigned` | [현재 해당하는 팬딩로그 번호 (FK to t_fanding_log.no)] |
| `member_no` | `int(11) unsigned` | [해당 멤버십에 가입한 멤버의 member_no (FK to t_member.no)] |
| `creator_no` | `int(11) unsigned` | [해당 멤버십을 제공하는 크리에이터의 creator_no (FK to t_creator.no)] |
| `fanding_status` | `char(1)` | [현재 시점의 멤버십 상태 ('T': 가입 중/활성, 'F': 이탈/비활성). 주의: 이 값은 과거 특정 시점의 상태가 아닌, 현재 상태만을 나타냅니다.] |
| `ins_datetime` | `datetime` | [해당 멤버가 이 크리에이터의 멤버십에 최초로 가입(또는 팬딩 관계 생성)한 날짜 및 시간. (멤버의 '첫 팬딩 시작일' 계산에 사용)] |


---

## `t_fanding_log`

**설명:** [여기에 테이블에 대한 설명을 작성해주세요]

| 컬럼명 | 데이터 타입 | 설명 |
| --- | --- | --- |
| `no` | `int(11) unsigned` | [No: 로그 레코드 자체의 고유 ID (PK)] |
| `fanding_no` | `int(11) unsigned` | [t_fanding 테이블의 No를 참조 (FK)] |
| `edition` | `smallint(5) unsigned` | [멤버십을 몇 번째 구매하고 있는지 나타내는 횟수.] |
| `period` | `smallint(5) unsigned` | [사용중인 멤버십 상품의 기간 (개월수)] |
| `tier_log_no` | `int(10) unsigned` | [사용중인 멤버십 정보 로그 번호 (FK to t_tier_log.no)] |
| `currency_no` | `tinyint(3) unsigned` | [통화 구분 (1: 원화, 2: 달러 등. 환율 적용 필요)] |
| `price` | `decimal(9,2) unsigned` | [해당 멤버십 기간의 가격] |
| `heat` | `int(10) unsigned` | [사용된 히트(서비스 내 재화)] |
| `coupon_member_no` | `int(10) unsigned` | [해당 팬딩로그 기간 내 사용한 쿠폰의 쿠폰 로그 번호 (FK to t_creator_coupon_member.no)] |
| `start_date` | `date` | [해당 멤버십 기간(또는 갱신 기간)의 시작일] |
| `end_date` | `date` | [해당 멤버십 기간(또는 갱신 기간)의 종료일] |

---

## `t_fanding_reserve_log`

**설명:** [여기에 테이블에 대한 설명을 작성해주세요]

| 컬럼명 | 데이터 타입 | 설명 |
| --- | --- | --- |
| `no` | `int(11) unsigned` | [멤버십 갱신 중단 설정 로그 번호 (PK)] |
| `fanding_no` | `int(11) unsigned` | [멤버십 정보 번호] |
| `status` | `char(1)` | [갱신 설정 상태 (t=갱신 활성화,f=갱신 비활성화)] |
| `tier_no` | `int(10) unsigned` | [이용중인 멤버십 번호 (FK to t_tier.no)] |
| `is_complete` | `char(1)` | [갱신 중단 실행 여부] |
| `ins_datetime` | `datetime` | [갱신 중단 설정 날짜] |
| `del_datetime` | `datetime` | [신규 요청으로 인한 기존 로그 삭제일] |
---

## `t_follow`

**설명:** [여기에 테이블에 대한 설명을 작성해주세요]

| 컬럼명 | 데이터 타입 | 설명 |
| --- | --- | --- |
| `no` | `int(10) unsigned` | [팔로우 액션의 고유 ID (PK)] |
| `creator_no` | `int(10) unsigned` | [ 팔로우를 받은 크리에이터의 creator_no (FK to t_creator.no)] |
| `member_no` | `int(10) unsigned` | [팔로우를 한 멤버의 member_no (FK to t_member_info.member_no)] |
| `ins_datetime` | `datetime` | [팔로우 액션이 발생한 날짜 및 시간.] |

---

## `t_member`

**설명:** [여기에 테이블에 대한 설명을 작성해주세요]

| 컬럼명 | 데이터 타입 | 설명 |
| --- | --- | --- |
| `no` | `int(11) unsigned` | [팬딩에 가입돼있는 회원의 고유 번호 (PK)] |
| `email` | `varchar(200)` | [회원 이메일] |
| `nickname` | `varchar(100)` | [가입된 회원의 닉네임] |
| `status` | `char(1)` | [가입 상태 (A=가입/인증 완료, J=가입완료)] |
| `is_admin` | `char(1)` | [플랫폼 어드민 권한 여부] |

---

## `t_member_join_phone_number`

**설명:** [여기에 테이블에 대한 설명을 작성해주세요]

| 컬럼명 | 데이터 타입 | 설명 |
| --- | --- | --- |
| `no` | `int(11) unsigned` | [회원 전화번호 로그 번호 (PK)] |
| `phone_country_no` | `int(11) unsigned` | [전화번호 지역번호] |
| `member_no` | `int(11) unsigned` | [회원의 회원 번호 (FK to t_member.no)] |
| `phone_number` | `varchar(20)` | [회원의 전화번호] |

---

## `t_payment`

**설명:** [여기에 테이블에 대한 설명을 작성해주세요]

| 컬럼명 | 데이터 타입 | 설명 |
| --- | --- | --- |
| `no` | `int(10) unsigned` | [결제 건의 고유 ID (PK)] |
| `member_no` | `int(11) unsigned` | [결제를 한 멤버의 member_no (FK to t_member_info.member_no)] |
| `seller_creator_no` | `int(10) unsigned` | [매출이 발생한 크리에이터의 creator_no (FK to t_creator.no)] |
| `tier_no` | `int(10) unsigned` | [멤버십을 구매한 경우 구매한 멤버십 상품의 번호(FK to t_tier.no)] |
| `item` | `varchar(20)` | [결제 상품 구분 (F: 멤버십 상품, C:프리미엄 컨텐츠, V:온라인 강의, E:유료 이벤트)] |
| `order_name` | `varchar(300)` | [구매한 상품 이름] |
| `currency_no` | `tinyint(3) unsigned` | [통화 구분 (1: 원화, 2: 달러 - 환율 1360원 적용 필요)] null값은 히트로 계산 |
| `heat` | `int(10) unsigned` | [결제 히트] |
| `remain_heat` | `int(10) unsigned` | [실제 사용/결제된 히트 (1히트 = 100원으로 환산하여 매출에 합산)] |
| `price` | `decimal(10,2) unsigned` | [결제 금액] |
| `remain_price` | `decimal(10,2) unsigned` | [실제 결제된 금액 (통화 적용 전)] |
| `is_tax_free` | `char(1)` | [면세 여부] |
| `status` | `char(1)` | [결제 상태 ('T' 또는 'P'가 실제 결제를 의미)] |
| `ins_datetime` | `datetime` | [결제 요청일] |
| `pay_datetime` | `datetime` | [결제가 발생(완료)된 날짜 및 시간 (매출 발생 시점)] |

---

## `t_post`

**설명:** [여기에 테이블에 대한 설명을 작성해주세요]

| 컬럼명 | 데이터 타입 | 설명 |
| --- | --- | --- |
| `no` | `int(11) unsigned` | [포스트의 고유 번호 (PK)] |
| `member_no` | `int(11) unsigned` | [포스트 작성자의 회원 번호 (FK)] |
| `title` | `varchar(210)` | [포스트 제목] |
| `content` | `mediumtext` | [포스트 내용] |
| `status` | `varchar(10)` | [포스트 발행 상태 (public=발행완료)] |
| `public_range` | `char(1)` | [포스트 공개 범위 (A:전체 공개, F:회원 대상 공개, C:유료, T:멤버십 상품 지정 공개)] |
| `content_type` | `char(1)` | [컨텐츠 유형 (M:비디오, I:이미지, A:오디오, 미지정시 복합)] |
| `is_fix_home` | `char(1)` | [홈화면 고정 여부] |
| `is_fix_top` | `char(1)` | [상단 고정 여부] |
| `view_count` | `int(11) unsigned` | [조회 수] |
| `like_count` | `int(11) unsigned` | [좋아요 수] |
| `ins_datetime` | `datetime` | [업로드 날짜] |
| `mod_datetime` | `datetime` | [수정한 날짜] |
| `del_datetime` | `datetime` | [삭제한 날짜] |

---

## `t_post_like_log`

**설명:** [여기에 테이블에 대한 설명을 작성해주세요]

| 컬럼명 | 데이터 타입 | 설명 |
| --- | --- | --- |
| `no` | `int(10) unsigned` | [포스트 좋아요 로그 번호 (PK)] |
| `post_no` | `int(10) unsigned` | [좋아요를 클릭한 포스트 번호 (FK to t_post.no)] |
| `member_no` | `int(10) unsigned` | [좋아요를 클릭한 회원의 회원 번호 (FK to t_member.no)] |
| `ins_datetime` | `datetime` | [좋아요를 클릭한 날짜] |

---

## `t_post_reply_like_log`

**설명:** [여기에 테이블에 대한 설명을 작성해주세요]

| 컬럼명 | 데이터 타입 | 설명 |
| --- | --- | --- |
| `no` | `int(10) unsigned` | [댓글에 좋아요를 누른 로그 번호 (PK)] |
| `reply_no` | `int(11) unsigned` | [좋아요를 클릭한 댓글 번호 (FK to t_port_reply.no)] |
| `member_no` | `int(11) unsigned` | [댓글에 좋아요를 클릭한 회원의 회원 번호 (FK to t_member.no)] |
| `ins_datetime` | `datetime` | [댓글에 좋아요를 클릭한 날짜] |

---

## `t_post_view_log`

**설명:** [여기에 테이블에 대한 설명을 작성해주세요]

| 컬럼명 | 데이터 타입 | 설명 |
| --- | --- | --- |
| `no` | `int(10) unsigned` | [포스트를 조회한 로그 번호 (PK)] |
| `post_no` | `int(10) unsigned` | [조회한 포스트의 번호 (FK to t_port.no)] |
| `member_no` | `int(10) unsigned` | [포스트를 조회한 회원의 회원 번호 (FK to t_member.no)] |
| `is_auth` | `char(1)` | [설명] |
| `ins_datetime` | `datetime` | [조회가 발생한 날짜] |

---

## `t_tier`

**설명:** [여기에 테이블에 대한 설명을 작성해주세요]

| 컬럼명 | 데이터 타입 | 설명 |
| --- | --- | --- |
| `no` | `int(10) unsigned` | [멤버십 상품의 고유 ID (PK).] |
| `creator_no` | `int(10) unsigned` | [해당 멤버십 상품을 제공하는 크리에이터의 ID (t_creator.no를 참조하는 FK)] |
| `public_status` | `varchar(10)` | [설명] |
| `is_renewable` | `char(1)` | [갱신 가능 여부] |
| `end_criteria` | `varchar(10)` | [설명] |
| `name` | `varchar(60)` | [멤버십 상품 이름 (예: '눈팅족')] |
| `regular_price` | `int(10) unsigned` | [설명] |
| `price` | `int(10) unsigned` | [멤버십 상품의 가격] |
| `regular_heat` | `int(10) unsigned` | [설명] |
| `heat` | `int(10) unsigned` | [설명] |
| `sponsor_limit` | `int(11)` | [설명] |
| `is_private` | `char(1)` | [설명] |
| `is_approval_required` | `char(1)` | [설명] |
| `is_monthly_pass_allowed` | `char(1)` | [설명] |
| `period` | `tinyint(3) unsigned` | [멤버십 상품의 기간, 개월수로 카운트] |
| `end_date` | `date` | [설명] |
| `join_start_date` | `date` | [설명] |
| `join_end_date` | `date` | [설명] |

---

## 테이블 간의 주요 관계

- [`t_creator.no`는 다른 테이블에서 `creator_no` 또는 `seller_creator_no`를 참조합니다.]
- [`t_fanding.no` 값이 `t_fanding_log`에서 `fanding_no`로 참조됨]

---


