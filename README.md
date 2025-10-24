# 🤖 자연어 SQL 변환 슬랙봇

데이터베이스 쿼리를 자연어로 질문하면 자동으로 SQL로 변환해주는 슬랙봇입니다. 사용할수록 똑똑해지는 자동 학습 기능이 포함되어 있습니다.

## 주요 기능

### 핵심 기능
- **자연어 처리**: 한국어/영어 질문을 SQL로 변환
- **실시간 스키마 매핑**: 데이터베이스 구조를 자동으로 파악하고 매핑
- **템플릿 매칭**: 미리 정의된 SQL 템플릿을 지능적으로 선택
- **자동 학습**: 사용자와의 상호작용을 통해 지속적으로 개선
- **슬랙 연동**: 슬랙에서 바로 사용 가능한 봇 기능
- **API 제공**: 외부 시스템과 연동 가능한 REST API

### 고급 기능
- **자동 학습**: 사용자 질문 패턴을 분석하여 성능 향상
- **패턴 인식**: 유사한 질문들을 그룹화하여 처리 효율성 증대
- **성능 최적화**: 자주 사용되는 쿼리 패턴을 우선순위로 설정
- **실시간 모니터링**: 시스템 상태와 성능 지표를 실시간으로 확인
- **배치 처리**: 대량의 학습 데이터를 효율적으로 처리
- **백업 시스템**: 최적화 전 자동 백업으로 데이터 안전성 보장

## 시스템 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                        사용자 인터페이스                        │
├─────────────────────────────────────────────────────────────────┤
│  슬랙봇 명령어        │    API 엔드포인트        │    웹 대시보드    │
│  /학습-분석          │    /learning/insights   │    실시간 모니터링 │
│  /학습-최적화        │    /learning/optimize   │    성능 분석      │
│  /학습-상태          │    /learning/status     │    보고서        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      핵심 처리 엔진                            │
├─────────────────────────────────────────────────────────────────┤
│  자연어 분석기        │  스키마 매핑기          │  SQL 생성기      │
│  └─ 자동학습 연동     │  └─ 자동학습 연동       │  └─ 자동학습 연동 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      자동 학습 엔진                            │
├─────────────────────────────────────────────────────────────────┤
│  • 사용자 질문 패턴 분석 및 저장                                │
│  • 성공/실패 데이터 수집 및 분석                                │
│  • 자주 사용되는 쿼리 패턴 우선순위 조정                        │
│  • 성능 개선 제안 자동 생성                                    │
│  • 매핑 규칙 실시간 업데이트                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 설치 및 설정

### 필요 조건
- Python 3.9 이상
- MySQL 또는 PostgreSQL 데이터베이스
- 슬랙 앱 (봇 기능 사용 시)

### 설치 과정

1. **uv 패키지 매니저 설치**
```bash
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **저장소 클론**
```bash
git clone https://github.com/fanding-growth-lab/nl-to-sql_AI_BOT.git
cd nl-to-sql_AI_BOT
```

3. **의존성 설치**
```bash
# 기본 패키지 설치
uv sync

# 개발용 패키지 포함 설치
uv sync --extra dev

# 문서 생성용 패키지 포함 설치
uv sync --extra docs
```

4. **환경 변수 설정**
```bash
# 환경 변수 템플릿 복사
cp env.template .env

# .env 파일을 편집하여 다음 값들을 설정:
# - DATABASE_URL: 데이터베이스 연결 문자열
# - SLACK_BOT_TOKEN: 슬랙 봇 토큰
# - SLACK_SIGNING_SECRET: 슬랙 서명 시크릿
# - LLM_API_KEY: AI 모델 API 키 (OpenAI/Anthropic/Google)
```

5. **데이터베이스 설정**
```bash
# 데이터베이스가 실행 중인지 확인
# 시스템이 자동으로 스키마를 감지하고 로드합니다
```

6. **애플리케이션 실행**
```bash
# uv를 사용하여 실행
uv run python src/main.py

# 또는 가상환경 활성화 후 실행
uv shell
python src/main.py
```

## 설정 가이드

### 환경 변수

| 변수명 | 설명 | 필수 여부 |
|--------|------|-----------|
| `DATABASE_URL` | 데이터베이스 연결 문자열 | 필수 |
| `SLACK_BOT_TOKEN` | 슬랙 봇 토큰 | 필수 |
| `SLACK_SIGNING_SECRET` | 슬랙 서명 시크릿 | 필수 |
| `LLM_API_KEY` | AI 모델 API 키 | 필수 |
| `LLM_MODEL` | 사용할 AI 모델명 | 선택 |
| `LOG_LEVEL` | 로그 레벨 | 선택 |

### 자동 학습 시스템 설정

자동 학습 기능은 `AutoLearningSystem` 클래스를 통해 설정할 수 있습니다:

```python
# 학습 기능 활성화/비활성화
learning_system.learning_enabled = True

# 배치 처리 설정
learning_system._batch_size = 10  # 한 번에 처리할 데이터 수
learning_system._save_interval = 60  # 저장 간격(초)

# 신뢰도 임계값 설정
LLM_CONFIDENCE_THRESHOLD_HIGH = 0.8    # 높은 신뢰도
LLM_CONFIDENCE_THRESHOLD_MEDIUM = 0.6  # 중간 신뢰도
LLM_CONFIDENCE_THRESHOLD_LOW = 0.4     # 낮은 신뢰도
```

## 사용 방법

### 슬랙봇 명령어

#### 기본 명령어
- `/help` - 사용 가능한 명령어와 예시 보기
- `/status` - 봇 상태 및 건강도 확인
- `/ping` - 봇 연결 테스트

#### 학습 관련 명령어
- `/학습-분석` - AI 학습 시스템 분석 결과 보기
- `/학습-최적화` - 학습 기반 최적화 적용
- `/학습-상태` - 학습 시스템 상태 확인

#### 자연어 질문
자연스러운 언어로 질문하면 됩니다:
- "지난 달 매출 데이터 보여줘"
- "상위 10명 고객은 누구야?"
- "어제 주문이 몇 개 들어왔어?"
- "8월 신규 회원 수를 알려줘"
- "9월 멤버십 성과 분석해줄수있어?"

### API 엔드포인트

#### 학습 시스템 API
```bash
# 학습 분석 결과 조회
GET /learning/insights

# 성능 지표 조회
GET /learning/performance

# 최적화 적용
POST /learning/optimize

# 최적화 상태 확인
GET /learning/optimization-status

# 학습 데이터 내보내기
POST /learning/export

# 학습 데이터 강제 저장
POST /learning/force-save
```

#### 상태 확인
```bash
# 기본 상태 확인
GET /health

# 상세 상태 정보
GET /status
```

## 자동 학습 시스템

### 작동 원리

1. **질문 분석**: 모든 사용자 질문을 자동으로 분석하고 기록
2. **패턴 추출**: 질문을 의미 있는 패턴으로 정규화
3. **성공 추적**: 성공률과 신뢰도 점수를 추적
4. **자동 최적화**: 학습 데이터를 바탕으로 패턴 자동 개선
5. **지속적 개선**: 상호작용할수록 더 똑똑해짐

### 학습 데이터 구조

```json
{
  "query_patterns": {
    "9월_멤버십_성과_분석_분석": {
      "pattern": "9월 멤버십 성과 분석해줄수있어?",
      "frequency": 8,
      "success_rate": 1.0,
      "avg_confidence": 0.95,
      "last_used": "2025-10-17T13:55:03.719599",
      "user_feedback": ["정확한 분석이었습니다"]
    }
  }
}
```

### 성능 지표

- **총 질문 수**: 처리된 질문의 총 개수
- **성공률**: 성공적으로 매핑된 질문의 비율
- **학습된 패턴**: 식별된 고유 패턴의 수
- **평균 신뢰도**: 모든 질문의 평균 신뢰도 점수
- **패턴 커버리지**: 학습된 패턴으로 커버되는 질문의 비율

## 모니터링 및 분석

### 실시간 지표
- 대기열 크기 및 처리 상태
- 백그라운드 저장 작업 상태
- 학습 시스템 건강도
- 패턴 통계
- 성능 트렌드

### 최적화 보고서
- 적용된 변경사항 요약
- 백업 생성 상태
- 패턴 우선순위 조정
- 신뢰도 점수 개선 사항

## 개발 가이드

### 프로젝트 구조
```
src/
├── agentic_flow/          # 핵심 AI 처리 모듈
│   ├── auto_learning_system.py    # 자동 학습 시스템
│   ├── enhanced_rag_mapper.py     # RAG 기반 매핑
│   ├── nodes.py                   # LangGraph 노드
│   └── learning_data.json         # 학습 데이터 저장소
├── core/                   # 핵심 유틸리티
│   ├── config.py          # 설정 관리
│   ├── db.py              # 데이터베이스 작업
│   └── logging.py         # 로깅 유틸리티
├── slack/                  # 슬랙 연동
│   └── bot.py               # 슬랙봇 구현
└── main.py                # FastAPI 애플리케이션
```

### 개발 명령어

#### 테스트
```bash
# 모든 테스트 실행
uv run pytest

# 특정 테스트 카테고리 실행
uv run pytest tests/unit/
uv run pytest tests/integration/

# 커버리지와 함께 실행
uv run pytest --cov=src

# 개발 의존성과 함께 테스트 실행
uv run --extra dev pytest
```

#### 코드 품질
```bash
# 코드 포맷팅
uv run black src/ tests/

# import 정렬
uv run isort src/ tests/

# 코드 린팅
uv run flake8 src/ tests/

# 타입 체킹
uv run mypy src/
```

#### 문서화
```bash
# 문서 생성
uv run --extra docs sphinx-build -b html docs/ docs/_build/html

# 문서 보기
open docs/_build/html/index.html
```

#### 패키지 관리
```bash
# 새 의존성 추가
uv add package-name

# 개발 의존성 추가
uv add --dev package-name

# 선택적 의존성 추가
uv add --optional docs sphinx

# 의존성 업데이트
uv sync --upgrade

# 의존성 트리 보기
uv tree
```

## 기여하기

1. 저장소를 포크합니다
2. 기능 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m '멋진 기능 추가'`)
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다

## 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 감사의 말

- LangGraph: 에이전트 프레임워크 제공
- FastAPI: 웹 프레임워크 제공
- Slack Bolt: 봇 통합 제공
- SQLAlchemy: 데이터베이스 작업 제공
- 다양한 LLM 제공업체: AI 기능 제공

## 지원

문의사항이나 지원이 필요한 경우:
- 저장소에 이슈를 생성하세요
- `docs/` 폴더의 문서를 확인하세요
- 애플리케이션 실행 시 `/docs`에서 API 문서를 확인하세요

---

**지능적인 데이터베이스 쿼리를 위해 ❤️로 제작되었습니다**