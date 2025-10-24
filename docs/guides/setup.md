# 개발 환경 설정 가이드

## 개요

이 가이드는 NL-to-SQL 변환 시스템의 개발 환경을 설정하는 방법을 단계별로 설명합니다. 새로운 개발자가 프로젝트를 쉽게 설정하고 실행할 수 있도록 상세한 안내를 제공합니다.

## 필수 요구사항

### 시스템 요구사항
- **운영체제**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **메모리**: 최소 8GB RAM (권장: 16GB+)
- **저장공간**: 최소 10GB 여유 공간
- **네트워크**: 인터넷 연결 (API 호출용)

### 소프트웨어 요구사항
- **Python**: 3.9 이상 (권장: 3.11)
- **Node.js**: 16 이상 (문서 자동화용)
- **Git**: 2.30 이상
- **Docker**: 20.10 이상 (선택사항)

## 1. 저장소 클론 및 초기 설정

### 1.1 저장소 클론
```bash
# 저장소 클론
git clone https://github.com/your-org/nl-to-sql-project.git
cd nl-to-sql-project

# 최신 코드 확인
git checkout main
git pull origin main
```

### 1.2 프로젝트 구조 확인
```bash
# 프로젝트 구조 확인
tree -L 3
# 또는
ls -la
```

**예상 구조:**
```
nl-to-sql-project/
├── src/                    # 소스 코드
│   ├── agentic_flow/      # 핵심 로직
│   ├── core/              # 인프라
│   └── slack/             # Slack 봇
├── docs/                 # 문서
├── tests/                 # 테스트
├── requirements.txt       # Python 의존성
├── .env.template         # 환경 변수 템플릿
└── README.md             # 프로젝트 설명
```

## 2. Python 환경 설정

### 2.1 Python 버전 확인
```bash
# Python 버전 확인
python --version
# 또는
python3 --version

# Python 3.9 이상이어야 함
```

### 2.2 가상 환경 생성
```bash
# 가상 환경 생성
python -m venv venv

# 가상 환경 활성화
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 가상 환경 확인
which python
# Windows: where python
```

### 2.3 의존성 설치
```bash
# pip 업그레이드
python -m pip install --upgrade pip

# 의존성 설치
pip install -r requirements.txt

# 설치 확인
pip list
```

## 3. 환경 변수 설정

### 3.1 환경 변수 파일 생성
```bash
# 템플릿 복사
cp .env.template .env

# 파일 편집 (선호하는 에디터 사용)
# Windows
notepad .env
# macOS
open -e .env
# Linux
nano .env
```

### 3.2 필수 환경 변수 설정
```bash
# .env 파일 내용 예시
# LLM API 키
GOOGLE_API_KEY=your_google_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# 데이터베이스 설정
DB_HOST=localhost
DB_PORT=3306
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password

# Slack 설정
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token
SLACK_SIGNING_SECRET=your_signing_secret

# 애플리케이션 설정
DEBUG=True
LOG_LEVEL=INFO
```

### 3.3 API 키 획득 방법

#### Google API 키
1. [Google Cloud Console](https://console.cloud.google.com/) 접속
2. 프로젝트 생성 또는 선택
3. "API 및 서비스" > "사용 설정된 API" 이동
4. "Generative AI API" 활성화
5. "사용자 인증 정보" > "API 키" 생성

#### Anthropic API 키
1. [Anthropic Console](https://console.anthropic.com/) 접속
2. 계정 생성 또는 로그인
3. "API Keys" 섹션에서 새 키 생성

#### OpenAI API 키
1. [OpenAI Platform](https://platform.openai.com/) 접속
2. 계정 생성 또는 로그인
3. "API Keys" 섹션에서 새 키 생성

## 4. 데이터베이스 설정

### 4.1 MySQL/MariaDB 설치

#### Windows
```bash
# MySQL 설치 (Chocolatey 사용)
choco install mysql

# 또는 MySQL Installer 다운로드
# https://dev.mysql.com/downloads/installer/
```

#### macOS
```bash
# Homebrew 사용
brew install mysql

# 서비스 시작
brew services start mysql
```

#### Ubuntu/Debian
```bash
# MySQL 설치
sudo apt update
sudo apt install mysql-server

# 서비스 시작
sudo systemctl start mysql
sudo systemctl enable mysql
```

### 4.2 데이터베이스 생성
```sql
-- MySQL 접속
mysql -u root -p

-- 데이터베이스 생성
CREATE DATABASE nl_to_sql_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 사용자 생성
CREATE USER 'nl_to_sql_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON nl_to_sql_db.* TO 'nl_to_sql_user'@'localhost';
FLUSH PRIVILEGES;

-- 데이터베이스 선택
USE nl_to_sql_db;
```

### 4.3 스키마 생성
```sql
-- 예시 테이블 생성
CREATE TABLE t_member (
    member_no INT PRIMARY KEY AUTO_INCREMENT,
    member_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE,
    status CHAR(1) DEFAULT 'A',
    ins_datetime DATETIME DEFAULT CURRENT_TIMESTAMP,
    upd_datetime DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE t_member_login_log (
    log_no INT PRIMARY KEY AUTO_INCREMENT,
    member_no INT NOT NULL,
    login_datetime DATETIME DEFAULT CURRENT_TIMESTAMP,
    login_type VARCHAR(20),
    FOREIGN KEY (member_no) REFERENCES t_member(member_no)
);

-- 샘플 데이터 삽입
INSERT INTO t_member (member_name, email, status) VALUES
('홍길동', 'hong@example.com', 'A'),
('김철수', 'kim@example.com', 'A'),
('이영희', 'lee@example.com', 'I');

INSERT INTO t_member_login_log (member_no, login_type) VALUES
(1, 'web'),
(2, 'mobile'),
(1, 'web');
```

## 5. Slack 봇 설정

### 5.1 Slack 앱 생성
1. [Slack API](https://api.slack.com/apps) 접속
2. "Create New App" 클릭
3. "From scratch" 선택
4. 앱 이름과 워크스페이스 선택

### 5.2 봇 권한 설정
1. "OAuth & Permissions" 탭 이동
2. "Scopes" 섹션에서 다음 권한 추가:
   - `app_mentions:read`
   - `channels:history`
   - `chat:write`
   - `im:history`
   - `im:read`
   - `im:write`

### 5.3 이벤트 구독 설정
1. "Event Subscriptions" 탭 이동
2. "Enable Events" 토글 활성화
3. "Request URL" 설정 (로컬 개발 시 ngrok 사용)
4. "Subscribe to bot events" 섹션에서 다음 이벤트 추가:
   - `app_mention`
   - `message.im`

### 5.4 ngrok 설정 (로컬 개발용)
```bash
# ngrok 설치
# https://ngrok.com/download

# ngrok 실행
ngrok http 8000

# 생성된 URL을 Slack 앱의 Request URL에 설정
# 예: https://abc123.ngrok.io/slack/events
```

## 6. 애플리케이션 실행

### 6.1 설정 확인
```bash
# 환경 변수 확인
python -c "import os; print('GOOGLE_API_KEY:', 'OK' if os.getenv('GOOGLE_API_KEY') else 'MISSING')"

# 데이터베이스 연결 확인
python -c "from src.core.db import get_db_connection; print('DB 연결:', 'OK' if get_db_connection() else 'FAILED')"
```

### 6.2 애플리케이션 실행
```bash
# 개발 모드로 실행
python src/main.py

# 또는
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 6.3 실행 확인
```bash
# API 엔드포인트 확인
curl http://localhost:8000/health

# Slack 봇 테스트
# Slack 채널에서 @봇이름 안녕하세요
```

## 7. 개발 도구 설정

### 7.1 코드 포맷터 설정
```bash
# Black 설치
pip install black

# 설정 파일 생성
echo "[tool.black]
line-length = 88
target-version = ['py39']
" > pyproject.toml
```

### 7.2 린터 설정
```bash
# flake8 설치
pip install flake8

# 설정 파일 생성
echo "[flake8]
max-line-length = 88
ignore = E203, W503
" > .flake8
```

### 7.3 IDE 설정 (VS Code)
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

## 8. 테스트 환경 설정

### 8.1 테스트 데이터베이스 설정
```sql
-- 테스트용 데이터베이스 생성
CREATE DATABASE nl_to_sql_test CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
GRANT ALL PRIVILEGES ON nl_to_sql_test.* TO 'nl_to_sql_user'@'localhost';
```

### 8.2 테스트 실행
```bash
# 단위 테스트 실행
pytest tests/unit/ -v

# 통합 테스트 실행
pytest tests/integration/ -v

# 전체 테스트 실행
pytest tests/ -v

# 커버리지 포함 테스트
pytest tests/ --cov=src --cov-report=html
```

## 9. 문제 해결

### 9.1 일반적인 문제

#### Python 버전 문제
```bash
# Python 버전 확인
python --version

# 가상 환경 재생성
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 의존성 설치 실패
```bash
# pip 업그레이드
python -m pip install --upgrade pip

# 캐시 클리어
pip cache purge

# 개별 설치
pip install package_name --no-cache-dir
```

#### 데이터베이스 연결 실패
```bash
# 연결 정보 확인
python -c "
from src.core.config import get_config
config = get_config()
print(f'DB Host: {config.DB_HOST}')
print(f'DB Port: {config.DB_PORT}')
print(f'DB Name: {config.DB_NAME}')
"

# 데이터베이스 서비스 확인
# Windows
net start mysql
# macOS/Linux
sudo systemctl status mysql
```

#### Slack 봇 응답 없음
```bash
# 봇 토큰 확인
python -c "
import os
print('SLACK_BOT_TOKEN:', 'OK' if os.getenv('SLACK_BOT_TOKEN') else 'MISSING')
"

# ngrok 상태 확인
curl http://localhost:4040/api/tunnels
```

### 9.2 로그 확인
```bash
# 애플리케이션 로그 확인
tail -f logs/datatalk.log

# 특정 레벨 로그 확인
grep "ERROR" logs/datatalk.log
grep "WARNING" logs/datatalk.log
```

### 9.3 성능 문제
```bash
# 메모리 사용량 확인
ps aux | grep python

# CPU 사용량 확인
top -p $(pgrep -f "python src/main.py")

# 데이터베이스 연결 확인
mysql -u nl_to_sql_user -p -e "SHOW PROCESSLIST;"
```

## 10. 추가 설정

### 10.1 Docker 설정 (선택사항)
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "src/main.py"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DB_HOST=db
    depends_on:
      - db
  
  db:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: rootpassword
      MYSQL_DATABASE: nl_to_sql_db
    ports:
      - "3306:3306"
```

### 10.2 모니터링 설정
```bash
# Prometheus 설정 (선택사항)
pip install prometheus-client

# Grafana 설정 (선택사항)
# https://grafana.com/docs/grafana/latest/installation/
```

## 11. 다음 단계

### 11.1 개발 시작
- [기본 사용법 가이드](basic_usage.md) 참조
- [테스트 가이드](testing.md) 참조
- [성능 최적화 가이드](performance.md) 참조

### 11.2 추가 학습
- [모듈별 문서](../modules/) 참조
- [아키텍처 문서](../architecture.md) 참조
- [문제 해결 가이드](troubleshooting.md) 참조

### 11.3 기여하기
- [기여 가이드](../CONTRIBUTING.md) 참조
- [코드 스타일 가이드](../STYLE.md) 참조
- [이슈 리포트](../issues) 참조

---

**문제가 발생하면 [문제 해결 가이드](troubleshooting.md)를 참조하거나 [GitHub Issues](https://github.com/your-org/nl-to-sql-project/issues)에 문의하세요.**

