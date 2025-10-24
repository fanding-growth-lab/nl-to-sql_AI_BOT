# 테스트 및 디버깅 가이드

## 개요

이 가이드는 NL-to-SQL 변환 시스템의 테스트 방법과 디버깅 기법을 설명합니다. 단위 테스트, 통합 테스트, 성능 테스트부터 일반적인 문제 해결까지 포괄적으로 다룹니다.

## 테스트 구조

### 테스트 디렉토리 구조
```
tests/
├── unit/                    # 단위 테스트
│   ├── test_agentic_flow_nodes.py
│   ├── test_core_config.py
│   └── test_slack_bot.py
├── integration/             # 통합 테스트
│   └── test_pipeline_integration.py
├── conftest.py             # 테스트 설정
└── __init__.py
```

### 테스트 분류
- **단위 테스트**: 개별 모듈/함수 테스트
- **통합 테스트**: 모듈 간 상호작용 테스트
- **성능 테스트**: 응답 시간, 메모리 사용량 테스트
- **보안 테스트**: SQL 인젝션, 권한 테스트

## 단위 테스트

### 1. 기본 테스트 실행

#### 전체 단위 테스트
```bash
# 모든 단위 테스트 실행
pytest tests/unit/ -v

# 특정 파일 테스트
pytest tests/unit/test_agentic_flow_nodes.py -v

# 특정 함수 테스트
pytest tests/unit/test_agentic_flow_nodes.py::test_nl_processor -v
```

#### 커버리지 포함 테스트
```bash
# 커버리지 측정
pytest tests/unit/ --cov=src --cov-report=html

# 커버리지 보고서 확인
open htmlcov/index.html
```

### 2. 핵심 모듈 테스트

#### 인텐트 분류기 테스트
```python
# tests/unit/test_intent_classifier.py
import pytest
from src.agentic_flow.llm_intent_classifier import LLMIntentClassifier

class TestIntentClassifier:
    def test_data_query_classification(self):
        """데이터 쿼리 분류 테스트"""
        classifier = LLMIntentClassifier(config)
        result = classifier.classify_intent("8월 신규 가입자 수를 알려줘")
        
        assert result.intent == "DATA_QUERY"
        assert result.confidence > 0.8
    
    def test_greeting_classification(self):
        """인사말 분류 테스트"""
        classifier = LLMIntentClassifier(config)
        result = classifier.classify_intent("안녕하세요")
        
        assert result.intent == "GREETING"
        assert result.confidence > 0.9
    
    def test_general_chat_classification(self):
        """일반 대화 분류 테스트"""
        classifier = LLMIntentClassifier(config)
        result = classifier.classify_intent("오늘 날씨가 어떤가요?")
        
        assert result.intent == "GENERAL_CHAT"
        assert result.confidence > 0.7
```

#### 엔티티 추출기 테스트
```python
# tests/unit/test_entity_extractor.py
import pytest
from src.agentic_flow.enhanced_entity_extractor import EnhancedEntityExtractor

class TestEntityExtractor:
    def test_basic_entity_extraction(self):
        """기본 엔티티 추출 테스트"""
        extractor = EnhancedEntityExtractor()
        result = extractor.extract_entities("8월 신규 가입자 수")
        
        assert result.total_entities > 0
        assert any(entity.value == "8월" for entity in result.entities)
        assert any(entity.value == "가입자" for entity in result.entities)
    
    def test_composite_entity_extraction(self):
        """복합 엔티티 추출 테스트"""
        extractor = EnhancedEntityExtractor()
        result = extractor.extract_entities("상위 5 크리에이터")
        
        composite_entities = [e for e in result.entities if e.source == 'composite']
        assert len(composite_entities) > 0
        assert any("상위" in entity.value for entity in composite_entities)
    
    def test_entity_importance_scoring(self):
        """엔티티 중요도 점수 테스트"""
        extractor = EnhancedEntityExtractor()
        result = extractor.extract_entities("매출 분석 리포트")
        
        # 비즈니스 용어는 높은 중요도
        business_entities = [e for e in result.entities if e.entity_type == 'business']
        assert all(entity.importance_score > 0.5 for entity in business_entities)
```

#### 스키마 매퍼 테스트
```python
# tests/unit/test_schema_mapper.py
import pytest
from src.agentic_flow.schema_mapper import SchemaMapper

class TestSchemaMapper:
    def test_table_mapping(self):
        """테이블 매핑 테스트"""
        mapper = SchemaMapper(config)
        entities = ["회원", "가입자"]
        results = mapper.map_to_tables(entities)
        
        assert len(results) > 0
        assert any(result.type == "table" for result in results)
        assert any("member" in result.name.lower() for result in results)
    
    def test_column_mapping(self):
        """컬럼 매핑 테스트"""
        mapper = SchemaMapper(config)
        entities = ["가입일", "상태"]
        results = mapper.map_to_columns(entities)
        
        assert len(results) > 0
        assert any(result.type == "column" for result in results)
        assert any("datetime" in result.name.lower() for result in results)
    
    def test_mapping_confidence(self):
        """매핑 신뢰도 테스트"""
        mapper = SchemaMapper(config)
        entities = ["회원", "가입자"]
        results = mapper.map_entities(entities)
        
        # 높은 신뢰도 매핑이 있어야 함
        assert any(result.confidence > 0.8 for result in results)
```

#### SQL 생성기 테스트
```python
# tests/unit/test_sql_generator.py
import pytest
from src.agentic_flow.dynamic_sql_generator import DynamicSQLGenerator

class TestSQLGenerator:
    def test_simple_sql_generation(self):
        """단순 SQL 생성 테스트"""
        generator = DynamicSQLGenerator(config)
        schema_info = {
            "tables": ["t_member"],
            "columns": {"t_member": ["member_no", "status"]}
        }
        
        result = generator.generate_dynamic_sql("회원 수", schema_info)
        
        assert result.sql_query is not None
        assert "SELECT" in result.sql_query.upper()
        assert "t_member" in result.sql_query
        assert result.confidence > 0.5
    
    def test_complex_sql_generation(self):
        """복잡한 SQL 생성 테스트"""
        generator = DynamicSQLGenerator(config)
        schema_info = {
            "tables": ["t_member", "t_member_login_log"],
            "columns": {
                "t_member": ["member_no", "status"],
                "t_member_login_log": ["member_no", "login_datetime"]
            }
        }
        
        result = generator.generate_dynamic_sql("8월 신규 가입자 수", schema_info)
        
        assert result.sql_query is not None
        assert "JOIN" in result.sql_query.upper() or "WHERE" in result.sql_query.upper()
        assert result.confidence > 0.3
    
    def test_json_parsing_robustness(self):
        """JSON 파싱 강건성 테스트"""
        generator = DynamicSQLGenerator(config)
        
        # 잘못된 JSON 형식 테스트
        malformed_json = '{"sql_query": "SELECT * FROM test", "confidence": 0.8,}'
        result = generator._extract_json_from_text(malformed_json)
        
        assert result is not None
        assert "sql_query" in result
        assert result["sql_query"] == "SELECT * FROM test"
```

## 통합 테스트

### 1. 전체 파이프라인 테스트

#### 기본 파이프라인 테스트
```python
# tests/integration/test_pipeline_integration.py
import pytest
from src.agentic_flow.nodes import NLProcessor, SchemaMapper, DynamicSQLGenerator

class TestPipelineIntegration:
    def test_end_to_end_pipeline(self):
        """전체 파이프라인 테스트"""
        # 1. 자연어 처리
        processor = NLProcessor(config)
        state = {"query": "8월 신규 가입자 수를 알려줘"}
        result = processor.process(state)
        
        assert result["intent"] == "DATA_QUERY"
        assert len(result["entities"]) > 0
        
        # 2. 스키마 매핑
        mapper = SchemaMapper(config)
        mapping_results = mapper.map_entities(result["entities"])
        
        assert len(mapping_results) > 0
        assert any(result.type == "table" for result in mapping_results)
        
        # 3. SQL 생성
        generator = DynamicSQLGenerator(config)
        sql_result = generator.generate_dynamic_sql(
            result["query"], 
            {"tables": ["t_member"], "columns": {"t_member": ["member_no"]}}
        )
        
        assert sql_result.sql_query is not None
        assert "SELECT" in sql_result.sql_query.upper()
    
    def test_error_handling_pipeline(self):
        """오류 처리 파이프라인 테스트"""
        processor = NLProcessor(config)
        
        # 잘못된 입력 테스트
        state = {"query": ""}
        result = processor.process(state)
        
        assert result["intent"] == "GENERAL_CHAT"  # Fallback
        
        # SQL 생성 실패 테스트
        generator = DynamicSQLGenerator(config)
        result = generator.generate_dynamic_sql("", {})
        
        assert result.sql_query is not None  # Fallback SQL
        assert result.confidence < 0.5
```

### 2. 데이터베이스 연동 테스트

#### 데이터베이스 연결 테스트
```python
# tests/integration/test_database_integration.py
import pytest
from src.core.db import get_db_connection
from src.agentic_flow.async_sql_executor import AsyncSQLExecutor

class TestDatabaseIntegration:
    def test_database_connection(self):
        """데이터베이스 연결 테스트"""
        connection = get_db_connection()
        assert connection is not None
        
        # 연결 테스트
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result[0] == 1
        
        cursor.close()
        connection.close()
    
    def test_sql_execution(self):
        """SQL 실행 테스트"""
        executor = AsyncSQLExecutor(config)
        
        # 단순 쿼리 실행
        result = executor.execute_sql("SELECT COUNT(*) FROM t_member")
        
        assert result is not None
        assert "rows" in result
        assert result["execution_time"] > 0
    
    def test_complex_sql_execution(self):
        """복잡한 SQL 실행 테스트"""
        executor = AsyncSQLExecutor(config)
        
        # 조인 쿼리 실행
        sql = """
        SELECT COUNT(*) as member_count
        FROM t_member m
        LEFT JOIN t_member_login_log l ON m.member_no = l.member_no
        WHERE m.status = 'A'
        """
        
        result = executor.execute_sql(sql)
        
        assert result is not None
        assert result["rows"] > 0
        assert result["execution_time"] < 5.0  # 5초 이내
```

## 성능 테스트

### 1. 응답 시간 테스트

#### 단위 성능 테스트
```python
# tests/performance/test_response_time.py
import pytest
import time
from src.agentic_flow.enhanced_entity_extractor import EnhancedEntityExtractor

class TestPerformance:
    def test_entity_extraction_performance(self):
        """엔티티 추출 성능 테스트"""
        extractor = EnhancedEntityExtractor()
        
        # 성능 측정
        start_time = time.time()
        result = extractor.extract_entities("8월 신규 가입자 수를 알려줘")
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # 성능 기준: 1초 이내
        assert execution_time < 1.0
        assert result.total_entities > 0
    
    def test_sql_generation_performance(self):
        """SQL 생성 성능 테스트"""
        generator = DynamicSQLGenerator(config)
        
        queries = [
            "8월 신규 가입자 수",
            "지난달 대비 이번달 매출 증가율",
            "상위 5 크리에이터"
        ]
        
        for query in queries:
            start_time = time.time()
            result = generator.generate_dynamic_sql(query, schema_info)
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # 성능 기준: 5초 이내
            assert execution_time < 5.0
            assert result.sql_query is not None
```

### 2. 메모리 사용량 테스트

#### 메모리 사용량 테스트
```python
# tests/performance/test_memory_usage.py
import pytest
import psutil
import os
from src.agentic_flow.enhanced_entity_extractor import EnhancedEntityExtractor

class TestMemoryUsage:
    def test_memory_usage_during_extraction(self):
        """엔티티 추출 중 메모리 사용량 테스트"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        extractor = EnhancedEntityExtractor()
        
        # 여러 번 추출 수행
        for _ in range(100):
            result = extractor.extract_entities("8월 신규 가입자 수를 알려줘")
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # 메모리 증가량: 100MB 이내
        assert memory_increase < 100
    
    def test_memory_leak_detection(self):
        """메모리 누수 감지 테스트"""
        process = psutil.Process(os.getpid())
        
        # 초기 메모리
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # 반복 실행
        for i in range(1000):
            extractor = EnhancedEntityExtractor()
            result = extractor.extract_entities(f"테스트 쿼리 {i}")
            del extractor, result
        
        # 가비지 컬렉션
        import gc
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # 메모리 누수: 50MB 이내
        assert memory_increase < 50
```

## 디버깅 기법

### 1. 로그 기반 디버깅

#### 로그 레벨 설정
```python
# 디버깅용 로그 설정
import logging

# 로그 레벨 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# 특정 모듈 로그 설정
logger = logging.getLogger('src.agentic_flow.nodes')
logger.setLevel(logging.DEBUG)
```

#### 로그 분석
```bash
# 오류 로그 확인
grep "ERROR" logs/datatalk.log

# 특정 모듈 로그 확인
grep "NLProcessor" logs/datatalk.log

# 성능 관련 로그 확인
grep "execution_time" logs/datatalk.log
```

### 2. 단계별 디버깅

#### 파이프라인 단계별 디버깅
```python
# 단계별 디버깅 함수
def debug_pipeline_step_by_step(query: str):
    """파이프라인 단계별 디버깅"""
    
    # 1. 인텐트 분류 디버깅
    print("=== 1. 인텐트 분류 ===")
    processor = NLProcessor(config)
    state = {"query": query}
    result = processor.process(state)
    print(f"인텐트: {result['intent']}")
    print(f"신뢰도: {result.get('confidence', 'N/A')}")
    
    # 2. 엔티티 추출 디버깅
    print("\n=== 2. 엔티티 추출 ===")
    extractor = EnhancedEntityExtractor()
    entities = extractor.extract_entities(query)
    print(f"추출된 엔티티 수: {entities.total_entities}")
    for entity in entities.entities[:5]:
        print(f"  - {entity.value} (타입: {entity.entity_type}, 중요도: {entity.importance_score:.3f})")
    
    # 3. 스키마 매핑 디버깅
    print("\n=== 3. 스키마 매핑 ===")
    mapper = SchemaMapper(config)
    mapping_results = mapper.map_entities([e.value for e in entities.entities])
    print(f"매핑 결과 수: {len(mapping_results)}")
    for result in mapping_results[:5]:
        print(f"  - {result.entity} -> {result.name} (타입: {result.type}, 신뢰도: {result.confidence:.3f})")
    
    # 4. SQL 생성 디버깅
    print("\n=== 4. SQL 생성 ===")
    generator = DynamicSQLGenerator(config)
    sql_result = generator.generate_dynamic_sql(query, schema_info)
    print(f"생성된 SQL: {sql_result.sql_query}")
    print(f"신뢰도: {sql_result.confidence:.3f}")
    print(f"생성 방법: {sql_result.generation_method}")

# 사용 예시
debug_pipeline_step_by_step("8월 신규 가입자 수를 알려줘")
```

### 3. 오류 상황 디버깅

#### 일반적인 오류 처리
```python
# 오류 처리 디버깅 함수
def debug_error_scenarios():
    """오류 상황 디버깅"""
    
    # 1. 빈 입력 처리
    print("=== 빈 입력 테스트 ===")
    try:
        processor = NLProcessor(config)
        result = processor.process({"query": ""})
        print(f"빈 입력 결과: {result['intent']}")
    except Exception as e:
        print(f"빈 입력 오류: {e}")
    
    # 2. 잘못된 SQL 처리
    print("\n=== 잘못된 SQL 테스트 ===")
    try:
        executor = AsyncSQLExecutor(config)
        result = executor.execute_sql("INVALID SQL")
        print(f"잘못된 SQL 결과: {result}")
    except Exception as e:
        print(f"잘못된 SQL 오류: {e}")
    
    # 3. LLM API 오류 처리
    print("\n=== LLM API 오류 테스트 ===")
    try:
        generator = DynamicSQLGenerator(config)
        result = generator.generate_dynamic_sql("테스트", {})
        print(f"LLM 오류 결과: {result.sql_query}")
    except Exception as e:
        print(f"LLM 오류: {e}")
```

## 테스트 자동화

### 1. CI/CD 파이프라인 설정

#### GitHub Actions 설정
```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

### 2. 테스트 스케줄링

#### 정기 테스트 실행
```bash
# cron 설정 (Linux/macOS)
# 매일 오전 2시에 테스트 실행
0 2 * * * cd /path/to/project && python -m pytest tests/ --cov=src

# Windows Task Scheduler 설정
# 작업 스케줄러에서 Python 스크립트 실행
```

## 성능 모니터링

### 1. 성능 메트릭 수집

#### 성능 모니터링 설정
```python
# performance_monitor.py
import time
import psutil
import logging
from functools import wraps

def monitor_performance(func):
    """성능 모니터링 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        logging.info(f"Function: {func.__name__}")
        logging.info(f"Execution time: {execution_time:.3f}s")
        logging.info(f"Memory usage: {memory_usage:.3f}MB")
        
        return result
    return wrapper

# 사용 예시
@monitor_performance
def extract_entities(query: str):
    extractor = EnhancedEntityExtractor()
    return extractor.extract_entities(query)
```

### 2. 성능 임계값 설정

#### 성능 기준 설정
```python
# performance_thresholds.py
PERFORMANCE_THRESHOLDS = {
    'entity_extraction': {
        'max_time': 1.0,  # 1초
        'max_memory': 50  # 50MB
    },
    'sql_generation': {
        'max_time': 5.0,  # 5초
        'max_memory': 100  # 100MB
    },
    'sql_execution': {
        'max_time': 10.0,  # 10초
        'max_memory': 200  # 200MB
    }
}

def check_performance_threshold(operation: str, execution_time: float, memory_usage: float):
    """성능 임계값 확인"""
    thresholds = PERFORMANCE_THRESHOLDS.get(operation, {})
    
    if execution_time > thresholds.get('max_time', float('inf')):
        logging.warning(f"{operation} 실행 시간 초과: {execution_time:.3f}s > {thresholds['max_time']}s")
    
    if memory_usage > thresholds.get('max_memory', float('inf')):
        logging.warning(f"{operation} 메모리 사용량 초과: {memory_usage:.3f}MB > {thresholds['max_memory']}MB")
```

## 문제 해결

### 1. 일반적인 문제

#### 테스트 실패 문제
```bash
# 의존성 문제 해결
pip install --upgrade -r requirements.txt

# 가상 환경 재생성
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 캐시 클리어
pytest --cache-clear
```

#### 성능 문제
```bash
# 메모리 사용량 확인
ps aux | grep python

# CPU 사용량 확인
top -p $(pgrep -f "python")

# 디스크 사용량 확인
df -h
```

### 2. 디버깅 도구

#### 프로파일링 도구
```python
# cProfile 사용
import cProfile
import pstats

def profile_function(func):
    """함수 프로파일링"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = func()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
    
    return result
```

#### 메모리 프로파일링
```python
# memory_profiler 사용
from memory_profiler import profile

@profile
def extract_entities_with_profiling(query: str):
    extractor = EnhancedEntityExtractor()
    return extractor.extract_entities(query)
```

---

**테스트 및 디버깅에 대한 추가 정보는 [문제 해결 가이드](troubleshooting.md)를 참조하세요.**

