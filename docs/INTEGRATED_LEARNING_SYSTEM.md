# 통합 학습 시스템 (Integrated Learning System) 적용 가이드

## 개요

Intent 분류 통계 시스템과 AutoLearning 시스템을 통합하여, 모든 사용자 쿼리의 전체 라이프사이클을 추적하고 학습하는 시스템입니다.

---

## 시스템 흐름도

```
사용자 쿼리 입력
    ↓
LangGraph 파이프라인 실행
    ↓
[Intent 분류] → LLMIntentClassifier
    ↓
[SQL 생성] → SQLGenerationNode  
    ↓
[SQL 검증] → SQLValidationNode
    ↓
[SQL 실행] → SQLExecutionNode
    ↓
[결과 요약] → DataSummarizationNode
    ↓
runner.py: _record_integrated_metrics() 호출  ⭐ 여기서 통합 시작
    ↓
LearningDataIntegrator.record_complete_query_interaction()
    ├─→ StatisticsCollector (Intent 통계 기록)
    └─→ AutoLearningSystem (쿼리 패턴 학습)
```

---

## 자동 적용 과정

### 1. 쿼리 처리 시 자동 기록

모든 사용자 쿼리가 처리될 때마다 자동으로 통합 데이터가 기록됩니다:

```python
# src/agentic_flow/graph/runner.py

def process_query(self, user_query: str, ...):
    # 파이프라인 실행
    final_state = self.graph.invoke(...)
    
    # 결과 생성
    result = self._create_execution_result(...)
    
    # ⭐ 통합 메트릭 자동 기록
    self._record_integrated_metrics(
        final_state=final_state,
        result=result,
        user_id=user_id,
        session_id=session_id,
        execution_time_ms=execution_time * 1000
    )
```

### 2. 통합 메트릭스 생성

```python
# src/agentic_flow/graph/runner.py

metrics = QueryInteractionMetrics(
    # 기본 정보
    user_query="4월 신규 회원 현황 알려줘",
    user_id="U12345",
    session_id="session_abc",
    
    # Intent 분류 정보
    intent="DATA_QUERY",
    intent_confidence=0.95,
    intent_reasoning="사용자가 특정 기간의 데이터를 요청하는 쿼리",
    
    # 처리 결과
    sql_query="SELECT ... FROM t_member_info WHERE ...",
    validation_passed=True,
    execution_success=True,
    execution_result_count=150,
    
    # 성능
    response_time_ms=1250.0,
    
    # 학습 정보
    template_used="new_members_specific_month",
    mapping_result={...},
    
    timestamp=time.time()
)
```

### 3. 두 시스템에 분산 기록

```python
# src/agentic_flow/intent_classification_stats.py

def record_complete_query_interaction(self, metrics):
    # 1️⃣ StatisticsCollector에 기록 (Intent 통계)
    self.stats_collector.record_classification(
        intent=metrics.intent,
        confidence=metrics.intent_confidence,
        response_time_ms=metrics.response_time_ms,
        is_error=metrics.is_error
    )
    
    # 2️⃣ AutoLearningSystem에 기록 (쿼리 패턴 학습)
    self.auto_learning_system.record_query_interaction(
        user_id=metrics.user_id,
        query=metrics.user_query,
        mapping_result=metrics.mapping_result,
        confidence=metrics.intent_confidence,
        success=metrics.execution_success,
        user_feedback=metrics.user_feedback
    )
```

---

## 데이터 저장 위치

### StatisticsCollector
- **메모리**: `StatisticsCollector._stats` (IntentClassifierStats)
- **영구 저장**: `statistics_persistence.py`의 SQLite/JSON 저장소
- **경로**: `.taskmaster/stats/` (설정 가능)

### AutoLearningSystem  
- **메모리**: `AutoLearningSystem.query_patterns`, `user_behaviors`
- **영구 저장**: `learning_data.json`
- **경로**: `src/agentic_flow/learning_data.json`

---

## 사용 예시

### 1. 통합 인사이트 조회

```python
from agentic_flow.intent_classification_stats import get_integrator

integrator = get_integrator()
insights = integrator.get_unified_insights()

print(insights)
# {
#     "intent_classification": {
#         "total_classifications": 1250,
#         "average_confidence": 0.87,
#         "error_rate": 5.2,
#         "intent_distribution": {
#             "DATA_QUERY": 850,
#             "GREETING": 200,
#             "SCHEMA_QUERY": 200
#         }
#     },
#     "query_learning": {
#         "total_queries": 1250,
#         "success_rate": 94.8,
#         "total_patterns": 45,
#         "avg_confidence": 0.87
#     },
#     "performance_metrics": {
#         "queue_size": 5,
#         "memory_usage_percent": 12.3,
#         "batch_processing_rate": 25.5
#     }
# }
```

### 2. 쿼리 패턴 분석

```python
patterns = integrator.get_query_pattern_analysis()

print(patterns)
# {
#     "intent_patterns": {
#         "distribution": {...},
#         "confidence_by_intent": {...}
#     },
#     "query_patterns": {
#         "total_patterns": 45,
#         "most_common": [...],
#         "high_success_rate": [...]
#     },
#     "correlations": {...}  # 향후 구현
# }
```

### 3. 최적화 제안 조회

```python
suggestions = integrator.optimize_based_on_data()

for suggestion in suggestions:
    print(f"💡 {suggestion}")

# 💡 Intent 분류 오류율이 5.2%로 관리 가능 범위입니다.
# 💡 평균 신뢰도가 0.87로 양호합니다.
# 💡 "4월" 키워드가 포함된 쿼리의 성공률이 낮습니다. SQL 템플릿 개선을 고려하세요.
```

---

## API 엔드포인트 (main.py)

### 통합 인사이트 API

```python
# GET /stats/integrated/insights
@app.get("/stats/integrated/insights")
async def get_integrated_insights():
    integrator = get_integrator()
    return integrator.get_unified_insights()
```

### 패턴 분석 API

```python
# GET /stats/integrated/patterns
@app.get("/stats/integrated/patterns")
async def get_pattern_analysis():
    integrator = get_integrator()
    return integrator.get_query_pattern_analysis()
```

### 최적화 제안 API

```python
# GET /stats/integrated/optimizations
@app.get("/stats/integrated/optimizations")
async def get_optimizations():
    integrator = get_integrator()
    return integrator.optimize_based_on_data()
```

---

## 장점

### 1. 자동화
- ✅ 코드 변경 불필요: 모든 쿼리가 자동으로 기록됨
- ✅ 투명성: 파이프라인 실행 중 자동 기록

### 2. 데이터 통합
- ✅ Intent 분류와 쿼리 학습 데이터가 하나의 구조로 관리
- ✅ 상관관계 분석 가능 (예: 특정 Intent의 쿼리가 특정 패턴으로 매핑되는 빈도)

### 3. 학습 효율성
- ✅ 전체 쿼리 라이프사이클 추적
- ✅ Intent 분류 정확도와 실행 성공률 연계 분석
- ✅ 사용자별 선호도와 패턴 분석

### 4. 통합 인사이트
- ✅ 두 시스템 데이터를 한 번에 조회
- ✅ 종합적인 최적화 제안 생성

---

## 모니터링

### 로그 확인

```bash
# 통합 메트릭 기록 로그
grep "Recorded integrated metrics" logs/datatalk.log

# 통합 인사이트 조회 로그
grep "get_unified_insights" logs/datatalk.log
```

### 성능 영향

- **기록 비용**: 비동기 처리로 메인 파이프라인에 영향 최소화
- **메모리 사용**: 배치 처리 및 지속성 저장으로 효율적 관리
- **디스크 사용**: 설정 가능한 데이터 보존 정책

---

## 향후 개선 사항

1. **상관관계 분석**: Intent 분류와 쿼리 패턴 간 상관관계 분석
2. **실시간 대시보드**: 웹 대시보드로 통합 인사이트 시각화
3. **자동 최적화**: 제안된 최적화를 자동으로 적용하는 기능
4. **A/B 테스팅**: 다른 Intent 분류 모델 비교 분석

