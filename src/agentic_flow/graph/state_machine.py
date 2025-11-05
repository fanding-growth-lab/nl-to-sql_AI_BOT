"""
LangGraph State Machine Core Implementation

This module implements the core state machine logic for the NL-to-SQL pipeline,
including state definitions, node connections, and conditional routing.
"""

import time
import logging
from typing import Dict, List, Optional, Any, TypedDict, Union, Callable
from dataclasses import dataclass
from enum import Enum

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ..nodes import (
    NLProcessor, SQLGenerationNode, 
    SQLValidationNode, DataSummarizationNode
)
# SchemaMapper 제거: RAGSchemaRetrieverNode에 통합됨
# from ..nodes import SchemaMapper
from ..llm_intent_classifier import LLMIntentClassifier
# DynamicSQLGenerator 통합으로 인한 import 제거
# from ..dynamic_sql_generator import DynamicSQLGenerator
from ..rag_schema_retriever import RAGSchemaRetrieverNode
from ..validation_node import ValidationNode
from ..user_review_node import UserReviewNode, ReviewStatus
from ..python_code_generator import PythonCodeGeneratorNode
from ..code_executor import CodeExecutorNode
from ..result_integrator import ResultIntegratorNode
from ..hybrid_query_processor import HybridQueryProcessor
from ..monitoring import PipelineMonitor
# 공통 유틸리티 함수 import
from ..utils import is_intent_equal, is_intent_in
from core.config import get_settings
from core.db import get_db_session, execute_query
from core.logging import get_logger

logger = get_logger(__name__)


class ExecutionStatus(Enum):
    """Execution status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class NodeType(Enum):
    """Node type enumeration."""
    LLM_INTENT_CLASSIFICATION = "llm_intent_classification"
    NLP_PROCESSING = "nlp_processing"
    # SCHEMA_MAPPING = "schema_mapping"  # 제거: RAGSchemaRetrieverNode에 통합됨
    # DynamicSQLGenerator 통합으로 인한 노드 제거
    # DYNAMIC_SQL_GENERATION = "dynamic_sql_generation"
    SQL_GENERATION = "sql_generation"
    SQL_VALIDATION = "sql_validation"
    VALIDATION_CHECK = "validation_check"
    USER_REVIEW = "user_review"
    SQL_EXECUTION = "sql_execution"
    DATA_SUMMARIZATION = "data_summarization"


@dataclass
class NodeExecutionResult:
    """Result of node execution."""
    node_type: NodeType
    success: bool
    execution_time: float
    confidence: float
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentState(TypedDict):
    """
    State definition for the LangGraph pipeline.
    
    This TypedDict defines all the data that flows through the pipeline,
    including input, intermediate results, and output.
    """
    # Input
    user_query: str
    user_id: Optional[str]
    channel_id: Optional[str]
    session_id: Optional[str]
    context: Optional[Dict[str, Any]]
    
    # Processing stages
    normalized_query: Optional[str]
    intent: Optional[str]
    llm_intent_result: Optional[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    agent_schema_mapping: Optional[Dict[str, Any]]
    sql_query: Optional[str]
    validated_sql: Optional[str]
    sql_params: Optional[Dict[str, Any]]  # SQL 파라미터 (SQL Injection 방지)
    query_result: List[Dict[str, Any]]
    data_summary: Optional[str]
    
    # Conversation handling
    skip_sql_generation: Optional[bool]
    conversation_response: Optional[str]
    needs_clarification: Optional[bool]  # 사용자 재입력이 필요한지 표시
    
    # Fanding templates
    fanding_template: Optional[Any]
    
    # Validation and error handling
    validation_result: Optional[Dict[str, Any]]
    processing_decision: Optional[Dict[str, Any]]
    is_valid: bool
    error_message: Optional[str]
    retry_count: int
    max_retries: int
    
    # Execution tracking
    current_node: Optional[str]
    execution_status: str
    node_results: List[NodeExecutionResult]
    
    # Metadata
    processing_time: float
    execution_time: float
    confidence_scores: Dict[str, float]
    debug_info: Dict[str, Any]
    
    # RAG schema retrieval
    rag_schema_chunks: Optional[List[Dict[str, Any]]]  # RAG schema retrieval results
    rag_schema_context: Optional[str]  # Formatted schema context for prompts
    
    # Conversation history for context awareness
    conversation_history: Optional[List[Dict[str, str]]]  # [{"role": "user|assistant", "content": "..."}]
    
    # Additional fields for review
    review_status: Optional[str]
    review_result: Optional[Any]
    
    # Output
    final_sql: Optional[str]
    explanation: Optional[str]
    success: bool


def _create_performance_wrapper(node_name: str, node_func: Callable[[AgentState], AgentState]) -> Callable[[AgentState], AgentState]:
    """
    노드 실행 시간 측정 및 성능 메타데이터 수집 래퍼
    
    Args:
        node_name: 노드 이름
        node_func: 실제 노드 함수
        
    Returns:
        성능 측정 기능이 추가된 노드 함수
    """
    def wrapper(state: AgentState) -> AgentState:
        node_start_time = time.time()
        state["current_node"] = node_name
        
        try:
            result = node_func(state)
            node_execution_time = time.time() - node_start_time
            
            # 성능 메타데이터 추가
            # debug_info가 없으면 초기화, 있으면 기존 것을 사용
            if "debug_info" not in result:
                result["debug_info"] = {}
            elif result["debug_info"] is None:
                result["debug_info"] = {}
            
            # node_performance 초기화 (기존 데이터 유지)
            if "node_performance" not in result["debug_info"]:
                result["debug_info"]["node_performance"] = {}
            
            # 노드별 추가 메타데이터 수집
            metadata = {
                "execution_time": node_execution_time,
                "success": True
            }
            
            # 노드별 특수 메트릭 추가
            if node_name == "rag_schema_retrieval":
                metadata["chunks_retrieved"] = len(result.get("rag_schema_chunks", []))
            elif node_name == "sql_generation":
                if result.get("sql_query"):
                    metadata["sql_length"] = len(result.get("sql_query", ""))
                    metadata["sql_generated"] = True
                else:
                    metadata["sql_generated"] = False
            # DynamicSQLGenerator 통합으로 인한 노드 제거
            # dynamic_sql_result는 이제 SQLGenerationNode에서 생성되므로
            # sql_generation 노드의 메타데이터에서 처리됨
            elif node_name == "llm_intent_classification":
                intent_result = result.get("llm_intent_result")
                if intent_result:
                    metadata["intent"] = intent_result.get("intent", "UNKNOWN")
                    metadata["confidence"] = intent_result.get("confidence", 0.0)
            elif node_name == "validation_check":
                validation_result = result.get("validation_result", {})
                if isinstance(validation_result, dict):
                    metadata["validation_passed"] = validation_result.get("status") != "FAILED"
                    metadata["validation_confidence"] = validation_result.get("confidence", 0.0)
            
            result["debug_info"]["node_performance"][node_name] = metadata
            
            logger.debug(f"Node {node_name} completed in {node_execution_time:.3f}s")
            return result
            
        except Exception as e:
            node_execution_time = time.time() - node_start_time
            logger.error(f"Node {node_name} failed after {node_execution_time:.3f}s: {str(e)}")
            
            # 에러 정보를 debug_info에 추가
            if "debug_info" not in state:
                state["debug_info"] = {}
            if "node_performance" not in state["debug_info"]:
                state["debug_info"]["node_performance"] = {}
            
            state["debug_info"]["node_performance"][node_name] = {
                "execution_time": node_execution_time,
                "success": False,
                "error": str(e)
            }
            
            # 에러 상태 설정
            state["error_message"] = f"{node_name} failed: {str(e)}"
            return state
    
    return wrapper


def _identify_bottlenecks(node_performance: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    노드 성능 데이터에서 병목 지점 식별
    
    Args:
        node_performance: 노드별 성능 메트릭
        
    Returns:
        병목 지점 분석 결과
    """
    if not node_performance:
        return {
            "bottleneck_nodes": [],
            "total_time": 0.0,
            "analysis": "No performance data available"
        }
    
    # 실행 시간 기준으로 정렬
    node_times = {
        node_name: metrics.get("execution_time", 0.0)
        for node_name, metrics in node_performance.items()
    }
    
    total_time = sum(node_times.values())
    
    if total_time == 0:
        return {
            "bottleneck_nodes": [],
            "total_time": 0.0,
            "analysis": "No execution time data available"
        }
    
    # 실행 시간 기준 정렬
    sorted_nodes = sorted(node_times.items(), key=lambda x: x[1], reverse=True)
    
    # 총 시간의 20% 이상을 차지하는 노드를 병목으로 식별
    bottleneck_threshold = total_time * 0.2
    bottleneck_nodes = [
        {
            "node_name": node_name,
            "execution_time": exec_time,
            "percentage": (exec_time / total_time) * 100,
            "metrics": node_performance[node_name]
        }
        for node_name, exec_time in sorted_nodes
        if exec_time >= bottleneck_threshold
    ]
    
    # 평균 실행 시간 계산
    avg_time = total_time / len(node_times)
    
    # 최적화 제안 생성
    optimization_suggestions = []
    for node in bottleneck_nodes:
        node_name = node["node_name"]
        percentage = node["percentage"]
        
        if percentage > 50:
            optimization_suggestions.append({
                "node": node_name,
                "priority": "high",
                "suggestion": f"{node_name}가 전체 실행 시간의 {percentage:.1f}%를 차지합니다. 캐싱 또는 최적화를 고려하세요."
            })
        elif percentage > 30:
            optimization_suggestions.append({
                "node": node_name,
                "priority": "medium",
                "suggestion": f"{node_name}가 전체 실행 시간의 {percentage:.1f}%를 차지합니다. 최적화 여지가 있습니다."
            })
    
    return {
        "bottleneck_nodes": bottleneck_nodes,
        "total_time": total_time,
        "average_node_time": avg_time,
        "slowest_node": sorted_nodes[0][0] if sorted_nodes else None,
        "fastest_node": sorted_nodes[-1][0] if sorted_nodes else None,
        "optimization_suggestions": optimization_suggestions,
        "node_count": len(node_times)
    }


def create_agent_graph(
    db_schema: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    checkpointer: Optional[Any] = None
) -> Any:
    """
    Create the LangGraph state machine for NL-to-SQL processing.
    
    Args:
        db_schema: Database schema information
        config: Configuration parameters
        
    Returns:
        Compiled StateGraph instance
    """
    logger.info("Creating LangGraph state machine")
    
    # Get configuration
    settings = get_settings()
    
    # db_schema가 제공되지 않으면 get_cached_db_schema()로 로드 (성능 최적화)
    if db_schema is None or len(db_schema) == 0:
        from core.db import get_cached_db_schema
        db_schema = get_cached_db_schema()
        logger.debug("db_schema was not provided, loaded from cache during graph creation")
    
    if config is None:
        config = {
            "llm": {
                "model": settings.llm.model,
                "api_key": settings.llm.api_key,
                "temperature": settings.llm.temperature,
                "max_tokens": settings.llm.max_tokens
            },
            "db_schema": db_schema,  # 이미 로드된 db_schema 사용
            "max_retries": settings.pipeline.max_retries,
            "confidence_threshold": settings.pipeline.confidence_threshold,
            "enable_debug": settings.pipeline.enable_debug,
            "enable_monitoring": settings.pipeline.enable_monitoring
        }
    else:
        # config가 제공되었지만 db_schema가 없으면 추가
        if "db_schema" not in config or not config.get("db_schema") or len(config.get("db_schema", {})) == 0:
            config["db_schema"] = db_schema
            logger.debug("db_schema was missing in provided config, added from parameter")
    
    # Create the state graph
    graph = StateGraph(AgentState)
    
    # Initialize nodes
    nodes = _initialize_nodes(config)
    
    # Add nodes to the graph with performance monitoring wrappers
    # 모든 노드에 성능 측정 래퍼 적용
    graph.add_node(
        "llm_intent_classification",
        _create_performance_wrapper("llm_intent_classification", lambda state: nodes["llm_intent_classifier"].process(state))
    )
    
    graph.add_node(
        "nlp_processing",
        _create_performance_wrapper("nlp_processing", lambda state: nodes["nlp_processor"].process(state))
    )
    
    # schema_mapping 노드 제거: RAGSchemaRetrieverNode에 통합됨
    # RAG 노드: 에러 발생 시 폴백 처리 및 성능 모니터링 포함
    def _rag_node_with_fallback(state: AgentState) -> AgentState:
        """RAG 노드 실행, 에러 처리 및 성능 모니터링"""
        node_start_time = time.time()
        state["current_node"] = "rag_schema_retrieval"
        
        try:
            result = nodes["rag_schema_retriever"].process(state)
            node_execution_time = time.time() - node_start_time
            
            # 성능 메타데이터 추가
            if "debug_info" not in result:
                result["debug_info"] = {}
            if "node_performance" not in result["debug_info"]:
                result["debug_info"]["node_performance"] = {}
            
            result["debug_info"]["node_performance"]["rag_schema_retrieval"] = {
                "execution_time": node_execution_time,
                "success": True,
                "chunks_retrieved": len(result.get("rag_schema_chunks", []))
            }
            
            logger.debug(f"RAG retrieval completed in {node_execution_time:.3f}s, retrieved {len(result.get('rag_schema_chunks', []))} chunks")
            return result
        except Exception as e:
            node_execution_time = time.time() - node_start_time
            logger.warning(f"RAG retrieval failed after {node_execution_time:.3f}s: {e}, continuing without RAG context")
            
            # 에러 발생 시 RAG 필드를 빈 값으로 설정하고 계속 진행
            state["rag_schema_chunks"] = []
            state["rag_schema_context"] = None
            
            # 에러 정보를 debug_info에 추가
            if "debug_info" not in state:
                state["debug_info"] = {}
            if "node_performance" not in state["debug_info"]:
                state["debug_info"]["node_performance"] = {}
            
            state["debug_info"]["node_performance"]["rag_schema_retrieval"] = {
                "execution_time": node_execution_time,
                "success": False,
                "error": str(e),
                "chunks_retrieved": 0
            }
            
            return state
    
    graph.add_node("rag_schema_retrieval", _rag_node_with_fallback)
    
    # dynamic_sql_generation 노드 제거 (SQLGenerationNode에 통합됨)
    # graph.add_node(
    #     "dynamic_sql_generation",
    #     _create_performance_wrapper("dynamic_sql_generation", lambda state: nodes["dynamic_sql_generator"].process(state))
    # )

    graph.add_node(
        "sql_generation",
        _create_performance_wrapper("sql_generation", lambda state: nodes["sql_generator"].process(state))
    )

    graph.add_node(
        "sql_validation",
        _create_performance_wrapper("sql_validation", lambda state: nodes["sql_validator"].process(state))
    )
    
    graph.add_node(
        "validation_check",
        _create_performance_wrapper("validation_check", lambda state: nodes["validation_node"].process(state))
    )
    
    graph.add_node(
        "user_review",
        _create_performance_wrapper("user_review", lambda state: nodes["user_review_node"].process(state))
    )
    
    graph.add_node(
        "sql_execution",
        _create_performance_wrapper("sql_execution", _execute_sql_query)
    )
    
    graph.add_node(
        "data_summarization",
        _create_performance_wrapper("data_summarization", lambda state: nodes["data_summarizer"].process(state))
    )
    
    # Python 경로 노드 추가 (하이브리드 시스템)
    graph.add_node(
        "python_code_generation",
        _create_performance_wrapper("python_code_generation", lambda state: nodes["python_code_generator"].process(state))
    )
    
    graph.add_node(
        "code_execution",
        _create_performance_wrapper("code_execution", lambda state: nodes["code_executor"].process(state))
    )
    
    graph.add_node(
        "result_integration",
        _create_performance_wrapper("result_integration", lambda state: nodes["result_integrator"].process(state))
    )
    
    # Define the flow
    graph.add_edge(START, "llm_intent_classification")
    
    # Conditional edge after intent classification (3갈래 분기: 중앙 관제탑/관문)
    # 최적화: CHAT_PATH는 nlp_processing을 건너뛰고 바로 data_summarization으로 직행
    graph.add_conditional_edges(
        "llm_intent_classification",
        route_after_intent_classification,
        {
            "CHAT_PATH": "data_summarization",  # CHAT_PATH: 인사/도움말 (즉시 응답, nlp_processing 건너뛰기)
            "SQL_PATH": "nlp_processing",       # SQL_PATH: SIMPLE_AGGREGATION (Fast Path)
            "PYTHON_PATH": "nlp_processing"     # PYTHON_PATH: COMPLEX_ANALYSIS (Safe Path)
        }
    )
    
    # Conditional edge after NLP processing
    # 데이터 경로만 여기를 거침 (CHAT_PATH는 이미 llm_intent_classification에서 바로 data_summarization으로 감)
    # 최적화: schema_mapping 노드 제거, nlp_processing에서 직접 rag_schema_retrieval로 연결
    graph.add_conditional_edges(
        "nlp_processing",
        route_after_nlp_processing,
        {
            "SQL_PATH": "rag_schema_retrieval",        # SQL_PATH: SIMPLE_AGGREGATION (RAG 스키마 검색으로 직접 이동)
            "PYTHON_PATH": "rag_schema_retrieval",     # PYTHON_PATH: COMPLEX_ANALYSIS (RAG 스키마 검색으로 직접 이동)
            "end": END                                  # 조기 종료 (명확화 질문 등)
        }
    )
    
    # === SQL_PATH 플로우 (Fast Path ⚡) ===
    # schema_mapping 노드 제거됨, rag_schema_retrieval이 schema_mapping 기능 포함
    
    # RAG 검색 후 intent 기반 라우팅
    # COMPLEX_ANALYSIS: SQL 생성 건너뛰고 바로 python_code_generation으로
    # SIMPLE_AGGREGATION: sql_generation으로
    graph.add_conditional_edges(
        "rag_schema_retrieval",
        route_after_rag_retrieval,
        {
            "sql_generation": "sql_generation",  # SQL_PATH: SIMPLE_AGGREGATION
            "python_code_generation": "python_code_generation"  # PYTHON_PATH: COMPLEX_ANALYSIS
        }
    )
    
    # 더 단순한 대안 - 조건부 로직 완전 제거 (향후 고려)
    # graph.add_edge("rag_schema_retrieval", "sql_generation")  # 모든 경우에 직접 연결
    
    # dynamic_sql_generation 노드 제거로 인한 엣지 제거
    # graph.add_edge("dynamic_sql_generation", "sql_generation")
    
    # SQL 생성 후: SQL_PATH만 여기를 거침 (PYTHON_PATH는 이미 rag_schema_retrieval에서 python_code_generation으로 라우팅됨)
    # 하지만 예외 상황(fallback)을 대비해 python_code_generation 경로도 포함
    graph.add_conditional_edges(
        "sql_generation",
        route_after_sql_generation,
        {
            "sql_validation": "sql_validation",  # SQL_PATH: Fast Path (검증 후 바로 실행)
            "python_code_generation": "python_code_generation"  # Fallback: COMPLEX_ANALYSIS가 여기에 도달한 경우 (일반적으로 발생하지 않아야 함)
        }
    )
    
    # === PYTHON_PATH 플로우 (Safe Path 🐍) ===
    # Python 경로는 RAG를 거쳐 python_code_generator로 직접 이동
    # python_code_generator가 data_gathering_sql과 python_code를 모두 생성
    # 이후 sql_execution → code_execution 순서로 진행
    
    # SQL 검증 후 라우팅
    # Phase 1: SIMPLE_AGGREGATION Fast Path - 검증 통과 시 바로 실행
    graph.add_conditional_edges(
        "sql_validation",
        route_after_validation,
        {
            "retry": "sql_generation",  # 재시도
            "validate": "validation_check",  # 낮은 신뢰도 또는 기타 의도 (안전장치)
            "sql_execution": "sql_execution"  # Fast Path: SIMPLE_AGGREGATION 검증 통과 시 바로 실행
        }
    )
    
    # 검증 체크 후 라우팅
    graph.add_conditional_edges(
        "validation_check",
        route_after_validation_check,
        {
            "sql_execution": "sql_execution",  # 자동 승인 또는 사용자 승인
            "user_review": "user_review",  # 사용자 검토 필요
            "reject": "nlp_processing",  # 거부 (재시작)
            "result_integration": "result_integration"  # 이미 SQL 실행된 경우 (일반적으로 발생하지 않음)
        }
    )
    
    # 사용자 검토 후 라우팅
    graph.add_conditional_edges(
        "user_review",
        route_after_user_review,
        {
            "sql_execution": "sql_execution",  # 사용자 승인
            "reject": "nlp_processing",  # 거부 (재시작)
            "modify": "sql_generation",  # 수정 (SQL 재생성)
            "pending": END  # 대기 중
        }
    )
    
    # === SQL 실행 후 라우팅 (SQL_PATH vs PYTHON_PATH 분기) ===
    # SQL_PATH: sql_execution → result_integration
    # PYTHON_PATH: sql_execution → code_execution (Python 코드 실행)
    graph.add_conditional_edges(
        "sql_execution",
        route_after_sql_execution,
        {
            "result_integration": "result_integration",  # SQL_PATH: 결과 통합
            "code_execution": "code_execution"           # PYTHON_PATH: Python 코드 실행
        }
    )
    
    # === PYTHON_PATH 플로우 계속 (Safe Path 🐍) ===
    # python_code_generation에서 data_gathering_sql과 python_code를 생성
    # 다음 노드: sql_execution (data_gathering_sql 실행)
    graph.add_edge("python_code_generation", "sql_execution")
    
    # Python 경로용 SQL 실행: data_gathering_sql 실행 (간단한 SQL이므로 복잡한 검증 불필요)
    # sql_execution 노드는 기존 것을 재사용 (의도에 따라 간단/복잡 검증 선택)
    
    # Python 코드 실행 후: 성공 시 결과 통합, 실패 시 SQL 경로로 폴백
    graph.add_conditional_edges(
        "code_execution",
        route_after_code_execution,
        {
            "result_integration": "result_integration",  # 성공: 결과 통합
            "sql_validation": "sql_validation"  # 실패: SQL 경로로 폴백 (검증부터 시작)
        }
    )
    
    # === 공통 종료 지점 ===
    # 결과 통합: SQL 경로와 Python 경로 모두 여기로 수렴
    graph.add_edge("result_integration", "data_summarization")
    graph.add_edge("data_summarization", END)
    
    # Compile the graph with memory
    # Use provided checkpointer if available, otherwise create new one
    memory = checkpointer or MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)
    
    logger.info("LangGraph state machine created successfully")
    return compiled_graph


def _initialize_nodes(config: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize all pipeline nodes."""
    logger.info("Initializing pipeline nodes")
    
    # Log config summary (without db_schema details to avoid cluttering logs)
    config_summary = {k: v for k, v in config.items() if k != "db_schema"}
    if "db_schema" in config:
        db_schema = config.get("db_schema", {})
        config_summary["db_schema"] = f"<{len(db_schema)} tables>"
    logger.debug(f"Config passed to _initialize_nodes: {config_summary}")
    
    # LLM 서비스를 config에 추가
    from agentic_flow.llm_service import get_llm_service
    llm_service = get_llm_service()
    config["llm_service"] = llm_service
    
    nodes = {
        "llm_intent_classifier": LLMIntentClassifier(config),
        "nlp_processor": NLProcessor(config),
        # "schema_mapper": SchemaMapper(config),  # 제거: RAGSchemaRetrieverNode에 통합됨
        "rag_schema_retriever": RAGSchemaRetrieverNode(config),
        # DynamicSQLGenerator 제거 (SQLGenerationNode에 통합됨)
        # "dynamic_sql_generator": DynamicSQLGenerator(config),
        "sql_generator": SQLGenerationNode(config),
        "sql_validator": SQLValidationNode(config),
        "validation_node": ValidationNode(),  # 독립적으로 작동, config 불필요
        "user_review_node": UserReviewNode(),  # 독립적으로 작동, config 불필요
        "python_code_generator": PythonCodeGeneratorNode(config),  # COMPLEX_ANALYSIS용 Python 코드 생성
        "code_executor": CodeExecutorNode(config),  # Python 코드 실행 (샌드박스)
        "result_integrator": ResultIntegratorNode(config),  # SQL/Python 결과 통합
        "data_summarizer": DataSummarizationNode(config)
    }
    
    logger.info(f"Initialized {len(nodes)} pipeline nodes")
    return nodes


def _get_default_schema() -> Dict[str, Any]:
    """Get default database schema."""
    return {
        "t_member": {
            "description": "회원 정보 테이블",
            "columns": {
                "id": {"type": "int", "description": "회원 ID"},
                "email": {"type": "varchar", "description": "이메일 주소"},
                "nickname": {"type": "varchar", "description": "닉네임"},
                "status": {"type": "varchar", "description": "회원 상태"},
                "created_at": {"type": "timestamp", "description": "가입일"}
            }
        },
        "t_creator": {
            "description": "크리에이터 정보 테이블",
            "columns": {
                "id": {"type": "int", "description": "크리에이터 ID"},
                "nickname": {"type": "varchar", "description": "크리에이터 닉네임"},
                "description": {"type": "text", "description": "크리에이터 소개"},
                "category": {"type": "varchar", "description": "카테고리"}
            }
        },
        "t_funding": {
            "description": "펀딩 프로젝트 테이블",
            "columns": {
                "id": {"type": "int", "description": "프로젝트 ID"},
                "title": {"type": "varchar", "description": "프로젝트 제목"},
                "goal_amount": {"type": "int", "description": "목표 금액"},
                "current_amount": {"type": "int", "description": "현재 모금액"},
                "status": {"type": "varchar", "description": "프로젝트 상태"},
                "created_at": {"type": "timestamp", "description": "생성일"}
            }
        }
    }


def route_after_intent_classification(state: AgentState) -> str:
    """
    인텐트 분류 후 라우팅 결정 (3갈래 분기: CHAT_PATH, SQL_PATH, PYTHON_PATH)
    
    Phase 2 최적화: llm_intent_classification에서 바로 3개의 명확한 경로로 분기
    
    경로:
    - CHAT_PATH: 비데이터 의도 (GREETING, HELP_REQUEST, GENERAL_CHAT)
    - SQL_PATH: SIMPLE_AGGREGATION (Fast Path 적용)
    - PYTHON_PATH: COMPLEX_ANALYSIS (Safe Path)
    
    Args:
        state: Current pipeline state
        
    Returns:
        Next node name ("CHAT_PATH", "SQL_PATH", "PYTHON_PATH")
    """
    from agentic_flow.state import QueryIntent
    
    llm_intent_result = state.get("llm_intent_result")
    conversation_response = state.get("conversation_response")
    skip_sql = state.get("skip_sql_generation", False)
    
    # 라우팅 결정을 debug_info에 기록
    if "debug_info" not in state:
        state["debug_info"] = {}
    if "routing_decisions" not in state["debug_info"]:
        state["debug_info"]["routing_decisions"] = {}
    
    # LLM 분류 실패 시 SQL_PATH로 진행 (fallback, 안전한 기본값)
    if not llm_intent_result:
        logger.warning("LLM intent classification failed, defaulting to SQL_PATH for safety")
        state["debug_info"]["routing_decisions"]["intent_classification"] = {
            "decision": "SQL_PATH",
            "reason": "llm_classification_failed"
        }
        return "SQL_PATH"
    
    # 인텐트 추출
    intent_str = llm_intent_result.get("intent", "").upper()
    try:
        intent = QueryIntent(intent_str)
    except ValueError:
        logger.warning(f"Unknown intent: {intent_str}, defaulting to SQL_PATH for safety")
        state["debug_info"]["routing_decisions"]["intent_classification"] = {
            "decision": "SQL_PATH",
            "reason": "unknown_intent"
        }
        return "SQL_PATH"
    
    # CHAT_PATH: 비데이터 의도 (인사/도움말/일반 대화)
    # 최적화: nlp_processing을 건너뛰고 바로 data_summarization으로 직행
    # 응답 생성은 data_summarization 노드에서 수행 (봇 기능에 맞춘 응답)
    # 공통 유틸리티 함수 사용
    if is_intent_in(intent, [QueryIntent.GREETING, QueryIntent.HELP_REQUEST, QueryIntent.GENERAL_CHAT]):
        logger.info(f"CHAT_PATH: Non-data intent ({intent.value}) detected, routing directly to data_summarization (skipping nlp_processing)")
        state["skip_sql_generation"] = True
        state["debug_info"]["routing_decisions"]["intent_classification"] = {
            "decision": "CHAT_PATH",
            "reason": f"non_data_intent_{intent.value}_direct_to_summarization",
            "intent": intent.value,
            "optimization": "skipped_nlp_processing",
            "note": "Response will be generated in data_summarization node with bot-specific information"
        }
        return "CHAT_PATH"
    
    # SQL_PATH: SIMPLE_AGGREGATION (Fast Path)
    # 공통 유틸리티 함수 사용
    elif is_intent_equal(intent, QueryIntent.SIMPLE_AGGREGATION):
        logger.info(f"SQL_PATH: SIMPLE_AGGREGATION detected, routing to SQL path")
        state["debug_info"]["routing_decisions"]["intent_classification"] = {
            "decision": "SQL_PATH",
            "reason": "simple_aggregation_intent",
            "intent": intent.value
        }
        return "SQL_PATH"
    
    # PYTHON_PATH: COMPLEX_ANALYSIS (Safe Path)
    # 공통 유틸리티 함수 사용
    elif is_intent_equal(intent, QueryIntent.COMPLEX_ANALYSIS):
        logger.info(f"PYTHON_PATH: COMPLEX_ANALYSIS detected, routing to Python path")
        state["debug_info"]["routing_decisions"]["intent_classification"] = {
            "decision": "PYTHON_PATH",
            "reason": "complex_analysis_intent",
            "intent": intent.value
        }
        return "PYTHON_PATH"
    
    # 기타: SQL_PATH로 진행 (fallback, 안전한 기본값)
    else:
        logger.warning(f"Unknown intent ({intent.value}), defaulting to SQL_PATH for safety")
        state["debug_info"]["routing_decisions"]["intent_classification"] = {
            "decision": "SQL_PATH",
            "reason": "unknown_intent_fallback",
            "intent": str(intent)
        }
        return "SQL_PATH"


def route_after_rag_retrieval(state: AgentState) -> str:
    """
    RAG 검색 후 라우팅 결정
    
    Intent를 확인하여 경로별로 분기:
    - COMPLEX_ANALYSIS: SQL 생성 건너뛰고 바로 python_code_generation으로
    - SIMPLE_AGGREGATION: sql_generation으로
    
    Args:
        state: Current pipeline state
        
    Returns:
        Next node name ("sql_generation" 또는 "python_code_generation")
    """
    from agentic_flow.state import QueryIntent
    
    intent = state.get("intent")
    rag_schema_chunks = state.get("rag_schema_chunks", [])
    rag_schema_context = state.get("rag_schema_context")
    
    # 라우팅 결정을 debug_info에 기록 (디버깅 목적)
    if "debug_info" not in state:
        state["debug_info"] = {}
    if "routing_decisions" not in state["debug_info"]:
        state["debug_info"]["routing_decisions"] = {}
    
    # RAG 결과 품질 평가 (디버깅 및 로깅 목적)
    high_quality_chunks = [
        chunk for chunk in rag_schema_chunks
        if isinstance(chunk, dict) and chunk.get("relevance_score", 0.0) >= 0.5
    ]
    
    avg_relevance = 0.0
    if high_quality_chunks:
        avg_relevance = sum(
            chunk.get("relevance_score", 0.0) for chunk in high_quality_chunks
        ) / len(high_quality_chunks)
    
    # PYTHON_PATH: SQL 생성 건너뛰고 바로 python_code_generation으로
    # 공통 유틸리티 함수 사용
    if is_intent_equal(intent, QueryIntent.COMPLEX_ANALYSIS):
        logger.info(
            f"PYTHON_PATH: Routing directly to python_code_generation "
            f"(skipping sql_generation, RAG chunks: {len(rag_schema_chunks)})"
        )
        decision = "python_code_generation"
        reason = "complex_analysis_intent_skip_sql_generation"
        
        # 디버깅 정보 기록
        state["debug_info"]["routing_decisions"]["rag_retrieval"] = {
            "decision": decision,
            "reason": reason,
            "intent": "COMPLEX_ANALYSIS",
            "chunks_count": len(rag_schema_chunks),
            "high_quality_chunks": len(high_quality_chunks),
            "avg_relevance": avg_relevance,
            "note": "SQL generation skipped for COMPLEX_ANALYSIS"
        }
        return decision
    
    # SQL_PATH: sql_generation으로
    # 공통 유틸리티 함수 사용
    if is_intent_equal(intent, QueryIntent.SIMPLE_AGGREGATION):
        if rag_schema_chunks and rag_schema_context:
            logger.info(
                f"SQL_PATH: RAG results found ({len(rag_schema_chunks)} chunks, "
                f"avg relevance: {avg_relevance:.3f}), proceeding to SQL generation"
            )
            decision = "sql_generation"
            reason = "rag_results_available"
        else:
            logger.info("SQL_PATH: No RAG results found, proceeding to SQL generation (fallback)")
            decision = "sql_generation"
            reason = "no_rag_results"
        
        # 디버깅 정보 기록
        state["debug_info"]["routing_decisions"]["rag_retrieval"] = {
            "decision": decision,
            "reason": reason,
            "intent": "SIMPLE_AGGREGATION",
            "chunks_count": len(rag_schema_chunks),
            "high_quality_chunks": len(high_quality_chunks),
            "avg_relevance": avg_relevance,
            "note": "Proceeding to SQL generation"
        }
        return decision
    
    # 기타/알 수 없음: 기본적으로 SQL 경로로 진행 (안전)
    logger.warning(f"Unknown intent ({intent}) after RAG retrieval, defaulting to SQL generation path")
    decision = "sql_generation"
    reason = "unknown_intent_fallback"
    
    # 디버깅 정보 기록
    state["debug_info"]["routing_decisions"]["rag_retrieval"] = {
        "decision": decision,
        "reason": reason,
        "intent": str(intent),
        "chunks_count": len(rag_schema_chunks),
        "high_quality_chunks": len(high_quality_chunks),
        "avg_relevance": avg_relevance,
        "note": "Unknown intent, defaulting to SQL generation"
    }
    
    return decision


def route_after_nlp_processing(state: AgentState) -> str:
    """
    NLP 처리 후 경로 분기 결정 (데이터 경로만 처리)
    
    최적화: CHAT_PATH는 이미 llm_intent_classification에서 바로 data_summarization으로 라우팅되었으므로,
    이 함수는 SQL_PATH와 PYTHON_PATH만 처리합니다.
    
    schema_mapping 노드 제거, rag_schema_retrieval로 직접 라우팅
    
    Args:
        state: Current pipeline state
        
    Returns:
        Next node name ("rag_schema_retrieval" or "end")
    """
    from agentic_flow.state import QueryIntent
    
    # 재입력 요청이 필요한 경우 조기 종료
    if state.get("needs_clarification", False):
        logger.info("Clarification needed, ending pipeline early")
        return "end"
    
    # 의도 확인 (NLP processing에서 설정됨)
    intent = state.get("intent")
    
    # 라우팅 결정을 debug_info에 기록
    if "debug_info" not in state:
        state["debug_info"] = {}
    if "routing_decisions" not in state["debug_info"]:
        state["debug_info"]["routing_decisions"] = {}
    
    # SQL_PATH: SIMPLE_AGGREGATION (Fast Path)
    # 공통 유틸리티 함수 사용
    if is_intent_equal(intent, QueryIntent.SIMPLE_AGGREGATION):
        logger.info("SQL_PATH: SIMPLE_AGGREGATION detected, routing to rag_schema_retrieval (schema_mapping integrated)")
        state["debug_info"]["routing_decisions"]["nlp_processing"] = {
            "decision": "SQL_PATH",
            "intent": "SIMPLE_AGGREGATION",
            "reason": "simple_aggregation_intent",
            "next_node": "rag_schema_retrieval",
            "note": "schema_mapping functionality integrated into rag_schema_retrieval"
        }
        return "SQL_PATH"
    
    # PYTHON_PATH: COMPLEX_ANALYSIS (Safe Path)
    # 공통 유틸리티 함수 사용
    elif is_intent_equal(intent, QueryIntent.COMPLEX_ANALYSIS):
        logger.info("PYTHON_PATH: COMPLEX_ANALYSIS detected, routing to rag_schema_retrieval (schema_mapping integrated)")
        state["debug_info"]["routing_decisions"]["nlp_processing"] = {
            "decision": "PYTHON_PATH",
            "intent": "COMPLEX_ANALYSIS",
            "reason": "complex_analysis_intent",
            "next_node": "rag_schema_retrieval",
            "note": "schema_mapping functionality integrated into rag_schema_retrieval"
        }
        return "PYTHON_PATH"  # Python 경로도 스키마 정보가 필요하므로 rag_schema_retrieval로 라우팅
    
    # CHAT_PATH는 여기에 도달하지 않아야 함 (이미 llm_intent_classification에서 처리됨)
    # 하지만 혹시 모를 경우를 대비한 fallback
    # 공통 유틸리티 함수 사용
    elif is_intent_in(intent, [QueryIntent.GREETING, QueryIntent.HELP_REQUEST, QueryIntent.GENERAL_CHAT]):
        logger.warning(f"CHAT_PATH reached nlp_processing (unexpected), this should have been handled earlier")
        # 이미 llm_intent_classification에서 처리되었어야 하는데 여기 도달했다면 오류
        # 하지만 안전을 위해 조기 종료
        return "end"
    
    # 기타/알 수 없음: 기본적으로 SQL 경로로 진행 (하위 호환성)
    else:
        logger.warning(f"Unknown or unclear intent ({intent}), defaulting to SQL_PATH for safety")
        state["debug_info"]["routing_decisions"]["nlp_processing"] = {
            "decision": "SQL_PATH",
            "intent": str(intent),
            "reason": "unknown_intent_fallback"
        }
        return "SQL_PATH"


def route_after_validation(state: AgentState) -> str:
    """
    Route after SQL validation based on validation result and retry count.
    
    Phase 1 최적화: SIMPLE_AGGREGATION 경로 Fast Path 구현
    - 검증 통과 시 validation_check 건너뛰고 바로 sql_execution으로 이동
    
    Args:
        state: Current pipeline state
        
    Returns:
        Next node name ("retry", "validate", "sql_execution")
    """
    from agentic_flow.state import QueryIntent
    
    validation_result = state.get("validation_result", {})
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    intent = state.get("intent")
    
    # Check if this is a conversation response (greeting, help, etc.)
    # 단, SQL이 성공적으로 생성되었으면 conversation_response를 무시하고 SQL 실행
    conversation_response = state.get("conversation_response")
    sql_query = state.get("sql_query")
    dynamic_sql_result = state.get("dynamic_sql_result", {})
    has_valid_sql = (
        sql_query or 
        (isinstance(dynamic_sql_result, dict) and dynamic_sql_result.get("sql_query"))
    )
    
    if conversation_response and not has_valid_sql:
        # SQL이 없고 conversation_response만 있는 경우 (GREETING, HELP_REQUEST 등)
        logger.info("Conversation response detected (no SQL), skipping validation and proceeding to execution")
        return "sql_execution"
    elif conversation_response and has_valid_sql:
        # SQL이 생성되었는데 conversation_response도 있는 경우 (clarification이 있었지만 SQL 생성 성공)
        logger.info("SQL successfully generated despite previous clarification request, clearing conversation_response and proceeding with SQL execution")
        state["conversation_response"] = None
        state["needs_clarification"] = False
    
    # SIMPLE_AGGREGATION Fast Path 강화
    # 검증 통과 시 validation_check 건너뛰고 바로 실행 (임계값 낮춤)
    # 공통 유틸리티 함수 사용
    if is_intent_equal(intent, QueryIntent.SIMPLE_AGGREGATION):
        if validation_result and validation_result.get("is_valid", False):
            confidence = validation_result.get("confidence", 0.0)
            
            # Fast Path 임계값을 0.5로 낮춤 (더 많은 쿼리가 빠른 경로로 처리됨)
            # 간단한 집계 쿼리는 기본적으로 빠른 경로로 처리
            if confidence >= 0.5:
                logger.info(
                    f"SIMPLE_AGGREGATION Fast Path: Validation passed "
                    f"(confidence: {confidence:.2f}), executing immediately"
                )
                return "sql_execution"
            
            # 매우 낮은 신뢰도만 validation_check로 이동 (안전장치)
            logger.warning(
                f"SIMPLE_AGGREGATION: Very low confidence ({confidence:.2f}), "
                f"proceeding to validation check for review"
            )
            return "validate"
    
    # 기타 의도 또는 SIMPLE_AGGREGATION 검증 실패 시 기존 로직
    # Check if validation passed
    if validation_result and validation_result.get("is_valid", False):
        logger.info("SQL validation passed, proceeding to validation check")
        return "validate"
    
    # Check retry count
    if retry_count < max_retries:
        logger.warning(f"SQL validation failed, retrying ({retry_count + 1}/{max_retries})")
        # 재시도 카운트 증가
        state["retry_count"] = retry_count + 1
        return "retry"
    else:
        logger.warning("Max retries exceeded, proceeding to validation check for final decision")
        # 재시도 카운트 초기화하고 validation_check에서 최종 결정
        state["retry_count"] = 0
        return "validate"


def route_after_validation_check(state: AgentState) -> str:
    """
    검증 체크 후 라우팅 결정
    
    SIMPLE_AGGREGATION 쿼리에 대해서는 간소화된 경로 적용:
    - 높은 신뢰도 또는 기본 검증 통과 시 자동 승인
    - UserReviewNode 건너뛰고 직접 SQL 실행
    
    Args:
        state: Current pipeline state
        
    Returns:
        str: 다음 노드 이름 ("sql_execution", "user_review", "reject")
    """
    from agentic_flow.state import QueryIntent
    
    validation_result = state.get("validation_result")
    processing_decision = state.get("processing_decision") or {}
    intent = state.get("intent")
    
    # SIMPLE_AGGREGATION 쿼리 간소화 처리 강화
    # 공통 유틸리티 함수 사용
    if is_intent_equal(intent, QueryIntent.SIMPLE_AGGREGATION):
        if not validation_result:
            logger.info("SIMPLE_AGGREGATION: No validation result, auto-approving")
            return "sql_execution"
        
        # 기본 SQL 구문 검증 통과 시 자동 승인
        if isinstance(validation_result, dict):
            is_valid = validation_result.get("is_valid", False)
            confidence = validation_result.get("confidence", 0)
            
            # Fast Path 임계값을 0.5로 낮춤 (더 많은 쿼리가 자동 승인됨)
            # 간단한 집계 쿼리는 기본적으로 자동 승인
            if is_valid and confidence >= 0.5:
                logger.info(
                    f"SIMPLE_AGGREGATION: Auto-approving simple query "
                    f"(confidence: {confidence:.2f}, valid: {is_valid})"
                )
                return "sql_execution"
            
            # 매우 낮은 신뢰도만 사용자 검토 (임계값 0.5 미만)
            if confidence < 0.5:
                logger.warning(
                    f"SIMPLE_AGGREGATION: Very low confidence ({confidence:.2f}), "
                    f"requiring user review"
                )
                return "user_review"
            
            # 검증 실패 시에도 중간 신뢰도면 자동 승인 (간소화)
            logger.info(
                f"SIMPLE_AGGREGATION: Auto-approving with intermediate confidence "
                f"({confidence:.2f}) despite validation status"
            )
            return "sql_execution"
    
    # COMPLEX_ANALYSIS 및 기타 쿼리는 기존 로직 유지
    if not validation_result:
        logger.warning("No validation result found, defaulting to user review")
        return "user_review"
    
    # 자동 승인 조건 확인 (높은 신뢰도)
    if processing_decision and processing_decision.get("auto_approve", False):
        logger.info("Auto-approving query based on high confidence")
        return "sql_execution"
    
    # 높은 신뢰도면 자동 승인
    if isinstance(validation_result, dict) and validation_result.get("confidence", 0) >= 0.85:
        confidence = validation_result.get("confidence", 0)
        logger.info(f"High confidence ({confidence:.2f}), auto-approving")
        return "sql_execution"
    
    # 재시도 횟수 체크 (무한 루프 방지)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    
    if retry_count >= max_retries:
        logger.error(f"Max retries ({max_retries}) exceeded, rejecting query")
        return "reject"
    
    # 검증 실패 시 사용자 검토로 이동 (강제 실행 제거)
    if isinstance(validation_result, dict):
        status_value = validation_result.get("status")
        if status_value and (hasattr(status_value, "value") and status_value.value == "rejected" or status_value == "rejected"):
            logger.warning("Query rejected due to validation issues, requiring user review")
            return "user_review"
    
    # 사용자 검토 필요 조건 확인
    if processing_decision and processing_decision.get("needs_user_review", False):
        logger.info("Query requires user review")
        return "user_review"
    
    # 기본적으로 사용자 검토로 이동
    logger.info("Defaulting to user review for safety")
    return "user_review"


def route_after_sql_generation(state: AgentState) -> str:
    """
    SQL 생성 후 라우팅 결정
    
    주의: 이 함수는 SQL_PATH (SIMPLE_AGGREGATION)에서만 호출됩니다.
    PYTHON_PATH (COMPLEX_ANALYSIS)는 이미 rag_schema_retrieval에서 python_code_generation으로 라우팅되었습니다.
    
    Args:
        state: Current pipeline state
        
    Returns:
        str: 다음 노드 이름 ("sql_validation")
    """
    from agentic_flow.state import QueryIntent
    
    intent = state.get("intent")
    
    # SQL_PATH: 검증 파이프라인으로 (Fast Path)
    # 공통 유틸리티 함수 사용
    if is_intent_equal(intent, QueryIntent.SIMPLE_AGGREGATION):
        logger.info("SQL_PATH: Routing to SQL validation (Fast Path)")
        return "sql_validation"
    
    # COMPLEX_ANALYSIS가 여기에 도달하면 안 됨 (이미 rag_schema_retrieval에서 처리됨)
    # 하지만 혹시 모를 경우를 대비한 fallback
    if is_intent_equal(intent, QueryIntent.COMPLEX_ANALYSIS):
        logger.warning("COMPLEX_ANALYSIS reached route_after_sql_generation (unexpected), this should have been handled earlier")
        # 안전을 위해 python_code_generation으로 라우팅 (하지만 일반적으로 발생하지 않아야 함)
        return "python_code_generation"
    
    # 기타: 기본적으로 SQL 검증 경로 (안전)
    logger.info(f"Unknown intent ({intent}), defaulting to SQL validation path")
    return "sql_validation"


def route_after_sql_execution(state: AgentState) -> str:
    """
    SQL 실행 후 라우팅 결정 (SQL_PATH vs PYTHON_PATH 분기)
    
    SQL_PATH (SIMPLE_AGGREGATION): result_integration으로
    PYTHON_PATH (COMPLEX_ANALYSIS): code_execution으로
    
    Args:
        state: Current pipeline state
        
    Returns:
        str: 다음 노드 이름 ("result_integration" 또는 "code_execution")
    """
    from agentic_flow.state import QueryIntent
    
    intent = state.get("intent")
    
    # PYTHON_PATH: Python 코드 실행 필요
    # 공통 유틸리티 함수 사용
    if is_intent_equal(intent, QueryIntent.COMPLEX_ANALYSIS):
        # python_code가 있으면 code_execution으로
        if state.get("python_code"):
            logger.info("PYTHON_PATH: Routing to code_execution after SQL execution")
            return "code_execution"
        else:
            logger.warning("PYTHON_PATH: No python_code found, routing to result_integration")
            return "result_integration"
    
    # SQL_PATH: 결과 통합으로
    logger.info("SQL_PATH: Routing to result_integration after SQL execution")
    return "result_integration"


def route_after_code_execution(state: AgentState) -> str:
    """
    Python 코드 실행 후 라우팅 결정 (단순화됨)
    
    성공 시 바로 결과 통합, 실패 시 SQL 경로로 폴백
    
    Args:
        state: Current pipeline state
        
    Returns:
        str: 다음 노드 이름 ("result_integration" 또는 "sql_validation")
    """
    python_execution_result = state.get("python_execution_result")
    
    if python_execution_result and python_execution_result.get("success"):
        # Python 실행 성공: 바로 결과 통합
        logger.info("Python execution successful, proceeding to result integration")
        return "result_integration"
    else:
        # Python 실행 실패: SQL 경로로 폴백 (검증부터 시작)
        logger.warning("Python execution failed, falling back to SQL validation path")
        error_msg = python_execution_result.get("error_message", "Unknown error") if python_execution_result else "Execution failed"
        logger.info(f"Python execution error: {error_msg}, switching to SQL validation path")
        return "sql_validation"


def route_after_user_review(state: AgentState) -> str:
    """
    사용자 검토 후 라우팅 결정
    
    Args:
        state: Current pipeline state
        
    Returns:
        str: 다음 노드 이름
    """
    review_status = state.get("review_status")
    review_result = state.get("review_result")
    
    # 자동 승인된 경우 처리
    if review_status == ReviewStatus.AUTO_APPROVED:
        logger.info("Query auto-approved, proceeding to execution")
        return "sql_execution"
    
    # PENDING 상태인 경우 사용자 응답 대기 (무한 루프 방지)
    if review_status == ReviewStatus.PENDING:
        logger.info("Review pending, waiting for user response. Ending graph execution.")
        return "pending"
    
    # 사용자 검토 결과가 있는 경우
    if review_result and hasattr(review_result, 'status'):
        if review_result.status == ReviewStatus.APPROVED:
            logger.info("User approved query, proceeding to execution")
            return "sql_execution"
        elif review_result.status == ReviewStatus.REJECTED:
            logger.info("User rejected query, starting over")
            return "reject"
        elif review_result.status == ReviewStatus.MODIFIED:
            logger.info("User modified query, regenerating SQL")
            return "modify"
    
    # review_status가 있는 경우
    if review_status:
        if review_status == ReviewStatus.APPROVED:
            logger.info("User approved query, proceeding to execution")
            return "sql_execution"
        elif review_status == ReviewStatus.REJECTED:
            logger.info("User rejected query, starting over")
            return "reject"
        elif review_status == ReviewStatus.MODIFIED:
            logger.info("User modified query, regenerating SQL")
            return "modify"
    
    # 기본값: 명확하지 않은 상태에서는 안전을 위해 거부
    logger.warning("No clear review status found, defaulting to reject for safety")
    return "reject"


def _execute_sql_query_simple(state: AgentState) -> AgentState:
    """
    간단한 SQL 실행 (Python 경로용, 데이터 추출 전용)
    
    복잡한 검증 없이 기본 구문/보안 검사만 수행하고 바로 실행합니다.
    Python 코드 실행을 위한 데이터 추출이 목적이므로, 
    validation_check, user_review 등은 생략합니다.
    
    Args:
        state: Current pipeline state
        
    Returns:
        Updated state with query results
    """
    logger.info("Executing SQL query (simple mode for Python path)")
    
    start_time = time.time()
    state["current_node"] = "simple_sql_execution"
    state["execution_status"] = ExecutionStatus.IN_PROGRESS.value
    
    try:
        sql_query = state.get("sql_query")
        if not sql_query:
            raise ValueError("No SQL query to execute")
        
        # 간단한 구문 검사만 (복잡한 검증 생략)
        # SQL 주입 방지를 위한 기본적인 검사만 수행
        dangerous_keywords = ["DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE", "GRANT", "REVOKE"]
        sql_upper = sql_query.upper()
        
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                raise ValueError(f"Dangerous SQL keyword detected: {keyword}")
        
        # SQL 파라미터 가져오기 (SQL Injection 방지)
        sql_params = state.get("sql_params")
        
        # Execute the query with parameters (if available)
        result = execute_query(sql_query, params=sql_params, readonly=True)
        execution_time = time.time() - start_time
        
        # Handle different return types
        if isinstance(result, int):
            query_result: List[Dict[str, Any]] = []
            logger.info(f"Query executed successfully (affected rows: {result})")
        else:
            query_result = result if isinstance(result, list) else []
            if result is None:
                logger.warning("SQL query returned None, treating as empty result")
        
        # Update state
        state["query_result"] = query_result
        state["execution_time"] = execution_time
        state["success"] = True
        state["execution_status"] = ExecutionStatus.COMPLETED.value
        
        logger.info(f"Simple SQL execution completed in {execution_time:.2f}s, returned {len(query_result)} rows")
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Simple SQL execution failed: {str(e)}"
        
        logger.error(error_msg)
        
        # Update state with error
        state["error_message"] = error_msg
        state["success"] = False
        state["execution_status"] = ExecutionStatus.FAILED.value
    
    return state


def _execute_sql_query(state: AgentState) -> AgentState:
    """
    Execute the validated SQL query.
    
    Phase 2: Python 경로의 data_gathering_sql도 처리합니다.
    
    Args:
        state: Current pipeline state
        
    Returns:
        Updated state with query results
    """
    from agentic_flow.state import QueryIntent
    
    intent = state.get("intent")
    
    # Python 경로: data_gathering_sql 사용
    # 공통 유틸리티 함수 사용
    if is_intent_equal(intent, QueryIntent.COMPLEX_ANALYSIS):
        data_gathering_sql = state.get("data_gathering_sql")
        if data_gathering_sql:
            logger.info(f"PYTHON_PATH: Executing data_gathering_sql for Python code")
            state["sql_query"] = data_gathering_sql  # sql_execution이 사용하도록 설정
        else:
            logger.warning("PYTHON_PATH: data_gathering_sql not found, using sql_query")
    
    logger.info("Executing SQL query")
    
    start_time = time.time()
    state["current_node"] = "sql_execution"
    state["execution_status"] = ExecutionStatus.IN_PROGRESS.value
    
    try:
        # 일반 대화인 경우 SQL 실행 건너뛰기
        if state.get("skip_sql_generation", False):
            logger.info("Skipping SQL execution for conversational query")
            state["query_result"] = []
            state["success"] = True
            state["execution_time"] = 0.0
            state["execution_status"] = ExecutionStatus.COMPLETED.value
            return state
        
        sql_query = state.get("validated_sql") or state.get("sql_query")
        if not sql_query:
            raise ValueError("No SQL query to execute")
        
        # SQL 파라미터 가져오기 (SQL Injection 방지)
        sql_params = state.get("sql_params")
        
        # Execute the query with parameters (if available)
        result = execute_query(sql_query, params=sql_params, readonly=True)
        execution_time = time.time() - start_time
        
        # Handle different return types: int (affected rows) or List[Dict[str, Any]] (query results)
        if isinstance(result, int):
            # For non-SELECT queries, convert to empty list
            query_result: List[Dict[str, Any]] = []
            logger.info(f"Query executed successfully (affected rows: {result})")
        else:
            # For SELECT queries, ensure result is a list
            query_result = result if isinstance(result, list) else []
            if result is None:
                logger.warning("SQL query returned None, treating as empty result")
        
        # Update state
        state["query_result"] = query_result
        state["execution_time"] = execution_time
        state["success"] = True
        state["execution_status"] = ExecutionStatus.COMPLETED.value
        
        # Record node execution result
        node_result = NodeExecutionResult(
            node_type=NodeType.SQL_EXECUTION,
            success=True,
            execution_time=execution_time,
            confidence=1.0,
            metadata={"rows_returned": len(query_result)}
        )
        state["node_results"].append(node_result)
        
        logger.info(f"Query executed successfully in {execution_time:.2f}s, returned {len(query_result)} rows")
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"SQL execution failed: {str(e)}"
        
        logger.error(error_msg)
        
        # Update state with error
        state["error_message"] = error_msg
        state["success"] = False
        state["execution_status"] = ExecutionStatus.FAILED.value
        
        # Record node execution result
        node_result = NodeExecutionResult(
            node_type=NodeType.SQL_EXECUTION,
            success=False,
            execution_time=execution_time,
            confidence=0.0,
            error_message=error_msg
        )
        state["node_results"].append(node_result)
    
    return state


def initialize_state(
    user_query: str,
    user_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    session_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> AgentState:
    """
    Initialize the pipeline state.
    
    Args:
        user_query: User's natural language query
        user_id: User identifier
        channel_id: Channel identifier
        session_id: Session identifier
        context: Additional context
        max_retries: Maximum number of retries
        conversation_history: Previous conversation history for context awareness
        
    Returns:
        Initialized AgentState
    """
    return AgentState(
        # Input
        user_query=user_query,
        user_id=user_id,
        channel_id=channel_id,
        session_id=session_id,
        context=context or {},
        
        # Processing stages
        normalized_query=None,
        intent=None,
        llm_intent_result=None,
        entities=[],
        agent_schema_mapping=None,
        sql_query=None,
        validated_sql=None,
        query_result=[],
        data_summary=None,
        
        # Conversation handling
        skip_sql_generation=False,
        conversation_response=None,
        needs_clarification=None,
        
        # Fanding templates
        fanding_template=None,
        
        # Validation and error handling
        validation_result=None,
        processing_decision=None,
        is_valid=True,
        error_message=None,
        retry_count=0,
        max_retries=max_retries,
        
        # Execution tracking
        current_node=None,
        execution_status=ExecutionStatus.PENDING.value,
        node_results=[],
        
        # Metadata
        processing_time=0.0,
        execution_time=0.0,
        confidence_scores={},
        debug_info={},
        
        # RAG schema retrieval
        rag_schema_chunks=None,
        rag_schema_context=None,
        
        # Conversation history
        conversation_history=conversation_history or [],
        
        # Additional fields for review
        review_status=None,
        review_result=None,
        
        # Output
        final_sql=None,
        explanation=None,
        success=False
    )
