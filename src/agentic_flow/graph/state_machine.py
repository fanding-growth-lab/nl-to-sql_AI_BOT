"""
LangGraph State Machine Core Implementation

This module implements the core state machine logic for the NL-to-SQL pipeline,
including state definitions, node connections, and conditional routing.
"""

import time
import logging
from typing import Dict, List, Optional, Any, TypedDict, Union
from dataclasses import dataclass
from enum import Enum

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ..nodes import (
    NLProcessor, SchemaMapper, SQLGenerationNode, 
    SQLValidationNode, DataSummarizationNode
)
from ..llm_intent_classifier import LLMIntentClassifier
from ..dynamic_sql_generator import DynamicSQLGenerator
from ..validation_node import ValidationNode
from ..user_review_node import UserReviewNode, ReviewStatus
from ..monitoring import PipelineMonitor
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
    SCHEMA_MAPPING = "schema_mapping"
    DYNAMIC_SQL_GENERATION = "dynamic_sql_generation"
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
    entities: List[Dict[str, Any]]
    agent_schema_mapping: Optional[Dict[str, Any]]
    sql_query: Optional[str]
    validated_sql: Optional[str]
    query_result: List[Dict[str, Any]]
    data_summary: Optional[str]
    
    # Conversation handling
    skip_sql_generation: Optional[bool]
    conversation_response: Optional[str]
    
    # Fanding templates
    fanding_template: Optional[Any]
    
    # Validation and error handling
    validation_result: Optional[Dict[str, Any]]
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
    confidence_scores: Dict[str, float]
    debug_info: Dict[str, Any]
    
    # Output
    final_sql: Optional[str]
    explanation: Optional[str]
    success: bool


def create_agent_graph(
    db_schema: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> StateGraph:
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
    if config is None:
        config = {
            "llm_config": {
                "model": settings.llm.model,
                "api_key": settings.llm.api_key,
                "temperature": settings.llm.temperature,
                "max_tokens": settings.llm.max_tokens
            },
            "llm": {
                "model": settings.llm.model,
                "api_key": settings.llm.api_key,
                "temperature": settings.llm.temperature,
                "max_tokens": settings.llm.max_tokens
            },
            "db_schema": db_schema or _get_default_schema(),
            "max_retries": settings.pipeline.max_retries,
            "confidence_threshold": settings.pipeline.confidence_threshold,
            "enable_debug": settings.pipeline.enable_debug,
            "enable_monitoring": settings.pipeline.enable_monitoring
        }
    
    # Create the state graph
    graph = StateGraph(AgentState)
    
    # Initialize nodes
    nodes = _initialize_nodes(config)
    
    # Add nodes to the graph (wrap node objects in lambda functions)
    graph.add_node("llm_intent_classification", lambda state: nodes["llm_intent_classifier"].process(state))
    graph.add_node("nlp_processing", lambda state: nodes["nlp_processor"].process(state))
    graph.add_node("schema_mapping", lambda state: nodes["schema_mapper"].process(state))
    graph.add_node("dynamic_sql_generation", lambda state: nodes["dynamic_sql_generator"].process(state))
    graph.add_node("sql_generation", lambda state: nodes["sql_generator"].process(state))
    graph.add_node("sql_validation", lambda state: nodes["sql_validator"].process(state))
    graph.add_node("validation_check", lambda state: nodes["validation_node"].process(state))
    graph.add_node("user_review", lambda state: nodes["user_review_node"].process(state))
    graph.add_node("sql_execution", _execute_sql_query)
    graph.add_node("data_summarization", lambda state: nodes["data_summarizer"].process(state))
    
    # Define the flow
    graph.add_edge(START, "llm_intent_classification")
    graph.add_edge("llm_intent_classification", "nlp_processing")
    graph.add_edge("nlp_processing", "schema_mapping")
    graph.add_edge("schema_mapping", "dynamic_sql_generation")
    graph.add_edge("dynamic_sql_generation", "sql_generation")
    graph.add_edge("sql_generation", "sql_validation")
    
    # Conditional edge for SQL validation
    graph.add_conditional_edges(
        "sql_validation",
        route_after_validation,
        {
            "retry": "sql_generation",
            "validate": "validation_check",
            "sql_execution": "sql_execution"
        }
    )
    
    # Note: validation_check is only reached via conditional edge
    
    # Conditional edge for validation check
    graph.add_conditional_edges(
        "validation_check",
        route_after_validation_check,
        {
            "sql_execution": "sql_execution",
            "user_review": "user_review",
            "reject": "nlp_processing"  # Start over with better context
        }
    )
    
    # Conditional edge for user review
    graph.add_conditional_edges(
        "user_review",
        route_after_user_review,
        {
            "sql_execution": "sql_execution",
            "reject": "nlp_processing",
            "modify": "sql_generation"
        }
    )
    
    graph.add_edge("sql_execution", "data_summarization")
    graph.add_edge("data_summarization", END)
    
    # Compile the graph with memory
    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)
    
    logger.info("LangGraph state machine created successfully")
    return compiled_graph


def _initialize_nodes(config: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize all pipeline nodes."""
    logger.info("Initializing pipeline nodes")
    logger.info(f"Config passed to _initialize_nodes: {config}")
    
    nodes = {
        "llm_intent_classifier": LLMIntentClassifier(config),
        "nlp_processor": NLProcessor(config),
        "schema_mapper": SchemaMapper(config),
        "dynamic_sql_generator": DynamicSQLGenerator(config),
        "sql_generator": SQLGenerationNode(config),
        "sql_validator": SQLValidationNode(config),
        "validation_node": ValidationNode(),
        "user_review_node": UserReviewNode(),
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


def route_after_validation(state: AgentState) -> str:
    """
    Route after SQL validation based on validation result and retry count.
    
    Args:
        state: Current pipeline state
        
    Returns:
        Next node name ("retry" or "execute")
    """
    validation_result = state.get("validation_result", {})
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    
    # Check if this is a conversation response (greeting, help, etc.)
    if state.get("conversation_response", False):
        logger.info("Conversation response detected, skipping validation and proceeding to execution")
        return "sql_execution"
    
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
        logger.warning("Max retries exceeded, proceeding with invalid SQL to avoid infinite loop")
        # 재시도 카운트 초기화
        state["retry_count"] = 0
        return "validate"


def route_after_validation_check(state: AgentState) -> str:
    """
    검증 체크 후 라우팅 결정
    
    Args:
        state: Current pipeline state
        
    Returns:
        str: 다음 노드 이름
    """
    validation_result = state.get("validation_result")
    processing_decision = state.get("processing_decision", {})
    
    if not validation_result:
        logger.warning("No validation result found, defaulting to user review")
        return "user_review"
    
    # 자동 승인 조건 확인
    if processing_decision.get("auto_approve", False):
        logger.info("Auto-approving query based on high confidence")
        return "sql_execution"
    
    # 검증 실패 시 거부 - 무한 루프 방지를 위해 sql_execution으로 강제 진행
    if validation_result.status.value == "rejected":
        logger.warning("Query rejected due to validation issues, but proceeding to execution to avoid infinite loop")
        return "sql_execution"
    
    # 사용자 검토 필요 조건 확인
    if processing_decision.get("needs_user_review", False):
        logger.info("Query requires user review")
        return "user_review"
    
    # 높은 신뢰도면 자동 승인
    if validation_result.confidence >= 0.85:
        logger.info(f"High confidence ({validation_result.confidence:.2f}), auto-approving")
        return "sql_execution"
    
    # 기본적으로 사용자 검토로 이동
    logger.info("Defaulting to user review")
    return "user_review"


def route_after_user_review(state: AgentState) -> str:
    """
    사용자 검토 후 라우팅 결정
    
    Args:
        state: Current pipeline state
        
    Returns:
        str: 다음 노드 이름
    """
    review_status = state.get("review_status")
    
    if not review_status:
        logger.warning("No review status found, defaulting to reject")
        return "reject"
    
    if review_status == ReviewStatus.APPROVED:
        logger.info("User approved query, proceeding to execution")
        return "sql_execution"
    elif review_status == ReviewStatus.REJECTED:
        logger.info("User rejected query, starting over")
        return "reject"
    elif review_status == ReviewStatus.MODIFIED:
        logger.info("User modified query, regenerating SQL")
        return "modify"
    else:
        logger.warning(f"Unknown review status: {review_status}, defaulting to reject")
        return "reject"


def should_retry_generation(state: AgentState) -> bool:
    """
    Determine if SQL generation should be retried.
    
    Args:
        state: Current pipeline state
        
    Returns:
        True if should retry, False otherwise
    """
    validation_result = state.get("validation_result", {})
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    
    return (
        not validation_result.get("is_valid", False) and 
        retry_count < max_retries
    )


def _execute_sql_query(state: AgentState) -> AgentState:
    """
    Execute the validated SQL query.
    
    Args:
        state: Current pipeline state
        
    Returns:
        Updated state with query results
    """
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
        
        # Execute the query
        result = execute_query(sql_query, readonly=True)
        execution_time = time.time() - start_time
        
        # NoneType 에러 방지: result가 None인 경우 빈 리스트로 처리
        if result is None:
            result = []
            logger.warning("SQL query returned None, treating as empty result")
        
        # Update state
        state["query_result"] = result
        state["execution_time"] = execution_time
        state["success"] = True
        state["execution_status"] = ExecutionStatus.COMPLETED.value
        
        # Record node execution result
        node_result = NodeExecutionResult(
            node_type=NodeType.SQL_EXECUTION,
            success=True,
            execution_time=execution_time,
            confidence=1.0,
            metadata={"rows_returned": len(result)}
        )
        state["node_results"].append(node_result)
        
        logger.info(f"Query executed successfully in {execution_time:.2f}s, returned {len(result)} rows")
        
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


def handle_errors(state: AgentState) -> AgentState:
    """
    Handle errors in the pipeline state.
    
    Args:
        state: Current pipeline state
        
    Returns:
        Updated state with error handling applied
    """
    error_message = state.get("error_message")
    if error_message:
        logger.error(f"Pipeline error: {error_message}")
        
        # Set final state
        state["success"] = False
        state["execution_status"] = ExecutionStatus.FAILED.value
        state["final_sql"] = state.get("sql_query")
        state["explanation"] = f"처리 중 오류가 발생했습니다: {error_message}"
    
    return state


def initialize_state(
    user_query: str,
    user_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    session_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    max_retries: int = 3
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
        entities=[],
        agent_schema_mapping=None,
        sql_query=None,
        validated_sql=None,
        query_result=[],
        data_summary=None,
        
        # Conversation handling
        skip_sql_generation=False,
        conversation_response=None,
        
        # LLM Intent Classification
        llm_intent_result=None,
        
        # Fanding templates
        fanding_template=None,
        
        # Validation and error handling
        validation_result=None,
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
        confidence_scores={},
        debug_info={},
        
        # Output
        final_sql=None,
        explanation=None,
        success=False
    )
