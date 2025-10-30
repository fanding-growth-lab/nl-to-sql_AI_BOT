"""
Graph State Management for NL-to-SQL Pipeline

This module defines the state structure used throughout the LangGraph pipeline.
"""

from typing import Dict, List, Optional, Any, TypedDict
from dataclasses import dataclass
from enum import Enum


class QueryIntent(Enum):
    """Query intent classification."""
    # 데이터 조회 관련
    DATA_QUERY = "DATA_QUERY"       # 데이터 조회 의도
    SELECT = "SELECT"
    COUNT = "COUNT" 
    AGGREGATE = "AGGREGATE"
    FILTER = "FILTER"
    JOIN = "JOIN"
    
    # 일반 대화 관련
    GREETING = "GREETING"           # 인사말
    GENERAL_CHAT = "GENERAL_CHAT"   # 일반 대화
    HELP_REQUEST = "HELP_REQUEST"   # 도움말 요청
    
    # 알 수 없음
    UNKNOWN = "UNKNOWN"


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"


@dataclass
class Entity:
    """Extracted entity from natural language query."""
    name: str
    type: str  # table, column, value, condition
    confidence: float
    context: Optional[str] = None


@dataclass
class SchemaMapping:
    """Database schema mapping result."""
    relevant_tables: List[str]
    relevant_columns: List[str]
    relationships: List[Dict[str, str]]
    confidence: float


@dataclass
class SQLResult:
    """Generated SQL query result."""
    sql_query: str
    confidence: float
    complexity: QueryComplexity
    estimated_rows: Optional[int] = None
    execution_plan: Optional[str] = None


class GraphState(TypedDict):
    """Main state structure for the LangGraph pipeline."""


    # Input
    user_query: str
    user_id: Optional[str]
    channel_id: Optional[str]
    context: Optional[Dict[str, Any]]
    
    # Processing stages
    normalized_query: Optional[str]
    intent: Optional[QueryIntent]
    llm_intent_result: Optional[Dict[str, Any]]  # LLM intent classification result
    entities: List[Entity]
    schema_mapping: Optional[SchemaMapping]
    sql_result: Optional[SQLResult]
    
    # SQL Generation
    sql_query: Optional[str]
    validated_sql: Optional[str]
    sql_corrected: Optional[str]
    sql_validation: Optional[Dict[str, Any]]
    rag_mapping_result: Optional[Any]
    dynamic_pattern: Optional[Any]
    fanding_template: Optional[Any]
    agent_schema_mapping: Optional[Dict[str, Any]]
    
    # Query execution results
    query_result: List[Dict[str, Any]]
    
    # Conversation handling
    skip_sql_generation: Optional[bool]
    conversation_response: Optional[str]
    conversation_text: Optional[str]
    needs_clarification: Optional[bool]  # 사용자 재입력이 필요한지 표시
    clarification_question: Optional[str]
    slots: Optional[Dict[str, Any]]
    
    # Validation and error handling
    is_valid: bool
    validation_result: Optional[Dict[str, Any]]
    processing_decision: Optional[Dict[str, Any]]
    sql_validation_failed: Optional[bool]
    error_message: Optional[str]
    retry_count: int
    max_retries: int
    
    # Metadata
    processing_time: float
    confidence_scores: Dict[str, float]
    debug_info: Dict[str, Any]
    sql_generation_metadata: Optional[Dict[str, Any]]
    
    # Output
    final_sql: Optional[str]
    data_summary: Optional[str]
    insight_report: Optional[Dict[str, Any]]
    business_insights: Optional[Dict[str, Any]]
    result_statistics: Optional[Dict[str, Any]]
    explanation: Optional[str]
    success: bool


class PipelineConfig:
    """Configuration for the NL-to-SQL pipeline."""
    
    def __init__(
        self,
        llm_config: Dict[str, Any],
        db_schema: Dict[str, Any],
        max_retries: int = 3,
        confidence_threshold: float = 0.7,
        enable_debug: bool = False
    ):
        self.llm_config = llm_config
        self.db_schema = db_schema
        self.max_retries = max_retries
        self.confidence_threshold = confidence_threshold
        self.enable_debug = enable_debug


def create_initial_state(
    user_query: str,
    user_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    max_retries: int = 3
) -> GraphState:
    """Create initial state for the pipeline."""
    return {
        "user_query": user_query,
        "user_id": user_id,
        "channel_id": channel_id,
        "context": context or {},
        "normalized_query": None,
        "intent": None,
        "llm_intent_result": None,
        "entities": [],
        "schema_mapping": None,
        "sql_result": None,
        "sql_query": None,
        "validated_sql": None,
        "sql_corrected": None,
        "sql_validation": None,
        "rag_mapping_result": None,
        "dynamic_pattern": None,
        "fanding_template": None,
        "agent_schema_mapping": None,
        "query_result": [],
        "skip_sql_generation": None,
        "conversation_response": None,
        "conversation_text": None,
        "needs_clarification": None,
        "clarification_question": None,
        "slots": None,
        "is_valid": False,
        "validation_result": None,
        "processing_decision": None,
        "sql_validation_failed": None,
        "error_message": None,
        "retry_count": 0,
        "max_retries": max_retries,
        "processing_time": 0.0,
        "confidence_scores": {},
        "debug_info": {},
        "sql_generation_metadata": None,
        "final_sql": None,
        "data_summary": None,
        "insight_report": None,
        "business_insights": None,
        "result_statistics": None,
        "explanation": None,
        "success": False
    }


def update_state_confidence(state: GraphState, component: str, confidence: float) -> GraphState:
    """Update confidence score for a specific component."""
    state["confidence_scores"][component] = confidence
    return state


def add_debug_info(state: GraphState, key: str, value: Any) -> GraphState:
    """Add debug information to the state."""
    if state.get("debug_info") is None:
        state["debug_info"] = {}
    state["debug_info"][key] = value
    return state


def should_retry(state: GraphState) -> bool:
    """Determine if the pipeline should retry processing."""
    return (
        not state.get("is_valid", False) and 
        state.get("retry_count", 0) < state.get("max_retries", 3)
    )


def is_pipeline_complete(state: GraphState) -> bool:
    """Check if the pipeline processing is complete."""
    return (
        state.get("is_valid", False) or 
        state.get("retry_count", 0) >= state.get("max_retries", 3)
    )


def set_sql_result(state: GraphState, sql_query: str, confidence: float = 1.0) -> GraphState:
    """Set SQL query result in state."""
    state["sql_query"] = sql_query
    state["confidence_scores"]["sql_generation"] = confidence
    return state


def set_rag_mapping_result(state: GraphState, rag_result: Any) -> GraphState:
    """Set RAG mapping result in state."""
    state["rag_mapping_result"] = rag_result
    state["sql_query"] = rag_result.sql_template if hasattr(rag_result, 'sql_template') else None
    return state


def set_dynamic_pattern(state: GraphState, pattern: Any) -> GraphState:
    """Set dynamic pattern result in state."""
    state["dynamic_pattern"] = pattern
    state["sql_query"] = pattern.sql_template if hasattr(pattern, 'sql_template') else None
    return state


def set_fanding_template(state: GraphState, template: Any) -> GraphState:
    """Set Fanding template result in state."""
    state["fanding_template"] = template
    state["sql_query"] = template.sql_template if hasattr(template, 'sql_template') else None
    return state


def set_conversation_response(state: GraphState, response: str, skip_sql: bool = True) -> GraphState:
    """Set conversation response and skip SQL generation."""
    state["conversation_response"] = response
    state["skip_sql_generation"] = skip_sql
    return state


def clear_sql_generation(state: GraphState) -> GraphState:
    """Clear all SQL generation related fields."""
    state["sql_query"] = None
    state["rag_mapping_result"] = None
    state["dynamic_pattern"] = None
    state["fanding_template"] = None
    return state


def get_effective_sql(state: GraphState) -> Optional[str]:
    """Get the effective SQL query from various sources."""
    if state.get("sql_query"):
        return state["sql_query"]
    if state.get("final_sql"):
        return state["final_sql"]
    return None


def is_sql_generation_skipped(state: GraphState) -> bool:
    """Check if SQL generation should be skipped."""
    return state.get("skip_sql_generation", False) or state.get("conversation_response") is not None


# === New State Helper Functions for Consistency ===

def set_skip_sql(state: GraphState, skip: bool = True) -> None:
    """Set skip_sql_generation flag."""
    state["skip_sql_generation"] = skip


def set_intent(state: GraphState, intent: QueryIntent) -> None:
    """Set query intent."""
    state["intent"] = intent


def set_entities(state: GraphState, entities: List[Entity]) -> None:
    """Set extracted entities."""
    state["entities"] = entities


def set_normalized_query(state: GraphState, query: str) -> None:
    """Set normalized query."""
    state["normalized_query"] = query


def set_confidence_score(state: GraphState, component: str, score: float) -> None:
    """Set confidence score for a specific component."""
    if "confidence_scores" not in state:
        state["confidence_scores"] = {}
    state["confidence_scores"][component] = score


def set_error_message(state: GraphState, message: str) -> None:
    """Set error message."""
    state["error_message"] = message


def set_success(state: GraphState, success: bool = True) -> None:
    """Set success flag."""
    state["success"] = success


def set_sql_query(state: GraphState, sql: Optional[str]) -> None:
    """Set SQL query."""
    state["sql_query"] = sql


def set_validated_sql(state: GraphState, sql: Optional[str]) -> None:
    """Set validated SQL query."""
    state["validated_sql"] = sql


def set_query_result(state: GraphState, result: List[Dict[str, Any]]) -> None:
    """Set query execution result."""
    state["query_result"] = result


def set_data_summary(state: GraphState, summary: str) -> None:
    """Set data summary."""
    state["data_summary"] = summary


def set_schema_mapping(state: GraphState, mapping: SchemaMapping) -> None:
    """Set schema mapping result."""
    state["schema_mapping"] = mapping


def set_validation_result(state: GraphState, result: Dict[str, Any]) -> None:
    """Set validation result."""
    state["validation_result"] = result


def set_is_valid(state: GraphState, is_valid: bool) -> None:
    """Set validation flag."""
    state["is_valid"] = is_valid


def set_sql_validation_failed(state: GraphState, failed: bool) -> None:
    """Set SQL validation failed flag."""
    state["sql_validation_failed"] = failed


def set_conversation_text(state: GraphState, text: str) -> None:
    """Set conversation text."""
    state["conversation_text"] = text


def set_clarification_question(state: GraphState, question: str) -> None:
    """Set clarification question."""
    state["clarification_question"] = question


def set_slots(state: GraphState, slots: Dict[str, Any]) -> None:
    """Set extracted slots."""
    state["slots"] = slots


def set_llm_intent_result(state: GraphState, result: Optional[Dict[str, Any]]) -> None:
    """Set LLM intent classification result."""
    state["llm_intent_result"] = result


def set_processing_decision(state: GraphState, decision: Optional[Dict[str, Any]]) -> None:
    """Set processing decision."""
    state["processing_decision"] = decision