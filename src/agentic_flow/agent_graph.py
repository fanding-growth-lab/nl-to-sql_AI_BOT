"""
LangGraph Agent Pipeline for NL-to-SQL Processing

This module implements the complete LangGraph pipeline that integrates all agent nodes
for natural language to SQL conversion with validation and summarization.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from functools import lru_cache

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import GraphState, create_initial_state, should_retry, is_pipeline_complete
from .nodes import (
    NLProcessor, SchemaMapper, SQLGenerationNode, 
    SQLValidationNode, DataSummarizationNode
)
from .monitoring import PipelineMonitor, get_metrics_summary, get_health_status
from core.config import get_settings
from core.db import get_db_session, execute_query, get_cached_db_schema
from core.logging import get_logger

logger = get_logger(__name__)


class SQLAgentPipeline:
    """Complete SQL Agent Pipeline using LangGraph"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(self.__class__.__name__)
        self.config = config or self._get_default_config()
        self.graph = self._build_pipeline()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        settings = get_settings()
        
        # Get database schema (캐싱된 스키마 사용)
        db_schema = get_cached_db_schema()
        
        return {
            "llm_config": {
                "model": settings.llm.model,
                "api_key": settings.llm.api_key,
                "temperature": settings.llm.temperature,
                "max_tokens": settings.llm.max_tokens
            },
            "db_schema": db_schema,
            "max_retries": settings.pipeline.max_retries,
            "confidence_threshold": settings.pipeline.confidence_threshold,
            "enable_debug": settings.pipeline.enable_debug or settings.environment == "development",
            "enable_monitoring": settings.pipeline.enable_monitoring,
            "max_history": settings.pipeline.max_history,
            "dangerous_sql_keywords": settings.pipeline.dangerous_sql_keywords
        }
    
    
    
    def _build_pipeline(self) -> StateGraph:
        """Build the LangGraph pipeline."""
        # Create the state graph
        graph = StateGraph(GraphState)
        
        # Initialize nodes
        nl_processor = NLProcessor(self.config) #자연어
        schema_mapper = SchemaMapper(self.config)  #
        sql_generator = SQLGenerationNode(self.config)
        sql_validator = SQLValidationNode(self.config)
        data_summarizer = DataSummarizationNode(self.config)
        
        # Add nodes to the graph
        graph.add_node("nl_processing", nl_processor.process)
        graph.add_node("schema_mapping", schema_mapper.process)
        graph.add_node("sql_generation", sql_generator.process)
        graph.add_node("sql_validation", sql_validator.process)
        graph.add_node("sql_execution", self._execute_sql_query)
        graph.add_node("data_summarization", data_summarizer.process)
        
        # Define the flow
        graph.add_edge(START, "nl_processing")
        graph.add_edge("nl_processing", "schema_mapping")
        graph.add_edge("schema_mapping", "sql_generation")
        graph.add_edge("sql_generation", "sql_validation")
        
        # Conditional edge for SQL validation
        graph.add_conditional_edges(
            "sql_validation",
            self._should_retry_sql_generation,
            {
                "retry": "sql_generation",
                "execute": "sql_execution",
                "end": END  # 대화 응답인 경우 파이프라인 종료
            }
        )
        
        graph.add_edge("sql_execution", "data_summarization")
        graph.add_edge("data_summarization", END)
        
        # Compile the graph with memory
        memory = MemorySaver()
        return graph.compile(checkpointer=memory)
    
    def _should_retry_sql_generation(self, state: GraphState) -> str:
        """Determine if SQL generation should be retried."""
        # 일반 대화인 경우 SQL 생성 건너뛰기
        if state.get("skip_sql_generation", False):
            return "end"  # 대화 응답이 있는 경우 파이프라인 종료
        
        validation_result = state.get("sql_validation", {})
        
        # Check if validation passed
        if validation_result.get("is_valid", False):
            return "execute"
        
        # Check retry count
        retry_count = state.get("retry_count", 0)
        max_retries = self.config.get("max_retries", 3)
        
        if retry_count < max_retries:
            # Increment retry count
            state["retry_count"] = retry_count + 1
            return "retry"
        else:
            # Max retries exceeded, terminate the pipeline
            self.logger.error("Max retries exceeded. SQL generation failed.")
            state["error_message"] = "최대 재시도 횟수를 초과하여 유효한 SQL 쿼리를 생성하지 못했습니다."
            state["success"] = False
            return "end"  # 'execute' 대신 'end'로 변경
    
    def _execute_sql_query(self, state: GraphState) -> GraphState:
        """Execute the SQL query with enhanced error handling."""
        self.logger.info("Executing SQL query")
        
        try:
            # 일반 대화인 경우 SQL 실행 건너뛰기
            if state.get("skip_sql_generation", False):
                self.logger.info("Skipping SQL execution for conversational query")
                state["query_result"] = []
                state["success"] = True
                return state
            
            sql_query = state.get("sql_query")
            if not sql_query:
                state["error_message"] = "No SQL query to execute"
                state["success"] = False
                return state
            
            # SQL 쿼리 사전 검증
            if not self._is_safe_sql(sql_query):
                state["error_message"] = "Unsafe SQL query detected"
                state["success"] = False
                return state
            
            # Execute the query
            start_time = time.time()
            result = execute_query(sql_query, readonly=True)
            execution_time = time.time() - start_time
            
            # NoneType 에러 방지: result가 None인 경우 빈 리스트로 처리
            if result is None:
                result = []
                self.logger.warning("SQL query returned None, treating as empty result")
            
            # Store results
            state["query_result"] = result
            state["execution_time"] = execution_time
            state["success"] = True
            
            self.logger.info(f"Query executed successfully in {execution_time:.2f}s, returned {len(result)} rows")
            
        except Exception as e:
            self.logger.error(f"SQL execution failed: {str(e)}")
            state["error_message"] = f"SQL execution failed: {str(e)}"
            state["success"] = False
        
        return state
    
    def _is_safe_sql(self, sql_query: str) -> bool:
        """Check if SQL query is safe to execute."""
        # 설정에서 키워드 리스트를 가져옴
        dangerous_keywords = self.config.get("dangerous_sql_keywords", [])
        sql_upper = sql_query.upper()
        
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                self.logger.warning(f"Dangerous SQL keyword detected: {keyword}")
                return False
        return True
    
    def process(self, user_query: str, user_id: Optional[str] = None, 
                channel_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a natural language query and return SQL result."""
        
        with PipelineMonitor(user_id, channel_id) as monitor:
            try:
                # Create initial state
                initial_state = create_initial_state(
                    user_query=user_query,
                    user_id=user_id,
                    channel_id=channel_id,
                    context=context,
                    max_retries=self.config.get("max_retries", 3)
                )
                
                # Create config for checkpointer
                run_config = {
                    "configurable": {
                        "thread_id": f"{user_id or 'default'}_{channel_id or 'default'}"
                    }
                }
                
                # 전체 파이프라인 실행 시간 측정
                monitor.start_component("full_pipeline")
                result = self.graph.invoke(initial_state, config=run_config)
                monitor.end_component("full_pipeline")
                
                # Finish monitoring and collect final metrics
                final_metrics = monitor.finish(result)
                
                # Log results
                self._log_results(result)
                
                # Add monitoring info to result if debug is enabled
                formatted_result = self._format_result(result)
                if self.config.get("enable_debug", False) and final_metrics:
                    formatted_result["monitoring"] = {
                        "processing_time": final_metrics.processing_time,
                        "confidence": final_metrics.final_confidence,
                        "success": final_metrics.success,
                        "retry_count": final_metrics.retry_count
                    }
                
                return formatted_result
                
            except Exception as e:
                self.logger.error(f"Pipeline processing failed: {str(e)}")
                # Finish monitoring even on error
                error_state = {
                    "success": False,
                    "error_message": str(e),
                    "retry_count": 0
                }
                monitor.finish(error_state)
                
                return {
                    "success": False,
                    "error": str(e),
                    "sql": None,
                    "explanation": "Pipeline processing failed"
                }
    
    def _log_results(self, result: GraphState):
        """Log pipeline results with structured logging."""
        if result.get("success", False):
            # INFO 레벨: 요약 정보
            self.logger.info(
                "Pipeline completed successfully",
                extra={
                    "user_id": result.get("user_id"),
                    "channel_id": result.get("channel_id"),
                    "sql_summary": result.get("sql_query", "")[:100],
                    "execution_time": result.get("execution_time", 0),
                    "retry_count": result.get("retry_count", 0)
                }
            )
            
            # DEBUG 레벨: 전체 SQL 쿼리
            self.logger.debug(
                "Full SQL query for successful pipeline",
                extra={
                    "full_sql": result.get("sql_query"),
                    "query_result_count": len(result.get("query_result", [])),
                    "confidence_scores": result.get("confidence_scores", {})
                }
            )
        else:
            # ERROR 레벨: 실패 정보
            self.logger.error(
                "Pipeline failed",
                extra={
                    "user_id": result.get("user_id"),
                    "channel_id": result.get("channel_id"),
                    "error_message": result.get("error_message", "Unknown error"),
                    "retry_count": result.get("retry_count", 0),
                    "sql_query": result.get("sql_query", "")[:200]  # 실패한 SQL도 로깅
                }
            )
    
    def _format_result(self, result: GraphState) -> Dict[str, Any]:
        """Format the final result."""
        return {
            "success": result.get("success", False),
            "sql": result.get("sql_query"),
            "explanation": result.get("data_summary", "No summary available"),
            "query_result": result.get("query_result", []),
            "confidence_scores": result.get("confidence_scores", {}),
            "execution_time": result.get("execution_time", 0),
            "retry_count": result.get("retry_count", 0),
            "error": result.get("error_message") if not result.get("success", False) else None
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get pipeline performance metrics summary."""
        return get_metrics_summary()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get pipeline health status."""
        return get_health_status()
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline configuration and status information."""
        return {
            "pipeline_type": "SQLAgentPipeline",
            "config": {
                "max_retries": self.config.get("max_retries", 3),
                "confidence_threshold": self.config.get("confidence_threshold", 0.7),
                "enable_debug": self.config.get("enable_debug", False),
                "schema_tables": len(self.config.get("db_schema", {}))
            },
            "llm_config": {
                "model": self.config.get("llm_config", {}).get("model", "unknown"),
                "temperature": self.config.get("llm_config", {}).get("temperature", 0.0),
                "max_tokens": self.config.get("llm_config", {}).get("max_tokens", 0)
            },
            "performance": get_metrics_summary(),
            "health": get_health_status()
        }


@lru_cache(maxsize=1)
def build_sql_agent_pipeline(config: Optional[Dict[str, Any]] = None) -> SQLAgentPipeline:
    """Build and cache the SQL agent pipeline."""
    return SQLAgentPipeline(config)


def create_sql_agent_graph() -> StateGraph:
    """Create a SQL agent graph for external use."""
    pipeline = SQLAgentPipeline()
    return pipeline.graph
