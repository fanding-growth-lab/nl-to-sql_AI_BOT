"""
Simplified LangGraph Pipeline Integration

This module integrates all components into a complete LangGraph pipeline with simplified schema loading.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from functools import lru_cache

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import GraphState, create_initial_state, should_retry, is_pipeline_complete
from .nodes import NLProcessor, SchemaMapper
from .sql_generator import SQLGenerator, SQLValidator
from .monitoring import PipelineMonitor, get_metrics_summary, get_health_status
from core.config import get_settings
from core.db import get_db_session
from core.logging import get_logger

logger = get_logger(__name__)


class NLToSQLPipeline:
    """Main pipeline class for natural language to SQL conversion."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = get_logger(self.__class__.__name__)
        self.graph = self._build_pipeline()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        settings = get_settings()
        
        # Get database schema
        db_schema = self._load_database_schema()
        
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
            "max_history": settings.pipeline.max_history
        }
    
    def _load_database_schema(self) -> Dict[str, Any]:
        """Load database schema from the connected database."""
        try:
            # 실제 DB 스키마 (주요 테이블만)
            schema = {
                "t_member": {
                    "comment": "회원 정보 테이블",
                    "columns": {
                        "no": {"type": "int", "nullable": False, "default": None, "comment": "회원 번호"},
                        "email": {"type": "varchar", "nullable": True, "default": None, "comment": "이메일"},
                        "nickname": {"type": "varchar", "nullable": False, "default": None, "comment": "닉네임"},
                        "status": {"type": "char", "nullable": False, "default": None, "comment": "상태 (A=활성)"},
                        "is_admin": {"type": "char", "nullable": False, "default": "F", "comment": "관리자 여부"}
                    }
                },
                "t_member_info": {
                    "comment": "회원 상세 정보 테이블",
                    "columns": {
                        "member_no": {"type": "int", "nullable": False, "default": None, "comment": "회원 번호"},
                        "login_datetime": {"type": "datetime", "nullable": False, "default": None, "comment": "마지막 로그인"},
                        "ins_datetime": {"type": "datetime", "nullable": False, "default": None, "comment": "가입일"}
                    }
                },
                "t_creator": {
                    "comment": "크리에이터 정보 테이블",
                    "columns": {
                        "no": {"type": "int", "nullable": False, "default": None, "comment": "크리에이터 번호"},
                        "nickname": {"type": "varchar", "nullable": False, "default": None, "comment": "닉네임"},
                        "name": {"type": "varchar", "nullable": True, "default": None, "comment": "이름"}
                    }
                },
                "t_funding": {
                    "comment": "펀딩 정보 테이블",
                    "columns": {
                        "no": {"type": "int", "nullable": False, "default": None, "comment": "펀딩 번호"},
                        "creator_no": {"type": "int", "nullable": False, "default": None, "comment": "크리에이터 번호"},
                        "title": {"type": "varchar", "nullable": True, "default": None, "comment": "펀딩 제목"},
                        "status": {"type": "varchar", "nullable": True, "default": None, "comment": "펀딩 상태"},
                        "start_datetime": {"type": "datetime", "nullable": True, "default": None, "comment": "시작일"},
                        "end_datetime": {"type": "datetime", "nullable": True, "default": None, "comment": "종료일"}
                    }
                },
                "t_follow": {
                    "comment": "팔로우 관계 테이블",
                    "columns": {
                        "creator_no": {"type": "int", "nullable": False, "default": None, "comment": "크리에이터 번호"},
                        "member_no": {"type": "int", "nullable": False, "default": None, "comment": "회원 번호"},
                        "type": {"type": "varchar", "nullable": True, "default": None, "comment": "타입"},
                        "ins_datetime": {"type": "datetime", "nullable": False, "default": None, "comment": "팔로우 일시"}
                    }
                }
            }
            
            logger.info(f"Loaded schema for {len(schema)} tables")
            return schema
                
        except Exception as e:
            logger.error(f"Error loading database schema: {str(e)}")
            return {}
    
    def _build_pipeline(self) -> StateGraph:
        """Build the LangGraph pipeline."""
        # Initialize nodes
        nl_processor = NLProcessor(self.config)
        schema_mapper = SchemaMapper(self.config)
        sql_generator = SQLGenerator(self.config)
        sql_validator = SQLValidator(self.config)
        
        # Create state graph (최신 LangGraph 문법)
        graph = StateGraph(GraphState)
        
        # Add nodes
        graph.add_node("process_query", nl_processor.process)
        graph.add_node("map_schema", schema_mapper.process)
        graph.add_node("generate_sql", sql_generator.process)
        graph.add_node("validate_sql", sql_validator.process)
        
        # Add edges (최신 문법: START 상수 사용)
        graph.add_edge(START, "process_query")
        graph.add_edge("process_query", "map_schema")
        graph.add_edge("map_schema", "generate_sql")
        graph.add_edge("generate_sql", "validate_sql")
        
        # Add conditional edges for validation
        graph.add_conditional_edges(
            "validate_sql",
            self._should_retry,
            {
                "retry": "generate_sql",
                "end": END
            }
        )
        
        # Compile the graph with checkpointer (최신 LangGraph 문법)
        checkpointer = MemorySaver()
        return graph.compile(checkpointer=checkpointer)
    
    def _should_retry(self, state: GraphState) -> str:
        """Determine if the pipeline should retry or end."""
        if state.get("is_valid", False):
            return "end"
        elif should_retry(state):
            return "retry"
        else:
            return "end"
    
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
                
                # Create config for checkpointer (최신 LangGraph 문법)
                run_config = {
                    "configurable": {
                        "thread_id": f"{user_id or 'default'}_{channel_id or 'default'}"
                    }
                }
                
                # Run the pipeline with component monitoring
                monitor.start_component("nl_processing")
                result = self.graph.invoke(initial_state, config=run_config)
                monitor.end_component("nl_processing")
                
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
        """Log pipeline results."""
        if result.get("success", False):
            self.logger.info(
                f"Pipeline completed successfully",
                user_id=result.get("user_id"),
                channel_id=result.get("channel_id"),
                processing_time=result.get("processing_time", 0),
                confidence=result.get("confidence_scores", {}).get("sql_generation", 0)
            )
        else:
            self.logger.warning(
                f"Pipeline failed",
                user_id=result.get("user_id"),
                channel_id=result.get("channel_id"),
                error=result.get("error_message"),
                retry_count=result.get("retry_count", 0)
            )
    
    def _format_result(self, result: GraphState) -> Dict[str, Any]:
        """Format the result for API response."""
        return {
            "success": result.get("success", False),
            "sql": result.get("final_sql"),
            "explanation": result.get("explanation"),
            "confidence": result.get("confidence_scores", {}).get("sql_generation", 0),
            "complexity": result.get("sql_result").complexity.value if result.get("sql_result") else "UNKNOWN",
            "processing_time": result.get("processing_time", 0),
            "error": result.get("error_message"),
            "debug_info": result.get("debug_info") if self.config.get("enable_debug", False) else None
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
            "name": "NLToSQLPipeline",
            "pipeline_type": "NLToSQLPipeline",
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
            "components": ["nl_processor", "schema_mapper", "sql_generator", "sql_validator", "data_summarizer"],
            "performance": get_metrics_summary(),
            "health": get_health_status()
        }


@lru_cache(maxsize=1)
def build_nl_to_sql_pipeline(config: Optional[Dict[str, Any]] = None) -> NLToSQLPipeline:
    """Build and cache the NL-to-SQL pipeline."""
    return NLToSQLPipeline(config)


def test_pipeline():
    """Test the pipeline with sample queries."""
    pipeline = build_nl_to_sql_pipeline()
    
    test_queries = [
        "모든 사용자의 이메일 주소를 보여줘",
        "지난 달에 가입한 사용자 수를 세어줘",
        "사용자 이름과 이메일을 함께 보여줘"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: {query}")
        result = pipeline.process(query)
        print(f"Result: {result}")


if __name__ == "__main__":
    test_pipeline()