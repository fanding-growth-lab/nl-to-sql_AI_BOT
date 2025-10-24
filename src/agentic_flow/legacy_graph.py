"""
LangGraph Pipeline Integration

This module integrates all components into a complete LangGraph pipeline.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from functools import lru_cache

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import text

from .state import GraphState, create_initial_state, should_retry, is_pipeline_complete
from .nodes import NLProcessor, SchemaMapper
from .sql_generator import SQLGenerator, SQLValidator
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
            "max_retries": 3,
            "confidence_threshold": 0.7,
            "enable_debug": settings.environment == "development"
        }
    
    def _load_database_schema(self) -> Dict[str, Any]:
        """Load database schema from the connected database."""
        try:
            with get_db_session() as session:
                # Get table information
                tables_query = """
                SELECT 
                    TABLE_NAME,
                    TABLE_COMMENT
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = DATABASE()
                """
                
                tables_result = session.execute(text(tables_query)).fetchall()
                schema = {}
                
                for row in tables_result:
                    table_name = row[0]
                    table_comment = row[1]
                    # Get column information for each table
                    columns_query = """
                    SELECT 
                        COLUMN_NAME,
                        DATA_TYPE,
                        IS_NULLABLE,
                        COLUMN_DEFAULT,
                        COLUMN_COMMENT
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_SCHEMA = DATABASE() 
                    AND TABLE_NAME = %s
                    ORDER BY ORDINAL_POSITION
                    """
                    
                    columns_result = session.execute(text(columns_query), (table_name,)).fetchall()
                    columns = {}
                    
                    for row in columns_result:
                        col_name = row[0]
                        data_type = row[1]
                        is_nullable = row[2]
                        default_val = row[3]
                        comment = row[4]
                        columns[col_name] = {
                            "type": data_type,
                            "nullable": is_nullable == "YES",
                            "default": default_val,
                            "comment": comment
                        }
                    
                    schema[table_name] = {
                        "comment": table_comment,
                        "columns": columns
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
        
        # Create state graph
        graph = StateGraph(GraphState)
        
        # Add nodes
        graph.add_node("process_query", nl_processor.process)
        graph.add_node("map_schema", schema_mapper.process)
        graph.add_node("generate_sql", sql_generator.process)
        graph.add_node("validate_sql", sql_validator.process)
        
        # Add edges (최신 LangGraph 문법: START 상수 사용)
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
        start_time = time.time()
        
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
            
            # Run the pipeline
            result = self.graph.invoke(initial_state, config=run_config)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            
            # Log results
            self._log_results(result)
            
            return self._format_result(result)
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {str(e)}")
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
            "complexity": result.get("sql_result", {}).get("complexity", "UNKNOWN") if result.get("sql_result") else "UNKNOWN",
            "processing_time": result.get("processing_time", 0),
            "error": result.get("error_message"),
            "debug_info": result.get("debug_info") if self.config.get("enable_debug", False) else None
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