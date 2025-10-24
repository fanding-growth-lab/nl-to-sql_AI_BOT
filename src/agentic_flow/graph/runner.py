"""
Graph Runner Implementation for LangGraph State Machine

This module implements the AgentGraphRunner class for executing the LangGraph
state machine with async support, progress monitoring, and error handling.
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from dataclasses import dataclass, asdict
from enum import Enum

# CompiledGraph is not directly importable in newer versions

from .state_machine import AgentState, create_agent_graph, initialize_state
from .context import AgentContextManager, MessageRole
from ..monitoring import PipelineMonitor, MetricsCollector
from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)


class ExecutionMode(Enum):
    """Execution mode enumeration."""
    STANDARD = "standard"  # Alias for SYNC
    SYNC = "sync"
    ASYNC = "async"
    STREAM = "stream"


@dataclass
class ExecutionProgress:
    """Execution progress information."""
    session_id: str
    current_node: str
    progress_percentage: float
    elapsed_time: float
    estimated_remaining: Optional[float] = None
    status: str = "running"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GraphExecutionResult:
    """Graph execution result."""
    session_id: str
    success: bool
    user_query: str
    final_sql: Optional[str]
    query_result: List[Dict[str, Any]]
    data_summary: Optional[str]
    error_message: Optional[str]
    execution_time: float
    confidence_scores: Dict[str, float]
    node_results: List[Dict[str, Any]]
    debug_info: Dict[str, Any]
    conversation_response: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphExecutionResult':
        """Create from dictionary."""
        return cls(**data)


class AgentGraphRunner:
    """
    Agent Graph Runner for executing LangGraph state machine.
    
    This class provides synchronous and asynchronous execution of the NL-to-SQL
    pipeline with progress monitoring, error handling, and session management.
    """
    
    def __init__(
        self,
        db_schema: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        enable_monitoring: bool = True
    ):
        """
        Initialize the graph runner.
        
        Args:
            db_schema: Database schema information
            config: Configuration parameters
            enable_monitoring: Whether to enable performance monitoring
        """
        self.db_schema = db_schema
        self.config = config or {}
        self.enable_monitoring = enable_monitoring
        
        # Initialize components
        self.graph = create_agent_graph(db_schema, config)
        self.context_manager = AgentContextManager(storage_backend="memory")
        
        if enable_monitoring:
            self.monitor = PipelineMonitor()
            self.metrics_collector = MetricsCollector()
        else:
            self.monitor = None
            self.metrics_collector = None
        
        logger.info("AgentGraphRunner initialized successfully")
    
    def process_query(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> GraphExecutionResult:
        """
        Process a natural language query synchronously.
        
        Args:
            user_query: User's natural language query
            session_id: Session identifier
            user_id: User identifier
            channel_id: Channel identifier
            context: Additional context
            max_retries: Maximum number of retries
            
        Returns:
            GraphExecutionResult with the processing outcome
        """
        if session_id is None:
            session_id = self.context_manager.create_session()
        
        logger.info(f"Processing query synchronously for session {session_id}")
        
        # Add user message to conversation history
        self.context_manager.add_user_message(session_id, user_query)
        
        # Initialize state
        initial_state = initialize_state(
            user_query=user_query,
            user_id=user_id,
            channel_id=channel_id,
            session_id=session_id,
            context=context,
            max_retries=max_retries
        )
        
        # Add conversation context to state
        conversation_context = self.context_manager.get_conversation_context(session_id)
        if conversation_context:
            initial_state["context"]["conversation_history"] = conversation_context
        
        # Execute the graph
        start_time = time.time()
        
        try:
            if self.enable_monitoring:
                self.monitor.start_execution(session_id)
            
            # Create run configuration
            run_config = {
                "configurable": {
                    "thread_id": session_id
                }
            }
            
            # Execute the graph
            final_state = self.graph.invoke(initial_state, config=run_config)
            
            execution_time = time.time() - start_time
            
            # Debug logging
            logger.info(f"Graph execution completed. final_state type: {type(final_state)}, value: {final_state}")
            
            # Create result
            result = self._create_execution_result(
                session_id=session_id,
                final_state=final_state,
                execution_time=execution_time,
                user_query=user_query
            )
            
            # Add assistant message to conversation history
            summary = result.data_summary or "쿼리가 처리되었습니다."
            self.context_manager.add_assistant_message(
                session_id,
                summary,
                {"sql_query": result.final_sql, "success": result.success}
            )
            
            # Update session state
            self.context_manager.update_session_state(session_id, final_state)
            
            if self.enable_monitoring:
                self.monitor.end_execution(session_id, success=result.success)
            
            logger.info(f"Query processed successfully in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Query processing failed: {str(e)}"
            
            logger.error(error_msg)
            
            if self.enable_monitoring:
                self.monitor.end_execution(session_id, success=False)
            
            # Create error result
            return GraphExecutionResult(
                session_id=session_id,
                success=False,
                user_query=user_query,
                final_sql=None,
                query_result=[],
                data_summary=None,
                error_message=error_msg,
                execution_time=execution_time,
                confidence_scores={},
                node_results=[],
                debug_info={"error": str(e)}
            )
    
    def run_sync(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> GraphExecutionResult:
        """
        Synchronous wrapper for process_query.
        
        Args:
            user_query: User's natural language query
            session_id: Session identifier
            user_id: User identifier
            channel_id: Channel identifier
            context: Additional context
            max_retries: Maximum number of retries
            
        Returns:
            GraphExecutionResult with the processing outcome
        """
        return self.process_query(
            user_query=user_query,
            session_id=session_id,
            user_id=user_id,
            channel_id=channel_id,
            context=context,
            max_retries=max_retries
        )
    
    async def process_query_async(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> GraphExecutionResult:
        """
        Process a natural language query asynchronously.
        
        Args:
            user_query: User's natural language query
            session_id: Session identifier
            user_id: User identifier
            channel_id: Channel identifier
            context: Additional context
            max_retries: Maximum number of retries
            
        Returns:
            GraphExecutionResult with the processing outcome
        """
        if session_id is None:
            session_id = self.context_manager.create_session()
        
        logger.info(f"Processing query asynchronously for session {session_id}")
        
        # Add user message to conversation history
        self.context_manager.add_user_message(session_id, user_query)
        
        # Initialize state
        initial_state = initialize_state(
            user_query=user_query,
            user_id=user_id,
            channel_id=channel_id,
            session_id=session_id,
            context=context,
            max_retries=max_retries
        )
        
        # Add conversation context to state
        conversation_context = self.context_manager.get_conversation_context(session_id)
        if conversation_context:
            initial_state["context"]["conversation_history"] = conversation_context
        
        # Execute the graph asynchronously
        start_time = time.time()
        
        try:
            if self.enable_monitoring:
                self.monitor.start_execution(session_id)
            
            # Create run configuration
            run_config = {
                "configurable": {
                    "thread_id": session_id
                }
            }
            
            # Execute the graph asynchronously with recursion limit
            loop = asyncio.get_event_loop()
            try:
                # 재귀 제한 설정 (무한 루프 방지를 위해 100으로 증가)
                run_config["recursion_limit"] = 100
                final_state = await loop.run_in_executor(
                    None,
                    lambda: self.graph.invoke(initial_state, config=run_config)
                )
            except Exception as e:
                logger.error(f"Graph execution failed: {str(e)}")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error(f"Exception details: {str(e)}", exc_info=True)
                final_state = None
            
            execution_time = time.time() - start_time
            
            # Create result with safe final_state handling
            try:
                result = self._create_execution_result(
                    session_id=session_id,
                    final_state=final_state,
                    execution_time=execution_time,
                    user_query=user_query
                )
            except Exception as e:
                logger.error(f"Failed to create execution result: {str(e)}")
                logger.error(f"Final state type: {type(final_state)}")
                logger.error(f"Final state value: {final_state}")
                
                # Create a safe fallback result
                result = GraphExecutionResult(
                    session_id=session_id,
                    success=False,
                    user_query=user_query,
                    final_sql=None,
                    query_result=[],
                    data_summary=None,
                    error_message=f"Failed to create execution result: {str(e)}",
                    execution_time=execution_time,
                    confidence_scores={},
                    node_results=[],
                    conversation_response=None
                )
            
            # Add assistant message to conversation history
            summary = result.data_summary or "쿼리가 처리되었습니다."
            self.context_manager.add_assistant_message(
                session_id,
                summary,
                {"sql_query": result.final_sql, "success": result.success}
            )
            
            # Update session state
            self.context_manager.update_session_state(session_id, final_state)
            
            if self.enable_monitoring:
                self.monitor.end_execution(session_id, success=result.success)
            
            logger.info(f"Query processed asynchronously in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Async query processing failed: {str(e)}"
            
            logger.error(error_msg)
            
            if self.enable_monitoring:
                self.monitor.end_execution(session_id, success=False)
            
            # Create error result
            return GraphExecutionResult(
                session_id=session_id,
                success=False,
                user_query=user_query,
                final_sql=None,
                query_result=[],
                data_summary=None,
                error_message=error_msg,
                execution_time=execution_time,
                confidence_scores={},
                node_results=[],
                debug_info={"error": str(e)}
            )
    
    async def stream_query_execution(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> AsyncGenerator[ExecutionProgress, None]:
        """
        Stream query execution progress asynchronously.
        
        Args:
            user_query: User's natural language query
            session_id: Session identifier
            user_id: User identifier
            channel_id: Channel identifier
            context: Additional context
            max_retries: Maximum number of retries
            
        Yields:
            ExecutionProgress objects with current progress information
        """
        if session_id is None:
            session_id = self.context_manager.create_session()
        
        logger.info(f"Streaming query execution for session {session_id}")
        
        # Add user message to conversation history
        self.context_manager.add_user_message(session_id, user_query)
        
        # Initialize state
        initial_state = initialize_state(
            user_query=user_query,
            user_id=user_id,
            channel_id=channel_id,
            session_id=session_id,
            context=context,
            max_retries=max_retries
        )
        
        # Add conversation context to state
        conversation_context = self.context_manager.get_conversation_context(session_id)
        if conversation_context:
            initial_state["context"]["conversation_history"] = conversation_context
        
        start_time = time.time()
        
        try:
            if self.enable_monitoring:
                self.monitor.start_execution(session_id)
            
            # Create run configuration
            run_config = {
                "configurable": {
                    "thread_id": session_id
                }
            }
            
            # Stream the graph execution
            node_order = [
                "nlp_processing", "schema_mapping", "sql_generation",
                "sql_validation", "sql_execution", "data_summarization"
            ]
            
            async for event in self.graph.astream(initial_state, config=run_config):
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # Determine current node and progress
                current_node = "unknown"
                progress_percentage = 0.0
                
                if event:
                    for node_name in node_order:
                        if node_name in event:
                            current_node = node_name
                            progress_percentage = (node_order.index(node_name) + 1) / len(node_order) * 100
                            break
                
                # Estimate remaining time
                if progress_percentage > 0:
                    estimated_total = elapsed_time / (progress_percentage / 100)
                    estimated_remaining = estimated_total - elapsed_time
                else:
                    estimated_remaining = None
                
                # Create progress update
                progress = ExecutionProgress(
                    session_id=session_id,
                    current_node=current_node,
                    progress_percentage=progress_percentage,
                    elapsed_time=elapsed_time,
                    estimated_remaining=estimated_remaining,
                    status="running",
                    metadata={"event": event}
                )
                
                yield progress
            
            # Final progress update
            final_time = time.time()
            final_elapsed = final_time - start_time
            
            final_progress = ExecutionProgress(
                session_id=session_id,
                current_node="completed",
                progress_percentage=100.0,
                elapsed_time=final_elapsed,
                estimated_remaining=0.0,
                status="completed",
                metadata={"execution_completed": True}
            )
            
            yield final_progress
            
            if self.enable_monitoring:
                self.monitor.end_execution(session_id, success=True)
            
            logger.info(f"Query execution streamed successfully in {final_elapsed:.2f}s")
            
        except Exception as e:
            error_time = time.time()
            error_elapsed = error_time - start_time
            
            logger.error(f"Streaming execution failed: {str(e)}")
            
            # Error progress update
            error_progress = ExecutionProgress(
                session_id=session_id,
                current_node="error",
                progress_percentage=0.0,
                elapsed_time=error_elapsed,
                estimated_remaining=None,
                status="failed",
                metadata={"error": str(e)}
            )
            
            yield error_progress
            
            if self.enable_monitoring:
                self.monitor.end_execution(session_id, success=False)
    
    def _create_execution_result(
        self,
        session_id: str,
        final_state: AgentState,
        execution_time: float,
        user_query: str
    ) -> GraphExecutionResult:
        """Create execution result from final state."""
        # Handle None or invalid final_state
        if final_state is None:
            return GraphExecutionResult(
                session_id=session_id,
                success=False,
                user_query=user_query,
                final_sql=None,
                query_result=[],
                data_summary=None,
                error_message="Graph execution returned no result",
                execution_time=execution_time,
                confidence_scores={},
                node_results=[],
                debug_info={"error": "final_state is None"}
            )
        
        # Convert node results to dictionaries
        node_results = []
        if hasattr(final_state, "get") and "node_results" in final_state:
            for result in final_state["node_results"]:
                if hasattr(result, 'to_dict'):
                    node_results.append(result.to_dict())
                else:
                    node_results.append(result.__dict__ if hasattr(result, '__dict__') else str(result))
        elif hasattr(final_state, "node_results") and final_state.node_results:
            for result in final_state.node_results:
                if hasattr(result, 'to_dict'):
                    node_results.append(result.to_dict())
                else:
                    node_results.append(result.__dict__ if hasattr(result, '__dict__') else str(result))
        
        # Safe access to final_state attributes with None checks
        try:
            if hasattr(final_state, "get") and final_state is not None:
                # Dictionary-like access
                success = final_state.get("success", False)
                final_sql = final_state.get("final_sql") or final_state.get("sql_query")
                query_result = final_state.get("query_result", [])
                data_summary = final_state.get("data_summary")
                error_message = final_state.get("error_message")
                confidence_scores = final_state.get("confidence_scores", {})
                debug_info = final_state.get("debug_info", {})
                conversation_response = final_state.get("conversation_response")
            else:
                # Object attribute access with safe defaults
                success = getattr(final_state, "success", False) if final_state else False
                final_sql = getattr(final_state, "final_sql", None) or getattr(final_state, "sql_query", None) if final_state else None
                query_result = getattr(final_state, "query_result", []) if final_state else []
                data_summary = getattr(final_state, "data_summary", None) if final_state else None
                error_message = getattr(final_state, "error_message", None) if final_state else None
                confidence_scores = getattr(final_state, "confidence_scores", {}) if final_state else {}
                debug_info = getattr(final_state, "debug_info", {}) if final_state else {}
                conversation_response = getattr(final_state, "conversation_response", None) if final_state else None
        except Exception as e:
            logger.error(f"Error accessing final_state attributes: {str(e)}")
            # Fallback to safe defaults
            success = False
            final_sql = None
            query_result = []
            data_summary = None
            error_message = f"State access error: {str(e)}"
            confidence_scores = {}
            debug_info = {"error": str(e)}
            conversation_response = None
        
        return GraphExecutionResult(
            session_id=session_id,
            success=success,
            user_query=user_query,
            final_sql=final_sql,
            query_result=query_result,
            data_summary=data_summary,
            error_message=error_message,
            execution_time=execution_time,
            confidence_scores=confidence_scores,
            node_results=node_results,
            debug_info=debug_info,
            conversation_response=conversation_response
        )
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a session."""
        session = self.context_manager.get_session(session_id)
        if session:
            return {
                "session_id": session.session_id,
                "created_at": session.created_at,
                "last_accessed": session.last_accessed,
                "message_count": len(session.conversation_history.messages),
                "current_state": session.current_state is not None
            }
        return None
    
    def get_conversation_history(self, session_id: str, max_messages: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        session = self.context_manager.get_session(session_id)
        if session:
            recent_messages = session.conversation_history.get_recent_messages(max_messages)
            return [msg.to_dict() for msg in recent_messages]
        return []
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a session and its history."""
        return self.context_manager.remove_session(session_id)
    
    def get_monitoring_metrics(self) -> Optional[Dict[str, Any]]:
        """Get performance monitoring metrics."""
        if self.monitor:
            return self.monitor.get_metrics_summary()
        return None
    
    def get_health_status(self) -> Optional[Dict[str, Any]]:
        """Get system health status."""
        if self.monitor:
            return self.monitor.get_health_status()
        return None
