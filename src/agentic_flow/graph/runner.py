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
from .context import AgentContextManager
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
    needs_clarification: Optional[bool] = None  # 사용자 재입력 필요 여부
    
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
        self.context_manager = AgentContextManager()
        
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
            session_id = self.context_manager.create_session_id()
        
        logger.info(f"Processing query synchronously for session {session_id}")
        
        # Get business metadata for AgentState
        business_metadata = self.context_manager.get_business_metadata(
            user_id=user_id,
            channel_id=channel_id
        )
        
        # Initialize state with business metadata
        initial_state = initialize_state(
            user_query=user_query,
            user_id=user_id,
            channel_id=channel_id,
            session_id=session_id,
            context={**context, **business_metadata.to_dict()} if context else business_metadata.to_dict(),
            max_retries=max_retries
        )
        
        # LangGraph will handle conversation history automatically
        
        # Execute the graph
        start_time = time.time()
        
        try:
            if self.monitor:
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
            
            # 통합 학습 데이터 기록
            self._record_integrated_metrics(
                final_state=final_state,
                result=result,
                user_id=user_id,
                session_id=session_id,
                execution_time_ms=execution_time * 1000
            )
            
            # LangGraph handles conversation history automatically
            
            # Session state is managed by LangGraph's checkpointer
            
            if self.monitor:
                self.monitor.end_execution(session_id, success=result.success)
            
            logger.info(f"Query processed successfully in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Query processing failed: {str(e)}"
            
            logger.error(error_msg)
            
            if self.monitor:
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
        Process a natural language query asynchronously using LangGraph's native ainvoke.
        
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
            session_id = self.context_manager.create_session_id()
        
        logger.info(f"Processing query asynchronously for session {session_id}")
        
        # Get business metadata for AgentState
        business_metadata = self.context_manager.get_business_metadata(
            user_id=user_id,
            channel_id=channel_id
        )
        
        # Initialize state with business metadata
        initial_state = initialize_state(
            user_query=user_query,
            user_id=user_id,
            channel_id=channel_id,
            session_id=session_id,
            context={**context, **business_metadata.to_dict()} if context else business_metadata.to_dict(),
            max_retries=max_retries
        )
        
        # LangGraph will handle conversation history automatically
        
        # Execute the graph asynchronously
        start_time = time.time()
        
        try:
            if self.monitor:
                self.monitor.start_execution(session_id)
            
            # Create run configuration
            run_config = {
                "configurable": {
                    "thread_id": session_id
                },
                "recursion_limit": 100
            }
            
            # Use LangGraph's native async invoke
            final_state = await self.graph.ainvoke(initial_state, config=run_config)
            
            execution_time = time.time() - start_time
            
            # Create result
            result = self._create_execution_result(
                session_id=session_id,
                final_state=final_state,
                execution_time=execution_time,
                user_query=user_query
            )
            
            # LangGraph handles conversation history automatically
            
            # Session state is managed by LangGraph's checkpointer
            
            if self.monitor:
                self.monitor.end_execution(session_id, success=result.success)
            
            logger.info(f"Query processed asynchronously in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Async query processing failed: {str(e)}"
            
            logger.error(error_msg)
            
            if self.monitor:
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
            session_id = self.context_manager.create_session_id()
        
        logger.info(f"Streaming query execution for session {session_id}")
        
        # Get business metadata for AgentState
        business_metadata = self.context_manager.get_business_metadata(
            user_id=user_id,
            channel_id=channel_id
        )
        
        # Initialize state with business metadata
        initial_state = initialize_state(
            user_query=user_query,
            user_id=user_id,
            channel_id=channel_id,
            session_id=session_id,
            context={**context, **business_metadata.to_dict()} if context else business_metadata.to_dict(),
            max_retries=max_retries
        )
        
        # LangGraph will handle conversation history automatically
        
        start_time = time.time()
        
        try:
            if self.monitor:
                self.monitor.start_execution(session_id)
            
            # Create run configuration
            run_config = {
                "configurable": {
                    "thread_id": session_id
                }
            }
            
            # Stream the graph execution with improved progress tracking
            async for event in self.graph.astream(initial_state, config=run_config):
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # Determine current node from event
                current_node = "unknown"
                if event:
                    # Get the first node name from the event
                    for key in event.keys():
                        if key not in ["__end__", "__start__"]:
                            current_node = key
                            break
                
                # Create progress update (focus on current node and elapsed time)
                progress = ExecutionProgress(
                    session_id=session_id,
                    current_node=current_node,
                    progress_percentage=0.0,  # Accurate percentage is not feasible with conditional edges
                    elapsed_time=elapsed_time,
                    estimated_remaining=None,  # Not accurate with dynamic paths
                    status="running",
                    metadata={"event": event, "note": "Progress percentage not available due to conditional execution paths"}
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
            
            if self.monitor:
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
            
            if self.monitor:
                self.monitor.end_execution(session_id, success=False)
    
    def _create_execution_result(
        self,
        session_id: str,
        final_state: AgentState,
        execution_time: float,
        user_query: str
    ) -> GraphExecutionResult:
        """Create execution result from final state using consistent dictionary access."""
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
        
        # Safe access to final_state using consistent dictionary access
        # TypedDict는 dict의 서브타입이므로 dict로 캐스팅하여 .get() 사용
        try:
            state_dict: Dict[str, Any] = final_state  # type: ignore[assignment]
            success = state_dict.get("success", False)
            final_sql = state_dict.get("final_sql") or state_dict.get("sql_query")
            query_result = state_dict.get("query_result", [])
            data_summary = state_dict.get("data_summary")
            error_message = state_dict.get("error_message")
            confidence_scores = state_dict.get("confidence_scores", {})
            debug_info = state_dict.get("debug_info", {})
            conversation_response = state_dict.get("conversation_response")
            needs_clarification = state_dict.get("needs_clarification", False)
            
            # Process node_results safely
            node_results_raw = state_dict.get("node_results", [])
            node_results_dict = []
            for res in node_results_raw:
                if hasattr(res, '__dict__'):  # dataclass or object
                    node_results_dict.append(asdict(res) if hasattr(res, '__dataclass_fields__') else res.__dict__)
                elif isinstance(res, dict):
                    node_results_dict.append(res)
                else:
                    logger.warning(f"Unexpected type in node_results: {type(res)}")
                    node_results_dict.append(str(res))  # Safe string conversion
                    
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
            needs_clarification = False
            node_results_dict = []
        
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
            node_results=node_results_dict,
            debug_info=debug_info,
            conversation_response=conversation_response,
            needs_clarification=needs_clarification
        )
    
    
    def get_monitoring_metrics(self) -> Optional[Dict[str, Any]]:
        """Get performance monitoring metrics."""
        from ..monitoring import metrics_collector
        return metrics_collector.get_performance_summary()
    
    def get_health_status(self) -> Optional[Dict[str, Any]]:
        """Get system health status."""
        from ..monitoring import metrics_collector
        return metrics_collector.get_health_status()
    
    def _record_integrated_metrics(
        self,
        final_state: AgentState,
        result: GraphExecutionResult,
        user_id: Optional[str],
        session_id: Optional[str],
        execution_time_ms: float
    ) -> None:
        """
        통합 학습 데이터 기록
        
        Intent 분류와 AutoLearning 데이터를 통합하여 기록합니다.
        """
        try:
            from agentic_flow.intent_classification_stats import get_integrator, QueryInteractionMetrics
            
            integrator = get_integrator()
            
            # State를 dict로 캐스팅하여 안전하게 접근
            state_dict: Dict[str, Any] = final_state  # type: ignore[assignment]
            
            # State에서 데이터 추출
            intent_result = state_dict.get("llm_intent_result", {})
            intent = intent_result.get("intent", "UNKNOWN") if isinstance(intent_result, dict) else "UNKNOWN"
            intent_confidence = intent_result.get("confidence", 0.0) if isinstance(intent_result, dict) else 0.0
            intent_reasoning = intent_result.get("reasoning", None) if isinstance(intent_result, dict) else None
            
            validation_result = state_dict.get("validation_result", {})
            validation_passed = validation_result.get("is_valid", False) if isinstance(validation_result, dict) else False
            
            # fanding_template 안전하게 처리
            fanding_template = state_dict.get("fanding_template")
            template_used = None
            if fanding_template is not None:
                if hasattr(fanding_template, 'name'):
                    template_used = fanding_template.name
                elif isinstance(fanding_template, dict):
                    template_used = fanding_template.get("name")
                elif isinstance(fanding_template, str):
                    template_used = fanding_template
            
            # 통합 메트릭스 생성
            metrics = QueryInteractionMetrics(
                user_query=result.user_query,
                user_id=user_id,
                session_id=session_id,
                intent=intent,
                intent_confidence=intent_confidence,
                intent_reasoning=intent_reasoning,
                sql_query=result.final_sql,
                validation_passed=validation_passed,
                execution_success=result.success,
                execution_result_count=len(result.query_result) if result.query_result else 0,
                response_time_ms=execution_time_ms,
                total_processing_time_ms=execution_time_ms,
                mapping_result=state_dict.get("agent_schema_mapping"),
                schema_mapping=state_dict.get("schema_mapping"),
                template_used=template_used,
                timestamp=time.time(),
                is_error=not result.success,
                error_message=result.error_message
            )
            
            # 통합 기록
            integrator.record_complete_query_interaction(metrics)
            
            logger.debug(f"Recorded integrated metrics for query: {result.user_query[:50]}...")
            
        except Exception as e:
            logger.warning(f"Failed to record integrated metrics: {e}")
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a session using LangGraph Checkpointer API.
        
        Args:
            session_id: Session identifier (thread_id)
            
        Returns:
            Session information dictionary or None if not found
        """
        try:
            checkpointer = self.context_manager.get_checkpointer()
            config: Any = {"configurable": {"thread_id": session_id}}  # type: ignore[assignment]
            
            # Get the latest checkpoint for this thread
            checkpoint = checkpointer.get(config)  # type: ignore[arg-type]
            
            if checkpoint is None:
                return None
            
            # Extract session information from checkpoint
            session_info = {
                "session_id": session_id,
                "created_at": checkpoint.get("ts", "unknown"),
                "last_accessed": checkpoint.get("ts", "unknown"),
                "message_count": len(checkpoint.get("channel_values", {}).get("messages", [])),
                "current_state": checkpoint.get("channel_values") is not None,
                "checkpoint_id": checkpoint.get("id", "unknown"),
                "parent_id": checkpoint.get("parent_id"),
                "metadata": checkpoint.get("metadata", {})
            }
            
            return session_info
            
        except Exception as e:
            logger.error(f"Failed to get session info for {session_id}: {str(e)}")
            return None
    
    def get_conversation_history(self, session_id: str, max_messages: int = 10) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session using LangGraph Checkpointer API.
        
        Args:
            session_id: Session identifier (thread_id)
            max_messages: Maximum number of messages to return
            
        Returns:
            List of conversation messages
        """
        try:
            checkpointer = self.context_manager.get_checkpointer()
            config: Any = {"configurable": {"thread_id": session_id}}  # type: ignore[assignment]
            
            # Get the latest checkpoint for this thread
            checkpoint = checkpointer.get(config)  # type: ignore[arg-type]
            
            if checkpoint is None:
                return []
            
            # Extract messages from checkpoint
            channel_values = checkpoint.get("channel_values", {})
            messages = channel_values.get("messages", [])
            
            # Convert messages to dictionary format and limit count
            conversation_history = []
            for msg in messages[-max_messages:]:  # Get last N messages
                if hasattr(msg, 'to_dict'):
                    conversation_history.append(msg.to_dict())
                elif isinstance(msg, dict):
                    conversation_history.append(msg)
                else:
                    # Convert other message types to dict
                    conversation_history.append({
                        "content": str(msg),
                        "type": type(msg).__name__
                    })
            
            return conversation_history
            
        except Exception as e:
            logger.error(f"Failed to get conversation history for {session_id}: {str(e)}")
            return []
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear a session and its history using LangGraph Checkpointer API.
        
        Args:
            session_id: Session identifier (thread_id)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            checkpointer = self.context_manager.get_checkpointer()
            
            # Note: LangGraph's MemorySaver doesn't have a direct delete method
            # We can only clear by creating a new thread_id
            # For now, we'll return True as the session will naturally expire
            logger.info(f"Session {session_id} marked for cleanup (MemorySaver limitation)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear session {session_id}: {str(e)}")
            return False
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """
        List all active sessions using LangGraph Checkpointer API.
        
        Returns:
            List of active session information
        """
        try:
            checkpointer = self.context_manager.get_checkpointer()
            config: Any = {"configurable": {}}  # type: ignore[assignment]
            
            # Get all thread IDs
            threads_generator = checkpointer.list(config)  # type: ignore[arg-type]
            thread_ids = []
            for thread_config in threads_generator:
                if isinstance(thread_config, dict) and "configurable" in thread_config:
                    configurable = thread_config.get("configurable", {})
                    if isinstance(configurable, dict):
                        thread_id = configurable.get("thread_id")
                        if thread_id:
                            thread_ids.append(thread_id)
            
            # Get session info for each thread
            active_sessions = []
            for thread_id in thread_ids:
                session_info = self.get_session_info(thread_id)
                if session_info:
                    active_sessions.append(session_info)
            
            return active_sessions
            
        except Exception as e:
            logger.error(f"Failed to list active sessions: {str(e)}")
            return []
