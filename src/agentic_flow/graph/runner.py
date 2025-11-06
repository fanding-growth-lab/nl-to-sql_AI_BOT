"""
Graph Runner Implementation for LangGraph State Machine

This module implements the AgentGraphRunner class for executing the LangGraph
state machine with async support, progress monitoring, and error handling.
"""

import asyncio
import time
import uuid
import threading
import queue
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum

# CompiledGraph is not directly importable in newer versions

from .state_machine import AgentState, create_agent_graph, initialize_state, _identify_bottlenecks
from langgraph.checkpoint.memory import MemorySaver
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
    rag_schema_chunks_count: int = 0  # RAG로 검색된 스키마 청크 수
    
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
        # db_schema가 제공되지 않으면 get_cached_db_schema()로 로드 (성능 최적화)
        if db_schema is None or len(db_schema) == 0:
            from core.db import get_cached_db_schema
            db_schema = get_cached_db_schema()
            logger.debug("db_schema was not provided to AgentGraphRunner, loaded from cache")
        
        self.db_schema = db_schema
        self.config = config or {}
        self.enable_monitoring = enable_monitoring
        
        # Initialize components
        # Create checkpointer directly and share with graph
        self.checkpointer = MemorySaver()
        # Pass the checkpointer to graph so they share the same memory
        # db_schema는 create_agent_graph에서도 한 번 더 확인하지만, 여기서 이미 로드했으므로 빠름
        self.graph = create_agent_graph(db_schema, config, checkpointer=self.checkpointer)
        
        if enable_monitoring:
            self.monitor = PipelineMonitor()
            self.metrics_collector = MetricsCollector()
        else:
            self.monitor = None
            self.metrics_collector = None
        
        # 비동기 작업 큐 및 스레드 풀
        self._async_queue: Optional[queue.Queue] = None
        self._async_workers: List[threading.Thread] = []
        self._max_async_workers = self.config.get("max_async_workers", 4)
        self._async_active = False
        
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
            session_id = str(uuid.uuid4())
        
        logger.info(
            f"Processing query synchronously for session {session_id} "
            f"(user_id={user_id}, channel_id={channel_id})"
        )
        
        # Create business metadata dict for AgentState
        business_metadata = {
            "user_id": user_id,
            "channel_id": channel_id
        }
        
        # Restore conversation history from checkpointer for context awareness
        conversation_history = self._restore_conversation_history(
            session_id=session_id,
            user_id=user_id,
            channel_id=channel_id,
            max_messages=10
        )
        
        # Restore query result cache from previous state (하이브리드 접근)
        query_result_cache = self._restore_query_result_cache(
            session_id=session_id
        )
        
        # Initialize state with business metadata and conversation history
        initial_state = initialize_state(
            user_query=user_query,
            user_id=user_id,
            channel_id=channel_id,
            session_id=session_id,
            context={**context, **business_metadata} if context else business_metadata,
            max_retries=max_retries,
            conversation_history=conversation_history
        )
        
        # Restore query result cache to initial state
        if query_result_cache:
            initial_state["query_result_cache"] = query_result_cache
            logger.debug(f"Restored {len(query_result_cache)} cached query results for session {session_id}")
        
        # Log conversation history for debugging
        if conversation_history:
            logger.debug(f"Restored {len(conversation_history)} previous messages for session {session_id}")
        else:
            logger.debug(f"No previous conversation history found for session {session_id}")
        
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
            # LangGraph will automatically:
            # 1. Merge initial_state with previous checkpoint (if exists)
            # 2. Save final_state to checkpoint after execution
            # 3. Maintain conversation history through checkpointer
            final_state = self.graph.invoke(initial_state, config=run_config)
            
            # Update conversation history in state after execution
            # This ensures the current query and response are included in next call
            # Note: DataSummarizationNode should update conversation_history in state
            # so that LangGraph automatically saves it to checkpointer
            self._update_conversation_history_in_state(final_state, user_query)
            
            execution_time = time.time() - start_time
            
            # Debug logging (excluding RAG chunks/context to avoid cluttering logs)
            if hasattr(final_state, 'keys'):
                state_summary = {k: v for k, v in final_state.items() 
                               if k not in ['rag_schema_chunks', 'rag_schema_context']}
                # Add summary for RAG fields
                rag_chunks = final_state.get("rag_schema_chunks", [])
                rag_context = final_state.get("rag_schema_context", "")
                if rag_chunks:
                    state_summary['rag_schema_chunks'] = f"<{len(rag_chunks)} chunks>"
                if rag_context:
                    state_summary['rag_schema_context'] = f"<{len(rag_context)} chars>"
                
                # Debug info 확인
                debug_info_check = final_state.get("debug_info", {})
                if isinstance(debug_info_check, dict):
                    node_perf = debug_info_check.get("node_performance", {})
                    if node_perf:
                        state_summary['debug_info'] = f"<node_performance: {len(node_perf)} nodes>"
                
                logger.debug(f"Graph execution completed. final_state type: {type(final_state)}, summary: {state_summary}")
            else:
                logger.debug(f"Graph execution completed. final_state type: {type(final_state)}")
            
            # RAG 정보 디버깅: final_state에 rag_schema_chunks가 있는지 확인
            if hasattr(final_state, 'get'):
                rag_chunks = final_state.get("rag_schema_chunks", [])
                rag_context = final_state.get("rag_schema_context", "")
                if rag_chunks:
                    logger.info(f"✅ Found {len(rag_chunks)} RAG chunks in final_state")
                elif rag_context:
                    logger.info(f"✅ Found RAG context (length: {len(rag_context)}) but no chunks")
                else:
                    logger.warning(f"⚠️ No RAG information found in final_state. Keys: {list(final_state.keys())[:30] if hasattr(final_state, 'keys') else 'N/A'}")
                
                # Debug info 확인
                debug_info_check = final_state.get("debug_info", {})
                if isinstance(debug_info_check, dict):
                    node_perf = debug_info_check.get("node_performance", {})
                    if node_perf:
                        logger.info(f"[PERF] Found {len(node_perf)} node performance metrics in final_state")
                        for node_name in node_perf.keys():
                            logger.debug(f"  - {node_name}: {node_perf[node_name].get('execution_time', 0):.3f}s")
                    else:
                        logger.warning(f"[PERF] No node_performance found in final_state.debug_info. Keys: {list(debug_info_check.keys()) if debug_info_check else 'empty'}")
                else:
                    logger.warning(f"[PERF] debug_info is not a dict in final_state: {type(debug_info_check)}")
            
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
            
            # 정확도 평가 및 피드백 루프 통합 (태스크 7.5)
            self._evaluate_and_learn(
                result=result,
                final_state=final_state,
                session_id=session_id,
                user_id=user_id
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
            session_id = str(uuid.uuid4())
        
        logger.info(f"Processing query asynchronously for session {session_id}")
        
        # Create business metadata dict for AgentState
        business_metadata = {
            "user_id": user_id,
            "channel_id": channel_id
        }
        
        # Restore conversation history from checkpointer for context awareness
        conversation_history = self._restore_conversation_history(
            session_id=session_id,
            user_id=user_id,
            channel_id=channel_id,
            max_messages=10
        )
        
        # Restore query result cache from previous state (하이브리드 접근)
        query_result_cache = self._restore_query_result_cache(
            session_id=session_id
        )
        
        # Initialize state with business metadata and conversation history
        initial_state = initialize_state(
            user_query=user_query,
            user_id=user_id,
            channel_id=channel_id,
            session_id=session_id,
            context={**context, **business_metadata} if context else business_metadata,
            max_retries=max_retries,
            conversation_history=conversation_history
        )
        
        # Restore query result cache to initial state
        if query_result_cache:
            initial_state["query_result_cache"] = query_result_cache
            logger.debug(f"Restored {len(query_result_cache)} cached query results for async session {session_id}")
        
        # Log conversation history for debugging
        if conversation_history:
            logger.debug(f"Restored {len(conversation_history)} previous messages for async session {session_id}")
        
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
            # LangGraph will automatically:
            # 1. Merge initial_state with previous checkpoint (if exists)
            # 2. Save final_state to checkpoint after execution
            final_state = await self.graph.ainvoke(initial_state, config=run_config)
            
            # Update conversation history in state after execution
            # Note: DataSummarizationNode should update conversation_history in state
            # so that LangGraph automatically saves it to checkpointer
            self._update_conversation_history_in_state(final_state, user_query)
            
            execution_time = time.time() - start_time
            
            # Create result
            result = self._create_execution_result(
                session_id=session_id,
                final_state=final_state,
                execution_time=execution_time,
                user_query=user_query
            )
            
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
            session_id = str(uuid.uuid4())
        
        logger.info(f"Streaming query execution for session {session_id}")
        
        # Create business metadata dict for AgentState
        business_metadata = {
            "user_id": user_id,
            "channel_id": channel_id
        }
        
        # Initialize state with business metadata
        initial_state = initialize_state(
            user_query=user_query,
            user_id=user_id,
            channel_id=channel_id,
            session_id=session_id,
            context={**context, **business_metadata} if context else business_metadata,
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
            
            # RAG 스키마 청크 수 추출
            rag_schema_chunks = state_dict.get("rag_schema_chunks", [])
            rag_schema_chunks_count = len(rag_schema_chunks) if isinstance(rag_schema_chunks, list) else 0
            
            # 디버깅: RAG 정보 확인
            if rag_schema_chunks_count == 0:
                # state_dict의 모든 키 확인 (디버깅용)
                logger.debug(f"RAG chunks not found in final_state. Available keys: {list(state_dict.keys())[:20]}")
                # rag_schema로 시작하는 키 확인
                rag_keys = [k for k in state_dict.keys() if 'rag' in k.lower() or 'schema' in k.lower()]
                if rag_keys:
                    logger.debug(f"Found RAG-related keys: {rag_keys}")
            
            # debug_info 초기화 및 복사 (None 방지)
            if debug_info is None:
                debug_info = {}
            elif not isinstance(debug_info, dict):
                debug_info = {}
            else:
                # 기존 dict를 복사하여 수정 (원본 보존)
                debug_info = dict(debug_info)
            
            # debug_info에 RAG 정보 추가 (테스트 및 디버깅용)
            if rag_schema_chunks_count > 0:
                # 전체 청크 내용 대신 요약만 저장
                debug_info["rag_schema_chunks_count"] = rag_schema_chunks_count
                # 테이블 이름과 관련도만 저장 (전체 내용 제외)
                if isinstance(rag_schema_chunks, list):
                    rag_summary = []
                    for chunk in rag_schema_chunks[:5]:
                        if isinstance(chunk, dict):
                            rag_summary.append({
                                'table_name': chunk.get('table_name', 'Unknown'),
                                'relevance_score': chunk.get('relevance_score', 0.0)
                            })
                    debug_info["rag_schema_summary"] = rag_summary
                rag_context = state_dict.get("rag_schema_context", "")
                if rag_context:
                    debug_info["rag_schema_context_length"] = len(rag_context)
            else:
                # RAG 정보가 없어도 debug_info에 추가 (0으로 표시)
                debug_info["rag_schema_chunks_count"] = 0
            
            # 병목 지점 분석
            node_performance = debug_info.get("node_performance", {})
            if node_performance:
                bottleneck_analysis = _identify_bottlenecks(node_performance)
                debug_info["bottleneck_analysis"] = bottleneck_analysis
                
                # 병목 지점이 있으면 로깅
                if bottleneck_analysis.get("bottleneck_nodes"):
                    logger.info(
                        f"Performance analysis: {len(bottleneck_analysis['bottleneck_nodes'])} bottleneck(s) identified. "
                        f"Total execution time: {bottleneck_analysis['total_time']:.3f}s"
                    )
                    for bottleneck in bottleneck_analysis["bottleneck_nodes"]:
                        logger.info(
                            f"  - {bottleneck['node_name']}: {bottleneck['execution_time']:.3f}s "
                            f"({bottleneck['percentage']:.1f}% of total)"
                        )
                    
                    # 최적화 제안 로깅
                    suggestions = bottleneck_analysis.get("optimization_suggestions", [])
                    if suggestions:
                        logger.info("Optimization suggestions:")
                        for suggestion in suggestions:
                            logger.info(f"  [{suggestion['priority'].upper()}] {suggestion['suggestion']}")
            else:
                logger.debug(f"[PERF] No node_performance found in debug_info for bottleneck analysis. debug_info keys: {list(debug_info.keys())}")
                    
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
            rag_schema_chunks_count = 0
        
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
            needs_clarification=needs_clarification,
            rag_schema_chunks_count=rag_schema_chunks_count
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
    
    def _evaluate_and_learn(
        self,
        result: GraphExecutionResult,
        final_state: AgentState,
        session_id: str,
        user_id: Optional[str]
    ) -> None:
        """
        정확도 평가 및 학습 데이터 통합 (태스크 7.5)
        
        Args:
            result: 그래프 실행 결과
            final_state: 최종 상태
            session_id: 세션 ID
            user_id: 사용자 ID
        """
        try:
            # 대화 응답이거나 SQL이 없는 경우 건너뛰기
            if result.conversation_response or not result.final_sql:
                return
            
            from agentic_flow.sql_accuracy_evaluator import get_accuracy_evaluator
            from agentic_flow.learning_data_integrator import get_learning_integrator
            from agentic_flow.validation_node import ValidationResult
            
            # 정확도 평가기 및 학습 통합기 가져오기
            accuracy_evaluator = get_accuracy_evaluator()
            learning_integrator = get_learning_integrator()
            
            # 검증 결과 추출
            state_dict: Dict[str, Any] = final_state  # type: ignore[assignment]
            validation_result = state_dict.get("validation_result", {})
            sql_params = state_dict.get("sql_params")
            
            # SQL 정확도 평가
            accuracy_metrics = accuracy_evaluator.evaluate_sql(
                sql_query=result.final_sql,
                user_query=result.user_query,
                validation_result=validation_result if isinstance(validation_result, dict) else None,
                sql_params=sql_params if isinstance(sql_params, dict) else None
            )
            
            # 정확도 메트릭을 debug_info에 추가 (기존 데이터 보존)
            if result.debug_info is None:
                result.debug_info = {}
            elif not isinstance(result.debug_info, dict):
                result.debug_info = {}
            # 기존 debug_info의 내용을 유지하면서 accuracy_evaluation만 추가
            result.debug_info["accuracy_evaluation"] = {
                "accuracy_score": accuracy_metrics.accuracy_score,
                "accuracy_level": accuracy_metrics.accuracy_level.value,
                "syntax_valid": accuracy_metrics.syntax_valid,
                "execution_success": accuracy_metrics.execution_success,
                "schema_compatibility": accuracy_metrics.schema_compatibility,
                "result_quality": accuracy_metrics.result_quality
            }
            
            # 학습 데이터로 통합
            learning_integrator.integrate_accuracy_evaluation(
                user_query=result.user_query,
                generated_sql=result.final_sql,
                accuracy_metrics=accuracy_metrics,
                session_id=session_id
            )
            
            logger.debug(
                f"Accuracy evaluation completed: {accuracy_metrics.accuracy_level.value} "
                f"({accuracy_metrics.accuracy_score:.2%})"
            )
            
        except Exception as e:
            logger.warning(f"Failed to evaluate and learn: {e}")
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a session using LangGraph Checkpointer API.
        
        Args:
            session_id: Session identifier (thread_id)
            
        Returns:
            Session information dictionary or None if not found
        """
        try:
            checkpointer = self.checkpointer
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
            List of conversation messages (raw format from checkpointer)
        """
        try:
            checkpointer = self.checkpointer
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
    
    def _restore_conversation_history(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        max_messages: int = 10
    ) -> List[Dict[str, str]]:
        """
        Restore conversation history from checkpointer and convert to standardized format.
        
        This method:
        1. Retrieves history from LangGraph checkpointer using session_id (thread_id)
        2. Extracts previous user queries and assistant responses from AgentState
        3. For DM Thread sessions, also merges history from the main DM session
        4. Converts to standardized format: [{"role": "user|assistant", "content": "..."}]
        
        Args:
            session_id: Session identifier (thread_id for LangGraph)
            user_id: User identifier (for validation and DM session lookup)
            channel_id: Channel identifier (for validation)
            max_messages: Maximum number of messages to restore
            
        Returns:
            List of conversation messages in standardized format
        """
        try:
            checkpointer = self.checkpointer
            config: Any = {"configurable": {"thread_id": session_id}}  # type: ignore[assignment]
            
            # Get the latest checkpoint for this thread
            checkpoint = checkpointer.get(config)  # type: ignore[arg-type]
            
            # Check if this is a DM Thread session (starts with "slack_" and contains channel_id)
            # DM Thread: slack_D{channel_id}_{thread_ts}
            # DM Main: slack_dm_user_{user_id}
            is_dm_thread = (
                session_id.startswith("slack_") and 
                not session_id.startswith("slack_dm_user_") and
                channel_id and 
                channel_id.startswith('D') and
                user_id is not None
            )
            
            # For DM Thread sessions, also restore history from main DM session
            main_dm_history: List[Dict[str, str]] = []
            if is_dm_thread and user_id:
                main_dm_session_id = f"slack_dm_user_{user_id}"
                main_dm_config: Any = {"configurable": {"thread_id": main_dm_session_id}}  # type: ignore[assignment]
                main_dm_checkpoint = checkpointer.get(main_dm_config)  # type: ignore[arg-type]
                
                if main_dm_checkpoint:
                    main_dm_values = main_dm_checkpoint.get("channel_values", {})
                    main_dm_history_raw = main_dm_values.get("conversation_history", [])
                    if main_dm_history_raw and isinstance(main_dm_history_raw, list):
                        # Get last N messages from main DM session (to avoid token overflow)
                        main_dm_history = main_dm_history_raw[-max_messages:]
                        logger.debug(
                            f"Restored {len(main_dm_history)} messages from main DM session "
                            f"({main_dm_session_id}) for thread {session_id}"
                        )
            
            if checkpoint is None:
                # If no thread checkpoint but we have main DM history, return that
                if main_dm_history:
                    logger.debug(
                        f"No checkpoint found for thread {session_id}, "
                        f"using main DM history ({len(main_dm_history)} messages)"
                    )
                    return main_dm_history
                logger.debug(f"No checkpoint found for session {session_id}, starting fresh conversation")
                return []
            
            # Extract state from checkpoint
            channel_values = checkpoint.get("channel_values", {})
            if not channel_values:
                # If no thread values but we have main DM history, return that
                if main_dm_history:
                    logger.debug(
                        f"No channel values in thread {session_id}, "
                        f"using main DM history ({len(main_dm_history)} messages)"
                    )
                    return main_dm_history
                logger.debug(f"No channel values in checkpoint for session {session_id}")
                return []
            
            # First, try to get conversation_history directly from state (our custom field)
            conversation_history = channel_values.get("conversation_history")
            if conversation_history and isinstance(conversation_history, list):
                logger.debug(f"Found conversation_history in checkpoint: {len(conversation_history)} messages")
                
                # For DM Thread sessions, merge with main DM history
                if is_dm_thread and main_dm_history:
                    # Merge histories: main DM history first, then thread history
                    # Remove duplicates by content (simple deduplication)
                    merged_history = []
                    seen_content = set()
                    
                    # Add main DM history first (context from main DM)
                    for msg in main_dm_history:
                        content = msg.get("content", "")
                        content_key = f"{msg.get('role', '')}:{content[:50]}"  # First 50 chars for dedup
                        if content_key not in seen_content:
                            merged_history.append(msg)
                            seen_content.add(content_key)
                    
                    # Add thread history (more recent context)
                    for msg in conversation_history:
                        content = msg.get("content", "")
                        content_key = f"{msg.get('role', '')}:{content[:50]}"
                        if content_key not in seen_content:
                            merged_history.append(msg)
                            seen_content.add(content_key)
                    
                    # Limit total size
                    if len(merged_history) > max_messages * 2:  # Allow more for merged history
                        merged_history = merged_history[-max_messages * 2:]
                    
                    logger.debug(
                        f"Merged conversation history: {len(main_dm_history)} from main DM + "
                        f"{len(conversation_history)} from thread = {len(merged_history)} total"
                    )
                    return merged_history
                
                # Validate that user_id matches (if provided)
                # For DM sessions (session_id starts with "slack_dm_user_"), we only validate user_id
                # For channel sessions, we validate both user_id and channel_id
                is_dm_session = session_id.startswith("slack_dm_user_")
                
                if user_id:
                    state_user_id = channel_values.get("user_id")
                    
                    if state_user_id != user_id:
                        logger.warning(
                            f"Session {session_id} user_id mismatch: "
                            f"expected user={user_id}, found user={state_user_id}. "
                            f"Starting fresh conversation for security."
                        )
                        return main_dm_history if is_dm_thread and main_dm_history else []
                
                # For channel sessions, also validate channel_id
                if channel_id and not is_dm_session:
                    state_channel_id = channel_values.get("channel_id")
                    if state_channel_id and state_channel_id != channel_id:
                        logger.warning(
                            f"Session {session_id} channel_id mismatch: "
                            f"expected channel={channel_id}, found channel={state_channel_id}. "
                            f"Starting fresh conversation for security."
                        )
                        return []
                
                # Return last N messages
                return conversation_history[-max_messages:]
            
            # Fallback: Extract from user_query and conversation_response/data_summary
            conversation_history = []
            
            # Get messages from LangGraph's message list (if available)
            messages = channel_values.get("messages", [])
            if messages:
                for msg in messages[-max_messages:]:
                    if hasattr(msg, 'content'):
                        # LangGraph message format
                        content = msg.content if hasattr(msg, 'content') else str(msg)
                        msg_type = msg.type if hasattr(msg, 'type') else 'unknown'
                        
                        # Map LangGraph message types to our format
                        role = "assistant" if msg_type in ["ai", "assistant", "system"] else "user"
                        conversation_history.append({
                            "role": role,
                            "content": content
                        })
                    elif isinstance(msg, dict):
                        # Dictionary format
                        content = msg.get("content", str(msg))
                        role = msg.get("role", "user")
                        conversation_history.append({
                            "role": role,
                            "content": content
                        })
            
            # Also extract from state if messages are not available
            if not conversation_history:
                # Try to extract from previous state's user_query and conversation_response
                prev_user_query = channel_values.get("user_query")
                prev_response = channel_values.get("conversation_response") or channel_values.get("data_summary")
                
                if prev_user_query:
                    conversation_history.append({
                        "role": "user",
                        "content": prev_user_query
                    })
                
                if prev_response:
                    conversation_history.append({
                        "role": "assistant",
                        "content": prev_response
                    })
            
            # For DM Thread sessions without thread history, merge with main DM history
            if is_dm_thread and main_dm_history and not conversation_history:
                logger.debug(
                    f"No thread history found for {session_id}, "
                    f"using main DM history ({len(main_dm_history)} messages)"
                )
                return main_dm_history
            
            # For DM Thread sessions with thread history, merge with main DM history
            if is_dm_thread and main_dm_history and conversation_history:
                # Merge histories: main DM history first, then thread history
                merged_history = []
                seen_content = set()
                
                # Add main DM history first (context from main DM)
                for msg in main_dm_history:
                    content = msg.get("content", "")
                    content_key = f"{msg.get('role', '')}:{content[:50]}"
                    if content_key not in seen_content:
                        merged_history.append(msg)
                        seen_content.add(content_key)
                
                # Add thread history (more recent context)
                for msg in conversation_history:
                    content = msg.get("content", "")
                    content_key = f"{msg.get('role', '')}:{content[:50]}"
                    if content_key not in seen_content:
                        merged_history.append(msg)
                        seen_content.add(content_key)
                
                # Limit total size
                if len(merged_history) > max_messages * 2:
                    merged_history = merged_history[-max_messages * 2:]
                
                logger.debug(
                    f"Merged conversation history (fallback): {len(main_dm_history)} from main DM + "
                    f"{len(conversation_history)} from thread = {len(merged_history)} total"
                )
                return merged_history
            
            # Validate that user_id matches (if provided)
            # For DM sessions (session_id starts with "slack_dm_user_"), we only validate user_id
            # For channel sessions, we validate both user_id and channel_id
            is_dm_session = session_id.startswith("slack_dm_user_")
            
            if user_id:
                state_user_id = channel_values.get("user_id")
                
                if state_user_id != user_id:
                    logger.warning(
                        f"Session {session_id} user_id mismatch: "
                        f"expected user={user_id}, found user={state_user_id}. "
                        f"Starting fresh conversation for security."
                    )
                    return main_dm_history if is_dm_thread and main_dm_history else []
            
            # For channel sessions, also validate channel_id
            if channel_id and not is_dm_session:
                state_channel_id = channel_values.get("channel_id")
                if state_channel_id and state_channel_id != channel_id:
                    logger.warning(
                        f"Session {session_id} channel_id mismatch: "
                        f"expected channel={channel_id}, found channel={state_channel_id}. "
                        f"Starting fresh conversation for security."
                    )
                    return []
            
            logger.debug(f"Restored {len(conversation_history)} messages from checkpoint for session {session_id}")
            return conversation_history[-max_messages:]  # Return last N messages
            
        except Exception as e:
            logger.error(f"Failed to restore conversation history for {session_id}: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _update_conversation_history_in_state(
        self,
        final_state: AgentState,
        current_user_query: str
    ) -> None:
        """
        Update conversation history in final state to include current query and response.
        
        This ensures that the next call to process_query will have access to the
        current conversation turn.
        
        Args:
            final_state: Final state from graph execution
            current_user_query: Current user query that was processed
        """
        try:
            state_dict: Dict[str, Any] = final_state  # type: ignore[assignment]
            
            # Get existing history or initialize empty list
            history = state_dict.get("conversation_history", [])
            if not isinstance(history, list):
                history = []
            
            # Add current user query
            history.append({
                "role": "user",
                "content": current_user_query
            })
            
            # Add assistant response (conversation_response or data_summary)
            response = state_dict.get("conversation_response") or state_dict.get("data_summary")
            if response:
                history.append({
                    "role": "assistant",
                    "content": response
                })
            
            # Limit history size to prevent token overflow
            max_history = 20  # Keep last 20 messages (10 user + 10 assistant)
            if len(history) > max_history:
                history = history[-max_history:]
            
            # Update state with updated history
            state_dict["conversation_history"] = history
            
            logger.debug(f"Updated conversation history: {len(history)} messages total")
            
        except Exception as e:
            logger.warning(f"Failed to update conversation history in state: {str(e)}")
            # Non-critical error, continue execution
    
    def _restore_query_result_cache(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Restore query result cache from previous state using checkpointer (하이브리드 접근)
        
        Args:
            session_id: Session identifier (thread_id for LangGraph)
            
        Returns:
            Query result cache dictionary or None if not found
        """
        try:
            checkpointer = self.checkpointer
            config: Any = {"configurable": {"thread_id": session_id}}  # type: ignore[assignment]
            
            # Get the latest checkpoint for this thread
            checkpoint = checkpointer.get(config)  # type: ignore[arg-type]
            
            if not checkpoint or not checkpoint.get("channel_values"):
                logger.debug(f"No checkpoint found for session {session_id}")
                return None
            
            # Extract state from checkpoint
            state = checkpoint["channel_values"]
            
            # Extract query_result_cache from state
            query_result_cache = state.get("query_result_cache")
            
            if query_result_cache and isinstance(query_result_cache, dict):
                logger.debug(f"Restored query result cache with {len(query_result_cache)} entries for session {session_id}")
                return query_result_cache
            else:
                logger.debug(f"No query result cache found in checkpoint for session {session_id}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to restore query result cache for session {session_id}: {str(e)}")
            return None
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear a session and its history using LangGraph Checkpointer API.
        
        Args:
            session_id: Session identifier (thread_id)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            checkpointer = self.checkpointer
            
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
            checkpointer = self.checkpointer
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
