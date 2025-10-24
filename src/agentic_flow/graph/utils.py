"""
Utility Functions for LangGraph State Machine

This module provides utility functions for state management, validation,
serialization, and other common operations in the LangGraph pipeline.
"""

import json
import pickle
import base64
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import asdict, is_dataclass

from .state_machine import AgentState, ExecutionStatus, NodeExecutionResult
from ..monitoring import PipelineMonitor
from core.logging import get_logger

logger = get_logger(__name__)


def validate_state_transition(
    current_state: AgentState,
    next_node: str,
    validation_rules: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate state transition to next node.
    
    Args:
        current_state: Current pipeline state
        next_node: Next node to transition to
        validation_rules: Optional validation rules
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Basic state validation
        if not current_state.get("user_query"):
            return False, "Missing user query in state"
        
        # Node-specific validation
        if next_node == "sql_execution":
            if not current_state.get("sql_query") and not current_state.get("validated_sql"):
                return False, "Missing SQL query for execution"
        
        elif next_node == "data_summarization":
            if not current_state.get("query_result"):
                return False, "Missing query result for summarization"
        
        # Custom validation rules
        if validation_rules:
            for rule_name, rule_func in validation_rules.items():
                try:
                    if not rule_func(current_state, next_node):
                        return False, f"Validation rule '{rule_name}' failed"
                except Exception as e:
                    logger.warning(f"Validation rule '{rule_name}' error: {e}")
        
        return True, None
        
    except Exception as e:
        logger.error(f"State transition validation error: {e}")
        return False, f"Validation error: {str(e)}"


def serialize_state(state: AgentState, format_type: str = "json") -> str:
    """
    Serialize pipeline state to string.
    
    Args:
        state: Pipeline state to serialize
        format_type: Serialization format ("json" or "pickle")
        
    Returns:
        Serialized state as string
    """
    try:
        if format_type == "json":
            # Convert complex objects to serializable format
            serializable_state = _make_serializable(state)
            return json.dumps(serializable_state, ensure_ascii=False, indent=2)
        
        elif format_type == "pickle":
            # Use pickle for binary serialization
            pickled = pickle.dumps(state)
            return base64.b64encode(pickled).decode('utf-8')
        
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
            
    except Exception as e:
        logger.error(f"State serialization error: {e}")
        raise


def deserialize_state(serialized_state: str, format_type: str = "json") -> AgentState:
    """
    Deserialize string back to pipeline state.
    
    Args:
        serialized_state: Serialized state string
        format_type: Serialization format ("json" or "pickle")
        
    Returns:
        Deserialized pipeline state
    """
    try:
        if format_type == "json":
            state_dict = json.loads(serialized_state)
            return _restore_from_serializable(state_dict)
        
        elif format_type == "pickle":
            # Decode from base64 and unpickle
            pickled = base64.b64decode(serialized_state.encode('utf-8'))
            return pickle.loads(pickled)
        
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
            
    except Exception as e:
        logger.error(f"State deserialization error: {e}")
        raise


def _make_serializable(obj: Any) -> Any:
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {key: _make_serializable(value) for key, value in obj.items()}
    
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    
    elif is_dataclass(obj):
        return _make_serializable(asdict(obj))
    
    elif isinstance(obj, datetime):
        return obj.isoformat()
    
    elif hasattr(obj, '__dict__'):
        return _make_serializable(obj.__dict__)
    
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    
    else:
        # For other types, convert to string
        return str(obj)


def _restore_from_serializable(obj: Any) -> Any:
    """Restore objects from JSON-serializable format."""
    if isinstance(obj, dict):
        # Check for special markers
        if '__type__' in obj:
            return _restore_special_object(obj)
        else:
            return {key: _restore_from_serializable(value) for key, value in obj.items()}
    
    elif isinstance(obj, (list, tuple)):
        return [_restore_from_serializable(item) for item in obj]
    
    elif isinstance(obj, str):
        # Try to parse datetime
        try:
            return datetime.fromisoformat(obj)
        except (ValueError, TypeError):
            return obj
    
    else:
        return obj


def _restore_special_object(obj: Dict[str, Any]) -> Any:
    """Restore special objects from serialized format."""
    obj_type = obj.get('__type__')
    
    if obj_type == 'datetime':
        return datetime.fromisoformat(obj['value'])
    
    elif obj_type == 'NodeExecutionResult':
        return NodeExecutionResult(**obj['value'])
    
    else:
        return obj


def merge_states(base_state: AgentState, update_state: Dict[str, Any]) -> AgentState:
    """
    Merge update state into base state.
    
    Args:
        base_state: Base pipeline state
        update_state: State updates to merge
        
    Returns:
        Merged pipeline state
    """
    try:
        merged_state = base_state.copy()
        
        for key, value in update_state.items():
            if key in merged_state:
                # Handle special merging logic for specific fields
                if key == "confidence_scores" and isinstance(value, dict):
                    if isinstance(merged_state[key], dict):
                        merged_state[key].update(value)
                    else:
                        merged_state[key] = value
                
                elif key == "debug_info" and isinstance(value, dict):
                    if isinstance(merged_state[key], dict):
                        merged_state[key].update(value)
                    else:
                        merged_state[key] = value
                
                elif key == "node_results" and isinstance(value, list):
                    if isinstance(merged_state[key], list):
                        merged_state[key].extend(value)
                    else:
                        merged_state[key] = value
                
                else:
                    merged_state[key] = value
            else:
                merged_state[key] = value
        
        return merged_state
        
    except Exception as e:
        logger.error(f"State merge error: {e}")
        return base_state


def extract_state_summary(state: AgentState) -> Dict[str, Any]:
    """
    Extract a summary of the pipeline state.
    
    Args:
        state: Pipeline state
        
    Returns:
        State summary dictionary
    """
    try:
        summary = {
            "session_id": state.get("session_id"),
            "user_query": state.get("user_query", "")[:100] + "..." if len(state.get("user_query", "")) > 100 else state.get("user_query", ""),
            "current_node": state.get("current_node"),
            "execution_status": state.get("execution_status"),
            "success": state.get("success", False),
            "processing_time": state.get("processing_time", 0.0),
            "retry_count": state.get("retry_count", 0),
            "max_retries": state.get("max_retries", 3),
            "has_sql_query": bool(state.get("sql_query") or state.get("validated_sql")),
            "has_query_result": bool(state.get("query_result")),
            "has_error": bool(state.get("error_message")),
            "confidence_scores": state.get("confidence_scores", {}),
            "node_count": len(state.get("node_results", [])),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add SQL query preview if available
        sql_query = state.get("sql_query") or state.get("validated_sql")
        if sql_query:
            summary["sql_preview"] = sql_query[:50] + "..." if len(sql_query) > 50 else sql_query
        
        # Add result count if available
        query_result = state.get("query_result", [])
        if query_result:
            summary["result_count"] = len(query_result)
        
        return summary
        
    except Exception as e:
        logger.error(f"State summary extraction error: {e}")
        return {"error": str(e), "timestamp": datetime.now().isoformat()}


def validate_state_integrity(state: AgentState) -> Tuple[bool, List[str]]:
    """
    Validate the integrity of pipeline state.
    
    Args:
        state: Pipeline state to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    try:
        # Required fields validation
        required_fields = ["user_query", "session_id"]
        for field in required_fields:
            if not state.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Data type validation
        if not isinstance(state.get("confidence_scores", {}), dict):
            errors.append("confidence_scores must be a dictionary")
        
        if not isinstance(state.get("node_results", []), list):
            errors.append("node_results must be a list")
        
        if not isinstance(state.get("retry_count", 0), int):
            errors.append("retry_count must be an integer")
        
        if not isinstance(state.get("max_retries", 3), int):
            errors.append("max_retries must be an integer")
        
        # Range validation
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        if retry_count > max_retries:
            errors.append("retry_count cannot exceed max_retries")
        
        # SQL query validation
        sql_query = state.get("sql_query") or state.get("validated_sql")
        if sql_query and not isinstance(sql_query, str):
            errors.append("SQL query must be a string")
        
        # Query result validation
        query_result = state.get("query_result", [])
        if query_result and not isinstance(query_result, list):
            errors.append("query_result must be a list")
        
        # Confidence scores validation
        confidence_scores = state.get("confidence_scores", {})
        for score_name, score_value in confidence_scores.items():
            if not isinstance(score_value, (int, float)):
                errors.append(f"Confidence score '{score_name}' must be numeric")
            elif not 0 <= score_value <= 1:
                errors.append(f"Confidence score '{score_name}' must be between 0 and 1")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        logger.error(f"State integrity validation error: {e}")
        return False, [f"Validation error: {str(e)}"]


def create_state_checkpoint(state: AgentState, checkpoint_name: Optional[str] = None) -> str:
    """
    Create a checkpoint of the current state.
    
    Args:
        state: Pipeline state to checkpoint
        checkpoint_name: Optional checkpoint name
        
    Returns:
        Checkpoint identifier
    """
    try:
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Validate state integrity before checkpointing
        is_valid, errors = validate_state_integrity(state)
        if not is_valid:
            logger.warning(f"State integrity issues before checkpoint: {errors}")
        
        # Create checkpoint data
        checkpoint_data = {
            "checkpoint_name": checkpoint_name,
            "timestamp": datetime.now().isoformat(),
            "state": state,
            "state_summary": extract_state_summary(state),
            "integrity_errors": errors if not is_valid else []
        }
        
        # Serialize checkpoint
        serialized_checkpoint = serialize_state(checkpoint_data, format_type="json")
        
        logger.info(f"Created checkpoint: {checkpoint_name}")
        return serialized_checkpoint
        
    except Exception as e:
        logger.error(f"Checkpoint creation error: {e}")
        raise


def restore_state_from_checkpoint(checkpoint_data: str) -> Tuple[AgentState, Dict[str, Any]]:
    """
    Restore state from a checkpoint.
    
    Args:
        checkpoint_data: Serialized checkpoint data
        
    Returns:
        Tuple of (restored_state, checkpoint_info)
    """
    try:
        # Deserialize checkpoint data
        checkpoint = deserialize_state(checkpoint_data, format_type="json")
        
        # Extract state and info
        restored_state = checkpoint["state"]
        checkpoint_info = {
            "checkpoint_name": checkpoint["checkpoint_name"],
            "timestamp": checkpoint["timestamp"],
            "state_summary": checkpoint["state_summary"],
            "integrity_errors": checkpoint["integrity_errors"]
        }
        
        # Validate restored state
        is_valid, errors = validate_state_integrity(restored_state)
        if not is_valid:
            logger.warning(f"Restored state has integrity issues: {errors}")
            checkpoint_info["restoration_warnings"] = errors
        
        logger.info(f"Restored state from checkpoint: {checkpoint_info['checkpoint_name']}")
        return restored_state, checkpoint_info
        
    except Exception as e:
        logger.error(f"State restoration error: {e}")
        raise


def get_node_execution_stats(node_results: List[Any]) -> Dict[str, Any]:
    """
    Get execution statistics from node results.
    
    Args:
        node_results: List of node execution results
        
    Returns:
        Execution statistics dictionary
    """
    try:
        if not node_results:
            return {
                "total_nodes": 0,
                "successful_nodes": 0,
                "failed_nodes": 0,
                "total_execution_time": 0.0,
                "average_execution_time": 0.0,
                "node_types": []
            }
        
        total_nodes = len(node_results)
        successful_nodes = sum(1 for result in node_results if getattr(result, 'success', True))
        failed_nodes = total_nodes - successful_nodes
        
        execution_times = []
        node_types = []
        
        for result in node_results:
            if hasattr(result, 'execution_time'):
                execution_times.append(result.execution_time)
            if hasattr(result, 'node_type'):
                node_types.append(str(result.node_type))
        
        total_execution_time = sum(execution_times)
        average_execution_time = total_execution_time / len(execution_times) if execution_times else 0.0
        
        return {
            "total_nodes": total_nodes,
            "successful_nodes": successful_nodes,
            "failed_nodes": failed_nodes,
            "total_execution_time": total_execution_time,
            "average_execution_time": average_execution_time,
            "node_types": list(set(node_types)),
            "success_rate": successful_nodes / total_nodes if total_nodes > 0 else 0.0
        }
        
    except Exception as e:
        logger.error(f"Node execution stats error: {e}")
        return {"error": str(e)}


def format_execution_report(result: Dict[str, Any]) -> str:
    """
    Format execution result into a human-readable report.
    
    Args:
        result: Execution result dictionary
        
    Returns:
        Formatted report string
    """
    try:
        report_lines = []
        
        # Header
        report_lines.append("=" * 60)
        report_lines.append("NL-to-SQL Pipeline Execution Report")
        report_lines.append("=" * 60)
        
        # Basic info
        report_lines.append(f"Session ID: {result.get('session_id', 'N/A')}")
        report_lines.append(f"Success: {'✅ Yes' if result.get('success') else '❌ No'}")
        report_lines.append(f"Execution Time: {result.get('execution_time', 0):.2f}s")
        
        # Query info
        user_query = result.get('user_query', '')
        report_lines.append(f"User Query: {user_query[:100]}{'...' if len(user_query) > 100 else ''}")
        
        # SQL info
        sql_query = result.get('final_sql', '')
        if sql_query:
            report_lines.append(f"Generated SQL: {sql_query[:100]}{'...' if len(sql_query) > 100 else ''}")
        
        # Results info
        query_result = result.get('query_result', [])
        if query_result:
            report_lines.append(f"Result Count: {len(query_result)} rows")
        
        # Summary
        summary = result.get('data_summary', '')
        if summary:
            report_lines.append(f"Summary: {summary[:200]}{'...' if len(summary) > 200 else ''}")
        
        # Error info
        error_message = result.get('error_message', '')
        if error_message:
            report_lines.append(f"Error: {error_message}")
        
        # Confidence scores
        confidence_scores = result.get('confidence_scores', {})
        if confidence_scores:
            report_lines.append("\nConfidence Scores:")
            for score_name, score_value in confidence_scores.items():
                report_lines.append(f"  {score_name}: {score_value:.2f}")
        
        # Node execution stats
        node_results = result.get('node_results', [])
        if node_results:
            stats = get_node_execution_stats(node_results)
            report_lines.append(f"\nNode Execution Stats:")
            report_lines.append(f"  Total Nodes: {stats.get('total_nodes', 0)}")
            report_lines.append(f"  Success Rate: {stats.get('success_rate', 0):.1%}")
            report_lines.append(f"  Average Time: {stats.get('average_execution_time', 0):.2f}s")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
        
    except Exception as e:
        logger.error(f"Report formatting error: {e}")
        return f"Error formatting report: {str(e)}"
