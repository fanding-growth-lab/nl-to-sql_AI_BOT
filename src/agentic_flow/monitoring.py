"""
Performance Monitoring and Metrics Collection for NL-to-SQL Pipeline

This module provides monitoring capabilities for the LangGraph pipeline.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import threading
from collections import defaultdict, deque

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineMetrics:
    """Metrics collected during pipeline execution."""
    start_time: float
    end_time: Optional[float] = None
    processing_time: float = 0.0
    
    # Component timings
    nl_processing_time: float = 0.0
    schema_mapping_time: float = 0.0
    sql_generation_time: float = 0.0
    sql_validation_time: float = 0.0
    
    # Success/failure metrics
    success: bool = False
    retry_count: int = 0
    error_message: Optional[str] = None
    
    # Quality metrics
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    final_confidence: float = 0.0
    complexity_level: str = "UNKNOWN"
    
    # Input/Output metrics
    query_length: int = 0
    entities_extracted: int = 0
    tables_mapped: int = 0
    sql_length: int = 0
    
    # User context
    user_id: Optional[str] = None
    channel_id: Optional[str] = None


class MetricsCollector:
    """Collects and aggregates pipeline metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.component_times: Dict[str, List[float]] = defaultdict(list)
        self.success_rates: Dict[str, float] = defaultdict(float)
        self.confidence_scores: Dict[str, List[float]] = defaultdict(list)
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
    
    def start_metrics(self, user_id: Optional[str] = None, 
                     channel_id: Optional[str] = None) -> PipelineMetrics:
        """Start collecting metrics for a new pipeline execution."""
        return PipelineMetrics(
            start_time=time.time(),
            user_id=user_id,
            channel_id=channel_id
        )
    
    def update_component_time(self, metrics: PipelineMetrics, 
                            component: str, duration: float):
        """Update component-specific timing."""
        setattr(metrics, f"{component}_time", duration)
        
        with self._lock:
            self.component_times[component].append(duration)
            if len(self.component_times[component]) > self.max_history:
                self.component_times[component].pop(0)
    
    def finish_metrics(self, metrics: PipelineMetrics, 
                      state: Dict[str, Any]):
        """Finish collecting metrics and update aggregations."""
        metrics.end_time = time.time()
        metrics.processing_time = metrics.end_time - metrics.start_time
        
        # Extract metrics from state
        metrics.success = state.get("success", False)
        metrics.retry_count = state.get("retry_count", 0)
        metrics.error_message = state.get("error_message")
        
        metrics.confidence_scores = state.get("confidence_scores", {})
        metrics.final_confidence = max(metrics.confidence_scores.values()) if metrics.confidence_scores else 0.0
        
        if state.get("sql_result"):
            metrics.complexity_level = state["sql_result"].complexity.value
        
        metrics.query_length = len(state.get("user_query", ""))
        metrics.entities_extracted = len(state.get("entities", []))
        
        # Handle SchemaMapping object properly
        schema_mapping = state.get("schema_mapping")
        if schema_mapping:
            metrics.tables_mapped = len(schema_mapping.relevant_tables)
        else:
            metrics.tables_mapped = 0
            
        metrics.sql_length = len(state.get("final_sql", ""))
        
        # Update aggregations
        with self._lock:
            self.metrics_history.append(metrics)
            
            # Update success rates
            component = "overall"
            success_count = sum(1 for m in self.metrics_history if m.success)
            self.success_rates[component] = success_count / len(self.metrics_history)
            
            # Update confidence scores
            if metrics.final_confidence > 0:
                self.confidence_scores[component].append(metrics.final_confidence)
                if len(self.confidence_scores[component]) > self.max_history:
                    self.confidence_scores[component].pop(0)
            
            # Update error patterns
            if metrics.error_message:
                error_type = metrics.error_message.split(":")[0] if ":" in metrics.error_message else "Unknown"
                self.error_patterns[error_type] += 1
        
        logger.info(
            f"Pipeline completed - Success: {metrics.success}, "
            f"Time: {metrics.processing_time:.2f}s, "
            f"Confidence: {metrics.final_confidence:.2f}, "
            f"Retries: {metrics.retry_count}"
        )
        
        return metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        with self._lock:
            if not self.metrics_history:
                return {"message": "No metrics available"}
            
            recent_metrics = list(self.metrics_history)[-100:]  # Last 100 executions
            
            # Calculate averages
            avg_processing_time = sum(m.processing_time for m in recent_metrics) / len(recent_metrics)
            avg_confidence = sum(m.final_confidence for m in recent_metrics if m.final_confidence > 0) / max(1, len([m for m in recent_metrics if m.final_confidence > 0]))
            
            # Calculate component averages
            component_averages = {}
            for component, times in self.component_times.items():
                if times:
                    component_averages[component] = {
                        "avg_time": sum(times) / len(times),
                        "min_time": min(times),
                        "max_time": max(times),
                        "count": len(times)
                    }
            
            return {
                "total_executions": len(self.metrics_history),
                "recent_executions": len(recent_metrics),
                "success_rate": self.success_rates.get("overall", 0.0),
                "avg_processing_time": avg_processing_time,
                "avg_confidence": avg_confidence,
                "component_performance": component_averages,
                "common_errors": dict(sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]),
                "last_updated": datetime.now().isoformat()
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get pipeline health status."""
        summary = self.get_performance_summary()
        
        # Determine health status
        health_score = 0.0
        health_issues = []
        
        if summary.get("success_rate", 0) >= 0.9:
            health_score += 40
        elif summary.get("success_rate", 0) >= 0.7:
            health_score += 20
            health_issues.append("Success rate below 90%")
        else:
            health_issues.append("Low success rate")
        
        if summary.get("avg_processing_time", 0) <= 20:
            health_score += 30
        elif summary.get("avg_processing_time", 0) <= 30:
            health_score += 15
            health_issues.append("Processing time above 20s")
        else:
            health_issues.append("High processing time")
        
        if summary.get("avg_confidence", 0) >= 0.8:
            health_score += 30
        elif summary.get("avg_confidence", 0) >= 0.6:
            health_score += 15
            health_issues.append("Low confidence scores")
        else:
            health_issues.append("Very low confidence scores")
        
        # Determine overall health
        if health_score >= 90:
            health_status = "excellent"
        elif health_score >= 70:
            health_status = "good"
        elif health_score >= 50:
            health_status = "fair"
        else:
            health_status = "poor"
        
        return {
            "health_status": health_status,
            "health_score": health_score,
            "health_issues": health_issues,
            "last_check": datetime.now().isoformat(),
            **summary
        }


# Global metrics collector instance
metrics_collector = MetricsCollector()


class PipelineMonitor:
    """Context manager for monitoring pipeline execution."""
    
    def __init__(self, user_id: Optional[str] = None, 
                 channel_id: Optional[str] = None):
        self.metrics = None
        self.user_id = user_id
        self.channel_id = channel_id
        self.component_start_times = {}
    
    def __enter__(self):
        self.metrics = metrics_collector.start_metrics(
            self.user_id, self.channel_id
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.metrics:
            # This will be called by the pipeline when it finishes
            pass
    
    def start_component(self, component: str):
        """Start timing a component."""
        self.component_start_times[component] = time.time()
    
    def end_component(self, component: str):
        """End timing a component."""
        if component in self.component_start_times and self.metrics:
            duration = time.time() - self.component_start_times[component]
            metrics_collector.update_component_time(self.metrics, component, duration)
    
    def start_execution(self, session_id: str) -> None:
        """Start execution tracking for a session."""
        # Implementation for tracking execution start
        pass
    
    def end_execution(self, session_id: str, success: bool = True) -> None:
        """End execution tracking for a session."""
        # Implementation for tracking execution end
        pass
    
    def finish(self, state: Dict[str, Any]) -> PipelineMetrics:
        """Finish monitoring and return final metrics."""
        if self.metrics:
            return metrics_collector.finish_metrics(self.metrics, state)
        return None


def get_metrics_summary() -> Dict[str, Any]:
    """Get current metrics summary."""
    return metrics_collector.get_performance_summary()


def get_health_status() -> Dict[str, Any]:
    """Get pipeline health status."""
    return metrics_collector.get_health_status()
