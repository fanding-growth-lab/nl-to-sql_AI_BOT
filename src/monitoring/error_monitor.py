"""
오류 모니터링 및 알림 시스템
실시간으로 오류를 감지하고 알림을 보내는 시스템
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

import logging

logger = logging.getLogger(__name__)


@dataclass
class ErrorEvent:
    """오류 이벤트 정보"""
    timestamp: datetime
    error_type: str
    error_message: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    component: str  # 'slack', 'database', 'sql_generator', etc.
    user_id: Optional[str] = None
    channel_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """알림 규칙"""
    name: str
    condition: Callable[[List[ErrorEvent]], bool]
    severity_threshold: str
    time_window_minutes: int
    max_occurrences: int
    cooldown_minutes: int = 30
    last_triggered: Optional[datetime] = None


class ErrorMonitor:
    """오류 모니터링 시스템"""
    
    def __init__(self, max_events: int = 10000):
        """
        오류 모니터 초기화
        
        Args:
            max_events: 최대 저장할 오류 이벤트 수
        """
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self.alert_rules: List[AlertRule] = []
        self.alert_callbacks: List[Callable[[AlertRule, List[ErrorEvent]], None]] = []
        self.stats = defaultdict(int)
        self.component_stats = defaultdict(lambda: defaultdict(int))
        
        # Initialize default alert rules
        self._setup_default_alert_rules()
        
        logger.info("오류 모니터링 시스템 초기화 완료")
    
    def _setup_default_alert_rules(self):
        """기본 알림 규칙 설정"""
        
        # Critical errors (immediate alert)
        self.add_alert_rule(
            name="critical_errors",
            condition=lambda events: len([e for e in events if e.severity == 'critical']) > 0,
            severity_threshold="critical",
            time_window_minutes=1,
            max_occurrences=1,
            cooldown_minutes=0
        )
        
        # High severity error spike
        self.add_alert_rule(
            name="high_severity_spike",
            condition=lambda events: len([e for e in events if e.severity == 'high']) >= 5,
            severity_threshold="high",
            time_window_minutes=5,
            max_occurrences=5,
            cooldown_minutes=15
        )
        
        # Database connection failures
        self.add_alert_rule(
            name="database_failures",
            condition=lambda events: len([e for e in events 
                                        if e.component == 'database' and 'connection' in e.error_message.lower()]) >= 3,
            severity_threshold="high",
            time_window_minutes=10,
            max_occurrences=3,
            cooldown_minutes=20
        )
        
        # Slack API failures
        self.add_alert_rule(
            name="slack_api_failures",
            condition=lambda events: len([e for e in events 
                                        if e.component == 'slack' and 'api' in e.error_message.lower()]) >= 5,
            severity_threshold="medium",
            time_window_minutes=15,
            max_occurrences=5,
            cooldown_minutes=30
        )
        
        # SQL generation failures
        self.add_alert_rule(
            name="sql_generation_failures",
            condition=lambda events: len([e for e in events 
                                        if e.component == 'sql_generator']) >= 10,
            severity_threshold="medium",
            time_window_minutes=30,
            max_occurrences=10,
            cooldown_minutes=60
        )
        
        logger.info("기본 알림 규칙 설정 완료")
    
    def record_error(self, error_event: ErrorEvent):
        """
        오류 이벤트 기록
        
        Args:
            error_event: 기록할 오류 이벤트
        """
        self.events.append(error_event)
        self.stats[error_event.severity] += 1
        self.component_stats[error_event.component][error_event.severity] += 1
        
        logger.warning(
            f"오류 이벤트 기록: {error_event.component} - {error_event.error_type}",
            severity=error_event.severity,
            error_type=error_event.error_type,
            component=error_event.component
        )
        
        # Check alert rules
        self._check_alert_rules()
    
    def add_alert_rule(self, name: str, condition: Callable[[List[ErrorEvent]], bool],
                      severity_threshold: str, time_window_minutes: int, 
                      max_occurrences: int, cooldown_minutes: int = 30):
        """
        알림 규칙 추가
        
        Args:
            name: 규칙 이름
            condition: 알림 조건 함수
            severity_threshold: 심각도 임계값
            time_window_minutes: 시간 윈도우 (분)
            max_occurrences: 최대 발생 횟수
            cooldown_minutes: 쿨다운 시간 (분)
        """
        rule = AlertRule(
            name=name,
            condition=condition,
            severity_threshold=severity_threshold,
            time_window_minutes=time_window_minutes,
            max_occurrences=max_occurrences,
            cooldown_minutes=cooldown_minutes
        )
        
        self.alert_rules.append(rule)
        logger.info(f"알림 규칙 추가: {name}")
    
    def add_alert_callback(self, callback: Callable[[AlertRule, List[ErrorEvent]], None]):
        """
        알림 콜백 함수 추가
        
        Args:
            callback: 알림 발생 시 호출될 함수
        """
        self.alert_callbacks.append(callback)
        logger.info("알림 콜백 함수 추가")
    
    def _check_alert_rules(self):
        """알림 규칙 확인"""
        current_time = datetime.now()
        
        for rule in self.alert_rules:
            # Check cooldown
            if (rule.last_triggered and 
                current_time - rule.last_triggered < timedelta(minutes=rule.cooldown_minutes)):
                continue
            
            # Get events within time window
            cutoff_time = current_time - timedelta(minutes=rule.time_window_minutes)
            recent_events = [e for e in self.events if e.timestamp >= cutoff_time]
            
            # Check if condition is met
            if rule.condition(recent_events):
                self._trigger_alert(rule, recent_events)
                rule.last_triggered = current_time
    
    def _trigger_alert(self, rule: AlertRule, events: List[ErrorEvent]):
        """알림 트리거"""
        logger.error(
            f"알림 트리거: {rule.name}",
            rule_name=rule.name,
            severity_threshold=rule.severity_threshold,
            event_count=len(events)
        )
        
        # Call all registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(rule, events)
            except Exception as e:
                logger.error(f"알림 콜백 실행 중 오류: {str(e)}")
    
    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        오류 통계 조회
        
        Args:
            hours: 조회할 시간 범위 (시간)
            
        Returns:
            오류 통계 정보
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_events = [e for e in self.events if e.timestamp >= cutoff_time]
        
        # Severity distribution
        severity_dist = defaultdict(int)
        for event in recent_events:
            severity_dist[event.severity] += 1
        
        # Component distribution
        component_dist = defaultdict(lambda: defaultdict(int))
        for event in recent_events:
            component_dist[event.component][event.severity] += 1
        
        # Error rate over time (hourly)
        hourly_rates = defaultdict(int)
        for event in recent_events:
            hour_key = event.timestamp.strftime("%Y-%m-%d %H:00")
            hourly_rates[hour_key] += 1
        
        return {
            "time_range_hours": hours,
            "total_events": len(recent_events),
            "severity_distribution": dict(severity_dist),
            "component_distribution": {k: dict(v) for k, v in component_dist.items()},
            "hourly_rates": dict(hourly_rates),
            "most_common_errors": self._get_most_common_errors(recent_events),
            "recent_critical_events": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "component": e.component,
                    "error_type": e.error_type,
                    "message": e.error_message[:100]
                }
                for e in recent_events if e.severity == 'critical'
            ][-10:]  # Last 10 critical events
        }
    
    def _get_most_common_errors(self, events: List[ErrorEvent], top_n: int = 10) -> List[Dict[str, Any]]:
        """가장 많이 발생한 오류 조회"""
        error_counts = defaultdict(int)
        error_examples = {}
        
        for event in events:
            error_key = f"{event.component}:{event.error_type}"
            error_counts[error_key] += 1
            if error_key not in error_examples:
                error_examples[error_key] = event.error_message
        
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                "error_key": error_key,
                "count": count,
                "example_message": error_examples[error_key],
                "component": error_key.split(':')[0],
                "error_type": error_key.split(':')[1]
            }
            for error_key, count in sorted_errors[:top_n]
        ]
    
    def get_health_status(self) -> Dict[str, Any]:
        """시스템 건강 상태 조회"""
        current_time = datetime.now()
        
        # Recent events (last hour)
        recent_events = [e for e in self.events 
                        if e.timestamp >= current_time - timedelta(hours=1)]
        
        # Critical events (last 24 hours)
        critical_events = [e for e in self.events 
                          if e.timestamp >= current_time - timedelta(hours=24) and e.severity == 'critical']
        
        # Determine overall health status
        if critical_events:
            health_status = "critical"
            health_message = f"{len(critical_events)}개의 치명적 오류가 지난 24시간 내에 발생했습니다"
        elif len([e for e in recent_events if e.severity == 'high']) >= 5:
            health_status = "warning"
            health_message = "높은 심각도의 오류가 최근에 많이 발생하고 있습니다"
        elif len(recent_events) >= 20:
            health_status = "warning"
            health_message = "오류 발생 빈도가 높습니다"
        else:
            health_status = "healthy"
            health_message = "시스템이 정상적으로 작동하고 있습니다"
        
        return {
            "status": health_status,
            "message": health_message,
            "last_updated": current_time.isoformat(),
            "recent_events_count": len(recent_events),
            "critical_events_24h": len(critical_events),
            "components_status": self._get_components_status()
        }
    
    def _get_components_status(self) -> Dict[str, str]:
        """컴포넌트별 상태 조회"""
        current_time = datetime.now()
        component_status = {}
        
        for component in ['slack', 'database', 'sql_generator', 'nl_processor']:
            recent_errors = [e for e in self.events 
                           if e.component == component and 
                           e.timestamp >= current_time - timedelta(hours=1)]
            
            if not recent_errors:
                component_status[component] = "healthy"
            elif len([e for e in recent_errors if e.severity in ['high', 'critical']]) > 0:
                component_status[component] = "critical"
            elif len(recent_errors) >= 5:
                component_status[component] = "warning"
            else:
                component_status[component] = "degraded"
        
        return component_status


# Global error monitor instance
_error_monitor: Optional[ErrorMonitor] = None


def get_error_monitor() -> ErrorMonitor:
    """전역 오류 모니터 인스턴스 반환"""
    global _error_monitor
    if _error_monitor is None:
        _error_monitor = ErrorMonitor()
    return _error_monitor


def record_error(error_type: str, error_message: str, severity: str = "medium",
                component: str = "unknown", user_id: Optional[str] = None,
                channel_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
    """
    오류 이벤트 기록 헬퍼 함수
    
    Args:
        error_type: 오류 타입
        error_message: 오류 메시지
        severity: 심각도 ('low', 'medium', 'high', 'critical')
        component: 컴포넌트명
        user_id: 사용자 ID (선택사항)
        channel_id: 채널 ID (선택사항)
        context: 추가 컨텍스트 정보 (선택사항)
    """
    error_event = ErrorEvent(
        timestamp=datetime.now(),
        error_type=error_type,
        error_message=error_message,
        severity=severity,
        component=component,
        user_id=user_id,
        channel_id=channel_id,
        context=context or {}
    )
    
    monitor = get_error_monitor()
    monitor.record_error(error_event)


# Convenience functions for common error types
def record_slack_error(error_message: str, severity: str = "medium", 
                      user_id: Optional[str] = None, channel_id: Optional[str] = None):
    """Slack 관련 오류 기록"""
    record_error("slack_error", error_message, severity, "slack", user_id, channel_id)


def record_database_error(error_message: str, severity: str = "high"):
    """데이터베이스 관련 오류 기록"""
    record_error("database_error", error_message, severity, "database")


def record_sql_generation_error(error_message: str, severity: str = "medium",
                               user_id: Optional[str] = None, channel_id: Optional[str] = None):
    """SQL 생성 관련 오류 기록"""
    record_error("sql_generation_error", error_message, severity, "sql_generator", user_id, channel_id)


def record_nl_processing_error(error_message: str, severity: str = "medium",
                              user_id: Optional[str] = None, channel_id: Optional[str] = None):
    """자연어 처리 관련 오류 기록"""
    record_error("nl_processing_error", error_message, severity, "nl_processor", user_id, channel_id)
