"""
알림 시스템
오류 모니터링 시스템과 연동하여 알림을 전송하는 시스템
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import logging
from src.monitoring.error_monitor import AlertRule, ErrorEvent

logger = logging.getLogger(__name__)


@dataclass
class AlertChannel:
    """알림 채널 정보"""
    name: str
    channel_type: str  # 'slack', 'email', 'webhook', 'log'
    config: Dict[str, Any]


class AlertManager:
    """알림 관리자"""
    
    def __init__(self):
        """알림 관리자 초기화"""
        self.channels: List[AlertChannel] = []
        self.alert_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        
        logger.info("알림 관리자 초기화 완료")
    
    def add_channel(self, name: str, channel_type: str, config: Dict[str, Any]):
        """
        알림 채널 추가
        
        Args:
            name: 채널 이름
            channel_type: 채널 타입
            config: 채널 설정
        """
        channel = AlertChannel(name=name, channel_type=channel_type, config=config)
        self.channels.append(channel)
        logger.info(f"알림 채널 추가: {name} ({channel_type})")
    
    async def send_alert(self, rule: AlertRule, events: List[ErrorEvent], 
                        severity_override: Optional[str] = None):
        """
        알림 전송
        
        Args:
            rule: 트리거된 알림 규칙
            events: 관련 오류 이벤트들
            severity_override: 심각도 오버라이드
        """
        try:
            # Create alert message
            alert_message = self._create_alert_message(rule, events, severity_override)
            
            # Determine which channels to use based on severity
            severity = severity_override or rule.severity_threshold
            target_channels = self._get_channels_for_severity(severity)
            
            # Send to all target channels
            for channel in target_channels:
                try:
                    await self._send_to_channel(channel, alert_message, rule, events)
                except Exception as e:
                    logger.error(f"채널 {channel.name}으로 알림 전송 실패: {str(e)}")
            
            # Record in alert history
            self._record_alert_history(rule, events, alert_message)
            
        except Exception as e:
            logger.error(f"알림 전송 중 오류 발생: {str(e)}")
    
    def _create_alert_message(self, rule: AlertRule, events: List[ErrorEvent], 
                            severity_override: Optional[str]) -> Dict[str, Any]:
        """알림 메시지 생성"""
        severity = severity_override or rule.severity_threshold
        current_time = datetime.now()
        
        # Group events by component
        component_events = {}
        for event in events:
            if event.component not in component_events:
                component_events[event.component] = []
            component_events[event.component].append(event)
        
        # Create summary
        total_events = len(events)
        critical_events = len([e for e in events if e.severity == 'critical'])
        high_events = len([e for e in events if e.severity == 'high'])
        
        # Determine alert emoji based on severity
        severity_emoji = {
            'critical': '🚨',
            'high': '⚠️',
            'medium': '🔶',
            'low': 'ℹ️'
        }.get(severity, '📢')
        
        # Create message
        message = {
            "title": f"{severity_emoji} DataTalk Bot 알림: {rule.name}",
            "timestamp": current_time.isoformat(),
            "rule_name": rule.name,
            "severity": severity,
            "summary": {
                "total_events": total_events,
                "critical_events": critical_events,
                "high_events": high_events,
                "time_window_minutes": rule.time_window_minutes,
                "threshold_exceeded": total_events >= rule.max_occurrences
            },
            "components_affected": list(component_events.keys()),
            "component_details": {}
        }
        
        # Add component-specific details
        for component, comp_events in component_events.items():
            error_types = {}
            for event in comp_events:
                if event.error_type not in error_types:
                    error_types[event.error_type] = []
                error_types[event.error_type].append(event.error_message)
            
            message["component_details"][component] = {
                "event_count": len(comp_events),
                "error_types": error_types,
                "latest_event": max(comp_events, key=lambda x: x.timestamp).error_message
            }
        
        # Add recent critical events
        recent_critical = [e for e in events if e.severity == 'critical'][-5:]
        if recent_critical:
            message["recent_critical_events"] = [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "component": e.component,
                    "error_type": e.error_type,
                    "message": e.error_message[:200]
                }
                for e in recent_critical
            ]
        
        return message
    
    def _get_channels_for_severity(self, severity: str) -> List[AlertChannel]:
        """심각도에 따른 알림 채널 선택"""
        # For now, use all channels regardless of severity
        # In production, you might want to configure different channels for different severities
        return self.channels
    
    async def _send_to_channel(self, channel: AlertChannel, message: Dict[str, Any],
                              rule: AlertRule, events: List[ErrorEvent]):
        """특정 채널로 알림 전송"""
        
        if channel.channel_type == 'slack':
            await self._send_slack_alert(channel, message, rule, events)
        elif channel.channel_type == 'email':
            await self._send_email_alert(channel, message, rule, events)
        elif channel.channel_type == 'webhook':
            await self._send_webhook_alert(channel, message, rule, events)
        elif channel.channel_type == 'log':
            await self._send_log_alert(channel, message, rule, events)
        else:
            logger.warning(f"지원하지 않는 채널 타입: {channel.channel_type}")
    
    async def _send_slack_alert(self, channel: AlertChannel, message: Dict[str, Any],
                               rule: AlertRule, events: List[ErrorEvent]):
        """Slack으로 알림 전송"""
        try:
            # Format message for Slack
            slack_message = self._format_slack_message(message)
            
            # In a real implementation, you would use Slack API here
            # For now, we'll just log the message
            logger.error(f"Slack 알림 (채널: {channel.name}): {slack_message}")
            
            # If you have Slack client available, you could do:
            # slack_client = channel.config.get('slack_client')
            # if slack_client:
            #     await slack_client.chat_postMessage(
            #         channel=channel.config.get('channel_id'),
            #         text=slack_message
            #     )
            
        except Exception as e:
            logger.error(f"Slack 알림 전송 실패: {str(e)}")
    
    def _format_slack_message(self, message: Dict[str, Any]) -> str:
        """Slack용 메시지 포맷팅"""
        severity_emoji = {
            'critical': '🚨',
            'high': '⚠️',
            'medium': '🔶',
            'low': 'ℹ️'
        }.get(message['severity'], '📢')
        
        slack_text = f"{severity_emoji} *DataTalk Bot 알림*\n\n"
        slack_text += f"*규칙:* {message['rule_name']}\n"
        slack_text += f"*심각도:* {message['severity'].upper()}\n"
        slack_text += f"*시간:* {message['timestamp']}\n\n"
        
        summary = message['summary']
        slack_text += f"*요약:*\n"
        slack_text += f"• 총 이벤트: {summary['total_events']}\n"
        slack_text += f"• 치명적 오류: {summary['critical_events']}\n"
        slack_text += f"• 높은 심각도: {summary['high_events']}\n"
        slack_text += f"• 시간 윈도우: {summary['time_window_minutes']}분\n\n"
        
        slack_text += f"*영향받은 컴포넌트:* {', '.join(message['components_affected'])}\n\n"
        
        # Add component details
        for component, details in message['component_details'].items():
            slack_text += f"*{component}:*\n"
            slack_text += f"• 이벤트 수: {details['event_count']}\n"
            slack_text += f"• 최근 오류: {details['latest_event'][:100]}...\n\n"
        
        # Add recent critical events if any
        if 'recent_critical_events' in message:
            slack_text += "*최근 치명적 오류:*\n"
            for event in message['recent_critical_events'][:3]:  # Show only first 3
                slack_text += f"• {event['component']}: {event['message'][:80]}...\n"
        
        return slack_text
    
    async def _send_email_alert(self, channel: AlertChannel, message: Dict[str, Any],
                               rule: AlertRule, events: List[ErrorEvent]):
        """이메일로 알림 전송"""
        try:
            # Format message for email
            email_subject = f"DataTalk Bot 알림: {rule.name} ({message['severity'].upper()})"
            email_body = json.dumps(message, indent=2, ensure_ascii=False)
            
            # In a real implementation, you would use email library here
            logger.error(f"이메일 알림 (수신자: {channel.config.get('recipients', 'N/A')}):\n제목: {email_subject}\n내용: {email_body}")
            
        except Exception as e:
            logger.error(f"이메일 알림 전송 실패: {str(e)}")
    
    async def _send_webhook_alert(self, channel: AlertChannel, message: Dict[str, Any],
                                 rule: AlertRule, events: List[ErrorEvent]):
        """웹훅으로 알림 전송"""
        try:
            import aiohttp
            
            webhook_url = channel.config.get('url')
            if not webhook_url:
                logger.warning(f"웹훅 URL이 설정되지 않음: {channel.name}")
                return
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=message,
                    headers={'Content-Type': 'application/json'},
                    timeout=10
                ) as response:
                    if response.status == 200:
                        logger.info(f"웹훅 알림 전송 성공: {channel.name}")
                    else:
                        logger.error(f"웹훅 알림 전송 실패: {response.status} - {channel.name}")
            
        except Exception as e:
            logger.error(f"웹훅 알림 전송 실패: {str(e)}")
    
    async def _send_log_alert(self, channel: AlertChannel, message: Dict[str, Any],
                             rule: AlertRule, events: List[ErrorEvent]):
        """로그로 알림 기록"""
        try:
            log_level = {
                'critical': 'CRITICAL',
                'high': 'ERROR',
                'medium': 'WARNING',
                'low': 'INFO'
            }.get(message['severity'], 'WARNING')
            
            log_message = f"알림 [{rule.name}] {message['severity'].upper()}: {message['summary']['total_events']}개 이벤트"
            
            if log_level == 'CRITICAL':
                logger.critical(log_message)
            elif log_level == 'ERROR':
                logger.error(log_message)
            elif log_level == 'WARNING':
                logger.warning(log_message)
            else:
                logger.info(log_message)
                
        except Exception as e:
            logger.error(f"로그 알림 기록 실패: {str(e)}")
    
    def _record_alert_history(self, rule: AlertRule, events: List[ErrorEvent], 
                            alert_message: Dict[str, Any]):
        """알림 히스토리 기록"""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "rule_name": rule.name,
            "severity": rule.severity_threshold,
            "event_count": len(events),
            "components_affected": list(alert_message['components_affected']),
            "message": alert_message
        }
        
        self.alert_history.append(history_entry)
        
        # Keep only recent history
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """알림 히스토리 조회"""
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            entry for entry in self.alert_history
            if datetime.fromisoformat(entry['timestamp']) >= cutoff_time
        ]


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """전역 알림 관리자 인스턴스 반환"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


async def setup_default_alerts():
    """기본 알림 설정"""
    alert_manager = get_alert_manager()
    
    # Add log channel (always available)
    alert_manager.add_channel(
        name="log_alerts",
        channel_type="log",
        config={}
    )
    
    # Add Slack channel if configured
    # In a real implementation, you would check for Slack configuration
    # and add the channel if available
    
    logger.info("기본 알림 채널 설정 완료")


# Alert callback for error monitor
async def alert_callback(rule: AlertRule, events: List[ErrorEvent]):
    """오류 모니터의 알림 콜백"""
    alert_manager = get_alert_manager()
    await alert_manager.send_alert(rule, events)
