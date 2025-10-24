"""
Base Handler for Slack Events

This module provides the base class for all Slack event handlers.
"""

import re
from typing import Dict, Any, Optional, Callable, Awaitable
from abc import ABC, abstractmethod
from slack_bolt import App
from core.logging import get_logger

logger = get_logger(__name__)


class BaseSlackHandler(ABC):
    """Base class for all Slack event handlers."""
    
    def __init__(self, app: App, agent_runner=None):
        """
        Initialize the base handler.
        
        Args:
            app: Slack Bolt App instance
            agent_runner: Agent runner for processing queries
        """
        self.app = app
        self.agent_runner = agent_runner
        self._register_handlers()
    
    @abstractmethod
    def _register_handlers(self):
        """Register event handlers with the Slack app."""
        pass
    
    def _extract_query_from_mention(self, text: str) -> str:
        """
        Extract query text from a mention message.
        
        Args:
            text: Message text containing mention
            
        Returns:
            Cleaned query text without mention
        """
        # Remove bot mention patterns like <@U1234567890>
        cleaned_text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
        
        # Remove extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        return cleaned_text
    
    def _is_direct_message(self, channel_type: str) -> bool:
        """
        Check if the message is from a direct message channel.
        
        Args:
            channel_type: Slack channel type
            
        Returns:
            True if it's a direct message
        """
        return channel_type == "im"
    
    def _is_bot_message(self, message: Dict[str, Any]) -> bool:
        """
        Check if the message is from a bot to avoid infinite loops.
        
        Args:
            message: Slack message event
            
        Returns:
            True if the message is from a bot
        """
        return message.get("subtype") == "bot_message" or "bot_id" in message
    
    def _get_thread_timestamp(self, message: Dict[str, Any]) -> str:
        """
        Get the appropriate thread timestamp for responses.
        
        Args:
            message: Slack message event
            
        Returns:
            Thread timestamp for responses
        """
        return message.get("thread_ts", message.get("ts"))
    
    def _format_error_message(self, error: Exception, show_details: bool = False) -> str:
        """
        Format error message for user display.
        
        Args:
            error: Exception that occurred
            show_details: Whether to show detailed error information
            
        Returns:
            Formatted error message
        """
        if show_details:
            return f"❌ 오류가 발생했습니다: {str(error)}"
        else:
            return "❌ 처리 중 오류가 발생했습니다. 나중에 다시 시도해주세요."
    
    def _log_event(self, event_type: str, event_data: Dict[str, Any]):
        """
        Log event information for debugging.
        
        Args:
            event_type: Type of event
            event_data: Event data dictionary
        """
        logger.info(
            f"Slack {event_type} event received",
            event_type=event_type,
            channel=event_data.get("channel"),
            user=event_data.get("user"),
            text_length=len(event_data.get("text", ""))
        )








