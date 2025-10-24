"""
Error Handler for Slack Events

This module provides comprehensive error handling for Slack events.
"""

import traceback
from typing import Dict, Any, Optional, Union
from slack_bolt import App
from slack_bolt.error import BoltError
from core.logging import get_logger

logger = get_logger(__name__)


class SlackErrorHandler:
    """Comprehensive error handler for Slack events."""
    
    def __init__(self, app: App, show_error_details: bool = False):
        """
        Initialize the error handler.
        
        Args:
            app: Slack Bolt App instance
            show_error_details: Whether to show detailed error information to users
        """
        self.app = app
        self.show_error_details = show_error_details
        self._register_error_handlers()
    
    def _register_error_handlers(self):
        """Register global error handlers."""
        # Register global error handler
        self.app.error(self.handle_global_error)
        
        # Register middleware for error logging
        self.app.middleware(self.error_logging_middleware)
        
        logger.info("Error handlers registered successfully")
    
    def handle_global_error(self, error: Exception, body: Dict[str, Any], logger):
        """
        Handle global errors that occur during event processing.
        
        Args:
            error: The exception that occurred
            body: Event payload that caused the error
            logger: Logger instance
        """
        try:
            # Log the error with context
            self._log_error(error, body)
            
            # Determine error type and appropriate response
            error_response = self._get_error_response(error, body)
            
            # Send error response to user if possible
            self._send_error_response(error_response, body)
            
        except Exception as e:
            # If error handling itself fails, log it
            logger.error(f"Error in error handler: {str(e)}", exc_info=True)
    
    def error_logging_middleware(self, args, next):
        """
        Middleware to log errors and performance metrics.
        
        Args:
            args: Middleware arguments
            next: Next middleware function
        """
        try:
            # Execute the next middleware/handler
            return next()
        except Exception as e:
            # Log the error
            self._log_middleware_error(e, args)
            raise
    
    def _log_error(self, error: Exception, body: Dict[str, Any]):
        """Log error with detailed context."""
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "event_type": body.get("type", "unknown"),
            "channel": body.get("channel", {}).get("id", "unknown"),
            "user": body.get("user", {}).get("id", "unknown"),
            "timestamp": body.get("event_ts", "unknown")
        }
        
        logger.error(
            f"Slack event processing error: {error_context['error_type']}",
            **error_context,
            exc_info=True
        )
        
        # Log stack trace for debugging
        logger.debug(f"Error stack trace: {traceback.format_exc()}")
    
    def _log_middleware_error(self, error: Exception, args):
        """Log middleware errors."""
        logger.error(
            f"Middleware error: {type(error).__name__} - {str(error)}",
            exc_info=True
        )
    
    def _get_error_response(self, error: Exception, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine appropriate error response based on error type.
        
        Args:
            error: The exception that occurred
            body: Event payload
            
        Returns:
            Error response dictionary
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        # Categorize errors
        if isinstance(error, TimeoutError):
            return {
                "type": "timeout",
                "user_message": "⏰ 요청 처리 시간이 초과되었습니다. 나중에 다시 시도해주세요.",
                "technical_message": f"Timeout error: {error_message}"
            }
        
        elif isinstance(error, ValueError):
            return {
                "type": "validation",
                "user_message": "❌ 입력값이 올바르지 않습니다. 다른 방식으로 질문해주세요.",
                "technical_message": f"Validation error: {error_message}"
            }
        
        elif isinstance(error, ConnectionError):
            return {
                "type": "connection",
                "user_message": "🔌 데이터베이스 연결에 문제가 있습니다. 잠시 후 다시 시도해주세요.",
                "technical_message": f"Connection error: {error_message}"
            }
        
        elif isinstance(error, PermissionError):
            return {
                "type": "permission",
                "user_message": "🔒 권한이 없습니다. 관리자에게 문의하세요.",
                "technical_message": f"Permission error: {error_message}"
            }
        
        elif isinstance(error, BoltError):
            return {
                "type": "slack_api",
                "user_message": "📱 Slack API 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                "technical_message": f"Slack API error: {error_message}"
            }
        
        else:
            # Generic error
            if self.show_error_details:
                return {
                    "type": "generic",
                    "user_message": f"❌ 오류가 발생했습니다: {error_message}",
                    "technical_message": f"Generic error: {error_message}"
                }
            else:
                return {
                    "type": "generic",
                    "user_message": "❌ 처리 중 오류가 발생했습니다. 관리자에게 문의하세요.",
                    "technical_message": f"Generic error: {error_message}"
                }
    
    def _send_error_response(self, error_response: Dict[str, Any], body: Dict[str, Any]):
        """Send error response to user if possible."""
        try:
            # Try to determine the appropriate channel to send the error
            channel_id = self._get_error_response_channel(body)
            
            if channel_id:
                # Send error message (this would need access to the Slack client)
                # For now, we just log that we would send a response
                logger.info(
                    f"Would send error response to channel {channel_id}",
                    error_type=error_response["type"],
                    user_message=error_response["user_message"]
                )
        
        except Exception as e:
            logger.error(f"Failed to send error response: {str(e)}")
    
    def _get_error_response_channel(self, body: Dict[str, Any]) -> Optional[str]:
        """Determine the appropriate channel to send error response."""
        try:
            # For direct messages
            if body.get("channel_type") == "im":
                return body.get("channel", {}).get("id")
            
            # For channel messages
            elif body.get("channel"):
                return body.get("channel", {}).get("id")
            
            # For app mentions
            elif body.get("event", {}).get("channel"):
                return body.get("event", {}).get("channel")
            
            return None
        
        except Exception:
            return None
    
    def handle_validation_error(self, error: ValueError, context: Dict[str, Any]) -> str:
        """
        Handle validation errors with specific guidance.
        
        Args:
            error: Validation error
            context: Error context
            
        Returns:
            User-friendly error message
        """
        error_message = str(error).lower()
        
        if "api key" in error_message:
            return "🔑 API 키 설정에 문제가 있습니다. 관리자에게 문의하세요."
        
        elif "database" in error_message or "connection" in error_message:
            return "🗄️ 데이터베이스 연결에 문제가 있습니다. 잠시 후 다시 시도해주세요."
        
        elif "sql" in error_message:
            return "📝 SQL 쿼리에 문제가 있습니다. 다른 방식으로 질문해주세요."
        
        elif "timeout" in error_message:
            return "⏰ 요청 시간이 초과되었습니다. 더 간단한 쿼리로 시도해주세요."
        
        else:
            return "❌ 입력값을 확인해주세요. 다른 방식으로 질문해보세요."
    
    def handle_rate_limit_error(self, error: Exception, retry_after: Optional[int] = None) -> str:
        """
        Handle rate limit errors.
        
        Args:
            error: Rate limit error
            retry_after: Seconds to wait before retrying
            
        Returns:
            User-friendly error message
        """
        if retry_after:
            return f"⏳ 요청이 너무 많습니다. {retry_after}초 후에 다시 시도해주세요."
        else:
            return "⏳ 요청이 너무 많습니다. 잠시 후 다시 시도해주세요."
    
    def handle_authentication_error(self, error: Exception) -> str:
        """
        Handle authentication errors.
        
        Args:
            error: Authentication error
            
        Returns:
            User-friendly error message
        """
        return "🔐 인증에 문제가 있습니다. 관리자에게 문의하세요."
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics for monitoring.
        
        Returns:
            Error statistics dictionary
        """
        # This would typically come from a metrics collection system
        return {
            "total_errors": 0,
            "error_types": {},
            "error_rate": 0.0,
            "last_error_time": None
        }
    
    def reset_error_statistics(self):
        """Reset error statistics."""
        # This would typically reset metrics in a monitoring system
        logger.info("Error statistics reset")








