"""
Custom exceptions for DataTalk bot.
Provides structured error handling with context and logging.
"""

from typing import Optional, Dict, Any, Union
from enum import Enum
import traceback
from dataclasses import dataclass

from .logging import get_logger, log_error


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Error context information."""
    user_id: Optional[str] = None
    channel_id: Optional[str] = None
    operation: Optional[str] = None
    query: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


class DataTalkBaseException(Exception):
    """
    Base exception class for all DataTalk bot errors.
    Provides structured error handling with context and logging.
    """
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            severity: Error severity level
            context: Error context information
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.context = context or ErrorContext()
        self.original_error = original_error
        self.traceback = traceback.format_exc()
        
        # Log the error
        self._log_error()
    
    def _log_error(self):
        """Log the error with context."""
        logger = get_logger("exceptions")
        logger.error(
            "DataTalk error occurred",
            error_type=self.__class__.__name__,
            message=self.message,
            severity=self.severity.value,
            context=self.context.__dict__,
            original_error=str(self.original_error) if self.original_error else None,
            traceback=self.traceback
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for API responses.
        
        Returns:
            Dict containing error information
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "severity": self.severity.value,
            "context": self.context.__dict__,
            "original_error": str(self.original_error) if self.original_error else None
        }


class ConfigurationError(DataTalkBaseException):
    """Raised when there's a configuration error."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            context=ErrorContext(additional_data={"config_key": config_key}),
            **kwargs
        )


class DatabaseError(DataTalkBaseException):
    """Raised when there's a database-related error."""
    
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        # Filter out conflicting kwargs but allow severity override
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['context', 'severity']}
        
        # Use severity from kwargs if provided, otherwise default to HIGH
        severity = kwargs.get('severity', ErrorSeverity.HIGH)
        
        super().__init__(
            message,
            severity=severity,
            context=ErrorContext(
                operation=operation,
                query=query
            ),
            **filtered_kwargs
        )


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    
    def __init__(self, message: str, **kwargs):
        # Filter out conflicting kwargs but allow severity override
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['operation']}
        
        super().__init__(
            message,
            operation="connection",
            severity=ErrorSeverity.CRITICAL,
            **filtered_kwargs
        )


class DatabaseQueryError(DatabaseError):
    """Raised when database query fails."""
    
    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            query=query,
            operation="query",
            **kwargs
        )


class SlackError(DataTalkBaseException):
    """Raised when there's a Slack-related error."""
    
    def __init__(
        self,
        message: str,
        user_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        **kwargs
    ):
        # Extract context from kwargs if present
        existing_context = kwargs.get('context')
        
        # Create context with user_id and channel_id
        context_data = {
            "user_id": user_id,
            "channel_id": channel_id,
            "operation": None,
            "query": None,
            "additional_data": None
        }
        
        # Merge with existing context if present
        if existing_context:
            context_data.update({
                "user_id": existing_context.user_id or user_id,
                "channel_id": existing_context.channel_id or channel_id,
                "operation": existing_context.operation,
                "query": existing_context.query,
                "additional_data": existing_context.additional_data
            })
        
        # Filter out conflicting kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['context']}
        
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            context=ErrorContext(**context_data),
            **filtered_kwargs
        )


class SlackAPIError(SlackError):
    """Raised when Slack API call fails."""
    
    def __init__(self, message: str, api_method: Optional[str] = None, **kwargs):
        # Filter out conflicting kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['context']}
        
        super().__init__(
            message,
            context=ErrorContext(additional_data={"api_method": api_method}),
            **filtered_kwargs
        )


class LLMError(DataTalkBaseException):
    """Raised when there's an LLM-related error."""
    
    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            context=ErrorContext(
                operation=operation,
                additional_data={"model": model}
            ),
            **kwargs
        )


class LLMAPIError(LLMError):
    """Raised when LLM API call fails."""
    
    def __init__(self, message: str, model: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            model=model,
            operation="api_call",
            **kwargs
        )


class LLMGenerationError(LLMError):
    """Raised when LLM fails to generate response."""
    
    def __init__(self, message: str, model: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            model=model,
            operation="generation",
            **kwargs
        )


class ValidationError(DataTalkBaseException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.LOW,
            context=ErrorContext(additional_data={"field": field}),
            **kwargs
        )


class SecurityError(DataTalkBaseException):
    """Raised when security validation fails."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )


class SQLSecurityError(SecurityError):
    """Raised when SQL security validation fails."""
    
    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            context=ErrorContext(query=query),
            **kwargs
        )


class RateLimitError(DataTalkBaseException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str, service: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            context=ErrorContext(additional_data={"service": service}),
            **kwargs
        )


class TimeoutError(DataTalkBaseException):
    """Raised when operation times out."""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            context=ErrorContext(operation=operation),
            **kwargs
        )


class BusinessLogicError(DataTalkBaseException):
    """Raised when business logic validation fails."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


def handle_exception(
    error: Exception,
    context: Optional[ErrorContext] = None,
    reraise: bool = True
) -> DataTalkBaseException:
    """
    Handle and wrap any exception in a DataTalk exception.
    
    Args:
        error: Original exception
        context: Additional context
        reraise: Whether to reraise the wrapped exception
        
    Returns:
        DataTalkBaseException: Wrapped exception
    """
    if isinstance(error, DataTalkBaseException):
        return error
    
    # Wrap unknown exceptions
    wrapped_error = DataTalkBaseException(
        message=f"Unexpected error: {str(error)}",
        severity=ErrorSeverity.HIGH,
        context=context,
        original_error=error
    )
    
    if reraise:
        raise wrapped_error
    
    return wrapped_error


def create_error_response(
    error: DataTalkBaseException,
    include_traceback: bool = False
) -> Dict[str, Any]:
    """
    Create a standardized error response for API endpoints.
    
    Args:
        error: DataTalk exception
        include_traceback: Whether to include traceback in response
        
    Returns:
        Dict containing error response
    """
    response = {
        "success": False,
        "error": error.to_dict()
    }
    
    if include_traceback:
        response["traceback"] = error.traceback
    
    return response