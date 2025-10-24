"""
Global error handler for DataTalk bot.
Provides centralized error handling and response formatting.
"""

import asyncio
from typing import Dict, Any, Optional, Callable
from functools import wraps
import traceback
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .exceptions import (
    DataTalkBaseException,
    handle_exception,
    create_error_response,
    ErrorContext,
    ErrorSeverity
)
from .logging import get_logger, log_error


logger = get_logger(__name__)


class ErrorHandler:
    """
    Centralized error handler for the application.
    """
    
    def __init__(self):
        self.error_handlers: Dict[type, Callable] = {}
        self._register_default_handlers()
    
    def register_handler(self, exception_type: type, handler: Callable):
        """
        Register a custom error handler.
        
        Args:
            exception_type: Exception type to handle
            handler: Handler function
        """
        self.error_handlers[exception_type] = handler
    
    def _register_default_handlers(self):
        """Register default error handlers."""
        self.error_handlers.update({
            DataTalkBaseException: self._handle_datatalk_error,
            HTTPException: self._handle_http_error,
            RequestValidationError: self._handle_validation_error,
            asyncio.TimeoutError: self._handle_timeout_error,
            Exception: self._handle_generic_error
        })
    
    def handle_error(self, error: Exception, request: Optional[Request] = None) -> JSONResponse:
        """
        Handle an error and return appropriate response.
        
        Args:
            error: Exception to handle
            request: FastAPI request object (optional)
            
        Returns:
            JSONResponse: Error response
        """
        # Get context from request if available
        context = self._extract_context_from_request(request)
        
        # Wrap unknown exceptions
        if not isinstance(error, DataTalkBaseException):
            error = handle_exception(error, context, reraise=False)
        
        # Log the error
        log_error(error, context.__dict__ if context else {})
        
        # Get appropriate handler
        handler = self._get_handler(type(error))
        
        # Handle the error
        return handler(error, request)
    
    def _get_handler(self, error_type: type) -> Callable:
        """
        Get the appropriate handler for an exception type.
        
        Args:
            error_type: Exception type
            
        Returns:
            Callable: Handler function
        """
        # Find the most specific handler
        for exception_type, handler in self.error_handlers.items():
            if issubclass(error_type, exception_type):
                return handler
        
        # Fall back to generic handler
        return self.error_handlers[Exception]
    
    def _extract_context_from_request(self, request: Optional[Request]) -> Optional[ErrorContext]:
        """
        Extract context information from FastAPI request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            ErrorContext: Extracted context
        """
        if not request:
            return None
        
        try:
            # Extract user and channel info from request headers or body
            user_id = request.headers.get("x-slack-user-id")
            channel_id = request.headers.get("x-slack-channel-id")
            
            return ErrorContext(
                user_id=user_id,
                channel_id=channel_id,
                additional_data={
                    "method": request.method,
                    "url": str(request.url),
                    "headers": dict(request.headers)
                }
            )
        except Exception:
            return None
    
    def _handle_datatalk_error(self, error: DataTalkBaseException, request: Optional[Request] = None) -> JSONResponse:
        """Handle DataTalk-specific errors."""
        status_code = self._get_status_code_for_severity(error.severity)
        
        return JSONResponse(
            status_code=status_code,
            content=create_error_response(error)
        )
    
    def _handle_http_error(self, error: HTTPException, request: Optional[Request] = None) -> JSONResponse:
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=error.status_code,
            content={
                "success": False,
                "error": {
                    "type": "HTTPException",
                    "message": error.detail,
                    "status_code": error.status_code
                }
            }
        )
    
    def _handle_validation_error(self, error: RequestValidationError, request: Optional[Request] = None) -> JSONResponse:
        """Handle request validation errors."""
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": {
                    "type": "ValidationError",
                    "message": "Request validation failed",
                    "details": error.errors()
                }
            }
        )
    
    def _handle_timeout_error(self, error: asyncio.TimeoutError, request: Optional[Request] = None) -> JSONResponse:
        """Handle timeout errors."""
        return JSONResponse(
            status_code=408,
            content={
                "success": False,
                "error": {
                    "type": "TimeoutError",
                    "message": "Request timed out",
                    "severity": ErrorSeverity.MEDIUM.value
                }
            }
        )
    
    def _handle_generic_error(self, error: Exception, request: Optional[Request] = None) -> JSONResponse:
        """Handle generic errors."""
        logger.error(
            "Unhandled error occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc()
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "type": "InternalServerError",
                    "message": "An internal server error occurred",
                    "severity": ErrorSeverity.CRITICAL.value
                }
            }
        )
    
    def _get_status_code_for_severity(self, severity: ErrorSeverity) -> int:
        """
        Get HTTP status code based on error severity.
        
        Args:
            severity: Error severity level
            
        Returns:
            int: HTTP status code
        """
        severity_mapping = {
            ErrorSeverity.LOW: 400,
            ErrorSeverity.MEDIUM: 400,
            ErrorSeverity.HIGH: 500,
            ErrorSeverity.CRITICAL: 500
        }
        
        return severity_mapping.get(severity, 500)


# Global error handler instance
error_handler = ErrorHandler()


def error_handler_decorator(func: Callable) -> Callable:
    """
    Decorator to handle errors in functions.
    
    Args:
        func: Function to wrap
        
    Returns:
        Callable: Wrapped function
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as error:
            logger.error(
                "Error in async function",
                function=func.__name__,
                error_type=type(error).__name__,
                error_message=str(error)
            )
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            logger.error(
                "Error in function",
                function=func.__name__,
                error_type=type(error).__name__,
                error_message=str(error)
            )
            raise
    
    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    reraise: bool = True,
    **kwargs
) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        default_return: Value to return if function fails
        reraise: Whether to reraise exceptions
        **kwargs: Function keyword arguments
        
    Returns:
        Any: Function result or default return value
    """
    try:
        return func(*args, **kwargs)
    except Exception as error:
        logger.error(
            "Error in safe_execute",
            function=func.__name__,
            error_type=type(error).__name__,
            error_message=str(error)
        )
        
        if reraise:
            raise
        
        return default_return


async def safe_execute_async(
    func: Callable,
    *args,
    default_return: Any = None,
    reraise: bool = True,
    **kwargs
) -> Any:
    """
    Safely execute an async function with error handling.
    
    Args:
        func: Async function to execute
        *args: Function arguments
        default_return: Value to return if function fails
        reraise: Whether to reraise exceptions
        **kwargs: Function keyword arguments
        
    Returns:
        Any: Function result or default return value
    """
    try:
        return await func(*args, **kwargs)
    except Exception as error:
        logger.error(
            "Error in safe_execute_async",
            function=func.__name__,
            error_type=type(error).__name__,
            error_message=str(error)
        )
        
        if reraise:
            raise
        
        return default_return