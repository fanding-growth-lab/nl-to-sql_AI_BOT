"""
Logging configuration module for DataTalk bot.
Provides structured logging using structlog with JSON formatting.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import structlog
from structlog.stdlib import LoggerFactory
from structlog.dev import ConsoleRenderer
from structlog.processors import JSONRenderer, TimeStamper

from .config import get_settings


def configure_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_json: bool = True
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        enable_json: Whether to use JSON formatting
    """
    settings = get_settings()
    
    # Use provided values or fall back to settings
    level = log_level or settings.logging.level
    file_path = log_file or settings.logging.file_path
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure standard library logging with UTF-8 encoding
    logging.basicConfig(
        level=numeric_level,
        format=settings.logging.format,
        handlers=_get_log_handlers(file_path, numeric_level),
        force=True,  # Force reconfiguration
        encoding='utf-8'  # Ensure UTF-8 encoding for console output
    )
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    # Add appropriate renderer based on environment
    if enable_json and not settings.debug:
        processors.append(JSONRenderer())
    else:
        processors.append(ConsoleRenderer(colors=settings.debug))
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _get_log_handlers(file_path: Optional[str], level: int) -> list:
    """
    Get appropriate log handlers based on configuration.
    
    Args:
        file_path: Path to log file
        level: Logging level
        
    Returns:
        List of log handlers
    """
    handlers = []
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setStream(sys.stdout)  # Ensure UTF-8 encoding
    handlers.append(console_handler)
    
    # File handler if specified
    if file_path:
        # Create log directory if it doesn't exist
        log_path = Path(file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler for production with UTF-8 encoding
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        handlers.append(file_handler)
    
    return handlers


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        structlog.BoundLogger: Structured logger instance
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class.
    """
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger for this class."""
        return get_logger(self.__class__.__module__ + "." + self.__class__.__name__)


def log_function_call(func_name: str, **kwargs) -> structlog.BoundLogger:
    """
    Log a function call with parameters.
    
    Args:
        func_name: Name of the function being called
        **kwargs: Function parameters to log
        
    Returns:
        structlog.BoundLogger: Logger instance
    """
    logger = get_logger("function_calls")
    return logger.bind(function=func_name, **kwargs)


def log_database_operation(operation: str, table: str = None, query: str = None, **kwargs) -> structlog.BoundLogger:
    """
    Log a database operation.
    
    Args:
        operation: Type of database operation
        table: Table name (optional)
        query: SQL query (optional)
        **kwargs: Additional parameters
        
    Returns:
        structlog.BoundLogger: Logger instance
    """
    logger = get_logger("database")
    return logger.bind(
        operation=operation,
        table=table,
        query=query[:100] + "..." if query and len(query) > 100 else query,
        **kwargs
    )


def log_slack_event(event_type: str, user: str = None, channel: str = None, **kwargs) -> structlog.BoundLogger:
    """
    Log a Slack event.
    
    Args:
        event_type: Type of Slack event
        user: User ID (optional)
        channel: Channel ID (optional)
        **kwargs: Additional parameters
        
    Returns:
        structlog.BoundLogger: Logger instance
    """
    logger = get_logger("slack")
    return logger.bind(
        event_type=event_type,
        user=user,
        channel=channel,
        **kwargs
    )


def log_llm_operation(operation: str, model: str = None, tokens: int = None, **kwargs) -> structlog.BoundLogger:
    """
    Log an LLM operation.
    
    Args:
        operation: Type of LLM operation
        model: Model name (optional)
        tokens: Token count (optional)
        **kwargs: Additional parameters
        
    Returns:
        structlog.BoundLogger: Logger instance
    """
    logger = get_logger("llm")
    return logger.bind(
        operation=operation,
        model=model,
        tokens=tokens,
        **kwargs
    )


def log_error(error: Exception, context: Dict[str, Any] = None) -> structlog.BoundLogger:
    """
    Log an error with context.
    
    Args:
        error: Exception instance
        context: Additional context (optional)
        
    Returns:
        structlog.BoundLogger: Logger instance
    """
    logger = get_logger("errors")
    return logger.bind(
        error_type=type(error).__name__,
        error_message=str(error),
        context=context or {}
    )


def log_performance(operation: str, duration: float, **kwargs) -> structlog.BoundLogger:
    """
    Log performance metrics.
    
    Args:
        operation: Operation name
        duration: Duration in seconds
        **kwargs: Additional metrics
        
    Returns:
        structlog.BoundLogger: Logger instance
    """
    logger = get_logger("performance")
    return logger.bind(
        operation=operation,
        duration=duration,
        **kwargs
    )


# Initialize logging when module is imported
configure_logging()