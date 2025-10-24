"""
Main Handler for Slack Events

This module integrates all Slack event handlers into a single interface.
"""

from typing import Optional, Dict, Any
from slack_bolt import App
from core.logging import get_logger
from .message_handler import MessageHandler
from .interactive_handler import InteractiveHandler
from .error_handler import SlackErrorHandler

logger = get_logger(__name__)


class SlackEventHandler:
    """Main handler that integrates all Slack event handlers."""
    
    def __init__(self, app: App, agent_runner=None, show_error_details: bool = False):
        """
        Initialize the main Slack event handler.
        
        Args:
            app: Slack Bolt App instance
            agent_runner: Agent runner for processing queries
            show_error_details: Whether to show detailed error information
        """
        self.app = app
        self.agent_runner = agent_runner
        self.show_error_details = show_error_details
        
        # Initialize handlers
        self.message_handler = None
        self.interactive_handler = None
        self.error_handler = None
        
        # Register all handlers
        self._initialize_handlers()
        
        logger.info("Slack event handler initialized successfully")
    
    def _initialize_handlers(self):
        """Initialize and register all event handlers."""
        try:
            # Initialize error handler first
            self.error_handler = SlackErrorHandler(
                self.app, 
                show_error_details=self.show_error_details
            )
            
            # Initialize message handler
            self.message_handler = MessageHandler(
                self.app,
                agent_runner=self.agent_runner
            )
            
            # Initialize interactive handler
            self.interactive_handler = InteractiveHandler(
                self.app,
                agent_runner=self.agent_runner
            )
            
            logger.info("All Slack handlers initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Slack handlers: {str(e)}", exc_info=True)
            raise
    
    def get_handler_status(self) -> Dict[str, Any]:
        """
        Get status of all handlers.
        
        Returns:
            Dictionary containing handler status information
        """
        return {
            "message_handler": {
                "initialized": self.message_handler is not None,
                "type": type(self.message_handler).__name__ if self.message_handler else None
            },
            "interactive_handler": {
                "initialized": self.interactive_handler is not None,
                "type": type(self.interactive_handler).__name__ if self.interactive_handler else None
            },
            "error_handler": {
                "initialized": self.error_handler is not None,
                "type": type(self.error_handler).__name__ if self.error_handler else None,
                "show_details": self.show_error_details
            },
            "agent_runner": {
                "available": self.agent_runner is not None,
                "type": type(self.agent_runner).__name__ if self.agent_runner else None
            }
        }
    
    def update_agent_runner(self, new_agent_runner):
        """
        Update the agent runner for all handlers.
        
        Args:
            new_agent_runner: New agent runner instance
        """
        self.agent_runner = new_agent_runner
        
        if self.message_handler:
            self.message_handler.agent_runner = new_agent_runner
        
        if self.interactive_handler:
            self.interactive_handler.agent_runner = new_agent_runner
        
        logger.info("Agent runner updated for all handlers")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics from the error handler.
        
        Returns:
            Error statistics dictionary
        """
        if self.error_handler:
            return self.error_handler.get_error_statistics()
        else:
            return {"error": "Error handler not initialized"}
    
    def reset_error_statistics(self):
        """Reset error statistics."""
        if self.error_handler:
            self.error_handler.reset_error_statistics()
        else:
            logger.warning("Error handler not initialized - cannot reset statistics")








