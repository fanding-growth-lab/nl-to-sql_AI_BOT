"""
Simplified Context Management using LangGraph's Built-in Features

This module provides minimal business-specific context management
while fully leveraging LangGraph's proven session and state management.
"""

import uuid
from typing import Dict, Optional, Any
from dataclasses import dataclass

from langgraph.checkpoint.memory import MemorySaver
from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BusinessMetadata:
    """Minimal business metadata for session tracking."""
    user_id: Optional[str] = None
    channel_id: Optional[str] = None
    session_created_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for AgentState."""
        return {
            "user_id": self.user_id,
            "channel_id": self.channel_id,
            "session_created_at": self.session_created_at
        }


class AgentContextManager:
    """
    Ultra-simplified context manager that leverages LangGraph's built-in capabilities.
    
    This class provides only essential business metadata while letting
    LangGraph handle all session and conversation management.
    """
    
    def __init__(self, checkpointer: Optional[MemorySaver] = None):
        """
        Initialize the context manager.
        
        Args:
            checkpointer: LangGraph checkpointer instance (uses MemorySaver by default)
        """
        self.checkpointer = checkpointer or MemorySaver()
        logger.info("AgentContextManager initialized with LangGraph checkpointer")
    
    def create_session_id(self) -> str:
        """
        Generate a unique session ID for LangGraph's thread_id.
        
        Returns:
            Session ID to be used as thread_id in LangGraph
        """
        session_id = str(uuid.uuid4())
        logger.debug(f"Generated session ID: {session_id}")
        return session_id
    
    def get_business_metadata(self, 
                           user_id: Optional[str] = None,
                           channel_id: Optional[str] = None) -> BusinessMetadata:
        """
        Create business metadata for AgentState.
        
        Args:
            user_id: Slack user ID
            channel_id: Slack channel ID
            
        Returns:
            BusinessMetadata object for AgentState
        """
        return BusinessMetadata(
            user_id=user_id,
            channel_id=channel_id,
            session_created_at=None  # Will be set by AgentState
        )
    
    def get_checkpointer(self) -> MemorySaver:
        """Get the LangGraph checkpointer instance."""
        return self.checkpointer
