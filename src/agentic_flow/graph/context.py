"""
Simplified Context Management using LangGraph's Built-in Features

This module provides a lightweight wrapper around LangGraph's built-in
context management capabilities, focusing on business-specific functionality
while leveraging the framework's proven session management.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from langgraph.checkpoint.memory import MemorySaver
from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationContext:
    """Simplified conversation context for business logic."""
    session_id: str
    user_id: Optional[str] = None
    channel_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LangGraph state."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "channel_id": self.channel_id,
            "metadata": self.metadata or {},
            "timestamp": datetime.now().isoformat()
        }


class AgentContextManager:
    """
    Simplified context manager leveraging LangGraph's built-in capabilities.
    
    This class provides business-specific context management while using
    LangGraph's proven session and state management under the hood.
    """
    
    def __init__(self, checkpointer: Optional[MemorySaver] = None):
        """
        Initialize the context manager.
        
        Args:
            checkpointer: LangGraph checkpointer instance (uses MemorySaver by default)
        """
        self.checkpointer = checkpointer or MemorySaver()
        self.business_contexts: Dict[str, ConversationContext] = {}
        
        logger.info("AgentContextManager initialized with LangGraph checkpointer")
    
    def create_session(self, 
                      session_id: Optional[str] = None,
                      user_id: Optional[str] = None,
                      channel_id: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new session with business context.
        
        Args:
            session_id: Optional custom session ID
            user_id: Slack user ID
            channel_id: Slack channel ID
            metadata: Additional business metadata
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Create business context
        business_context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            channel_id=channel_id,
            metadata=metadata
        )
        
        self.business_contexts[session_id] = business_context
        
        logger.info(f"Created session {session_id} for user {user_id} in channel {channel_id}")
        return session_id
    
    def get_business_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get business context for a session."""
        return self.business_contexts.get(session_id)
    
    def update_business_context(self, 
                              session_id: str, 
                              user_id: Optional[str] = None,
                              channel_id: Optional[str] = None,
                              metadata: Optional[Dict[str, Any]] = None):
        """Update business context for a session."""
        if session_id in self.business_contexts:
            context = self.business_contexts[session_id]
            if user_id is not None:
                context.user_id = user_id
            if channel_id is not None:
                context.channel_id = channel_id
            if metadata is not None:
                context.metadata = {**(context.metadata or {}), **metadata}
            
            logger.debug(f"Updated business context for session {session_id}")
        else:
            logger.warning(f"Session {session_id} not found for context update")
    
    def get_conversation_context(self, session_id: str, max_messages: int = 5) -> List[Dict[str, Any]]:
        """
        Get conversation context for LLM prompts.
        
        This method leverages LangGraph's built-in conversation history
        while adding business-specific context.
        
        Args:
            session_id: Session identifier
            max_messages: Maximum number of messages to include
            
        Returns:
            List of conversation messages formatted for prompts
        """
        try:
            # Get LangGraph's conversation history
            # Note: This would typically be called from within the graph execution
            # where we have access to the checkpointer and thread_id
            
            business_context = self.get_business_context(session_id)
            if not business_context:
                logger.warning(f"No business context found for session {session_id}")
                return []
            
            # Return business context for now
            # In a full implementation, this would merge with LangGraph's history
            return [business_context.to_dict()]
            
        except Exception as e:
            logger.error(f"Failed to get conversation context for session {session_id}: {e}")
            return []
    
    def add_user_message(self, session_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a user message to the session.
        
        Note: In LangGraph, this is typically handled automatically
        through the graph execution. This method is kept for compatibility.
        """
        business_context = self.get_business_context(session_id)
        if business_context:
            # Update metadata with message info
            message_metadata = {
                "last_user_message": content,
                "last_message_time": datetime.now().isoformat(),
                **(metadata or {})
            }
            self.update_business_context(session_id, metadata=message_metadata)
            logger.debug(f"Added user message to session {session_id}")
        else:
            logger.warning(f"Session {session_id} not found for user message")
    
    def add_assistant_message(self, session_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add an assistant message to the session.
        
        Note: In LangGraph, this is typically handled automatically
        through the graph execution. This method is kept for compatibility.
        """
        business_context = self.get_business_context(session_id)
        if business_context:
            # Update metadata with response info
            response_metadata = {
                "last_assistant_message": content,
                "last_response_time": datetime.now().isoformat(),
                **(metadata or {})
            }
            self.update_business_context(session_id, metadata=response_metadata)
            logger.debug(f"Added assistant message to session {session_id}")
        else:
            logger.warning(f"Session {session_id} not found for assistant message")
    
    def remove_session(self, session_id: str) -> bool:
        """Remove a session and its business context."""
        if session_id in self.business_contexts:
            del self.business_contexts[session_id]
            logger.info(f"Removed session: {session_id}")
            return True
        return False
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about managed sessions."""
        active_sessions = len(self.business_contexts)
        
        # Count sessions by channel type
        channel_stats = {}
        for context in self.business_contexts.values():
            channel_type = "dm" if context.channel_id and context.channel_id.startswith("D") else "channel"
            channel_stats[channel_type] = channel_stats.get(channel_type, 0) + 1
        
        return {
            "active_sessions": active_sessions,
            "channel_distribution": channel_stats,
            "checkpointer_type": type(self.checkpointer).__name__
        }
    
    def cleanup_expired_sessions(self, max_age_hours: int = 24):
        """
        Clean up expired sessions.
        
        Note: LangGraph's checkpointer handles most cleanup automatically.
        This method cleans up our business context.
        """
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        expired_sessions = []
        
        for session_id, context in self.business_contexts.items():
            # Check if session metadata indicates it's expired
            if context.metadata:
                last_activity = context.metadata.get("last_message_time") or context.metadata.get("last_response_time")
                if last_activity:
                    try:
                        activity_time = datetime.fromisoformat(last_activity).timestamp()
                        if activity_time < cutoff_time:
                            expired_sessions.append(session_id)
                    except ValueError:
                        # Invalid timestamp, consider expired
                        expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.remove_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
