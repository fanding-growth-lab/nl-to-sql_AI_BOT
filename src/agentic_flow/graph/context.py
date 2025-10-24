"""
Context Management for LangGraph State Machine

This module implements context management functionality for maintaining
conversation history, session state, and state persistence across pipeline executions.
"""

import json
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

from core.logging import get_logger

logger = get_logger(__name__)


class MessageRole(Enum):
    """Message role enumeration."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ConversationMessage:
    """Individual conversation message."""
    role: MessageRole
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMessage':
        """Create from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata")
        )


@dataclass
class ConversationHistory:
    """Conversation history management."""
    messages: List[ConversationMessage] = None
    max_history: int = 50
    max_age_hours: int = 24
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []
    
    def add_message(self, role: MessageRole, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a new message to the conversation."""
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            metadata=metadata
        )
        
        self.messages.append(message)
        
        # Clean up old messages
        self._cleanup_messages()
        
        logger.debug(f"Added {role.value} message to conversation history")
    
    def get_recent_messages(self, count: int = 10) -> List[ConversationMessage]:
        """Get recent messages from the conversation."""
        return self.messages[-count:] if self.messages else []
    
    def get_context_for_prompt(self, max_messages: int = 5) -> List[Dict[str, Any]]:
        """Get conversation context formatted for LLM prompts."""
        recent_messages = self.get_recent_messages(max_messages * 2)  # User + Assistant pairs
        
        context = []
        for message in recent_messages:
            context.append({
                "role": message.role.value,
                "content": message.content,
                "timestamp": message.timestamp
            })
        
        return context
    
    def _cleanup_messages(self):
        """Clean up old messages based on count and age."""
        # Remove excess messages
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
        
        # Remove old messages
        cutoff_time = datetime.now() - timedelta(hours=self.max_age_hours)
        self.messages = [
            msg for msg in self.messages 
            if datetime.fromisoformat(msg.timestamp) > cutoff_time
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "messages": [msg.to_dict() for msg in self.messages],
            "max_history": self.max_history,
            "max_age_hours": self.max_age_hours
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationHistory':
        """Create from dictionary."""
        messages = [ConversationMessage.from_dict(msg) for msg in data.get("messages", [])]
        return cls(
            messages=messages,
            max_history=data.get("max_history", 50),
            max_age_hours=data.get("max_age_hours", 24)
        )


@dataclass
class SessionState:
    """Session state management."""
    session_id: str
    created_at: str
    last_accessed: str
    conversation_history: ConversationHistory
    current_state: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def update_access_time(self):
        """Update last accessed time."""
        self.last_accessed = datetime.now().isoformat()
    
    def is_expired(self, max_idle_hours: int = 24) -> bool:
        """Check if session is expired."""
        last_accessed = datetime.fromisoformat(self.last_accessed)
        cutoff_time = datetime.now() - timedelta(hours=max_idle_hours)
        return last_accessed < cutoff_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "conversation_history": self.conversation_history.to_dict(),
            "current_state": self.current_state,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionState':
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            created_at=data["created_at"],
            last_accessed=data["last_accessed"],
            conversation_history=ConversationHistory.from_dict(data["conversation_history"]),
            current_state=data.get("current_state"),
            metadata=data.get("metadata")
        )


class AgentContextManager:
    """
    Agent context manager for maintaining conversation state and history.
    
    This class manages multiple sessions, conversation history, and provides
    context for LLM prompts.
    """
    
    def __init__(self, storage_backend: Optional[str] = None):
        """
        Initialize the context manager.
        
        Args:
            storage_backend: Storage backend type ("memory", "file", "redis")
        """
        self.sessions: Dict[str, SessionState] = {}
        self.storage_backend = storage_backend or "memory"
        self.max_sessions = 1000  # Maximum number of sessions to keep in memory
        self.session_timeout_hours = 24
        
        logger.info(f"AgentContextManager initialized with {self.storage_backend} backend")
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new session.
        
        Args:
            session_id: Optional custom session ID
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        current_time = datetime.now().isoformat()
        
        session_state = SessionState(
            session_id=session_id,
            created_at=current_time,
            last_accessed=current_time,
            conversation_history=ConversationHistory()
        )
        
        self.sessions[session_id] = session_state
        
        # Clean up old sessions if we exceed the limit
        self._cleanup_sessions()
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """
        Get session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionState if found, None otherwise
        """
        session = self.sessions.get(session_id)
        if session:
            session.update_access_time()
            
            # Check if session is expired
            if session.is_expired(self.session_timeout_hours):
                logger.warning(f"Session {session_id} is expired, removing")
                self.remove_session(session_id)
                return None
            
        return session
    
    def remove_session(self, session_id: str) -> bool:
        """
        Remove a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if removed, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Removed session: {session_id}")
            return True
        return False
    
    def add_user_message(self, session_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a user message to the session."""
        session = self.get_session(session_id)
        if session:
            session.conversation_history.add_message(
                MessageRole.USER,
                content,
                metadata
            )
        else:
            logger.warning(f"Session {session_id} not found for user message")
    
    def add_assistant_message(self, session_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add an assistant message to the session."""
        session = self.get_session(session_id)
        if session:
            session.conversation_history.add_message(
                MessageRole.ASSISTANT,
                content,
                metadata
            )
        else:
            logger.warning(f"Session {session_id} not found for assistant message")
    
    def get_conversation_context(self, session_id: str, max_messages: int = 5) -> List[Dict[str, Any]]:
        """
        Get conversation context for LLM prompts.
        
        Args:
            session_id: Session identifier
            max_messages: Maximum number of messages to include
            
        Returns:
            List of conversation messages formatted for prompts
        """
        session = self.get_session(session_id)
        if session:
            return session.conversation_history.get_context_for_prompt(max_messages)
        else:
            logger.warning(f"Session {session_id} not found for context retrieval")
            return []
    
    def update_session_state(self, session_id: str, state: Dict[str, Any]):
        """Update the current state for a session."""
        session = self.get_session(session_id)
        if session:
            session.current_state = state
            session.update_access_time()
        else:
            logger.warning(f"Session {session_id} not found for state update")
    
    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state for a session."""
        session = self.get_session(session_id)
        return session.current_state if session else None
    
    def save_session(self, session_id: str, filepath: Optional[str] = None) -> bool:
        """
        Save session to storage.
        
        Args:
            session_id: Session identifier
            filepath: Optional file path for file storage
            
        Returns:
            True if saved successfully, False otherwise
        """
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found for saving")
            return False
        
        try:
            if self.storage_backend == "file":
                if filepath is None:
                    filepath = f"sessions/{session_id}.json"
                
                # Ensure directory exists
                import os
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
                
                logger.info(f"Saved session {session_id} to {filepath}")
                return True
            
            elif self.storage_backend == "memory":
                # Already in memory, no action needed
                return True
            
            else:
                logger.warning(f"Unsupported storage backend: {self.storage_backend}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            return False
    
    def load_session(self, session_id: str, filepath: Optional[str] = None) -> bool:
        """
        Load session from storage.
        
        Args:
            session_id: Session identifier
            filepath: Optional file path for file storage
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if self.storage_backend == "file":
                if filepath is None:
                    filepath = f"sessions/{session_id}.json"
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                session = SessionState.from_dict(session_data)
                self.sessions[session_id] = session
                
                logger.info(f"Loaded session {session_id} from {filepath}")
                return True
            
            else:
                logger.warning(f"Unsupported storage backend: {self.storage_backend}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return False
    
    def _cleanup_sessions(self):
        """Clean up expired sessions."""
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if session.is_expired(self.session_timeout_hours):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.remove_session(session_id)
        
        # If still over limit, remove oldest sessions
        if len(self.sessions) > self.max_sessions:
            sessions_by_age = sorted(
                self.sessions.items(),
                key=lambda x: x[1].last_accessed
            )
            
            sessions_to_remove = len(self.sessions) - self.max_sessions
            for session_id, _ in sessions_by_age[:sessions_to_remove]:
                self.remove_session(session_id)
        
        if expired_sessions or len(self.sessions) > self.max_sessions:
            logger.info(f"Cleaned up sessions, current count: {len(self.sessions)}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about managed sessions."""
        active_sessions = len(self.sessions)
        total_messages = sum(
            len(session.conversation_history.messages)
            for session in self.sessions.values()
        )
        
        return {
            "active_sessions": active_sessions,
            "total_messages": total_messages,
            "storage_backend": self.storage_backend,
            "max_sessions": self.max_sessions,
            "session_timeout_hours": self.session_timeout_hours
        }
