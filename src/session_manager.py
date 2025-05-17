"""
Session Manager

This module implements a robust session management system for A2A conversations.
It maintains context across interactions, groups related tasks, stores relevant history,
implements forgetting mechanisms, and balances immediate context with long-term knowledge.
"""

import json
import uuid
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from enum import Enum
import heapq
import pickle
import re

# Import relationship tracker for long-term knowledge integration
from relationship_tracker import RelationshipTracker, InteractionType, InteractionQuality

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SessionManager")


class SessionStatus(Enum):
    """Session status states."""
    ACTIVE = "active"                    # Session is currently active
    IDLE = "idle"                        # Session is idle but can be resumed
    EXPIRED = "expired"                  # Session has expired but is still stored
    ARCHIVED = "archived"                # Session is archived for long-term storage
    TERMINATED = "terminated"            # Session has been terminated


class MessageRelevance(Enum):
    """Relevance levels for messages in a session."""
    CRITICAL = 5                         # Essential messages that must be preserved
    HIGH = 4                             # Highly relevant messages
    MEDIUM = 3                           # Moderately relevant messages
    LOW = 2                              # Less relevant messages
    MINIMAL = 1                          # Minimally relevant messages that can be forgotten


class Session:
    """
    Represents a conversation session between agents.
    
    A session contains contextual information about an ongoing interaction,
    including message history, topic tracking, relationship data, and
    relevance scoring for memory management.
    """
    
    def __init__(
        self,
        session_id: str,
        agent_id: str,
        initiator_id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_inactive_time: Optional[int] = 3600,  # 1 hour in seconds
        max_message_count: Optional[int] = 100    # Maximum messages to store
    ):
        """
        Initialize a new session.
        
        Args:
            session_id: Unique identifier for the session
            agent_id: ID of the agent owning this session
            initiator_id: ID of the agent who initiated the session
            name: Optional name for the session
            metadata: Additional session metadata
            max_inactive_time: Maximum time (seconds) a session can be inactive before considered expired
            max_message_count: Maximum number of messages to store in the session
        """
        self.session_id = session_id
        self.agent_id = agent_id
        self.initiator_id = initiator_id
        self.name = name or f"Session with {initiator_id}"
        self.metadata = metadata or {}
        self.max_inactive_time = max_inactive_time
        self.max_message_count = max_message_count
        
        # Session state
        self.created_at = datetime.utcnow()
        self.last_active_at = self.created_at
        self.messages: List[Dict[str, Any]] = []
        self.message_relevance: Dict[str, MessageRelevance] = {}  # message_id -> relevance
        self.status = SessionStatus.ACTIVE
        
        # Context tracking
        self.topics: List[str] = []
        self.topic_relevance: Dict[str, float] = {}  # topic -> relevance score (0.0-1.0)
        self.entities: Dict[str, Dict[str, Any]] = {}  # Entities mentioned in the session
        self.intents: List[str] = []  # List of intents in the session
        self.last_intent = None
        
        # Session summary information
        self.summary: Optional[str] = None
        self.summary_last_updated = None
        
        # Task grouping
        self.tasks: Dict[str, Dict[str, Any]] = {}  # task_id -> task metadata
        
        # Custom data storage
        self.context_data: Dict[str, Any] = {}
        
        logger.info(f"Created new session {session_id} between {agent_id} and {initiator_id}")
    
    def add_message(
        self,
        message: Dict[str, Any],
        intent: Optional[str] = None,
        relevance: MessageRelevance = MessageRelevance.MEDIUM
    ) -> str:
        """
        Add a message to the session.
        
        Args:
            message: The message to add
            intent: The intent of the message
            relevance: The relevance level of the message
            
        Returns:
            Message ID
        """
        # Generate message ID if not present
        if "id" not in message:
            message_id = f"msg-{uuid.uuid4().hex[:8]}"
            message["id"] = message_id
        else:
            message_id = message["id"]
            
        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()
            
        # Add intent if provided
        if intent:
            message["intent"] = intent
            self.intents.append(intent)
            self.last_intent = intent
            
        # Store the message
        self.messages.append(message)
        
        # Store relevance
        self.message_relevance[message_id] = relevance
        
        # Update session state
        self.last_active_at = datetime.utcnow()
        self.status = SessionStatus.ACTIVE
        
        # Extract topics from message if available
        if "topics" in message:
            for topic in message["topics"]:
                if topic not in self.topics:
                    self.topics.append(topic)
                    # Initialize topic relevance if new
                    if topic not in self.topic_relevance:
                        self.topic_relevance[topic] = 0.5  # Default mid-relevance
                        
        # Manage message count if exceeded
        if len(self.messages) > self.max_message_count:
            self._forget_least_relevant_messages()
            
        # Mark summary as outdated
        self.summary_last_updated = None
        
        return message_id
    
    def add_task(self, task_id: str, task_metadata: Dict[str, Any]) -> None:
        """
        Associate a task with this session.
        
        Args:
            task_id: The ID of the task
            task_metadata: Metadata about the task
        """
        self.tasks[task_id] = task_metadata
        self.last_active_at = datetime.utcnow()
        
    def update_topic_relevance(self, topic: str, relevance_delta: float) -> None:
        """
        Update the relevance score for a topic.
        
        Args:
            topic: The topic to update
            relevance_delta: Change in relevance (-1.0 to 1.0)
        """
        if topic not in self.topic_relevance:
            self.topic_relevance[topic] = 0.5  # Default mid-relevance
            
        # Update relevance, keeping within 0.0-1.0 range
        current = self.topic_relevance[topic]
        self.topic_relevance[topic] = max(0.0, min(1.0, current + relevance_delta))
        
    def update_message_relevance(self, message_id: str, relevance: MessageRelevance) -> bool:
        """
        Update the relevance level of a message.
        
        Args:
            message_id: ID of the message
            relevance: New relevance level
            
        Returns:
            True if successful, False if message not found
        """
        if message_id not in self.message_relevance:
            return False
            
        self.message_relevance[message_id] = relevance
        return True
        
    def is_expired(self) -> bool:
        """Check if the session has expired based on inactivity."""
        if self.status == SessionStatus.EXPIRED or self.status == SessionStatus.ARCHIVED:
            return True
            
        if self.max_inactive_time is None:
            return False
            
        inactive_seconds = (datetime.utcnow() - self.last_active_at).total_seconds()
        return inactive_seconds > self.max_inactive_time
        
    def activate(self) -> None:
        """Activate an idle session."""
        if self.status == SessionStatus.IDLE:
            self.status = SessionStatus.ACTIVE
            self.last_active_at = datetime.utcnow()
            logger.info(f"Session {self.session_id} activated")
            
    def idle(self) -> None:
        """Mark the session as idle."""
        if self.status == SessionStatus.ACTIVE:
            self.status = SessionStatus.IDLE
            logger.info(f"Session {self.session_id} marked as idle")
            
    def expire(self) -> None:
        """Mark the session as expired."""
        if self.status != SessionStatus.EXPIRED and self.status != SessionStatus.ARCHIVED:
            self.status = SessionStatus.EXPIRED
            logger.info(f"Session {self.session_id} marked as expired")
            
    def archive(self) -> None:
        """Archive the session for long-term storage."""
        self.status = SessionStatus.ARCHIVED
        logger.info(f"Session {self.session_id} archived")
        
    def terminate(self) -> None:
        """Terminate the session."""
        self.status = SessionStatus.TERMINATED
        logger.info(f"Session {self.session_id} terminated")
        
    def get_recent_messages(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent messages.
        
        Args:
            count: Number of messages to retrieve
            
        Returns:
            List of recent messages
        """
        return self.messages[-count:] if self.messages else []
        
    def get_relevant_messages(
        self, 
        min_relevance: MessageRelevance = MessageRelevance.MEDIUM,
        max_count: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get the most relevant messages in the session.
        
        Args:
            min_relevance: Minimum relevance level to include
            max_count: Maximum number of messages to return
            
        Returns:
            List of relevant messages
        """
        # Filter messages by relevance
        relevant_ids = [
            msg_id for msg_id, relevance in self.message_relevance.items()
            if relevance.value >= min_relevance.value
        ]
        
        # Get messages with these IDs
        relevant_messages = []
        for message in self.messages:
            if message["id"] in relevant_ids:
                relevant_messages.append(message)
                if len(relevant_messages) >= max_count:
                    break
                    
        return relevant_messages
        
    def search_messages(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for messages containing the query.
        
        Args:
            query: Text to search for
            
        Returns:
            List of matching messages
        """
        results = []
        for message in self.messages:
            # Search in content if present
            if "content" in message and isinstance(message["content"], str):
                if query.lower() in message["content"].lower():
                    results.append(message)
            # Search in text if present
            elif "text" in message and isinstance(message["text"], str):
                if query.lower() in message["text"].lower():
                    results.append(message)
            # Search in message JSON representation as fallback
            else:
                try:
                    message_str = json.dumps(message)
                    if query.lower() in message_str.lower():
                        results.append(message)
                except:
                    pass
                    
        return results
        
    def get_summary(self, force_update: bool = False) -> str:
        """
        Get a summary of the session.
        
        Args:
            force_update: Force regeneration of the summary
            
        Returns:
            Session summary
        """
        # Return existing summary if available and not forced to update
        if self.summary and self.summary_last_updated and not force_update:
            return self.summary
            
        # Generate a new summary
        summary = f"Session {self.name} between {self.agent_id} and {self.initiator_id}\n"
        summary += f"Status: {self.status.value}, "
        summary += f"Messages: {len(self.messages)}, "
        summary += f"Last active: {self.last_active_at.isoformat()}\n"
        
        if self.topics:
            summary += f"Topics: {', '.join(self.topics)}\n"
            
        # Add most common intents
        if self.intents:
            intent_counts = {}
            for intent in self.intents:
                if intent not in intent_counts:
                    intent_counts[intent] = 0
                intent_counts[intent] += 1
                
            # Get top 3 intents
            top_intents = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            summary += f"Top intents: {', '.join([f'{i[0]} ({i[1]})' for i in top_intents])}\n"
            
        # Summarize recent conversation
        recent = self.get_recent_messages(3)
        if recent:
            summary += "Recent exchanges:\n"
            for msg in recent:
                # Extract who said what
                sender = msg.get("sender_id", "Unknown")
                content = "..."
                if "content" in msg and isinstance(msg["content"], str):
                    content = msg["content"]
                elif "text" in msg and isinstance(msg["text"], str):
                    content = msg["text"]
                    
                # Truncate long content
                if len(content) > 50:
                    content = content[:47] + "..."
                    
                summary += f"- {sender}: {content}\n"
                
        # Update summary state
        self.summary = summary
        self.summary_last_updated = datetime.utcnow()
        
        return summary
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to a dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "initiator_id": self.initiator_id,
            "name": self.name,
            "metadata": self.metadata,
            "max_inactive_time": self.max_inactive_time,
            "max_message_count": self.max_message_count,
            "created_at": self.created_at.isoformat(),
            "last_active_at": self.last_active_at.isoformat(),
            "messages": self.messages,
            "message_relevance": {k: v.value for k, v in self.message_relevance.items()},
            "status": self.status.value,
            "topics": self.topics,
            "topic_relevance": self.topic_relevance,
            "entities": self.entities,
            "intents": self.intents,
            "last_intent": self.last_intent,
            "summary": self.summary,
            "summary_last_updated": self.summary_last_updated.isoformat() if self.summary_last_updated else None,
            "tasks": self.tasks,
            "context_data": self.context_data
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """
        Create a Session from a dictionary.
        
        Args:
            data: Session data dictionary
            
        Returns:
            Session object
        """
        session = cls(
            session_id=data["session_id"],
            agent_id=data["agent_id"],
            initiator_id=data["initiator_id"],
            name=data.get("name"),
            metadata=data.get("metadata", {}),
            max_inactive_time=data.get("max_inactive_time"),
            max_message_count=data.get("max_message_count")
        )
        
        # Restore created/updated times
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.last_active_at = datetime.fromisoformat(data["last_active_at"])
        
        # Restore messages
        session.messages = data.get("messages", [])
        
        # Restore message relevance
        relevance_dict = data.get("message_relevance", {})
        session.message_relevance = {
            k: MessageRelevance(v) for k, v in relevance_dict.items()
        }
        
        # Restore status
        session.status = SessionStatus(data.get("status", "active"))
        
        # Restore topics
        session.topics = data.get("topics", [])
        session.topic_relevance = data.get("topic_relevance", {})
        
        # Restore other fields
        session.entities = data.get("entities", {})
        session.intents = data.get("intents", [])
        session.last_intent = data.get("last_intent")
        session.summary = data.get("summary")
        if data.get("summary_last_updated"):
            session.summary_last_updated = datetime.fromisoformat(data["summary_last_updated"])
        session.tasks = data.get("tasks", {})
        session.context_data = data.get("context_data", {})
        
        return session
        
    def _forget_least_relevant_messages(self) -> None:
        """
        Forget the least relevant messages to stay within max_message_count.
        
        This implements the forgetting mechanism to avoid context overflow.
        Critical messages are always preserved.
        """
        # If we're under the limit, no action needed
        if len(self.messages) <= self.max_message_count:
            return
            
        # Build a list of (relevance_value, index, message_id) tuples
        message_relevance_with_index = []
        for i, message in enumerate(self.messages):
            msg_id = message["id"]
            # Default to LOW relevance if not specifically set
            relevance = self.message_relevance.get(msg_id, MessageRelevance.LOW)
            # Only consider non-CRITICAL messages for forgetting
            if relevance != MessageRelevance.CRITICAL:
                message_relevance_with_index.append((relevance.value, i, msg_id))
                
        # Sort by relevance (ascending) and then by index (ascending)
        # This will put the least relevant, oldest messages first
        message_relevance_with_index.sort()
        
        # Determine how many messages to forget
        excess = len(self.messages) - self.max_message_count
        to_forget = min(excess, len(message_relevance_with_index))
        
        if to_forget <= 0:
            return
            
        # Get the indices to remove (in descending order to avoid shifting)
        indices_to_remove = sorted([item[1] for item in message_relevance_with_index[:to_forget]], reverse=True)
        
        # Remove messages and their relevance entries
        for index in indices_to_remove:
            msg_id = self.messages[index]["id"]
            del self.messages[index]
            if msg_id in self.message_relevance:
                del self.message_relevance[msg_id]
                
        logger.info(f"Forgot {to_forget} least relevant messages in session {self.session_id}")


class SessionManager:
    """
    Manages conversation sessions between agents.
    
    This class provides methods for:
    1. Creating and tracking sessions
    2. Maintaining conversation context across multiple interactions
    3. Storing and retrieving relevant history
    4. Implementing forgetting mechanisms
    5. Balancing immediate context with long-term relationship data
    """
    
    def __init__(
        self,
        agent_id: str,
        relationship_tracker: Optional[RelationshipTracker] = None,
        storage_dir: str = "data/sessions",
        max_active_sessions: int = 100,
        default_session_timeout: int = 3600,  # 1 hour in seconds
        default_max_messages: int = 100,
        cleanup_interval: int = 3600  # 1 hour in seconds
    ):
        """
        Initialize the session manager.
        
        Args:
            agent_id: ID of the agent this session manager belongs to
            relationship_tracker: Optional relationship tracker for long-term knowledge
            storage_dir: Directory to store session data
            max_active_sessions: Maximum number of active sessions to maintain
            default_session_timeout: Default timeout for sessions in seconds
            default_max_messages: Default maximum messages per session
            cleanup_interval: Interval for session cleanup in seconds
        """
        self.agent_id = agent_id
        self.relationship_tracker = relationship_tracker
        self.storage_dir = storage_dir
        self.max_active_sessions = max_active_sessions
        self.default_session_timeout = default_session_timeout
        self.default_max_messages = default_max_messages
        self.cleanup_interval = cleanup_interval
        
        # Session storage
        self.active_sessions: Dict[str, Session] = {}  # session_id -> Session
        self.agent_sessions: Dict[str, List[str]] = {}  # agent_id -> [session_ids]
        
        # Tracking for session cleanup
        self.last_cleanup = datetime.utcnow()
        
        # Session retrieval indices
        self._session_topic_index: Dict[str, List[str]] = {}  # topic -> [session_ids]
        
        # Ensure storage directory exists
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load existing sessions
        self._load_sessions()
        
        logger.info(f"SessionManager initialized for agent {agent_id} with {len(self.active_sessions)} sessions")
    
    def create_session(
        self,
        initiator_id: str,
        session_id: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_inactive_time: Optional[int] = None,
        max_message_count: Optional[int] = None
    ) -> Session:
        """
        Create a new session.
        
        Args:
            initiator_id: ID of the agent initiating the session
            session_id: Optional custom session ID
            name: Optional session name
            metadata: Additional session metadata
            max_inactive_time: Maximum inactivity time in seconds
            max_message_count: Maximum number of messages to store
            
        Returns:
            The created session
        """
        # Generate session ID if not provided
        if not session_id:
            session_id = f"session-{uuid.uuid4().hex}"
            
        # Use default values if not specified
        if max_inactive_time is None:
            max_inactive_time = self.default_session_timeout
            
        if max_message_count is None:
            max_message_count = self.default_max_messages
            
        # Create new session
        session = Session(
            session_id=session_id,
            agent_id=self.agent_id,
            initiator_id=initiator_id,
            name=name,
            metadata=metadata,
            max_inactive_time=max_inactive_time,
            max_message_count=max_message_count
        )
        
        # Store session
        self.active_sessions[session_id] = session
        
        # Update agent sessions index
        if initiator_id not in self.agent_sessions:
            self.agent_sessions[initiator_id] = []
            
        self.agent_sessions[initiator_id].append(session_id)
        
        # Handle session limit
        if len(self.active_sessions) > self.max_active_sessions:
            self._expire_least_active_sessions()
            
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID.
        
        Args:
            session_id: ID of the session to get
            
        Returns:
            Session if found, None otherwise
        """
        # Check if in active sessions
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # Check if expired
            if session.is_expired():
                session.expire()
                
            return session
            
        # Try to load from storage
        session = self._load_session(session_id)
        if session:
            # Add to active sessions
            self.active_sessions[session_id] = session
            return session
            
        return None
    
    def get_or_create_session(
        self,
        initiator_id: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Session:
        """
        Get an existing session or create a new one.
        
        Args:
            initiator_id: ID of the initiating agent
            session_id: Optional session ID to retrieve
            metadata: Metadata for new session if created
            
        Returns:
            The retrieved or created session
        """
        # If session ID provided, try to get it
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
                
        # Find most recent active session with this agent
        agent_session_ids = self.agent_sessions.get(initiator_id, [])
        for sess_id in reversed(agent_session_ids):  # Reverse to check most recent first
            session = self.get_session(sess_id)
            if session and session.status == SessionStatus.ACTIVE:
                return session
                
        # Create new session if none found
        return self.create_session(
            initiator_id=initiator_id,
            session_id=session_id,
            metadata=metadata
        )
    
    def find_sessions_by_topic(self, topic: str) -> List[Session]:
        """
        Find sessions related to a specific topic.
        
        Args:
            topic: The topic to search for
            
        Returns:
            List of sessions related to the topic
        """
        session_ids = self._session_topic_index.get(topic, [])
        matching_sessions = []
        
        for session_id in session_ids:
            session = self.get_session(session_id)
            if session and session.status != SessionStatus.TERMINATED:
                matching_sessions.append(session)
                
        return matching_sessions
    
    def find_sessions_by_agent(
        self, 
        agent_id: str,
        active_only: bool = False
    ) -> List[Session]:
        """
        Find sessions with a specific agent.
        
        Args:
            agent_id: ID of the agent
            active_only: Only include active sessions
            
        Returns:
            List of sessions with the agent
        """
        session_ids = self.agent_sessions.get(agent_id, [])
        matching_sessions = []
        
        for session_id in session_ids:
            session = self.get_session(session_id)
            if session:
                if not active_only or session.status == SessionStatus.ACTIVE:
                    matching_sessions.append(session)
                    
        return matching_sessions
    
    def search_sessions(self, query: str) -> List[Tuple[Session, List[Dict[str, Any]]]]:
        """
        Search across all sessions for content matching the query.
        
        Args:
            query: Text to search for
            
        Returns:
            List of (session, matching_messages) tuples
        """
        results = []
        
        for session in self.active_sessions.values():
            # Skip terminated or archived sessions
            if session.status == SessionStatus.TERMINATED or session.status == SessionStatus.ARCHIVED:
                continue
                
            # Search in this session
            matching_messages = session.search_messages(query)
            if matching_messages:
                results.append((session, matching_messages))
                
        return results
    
    def archive_session(self, session_id: str) -> bool:
        """
        Archive a session for long-term storage.
        
        Args:
            session_id: ID of the session to archive
            
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session(session_id)
        if not session:
            return False
            
        # Archive the session
        session.archive()
        
        # Save to permanent storage
        self._save_session(session)
        
        # Remove from active sessions
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            
        return True
    
    def terminate_session(self, session_id: str) -> bool:
        """
        Terminate a session permanently.
        
        Args:
            session_id: ID of the session to terminate
            
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session(session_id)
        if not session:
            return False
            
        # Terminate the session
        session.terminate()
        
        # Save final state
        self._save_session(session)
        
        # Remove from active sessions
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            
        # Update indices
        self._update_session_indices(session, remove=True)
        
        return True
    
    def add_message_to_session(
        self,
        session_id: str,
        message: Dict[str, Any],
        intent: Optional[str] = None,
        relevance: MessageRelevance = MessageRelevance.MEDIUM
    ) -> Optional[str]:
        """
        Add a message to a session.
        
        Args:
            session_id: ID of the session
            message: Message to add
            intent: Optional intent of the message
            relevance: Relevance level of the message
            
        Returns:
            Message ID if successful, None otherwise
        """
        session = self.get_session(session_id)
        if not session:
            return None
            
        # Add message to session
        message_id = session.add_message(message, intent, relevance)
        
        # Update indices
        self._update_session_indices(session)
        
        # Update relationship data if available
        if self.relationship_tracker and "sender_id" in message:
            sender_id = message["sender_id"]
            quality = InteractionQuality.NEUTRAL
            
            # Record the interaction
            self.relationship_tracker.record_interaction(
                agent_id=sender_id,
                interaction_type=InteractionType.MESSAGE,
                content_summary=self._summarize_message(message),
                quality=quality,
                metadata={
                    "session_id": session_id,
                    "message_id": message_id,
                    "intent": intent
                }
            )
            
        # Save session state periodically
        self._save_session_if_needed(session)
        
        return message_id
    
    def get_relevant_context(
        self,
        session_id: str,
        max_messages: int = 10,
        include_relationship_data: bool = True
    ) -> Dict[str, Any]:
        """
        Get relevant context for a session, combining immediate context with relationship data.
        
        Args:
            session_id: ID of the session
            max_messages: Maximum number of messages to include
            include_relationship_data: Whether to include relationship data
            
        Returns:
            Dictionary with relevant context
        """
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}
            
        # Get session summary
        summary = session.get_summary()
        
        # Get relevant messages
        relevant_messages = session.get_relevant_messages(
            min_relevance=MessageRelevance.MEDIUM,
            max_count=max_messages
        )
        
        # Start building context
        context = {
            "session_summary": summary,
            "recent_messages": relevant_messages,
            "topics": list(session.topic_relevance.items()),
            "tasks": list(session.tasks.values())
        }
        
        # Add relationship data if available and requested
        if include_relationship_data and self.relationship_tracker:
            relationship = self.relationship_tracker.get_relationship(session.initiator_id)
            if relationship:
                context["relationship"] = {
                    "trust_level": relationship.trust_level.name,
                    "interaction_count": relationship.interaction_count,
                    "last_interaction": relationship.last_interaction.isoformat() if relationship.last_interaction else None,
                    "notes": relationship.notes
                }
                
                # Add recent interactions
                recent_interactions = self.relationship_tracker.get_recent_interactions(
                    agent_id=session.initiator_id,
                    count=5
                )
                
                if recent_interactions:
                    context["recent_interactions"] = recent_interactions
                    
        return context
    
    def add_task_to_session(
        self,
        session_id: str,
        task_id: str,
        task_metadata: Dict[str, Any]
    ) -> bool:
        """
        Add a task to a session.
        
        Args:
            session_id: ID of the session
            task_id: ID of the task
            task_metadata: Metadata about the task
            
        Returns:
            True if successful, False otherwise
        """
        session = self.get_session(session_id)
        if not session:
            return False
            
        # Add task to session
        session.add_task(task_id, task_metadata)
        
        # Save session state
        self._save_session_if_needed(session)
        
        return True
    
    def get_session_tasks(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get tasks associated with a session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            List of tasks associated with the session
        """
        session = self.get_session(session_id)
        if not session:
            return []
            
        return list(session.tasks.values())
    
    def get_sessions_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about all sessions.
        
        Returns:
            Dictionary with session statistics
        """
        total_sessions = len(self.active_sessions)
        active_count = 0
        idle_count = 0
        expired_count = 0
        archived_count = 0
        terminated_count = 0
        total_messages = 0
        
        for session in self.active_sessions.values():
            if session.status == SessionStatus.ACTIVE:
                active_count += 1
            elif session.status == SessionStatus.IDLE:
                idle_count += 1
            elif session.status == SessionStatus.EXPIRED:
                expired_count += 1
            elif session.status == SessionStatus.ARCHIVED:
                archived_count += 1
            elif session.status == SessionStatus.TERMINATED:
                terminated_count += 1
                
            total_messages += len(session.messages)
            
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_count,
            "idle_sessions": idle_count,
            "expired_sessions": expired_count,
            "archived_sessions": archived_count,
            "terminated_sessions": terminated_count,
            "total_messages": total_messages,
            "agent_count": len(self.agent_sessions)
        }
    
    def cleanup(self) -> Dict[str, int]:
        """
        Perform cleanup of expired sessions.
        
        Returns:
            Dictionary with cleanup statistics
        """
        # Check if cleanup is due
        now = datetime.utcnow()
        if (now - self.last_cleanup).total_seconds() < self.cleanup_interval:
            return {"expired": 0, "archived": 0, "saved": 0}
            
        expired_count = 0
        archived_count = 0
        saved_count = 0
        
        # Check each session
        for session_id, session in list(self.active_sessions.items()):
            # Check if expired
            if session.is_expired() and session.status != SessionStatus.EXPIRED:
                session.expire()
                expired_count += 1
                
                # Save expired session
                self._save_session(session)
                saved_count += 1
                
            # Archive sessions that have been expired for a long time
            if session.status == SessionStatus.EXPIRED:
                inactive_time = (now - session.last_active_at).total_seconds()
                # Archive after 30 days of inactivity
                if inactive_time > 30 * 24 * 3600:
                    session.archive()
                    archived_count += 1
                    
                    # Remove from active sessions
                    del self.active_sessions[session_id]
                    
                    # Save archived session
                    self._save_session(session)
                    saved_count += 1
                    
        # Update last cleanup time
        self.last_cleanup = now
        
        return {
            "expired": expired_count,
            "archived": archived_count,
            "saved": saved_count
        }
    
    def _save_session_if_needed(self, session: Session) -> None:
        """
        Save a session if needed (based on message count or time since last save).
        
        Args:
            session: The session to save
        """
        # Check if enough time has passed or enough messages have been added
        session_file = os.path.join(self.storage_dir, f"{session.session_id}.json")
        
        should_save = False
        
        # Save if file doesn't exist
        if not os.path.exists(session_file):
            should_save = True
        # Otherwise, check message count or time
        else:
            # Check if it's been at least 5 minutes since last modification
            last_modified = os.path.getmtime(session_file)
            if (time.time() - last_modified) > 300:  # 5 minutes in seconds
                should_save = True
                
        if should_save:
            self._save_session(session)
    
    def _save_session(self, session: Session) -> None:
        """
        Save a session to disk.
        
        Args:
            session: The session to save
        """
        try:
            # Convert session to dictionary
            session_data = session.to_dict()
            
            # Save to JSON file
            file_path = os.path.join(self.storage_dir, f"{session.session_id}.json")
            with open(file_path, 'w') as f:
                json.dump(session_data, f, indent=2)
                
            logger.debug(f"Saved session {session.session_id} to {file_path}")
        except Exception as e:
            logger.error(f"Error saving session {session.session_id}: {str(e)}")
    
    def _load_session(self, session_id: str) -> Optional[Session]:
        """
        Load a session from disk.
        
        Args:
            session_id: ID of the session to load
            
        Returns:
            Session if found and loaded, None otherwise
        """
        try:
            file_path = os.path.join(self.storage_dir, f"{session_id}.json")
            if not os.path.exists(file_path):
                return None
                
            with open(file_path, 'r') as f:
                session_data = json.load(f)
                
            # Create session from data
            session = Session.from_dict(session_data)
            
            logger.debug(f"Loaded session {session_id} from {file_path}")
            return session
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {str(e)}")
            return None
    
    def _load_sessions(self) -> None:
        """Load existing sessions from disk."""
        try:
            for filename in os.listdir(self.storage_dir):
                if filename.endswith('.json'):
                    session_id = filename[:-5]  # Remove .json extension
                    
                    # Only load a limited number of active sessions
                    if len(self.active_sessions) >= self.max_active_sessions:
                        break
                        
                    session = self._load_session(session_id)
                    if session:
                        # Skip terminated sessions
                        if session.status == SessionStatus.TERMINATED:
                            continue
                            
                        # Add to active sessions
                        self.active_sessions[session_id] = session
                        
                        # Update agent sessions index
                        initiator_id = session.initiator_id
                        if initiator_id not in self.agent_sessions:
                            self.agent_sessions[initiator_id] = []
                            
                        if session_id not in self.agent_sessions[initiator_id]:
                            self.agent_sessions[initiator_id].append(session_id)
                            
                        # Update indices
                        self._update_session_indices(session)
                        
            logger.info(f"Loaded {len(self.active_sessions)} sessions from {self.storage_dir}")
        except Exception as e:
            logger.error(f"Error loading sessions: {str(e)}")
    
    def _update_session_indices(self, session: Session, remove: bool = False) -> None:
        """
        Update session indices for efficient retrieval.
        
        Args:
            session: The session to update indices for
            remove: Whether to remove session from indices instead of adding
        """
        # Update topic index
        for topic in session.topics:
            if topic not in self._session_topic_index:
                self._session_topic_index[topic] = []
                
            if remove:
                if session.session_id in self._session_topic_index[topic]:
                    self._session_topic_index[topic].remove(session.session_id)
            else:
                if session.session_id not in self._session_topic_index[topic]:
                    self._session_topic_index[topic].append(session.session_id)
    
    def _expire_least_active_sessions(self) -> None:
        """Expire the least active sessions to stay within max_active_sessions."""
        # If under the limit, no action needed
        if len(self.active_sessions) <= self.max_active_sessions:
            return
            
        # Find least recently active sessions, prioritizing those already idle/expired
        sessions_with_activity = []
        for session_id, session in self.active_sessions.items():
            # Skip already archived or terminated sessions
            if session.status in [SessionStatus.ARCHIVED, SessionStatus.TERMINATED]:
                continue
                
            # Priority based on status and last active time
            priority = 0
            if session.status == SessionStatus.EXPIRED:
                priority = 1
            elif session.status == SessionStatus.IDLE:
                priority = 2
            else:  # ACTIVE
                priority = 3
                
            sessions_with_activity.append(
                (priority, session.last_active_at, session_id, session)
            )
            
        # Sort by priority and activity time
        sessions_with_activity.sort()
        
        # Determine how many to expire
        excess = len(self.active_sessions) - self.max_active_sessions
        to_expire = min(excess, len(sessions_with_activity))
        
        # Expire oldest, least important sessions
        for i in range(to_expire):
            _, _, session_id, session = sessions_with_activity[i]
            
            # Mark as expired if not already
            if session.status != SessionStatus.EXPIRED:
                session.expire()
                
            # Save to disk
            self._save_session(session)
            
        logger.info(f"Expired {to_expire} least active sessions")
    
    def _summarize_message(self, message: Dict[str, Any]) -> str:
        """
        Create a summary of a message for tracking.
        
        Args:
            message: The message to summarize
            
        Returns:
            A string summary of the message
        """
        # Extract content from different possible fields
        content = "..."
        if "content" in message and isinstance(message["content"], str):
            content = message["content"]
        elif "text" in message and isinstance(message["text"], str):
            content = message["text"]
        elif "data" in message:
            try:
                content = json.dumps(message["data"])
            except:
                content = str(message["data"])
                
        # Truncate long content
        if len(content) > 100:
            content = content[:97] + "..."
            
        return content
