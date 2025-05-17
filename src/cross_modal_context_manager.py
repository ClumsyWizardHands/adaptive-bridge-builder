#!/usr/bin/env python3
"""
Cross-Modal Context Manager

This module provides capabilities for maintaining continuous conversation context
across different communication channels, linking related interactions, and ensuring
context preservation when switching between modalities.
"""

import logging
import json
import uuid
import time
import re
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from communication_channel_manager import (
    CommunicationChannelManager, 
    ChannelType,
    ChannelMessage
)
from session_manager import SessionManager, Session, MessageRelevance
from human_interaction_styler import HumanInteractionStyler
from principle_engine import PrincipleEngine
from relationship_tracker import RelationshipTracker, InteractionType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CrossModalContextManager")


class ContextSensitivity(Enum):
    """Sensitivity levels for different types of context."""
    HIGH = "high"          # Highly sensitive/private information
    MEDIUM = "medium"      # Moderately sensitive information
    LOW = "low"            # Low sensitivity, can be shared broadly
    PUBLIC = "public"      # Public information, no sensitivity


class ModalityTransition(Enum):
    """Types of transitions between communication modalities."""
    EMAIL_TO_CHAT = "email_to_chat"
    CHAT_TO_EMAIL = "chat_to_email"
    EMAIL_TO_API = "email_to_api"
    API_TO_EMAIL = "api_to_email"
    CHAT_TO_API = "chat_to_api"
    API_TO_CHAT = "api_to_chat"
    SAME_MODALITY = "same_modality"
    OTHER = "other"
    
    @classmethod
    def from_channel_types(cls, from_type: ChannelType, to_type: ChannelType) -> 'ModalityTransition':
        """Get the modality transition based on channel types."""
        if from_type == to_type:
            return cls.SAME_MODALITY
            
        transition_name = f"{from_type.value}_to_{to_type.value}"
        for transition in cls:
            if transition.value == transition_name:
                return transition
                
        return cls.OTHER


@dataclass
class IdentityLink:
    """
    Links together different identities for the same entity across channels.
    
    This allows tracking a single human user who may have different identifiers
    depending on the communication channel (e.g., email address, chat username).
    """
    
    primary_id: str  # Primary identifier for the entity
    channel_identities: Dict[ChannelType, str] = field(default_factory=dict)  # Channel-specific identifiers
    verification_level: int = 0  # 0=unverified, 1=probable, 2=verified
    last_verification: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_channel_identity(
        self, 
        channel_type: ChannelType, 
        identity: str, 
        verified: bool = False
    ) -> None:
        """
        Add a channel-specific identity for this entity.
        
        Args:
            channel_type: The channel type
            identity: The identity on this channel
            verified: Whether this identity is verified
        """
        self.channel_identities[channel_type] = identity
        if verified and self.verification_level < 2:
            self.verification_level = 2
            self.last_verification = datetime.utcnow()
    
    def get_channel_identity(self, channel_type: ChannelType) -> Optional[str]:
        """
        Get the identity for a specific channel type.
        
        Args:
            channel_type: The channel type
            
        Returns:
            The identity for this channel if available, None otherwise
        """
        return self.channel_identities.get(channel_type)
    
    def get_all_identities(self) -> List[str]:
        """
        Get all identities for this entity.
        
        Returns:
            List of all identities including the primary ID
        """
        return [self.primary_id] + list(self.channel_identities.values())


@dataclass
class ContextLink:
    """
    Links together related context elements across different channels and sessions.
    
    This enables tracking related conversations even when they span different
    communication modes.
    """
    
    link_id: str
    primary_topic: str
    session_ids: List[str] = field(default_factory=list)
    message_ids: List[str] = field(default_factory=list)
    entity_ids: List[str] = field(default_factory=list)
    creation_time: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    status: str = "active"  # active, archived, expired
    metadata: Dict[str, Any] = field(default_factory=dict)
    sensitivity: ContextSensitivity = ContextSensitivity.MEDIUM
    expiry_time: Optional[datetime] = None
    
    def add_session(self, session_id: str) -> None:
        """Add a session to this context link."""
        if session_id not in self.session_ids:
            self.session_ids.append(session_id)
            self.last_updated = datetime.utcnow()
    
    def add_message(self, message_id: str) -> None:
        """Add a message to this context link."""
        if message_id not in self.message_ids:
            self.message_ids.append(message_id)
            self.last_updated = datetime.utcnow()
    
    def add_entity(self, entity_id: str) -> None:
        """Add an entity to this context link."""
        if entity_id not in self.entity_ids:
            self.entity_ids.append(entity_id)
            self.last_updated = datetime.utcnow()
            
    def is_expired(self) -> bool:
        """Check if this context link has expired."""
        if not self.expiry_time:
            return False
            
        return datetime.utcnow() >= self.expiry_time


class CrossModalContextManager:
    """
    Manages continuous conversation context across different communication channels.
    
    This class enables the agent to maintain context when conversations transition
    between different communication modalities, ensuring a seamless experience
    while respecting privacy and security requirements.
    
    Key capabilities:
    1. Maintains continuous conversation context regardless of the communication channel
    2. Links related interactions across different modalities (e.g., email to chat)
    3. Recognizes when a new interaction is related to previous ones
    4. Provides relevant history when switching modes of communication
    5. Respects privacy by maintaining appropriate separation between contexts
    6. Implements "Trust as the Foundation of Leadership" principle in managing sensitive information
    """
    
    def __init__(
        self,
        agent_id: str,
        session_manager: SessionManager,
        channel_manager: Optional[CommunicationChannelManager] = None,
        human_interaction_styler: Optional[HumanInteractionStyler] = None,
        principle_engine: Optional[PrincipleEngine] = None,
        relationship_tracker: Optional[RelationshipTracker] = None,
        context_expiry_days: int = 30,
        max_context_links: int = 10000
    ):
        """
        Initialize the CrossModalContextManager.
        
        Args:
            agent_id: ID of the agent using this manager
            session_manager: Manager for conversation sessions
            channel_manager: Manager for communication channels
            human_interaction_styler: Human interaction preference manager
            principle_engine: Engine for principle-aligned actions
            relationship_tracker: Tracker for long-term relationships
            context_expiry_days: Days after which context links expire
            max_context_links: Maximum number of context links to maintain
        """
        self.agent_id = agent_id
        self.session_manager = session_manager
        self.channel_manager = channel_manager
        self.human_interaction_styler = human_interaction_styler
        self.principle_engine = principle_engine
        self.relationship_tracker = relationship_tracker
        self.context_expiry_days = context_expiry_days
        self.max_context_links = max_context_links
        
        # Identity mapping
        self.identity_links: Dict[str, IdentityLink] = {}
        
        # Context linking
        self.context_links: Dict[str, ContextLink] = {}
        self.topic_index: Dict[str, List[str]] = {}  # topic -> [link_ids]
        self.entity_index: Dict[str, List[str]] = {}  # entity_id -> [link_ids]
        self.session_index: Dict[str, List[str]] = {}  # session_id -> [link_ids]
        self.message_index: Dict[str, str] = {}  # message_id -> link_id
        
        # Transition templates for different modality transitions
        self.transition_templates: Dict[ModalityTransition, str] = {
            ModalityTransition.EMAIL_TO_CHAT: "Continuing our email conversation about {topic} from {date}.",
            ModalityTransition.CHAT_TO_EMAIL: "I'm following up on our chat conversation about {topic} from {date}.",
            ModalityTransition.EMAIL_TO_API: "This request relates to our email discussion about {topic} from {date}.",
            ModalityTransition.API_TO_EMAIL: "I'm sending this email regarding the API request about {topic} from {date}.",
            ModalityTransition.CHAT_TO_API: "This API request continues our chat about {topic} from {date}.",
            ModalityTransition.API_TO_CHAT: "This relates to your API request about {topic} from {date}.",
            ModalityTransition.SAME_MODALITY: "Regarding our previous conversation about {topic} from {date}.",
            ModalityTransition.OTHER: "Continuing our discussion about {topic} from {date}."
        }
        
        # Privacy and security settings
        self.default_sensitivity = ContextSensitivity.MEDIUM
        self.channel_sensitivity_overrides: Dict[ChannelType, ContextSensitivity] = {}
        self.entity_sensitivity_preferences: Dict[str, ContextSensitivity] = {}
        
        # Recent activities for quick access
        self.recent_transitions: List[Dict[str, Any]] = []
        
        logger.info(f"CrossModalContextManager initialized for agent {agent_id}")
    
    def link_identity(
        self,
        primary_id: str,
        channel_type: ChannelType,
        channel_identity: str,
        verified: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Link a channel-specific identity to a primary identity.
        
        Args:
            primary_id: Primary identifier for the entity
            channel_type: The channel type
            channel_identity: The identity on this channel
            verified: Whether this identity link is verified
            metadata: Additional metadata about this identity link
        """
        if primary_id not in self.identity_links:
            self.identity_links[primary_id] = IdentityLink(
                primary_id=primary_id,
                verification_level=2 if verified else 1,
                last_verification=datetime.utcnow() if verified else None,
                metadata=metadata or {}
            )
            
        self.identity_links[primary_id].add_channel_identity(
            channel_type=channel_type,
            identity=channel_identity,
            verified=verified
        )
        
        # Update metadata if provided
        if metadata:
            for key, value in metadata.items():
                self.identity_links[primary_id].metadata[key] = value
                
        logger.info(f"Linked {channel_type} identity {channel_identity} to {primary_id}")
    
    def get_primary_id(
        self,
        channel_type: ChannelType,
        channel_identity: str
    ) -> Optional[str]:
        """
        Get the primary ID for a channel-specific identity.
        
        Args:
            channel_type: The channel type
            channel_identity: The identity on this channel
            
        Returns:
            The primary ID if found, None otherwise
        """
        for primary_id, identity_link in self.identity_links.items():
            if identity_link.get_channel_identity(channel_type) == channel_identity:
                return primary_id
                
        return None
    
    def get_channel_identity(
        self,
        primary_id: str,
        channel_type: ChannelType
    ) -> Optional[str]:
        """
        Get the channel-specific identity for an entity.
        
        Args:
            primary_id: Primary identifier for the entity
            channel_type: The channel type
            
        Returns:
            The channel-specific identity if found, None otherwise
        """
        if primary_id not in self.identity_links:
            return None
            
        return self.identity_links[primary_id].get_channel_identity(channel_type)
    
    def create_context_link(
        self,
        primary_topic: str,
        entity_ids: List[str],
        session_ids: Optional[List[str]] = None,
        message_ids: Optional[List[str]] = None,
        sensitivity: Optional[ContextSensitivity] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expiry_days: Optional[int] = None
    ) -> str:
        """
        Create a new context link to associate related interactions.
        
        Args:
            primary_topic: The main topic of this context
            entity_ids: IDs of entities involved
            session_ids: IDs of related sessions
            message_ids: IDs of related messages
            sensitivity: Sensitivity level of this context
            metadata: Additional metadata
            expiry_days: Days until this context link expires
            
        Returns:
            ID of the created context link
        """
        # Use default sensitivity if not specified
        if sensitivity is None:
            sensitivity = self.default_sensitivity
            
        # Generate a unique ID for this context link
        link_id = f"ctx-{uuid.uuid4().hex}"
        
        # Calculate expiry time if specified
        expiry_time = None
        if expiry_days is not None:
            expiry_time = datetime.utcnow() + timedelta(days=expiry_days)
        elif self.context_expiry_days:
            expiry_time = datetime.utcnow() + timedelta(days=self.context_expiry_days)
            
        # Create the context link
        context_link = ContextLink(
            link_id=link_id,
            primary_topic=primary_topic,
            entity_ids=entity_ids.copy(),
            session_ids=session_ids.copy() if session_ids else [],
            message_ids=message_ids.copy() if message_ids else [],
            metadata=metadata.copy() if metadata else {},
            sensitivity=sensitivity,
            expiry_time=expiry_time
        )
        
        # Store the context link
        self.context_links[link_id] = context_link
        
        # Update indices
        if primary_topic not in self.topic_index:
            self.topic_index[primary_topic] = []
        self.topic_index[primary_topic].append(link_id)
        
        for entity_id in entity_ids:
            if entity_id not in self.entity_index:
                self.entity_index[entity_id] = []
            self.entity_index[entity_id].append(link_id)
            
        if session_ids:
            for session_id in session_ids:
                if session_id not in self.session_index:
                    self.session_index[session_id] = []
                self.session_index[session_id].append(link_id)
                
        if message_ids:
            for message_id in message_ids:
                self.message_index[message_id] = link_id
                
        # Check if we're over the limit and clean up if needed
        if len(self.context_links) > self.max_context_links:
            self._cleanup_oldest_context_links()
            
        logger.info(f"Created context link {link_id} for topic '{primary_topic}'")
        return link_id
    
    def add_to_context_link(
        self,
        link_id: str,
        session_ids: Optional[List[str]] = None,
        message_ids: Optional[List[str]] = None,
        entity_ids: Optional[List[str]] = None
    ) -> bool:
        """
        Add elements to an existing context link.
        
        Args:
            link_id: ID of the context link
            session_ids: Session IDs to add
            message_ids: Message IDs to add
            entity_ids: Entity IDs to add
            
        Returns:
            Whether the update was successful
        """
        if link_id not in self.context_links:
            logger.warning(f"Context link not found: {link_id}")
            return False
            
        context_link = self.context_links[link_id]
        
        # Check if expired
        if context_link.is_expired():
            logger.warning(f"Cannot add to expired context link: {link_id}")
            return False
            
        # Add sessions
        if session_ids:
            for session_id in session_ids:
                context_link.add_session(session_id)
                if session_id not in self.session_index:
                    self.session_index[session_id] = []
                if link_id not in self.session_index[session_id]:
                    self.session_index[session_id].append(link_id)
                    
        # Add messages
        if message_ids:
            for message_id in message_ids:
                context_link.add_message(message_id)
                self.message_index[message_id] = link_id
                
        # Add entities
        if entity_ids:
            for entity_id in entity_ids:
                context_link.add_entity(entity_id)
                if entity_id not in self.entity_index:
                    self.entity_index[entity_id] = []
                if link_id not in self.entity_index[entity_id]:
                    self.entity_index[entity_id].append(link_id)
                    
        return True
    
    def get_context_links_by_entity(
        self,
        entity_id: str,
        active_only: bool = True
    ) -> List[ContextLink]:
        """
        Get context links involving a specific entity.
        
        Args:
            entity_id: ID of the entity
            active_only: Whether to include only active context links
            
        Returns:
            List of relevant context links
        """
        # Check identity links to find all identifiers for this entity
        all_identities = [entity_id]
        if entity_id in self.identity_links:
            all_identities.extend(self.identity_links[entity_id].get_all_identities())
            
        # Get unique context link IDs for all identities
        link_ids = set()
        for identity in all_identities:
            if identity in self.entity_index:
                link_ids.update(self.entity_index[identity])
                
        # Get context links
        result = []
        for link_id in link_ids:
            if link_id in self.context_links:
                context_link = self.context_links[link_id]
                if not active_only or (context_link.status == "active" and not context_link.is_expired()):
                    result.append(context_link)
                    
        # Sort by last updated (most recent first)
        result.sort(key=lambda x: x.last_updated, reverse=True)
        return result
    
    def get_context_links_by_topic(
        self,
        topic: str,
        active_only: bool = True
    ) -> List[ContextLink]:
        """
        Get context links related to a specific topic.
        
        Args:
            topic: The topic to search for
            active_only: Whether to include only active context links
            
        Returns:
            List of relevant context links
        """
        link_ids = self.topic_index.get(topic, [])
        
        # Also check for similar topics
        for indexed_topic in self.topic_index:
            if topic.lower() in indexed_topic.lower() or indexed_topic.lower() in topic.lower():
                link_ids.extend(self.topic_index[indexed_topic])
                
        # Get context links
        result = []
        for link_id in set(link_ids):  # Use set to deduplicate
            if link_id in self.context_links:
                context_link = self.context_links[link_id]
                if not active_only or (context_link.status == "active" and not context_link.is_expired()):
                    result.append(context_link)
                    
        # Sort by last updated (most recent first)
        result.sort(key=lambda x: x.last_updated, reverse=True)
        return result
    
    def get_context_link_by_message(self, message_id: str) -> Optional[ContextLink]:
        """
        Get the context link associated with a specific message.
        
        Args:
            message_id: ID of the message
            
        Returns:
            Associated context link if found, None otherwise
        """
        link_id = self.message_index.get(message_id)
        if not link_id or link_id not in self.context_links:
            return None
            
        return self.context_links[link_id]
    
    def find_related_context(
        self,
        entity_id: str,
        topic: Optional[str] = None,
        channel_type: Optional[ChannelType] = None,
        max_links: int = 3,
        sensitivity_threshold: ContextSensitivity = ContextSensitivity.MEDIUM
    ) -> List[Dict[str, Any]]:
        """
        Find context relevant to a new interaction.
        
        Args:
            entity_id: ID of the entity
            topic: Optional topic to search for
            channel_type: Optional channel type for context
            max_links: Maximum number of context links to return
            sensitivity_threshold: Maximum sensitivity level to include
            
        Returns:
            List of relevant context information
        """
        # Get context links by entity
        entity_links = self.get_context_links_by_entity(entity_id, active_only=True)
        
        # Get context links by topic if provided
        topic_links = []
        if topic:
            topic_links = self.get_context_links_by_topic(topic, active_only=True)
            
        # Combine and prioritize links that match both entity and topic
        combined_links = []
        for link in (topic_links if topic else entity_links):
            # Skip links that are too sensitive
            if link.sensitivity.value == ContextSensitivity.HIGH.value and sensitivity_threshold.value != ContextSensitivity.HIGH.value:
                continue
                
            # Check if this is a relevant link
            if link in entity_links or link in topic_links:
                combined_links.append(link)
                
        # Limit number of links
        combined_links = combined_links[:max_links]
        
        # Get contextual information for each link
        results = []
        for link in combined_links:
            context_info = {
                "link_id": link.link_id,
                "topic": link.primary_topic,
                "last_updated": link.last_updated.isoformat(),
                "summary": self._generate_context_summary(link, entity_id, channel_type)
            }
            
            # Add recent messages if available and channel-appropriate
            recent_messages = self._get_recent_messages_from_link(link, 3)
            if recent_messages and self._can_share_messages(recent_messages, channel_type, sensitivity_threshold):
                context_info["recent_messages"] = recent_messages
                
            results.append(context_info)
            
        return results
    
    def generate_transition_context(
        self,
        entity_id: str,
        from_channel: ChannelType,
        to_channel: ChannelType,
        topic: Optional[str] = None,
        context_link_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate context information for a channel transition.
        
        This is used when a conversation moves from one channel to another,
        providing relevant history and a transition message.
        
        Args:
            entity_id: ID of the entity
            from_channel: Previous channel type
            to_channel: New channel type
            topic: Optional topic to focus on
            context_link_id: Optional specific context link ID
            
        Returns:
            Dictionary with transition information
        """
        transition_type = ModalityTransition.from_channel_types(from_channel, to_channel)
        
        # Get context link if ID provided
        if context_link_id and context_link_id in self.context_links:
            context_link = self.context_links[context_link_id]
        # Otherwise find most relevant context
        elif topic:
            topic_links = self.get_context_links_by_topic(topic, active_only=True)
            entity_links = self.get_context_links_by_entity(entity_id, active_only=True)
            
            # Find links that contain both the topic and the entity
            combined_links = [link for link in topic_links if link in entity_links]
            if combined_links:
                context_link = combined_links[0]  # Use the most recent one
            elif entity_links:
                context_link = entity_links[0]  # Fall back to most recent entity link
            else:
                # No relevant context, create minimal info
                return {
                    "transition_type": transition_type.value,
                    "entity_id": entity_id,
                    "from_channel": from_channel.value,
                    "to_channel": to_channel.value,
                    "topic": topic,
                    "transition_message": self._generate_generic_transition_message(
                        transition_type, topic
                    )
                }
        else:
            # No topic or context link ID, find most recent context for entity
            entity_links = self.get_context_links_by_entity(entity_id, active_only=True)
            if not entity_links:
                # No context found, create minimal info
                return {
                    "transition_type": transition_type.value,
                    "entity_id": entity_id,
                    "from_channel": from_channel.value,
                    "to_channel": to_channel.value,
                    "transition_message": self._generate_generic_transition_message(
                        transition_type, None
                    )
                }
            context_link = entity_links[0]  # Use most recent context link
            
        # Get recent messages for context
        recent_messages = self._get_recent_messages_from_link(context_link, 5)
        
        # Check what level of context can be shared in the new channel
        to_channel_sensitivity = self.channel_sensitivity_overrides.get(
            to_channel, self.default_sensitivity
        )
        
        can_share_detail = (to_channel_sensitivity.value >= context_link.sensitivity.value or
                          to_channel_sensitivity == ContextSensitivity.PUBLIC)
                          
        # Generate transition message
        transition_message = self._generate_transition_message(
            transition_type=transition_type,
            context_link=context_link,
            include_details=can_share_detail
        )
        
        # Record this transition
        self._record_transition(
            entity_id=entity_id,
            from_channel=from_channel,
            to_channel=to_channel,
            context_link_id=context_link.link_id,
            topic=context_link.primary_topic
        )
        
        # Prepare result
        result = {
            "transition_type": transition_type.value,
            "context_link_id": context_link.link_id,
            "entity_id": entity_id,
            "from_channel": from_channel.value,
            "to_channel": to_channel.value,
            "topic": context_link.primary_topic,
            "last_interaction": context_link.last_updated.isoformat(),
            "transition_message": transition_message
        }
        
        # Add recent messages if they can be shared
        if can_share_detail and recent_messages:
            result["recent_messages"] = recent_messages
            
        return result
    
    def process_incoming_message(
        self,
        message: ChannelMessage,
        extract_topic: bool = True
    ) -> Dict[str, Any]:
        """
        Process an incoming message and identify relevant context.
        
        Args:
            message: The incoming message
            extract_topic: Whether to attempt topic extraction
            
        Returns:
            Dictionary with relevant context information
        """
        sender_id = message.sender_id
        channel_type = message.channel_type
        session_id = message.session_id
        
        # Extract topic if requested
        topic = None
        if extract_topic and hasattr(message, 'content') and message.content:
            topic = self._extract_topic(message.content)
            
        # Look for existing context links
        context_links = []
        
        # Check by session if available
        if session_id and session_id in self.session_index:
            for link_id in self.session_index[session_id]:
                if link_id in self.context_links:
                    context_links.append(self.context_links[link_id])
                    
        # If message references other messages, check those too
        if hasattr(message, 'references'):
            for ref_id in message.references:
                if ref_id in self.message_index:
                    link_id = self.message_index[ref_id]
                    if link_id in self.context_links and self.context_links[link_id] not in context_links:
                        context_links.append(self.context_links[link_id])
                    
        # If no context links found, check by entity and topic
        if not context_links:
            entity_links = self.get_context_links_by_entity(sender_id, active_only=True)
            if topic:
                topic_links = self.get_context_links_by_topic(topic, active_only=True)
                # Prioritize links that match both entity
