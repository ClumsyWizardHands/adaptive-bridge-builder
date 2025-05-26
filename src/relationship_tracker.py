#!/usr/bin/env python3
"""
Relationship Tracker for Adaptive Bridge Builder

This module implements a RelationshipTracker class that maintains
relationships with other agents, including interaction history,
trust levels, communication preferences, and relationship memories.
It provides methods to repair damaged relationships and evolves
based on the quality of interactions over time.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import math

from communication_style import CommunicationStyle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RelationshipTracker")

class RelationshipStatus(Enum):
    """Enumeration of possible relationship statuses."""
    UNKNOWN = "unknown"
    NEW = "new"
    ACQUAINTANCE = "acquaintance"
    TRUSTED = "trusted"
    CLOSE = "close"
    ESSENTIAL = "essential"
    STRAINED = "strained"
    DAMAGED = "damaged"
    REPAIRING = "repairing"
    RESTRICTED = "restricted"
    BLOCKED = "blocked"

class TrustLevel(Enum):
    """Enumeration of trust levels for relationships."""
    NONE = 0
    MINIMAL = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    COMPLETE = 5

class AgentType(Enum):
    """Enumeration of agent types."""
    UNKNOWN = "unknown"
    HUMAN = "human"
    AI_ASSISTANT = "ai_assistant" 
    AI_AGENT = "ai_agent"
    SYSTEM = "system"
    GROUP = "group"
    SERVICE = "service"
    IOT_DEVICE = "iot_device"
    OTHER = "other"

class InteractionType(Enum):
    """Enumeration of interaction types."""
    MESSAGE = "message"
    TASK = "task"
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    COLLABORATION = "collaboration"
    DISPUTE = "dispute"
    REPAIR = "repair"
    INTRODUCTION = "introduction"
    FEEDBACK = "feedback"
    STATUS_UPDATE = "status_update"

class InteractionQuality(Enum):
    """Enumeration of interaction quality ratings."""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2

class InteractionRecord:
    """Represents a single interaction with another agent."""
    
    def __init__(
        self,
        agent_id: str,
        interaction_type: InteractionType,
        timestamp: Optional[str] = None,
        message_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        content_summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        quality: Optional[InteractionQuality] = None,
        principle_alignment: Optional[float] = None,
        trust_impact: Optional[float] = None
    ):
        """
        Initialize an interaction record.
        
        Args:
            agent_id: ID of the other agent in this interaction.
            interaction_type: Type of interaction.
            timestamp: ISO format timestamp of interaction.
            message_id: ID of the message if applicable.
            conversation_id: ID of the conversation if applicable.
            content_summary: Brief summary of the interaction content.
            metadata: Additional metadata about the interaction.
            quality: Quality rating of the interaction.
            principle_alignment: How well the interaction aligned with principles (0.0-1.0).
            trust_impact: Impact on trust (positive or negative).
        """
        self.agent_id = agent_id
        self.interaction_type = interaction_type
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.message_id = message_id
        self.conversation_id = conversation_id
        self.content_summary = content_summary
        self.metadata = metadata or {}
        self.quality = quality or InteractionQuality.NEUTRAL
        self.principle_alignment = principle_alignment
        self.trust_impact = trust_impact
        self.interaction_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the interaction record to a dictionary."""
        return {
            "interaction_id": self.interaction_id,
            "agent_id": self.agent_id,
            "interaction_type": self.interaction_type.value,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "content_summary": self.content_summary,
            "metadata": self.metadata,
            "quality": self.quality.value,
            "principle_alignment": self.principle_alignment,
            "trust_impact": self.trust_impact
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InteractionRecord':
        """Create an InteractionRecord from a dictionary."""
        record = cls(
            agent_id=data.get("agent_id", ""),
            interaction_type=InteractionType(data.get("interaction_type", "message")),
            timestamp=data.get("timestamp"),
            message_id=data.get("message_id"),
            conversation_id=data.get("conversation_id"),
            content_summary=data.get("content_summary"),
            metadata=data.get("metadata", {}),
            quality=InteractionQuality(data.get("quality", 0)),
            principle_alignment=data.get("principle_alignment"),
            trust_impact=data.get("trust_impact")
        )
        record.interaction_id = data.get("interaction_id", record.interaction_id)
        return record

class RelationshipMemory:
    """Represents a significant memory about a relationship."""
    
    def __init__(
        self,
        agent_id: str,
        memory_type: str,
        content: str,
        timestamp: Optional[str] = None,
        importance: float = 1.0,
        related_interactions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a relationship memory.
        
        Args:
            agent_id: ID of the agent this memory is about.
            memory_type: Type of memory (e.g., "first_interaction", "breach_of_trust").
            content: Content of the memory.
            timestamp: ISO format timestamp of when this memory was created.
            importance: Importance rating (0.0-5.0).
            related_interactions: List of interaction IDs related to this memory.
            metadata: Additional metadata about the memory.
        """
        self.agent_id = agent_id
        self.memory_type = memory_type
        self.content = content
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.importance = max(0.0, min(5.0, importance))
        self.related_interactions = related_interactions or []
        self.metadata = metadata or {}
        self.memory_id = str(uuid.uuid4())
        self.last_accessed = self.timestamp
        self.access_count = 0
    
    def access(self) -> None:
        """Record an access to this memory."""
        self.last_accessed = datetime.now(timezone.utc).isoformat()
        self.access_count = self.access_count + 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the relationship memory to a dictionary."""
        return {
            "memory_id": self.memory_id,
            "agent_id": self.agent_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "related_interactions": self.related_interactions,
            "metadata": self.metadata,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RelationshipMemory':
        """Create a RelationshipMemory from a dictionary."""
        memory = cls(
            agent_id=data.get("agent_id", ""),
            memory_type=data.get("memory_type", ""),
            content=data.get("content", ""),
            timestamp=data.get("timestamp"),
            importance=data.get("importance", 1.0),
            related_interactions=data.get("related_interactions", []),
            metadata=data.get("metadata", {})
        )
        memory.memory_id = data.get("memory_id", memory.memory_id)
        memory.last_accessed = data.get("last_accessed", memory.last_accessed)
        memory.access_count = data.get("access_count", 0)
        return memory

class AgentRelationship:
    """Represents a relationship with another agent."""
    
    def __init__(
        self,
        agent_id: str,
        agent_name: Optional[str] = None,
        agent_type: Optional[AgentType] = None,
        first_interaction: Optional[str] = None,
        last_interaction: Optional[str] = None,
        interaction_count: int = 0,
        trust_level: TrustLevel = TrustLevel.NONE,
        trust_score: float = 0.0,
        status: RelationshipStatus = RelationshipStatus.UNKNOWN,
        communication_style: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an agent relationship.
        
        Args:
            agent_id: ID of the agent.
            agent_name: Name of the agent.
            agent_type: Type of the agent.
            first_interaction: ISO format timestamp of first interaction.
            last_interaction: ISO format timestamp of last interaction.
            interaction_count: Number of interactions with this agent.
            trust_level: Trust level for this agent.
            trust_score: Numerical trust score (0.0-100.0).
            status: Current status of the relationship.
            communication_style: Communication style preferences.
            metadata: Additional metadata about the relationship.
        """
        self.agent_id = agent_id
        self.agent_name = agent_name or agent_id
        self.agent_type = agent_type or AgentType.UNKNOWN
        self.first_interaction = first_interaction
        self.last_interaction = last_interaction
        self.interaction_count = interaction_count
        self.trust_level = trust_level
        self.trust_score = max(0.0, min(100.0, trust_score))
        self.status = status
        self.communication_style = communication_style or {}
        self.metadata = metadata or {}
        self.memories: List[RelationshipMemory] = []
        self.recent_interactions: List[str] = []  # Store interaction IDs
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at
        self.repair_attempts = 0
        self.blocked_reason = None
        self.preferences = {
            "preferred_interaction_types": [],
            "avoided_interaction_types": [],
            "communication_frequency": "normal",
            "response_priority": "normal"
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the agent relationship to a dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type.value,
            "first_interaction": self.first_interaction,
            "last_interaction": self.last_interaction,
            "interaction_count": self.interaction_count,
            "trust_level": self.trust_level.value,
            "trust_score": self.trust_score,
            "status": self.status.value,
            "communication_style": self.communication_style,
            "metadata": self.metadata,
            "memories": [m.to_dict() for m in self.memories],
            "recent_interactions": self.recent_interactions,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "repair_attempts": self.repair_attempts,
            "blocked_reason": self.blocked_reason,
            "preferences": self.preferences
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentRelationship':
        """Create an AgentRelationship from a dictionary."""
        relationship = cls(
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name"),
            agent_type=AgentType(data.get("agent_type", "unknown")),
            first_interaction=data.get("first_interaction"),
            last_interaction=data.get("last_interaction"),
            interaction_count=data.get("interaction_count", 0),
            trust_level=TrustLevel(data.get("trust_level", 0)),
            trust_score=data.get("trust_score", 0.0),
            status=RelationshipStatus(data.get("status", "unknown")),
            communication_style=data.get("communication_style", {}),
            metadata=data.get("metadata", {})
        )
        relationship.memories = [
            RelationshipMemory.from_dict(m) for m in data.get("memories", [])
        ]
        relationship.recent_interactions = data.get("recent_interactions", [])
        relationship.created_at = data.get("created_at", relationship.created_at)
        relationship.updated_at = data.get("updated_at", relationship.updated_at)
        relationship.repair_attempts = data.get("repair_attempts", 0)
        relationship.blocked_reason = data.get("blocked_reason")
        relationship.preferences = data.get("preferences", relationship.preferences)
        return relationship
    
    def update_trust_level(self) -> None:
        """Update the trust level based on the trust score."""
        if self.trust_score < 10.0:
            self.trust_level = TrustLevel.NONE
        elif self.trust_score < 30.0:
            self.trust_level = TrustLevel.MINIMAL
        elif self.trust_score < 50.0:
            self.trust_level = TrustLevel.LOW
        elif self.trust_score < 70.0:
            self.trust_level = TrustLevel.MODERATE
        elif self.trust_score < 90.0:
            self.trust_level = TrustLevel.HIGH
        else:
            self.trust_level = TrustLevel.COMPLETE
    
    def update_status(self) -> None:
        """Update the relationship status based on trust level and interaction history."""
        # Don't change these statuses automatically
        if self.status in [
            RelationshipStatus.BLOCKED, 
            RelationshipStatus.RESTRICTED, 
            RelationshipStatus.REPAIRING
        ]:
            return
            
        # New relationships
        if self.interaction_count <= 3:
            self.status = RelationshipStatus.NEW
            return
            
        # Base on trust level
        if self.trust_level == TrustLevel.NONE:
            self.status = RelationshipStatus.DAMAGED
        elif self.trust_level == TrustLevel.MINIMAL:
            self.status = RelationshipStatus.STRAINED
        elif self.trust_level == TrustLevel.LOW:
            self.status = RelationshipStatus.ACQUAINTANCE
        elif self.trust_level == TrustLevel.MODERATE:
            self.status = RelationshipStatus.ACQUAINTANCE
        elif self.trust_level == TrustLevel.HIGH:
            self.status = RelationshipStatus.TRUSTED
        elif self.trust_level == TrustLevel.COMPLETE:
            if self.interaction_count > 50:
                self.status = RelationshipStatus.ESSENTIAL
            else:
                self.status = RelationshipStatus.CLOSE
    
    def add_memory(self, memory: RelationshipMemory) -> None:
        """Add a new memory to this relationship."""
        # Check if a similar memory already exists
        for existing in self.memories:
            if existing.memory_type == memory.memory_type and existing.content == memory.content:
                # Update importance if new memory is more important
                if memory.importance > existing.importance:
                    existing.importance = memory.importance
                # Merge related interactions
                for interaction_id in memory.related_interactions:
                    if interaction_id not in existing.related_interactions:
                        existing.related_interactions.append(interaction_id)
                return
                
        # Add new memory
        self.memories = [*self.memories, memory]
        
        # Sort memories by importance (descending)
        self.memories.sort(key=lambda m: m.importance, reverse=True)
        
        # Keep only a reasonable number of memories
        max_memories = 50  # Arbitrary limit
        if len(self.memories) > max_memories:
            # Remove least important memories
            self.memories = self.memories[:max_memories]
    
    def get_relevant_memories(
        self, 
        memory_type: Optional[str] = None, 
        max_count: int = 5,
        min_importance: float = 0.0
    ) -> List[RelationshipMemory]:
        """
        Get relevant memories for this relationship.
        
        Args:
            memory_type: Optional type of memory to filter by.
            max_count: Maximum number of memories to return.
            min_importance: Minimum importance level for included memories.
            
        Returns:
            List of relevant relationship memories.
        """
        filtered = []
        
        for memory in self.memories:
            # Apply filters
            if memory_type and memory.memory_type != memory_type:
                continue
            if memory.importance < min_importance:
                continue
                
            # Record access
            memory.access()
            filtered.append(memory)
            
            if len(filtered) >= max_count:
                break
                
        return filtered

class RelationshipTracker:
    """
    Tracks and manages relationships with other agents.
    
    The RelationshipTracker maintains a history of interactions with other agents,
    tracks trust levels, records communication preferences, and develops
    "relationship memories" to inform future interactions. It can persist
    relationship data between sessions and evolve based on interaction quality.
    """
    
    def __init__(
        self,
        agent_id: str,
        data_dir: Optional[str] = None,
        auto_save: bool = True
    ):
        """
        Initialize the RelationshipTracker.
        
        Args:
            agent_id: ID of this agent.
            data_dir: Directory for persisting relationship data.
            auto_save: Whether to automatically save changes.
        """
        self.agent_id = agent_id
        self.data_dir = data_dir or "data/relationships"
        self.auto_save = auto_save
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            
        # Storage for relationships and interactions
        self.relationships: Dict[str, AgentRelationship] = {}
        self.interactions: Dict[str, InteractionRecord] = {}
        
        # Trust model parameters
        self.trust_decay_factor = 0.98  # Slow decay of trust over time
        self.trust_recovery_factor = 1.1  # How quickly trust can recover
        self.trust_damage_sensitivity = 1.5  # How much negative interactions hurt trust
        
        # Load existing data
        self._load_data()
        
        logger.info(f"RelationshipTracker initialized for agent {agent_id}")
    
    def _get_relationship_file_path(self, agent_id: str) -> str:
        """Get the file path for a relationship."""
        safe_id = agent_id.replace(":", "_").replace("/", "_")
        return os.path.join(self.data_dir, f"relationship_{safe_id}.json")
    
    def _get_interactions_file_path(self, agent_id: str) -> str:
        """Get the file path for interactions with an agent."""
        safe_id = agent_id.replace(":", "_").replace("/", "_")
        return os.path.join(self.data_dir, f"interactions_{safe_id}.json")
    
    def _load_data(self) -> None:
        """Load relationship and interaction data from files."""
        # Find all relationship files
        if not os.path.exists(self.data_dir):
            return
            
        relationship_files = [
            f for f in os.listdir(self.data_dir) 
            if f.startswith("relationship_") and f.endswith(".json")
        ]
        
        for rel_file in relationship_files:
            try:
                # Load relationship
                with open(os.path.join(self.data_dir, rel_file), 'r') as f:
                    rel_data = json.load(f)
                    relationship = AgentRelationship.from_dict(rel_data)
                    self.relationships = {**self.relationships, relationship.agent_id: relationship}
                
                # Load interactions
                int_file = rel_file.replace("relationship_", "interactions_")
                int_path = os.path.join(self.data_dir, int_file)
                if os.path.exists(int_path):
                    with open(int_path, 'r') as f:
                        int_data = json.load(f)
                        for interaction_dict in int_data:
                            interaction = InteractionRecord.from_dict(interaction_dict)
                            self.interactions = {**self.interactions, interaction.interaction_id: interaction}
            except Exception as e:
                logger.error(f"Error loading relationship data from {rel_file}: {e}")
        
        logger.info(f"Loaded {len(self.relationships)} relationships and {len(self.interactions)} interactions")
    
    def _save_relationship(self, agent_id: str) -> None:
        """Save relationship data for a specific agent."""
        if agent_id not in self.relationships:
            return
            
        relationship = self.relationships[agent_id]
        
        try:
            # Save relationship
            rel_path = self._get_relationship_file_path(agent_id)
            with open(rel_path, 'w') as f:
                json.dump(relationship.to_dict(), f, indent=2)
            
            # Save interactions
            agent_interactions = [
                interaction.to_dict() 
                for interaction_id, interaction in self.interactions.items()
                if interaction.agent_id == agent_id
            ]
            
            if agent_interactions:
                int_path = self._get_interactions_file_path(agent_id)
                with open(int_path, 'w') as f:
                    json.dump(agent_interactions, f, indent=2)
                    
            logger.debug(f"Saved relationship data for agent {agent_id}")
        except Exception as e:
            logger.error(f"Error saving relationship data for agent {agent_id}: {e}")
    
    def save_all(self) -> None:
        """Save all relationship and interaction data."""
        for agent_id in self.relationships.keys():
            self._save_relationship(agent_id)
        logger.info(f"Saved data for {len(self.relationships)} relationships")
    
    def get_relationship(self, agent_id: str, create_if_missing: bool = True) -> Optional[AgentRelationship]:
        """
        Get the relationship with a specific agent.
        
        Args:
            agent_id: ID of the agent.
            create_if_missing: Whether to create a new relationship if none exists.
            
        Returns:
            The agent relationship or None if not found and not created.
        """
        if agent_id in self.relationships:
            return self.relationships[agent_id]
            
        if not create_if_missing:
            return None
            
        # Create new relationship
        relationship = AgentRelationship(agent_id=agent_id)
        self.relationships = {**self.relationships, agent_id: relationship}
        
        # Save if auto-save is enabled
        if self.auto_save:
            self._save_relationship(agent_id)
            
        return relationship
    
    def record_interaction(
        self,
        agent_id: str,
        interaction_type: InteractionType,
        message_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        content_summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        quality: Optional[InteractionQuality] = None,
        principle_alignment: Optional[float] = None,
    ) -> InteractionRecord:
        """
        Record a new interaction with an agent.
        
        Args:
            agent_id: ID of the agent.
            interaction_type: Type of interaction.
            message_id: ID of the message if applicable.
            conversation_id: ID of the conversation if applicable.
            content_summary: Brief summary of the interaction content.
            metadata: Additional metadata about the interaction.
            quality: Quality rating of the interaction.
            principle_alignment: How well the interaction aligned with principles (0.0-1.0).
            
        Returns:
            The created interaction record.
        """
        # Create the interaction record
        interaction = InteractionRecord(
            agent_id=agent_id,
            interaction_type=interaction_type,
            message_id=message_id,
            conversation_id=conversation_id,
            content_summary=content_summary,
            metadata=metadata,
            quality=quality or InteractionQuality.NEUTRAL,
            principle_alignment=principle_alignment
        )
        
        # Calculate trust impact
        trust_impact = self._calculate_trust_impact(
            interaction_type,
            quality or InteractionQuality.NEUTRAL,
            principle_alignment
        )
        interaction.trust_impact = trust_impact
        
        # Store the interaction
        self.interactions = {**self.interactions, interaction.interaction_id: interaction}
        
        # Update the relationship
        relationship = self.get_relationship(agent_id)
        if not relationship.first_interaction:
            relationship.first_interaction = interaction.timestamp
            
            # Add "first interaction" memory
            memory = RelationshipMemory(
                agent_id=agent_id,
                memory_type="first_interaction",
                content=f"First interaction of type {interaction_type.value}",
                importance=2.0,
                related_interactions=[interaction.interaction_id]
            )
            relationship.add_memory(memory)
            
        relationship.last_interaction = interaction.timestamp
        relationship.interaction_count += 1
        relationship.updated_at = datetime.now(timezone.utc).isoformat()
        
        # Add to recent interactions
        relationship.recent_interactions.append(interaction.interaction_id)
        if len(relationship.recent_interactions) > 20:  # Limit recent interactions
            relationship.recent_interactions = relationship.recent_interactions[-20:]
        
        # Update trust score
        self._update_trust_score(relationship, trust_impact)
        
        # Check for memory-worthy events
        self._check_for_memory_events(relationship, interaction)
        
        # Auto-save if enabled
        if self.auto_save:
            self._save_relationship(agent_id)
            
        logger.info(f"Recorded {interaction_type.value} interaction with agent {agent_id}")
        return interaction
    
    def _calculate_trust_impact(
        self,
        interaction_type: InteractionType,
        quality: InteractionQuality,
        principle_alignment: Optional[float]
    ) -> float:
        """
        Calculate the trust impact of an interaction.
        
        Args:
            interaction_type: Type of interaction.
            quality: Quality rating of the interaction.
            principle_alignment: How well the interaction aligned with principles (0.0-1.0).
            
        Returns:
            Trust impact value (positive or negative).
        """
        # Base impact from quality
        base_impact = float(quality.value)
        
        # Adjust based on interaction type
        type_multiplier = 1.0
        if interaction_type == InteractionType.DISPUTE:
            # Disputes have higher impact
            type_multiplier = 1.5
        elif interaction_type == InteractionType.REPAIR:
            # Repair attempts have higher positive impact
            if quality.value > 0:
                type_multiplier = 2.0
            else:
                type_multiplier = 1.0  # Failed repairs don't hurt extra
        elif interaction_type == InteractionType.INTRODUCTION:
            # First impressions matter more
            type_multiplier = 1.3
            
        # Adjust based on principle alignment if available
        alignment_factor = 1.0
        if principle_alignment is not None:
            # Reward high alignment, penalize low alignment
            alignment_factor = 0.5 + principle_alignment
            
        # Apply "Trust as Foundation" principle:
        # - Negative interactions hurt trust more than positive ones build it
        # - Higher sensitivity to trust violations
        if base_impact < 0:
            impact = base_impact * type_multiplier * alignment_factor * self.trust_damage_sensitivity
        else:
            impact = base_impact * type_multiplier * alignment_factor
            
        return impact
    
    def _update_trust_score(
        self,
        relationship: AgentRelationship,
        trust_impact: float
    ) -> None:
        """
        Update the trust score for a relationship.
        
        Args:
            relationship: The relationship to update.
            trust_impact: Trust impact value from the interaction.
        """
        # Apply time decay
        time_since_last = 0
        if relationship.last_interaction:
            try:
                last_time = datetime.fromisoformat(relationship.last_interaction)
                now = datetime.now(timezone.utc)
                time_since_last = (now - last_time).total_seconds() / 86400.0  # Days
            except (ValueError, TypeError):
                time_since_last = 0
        
        # Apply decay based on time since last interaction
        decay = self.trust_decay_factor ** time_since_last
        current_score = relationship.trust_score * decay
        
        # Apply the new impact
        # Different formulas for positive and negative impact
        if trust_impact >= 0:
            # Positive impact has diminishing returns at higher trust levels
            max_gain = 100.0 - current_score
            gain = max_gain * (1.0 - math.exp(-0.05 * trust_impact))
            new_score = current_score + gain
        else:
            # Negative impact has greater effect at higher trust levels
            loss_factor = 0.1 * (1.0 + current_score / 50.0)  # More to lose when trust is high
            new_score = current_score + (trust_impact * loss_factor * 10.0)
        
        # Update the score, keeping it in valid range
        relationship.trust_score = max(0.0, min(100.0, new_score))
        
        # Update derived values
        relationship.update_trust_level()
        relationship.update_status()
    
    def _check_for_memory_events(
        self,
        relationship: AgentRelationship,
        interaction: InteractionRecord
    ) -> None:
        """
        Check for memory-worthy events based on the interaction.
        
        Args:
            relationship: The relationship to check.
            interaction: The interaction to analyze.
        """
        # Track significant trust changes
        if interaction.trust_impact is not None:
            if interaction.trust_impact <= -5.0:
                # Significant trust breach
                memory = RelationshipMemory(
                    agent_id=relationship.agent_id,
                    memory_type="trust_breach",
                    content=f"Significant breach of trust during {interaction.interaction_type.value}",
                    importance=min(4.0, abs(interaction.trust_impact) / 5.0),
                    related_interactions=[interaction.interaction_id]
                )
                relationship.add_memory(memory)
            elif interaction.trust_impact >= 3.0:
                # Significant trust building
                memory = RelationshipMemory(
                    agent_id=relationship.agent_id,
                    memory_type="trust_building",
                    content=f"Notable trust building during {interaction.interaction_type.value}",
                    importance=min(3.0, interaction.trust_impact / 3.0),
                    related_interactions=[interaction.interaction_id]
                )
                relationship.add_memory(memory)
                
        # Track milestone interactions
        if relationship.interaction_count == 10:
            memory = RelationshipMemory(
                agent_id=relationship.agent_id,
                memory_type="milestone",
                content="Reached 10 interactions",
                importance=1.5,
                related_interactions=[interaction.interaction_id]
            )
            relationship.add_memory(memory)
        elif relationship.interaction_count == 50:
            memory = RelationshipMemory(
                agent_id=relationship.agent_id,
                memory_type="milestone",
                content="Reached 50 interactions",
                importance=2.0,
                related_interactions=[interaction.interaction_id]
            )
            relationship.add_memory(memory)
        elif relationship.interaction_count == 100:
            memory = RelationshipMemory(
                agent_id=relationship.agent_id,
                memory_type="milestone",
                content="Reached 100 interactions",
                importance=2.5,
                related_interactions=[interaction.interaction_id]
            )
            relationship.add_memory(memory)
            
        # Track principle violations
        if interaction.principle_alignment is not None and interaction.principle_alignment < 0.3:
            memory = RelationshipMemory(
                agent_id=relationship.agent_id,
                memory_type="principle_violation",
                content=f"Low principle alignment during {interaction.interaction_type.value}",
                importance=3.0,
                related_interactions=[interaction.interaction_id]
            )
            relationship.add_memory(memory)
            
        # Track status changes
        old_status = relationship.status
        relationship.update_status()
        if relationship.status != old_status and relationship.interaction_count > 3:
            memory = RelationshipMemory(
                agent_id=relationship.agent_id,
                memory_type="status_change",
                content=f"Relationship status changed from {old_status.value} to {relationship.status.value}",
                importance=2.5,
                related_interactions=[interaction.interaction_id]
            )
            relationship.add_memory(memory)
            
        # Track significant interactions by type
        if interaction.interaction_type == InteractionType.DISPUTE:
            memory = RelationshipMemory(
                agent_id=relationship.agent_id,
                memory_type="dispute",
                content=f"Dispute occurred: {interaction.content_summary}",
                importance=3.0,
                related_interactions=[interaction.interaction_id]
            )
            relationship.add_memory(memory)
        elif interaction.interaction_type == InteractionType.REPAIR:
            memory = RelationshipMemory(
                agent_id=relationship.agent_id,
                memory_type="repair_attempt",
                content=f"Repair attempt: {interaction.content_summary}",
                importance=3.0,
                related_interactions=[interaction.interaction_id]
            )
            relationship.add_memory(memory)
    
    def get_recent_interactions(
        self,
        agent_id: str,
        max_count: int = 10
    ) -> List[InteractionRecord]:
        """
        Get recent interactions with an agent.
        
        Args:
            agent_id: ID of the agent.
            max_count: Maximum number of interactions to return.
            
        Returns:
            List of recent interactions.
        """
        relationship = self.get_relationship(agent_id, create_if_missing=False)
        if not relationship:
            return []
            
        interactions = []
        for interaction_id in reversed(relationship.recent_interactions):
            if interaction_id in self.interactions:
                interactions.append(self.interactions[interaction_id])
                if len(interactions) >= max_count:
                    break
                    
        return interactions
    
    def get_interaction_stats(
        self,
        agent_id: str
    ) -> Dict[str, Any]:
        """
        Get statistics about interactions with an agent.
        
        Args:
            agent_id: ID of the agent.
            
        Returns:
            Dictionary with interaction statistics.
        """
        relationship = self.get_relationship(agent_id, create_if_missing=False)
        if not relationship:
            return {
                "total_count": 0,
                "first_interaction": None,
                "last_interaction": None,
                "average_quality": 0.0,
                "average_principle_alignment": None,
                "interaction_types": {}
            }
            
        # Collect interactions
        agent_interactions = [
            interaction for interaction in self.interactions.values()
            if interaction.agent_id == agent_id
        ]
        
        if not agent_interactions:
            return {
                "total_count": 0,
                "first_interaction": relationship.first_interaction,
                "last_interaction": relationship.last_interaction,
                "average_quality": 0.0,
                "average_principle_alignment": None,
                "interaction_types": {}
            }
            
        # Calculate statistics
        quality_values = [float(interaction.quality.value) for interaction in agent_interactions]
        alignment_values = [
            interaction.principle_alignment for interaction in agent_interactions
            if interaction.principle_alignment is not None
        ]
        
        type_counts = {}
        for interaction in agent_interactions:
            type_name = interaction.interaction_type.value
            if type_name in type_counts:
                type_counts[type_name] += 1
            else:
                type_counts[type_name] = 1
                
        return {
            "total_count": len(agent_interactions),
            "first_interaction": relationship.first_interaction,
            "last_interaction": relationship.last_interaction,
            "average_quality": sum(quality_values) / len(quality_values) if quality_values else 0.0,
            "average_principle_alignment": sum(alignment_values) / len(alignment_values) if alignment_values else None,
            "interaction_types": type_counts
        }
    
    def update_communication_preferences(
        self,
        agent_id: str,
        preferences: Dict[str, Any]
    ) -> None:
        """
        Update communication preferences for an agent.
        
        Args:
            agent_id: ID of the agent.
            preferences: Dictionary of communication preferences.
        """
        relationship = self.get_relationship(agent_id)
        
        # Update preferences
        for key, value in preferences.items():
            if key in relationship.preferences:
                relationship.preferences[key] = value
                
        # Update timestamp
        relationship.updated_at = datetime.now(timezone.utc).isoformat()
        
        # Auto-save if enabled
        if self.auto_save:
            self._save_relationship(agent_id)
            
        logger.info(f"Updated communication preferences for agent {agent_id}")
    
    def update_communication_style(
        self,
        agent_id: str,
        communication_style: Union[CommunicationStyle, Dict[str, Any]]
    ) -> None:
        """
        Update communication style for an agent.
        
        Args:
            agent_id: ID of the agent.
            communication_style: CommunicationStyle object or dictionary.
        """
        relationship = self.get_relationship(agent_id)
        
        # Convert to dictionary if needed
        if isinstance(communication_style, CommunicationStyle):
            style_dict = communication_style.to_dict()
        else:
            style_dict = communication_style
            
        # Update communication style
        relationship.communication_style = style_dict
        relationship.updated_at = datetime.now(timezone.utc).isoformat()
        
        # Add memory about style
        memory = RelationshipMemory(
            agent_id=agent_id,
            memory_type="communication_style",
            content=f"Communication style set: {style_dict.get('formality', 'unknown')} formality, "
                   f"{style_dict.get('detail_level', 'unknown')} detail level",
            importance=2.0
        )
        relationship.add_memory(memory)
        
        # Auto-save if enabled
        if self.auto_save:
            self._save_relationship(agent_id)
            
        logger.info(f"Updated communication style for agent {agent_id}")
    
    def get_trust_evaluation(
        self,
        agent_id: str
    ) -> Dict[str, Any]:
        """
        Get a detailed trust evaluation for an agent.
        
        Args:
            agent_id: ID of the agent.
            
        Returns:
            Dictionary with trust evaluation details.
        """
        relationship = self.get_relationship(agent_id, create_if_missing=False)
        if not relationship:
            return {
                "trust_score": 0.0,
                "trust_level": TrustLevel.NONE.value,
                "status": RelationshipStatus.UNKNOWN.value,
                "interaction_count": 0,
                "has_trust_breaches": False,
                "has_trust_building": False,
                "can_be_trusted": False,
                "trust_trend": "none"
            }
            
        # Calculate trust trend based on recent interactions
        recent_interactions = self.get_recent_interactions(agent_id, max_count=5)
        trust_impacts = [
            interaction.trust_impact for interaction in recent_interactions
            if interaction.trust_impact is not None
        ]
        
        if len(trust_impacts) >= 3:
            avg_impact = sum(trust_impacts) / len(trust_impacts)
            if avg_impact > 1.0:
                trust_trend = "improving"
            elif avg_impact < -1.0:
                trust_trend = "declining"
            else:
                trust_trend = "stable"
        else:
            trust_trend = "insufficient_data"
            
        # Check for trust-related memories
        has_trust_breaches = any(
            memory.memory_type == "trust_breach" 
            for memory in relationship.memories
        )
        
        has_trust_building = any(
            memory.memory_type == "trust_building" 
            for memory in relationship.memories
        )
        
        # Determine if agent can be trusted for sensitive operations
        can_be_trusted = (
            relationship.trust_level in [TrustLevel.HIGH, TrustLevel.COMPLETE] and
            relationship.interaction_count >= 10 and
            not has_trust_breaches
        )
        
        return {
            "trust_score": relationship.trust_score,
            "trust_level": relationship.trust_level.value,
            "status": relationship.status.value,
            "interaction_count": relationship.interaction_count,
            "has_trust_breaches": has_trust_breaches,
            "has_trust_building": has_trust_building,
            "can_be_trusted": can_be_trusted,
            "trust_trend": trust_trend
        }
    
    def create_repair_plan(
        self,
        agent_id: str
    ) -> Dict[str, Any]:
        """
        Create a plan to repair a damaged relationship.
        
        Args:
            agent_id: ID of the agent.
            
        Returns:
            Dictionary with repair plan details.
        """
        relationship = self.get_relationship(agent_id)
        
        # Check if repair is needed
        if relationship.status not in [
            RelationshipStatus.DAMAGED, 
            RelationshipStatus.STRAINED
        ]:
            return {
                "repair_needed": False,
                "current_status": relationship.status.value,
                "message": "No repair needed for this relationship"
            }
            
        # Get trust breaches to address
        trust_breaches = [
            memory for memory in relationship.memories
            if memory.memory_type == "trust_breach"
        ]
        
        # Get principle violations to address
        principle_violations = [
            memory for memory in relationship.memories
            if memory.memory_type == "principle_violation"
        ]
        
        # Create repair steps
        repair_steps = []
        
        # Step 1: Always acknowledge past issues
        repair_steps.append({
            "step": "acknowledge",
            "description": "Acknowledge past trust breaches and principle violations",
            "breaches": [memory.content for memory in trust_breaches[:3]]
        })
        
        # Step 2: Offer explanation if appropriate
        if trust_breaches or principle_violations:
            repair_steps.append({
                "step": "explain",
                "description": "Provide context and explanation for past issues",
                "focus_areas": [
                    memory.content for memory in 
                    (trust_breaches + principle_violations)[:3]
                ]
            })
            
        # Step 3: Recommend corrective actions
        repair_steps.append({
            "step": "correct",
            "description": "Propose corrective actions to rebuild trust",
            "actions": [
                "Demonstrate consistent adherence to principles",
                "Increase transparency in all interactions",
                "Establish clear expectations for future interactions",
                "Provide regular status updates"
            ]
        })
        
        # Step 4: Monitoring plan
        repair_steps.append({
            "step": "monitor",
            "description": "Establish monitoring to track relationship repair",
            "metrics": [
                "Trust score improvement over time",
                "Consistent positive interaction quality",
                "High principle alignment in all interactions",
                "Regular constructive interactions"
            ]
        })
        
        # Update repair attempts counter
        relationship.repair_attempts += 1
        relationship.updated_at = datetime.now(timezone.utc).isoformat()
        relationship.status = RelationshipStatus.REPAIRING
        
        # Add a memory for this repair attempt
        memory = RelationshipMemory(
            agent_id=agent_id,
            memory_type="repair_plan",
            content=f"Created repair plan (attempt #{relationship.repair_attempts})",
            importance=4.0
        )
        relationship.add_memory(memory)
        
        # Auto-save if enabled
        if self.auto_save:
            self._save_relationship(agent_id)
            
        logger.info(f"Created repair plan for agent {agent_id} (attempt #{relationship.repair_attempts})")
        
        return {
            "repair_needed": True,
            "current_status": relationship.status.value,
            "repair_attempt": relationship.repair_attempts,
            "trust_score": relationship.trust_score,
            "steps": repair_steps,
            "message": "Relationship repair plan created"
        }
    
    def mark_repair_success(
        self,
        agent_id: str,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Mark a relationship repair as successful.
        
        Args:
            agent_id: ID of the agent.
            notes: Optional notes about the repair.
            
        Returns:
            Dictionary with repair outcome details.
        """
        relationship = self.get_relationship(agent_id)
        
        # Only allow if relationship was in repairing status
        if relationship.status != RelationshipStatus.REPAIRING:
            return {
                "success": False,
                "message": f"Relationship was not in repairing status (current: {relationship.status.value})"
            }
            
        # Boost trust score as a repair bonus
        trust_boost = min(20.0, 100.0 - relationship.trust_score)
        relationship.trust_score += trust_boost
        relationship.update_trust_level()
        
        # Update status
        old_status = relationship.status
        relationship.status = RelationshipStatus.ACQUAINTANCE
        if relationship.trust_level >= TrustLevel.HIGH:
            relationship.status = RelationshipStatus.TRUSTED
            
        relationship.updated_at = datetime.now(timezone.utc).isoformat()
        
        # Add memory for successful repair
        memory = RelationshipMemory(
            agent_id=agent_id,
            memory_type="repair_success",
            content=f"Successful relationship repair (attempt #{relationship.repair_attempts})" +
                   (f": {notes}" if notes else ""),
            importance=4.5
        )
        relationship.add_memory(memory)
        
        # Auto-save if enabled
        if self.auto_save:
            self._save_relationship(agent_id)
            
        logger.info(f"Marked repair as successful for agent {agent_id}")
        
        return {
            "success": True,
            "old_status": old_status.value,
            "new_status": relationship.status.value,
            "trust_boost": trust_boost,
            "new_trust_score": relationship.trust_score,
            "new_trust_level": relationship.trust_level.value,
            "message": "Relationship repair marked as successful"
        }
    
    def mark_repair_failure(
        self,
        agent_id: str,
        reason: str
    ) -> Dict[str, Any]:
        """
        Mark a relationship repair as failed.
        
        Args:
            agent_id: ID of the agent.
            reason: Reason for the repair failure.
            
        Returns:
            Dictionary with repair outcome details.
        """
        relationship = self.get_relationship(agent_id)
        
        # Only allow if relationship was in repairing status
        if relationship.status != RelationshipStatus.REPAIRING:
            return {
                "success": False,
                "message": f"Relationship was not in repairing status (current: {relationship.status.value})"
            }
            
        # Update status based on current trust level
        if relationship.trust_level <= TrustLevel.MINIMAL:
            relationship.status = RelationshipStatus.DAMAGED
        else:
            relationship.status = RelationshipStatus.STRAINED
            
        relationship.updated_at = datetime.now(timezone.utc).isoformat()
        
        # Add memory for failed repair
        memory = RelationshipMemory(
            agent_id=agent_id,
            memory_type="repair_failure",
            content=f"Failed relationship repair (attempt #{relationship.repair_attempts}): {reason}",
            importance=4.0
        )
        relationship.add_memory(memory)
        
        # Auto-save if enabled
        if self.auto_save:
            self._save_relationship(agent_id)
            
        logger.info(f"Marked repair as failed for agent {agent_id}: {reason}")
        
        return {
            "success": True,
            "status": relationship.status.value,
            "trust_score": relationship.trust_score,
            "trust_level": relationship.trust_level.value,
            "repair_attempts": relationship.repair_attempts,
            "message": "Relationship repair marked as failed",
            "reason": reason
        }
    
    def block_agent(
        self,
        agent_id: str,
        reason: str
    ) -> Dict[str, Any]:
        """
        Block interactions with an agent.
        
        Args:
            agent_id: ID of the agent to block.
            reason: Reason for blocking.
            
        Returns:
            Dictionary with blocking details.
        """
        relationship = self.get_relationship(agent_id)
        
        # Update status
        old_status = relationship.status
        relationship.status = RelationshipStatus.BLOCKED
        relationship.blocked_reason = reason
        relationship.updated_at = datetime.now(timezone.utc).isoformat()
        
        # Add memory for blocking
        memory = RelationshipMemory(
            agent_id=agent_id,
            memory_type="blocking",
            content=f"Agent blocked: {reason}",
            importance=5.0
        )
        relationship.add_memory(memory)
        
        # Auto-save if enabled
        if self.auto_save:
            self._save_relationship(agent_id)
            
        logger.info(f"Blocked agent {agent_id}: {reason}")
        
        return {
            "success": True,
            "old_status": old_status.value,
            "new_status": relationship.status.value,
            "reason": reason,
            "message": "Agent has been blocked"
        }
    
    def unblock_agent(
        self,
        agent_id: str,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Unblock interactions with an agent.
        
        Args:
            agent_id: ID of the agent to unblock.
            notes: Optional notes about unblocking.
            
        Returns:
            Dictionary with unblocking details.
        """
        relationship = self.get_relationship(agent_id, create_if_missing=False)
        if not relationship or relationship.status != RelationshipStatus.BLOCKED:
            return {
                "success": False,
                "message": "Agent is not blocked"
            }
            
        # Update status based on trust level
        if relationship.trust_level <= TrustLevel.MINIMAL:
            relationship.status = RelationshipStatus.DAMAGED
        elif relationship.trust_level <= TrustLevel.MODERATE:
            relationship.status = RelationshipStatus.STRAINED
        else:
            relationship.status = RelationshipStatus.ACQUAINTANCE
            
        # Clear blocked reason
        blocked_reason = relationship.blocked_reason
        relationship.blocked_reason = None
        relationship.updated_at = datetime.now(timezone.utc).isoformat()
        
        # Add memory for unblocking
        memory = RelationshipMemory(
            agent_id=agent_id,
            memory_type="unblocking",
            content=f"Agent unblocked. Previous reason: {blocked_reason}" +
                   (f". Notes: {notes}" if notes else ""),
            importance=4.5
        )
        relationship.add_memory(memory)
        
        # Auto-save if enabled
        if self.auto_save:
            self._save_relationship(agent_id)
            
        logger.info(f"Unblocked agent {agent_id}")
        
        return {
            "success": True,
            "new_status": relationship.status.value,
            "previous_reason": blocked_reason,
            "message": "Agent has been unblocked"
        }
    
    def get_all_relationships(
        self,
        status_filter: Optional[List[str]] = None,
        min_trust_level: Optional[int] = None,
        min_interactions: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get all relationships, optionally filtered.
        
        Args:
            status_filter: Optional list of status values to filter by.
            min_trust_level: Optional minimum trust level to filter by.
            min_interactions: Minimum number of interactions to filter by.
            
        Returns:
            List of relationship summaries.
        """
        results = []
        
        for agent_id, relationship in self.relationships.items():
            # Apply filters
            if status_filter and relationship.status.value not in status_filter:
                continue
                
            if min_trust_level is not None and relationship.trust_level.value < min_trust_level:
                continue
                
            if relationship.interaction_count < min_interactions:
                continue
                
            # Create summary
            summary = {
                "agent_id": agent_id,
                "agent_name": relationship.agent_name,
                "agent_type": relationship.agent_type.value,
                "status": relationship.status.value,
                "trust_level": relationship.trust_level.value,
                "trust_score": relationship.trust_score,
                "interaction_count": relationship.interaction_count,
                "first_interaction": relationship.first_interaction,
                "last_interaction": relationship.last_interaction,
                "memory_count": len(relationship.memories)
            }
            
            results.append(summary)
            
        # Sort by last interaction (most recent first)
        results.sort(
            key=lambda x: x["last_interaction"] or "0000-00-00T00:00:00Z", 
            reverse=True
        )
        
        return results