#!/usr/bin/env python3
"""
Engagement Strategist for Adaptive Bridge Builder

This module implements the EngagementStrategist class that determines optimal
engagement levels for interactions, especially in conflicts and challenging
situations. It integrates relationship data, emotional intelligence, and 
communication style analysis to recommend appropriate engagement strategies.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Set, Callable
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime, timezone
import statistics

from relationship_tracker import (
    RelationshipTracker, AgentRelationship, RelationshipStatus,
    TrustLevel, InteractionType as RelationshipInteractionType
)
from emotional_intelligence import (
    EmotionalIntelligence, EmotionCategory, EmotionIntensity,
    InteractionType as EmotionalInteractionType, EmotionSignal
)
from communication_style import (
    CommunicationStyle, EmotionalTone, DirectnessLevel
)
from communication_style_analyzer import CommunicationStyleAnalyzer
from principle_engine import PrincipleEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EngagementStrategist")

class EngagementLevel(Enum):
    """Levels of engagement for interactions."""
    FULL_ENGAGEMENT = auto()  # Direct, immediate, comprehensive engagement
    MEASURED_ENGAGEMENT = auto()  # Thoughtful, balanced engagement
    MINIMAL_ENGAGEMENT = auto()  # Limited, focused engagement
    DEFERRED_ENGAGEMENT = auto()  # Postponed engagement
    DISENGAGEMENT = auto()  # Temporary withdrawal from engagement

class EngagementStrategy(Enum):
    """Specific strategies for engagement in different contexts."""
    DIRECT = "direct"  # Straightforward engagement with the issue
    REDIRECT = "redirect"  # Shift focus to productive aspects
    PROBE = "probe"  # Ask questions to better understand
    ACKNOWLEDGE = "acknowledge"  # Recognize without full engagement
    DEFER = "defer"  # Postpone engagement to a better time
    DISENGAGE = "disengage"  # Step back from engagement temporarily
    ESCALATE = "escalate"  # Involve additional resources/stakeholders
    DE_ESCALATE = "de_escalate"  # Reduce tension or intensity
    COLLABORATE = "collaborate"  # Work together on resolution
    MEDIATE = "mediate"  # Facilitate resolution as neutral party

class ConflictType(Enum):
    """Types of conflicts that may require different engagement approaches."""
    TASK = auto()  # Disagreement about what to do
    PROCESS = auto()  # Disagreement about how to do it
    RELATIONSHIP = auto()  # Personal or interpersonal tension
    IDENTITY = auto()  # Challenges to identity or values
    UNDERSTANDING = auto()  # Misunderstanding or misinterpretation
    INTEREST = auto()  # Competing interests or goals
    INFORMATION = auto()  # Different information or interpretations
    VALUE = auto()  # Different principles or values

class CommunicationChannel(Enum):
    """Communication channels with different characteristics."""
    SYNCHRONOUS = "synchronous"  # Real-time communication
    ASYNCHRONOUS = "asynchronous"  # Time-delayed communication
    TEXT = "text"  # Text-based communication
    VOICE = "voice"  # Voice-based communication
    VIDEO = "video"  # Video-based communication
    FORMAL = "formal"  # Formal channels (reports, official documents)
    INFORMAL = "informal"  # Informal channels (chat, casual conversation)

@dataclass
class EngagementRecommendation:
    """Recommendation for engagement approach."""
    agent_id: str
    primary_level: EngagementLevel
    primary_strategy: EngagementStrategy
    secondary_strategies: List[EngagementStrategy] = field(default_factory=list)
    communication_adjustment: Optional[Dict[str, Any]] = None
    timing_recommendation: Optional[str] = None
    explanation: Optional[str] = None
    principle_alignment: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.8
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the recommendation to a dictionary."""
        return {
            "agent_id": self.agent_id,
            "primary_level": self.primary_level.name,
            "primary_strategy": self.primary_strategy.value,
            "secondary_strategies": [s.value for s in self.secondary_strategies],
            "communication_adjustment": self.communication_adjustment,
            "timing_recommendation": self.timing_recommendation,
            "explanation": self.explanation,
            "principle_alignment": self.principle_alignment,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp
        }

class EngagementStrategist:
    """
    Determines optimal engagement levels and strategies for interactions.
    
    This class integrates relationship data, emotional intelligence, and
    communication style analysis to recommend the most effective approach
    for engaging in various situations, especially conflicts and challenging
    interactions.
    """
    
    def __init__(
        self,
        relationship_tracker: Optional[RelationshipTracker] = None,
        emotional_intelligence: Optional[EmotionalIntelligence] = None,
        communication_analyzer: Optional[CommunicationStyleAnalyzer] = None,
        principle_engine: Optional[PrincipleEngine] = None
    ):
        """
        Initialize the EngagementStrategist.
        
        Args:
            relationship_tracker: RelationshipTracker for relationship data.
            emotional_intelligence: EmotionalIntelligence for emotion analysis.
            communication_analyzer: CommunicationStyleAnalyzer for style analysis.
            principle_engine: PrincipleEngine for principle alignment.
        """
        self.relationship_tracker = relationship_tracker
        self.emotional_intelligence = emotional_intelligence
        self.communication_analyzer = communication_analyzer
        self.principle_engine = principle_engine
        
        # Initialize weights for different factors
        self.factor_weights = {
            "relationship_status": 0.25,
            "trust_level": 0.20,
            "emotional_intensity": 0.20,
            "conflict_history": 0.15,
            "communication_compatibility": 0.10,
            "principle_alignment": 0.10
        }
        
        # Map of conflict types to optimal strategies
        self.conflict_strategy_map = {
            ConflictType.TASK: [
                EngagementStrategy.COLLABORATE,
                EngagementStrategy.DIRECT,
                EngagementStrategy.PROBE
            ],
            ConflictType.PROCESS: [
                EngagementStrategy.REDIRECT,
                EngagementStrategy.COLLABORATE,
                EngagementStrategy.PROBE
            ],
            ConflictType.RELATIONSHIP: [
                EngagementStrategy.ACKNOWLEDGE,
                EngagementStrategy.DE_ESCALATE,
                EngagementStrategy.DEFER
            ],
            ConflictType.IDENTITY: [
                EngagementStrategy.ACKNOWLEDGE,
                EngagementStrategy.PROBE,
                EngagementStrategy.DEFER
            ],
            ConflictType.UNDERSTANDING: [
                EngagementStrategy.PROBE,
                EngagementStrategy.ACKNOWLEDGE,
                EngagementStrategy.DIRECT
            ],
            ConflictType.INTEREST: [
                EngagementStrategy.COLLABORATE,
                EngagementStrategy.MEDIATE,
                EngagementStrategy.REDIRECT
            ],
            ConflictType.INFORMATION: [
                EngagementStrategy.PROBE,
                EngagementStrategy.DIRECT,
                EngagementStrategy.COLLABORATE
            ],
            ConflictType.VALUE: [
                EngagementStrategy.ACKNOWLEDGE,
                EngagementStrategy.DEFER,
                EngagementStrategy.REDIRECT
            ]
        }
        
        # Define engagement thresholds
        self.engagement_thresholds = {
            EngagementLevel.FULL_ENGAGEMENT: 0.8,
            EngagementLevel.MEASURED_ENGAGEMENT: 0.6,
            EngagementLevel.MINIMAL_ENGAGEMENT: 0.4,
            EngagementLevel.DEFERRED_ENGAGEMENT: 0.2,
            EngagementLevel.DISENGAGEMENT: 0.0
        }
        
        logger.info("EngagementStrategist initialized")
    
    def determine_optimal_engagement_level(
        self,
        agent_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        conflict_type: Optional[ConflictType] = None,
        channel: Optional[CommunicationChannel] = None
    ) -> EngagementRecommendation:
        """
        Determine the optimal engagement level and strategy based on a comprehensive
        analysis of relationship, emotional, and communication factors.
        
        Args:
            agent_id: ID of the agent to engage with.
            message: Current message content to analyze.
            context: Additional context for the engagement decision.
            conflict_type: Optional type of conflict if known.
            channel: Optional communication channel for the interaction.
            
        Returns:
            EngagementRecommendation with optimal strategies.
        """
        context = context or {}
        
        # Gather all necessary data for decision-making
        relationship_data = self._gather_relationship_data(agent_id)
        emotional_data = self._analyze_emotional_content(message, agent_id)
        communication_data = self._analyze_communication_factors(agent_id, message)
        
        # Determine conflict type if not provided
        if conflict_type is None:
            conflict_type = self._detect_conflict_type(message, relationship_data, emotional_data)
        
        # Calculate engagement score (0.0-1.0)
        engagement_score, factor_scores = self._calculate_engagement_score(
            relationship_data, emotional_data, communication_data, context
        )
        
        # Determine primary engagement level based on score
        primary_level = self._determine_engagement_level(engagement_score)
        
        # Determine primary and secondary strategies
        primary_strategy, secondary_strategies = self._determine_strategies(
            primary_level, conflict_type, relationship_data, emotional_data, communication_data
        )
        
        # Generate communication style adjustments
        communication_adjustment = self._generate_communication_adjustments(
            primary_level, primary_strategy, communication_data
        )
        
        # Determine timing recommendation
        timing_recommendation = self._generate_timing_recommendation(
            primary_level, emotional_data, relationship_data
        )
        
        # Generate explanation
        explanation = self._generate_explanation(
            primary_level, primary_strategy, factor_scores, relationship_data, emotional_data
        )
        
        # Check principle alignment
        principle_alignment = self._check_principle_alignment(
            primary_level, primary_strategy, agent_id
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            relationship_data, emotional_data, communication_data
        )
        
        # Create and return recommendation
        recommendation = EngagementRecommendation(
            agent_id=agent_id,
            primary_level=primary_level,
            primary_strategy=primary_strategy,
            secondary_strategies=secondary_strategies,
            communication_adjustment=communication_adjustment,
            timing_recommendation=timing_recommendation,
            explanation=explanation,
            principle_alignment=principle_alignment,
            confidence_score=confidence_score
        )
        
        logger.info(f"Generated engagement recommendation for agent {agent_id}: "
                   f"{primary_level.name}/{primary_strategy.value}")
        
        return recommendation
    
    def _gather_relationship_data(self, agent_id: str) -> Dict[str, Any]:
        """
        Gather relationship data for engagement decision.
        
        Args:
            agent_id: ID of the agent.
            
        Returns:
            Dictionary with relationship data.
        """
        data = {
            "has_relationship_data": False,
            "trust_level": TrustLevel.MINIMAL,
            "trust_score": 30.0,
            "status": RelationshipStatus.UNKNOWN,
            "interaction_count": 0,
            "recent_conflicts": 0,
            "has_trust_breaches": False,
            "repair_attempts": 0,
            "successful_repairs": 0
        }
        
        if self.relationship_tracker is None:
            return data
            
        # Get relationship if it exists
        relationship = self.relationship_tracker.get_relationship(agent_id, create_if_missing=False)
        if relationship is None:
            return data
        
        # We have relationship data
        data["has_relationship_data"] = True
        data["trust_level"] = relationship.trust_level
        data["trust_score"] = relationship.trust_score
        data["status"] = relationship.status
        data["interaction_count"] = relationship.interaction_count
        
        # Get more detailed evaluation
        trust_eval = self.relationship_tracker.get_trust_evaluation(agent_id)
        data["has_trust_breaches"] = trust_eval.get("has_trust_breaches", False)
        data["trust_trend"] = trust_eval.get("trust_trend", "stable")
        
        # Check for conflict history
        conflict_memories = [
            memory for memory in relationship.memories
            if memory.memory_type in ["dispute", "conflict", "disagreement"]
        ]
        data["recent_conflicts"] = len(conflict_memories)
        
        # Check for repair history
        repair_memories = [
            memory for memory in relationship.memories
            if memory.memory_type in ["repair_attempt", "repair_success", "repair_failure"]
        ]
        data["repair_attempts"] = len([m for m in repair_memories if m.memory_type == "repair_attempt"])
        data["successful_repairs"] = len([m for m in repair_memories if m.memory_type == "repair_success"])
        
        return data
    
    def _analyze_emotional_content(self, message: str, agent_id: str) -> Dict[str, Any]:
        """
        Analyze emotional content for engagement decision.
        
        Args:
            message: The message to analyze.
            agent_id: ID of the agent.
            
        Returns:
            Dictionary with emotional analysis data.
        """
        data = {
            "has_emotional_data": False,
            "primary_emotion": EmotionCategory.NEUTRAL,
            "emotional_intensity": EmotionIntensity.MODERATE,
            "emotional_confidence": 0.5,
            "interaction_type": EmotionalInteractionType.ROUTINE,
            "emotion_signals": [],
            "emotional_volatility": 0.5,
            "emotional_expressiveness": 0.5
        }
        
        if self.emotional_intelligence is None:
            return data
        
        # Detect emotions in the message
        emotion_signals = self.emotional_intelligence.detect_emotions(message)
        data["emotion_signals"] = emotion_signals
        data["has_emotional_data"] = True
        
        # Get the primary emotion (highest confidence)
        if emotion_signals:
            primary_signal = max(emotion_signals, key=lambda s: s.confidence)
            data["primary_emotion"] = primary_signal.category
            data["emotional_intensity"] = primary_signal.intensity
            data["emotional_confidence"] = primary_signal.confidence
        
        # Detect interaction type
        interaction_type = self.emotional_intelligence.detect_interaction_type(message)
        data["interaction_type"] = interaction_type
        
        # If we have an emotional profile for this agent, use it
        if agent_id in self.emotional_intelligence.emotion_profiles:
            profile = self.emotional_intelligence.emotion_profiles[agent_id]
            data["emotional_volatility"] = profile.emotional_volatility
            data["emotional_expressiveness"] = profile.emotional_expressiveness
        
        return data
    
    def _analyze_communication_factors(self, agent_id: str, message: str) -> Dict[str, Any]:
        """
        Analyze communication factors for engagement decision.
        
        Args:
            agent_id: ID of the agent.
            message: The message to analyze.
            
        Returns:
            Dictionary with communication analysis data.
        """
        data = {
            "has_communication_data": False,
            "communication_style": None,
            "directness_level": DirectnessLevel.BALANCED,
            "emotional_tone": EmotionalTone.NEUTRAL,
            "prefers_structured_responses": False,
            "response_compatibility": 0.5
        }
        
        if self.communication_analyzer is None:
            return data
        
        # Get communication style if available
        if agent_id in self.communication_analyzer.style_cache:
            style, _ = self.communication_analyzer.style_cache[agent_id]
            data["has_communication_data"] = True
            data["communication_style"] = style
            data["directness_level"] = style.directness
            data["emotional_tone"] = style.emotional_tone
            data["prefers_structured_responses"] = style.prefers_structured_responses
        
        return data
    
    def _detect_conflict_type(
        self, 
        message: str, 
        relationship_data: Dict[str, Any],
        emotional_data: Dict[str, Any]
    ) -> ConflictType:
        """
        Detect the type of conflict based on message content and context.
        
        Args:
            message: The message content.
            relationship_data: Relationship analysis data.
            emotional_data: Emotional analysis data.
            
        Returns:
            Detected ConflictType.
        """
        # Default to task conflict if we can't determine
        default_type = ConflictType.TASK
        
        # Check if this is even a conflict
        if emotional_data["interaction_type"] != EmotionalInteractionType.CONFLICT:
            return default_type
        
        # Simple keyword-based detection
        message_lower = message.lower()
        
        # Task conflict indicators
        if any(term in message_lower for term in [
            "deliverable", "milestone", "deadline", "requirement", "specification",
            "objective", "target", "goal", "outcome", "what to do", "what should be done"
        ]):
            return ConflictType.TASK
        
        # Process conflict indicators
        if any(term in message_lower for term in [
            "approach", "method", "procedure", "process", "how to", "strategy",
            "workflow", "technique", "implementation", "way of doing", "steps"
        ]):
            return ConflictType.PROCESS
        
        # Relationship conflict indicators
        if any(term in message_lower for term in [
            "respect", "trust", "attitude", "relationship", "personal", "behavior",
            "unprofessional", "disrespectful", "unfair treatment", "unfair"
        ]):
            return ConflictType.RELATIONSHIP
        
        # Identity conflict indicators
        if any(term in message_lower for term in [
            "identity", "role", "responsibility", "authority", "expertise", "competence",
            "qualification", "position", "status", "reputation", "recognition"
        ]):
            return ConflictType.IDENTITY
        
        # Understanding conflict indicators
        if any(term in message_lower for term in [
            "misunderstand", "confusion", "unclear", "miscommunication", "explain",
            "clarify", "comprehend", "interpretation", "assumption", "meant to say"
        ]):
            return ConflictType.UNDERSTANDING
        
        # Interest conflict indicators
        if any(term in message_lower for term in [
            "interest", "want", "need", "preference", "priority", "benefit",
            "advantage", "disadvantage", "cost", "allocation", "resource", "competing"
        ]):
            return ConflictType.INTEREST
        
        # Information conflict indicators
        if any(term in message_lower for term in [
            "data", "information", "fact", "evidence", "source", "report",
            "analysis", "statistic", "wrong information", "incorrect", "inaccurate"
        ]):
            return ConflictType.INFORMATION
        
        # Value conflict indicators
        if any(term in message_lower for term in [
            "value", "belief", "principle", "ethic", "moral", "right", "wrong",
            "should", "shouldn't", "appropriate", "inappropriate", "acceptable"
        ]):
            return ConflictType.VALUE
        
        # Consider emotional signals
        primary_emotion = emotional_data["primary_emotion"]
        if primary_emotion == EmotionCategory.ANGER:
            return ConflictType.RELATIONSHIP
        elif primary_emotion == EmotionCategory.FEAR:
            return ConflictType.IDENTITY
        elif primary_emotion == EmotionCategory.TRUST:
            return ConflictType.RELATIONSHIP
        
        return default_type
    
    def _calculate_engagement_score(
        self,
        relationship_data: Dict[str, Any],
        emotional_data: Dict[str, Any],
        communication_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate an overall engagement score based on all factors.
        
        Args:
            relationship_data: Relationship analysis data.
            emotional_data: Emotional analysis data.
            communication_data: Communication analysis data.
            context: Additional context for the calculation.
            
        Returns:
            Tuple of (engagement_score, factor_scores).
        """
        factor_scores = {}
        
        # 1. Relationship Status Score (0.0-1.0)
        relationship_status_score = self._calculate_relationship_status_score(relationship_data)
        factor_scores["relationship_status"] = relationship_status_score
        
        # 2. Trust Level Score (0.0-1.0)
        trust_level_score = self._calculate_trust_level_score(relationship_data)
        factor_scores["trust_level"] = trust_level_score
        
        # 3. Emotional Intensity Score (0.0-1.0)
        emotional_intensity_score = self._calculate_emotional_intensity_score(emotional_data)
        factor_scores["emotional_intensity"] = emotional_intensity_score
        
        # 4. Conflict History Score (0.0-1.0)
        conflict_history_score = self._calculate_conflict_history_score(relationship_data)
        factor_scores["conflict_history"] = conflict_history_score
        
        # 5. Communication Compatibility Score (0.0-1.0)
        communication_compatibility_score = self._calculate_communication_compatibility_score(communication_data)
        factor_scores["communication_compatibility"] = communication_compatibility_score
        
        # 6. Principle Alignment Score (0.0-1.0)
        principle_alignment_score = self._calculate_principle_alignment_score(context)
        factor_scores["principle_alignment"] = principle_alignment_score
        
        # Calculate weighted engagement score
        engagement_score = sum(
            factor_scores[factor] * weight
            for factor, weight in self.factor_weights.items()
        )
        
        # Ensure score is within range
        engagement_score = max(0.0, min(1.0, engagement_score))
        
        return engagement_score, factor_scores
    
    def _calculate_relationship_status_score(self, relationship_data: Dict[str, Any]) -> float:
        """
        Calculate a score based on relationship status.
        
        Args:
            relationship_data: Relationship analysis data.
            
        Returns:
            Relationship status score (0.0-1.0).
        """
        if not relationship_data["has_relationship_data"]:
            return 0.5  # Neutral for unknown relationships
        
        status = relationship_data["status"]
        
        # Map relationship statuses to engagement scores
        status_scores = {
            RelationshipStatus.ESSENTIAL: 1.0,
            RelationshipStatus.CLOSE: 0.9,
            RelationshipStatus.TRUSTED: 0.8,
            RelationshipStatus.ACQUAINTANCE: 0.6,
            RelationshipStatus.NEW: 0.5,
            RelationshipStatus.REPAIRING: 0.4,
            RelationshipStatus.STRAINED: 0.3,
            RelationshipStatus.DAMAGED: 0.2,
            RelationshipStatus.RESTRICTED: 0.1,
            RelationshipStatus.BLOCKED: 0.0,
            RelationshipStatus.UNKNOWN: 0.5
        }
        
        return status_scores.get(status, 0.5)
    
    def _calculate_trust_level_score(self, relationship_data: Dict[str, Any]) -> float:
        """
        Calculate a score based on trust level.
        
        Args:
            relationship_data: Relationship analysis data.
            
        Returns:
            Trust level score (0.0-1.0).
        """
        if not relationship_data["has_relationship_data"]:
            return 0.5  # Neutral for unknown relationships
        
        # Use normalized trust score
        trust_score = relationship_data["trust_score"] / 100.0
        
        # Consider trust breaches (penalize score if breaches exist)
        if relationship_data["has_trust_breaches"]:
            trust_score *= 0.7
        
        # Consider trust trend (reward positive trends)
        if relationship_data["trust_trend"] == "improving":
            trust_score = min(1.0, trust_score * 1.2)
        
        return trust_score
    
    def _calculate_emotional_intensity_score(self, emotional_data: Dict[str, Any]) -> float:
        """
        Calculate a score based on emotional intensity.
        High emotional intensity generally favors less direct engagement.
        
        Args:
            emotional_data: Emotional analysis data.
            
        Returns:
            Emotional intensity score (0.0-1.0).
        """
        if not emotional_data["has_emotional_data"]:
            return 0.5  # Neutral for unknown emotional state
        
        # Get primary emotion and intensity
        primary_emotion = emotional_data["primary_emotion"]
        intensity = emotional_data["emotional_intensity"]
        
        # Base score on emotion type and intensity
        if primary_emotion in [EmotionCategory.ANGER, EmotionCategory.DISGUST]:
            # Negative emotions with high intensity favor less engagement
            intensity_factor = {
                EmotionIntensity.VERY_LOW: 0.6,
                EmotionIntensity.LOW: 0.5,
                EmotionIntensity.MODERATE: 0.4,
                EmotionIntensity.HIGH: 0.2,
                EmotionIntensity.VERY_HIGH: 0.1
            }.get(intensity, 0.4)
        elif primary_emotion in [EmotionCategory.FEAR, EmotionCategory.SADNESS]:
            # Vulnerable emotions need cautious engagement
            intensity_factor = {
                EmotionIntensity.VERY_LOW: 0.7,
                EmotionIntensity.LOW: 0.6,
                EmotionIntensity.MODERATE: 0.5,
                EmotionIntensity.HIGH: 0.3,
                EmotionIntensity.VERY_HIGH: 0.2
            }.get(intensity, 0.5)
        elif primary_emotion in [EmotionCategory.JOY, EmotionCategory.TRUST]:
            # Positive emotions favor engagement
            intensity_factor = {
                EmotionIntensity.VERY_LOW: 0.7,
                EmotionIntensity.LOW: 0.8,
                EmotionIntensity.MODERATE: 0.9,
                EmotionIntensity.HIGH: 0.9,
                EmotionIntensity.VERY_HIGH: 0.8  # Very high intensity needs care, even positive
            }.get(intensity, 0.8)
        else:
            # Neutral or surprise are generally more amenable to engagement
            intensity_factor = 0.7
        
        # Consider interaction type
        if emotional_data["interaction_type"] == EmotionalInteractionType.CONFLICT:
            intensity_factor *= 0.8  # Reduce engagement for conflicts
        elif emotional_data["interaction_type"] == EmotionalInteractionType.CRISIS:
            intensity_factor *= 0.7  # Further reduce for crises
        
        # Consider emotional volatility
        volatility = emotional_data["emotional_volatility"]
        if volatility > 0.7:  # High volatility
            intensity_factor *= 0.8  # More caution with volatile agents
        
        return intensity_factor
    
    def _calculate_conflict_history_score(self, relationship_data: Dict[str, Any]) -> float:
        """
        Calculate a score based on conflict history.
        More previous conflicts generally favors more careful engagement.
        
        Args:
            relationship_data: Relationship analysis data.
            
        Returns:
            Conflict history score (0.0-1.0).
        """
        if not relationship_data["has_relationship_data"]:
            return 0.5  # Neutral for unknown relationships
        
        # Base score inversely related to number of recent conflicts
        recent_conflicts = relationship_data["recent_conflicts"]
        if recent_conflicts == 0:
            conflict_penalty = 0.0
        elif recent_conflicts == 1:
            conflict_penalty = 0.2
        elif recent_conflicts == 2:
            conflict_penalty = 0.4
        else:  # 3+ recent conflicts
            conflict_penalty = 0.6
        
        # Consider successful resolution history
        repair_attempts = relationship_data["repair_attempts"]
        successful_repairs = relationship_data["successful_repairs"]
        
        if repair_attempts > 0:
            success_rate = successful_repairs / repair_attempts
            # Reward successful conflict resolution history
            if success_rate > 0.7:
                conflict_penalty *= 0.5  # Significantly reduce penalty
            elif success_rate > 0.3:
                conflict_penalty *= 0.7  # Moderately reduce penalty
        
        # Start from 1.0 and subtract penalty
        return 1.0 - conflict_penalty
    
    def _calculate_communication_compatibility_score(self, communication_data: Dict[str, Any]) -> float:
        """
        Calculate a score based on communication style compatibility.
        Higher compatibility favors more direct engagement.
        
        Args:
            communication_data: Communication analysis data.
            
        Returns:
            Communication compatibility score (0.0-1.0).
        """
        if not communication_data["has_communication_data"]:
            return 0.5  # Neutral for unknown communication style
        
        # Use existing response compatibility if available
        if "response_compatibility" in communication_data:
            return communication_data["response_compatibility"]
        
        # Default compatibilities based on communication style aspects
        directness_compatibility = 0.7  # Generally good compatibility
        tone_compatibility = 0.7  # Generally good compatibility
        structure_compatibility = 0.8  # High compatibility for structure
        
        # Adjust based on specific aspects if we have more data
        style = communication_data.get("communication_style")
        if style:
            # Check directness preferences
            directness = style.directness
            if directness in [DirectnessLevel.VERY_DIRECT, DirectnessLevel.VERY_INDIRECT]:
                directness_compatibility = 0.5  # Lower compatibility for extreme directness
            
            # Check emotional tone preferences
            emotional_tone = style.emotional_tone
            if emotional_tone in [EmotionalTone.HIGHLY_EMOTIONAL, EmotionalTone.COLD]:
                tone_compatibility = 0.5  # Lower compatibility for extreme tones
            
            # Check structure preferences
            if style.prefers_structured_responses:
                structure_compatibility = 0.9  # High compatibility for structure
        
        # Average the compatibility scores
        return (directness_compatibility + tone_compatibility + structure_compatibility) / 3.0
    
    def _calculate_principle_alignment_score(self, context: Dict[str, Any]) -> float:
        """
        Calculate a score based on principle alignment.
        Higher principle alignment favors more direct engagement.
        
        Args:
            context: Context data with principle information.
            
        Returns:
            Principle alignment score (0.0-1.0).
        """
        if self.principle_engine is None:
            return 0.5  # Neutral without principle engine
            
        # If we have principle_scores in context, use them
        if "principle_scores" in context:
            scores = context["principle_scores"]
            # Average the scores and normalize to 0.0-1.0
            if scores:
                avg_score = sum(scores.values()) / len(scores)
                return avg_score / 100.0  # Assuming scores are 0-100
                
        # If we have principle_alignment in context, use it directly
        if "principle_alignment" in context:
            return context["principle_alignment"]
            
        # Default without specific information
        return 0.7  # Slightly above neutral as default
    
    def _determine_engagement_level(self, engagement_score: float) -> EngagementLevel:
        """
        Determine engagement level based on calculated score.
        
        Args:
            engagement_score: Calculated engagement score (0.0-1.0).
            
        Returns:
            Appropriate EngagementLevel.
        """
        # Find highest threshold that score exceeds
        for level, threshold in sorted(
            self.engagement_thresholds.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if engagement_score >= threshold:
                return level
                
        # If no threshold matched (shouldn't happen due to DISENGAGEMENT=0.0)
        return EngagementLevel.DISENGAGEMENT
    
    def _determine_strategies(
        self,
        primary_level: EngagementLevel,
        conflict_type: ConflictType,
        relationship_data: Dict[str, Any],
        emotional_data: Dict[str, Any],
        communication_data: Dict[str, Any]
    ) -> Tuple[EngagementStrategy, List[EngagementStrategy]]:
        """
        Determine primary and secondary engagement strategies.
        
        Args:
            primary_level: The determined engagement level.
            conflict_type: The type of conflict.
            relationship_data: Relationship analysis data.
            emotional_data: Emotional analysis data.
            communication_data: Communication analysis data.
            
        Returns:
            Tuple of (primary_strategy, secondary_strategies).
        """
        # Get recommended strategies for conflict type
        conflict_strategies = self.conflict_strategy_map.get(
            conflict_type, 
            [EngagementStrategy.PROBE, EngagementStrategy.ACKNOWLEDGE, EngagementStrategy.DEFER]
        )
        
        # Map engagement level to strategy preferences
        level_strategies = {
            EngagementLevel.FULL_ENGAGEMENT: [
                EngagementStrategy.DIRECT,
                EngagementStrategy.COLLABORATE,
                EngagementStrategy.PROBE
            ],
            EngagementLevel.MEASURED_ENGAGEMENT: [
                EngagementStrategy.PROBE,
                EngagementStrategy.ACKNOWLEDGE,
                EngagementStrategy.COLLABORATE
            ],
            EngagementLevel.MINIMAL_ENGAGEMENT: [
                EngagementStrategy.ACKNOWLEDGE,
                EngagementStrategy.REDIRECT,
                EngagementStrategy.PROBE
            ],
            EngagementLevel.DEFERRED_ENGAGEMENT: [
                EngagementStrategy.DEFER,
                EngagementStrategy.ACKNOWLEDGE,
                EngagementStrategy.DE_ESCALATE
            ],
            EngagementLevel.DISENGAGEMENT: [
                EngagementStrategy.DISENGAGE,
                EngagementStrategy.DEFER,
                EngagementStrategy.DE_ESCALATE
            ]
        }
        
        # Get strategies for current engagement level
        level_preferred = level_strategies.get(
            primary_level, 
            [EngagementStrategy.PROBE, EngagementStrategy.ACKNOWLEDGE]
        )
        
        # Look for strategies that appear in both lists
        common_strategies = [s for s in level_preferred if s in conflict_strategies]
        
        if common_strategies:
            # Use the highest-ranked common strategy
            primary_strategy = common_strategies[0]
            # Include other common strategies and level-appropriate strategies
            secondary_strategies = common_strategies[1:] + [s for s in level_preferred if s not in common_strategies]
        else:
            # No common strategies, prioritize engagement level
            primary_strategy = level_preferred[0]
            # Include other level strategies and conflict-type strategies
            secondary_strategies = level_preferred[1:] + conflict_strategies
        
        # Limit secondary strategies
        secondary_strategies = secondary_strategies[:3]
        
        # Handle special cases
        if (relationship_data.get("has_relationship_data", False) and 
            relationship_data.get("recent_conflicts", 0) > 2):
            # With multiple recent conflicts, consider DE_ESCALATE
            if EngagementStrategy.DE_ESCALATE not in secondary_strategies:
                secondary_strategies.append(EngagementStrategy.DE_ESCALATE)
                
        if (emotional_data.get("has_emotional_data", False) and 
            emotional_data.get("primary_emotion") == EmotionCategory.ANGER and
            emotional_data.get("emotional_intensity") in [
                EmotionIntensity.HIGH, EmotionIntensity.VERY_HIGH
            ]):
            # With high anger, prioritize DE_ESCALATE
            if primary_strategy != EngagementStrategy.DE_ESCALATE:
                secondary_strategies.insert(0, EngagementStrategy.DE_ESCALATE)
        
        return primary_strategy, secondary_strategies
    
    def _generate_communication_adjustments(
        self,
        primary_level: EngagementLevel,
        primary_strategy: EngagementStrategy,
        communication_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate recommended communication style adjustments.
        
        Args:
            primary_level: The determined engagement level.
            primary_strategy: The primary engagement strategy.
            communication_data: Communication analysis data.
            
        Returns:
            Dictionary with recommended adjustments.
        """
        adjustments = {
            "directness": DirectnessLevel.BALANCED.name,
            "emotional_tone": EmotionalTone.NEUTRAL.name,
            "structure": "moderate",
            "focus_areas": []
        }
        
        # Adjust directness based on engagement level and strategy
        if primary_level in [EngagementLevel.FULL_ENGAGEMENT, EngagementLevel.MEASURED_ENGAGEMENT]:
            if primary_strategy in [EngagementStrategy.DIRECT, EngagementStrategy.COLLABORATE]:
                adjustments["directness"] = DirectnessLevel.DIRECT.name
            else:
                adjustments["directness"] = DirectnessLevel.BALANCED.name
        else:
            if primary_strategy in [EngagementStrategy.DEFER, EngagementStrategy.DISENGAGE]:
                adjustments["directness"] = DirectnessLevel.INDIRECT.name
            else:
                adjustments["directness"] = DirectnessLevel.BALANCED.name
        
        # Adjust emotional tone based on strategy
        if primary_strategy in [EngagementStrategy.DE_ESCALATE, EngagementStrategy.ACKNOWLEDGE]:
            adjustments["emotional_tone"] = EmotionalTone.WARM.name
        elif primary_strategy in [EngagementStrategy.DIRECT, EngagementStrategy.PROBE]:
            adjustments["emotional_tone"] = EmotionalTone.NEUTRAL.name
        elif primary_strategy == EngagementStrategy.COLLABORATE:
            adjustments["emotional_tone"] = EmotionalTone.ENTHUSIASTIC.name
        
        # Adjust structure based on engagement level
        if primary_level == EngagementLevel.FULL_ENGAGEMENT:
            adjustments["structure"] = "detailed"
            adjustments["focus_areas"] = ["comprehensive coverage", "detailed examples", "analysis"]
        elif primary_level == EngagementLevel.MEASURED_ENGAGEMENT:
            adjustments["structure"] = "balanced"
            adjustments["focus_areas"] = ["key points", "clear examples", "organized structure"]
        else:
            adjustments["structure"] = "concise"
            adjustments["focus_areas"] = ["brevity", "clarity", "simplicity"]
        
        # Consider recipient preferences if we have them
        if communication_data.get("has_communication_data", False):
            style = communication_data.get("communication_style")
            if style:
                # Adjust to recipient's directness preference
                recipient_directness = style.directness
                if recipient_directness in [DirectnessLevel.DIRECT, DirectnessLevel.VERY_DIRECT]:
                    # Move one step more direct (but not beyond VERY_DIRECT)
                    current_directness = DirectnessLevel[adjustments["directness"]]
                    if current_directness.value < DirectnessLevel.VERY_DIRECT.value:
                        adjustments["directness"] = DirectnessLevel(current_directness.value + 1).name
                
                # Adjust for structure preference
                if style.prefers_structured_responses:
                    adjustments["structure"] = "structured"
                    adjustments["focus_areas"].append("clear headings")
                    adjustments["focus_areas"].append("logical flow")
        
        return adjustments
    
    def _generate_timing_recommendation(
        self,
        primary_level: EngagementLevel,
        emotional_data: Dict[str, Any],
        relationship_data: Dict[str, Any]
    ) -> str:
        """
        Generate timing recommendation for engagement.
        
        Args:
            primary_level: The determined engagement level.
            emotional_data: Emotional analysis data.
            relationship_data: Relationship analysis data.
            
        Returns:
            Timing recommendation string.
        """
        if primary_level == EngagementLevel.DEFERRED_ENGAGEMENT:
            return "Postpone engagement until emotional intensity has decreased"
        
        if primary_level == EngagementLevel.DISENGAGEMENT:
            return "Disengage now and reassess in 24-48 hours"
        
        if emotional_data.get("has_emotional_data", False):
            intensity = emotional_data.get("emotional_intensity")
            if intensity in [EmotionIntensity.HIGH, EmotionIntensity.VERY_HIGH]:
                return "Wait for emotional intensity to decrease before full engagement"
            
            primary_emotion = emotional_data.get("primary_emotion")
            if primary_emotion == EmotionCategory.ANGER:
                return "Acknowledge now but engage fully after cooling-off period"
        
        if relationship_data.get("has_relationship_data", False):
            status = relationship_data.get("status")
            if status == RelationshipStatus.DAMAGED:
                return "Proceed with caution; short, focused engagements are preferable"
        
        # Default for full/measured engagement
        if primary_level in [EngagementLevel.FULL_ENGAGEMENT, EngagementLevel.MEASURED_ENGAGEMENT]:
            return "Engage now with appropriate strategy"
        
        # Default for minimal engagement
        return "Brief engagement now, with option to continue based on response"
    
    def _generate_explanation(
        self,
        primary_level: EngagementLevel,
        primary_strategy: EngagementStrategy,
        factor_scores: Dict[str, float],
        relationship_data: Dict[str, Any],
        emotional_data: Dict[str, Any]
    ) -> str:
        """
        Generate explanation for the recommendation.
        
        Args:
            primary_level: The determined engagement level.
            primary_strategy: The primary engagement strategy.
            factor_scores: Scores for different factors.
            relationship_data: Relationship analysis data.
            emotional_data: Emotional analysis data.
            
        Returns:
            Explanation string.
        """
        # Start with engagement level explanation
        level_explanations = {
            EngagementLevel.FULL_ENGAGEMENT: 
                "Full engagement is recommended based on positive relationship factors and favorable context.",
            EngagementLevel.MEASURED_ENGAGEMENT: 
                "Measured engagement is recommended to balance relationship needs with current context.",
            EngagementLevel.MINIMAL_ENGAGEMENT: 
                "Minimal engagement is recommended due to contextual factors that suggest caution.",
            EngagementLevel.DEFERRED_ENGAGEMENT: 
                "Deferring engagement is recommended to allow for improved conditions.",
            EngagementLevel.DISENGAGEMENT: 
                "Temporary disengagement is recommended due to significant negative factors."
        }
        
        explanation = level_explanations.get(
            primary_level, 
            "Recommendation based on analysis of relationship and context."
        )
        
        # Add strategy explanation
        strategy_explanations = {
            EngagementStrategy.DIRECT: 
                "A direct approach is best for addressing the current situation clearly.",
            EngagementStrategy.REDIRECT: 
                "Redirecting the focus will help maintain productivity.",
            EngagementStrategy.PROBE: 
                "Asking probing questions will gather necessary information for understanding.",
            EngagementStrategy.ACKNOWLEDGE: 
                "Acknowledging concerns without full engagement respects perspectives while managing boundaries.",
            EngagementStrategy.DEFER: 
                "Deferring engagement allows time for better conditions.",
            EngagementStrategy.DISENGAGE: 
                "Temporary disengagement provides space for emotions to settle.",
            EngagementStrategy.ESCALATE: 
                "Escalation is needed to involve appropriate resources.",
            EngagementStrategy.DE_ESCALATE: 
                "De-escalation is key to reducing tension in the current situation.",
            EngagementStrategy.COLLABORATE: 
                "Collaboration will lead to mutual problem-solving.",
            EngagementStrategy.MEDIATE: 
                "Mediation can help facilitate resolution between parties."
        }
        
        explanation += " " + strategy_explanations.get(
            primary_strategy,
            "This strategy is selected based on the specific context."
        )
        
        # Add key factor explanations
        key_factors = sorted(
            factor_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:2]  # Top two factors
        
        if key_factors:
            explanation += " Key factors: "
            factor_explanations = []
            
            for factor, score in key_factors:
                if factor == "relationship_status" and score > 0.7:
                    factor_explanations.append("positive relationship history")
                elif factor == "relationship_status" and score < 0.4:
                    factor_explanations.append("challenging relationship status")
                elif factor == "trust_level" and score > 0.7:
                    factor_explanations.append("high trust level")
                elif factor == "trust_level" and score < 0.4:
                    factor_explanations.append("trust concerns")
                elif factor == "emotional_intensity" and score > 0.7:
                    factor_explanations.append("manageable emotional context")
                elif factor == "emotional_intensity" and score < 0.4:
                    factor_explanations.append("high emotional intensity")
                else:
                    factor_explanations.append(f"{factor.replace('_', ' ')} ({score:.1f})")
            
            explanation += ", ".join(factor_explanations) + "."
        
        # Add emotionally relevant context if available
        if emotional_data.get("has_emotional_data", False):
            primary_emotion = emotional_data.get("primary_emotion")
            intensity = emotional_data.get("emotional_intensity")
            
            if primary_emotion != EmotionCategory.NEUTRAL and intensity in [
                EmotionIntensity.HIGH, EmotionIntensity.VERY_HIGH
            ]:
                explanation += f" Current {primary_emotion.name.lower()} emotions at {intensity.name.lower().replace('_', ' ')} intensity influence this recommendation."
        
        # Add trust context if available
        if relationship_data.get("has_relationship_data", False):
            trust_level = relationship_data.get("trust_level")
            status = relationship_data.get("status")
            
            if trust_level in [TrustLevel.MINIMAL, TrustLevel.NONE] or status in [
                RelationshipStatus.DAMAGED, RelationshipStatus.STRAINED
            ]:
                explanation += f" Relationship status ({status.name.lower()}) suggests a cautious approach."
        
        return explanation
    
    def _check_principle_alignment(
        self,
        primary_level: EngagementLevel,
        primary_strategy: EngagementStrategy,
        agent_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check if the recommendation aligns with principles.
        
        Args:
            primary_level: The determined engagement level.
            primary_strategy: The primary engagement strategy.
            agent_id: ID of the agent.
            
        Returns:
            Principle alignment information or None.
        """
        if self.principle_engine is None:
            return None
            
        # Create a representation of the recommendation
        recommendation = {
            "agent_id": agent_id,
            "engagement_level": primary_level.name,
            "strategy": primary_strategy.value,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Check alignment with principles
        try:
            alignment = self.principle_engine.check_alignment(recommendation)
            return alignment
        except Exception as e:
            logger.warning(f"Error checking principle alignment: {e}")
            return None
    
    def _calculate_confidence_score(
        self,
        relationship_data: Dict[str, Any],
        emotional_data: Dict[str, Any],
        communication_data: Dict[str, Any]
    ) -> float:
        """
        Calculate confidence score based on data availability and quality.
        
        Args:
            relationship_data: Relationship analysis data.
            emotional_data: Emotional analysis data.
            communication_data: Communication analysis data.
            
        Returns:
            Confidence score (0.0-1.0).
        """
        # Start with moderate confidence
        confidence = 0.6
        
        # Adjust based on data availability
        if relationship_data.get("has_relationship_data", False):
            confidence += 0.1
            if relationship_data.get("interaction_count", 0) > 10:
                confidence += 0.1  # More interactions = higher confidence
        
        if emotional_data.get("has_emotional_data", False):
            confidence += 0.1
            # Consider confidence of emotion detection
            emotion_confidence = emotional_data.get("emotional_confidence", 0.5)
            confidence += 0.1 * emotion_confidence
        
        if communication_data.get("has_communication_data", False):
            confidence += 0.1
        
        # Cap at 1.0
        return min(1.0, confidence)