#!/usr/bin/env python3
"""
Conflict Engagement Module for Adaptive Bridge Builder

This module implements the engage_with_conflict function that extends the 
ConflictResolver's capabilities to proactively detect and engage with conflicts
rather than just resolving them after they've escalated. It applies the 
"Harmony Through Presence" principle by actively engaging with tensions 
as they emerge.
"""

import json
import logging
import copy
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union

from conflict_resolver import (
    ConflictResolver,
    ConflictType,
    ConflictSeverity,
    ConflictIndicator,
    ResolutionStrategy,
    ResolutionOutcome,
    ConflictResolutionStep,
    ConflictRecord
)
from relationship_tracker import (
    RelationshipTracker,
    RelationshipStatus,
    TrustLevel
)
from communication_style import CommunicationStyle
from communication_style_analyzer import CommunicationStyleAnalyzer
from emotional_intelligence import EmotionalIntelligence

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ConflictEngagement")

class EngagementType:
    """Types of engagement approaches."""
    CLARIFICATION = "clarification"
    REPHRASE = "rephrase"
    REFLECTION = "reflection"
    QUESTION = "question"
    MEDIATION = "mediation"
    ACKNOWLEDGMENT = "acknowledgment"
    PRINCIPLE_REMINDER = "principle_reminder"
    PERSPECTIVE_SHIFT = "perspective_shift"
    COMMON_GROUND = "common_ground"
    BREAK = "break"
    THIRD_PARTY = "third_party"

class MisunderstandingSign:
    """Signs that indicate a potential misunderstanding."""
    
    def __init__(
        self,
        name: str,
        description: str,
        pattern: str,
        severity: float = 0.5,
        engagement_types: List[str] = None
    ):
        """
        Initialize a misunderstanding sign.
        
        Args:
            name: Name of the sign
            description: Description of what this sign indicates
            pattern: Regex pattern to detect this sign
            severity: How severe the misunderstanding might be (0.0-1.0)
            engagement_types: Types of engagement recommended for this sign
        """
        self.name = name
        self.description = description
        self.pattern = pattern
        self.severity = severity
        self.engagement_types = engagement_types or [
            EngagementType.CLARIFICATION, 
            EngagementType.QUESTION
        ]

class EngagementAction:
    """
    A specific action to take to engage with a conflict or misunderstanding.
    """
    
    def __init__(
        self,
        engagement_type: str,
        content: str,
        context: Dict[str, Any] = None,
        confidence: float = 1.0,
        priority: int = 1
    ):
        """
        Initialize an engagement action.
        
        Args:
            engagement_type: Type of engagement (from EngagementType)
            content: The content of the engagement action (e.g., question text)
            context: Additional context about this action
            confidence: Confidence in the appropriateness of this action (0.0-1.0)
            priority: Priority of this action (higher number = higher priority)
        """
        self.engagement_type = engagement_type
        self.content = content
        self.context = context or {}
        self.confidence = confidence
        self.priority = priority
        self.action_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the engagement action to a dictionary."""
        return {
            "action_id": self.action_id,
            "engagement_type": self.engagement_type,
            "content": self.content,
            "context": self.context,
            "confidence": self.confidence,
            "priority": self.priority
        }

class EngagementPlan:
    """
    A plan for engaging with a conflict or misunderstanding.
    """
    
    def __init__(
        self,
        conflict_id: Optional[str] = None,
        conflict_type: Optional[ConflictType] = None,
        severity: float = 0.5,
        detected_signs: List[Dict[str, Any]] = None,
        context: Dict[str, Any] = None
    ):
        """
        Initialize an engagement plan.
        
        Args:
            conflict_id: ID of the related conflict (if one exists)
            conflict_type: Type of conflict or misunderstanding
            severity: Severity of the conflict or misunderstanding (0.0-1.0)
            detected_signs: Signs of conflict or misunderstanding detected
            context: Additional context about the situation
        """
        self.plan_id = str(uuid.uuid4())
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.conflict_id = conflict_id
        self.conflict_type = conflict_type
        self.severity = severity
        self.detected_signs = detected_signs or []
        self.context = context or {}
        self.primary_actions: List[EngagementAction] = []
        self.alternative_actions: List[EngagementAction] = []
        self.long_term_steps: List[Dict[str, Any]] = []
        self.explanation: str = ""
    
    def add_primary_action(self, action: EngagementAction) -> None:
        """Add a primary action to the plan."""
        self.primary_actions = [*self.primary_actions, action]
        # Keep primary actions sorted by priority (highest first)
        self.primary_actions.sort(key=lambda x: x.priority, reverse=True)
    
    def add_alternative_action(self, action: EngagementAction) -> None:
        """Add an alternative action to the plan."""
        self.alternative_actions = [*self.alternative_actions, action]
        # Keep alternative actions sorted by priority (highest first)
        self.alternative_actions.sort(key=lambda x: x.priority, reverse=True)
    
    def add_long_term_step(
        self,
        description: str,
        reasoning: str,
        expected_outcome: str,
        timeframe: Optional[str] = None
    ) -> None:
        """Add a long-term step to the plan."""
        self.long_term_steps.append({
            "description": description,
            "reasoning": reasoning,
            "expected_outcome": expected_outcome,
            "timeframe": timeframe
        })
    
    def set_explanation(self, explanation: str) -> None:
        """Set the explanation for the engagement plan."""
        self.explanation = explanation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the engagement plan to a dictionary."""
        return {
            "plan_id": self.plan_id,
            "created_at": self.created_at,
            "conflict_id": self.conflict_id,
            "conflict_type": self.conflict_type.value if self.conflict_type else None,
            "severity": self.severity,
            "detected_signs": self.detected_signs,
            "context": self.context,
            "primary_actions": [action.to_dict() for action in self.primary_actions],
            "alternative_actions": [action.to_dict() for action in self.alternative_actions],
            "long_term_steps": self.long_term_steps,
            "explanation": self.explanation
        }
    
    def get_best_action(self) -> Optional[EngagementAction]:
        """Get the highest priority primary action."""
        if self.primary_actions:
            return self.primary_actions[0]
        return None

def engage_with_conflict(
    conflict_resolver: ConflictResolver,
    message: Dict[str, Any],
    agent_id: str,
    conversation_id: Optional[str] = None,
    relationship_tracker: Optional[RelationshipTracker] = None,
    communication_analyzer: Optional[CommunicationStyleAnalyzer] = None,
    emotional_intelligence: Optional[EmotionalIntelligence] = None,
    context: Optional[Dict[str, Any]] = None
) -> EngagementPlan:
    """
    Detects signs of conflict or misunderstanding and generates an engagement plan
    for thoughtful intervention rather than avoidance.
    
    This function extends the ConflictResolver by focusing on early engagement
    with emerging conflicts, generating clarifying questions, rephrased information,
    and suggesting appropriate mediation steps.
    
    Args:
        conflict_resolver: ConflictResolver instance to work with
        message: The message to analyze for conflict signs
        agent_id: ID of the agent who sent the message
        conversation_id: ID of the ongoing conversation
        relationship_tracker: Optional RelationshipTracker for relationship context
        communication_analyzer: Optional CommunicationStyleAnalyzer for style matching
        emotional_intelligence: Optional EmotionalIntelligence for emotional awareness
        context: Additional context about the situation
        
    Returns:
        EngagementPlan containing actions and steps for addressing the conflict
    """
    # Initialize context
    context = context or {}
    
    # Step 1: Detect conflict indicators using ConflictResolver
    conflict_indicators = conflict_resolver.detect_conflicts(
        message=message,
        agent_id=agent_id,
        conversation_id=conversation_id
    )
    
    # Step 2: Detect additional signs of misunderstanding
    misunderstanding_signs = _detect_misunderstanding_signs(
        message=message,
        agent_id=agent_id
    )
    
    # Step 3: Analyze relationship context if tracker is available
    relationship_context = {}
    if relationship_tracker:
        relationship = relationship_tracker.get_relationship(agent_id)
        if relationship:
            relationship_context = {
                "status": relationship.status.value,
                "trust_level": relationship.trust_level.value,
                "trust_score": relationship.trust_score,
                "interaction_count": relationship.interaction_count,
                "recent_interactions": [
                    i.to_dict() for i in relationship.recent_interactions[:5]
                ] if hasattr(relationship, "recent_interactions") else []
            }
    
    # Step 4: Determine if we have an active conflict, potential conflict, or just misunderstanding
    engagement_plan = None
    
    # If we have significant conflict indicators, create or retrieve conflict record
    if conflict_indicators and any(i.severity in [ConflictSeverity.MODERATE, 
                                              ConflictSeverity.HIGH, 
                                              ConflictSeverity.CRITICAL] 
                              for i in conflict_indicators):
        # Create or retrieve conflict record
        conflict_record = _get_or_create_conflict_record(
            conflict_resolver=conflict_resolver,
            indicators=conflict_indicators,
            agent_id=agent_id,
            message=message,
            conversation_id=conversation_id,
            context=context
        )
        
        if conflict_record:
            # Create an engagement plan based on the conflict
            engagement_plan = _create_conflict_engagement_plan(
                conflict_resolver=conflict_resolver,
                conflict_record=conflict_record,
                message=message,
                relationship_context=relationship_context,
                communication_analyzer=communication_analyzer,
                emotional_intelligence=emotional_intelligence,
                context=context
            )
    
    # If no significant conflict but there are misunderstanding signs, create a misunderstanding engagement plan
    if not engagement_plan and misunderstanding_signs:
        engagement_plan = _create_misunderstanding_engagement_plan(
            misunderstanding_signs=misunderstanding_signs,
            message=message,
            agent_id=agent_id,
            relationship_context=relationship_context,
            communication_analyzer=communication_analyzer,
            emotional_intelligence=emotional_intelligence,
            context=context
        )
    
    # If we still don't have an engagement plan but have minor conflict indicators, create a minimal plan
    if not engagement_plan and conflict_indicators:
        engagement_plan = _create_minimal_engagement_plan(
            indicators=conflict_indicators,
            message=message,
            agent_id=agent_id,
            relationship_context=relationship_context,
            context=context
        )
    
    # If no signs of conflict or misunderstanding at all, create an empty plan
    if not engagement_plan:
        engagement_plan = EngagementPlan(
            severity=0.1,
            context={
                "agent_id": agent_id,
                "conversation_id": conversation_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message_id": message.get("id", "unknown")
            }
        )
        engagement_plan.set_explanation(
            "No signs of conflict or misunderstanding detected. Normal communication can continue."
        )
    
    return engagement_plan

def _detect_misunderstanding_signs(
    message: Dict[str, Any],
    agent_id: str
) -> List[Dict[str, Any]]:
    """
    Detect signs of misunderstanding that might not be full conflicts.
    
    Args:
        message: The message to analyze
        agent_id: ID of the agent who sent the message
        
    Returns:
        List of detected misunderstanding signs
    """
    # Initialize misunderstanding signs
    misunderstanding_signs = _initialize_misunderstanding_signs()
    
    # Extract message content
    content = _extract_message_content(message)
    if not content:
        return []
    
    # Check each sign against the message content
    detected_signs = []
    import re
    
    for sign in misunderstanding_signs:
        pattern = re.compile(sign.pattern, re.IGNORECASE)
        matches = pattern.findall(content)
        
        if matches:
            detected_signs.append({
                "name": sign.name,
                "description": sign.description,
                "matched_text": matches,
                "severity": sign.severity,
                "engagement_types": sign.engagement_types
            })
    
    return detected_signs

def _initialize_misunderstanding_signs() -> List[MisunderstandingSign]:
    """
    Initialize the set of misunderstanding signs to detect.
    
    Returns:
        List of MisunderstandingSign objects
    """
    signs = []
    
    # Confusion indicators
    signs.append(MisunderstandingSign(
        name="explicit_confusion",
        description="Explicit statement of confusion or lack of understanding",
        pattern=r"(confused|don't understand|not sure what you mean|unclear to me|what do you mean|not following)",
        severity=0.6,
        engagement_types=[
            EngagementType.CLARIFICATION,
            EngagementType.REPHRASE
        ]
    ))
    
    signs.append(MisunderstandingSign(
        name="request_clarification",
        description="Explicit request for clarification",
        pattern=r"(could you clarify|please explain|what exactly|can you elaborate|more details on|what specifically)",
        severity=0.5,
        engagement_types=[
            EngagementType.CLARIFICATION,
            EngagementType.REPHRASE
        ]
    ))
    
    # Misaligned expectations
    signs.append(MisunderstandingSign(
        name="expectation_mismatch",
        description="Indications of misaligned expectations",
        pattern=r"(expected|thought|assumed|was hoping|anticipated) (you|we) (would|to|that)",
        severity=0.6,
        engagement_types=[
            EngagementType.CLARIFICATION,
            EngagementType.ACKNOWLEDGMENT
        ]
    ))
    
    # Noncommittal responses
    signs.append(MisunderstandingSign(
        name="noncommittal",
        description="Vague or noncommittal response that may indicate misunderstanding",
        pattern=r"(I guess|perhaps|maybe|not really sure|if you say so|if that's what you want)",
        severity=0.4,
        engagement_types=[
            EngagementType.QUESTION,
            EngagementType.CLARIFICATION
        ]
    ))
    
    # Talking past each other
    signs.append(MisunderstandingSign(
        name="topic_shift",
        description="Abrupt topic shifts that may indicate misunderstanding",
        pattern=r"(anyway|moving on|let's talk about|changing the subject|back to|returning to)",
        severity=0.5,
        engagement_types=[
            EngagementType.REFLECTION,
            EngagementType.QUESTION
        ]
    ))
    
    # Repetition
    signs.append(MisunderstandingSign(
        name="repeated_explanation",
        description="Repeating the same explanation which may indicate misunderstanding",
        pattern=r"((as I said|as I mentioned|like I said|again|once more|to repeat),?)",
        severity=0.6,
        engagement_types=[
            EngagementType.REPHRASE,
            EngagementType.REFLECTION,
            EngagementType.PERSPECTIVE_SHIFT
        ]
    ))
    
    # Hesitation
    signs.append(MisunderstandingSign(
        name="hesitation",
        description="Hesitation or uncertainty that may indicate misunderstanding",
        pattern=r"(um|uh|hmm|well\.\.\.|so\.\.\.|actually,|honestly,|to be honest)",
        severity=0.3,
        engagement_types=[
            EngagementType.QUESTION,
            EngagementType.COMMON_GROUND
        ]
    ))
    
    # Defensive language
    signs.append(MisunderstandingSign(
        name="defensive_language",
        description="Defensive language that may indicate misunderstanding",
        pattern=r"(I never said|that's not what I meant|you're putting words|that's not fair|don't accuse)",
        severity=0.7,
        engagement_types=[
            EngagementType.REFLECTION,
            EngagementType.ACKNOWLEDGMENT
        ]
    ))
    
    return signs

def _extract_message_content(message: Dict[str, Any]) -> Optional[str]:
    """
    Extract content from a message for analysis.
    
    Args:
        message: The message to extract content from
        
    Returns:
        Extracted message content as string, or None if not extractable
    """
    # Extract from params field first
    params = message.get("params", {})
    if isinstance(params, dict):
        # Check common content fields
        for field in ["text", "content", "message", "data", "body"]:
            if field in params and isinstance(params[field], str):
                return params[field]
                
        # If data is a dict, convert to string
        if "data" in params and isinstance(params["data"], dict):
            return json.dumps(params["data"])
    
    # Check result field for responses
    result = message.get("result")
    if result:
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            return json.dumps(result)
    
    # Check error field for error messages
    error = message.get("error")
    if error:
        if isinstance(error, str):
            return error
        elif isinstance(error, dict) and "message" in error:
            return error["message"]
    
    # Try to extract from the message itself if it has text
    for field in ["text", "content", "message", "body"]:
        if field in message and isinstance(message[field], str):
            return message[field]
    
    # If no extractable content found
    return None

def _get_or_create_conflict_record(
    conflict_resolver: ConflictResolver,
    indicators: List[ConflictIndicator],
    agent_id: str,
    message: Dict[str, Any],
    conversation_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> Optional[ConflictRecord]:
    """
    Get an existing conflict record or create a new one if needed.
    
    Args:
        conflict_resolver: ConflictResolver instance
        indicators: List of conflict indicators
        agent_id: ID of the agent involved
        message: The message containing potential conflict
        conversation_id: ID of the conversation
        context: Additional context
        
    Returns:
        ConflictRecord if a conflict is found or created, None otherwise
    """
    # Check if there's an active conflict with this agent
    active_conflicts = conflict_resolver.get_active_conflicts(agent_id)
    
    if active_conflicts:
        # Use the most recent active conflict
        return conflict_resolver.active_conflicts[active_conflicts[0]["conflict_id"]]
    
    # Create a new conflict record
    return conflict_resolver.create_conflict_record(
        indicators=indicators,
        agent_id=agent_id,
        message=message,
        conversation_id=conversation_id,
        metadata=context
    )

def _create_minimal_engagement_plan(
    indicators: List[ConflictIndicator],
    message: Dict[str, Any],
    agent_id: str,
    relationship_context: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> EngagementPlan:
    """
    Create a minimal engagement plan for low-level conflict indicators.
    
    Args:
        indicators: List of conflict indicators
        message: The message being analyzed
        agent_id: ID of the agent who sent the message
        relationship_context: Context about the relationship with the agent
        context: Additional context
        
    Returns:
        EngagementPlan with minimal engagement actions
    """
    # Calculate severity from indicators
    severity_values = [0.3]  # Base minimal value
    for indicator in indicators:
        if indicator.severity == ConflictSeverity.MINIMAL:
            severity_values.append(0.2)
        elif indicator.severity == ConflictSeverity.LOW:
            severity_values.append(0.4)
        else:
            severity_values.append(0.5)
    
    severity = sum(severity_values) / len(severity_values)
    
    # Determine conflict type (use most common)
    type_counts = {}
    for indicator in indicators:
        conflict_type = indicator.conflict_type.value
        type_counts[conflict_type] = type_counts.get(conflict_type, 0) + 1
    
    most_common_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else "unknown"
    
    # Create the engagement plan
    plan = EngagementPlan(
        conflict_type=ConflictType(most_common_type) if most_common_type != "unknown" else None,
        severity=severity,
        detected_signs=[{
            "indicator_id": ind.indicator_id,
            "type": ind.conflict_type.value,
            "severity": ind.severity.value,
            "matched_text": ind.matched_text,
            "confidence": ind.confidence
        } for ind in indicators],
        context={
            "agent_id": agent_id,
            "message_id": message.get("id", "unknown"),
            "relationship_context": relationship_context,
            **(context or {})
        }
    )
    
    # Add lightweight engagement actions
    
    # Add a gentle clarification question
    clarification_content = "I want to make sure we're aligned. Could you help me understand your perspective on this matter?"
    plan.add_primary_action(EngagementAction(
        engagement_type=EngagementType.CLARIFICATION,
        content=clarification_content,
        priority=3
    ))
    
    # Add reflection on potential concerns
    reflection_content = "I noticed there might be some tension in our discussion. Let me reflect on what I understand from your perspective."
    plan.add_primary_action(EngagementAction(
        engagement_type=EngagementType.REFLECTION,
        content=reflection_content,
        priority=2
    ))
    
    # Add potential rephrasing as alternative
    rephrase_content = "Let me try to express my understanding differently to ensure we're on the same page."
    plan.add_alternative_action(EngagementAction(
        engagement_type=EngagementType.REPHRASE,
        content=rephrase_content,
        priority=1
    ))
    
    # Set explanation
    plan.set_explanation(
        f"Minor signs of potential {most_common_type} conflict detected with low severity. "
        "Taking a light-touch approach to ensure alignment and prevent escalation."
    )
    
    return plan

def _create_conflict_engagement_plan(
    conflict_resolver: ConflictResolver,
    conflict_record: ConflictRecord,
    message: Dict[str, Any],
    relationship_context: Dict[str, Any],
    communication_analyzer: Optional[CommunicationStyleAnalyzer] = None,
    emotional_intelligence: Optional[EmotionalIntelligence] = None,
    context: Optional[Dict[str, Any]] = None
) -> EngagementPlan:
    """
    Create an engagement plan for an active conflict.
    
    Args:
        conflict_resolver: ConflictResolver instance
        conflict_record: The conflict record to engage with
        message: The message being analyzed
        relationship_context: Context about the relationship with the agent
        communication_analyzer: Optional communication style analyzer
        emotional_intelligence: Optional emotional intelligence
        context: Additional context
        
    Returns:
        EngagementPlan for the conflict
    """
    # Create an engagement plan
    plan = EngagementPlan(
        conflict_id=conflict_record.conflict_id,
        conflict_type=conflict_record.conflict_type,
        severity=_convert_severity_to_float(conflict_record.severity),
        detected_signs=[ind.to_dict() for ind in conflict_record.indicators],
        context={
            "agent_id": conflict_record.agents[0] if conflict_record.agents[0] != conflict_resolver.agent_id else conflict_record.agents[1],
            "conversation_id": conflict_record.conversation_id,
            "conflict_status": conflict_record.status,
            "created_at": conflict_record.created_at,
            "relationship_context": relationship_context,
            **(context or {})
        }
    )
    
    # Get communication style preferences if analyzer is available
    communication_style = None
    if communication_analyzer:
        communication_style = communication_analyzer.analyze_message(message)
    
    # Get emotional context if emotional intelligence is available
    emotional_context = {}
    if emotional_intelligence:
        emotional_analysis = emotional_intelligence.analyze_message(message)
        if emotional_analysis:
            emotional_context = {
                "emotional_state": emotional_analysis.get("emotional_state", "neutral"),
                "dominant_emotions": emotional_analysis.get("dominant_emotions", []),
                "intensity": emotional_analysis.get("intensity", 0.5)
            }
    
    # Generate engagement actions based on conflict type
    if conflict_record.status == "detected":
        # Create resolution plan via conflict resolver
        resolution_plan = conflict_resolver.create_resolution_plan(conflict_record.conflict_id)
        
        # For newly detected conflicts, focus on acknowledgment and clarification
        _add_acknowledgment_actions(
            plan=plan, 
            conflict_record=conflict_record,
            communication_style=communication_style,
            emotional_context=emotional_context
        )
        
        _add_clarification_actions(
            plan=plan,
            conflict_record=conflict_record,
            communication_style=communication_style,
            emotional_context=emotional_context
        )
        
        # Add mediation suggestions for severe conflicts
        if conflict_record.severity in [ConflictSeverity.HIGH, ConflictSeverity.CRITICAL]:
            _add_mediation_actions(
                plan=plan,
                conflict_record=conflict_record,
                relationship_context=relationship_context
            )
    
    elif conflict_record.status == "planning":
        # For conflicts in planning phase, focus on implementing the first steps of the plan
        next_steps = conflict_record.get_next_actionable_steps()
        
        if next_steps:
            for step in next_steps[:2]:  # Focus on the first two steps only
                _add_step_implementation_actions(
                    plan=plan,
                    conflict_record=conflict_record,
                    step=step,
                    communication_style=communication_style,
                    emotional_context=emotional_context
                )
    
    elif conflict_record.status in ["implementing", "active"]:
        # For active conflicts, focus on continuing implementation and progress checks
        next_steps = conflict_record.get_next_actionable_steps()
        
        if next_steps:
            # Focus on the current step
            _add_step_implementation_actions(
                plan=plan,
                conflict_record=conflict_record,
                step=next_steps[0],
                communication_style=communication_style,
                emotional_context=emotional_context
            )
        
        # Add progress reflection
        progress = conflict_record.get_resolution_progress()
        _add_progress_reflection_actions(
            plan=plan,
            conflict_record=conflict_record,
            progress=progress
        )
    
    # Add common ground actions for any active conflict
    _add_common_ground_actions(
        plan=plan,
        conflict_record=conflict_record,
        relationship_context=relationship_context
    )
    
    # Generate explanation
    plan.set_explanation(_generate_conflict_engagement_explanation(
        conflict_record=conflict_record,
        message=message,
        plan=plan
    ))
    
    return plan

def _create_misunderstanding_engagement_plan(
    misunderstanding_signs: List[Dict[str, Any]],
    message: Dict[str, Any],
    agent_id: str,
    relationship_context: Dict[str, Any],
    communication_analyzer: Optional[CommunicationStyleAnalyzer] = None,
    emotional_intelligence: Optional[EmotionalIntelligence] = None,
    context: Optional[Dict[str, Any]] = None
) -> EngagementPlan:
    """
    Create an engagement plan for a potential misunderstanding.
    
    Args:
        misunderstanding_signs: List of detected misunderstanding signs
        message: The message being analyzed
        agent_id: ID of the agent who sent the message
        relationship_context: Context about the relationship with the agent
        communication_analyzer: Optional communication style analyzer
        emotional_intelligence: Optional emotional intelligence
        context: Additional context
        
    Returns:
        EngagementPlan for the misunderstanding
    """
    # Calculate overall severity
    severity = max([sign["severity"] for sign in misunderstanding_signs], default=0.3)
    
    # Create engagement plan
    plan = EngagementPlan(
        severity=severity,
        detected_signs=misunderstanding_signs,
        context={
            "agent_id": agent_id,
            "message_id": message.get("id", "unknown"),
            "relationship_context": relationship_context,
            **(context or {})
        }
    )
    
    # Get communication style preferences if analyzer is available
    communication_style = None
    if communication_analyzer:
        communication_style = communication_analyzer.analyze_