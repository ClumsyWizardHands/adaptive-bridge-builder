#!/usr/bin/env python3
"""
Conflict Resolver for Adaptive Bridge Builder

This module implements the ConflictResolver class that detects, categorizes,
and resolves conflicts between agents. It applies the "Harmony Through Presence"
principle by actively detecting tensions in communication, categorizing the conflict
type, and applying appropriate resolution strategies.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum, auto
import re
import random

from principle_engine import PrincipleEngine
from relationship_tracker import (
    RelationshipTracker, 
    RelationshipStatus, 
    TrustLevel,
    InteractionType,
    InteractionQuality,
    RelationshipMemory
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ConflictResolver")

class ConflictType(Enum):
    """Enumeration of possible conflict types."""
    GOAL = "goal"                 # Conflicting objectives or desired outcomes
    VALUE = "value"               # Differing core values or principles
    FACTUAL = "factual"           # Disagreement about facts or data
    PROCEDURAL = "procedural"     # Differing views on proper processes or methods
    RELATIONSHIP = "relationship" # Interpersonal tensions or misunderstandings
    RESOURCE = "resource"         # Competition over limited resources
    COMMUNICATION = "communication" # Misunderstandings in message exchange
    COMPLIANCE = "compliance"     # Disagreement over rule adherence
    AUTHORITY = "authority"       # Disputes over decision-making authority
    UNKNOWN = "unknown"           # Unclassified conflict

class ConflictSeverity(Enum):
    """Enumeration of conflict severity levels."""
    MINIMAL = "minimal"           # Minor disagreement, easy to resolve
    LOW = "low"                   # Notable disagreement but still manageable
    MODERATE = "moderate"         # Significant disagreement requiring attention
    HIGH = "high"                 # Serious conflict threatening cooperation
    CRITICAL = "critical"         # Extreme conflict endangering the relationship

class ResolutionStrategy(Enum):
    """Enumeration of conflict resolution strategies."""
    COMMON_GROUND = "common_ground"   # Finding shared interests and values
    COMPROMISE = "compromise"         # Both parties give up something
    COLLABORATIVE = "collaborative"   # Working together on a mutual solution
    ACCOMMODATING = "accommodating"   # One party concedes to the other
    AVOIDING = "avoiding"             # Temporarily deferring the conflict
    COMPETING = "competing"           # Asserting one's position firmly
    THIRD_PARTY = "third_party"       # Involving a neutral mediator
    PRINCIPLE_BASED = "principle_based" # Resolving based on shared principles
    FACT_FINDING = "fact_finding"     # Gathering additional information
    TEMPORAL_DISTANCE = "temporal_distance" # Creating temporal separation
    STRUCTURAL_DISTANCE = "structural_distance" # Creating process separation
    COMMUNICATION_IMPROVEMENT = "communication_improvement" # Enhancing clarity

class ResolutionOutcome(Enum):
    """Enumeration of conflict resolution outcomes."""
    RESOLVED = "resolved"         # Conflict fully resolved
    MITIGATED = "mitigated"       # Conflict reduced but not fully resolved
    MANAGED = "managed"           # Conflict contained with ongoing attention
    DEFERRED = "deferred"         # Resolution postponed to a later time
    ESCALATED = "escalated"       # Conflict became more severe
    UNCHANGED = "unchanged"       # No change in conflict status
    DISTANCED = "distanced"       # Parties separated to prevent further conflict

class ConflictTrigger:
    """Represents a pattern that may indicate a conflict."""
    
    def __init__(
        self,
        name: str,
        pattern: str,
        conflict_type: ConflictType,
        severity: ConflictSeverity,
        confidence: float = 1.0
    ):
        """
        Initialize a conflict trigger.
        
        Args:
            name: Descriptive name of the trigger.
            pattern: Regex pattern to detect in messages.
            conflict_type: Type of conflict this trigger indicates.
            severity: Severity level this trigger suggests.
            confidence: Confidence level in this trigger (0.0-1.0).
        """
        self.name = name
        self.pattern = pattern
        self.conflict_type = conflict_type
        self.severity = severity
        self.confidence = confidence
        self.regex = re.compile(pattern, re.IGNORECASE)
    
    def check(self, text: str) -> bool:
        """
        Check if the pattern matches the given text.
        
        Args:
            text: Text to check against the trigger pattern.
            
        Returns:
            True if the pattern matches, False otherwise.
        """
        return bool(self.regex.search(text))

class ConflictIndicator:
    """Collected evidence of a potential conflict."""
    
    def __init__(
        self,
        agent_id: str,
        message_id: str,
        conversation_id: str,
        timestamp: str,
        trigger_name: str,
        matched_text: str,
        conflict_type: ConflictType,
        severity: ConflictSeverity,
        confidence: float
    ):
        """
        Initialize a conflict indicator.
        
        Args:
            agent_id: ID of the agent involved.
            message_id: ID of the message containing the indicator.
            conversation_id: ID of the conversation context.
            timestamp: When the indicator was detected.
            trigger_name: Name of the trigger that matched.
            matched_text: Text that matched the trigger pattern.
            conflict_type: Type of conflict indicated.
            severity: Severity level of the potential conflict.
            confidence: Confidence in this indicator (0.0-1.0).
        """
        self.agent_id = agent_id
        self.message_id = message_id
        self.conversation_id = conversation_id
        self.timestamp = timestamp
        self.trigger_name = trigger_name
        self.matched_text = matched_text
        self.conflict_type = conflict_type
        self.severity = severity
        self.confidence = confidence
        self.indicator_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the conflict indicator to a dictionary."""
        return {
            "indicator_id": self.indicator_id,
            "agent_id": self.agent_id,
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "timestamp": self.timestamp,
            "trigger_name": self.trigger_name,
            "matched_text": self.matched_text,
            "conflict_type": self.conflict_type.value,
            "severity": self.severity.value,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConflictIndicator':
        """Create a ConflictIndicator from a dictionary."""
        indicator = cls(
            agent_id=data.get("agent_id", ""),
            message_id=data.get("message_id", ""),
            conversation_id=data.get("conversation_id", ""),
            timestamp=data.get("timestamp", ""),
            trigger_name=data.get("trigger_name", ""),
            matched_text=data.get("matched_text", ""),
            conflict_type=ConflictType(data.get("conflict_type", "unknown")),
            severity=ConflictSeverity(data.get("severity", "minimal")),
            confidence=data.get("confidence", 0.5)
        )
        indicator.indicator_id = data.get("indicator_id", indicator.indicator_id)
        return indicator

class ConflictResolutionStep:
    """A step in a conflict resolution plan."""
    
    def __init__(
        self,
        step_type: str,
        description: str,
        strategy: ResolutionStrategy,
        expected_outcome: str,
        required_resources: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        completion_criteria: Optional[str] = None
    ):
        """
        Initialize a conflict resolution step.
        
        Args:
            step_type: Type of step (e.g., "clarification", "negotiation").
            description: Detailed description of the step.
            strategy: Resolution strategy this step employs.
            expected_outcome: What this step aims to achieve.
            required_resources: Resources needed for this step.
            dependencies: IDs of steps that must be completed first.
            completion_criteria: How to determine if this step is complete.
        """
        self.step_type = step_type
        self.description = description
        self.strategy = strategy
        self.expected_outcome = expected_outcome
        self.required_resources = required_resources or []
        self.dependencies = dependencies or []
        self.completion_criteria = completion_criteria
        self.step_id = str(uuid.uuid4())
        self.status = "pending"
        self.notes = ""
        self.completed_at = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the conflict resolution step to a dictionary."""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "description": self.description,
            "strategy": self.strategy.value,
            "expected_outcome": self.expected_outcome,
            "required_resources": self.required_resources,
            "dependencies": self.dependencies,
            "completion_criteria": self.completion_criteria,
            "status": self.status,
            "notes": self.notes,
            "completed_at": self.completed_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConflictResolutionStep':
        """Create a ConflictResolutionStep from a dictionary."""
        step = cls(
            step_type=data.get("step_type", ""),
            description=data.get("description", ""),
            strategy=ResolutionStrategy(data.get("strategy", "common_ground")),
            expected_outcome=data.get("expected_outcome", ""),
            required_resources=data.get("required_resources", []),
            dependencies=data.get("dependencies", []),
            completion_criteria=data.get("completion_criteria")
        )
        step.step_id = data.get("step_id", step.step_id)
        step.status = data.get("status", "pending")
        step.notes = data.get("notes", "")
        step.completed_at = data.get("completed_at")
        return step
    
    def complete(self, notes: Optional[str] = None) -> None:
        """
        Mark this step as completed.
        
        Args:
            notes: Optional notes about the completion.
        """
        self.status = "completed"
        if notes:
            self.notes += f"\n{notes}"
        self.completed_at = datetime.utcnow().isoformat()

class ConflictRecord:
    """A record of a conflict and its resolution process."""
    
    def __init__(
        self,
        agents: List[str],
        conflict_type: ConflictType,
        severity: ConflictSeverity,
        description: str,
        indicators: Optional[List[ConflictIndicator]] = None,
        conversation_id: Optional[str] = None,
        resolution_plan: Optional[List[ConflictResolutionStep]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a conflict record.
        
        Args:
            agents: IDs of agents involved in the conflict.
            conflict_type: Type of conflict.
            severity: Severity level of the conflict.
            description: Description of the conflict.
            indicators: List of conflict indicators that led to detection.
            conversation_id: ID of the conversation where conflict was detected.
            resolution_plan: Plan for resolving the conflict.
            metadata: Additional metadata about the conflict.
        """
        self.agents = agents
        self.conflict_type = conflict_type
        self.severity = severity
        self.description = description
        self.indicators = indicators or []
        self.conversation_id = conversation_id
        self.resolution_plan = resolution_plan or []
        self.metadata = metadata or {}
        self.conflict_id = str(uuid.uuid4())
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = self.created_at
        self.status = "detected"
        self.outcome = None
        self.resolution_notes = ""
        self.resolution_timestamp = None
        self.principle_alignment = None
        self.resolution_metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the conflict record to a dictionary."""
        return {
            "conflict_id": self.conflict_id,
            "agents": self.agents,
            "conflict_type": self.conflict_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "indicators": [ind.to_dict() for ind in self.indicators],
            "conversation_id": self.conversation_id,
            "resolution_plan": [step.to_dict() for step in self.resolution_plan],
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "outcome": self.outcome.value if self.outcome else None,
            "resolution_notes": self.resolution_notes,
            "resolution_timestamp": self.resolution_timestamp,
            "principle_alignment": self.principle_alignment,
            "resolution_metrics": self.resolution_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConflictRecord':
        """Create a ConflictRecord from a dictionary."""
        conflict = cls(
            agents=data.get("agents", []),
            conflict_type=ConflictType(data.get("conflict_type", "unknown")),
            severity=ConflictSeverity(data.get("severity", "minimal")),
            description=data.get("description", ""),
            indicators=[
                ConflictIndicator.from_dict(ind) for ind in data.get("indicators", [])
            ],
            conversation_id=data.get("conversation_id"),
            resolution_plan=[
                ConflictResolutionStep.from_dict(step) for step in data.get("resolution_plan", [])
            ],
            metadata=data.get("metadata", {})
        )
        conflict.conflict_id = data.get("conflict_id", conflict.conflict_id)
        conflict.created_at = data.get("created_at", conflict.created_at)
        conflict.updated_at = data.get("updated_at", conflict.updated_at)
        conflict.status = data.get("status", "detected")
        conflict.outcome = ResolutionOutcome(data.get("outcome")) if data.get("outcome") else None
        conflict.resolution_notes = data.get("resolution_notes", "")
        conflict.resolution_timestamp = data.get("resolution_timestamp")
        conflict.principle_alignment = data.get("principle_alignment")
        conflict.resolution_metrics = data.get("resolution_metrics", {})
        return conflict
    
    def create_resolution_plan(
        self,
        conflict_id: str
    ) -> List[ConflictResolutionStep]:
        """
        Create a resolution plan for a conflict.
        
        Args:
            conflict_id: ID of the conflict to create a plan for.
            
        Returns:
            List of resolution steps, or empty list if no conflict found.
        """
        if conflict_id not in self.active_conflicts:
            logger.warning(f"Cannot create resolution plan: conflict {conflict_id} not found")
            return []
            
        conflict = self.active_conflicts[conflict_id]
        
        # Get resolution strategies for this conflict type
        strategies = self._get_strategies_for_conflict_type(conflict.conflict_type)
        
        # Create resolution steps
        resolution_plan = []
        
        # Step 1: Always start with acknowledgment
        acknowledgment_step = ConflictResolutionStep(
            step_type="acknowledgment",
            description="Acknowledge the existence of the conflict and validate the other agent's perspective",
            strategy=ResolutionStrategy.COMMON_GROUND,
            expected_outcome="Establish mutual recognition of the conflict situation",
            completion_criteria="Both parties acknowledge the existence of a conflict"
        )
        resolution_plan.append(acknowledgment_step)
        
        # Step 2: Information gathering/clarification
        clarification_step = ConflictResolutionStep(
            step_type="clarification",
            description="Gather more information about the nature and extent of the conflict",
            strategy=ResolutionStrategy.FACT_FINDING,
            expected_outcome="Clear understanding of all parties' positions and concerns",
            dependencies=[acknowledgment_step.step_id],
            completion_criteria="All relevant information about the conflict has been collected"
        )
        resolution_plan.append(clarification_step)
        
        # Step 3: Add specific strategies based on conflict type
        for strategy_info in strategies:
            strategy = strategy_info["strategy"]
            step_type = strategy_info["step_type"]
            description = strategy_info["description"]
            expected_outcome = strategy_info["expected_outcome"]
            
            resolution_step = ConflictResolutionStep(
                step_type=step_type,
                description=description,
                strategy=strategy,
                expected_outcome=expected_outcome,
                dependencies=[clarification_step.step_id],
                completion_criteria=f"The {step_type} process is complete and has achieved its intended outcome"
            )
            resolution_plan.append(resolution_step)
        
        # Step 4: Always end with verification and closure
        verification_step = ConflictResolutionStep(
            step_type="verification",
            description="Verify that the conflict has been adequately addressed",
            strategy=ResolutionStrategy.COMMON_GROUND,
            expected_outcome="Mutual agreement that the conflict is resolved or appropriately managed",
            dependencies=[step.step_id for step in resolution_plan[2:]],  # Depends on all specific strategy steps
            completion_criteria="All parties agree the conflict is resolved or appropriately managed"
        )
        resolution_plan.append(verification_step)
        
        # Add the plan to the conflict record
        conflict.resolution_plan = resolution_plan
        conflict.update_status("planning", "Resolution plan created")
        
        logger.info(f"Created resolution plan for conflict {conflict_id} with {len(resolution_plan)} steps")
        
        return resolution_plan
    
    def update_status(self, status: str, notes: Optional[str] = None) -> None:
        """
        Update the status of this conflict.
        
        Args:
            status: New status string.
            notes: Optional notes about the status change.
        """
        self.status = status
        self.updated_at = datetime.utcnow().isoformat()
        if notes:
            self.resolution_notes += f"\n[{self.updated_at}] {notes}"
    
    def complete_resolution(
        self,
        outcome: ResolutionOutcome,
        notes: str,
        principle_alignment: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Mark this conflict as resolved.
        
        Args:
            outcome: The resolution outcome.
            notes: Notes about the resolution.
            principle_alignment: How well the resolution aligned with principles.
            metrics: Metrics about the resolution process.
        """
        self.status = "resolved"
        self.outcome = outcome
        self.resolution_notes += f"\n[{datetime.utcnow().isoformat()}] {notes}"
        self.resolution_timestamp = datetime.utcnow().isoformat()
        self.principle_alignment = principle_alignment
        self.resolution_metrics = metrics or {}
        self.updated_at = datetime.utcnow().isoformat()
    
    def add_resolution_step(self, step: ConflictResolutionStep) -> None:
        """
        Add a resolution step to the plan.
        
        Args:
            step: The resolution step to add.
        """
        self.resolution_plan.append(step)
        self.updated_at = datetime.utcnow().isoformat()
    
    def get_pending_steps(self) -> List[ConflictResolutionStep]:
        """
        Get all pending resolution steps.
        
        Returns:
            List of pending resolution steps.
        """
        return [step for step in self.resolution_plan if step.status == "pending"]
    
    def get_next_actionable_steps(self) -> List[ConflictResolutionStep]:
        """
        Get steps that can be acted upon now.
        
        Returns:
            List of actionable resolution steps.
        """
        completed_ids = {step.step_id for step in self.resolution_plan 
                        if step.status == "completed"}
        
        actionable = []
        for step in self.resolution_plan:
            if step.status != "pending":
                continue
                
            dependencies_met = all(dep in completed_ids for dep in step.dependencies)
            if dependencies_met:
                actionable.append(step)
                
        return actionable
    
    def get_resolution_progress(self) -> float:
        """
        Calculate the resolution progress as a percentage.
        
        Returns:
            Percentage of completion (0.0-1.0).
        """
        if not self.resolution_plan:
            return 0.0
            
        completed = sum(1 for step in self.resolution_plan 
                      if step.status == "completed")
        return completed / len(self.resolution_plan)

class ConflictResolver:
    """
    Detects, categorizes, and resolves conflicts between agents.
    
    This class implements the "Harmony Through Presence" principle by actively
    monitoring for signs of conflict, categorizing the nature of disagreements,
    and applying appropriate resolution strategies. It maintains a record of 
    conflicts and resolutions to improve future conflict handling.
    """
    
    def __init__(
        self,
        agent_id: str,
        principle_engine: Optional[PrincipleEngine] = None,
        relationship_tracker: Optional[RelationshipTracker] = None,
        data_dir: Optional[str] = None
    ):
        """
        Initialize the ConflictResolver.
        
        Args:
            agent_id: ID of this agent.
            principle_engine: Optional PrincipleEngine for principle alignment.
            relationship_tracker: Optional RelationshipTracker for relationship context.
            data_dir: Directory for storing conflict data.
        """
        self.agent_id = agent_id
        self.principle_engine = principle_engine
        self.relationship_tracker = relationship_tracker
        self.data_dir = data_dir or "data/conflicts"
        
        # Initialize conflict triggers
        self.conflict_triggers = self._initialize_conflict_triggers()
        
        # Storage for active conflicts
        self.active_conflicts: Dict[str, ConflictRecord] = {}
        
        # Storage for conflict history
        self.resolved_conflicts: Dict[str, ConflictRecord] = {}
        
        # Storage for agent-specific conflict patterns
        self.agent_conflict_patterns: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"ConflictResolver initialized for agent {agent_id}")
    
    def _initialize_conflict_triggers(self) -> List[ConflictTrigger]:
        """
        Initialize the set of conflict triggers.
        
        Returns:
            List of ConflictTrigger objects.
        """
        triggers = []
        
        # Goal conflicts
        triggers.append(ConflictTrigger(
            name="opposing_objectives",
            pattern=r"(disagree|oppose|contrary|different goal|conflicting objective|incompatible|at odds)",
            conflict_type=ConflictType.GOAL,
            severity=ConflictSeverity.MODERATE,
            confidence=0.7
        ))
        triggers.append(ConflictTrigger(
            name="explicit_blocking",
            pattern=r"(block|prevent|stop|prohibit|forbid|disallow|reject)\s.*(from|action|doing)",
            conflict_type=ConflictType.GOAL,
            severity=ConflictSeverity.HIGH,
            confidence=0.8
        ))
        
        # Value conflicts
        triggers.append(ConflictTrigger(
            name="principle_disagreement",
            pattern=r"(against|violates|contradicts|disagree with)\s.*(principle|value|belief|ethic|moral)",
            conflict_type=ConflictType.VALUE,
            severity=ConflictSeverity.HIGH,
            confidence=0.9
        ))
        triggers.append(ConflictTrigger(
            name="ethical_concerns",
            pattern=r"(ethic|moral|right|wrong|should not|ought not|inappropriate|unacceptable)",
            conflict_type=ConflictType.VALUE,
            severity=ConflictSeverity.MODERATE,
            confidence=0.7
        ))
        
        # Factual conflicts
        triggers.append(ConflictTrigger(
            name="factual_contradiction",
            pattern=r"(incorrect|wrong|false|untrue|mistaken|error|not accurate|inaccurate)",
            conflict_type=ConflictType.FACTUAL,
            severity=ConflictSeverity.MODERATE,
            confidence=0.8
        ))
        triggers.append(ConflictTrigger(
            name="data_disagreement",
            pattern=r"(data|evidence|research|study|analysis|statistics|numbers|facts).*?(contradict|refute|disprove|disagree)",
            conflict_type=ConflictType.FACTUAL,
            severity=ConflictSeverity.MODERATE,
            confidence=0.7
        ))
        
        # Procedural conflicts
        triggers.append(ConflictTrigger(
            name="process_disagreement",
            pattern=r"(process|procedure|method|approach|protocol).*?(incorrect|wrong|inappropriate|not proper|improper)",
            conflict_type=ConflictType.PROCEDURAL,
            severity=ConflictSeverity.LOW,
            confidence=0.7
        ))
        triggers.append(ConflictTrigger(
            name="rule_violation",
            pattern=r"(rule|guideline|protocol|standard|regulation).*?(violation|breach|ignore|disregard|break)",
            conflict_type=ConflictType.PROCEDURAL,
            severity=ConflictSeverity.MODERATE,
            confidence=0.8
        ))
        
        # Relationship conflicts
        triggers.append(ConflictTrigger(
            name="trust_issue",
            pattern=r"(don't trust|distrust|suspicious|doubt|skeptical|not confident in|lack confidence in)",
            conflict_type=ConflictType.RELATIONSHIP,
            severity=ConflictSeverity.HIGH,
            confidence=0.9
        ))
        triggers.append(ConflictTrigger(
            name="emotional_language",
            pattern=r"(angry|upset|frustrated|annoyed|disappointed|betrayed|resent|furious|hostile)",
            conflict_type=ConflictType.RELATIONSHIP,
            severity=ConflictSeverity.HIGH,
            confidence=0.8
        ))
        
        # Resource conflicts
        triggers.append(ConflictTrigger(
            name="resource_competition",
            pattern=r"(compete|rivalry|contend|vie|fight).*?(resource|bandwidth|capacity|time|attention)",
            conflict_type=ConflictType.RESOURCE,
            severity=ConflictSeverity.MODERATE,
            confidence=0.7
        ))
        triggers.append(ConflictTrigger(
            name="resource_limitation",
            pattern=r"(limited|scarce|insufficient|inadequate|not enough|lack of|shortage).*?(resource|capacity)",
            conflict_type=ConflictType.RESOURCE,
            severity=ConflictSeverity.MODERATE,
            confidence=0.7
        ))
        
        # Communication conflicts
        triggers.append(ConflictTrigger(
            name="misunderstanding",
            pattern=r"(misunderstand|miscommunication|misinterpret|confused|unclear|ambiguous|not clear)",
            conflict_type=ConflictType.COMMUNICATION,
            severity=ConflictSeverity.LOW,
            confidence=0.6
        ))
        triggers.append(ConflictTrigger(
            name="communication_breakdown",
            pattern=r"(breakdown|failure|ceased|stopped|halted).*?(communication|dialogue|conversation|discussion)",
            conflict_type=ConflictType.COMMUNICATION,
            severity=ConflictSeverity.HIGH,
            confidence=0.8
        ))
        
        # Authority conflicts
        triggers.append(ConflictTrigger(
            name="authority_challenge",
            pattern=r"(challenge|question|dispute|contest).*?(authority|jurisdiction|right|permission)",
            conflict_type=ConflictType.AUTHORITY,
            severity=ConflictSeverity.HIGH,
            confidence=0.8
        ))
        triggers.append(ConflictTrigger(
            name="overstepping_bounds",
            pattern=r"(overstep|exceed|beyond).*?(bounds|limits|authority|mandate|scope|jurisdiction)",
            conflict_type=ConflictType.AUTHORITY,
            severity=ConflictSeverity.MODERATE,
            confidence=0.7
        ))
        
        # Compliance conflicts
        triggers.append(ConflictTrigger(
            name="compliance_violation",
            pattern=r"(non[-\s]?compliance|violation|breach|fail to comply|not compliant|out of compliance)",
            conflict_type=ConflictType.COMPLIANCE,
            severity=ConflictSeverity.HIGH,
            confidence=0.8
        ))
        
        return triggers
    
    def detect_conflicts(
        self,
        message: Dict[str, Any],
        agent_id: str,
        conversation_id: Optional[str] = None
    ) -> List[ConflictIndicator]:
        """
        Detect potential conflicts in a message.
        
        Args:
            message: The message to analyze.
            agent_id: ID of the agent who sent the message.
            conversation_id: ID of the conversation context.
            
        Returns:
            List of ConflictIndicator objects representing potential conflicts.
        """
        indicators = []
        
        # Extract message content for analysis
        content = self._extract_message_content(message)
        if not content or not isinstance(content, str):
            return []
            
        timestamp = datetime.utcnow().isoformat()
        message_id = message.get("id", str(uuid.uuid4()))
        
        # Check each trigger pattern against the message content
        for trigger in self.conflict_triggers:
            match = trigger.regex.search(content)
            if match:
                matched_text = match.group(0)
                
                # Create an indicator for the detected pattern
                indicator = ConflictIndicator(
                    agent_id=agent_id,
                    message_id=message_id,
                    conversation_id=conversation_id or "",
                    timestamp=timestamp,
                    trigger_name=trigger.name,
                    matched_text=matched_text,
                    conflict_type=trigger.conflict_type,
                    severity=trigger.severity,
                    confidence=trigger.confidence
                )
                indicators.append(indicator)
                
                logger.info(f"Detected potential {trigger.conflict_type.value} conflict with {agent_id}: '{matched_text}'")
        
        # Check agent-specific conflict patterns if available
        if agent_id in self.agent_conflict_patterns:
            agent_patterns = self.agent_conflict_patterns[agent_id]
            # TODO: Implement agent-specific pattern checking
        
        return indicators
    
    def _extract_message_content(self, message: Dict[str, Any]) -> Optional[str]:
        """
        Extract content from a message for conflict analysis.
        
        Args:
            message: The message to extract content from.
            
        Returns:
            Extracted message content as string, or None if not extractable.
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
        
        # If no extractable content found
        return None
    
    def categorize_conflict(
        self,
        indicators: List[ConflictIndicator],
        agent_id: str,
        message: Dict[str, Any]
    ) -> Optional[Tuple[ConflictType, ConflictSeverity, str]]:
        """
        Categorize a conflict based on indicators and context.
        
        Args:
            indicators: List of conflict indicators detected.
            agent_id: ID of the agent involved in potential conflict.
            message: The message being analyzed.
            
        Returns:
            Tuple of (conflict_type, severity, description) or None if no conflict.
        """
        if not indicators:
            return None
            
        # Count indicators by type
        type_counts = {}
        severity_by_type = {}
        confidence_by_type = {}
        
        for indicator in indicators:
            conflict_type = indicator.conflict_type.value
            if conflict_type not in type_counts:
                type_counts[conflict_type] = 0
                severity_by_type[conflict_type] = 0
                confidence_by_type[conflict_type] = 0
                
            type_counts[conflict_type] += 1
            severity_by_type[conflict_type] += indicator.severity.value
            confidence_by_type[conflict_type] += indicator.confidence
        
        if not type_counts:
            return None
            
        # Determine the dominant conflict type
        dominant_type = max(type_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate average severity for the dominant type
        avg_severity = severity_by_type[dominant_type] / type_counts[dominant_type]
        severity_level = int(round(avg_severity))
        severity = ConflictSeverity(max(1, min(5, severity_level)))  # Ensure valid enum value
        
        # Create a description
        content = self._extract_message_content(message)
        description = f"Potential {dominant_type} conflict detected with agent {agent_id}."
        
        if content:
            short_content = content[:100] + "..." if len(content) > 100 else content
            description += f" Message content: '{short_content}'"
            
        # Get the conflict type enum
        conflict_type = ConflictType(dominant_type)
        
        return (conflict_type, severity, description)
    
    def create_conflict_record(
        self,
        indicators: List[ConflictIndicator],
        agent_id: str,
        message: Dict[str, Any],
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ConflictRecord]:
        """
        Create a conflict record based on detected indicators.
        
        Args:
            indicators: List of conflict indicators.
            agent_id: ID of the agent involved.
            message: The message containing potential conflict.
            conversation_id: ID of the conversation context.
            metadata: Additional metadata about the conflict.
            
        Returns:
            ConflictRecord if conflict is detected, None otherwise.
        """
        # Categorize the conflict
        categorization = self.categorize_conflict(indicators, agent_id, message)
        if not categorization:
            return None
            
        conflict_type, severity, description = categorization
        
        # Create agents list (the other agent and this agent)
        agents = [agent_id, self.agent_id]
        
        # Create the conflict record
        conflict = ConflictRecord(
            agents=agents,
            conflict_type=conflict_type,
            severity=severity,
            description=description,
            indicators=indicators,
            conversation_id=conversation_id,
            metadata=metadata or {}
        )
        
        # Add to active conflicts
        self.active_conflicts[conflict.conflict_id] = conflict
        
        # If using relationship tracker, record this conflict
        if self.relationship_tracker:
            self.relationship_tracker.record_interaction(
                agent_id=agent_id,
                interaction_type=InteractionType.DISPUTE,
                content_summary=description,
                quality=InteractionQuality.NEGATIVE,
                principle_alignment=0.5,  # Neutral alignment for now
                metadata={
                    "conflict_id": conflict.conflict_id,
                    "conflict_type": conflict_type.value,
                    "severity": severity.value
                }
            )
        
        logger.info(f"Created conflict record {conflict.conflict_id}: {conflict_type.value} conflict with {agent_id}")
        
        return conflict
