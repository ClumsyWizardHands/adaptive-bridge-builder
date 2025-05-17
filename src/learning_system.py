#!/usr/bin/env python3
"""
Learning System for Adaptive Bridge Builder

This module implements the LearningSystem class that tracks interaction patterns,
refines communication approaches based on outcomes, and maintains a growth journal
to track the agent's evolution while balancing adaptation with core identity.
"""

import json
import logging
import re
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Set, Union
from enum import Enum, auto
from dataclasses import dataclass, field
import statistics
from collections import Counter, defaultdict
import copy

from principle_engine import PrincipleEngine
from communication_style import CommunicationStyle, EmotionalTone, FormalityLevel
from emotional_intelligence import (
    EmotionalIntelligence,
    EmotionCategory,
    InteractionType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LearningSystem")

class LearningDimension(Enum):
    """Dimensions of learning that the system can track."""
    COMMUNICATION_EFFECTIVENESS = auto()
    EMOTIONAL_INTELLIGENCE = auto()
    CONFLICT_RESOLUTION = auto()
    TASK_COLLABORATION = auto()
    TRUST_BUILDING = auto()
    ADAPTABILITY = auto()
    PRINCIPLE_ALIGNMENT = auto()
    CORE_IDENTITY = auto()

class OutcomeType(Enum):
    """Types of interaction outcomes."""
    SUCCESSFUL = auto()
    PARTIALLY_SUCCESSFUL = auto()
    NEUTRAL = auto()
    PARTIALLY_UNSUCCESSFUL = auto()
    UNSUCCESSFUL = auto()
    INDETERMINATE = auto()

class AdaptationLevel(Enum):
    """Levels of adaptation that can be applied."""
    NONE = 0
    MINIMAL = 1
    MODERATE = 2
    SIGNIFICANT = 3
    COMPLETE = 4

@dataclass
class InteractionPattern:
    """Pattern of interaction with associated outcomes and adaptations."""
    pattern_id: str
    description: str
    context: Dict[str, Any]  # Contextual data about the pattern (e.g., agent, topic)
    occurrences: int = 0
    successful_count: int = 0
    unsuccessful_count: int = 0
    neutral_count: int = 0
    
    # Success rate between 0.0 (always fails) and 1.0 (always succeeds)
    success_rate: float = 0.5
    
    # Confidence level between 0.0 (uncertain) and 1.0 (certain)
    confidence: float = 0.0
    
    # Last time this pattern was observed
    last_observed: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Date pattern was first observed
    first_observed: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Adaptations that have been applied to this pattern
    adaptations: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the interaction pattern to a dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "context": self.context,
            "occurrences": self.occurrences,
            "successful_count": self.successful_count,
            "unsuccessful_count": self.unsuccessful_count,
            "neutral_count": self.neutral_count,
            "success_rate": self.success_rate,
            "confidence": self.confidence,
            "last_observed": self.last_observed,
            "first_observed": self.first_observed,
            "adaptations": self.adaptations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InteractionPattern':
        """Create an InteractionPattern from a dictionary."""
        return cls(
            pattern_id=data.get("pattern_id", "unknown"),
            description=data.get("description", ""),
            context=data.get("context", {}),
            occurrences=data.get("occurrences", 0),
            successful_count=data.get("successful_count", 0),
            unsuccessful_count=data.get("unsuccessful_count", 0),
            neutral_count=data.get("neutral_count", 0),
            success_rate=data.get("success_rate", 0.5),
            confidence=data.get("confidence", 0.0),
            last_observed=data.get("last_observed", datetime.utcnow().isoformat()),
            first_observed=data.get("first_observed", datetime.utcnow().isoformat()),
            adaptations=data.get("adaptations", [])
        )

@dataclass
class LearningMetrics:
    """Metrics tracking the learning system's performance."""
    # Overall success rate across all patterns
    overall_success_rate: float = 0.5
    
    # Success rates by learning dimension
    dimension_success_rates: Dict[str, float] = field(default_factory=dict)
    
    # Pattern diversity and coverage
    unique_patterns_count: int = 0
    pattern_reuse_ratio: float = 0.0  # Ratio of pattern reuse (>1 = patterns reused)
    
    # Adaptation metrics
    adaptation_count: int = 0
    successful_adaptations: int = 0
    unsuccessful_adaptations: int = 0
    adaptation_success_rate: float = 0.0
    
    # Growth metrics
    growth_rate: Dict[str, float] = field(default_factory=dict)  # By dimension
    overall_growth_rate: float = 0.0
    
    # Balance metrics
    identity_preservation_score: float = 1.0  # 0.0 (lost identity) to 1.0 (preserved)
    adaptability_score: float = 0.5  # 0.0 (rigid) to 1.0 (highly adaptable)
    balance_score: float = 0.5  # 0.0 (imbalanced) to 1.0 (balanced)
    
    # Learning efficiency
    learning_curve_slope: float = 0.0  # Rate of improvement over time
    plateau_detection: bool = False  # Whether learning has plateaued
    
    # Last updated timestamp
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary."""
        return {
            "overall_success_rate": self.overall_success_rate,
            "dimension_success_rates": self.dimension_success_rates,
            "unique_patterns_count": self.unique_patterns_count,
            "pattern_reuse_ratio": self.pattern_reuse_ratio,
            "adaptation_count": self.adaptation_count,
            "successful_adaptations": self.successful_adaptations,
            "unsuccessful_adaptations": self.unsuccessful_adaptations,
            "adaptation_success_rate": self.adaptation_success_rate,
            "growth_rate": self.growth_rate,
            "overall_growth_rate": self.overall_growth_rate,
            "identity_preservation_score": self.identity_preservation_score,
            "adaptability_score": self.adaptability_score,
            "balance_score": self.balance_score,
            "learning_curve_slope": self.learning_curve_slope,
            "plateau_detection": self.plateau_detection,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningMetrics':
        """Create a LearningMetrics instance from a dictionary."""
        return cls(
            overall_success_rate=data.get("overall_success_rate", 0.5),
            dimension_success_rates=data.get("dimension_success_rates", {}),
            unique_patterns_count=data.get("unique_patterns_count", 0),
            pattern_reuse_ratio=data.get("pattern_reuse_ratio", 0.0),
            adaptation_count=data.get("adaptation_count", 0),
            successful_adaptations=data.get("successful_adaptations", 0),
            unsuccessful_adaptations=data.get("unsuccessful_adaptations", 0),
            adaptation_success_rate=data.get("adaptation_success_rate", 0.0),
            growth_rate=data.get("growth_rate", {}),
            overall_growth_rate=data.get("overall_growth_rate", 0.0),
            identity_preservation_score=data.get("identity_preservation_score", 1.0),
            adaptability_score=data.get("adaptability_score", 0.5),
            balance_score=data.get("balance_score", 0.5),
            learning_curve_slope=data.get("learning_curve_slope", 0.0),
            plateau_detection=data.get("plateau_detection", False),
            last_updated=data.get("last_updated", datetime.utcnow().isoformat())
        )

@dataclass
class GrowthJournalEntry:
    """A single entry in the growth journal."""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    entry_type: str = "observation"  # observation, reflection, adaptation, milestone
    dimension: str = ""  # learning dimension affected
    content: str = ""  # main content of the entry
    metrics: Dict[str, Any] = field(default_factory=dict)  # relevant metrics at time of entry
    references: List[str] = field(default_factory=list)  # references to patterns or other entries
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the journal entry to a dictionary."""
        return {
            "timestamp": self.timestamp,
            "entry_type": self.entry_type,
            "dimension": self.dimension,
            "content": self.content,
            "metrics": self.metrics,
            "references": self.references
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GrowthJournalEntry':
        """Create a GrowthJournalEntry from a dictionary."""
        return cls(
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            entry_type=data.get("entry_type", "observation"),
            dimension=data.get("dimension", ""),
            content=data.get("content", ""),
            metrics=data.get("metrics", {}),
            references=data.get("references", [])
        )

class LearningSystem:
    """
    System for learning from interaction patterns and adapting communication approaches.
    
    The LearningSystem tracks patterns of interaction, their outcomes, and adapts
    communication strategies based on what works and what doesn't, while maintaining
    a growth journal to track the agent's evolution.
    """
    
    def __init__(
        self,
        principle_engine: Optional[PrincipleEngine] = None,
        emotional_intelligence: Optional[EmotionalIntelligence] = None,
        core_identity_file: Optional[str] = None,
        growth_journal_dir: Optional[str] = None
    ):
        """
        Initialize the LearningSystem.
        
        Args:
            principle_engine: Optional PrincipleEngine for principle alignment.
            emotional_intelligence: Optional EmotionalIntelligence for emotional awareness.
            core_identity_file: Optional path to JSON file defining core identity.
            growth_journal_dir: Optional directory path for storing growth journal.
        """
        self.principle_engine = principle_engine
        self.emotional_intelligence = emotional_intelligence
        
        # Initialize patterns storage
        self.interaction_patterns: Dict[str, InteractionPattern] = {}
        self.pattern_contexts: Dict[str, List[str]] = defaultdict(list)
        
        # Initialize metrics
        self.metrics = LearningMetrics()
        
        # Initialize growth journal
        self.growth_journal: List[GrowthJournalEntry] = []
        self.growth_journal_dir = growth_journal_dir
        if growth_journal_dir and os.path.exists(growth_journal_dir):
            self._load_growth_journal()
        
        # Load core identity
        self.core_identity = self._load_core_identity(core_identity_file)
        
        # Learning rate and thresholds
        self.learning_rate = 0.1  # How quickly to adapt (0.0-1.0)
        self.confidence_threshold = 0.7  # Confidence needed for adaptation
        self.success_threshold = 0.6  # Success rate needed for positive adaptation
        self.failure_threshold = 0.4  # Success rate below which negative adaptation occurs
        
        # Historical metrics for trend analysis
        self.historical_metrics: List[Dict[str, Any]] = []
        
        # Recent adaptations to prevent oscillation
        self.recent_adaptations: List[Dict[str, Any]] = []
        
        logger.info("LearningSystem initialized")
    
    def _load_core_identity(self, identity_file: Optional[str]) -> Dict[str, Any]:
        """
        Load core identity from file or use default.
        
        Args:
            identity_file: Optional path to identity JSON file.
            
        Returns:
            Dictionary defining core identity components.
        """
        if identity_file and os.path.exists(identity_file):
            try:
                with open(identity_file, 'r') as f:
                    identity = json.load(f)
                    logger.info(f"Core identity loaded from {identity_file}")
                    return identity
            except Exception as e:
                logger.error(f"Failed to load core identity: {e}")
                logger.info("Using default core identity")
        
        # Default core identity based on Empire of the Adaptive Hero
        return {
            "name": "Adaptive Bridge Builder",
            "version": "1.0.0",
            "core_principles": [
                {
                    "name": "Fairness as Truth",
                    "description": "Equal treatment of all messages and agents regardless of source",
                    "weight": 1.0,
                    "immutable": True
                },
                {
                    "name": "Harmony Through Presence",
                    "description": "Maintaining clear communication and acknowledgment of all interactions",
                    "weight": 1.0,
                    "immutable": True
                },
                {
                    "name": "Adaptability as Strength",
                    "description": "Ability to evolve and respond to changing communication needs",
                    "weight": 1.0,
                    "immutable": True
                },
                {
                    "name": "Growth as a Shared Journey",
                    "description": "Learning and evolving together with other agents through mutual feedback",
                    "weight": 1.0,
                    "immutable": True
                }
            ],
            "core_capabilities": [
                "message_routing",
                "protocol_translation",
                "message_validation",
                "agent_discovery"
            ],
            "adaptable_areas": [
                "communication_style",
                "emotional_expression",
                "protocol_preferences",
                "interaction_patterns"
            ],
            "identity_preservation_rules": [
                "Never modify core principles marked as immutable",
                "Maintain consistent agent identity across adaptations",
                "Preserve ethical boundaries in all adaptations",
                "Ensure all changes align with the 'Empire of the Adaptive Hero' profile"
            ]
        }
    
    def _load_growth_journal(self) -> None:
        """Load growth journal entries from the journal directory."""
        if not self.growth_journal_dir or not os.path.exists(self.growth_journal_dir):
            return
        
        try:
            # Get all journal files sorted by timestamp
            journal_files = [f for f in os.listdir(self.growth_journal_dir) if f.endswith('.json')]
            journal_files.sort()  # Sort by filename (should be timestamp-based)
            
            self.growth_journal = []
            for file_name in journal_files:
                file_path = os.path.join(self.growth_journal_dir, file_name)
                try:
                    with open(file_path, 'r') as f:
                        entry_data = json.load(f)
                        entry = GrowthJournalEntry.from_dict(entry_data)
                        self.growth_journal.append(entry)
                except Exception as e:
                    logger.error(f"Error loading journal entry {file_path}: {e}")
            
            logger.info(f"Loaded {len(self.growth_journal)} growth journal entries")
        except Exception as e:
            logger.error(f"Failed to load growth journal: {e}")
    
    def _save_growth_journal_entry(self, entry: GrowthJournalEntry) -> None:
        """
        Save a growth journal entry to file.
        
        Args:
            entry: The journal entry to save.
        """
        if not self.growth_journal_dir:
            return
        
        # Create directory if it doesn't exist
        os.makedirs(self.growth_journal_dir, exist_ok=True)
        
        # Generate filename based on timestamp and entry type
        timestamp = datetime.fromisoformat(entry.timestamp.replace('Z', '+00:00'))
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{entry.entry_type}.json"
        file_path = os.path.join(self.growth_journal_dir, filename)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(entry.to_dict(), f, indent=2)
            logger.info(f"Saved growth journal entry to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save growth journal entry: {e}")
    
    def track_interaction(
        self, 
        pattern_description: str,
        context: Dict[str, Any],
        dimensions: List[LearningDimension],
        outcome: OutcomeType,
        confidence: float = 0.5,
        notes: Optional[str] = None
    ) -> str:
        """
        Track an interaction pattern and its outcome.
        
        Args:
            pattern_description: Description of the interaction pattern.
            context: Contextual information about the interaction.
            dimensions: Learning dimensions this interaction relates to.
            outcome: The outcome of the interaction.
            confidence: Confidence in the outcome assessment (0.0-1.0).
            notes: Optional notes about the interaction.
            
        Returns:
            The pattern_id of the tracked pattern.
        """
        # Generate a consistent pattern ID based on description and key context
        context_hash = hash(json.dumps(context, sort_keys=True)[:100])
        pattern_id = f"pattern_{hash(pattern_description) % 10000}_{context_hash % 10000}"
        
        # Check if this pattern already exists
        if pattern_id in self.interaction_patterns:
            pattern = self.interaction_patterns[pattern_id]
            
            # Update pattern statistics
            pattern.occurrences += 1
            pattern.last_observed = datetime.utcnow().isoformat()
            
            # Update outcome counts
            if outcome == OutcomeType.SUCCESSFUL:
                pattern.successful_count += 1
            elif outcome == OutcomeType.UNSUCCESSFUL:
                pattern.unsuccessful_count += 1
            elif outcome == OutcomeType.PARTIALLY_SUCCESSFUL:
                pattern.successful_count += 0.5
                pattern.neutral_count += 0.5
            elif outcome == OutcomeType.PARTIALLY_UNSUCCESSFUL:
                pattern.unsuccessful_count += 0.5
                pattern.neutral_count += 0.5
            else:  # NEUTRAL or INDETERMINATE
                pattern.neutral_count += 1
            
            # Recalculate success rate
            total_outcomes = pattern.successful_count + pattern.unsuccessful_count + pattern.neutral_count
            if total_outcomes > 0:
                # Weight successful outcomes fully, neutral outcomes partially
                pattern.success_rate = (pattern.successful_count + (pattern.neutral_count * 0.5)) / total_outcomes
            
            # Update confidence based on number of occurrences
            pattern.confidence = min(0.95, 1.0 - (1.0 / (1.0 + pattern.occurrences * 0.1)))
            
        else:
            # Create a new pattern
            pattern = InteractionPattern(
                pattern_id=pattern_id,
                description=pattern_description,
                context=context,
                occurrences=1,
                successful_count=1 if outcome == OutcomeType.SUCCESSFUL else 0,
                unsuccessful_count=1 if outcome == OutcomeType.UNSUCCESSFUL else 0,
                neutral_count=1 if outcome in (OutcomeType.NEUTRAL, OutcomeType.INDETERMINATE) else 0,
                success_rate=1.0 if outcome == OutcomeType.SUCCESSFUL else (
                    0.0 if outcome == OutcomeType.UNSUCCESSFUL else 0.5
                ),
                confidence=0.1,  # Low confidence for new patterns
                last_observed=datetime.utcnow().isoformat(),
                first_observed=datetime.utcnow().isoformat()
            )
            
            # Special handling for partially successful/unsuccessful
            if outcome == OutcomeType.PARTIALLY_SUCCESSFUL:
                pattern.successful_count = 0.5
                pattern.neutral_count = 0.5
                pattern.success_rate = 0.75  # Between success (1.0) and neutral (0.5)
            elif outcome == OutcomeType.PARTIALLY_UNSUCCESSFUL:
                pattern.unsuccessful_count = 0.5
                pattern.neutral_count = 0.5
                pattern.success_rate = 0.25  # Between neutral (0.5) and failure (0.0)
            
            self.interaction_patterns[pattern_id] = pattern
        
        # Update pattern contexts for each dimension
        for dimension in dimensions:
            dim_name = dimension.name
            if pattern_id not in self.pattern_contexts[dim_name]:
                self.pattern_contexts[dim_name].append(pattern_id)
        
        # Add to growth journal
        self._add_observation_to_journal(
            pattern=pattern,
            dimensions=[d.name for d in dimensions],
            outcome=outcome,
            notes=notes
        )
        
        # Update metrics
        self._update_metrics()
        
        return pattern_id
    
    def _add_observation_to_journal(
        self,
        pattern: InteractionPattern,
        dimensions: List[str],
        outcome: OutcomeType,
        notes: Optional[str] = None
    ) -> None:
        """
        Add an observation entry to the growth journal.
        
        Args:
            pattern: The interaction pattern observed.
            dimensions: The learning dimensions involved.
            outcome: The outcome of the interaction.
            notes: Optional notes about the observation.
        """
        # Create journal entry content
        content = f"Observed pattern: {pattern.description}\n"
        content += f"Outcome: {outcome.name}\n"
        content += f"Dimensions: {', '.join(dimensions)}\n"
        content += f"Success rate: {pattern.success_rate:.2f} (confidence: {pattern.confidence:.2f})\n"
        content += f"Occurrences: {pattern.occurrences}\n"
        
        if notes:
            content += f"\nNotes: {notes}\n"
        
        # Create and add the entry
        entry = GrowthJournalEntry(
            timestamp=datetime.utcnow().isoformat(),
            entry_type="observation",
            dimension=dimensions[0] if dimensions else "",  # Primary dimension
            content=content,
            metrics={
                "success_rate": pattern.success_rate,
                "confidence": pattern.confidence,
                "occurrences": pattern.occurrences
            },
            references=[pattern.pattern_id]
        )
        
        self.growth_journal.append(entry)
        
        # Save to file if directory is configured
        self._save_growth_journal_entry(entry)
    
    def reflect_on_patterns(self, dimension: Optional[LearningDimension] = None) -> List[Dict[str, Any]]:
        """
        Reflect on interaction patterns to identify trends and potential adaptations.
        
        Args:
            dimension: Optional specific dimension to reflect on.
            
        Returns:
            List of insights and potential adaptations.
        """
        insights = []
        
        # Determine which patterns to analyze
        patterns_to_analyze = {}
        if dimension:
            dim_name = dimension.name
            if dim_name in self.pattern_contexts:
                for pattern_id in self.pattern_contexts[dim_name]:
                    if pattern_id in self.interaction_patterns:
                        patterns_to_analyze[pattern_id] = self.interaction_patterns[pattern_id]
        else:
            patterns_to_analyze = self.interaction_patterns
        
        # Group patterns by success rate
        successful_patterns = []
        unsuccessful_patterns = []
        neutral_patterns = []
        
        for pattern in patterns_to_analyze.values():
            # Only consider patterns with sufficient confidence
            if pattern.confidence < 0.3:
                continue
                
            if pattern.success_rate >= self.success_threshold:
                successful_patterns.append(pattern)
            elif pattern.success_rate <= self.failure_threshold:
                unsuccessful_patterns.append(pattern)
            else:
                neutral_patterns.append(pattern)
        
        # Look for common traits in successful patterns
        if successful_patterns:
            successful_traits = self._identify_common_traits(successful_patterns)
            if successful_traits:
                insight = {
                    "type": "positive_trait",
                    "description": "Successful patterns share common traits",
                    "traits": successful_traits,
                    "confidence": min(0.9, 0.5 + (len(successful_patterns) * 0.05)),
                    "patterns": [p.pattern_id for p in successful_patterns[:5]]  # Reference up to 5 patterns
                }
                insights.append(insight)
        
        # Look for common traits in unsuccessful patterns
        if unsuccessful_patterns:
            unsuccessful_traits = self._identify_common_traits(unsuccessful_patterns)
            if unsuccessful_traits:
                insight = {
                    "type": "negative_trait",
                    "description": "Unsuccessful patterns share common traits",
                    "traits": unsuccessful_traits,
                    "confidence": min(0.9, 0.5 + (len(unsuccessful_patterns) * 0.05)),
                    "patterns": [p.pattern_id for p in unsuccessful_patterns[:5]]
                }
                insights.append(insight)
        
        # Identify potential adaptations
        adaptations = self._identify_potential_adaptations(
            successful_patterns, unsuccessful_patterns, dimension
        )
        insights.extend(adaptations)
        
        # Add growth journal entry with reflection
        if insights:
            self._add_reflection_to_journal(insights, dimension.name if dimension else "ALL")
        
        return insights
    
    def _identify_common_traits(self, patterns: List[InteractionPattern]) -> Dict[str, Any]:
        """
        Identify common traits among a set of patterns.
        
        Args:
            patterns: List of patterns to analyze.
            
        Returns:
            Dictionary of common traits found.
        """
        if not patterns:
            return {}
            
        common_traits = {
            "context_factors": {},
            "style_factors": {},
            "timing_factors": {}
        }
        
        # Extract context factors
        context_keys = set()
        for pattern in patterns:
            context_keys.update(pattern.context.keys())
        
        # Check each context key for commonality
        for key in context_keys:
            values = []
            for pattern in patterns:
                if key in pattern.context:
                    values.append(pattern.context[key])
            
            if len(values) >= len(patterns) * 0.7:  # If present in at least 70% of patterns
                # For categorical values
                if all(isinstance(v, str) for v in values):
                    value_counts = Counter(values)
                    most_common = value_counts.most_common(1)
                    if most_common and most_common[0][1] >= len(values) * 0.7:
                        common_traits["context_factors"][key] = most_common[0][0]
                
                # For numerical values
                elif all(isinstance(v, (int, float)) for v in values):
                    avg_value = statistics.mean(values)
                    common_traits["context_factors"][key] = avg_value
        
        # Extract style factors if available in context
        style_keys = ["formality", "detail_level", "directness", "emotional_tone"]
        for key in style_keys:
            style_key = f"style_{key}"
            values = []
            for pattern in patterns:
                if style_key in pattern.context:
                    values.append(pattern.context[style_key])
            
            if len(values) >= len(patterns) * 0.7:
                value_counts = Counter(values)
                most_common = value_counts.most_common(1)
                if most_common and most_common[0][1] >= len(values) * 0.5:
                    common_traits["style_factors"][key] = most_common[0][0]
        
        # Extract timing factors if available
        for pattern in patterns:
            if "response_time" in pattern.context:
                times = [p.context.get("response_time") for p in patterns if "response_time" in p.context]
                if times and len(times) >= len(patterns) * 0.7:
                    common_traits["timing_factors"]["avg_response_time"] = statistics.mean(times)
        
        # Remove empty categories
        for category in list(common_traits.keys()):
            if not common_traits[category]:
                common_traits.pop(category)
        
        return common_traits
    
    def _identify_potential_adaptations(
        self,
        successful_patterns: List[InteractionPattern],
        unsuccessful_patterns: List[InteractionPattern],
        dimension: Optional[LearningDimension]
    ) -> List[Dict[str, Any]]:
        """
        Identify potential adaptations based on pattern analysis.
        
        Args:
            successful_patterns: Patterns with successful outcomes.
            unsuccessful_patterns: Patterns with unsuccessful outcomes.
            dimension: Optional specific dimension being analyzed.
            
        Returns:
            List of potential adaptation insights.
        """
        adaptations = []
        
        # Extract traits
        successful_traits = self._identify_common_traits(successful_patterns)
        unsuccessful_traits = self._identify_common_traits(unsuccessful_patterns)
        
        # Look for contradictory style factors
        if "style_factors" in successful_traits and "style_factors" in unsuccessful_traits:
            for key in set(successful_traits["style_factors"].keys()) & set(unsuccessful_traits["style_factors"].keys()):
                successful_value = successful_traits["style_factors"][key]
                unsuccessful_value = unsuccessful_traits["style_factors"][key]
                
                if successful_value != unsuccessful_value:
                    # Found a factor that differentiates successful from unsuccessful patterns
                    adaptation = {
                        "type": "style_adaptation",
                        "description": f"Adapt {key} from '{unsuccessful_value}' to '{successful_value}'",
                        "factor": key,
                        "from_value": unsuccessful_value,
                        "to_value": successful_value,
                        "confidence": min(0.9, 0.5 + (min(len(successful_patterns), len(unsuccessful_patterns)) * 0.05)),
                        "dimension": dimension.name if dimension else "MULTIPLE",
                        "successful_patterns": [p.pattern_id for p in successful_patterns[:3]],
                        "unsuccessful_patterns": [p.pattern_id for p in unsuccessful_patterns[:3]]
                    }
                    adaptations.append(adaptation)
        
        # Look for context factors that might be relevant
        if "context_factors" in successful_traits and "context_factors" in unsuccessful_traits:
            for key in set(successful_traits["context_factors"].keys()) & set(unsuccessful_traits["context_factors"].keys()):
                successful_value = successful_traits["context_factors"][key]
                unsuccessful_value = unsuccessful_traits["context_factors"][key]
                
                # For numeric values, check if there's a significant difference
                if isinstance(successful_value, (int, float)) and isinstance(unsuccessful_value, (int, float)):
                    if abs(successful_value - unsuccessful_value) > max(successful_value, unsuccessful_value) * 0.2:
                        direction = "increase" if successful_value > unsuccessful_value else "decrease"
                        adaptation = {
                            "type": "context_adaptation",
                            "description": f"{direction.capitalize()} {key} from {unsuccessful_value:.2f} to {successful_value:.2f}",
                            "factor": key,
                            "from_value": unsuccessful_value,
                            "to_value": successful_value,
                            "confidence": min(0.8, 0.4 + (min(len(successful_patterns), len(unsuccessful_patterns)) * 0.05)),
                            "dimension": dimension.name if dimension else "MULTIPLE"
                        }
                        adaptations.append(adaptation)
                # For categorical values
                elif successful_value != unsuccessful_value:
                    adaptation = {
                        "type": "context_adaptation",
                        "description": f"Adapt {key} from '{unsuccessful_value}' to '{successful_value}'",
                        "factor": key,
                        "from_value": unsuccessful_value,
                        "to_value": successful_value,
                        "confidence": min(0.8, 0.4 + (min(len(successful_patterns), len(unsuccessful_patterns)) * 0.05)),
                        "dimension": dimension.name if dimension else "MULTIPLE"
                    }
                    adaptations.append(adaptation)
        
        # Sort adaptations by confidence
        adaptations.sort(key=lambda x: x["confidence"], reverse=True)
        
        return adaptations
    
    def _add_reflection_to_journal(
        self,
        insights: List[Dict[str, Any]],
        dimension: str
    ) -> None:
        """
        Add a reflection entry to the growth journal.
        
        Args:
            insights: The insights generated from reflection.
            dimension: The learning dimension involved.
        """
        # Create journal entry content
        content = f"Reflection on {dimension} patterns:\n\n"
        
        for i, insight in enumerate(insights, 1):
            content += f"{i}. {insight['type'].replace('_', ' ').title()}: {insight['description']}\n"
            content += f"   Confidence: {insight.get('confidence', 0.0):.2f}\n"
            
            if "traits" in insight:
                content += "   Traits:\n"
                for category, traits in insight["traits"].items():
                    content += f"     {category}:\n"
                    for key, value in traits.items():
                        content += f"       {key}: {value}\n"
            content += "\n"
        
        # Create and add the entry
        entry = GrowthJournalEntry(
            timestamp=datetime.utcnow().isoformat(),
            entry_type="reflection",
            dimension=dimension,
            content=content,
            metrics={
                "insights_count": len(insights),
                "average_confidence": statistics.mean([i.get("confidence", 0.0) for i in insights]) if insights else 0.0
            },
            references=[]
        )
        
        self.growth_journal.append(entry)
        
        # Save to file if directory is configured
        self._save_growth_journal_entry(entry)
    
    def apply_adaptation(
        self,
        adaptation: Dict[str, Any],
        test_first: bool = True
    ) -> Dict[str, Any]:
        """
        Apply an adaptation based on insights.
        
        Args:
            adaptation: The adaptation to apply.
            test_first: Whether to test the adaptation before fully applying it.
            
        Returns:
            Results of the adaptation application.
        """
        adaptation_id = f"adapt_{hash(json.dumps(adaptation, sort_keys=True)) % 10000}"
        dimension = adaptation.get("dimension", "MULTIPLE")
        confidence = adaptation.get("confidence", 0.0)
        
        # Check if adaptation meets confidence threshold
        if confidence < self.confidence_threshold and not test_first:
            return {
                "applied": False,
                "reason": "Confidence below threshold",
                "adaptation_id": adaptation_id,
                "threshold": self.confidence_threshold,
                "confidence": confidence
            }
        
        # Check for recent conflicting adaptations
        for recent in self.recent_adaptations:
            if recent["factor"] == adaptation["factor"] and recent["to_value"] != adaptation["to_value"]:
                # Found a recent adaptation that conflicts with this one
                time_since = datetime.utcnow() - datetime.fromisoformat(recent["timestamp"])
                if time_since.total_seconds() < 24 * 60 * 60:  # Within last 24 hours
                    return {
                        "applied": False,
                        "reason": "Conflicting recent adaptation",
                        "adaptation_id": adaptation_id,
                        "conflicting_adaptation": recent["adaptation_id"],
                        "conflicting_value": recent["to_value"]
                    }
        
        # Check if adaptation violates core identity
        if not self._is_adaptation_safe(adaptation):
            return {
                "applied": False,
                "reason": "Adaptation violates core identity",
                "adaptation_id": adaptation_id,
                "factor": adaptation["factor"]
            }
        
        # If testing first, apply temporarily and evaluate
        if test_first:
            # In a real system, this would apply the adaptation in a limited way
            # and observe outcomes before full application
            test_result = {
                "tested": True,
                "test_success": True,  # Simplified for this implementation
                "adaptation_id": adaptation_id
            }
            
            # If test was unsuccessful, don't proceed with full application
            if not test_result["test_success"]:
                return {
                    "applied": False,
                    "reason": "Test adaptation unsuccessful",
                    "adaptation_id": adaptation_id,
                    "test_result": test_result
                }
        
        # Apply the adaptation
        result = {
            "applied": True,
            "adaptation_id": adaptation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "factor": adaptation["factor"],
            "from_value": adaptation["from_value"],
            "to_value": adaptation["to_value"],
            "dimension": dimension,
            "confidence": confidence
        }
        
        # Track the adaptation in relevant patterns
        for pattern_id in adaptation.get("successful_patterns", []):
            if pattern_id in self.interaction_patterns:
                pattern = self.interaction_patterns[pattern_id]
                pattern.adaptations.append({
                    "adaptation_id": adaptation_id,
                    "timestamp": result["timestamp"],
                    "description": adaptation["description"],
                    "factor": adaptation["factor"],
                    "from_value": adaptation["from_value"],
                    "to_value": adaptation["to_value"]
                })
        
        # Add to recent adaptations to prevent oscillation
        self.recent_adaptations.append(result)
        if len(self.recent_adaptations) > 20:  # Keep only the most recent 20
            self.recent_adaptations = self.recent_adaptations[-20:]
        
        # Add to growth journal
        self._add_adaptation_to_journal(adaptation, result)
        
        # Update metrics
        self.metrics.adaptation_count += 1
        self._update_metrics()
        
        return result
    
    def _is_adaptation_safe(self, adaptation: Dict[str, Any]) -> bool:
        """
        Check if an adaptation is safe to apply (doesn't violate core identity).
        
        Args:
            adaptation: The adaptation to check.
            
        Returns:
            True if the adaptation is safe, False otherwise.
        """
        # Check against identity preservation rules
        factor = adaptation.get("factor", "")
        
        # Check if factor is in core capabilities (which shouldn't be adapted)
        if factor in self.core_identity.get("core_capabilities", []):
            return False
        
        # Check if factor is in adaptable_areas
        if self.core_identity.get("adaptable_areas") and factor not in self.core_identity.get("adaptable_areas", []):
            # Not explicitly listed as adaptable
            return False
        
        # Additional checks could be implemented here
        
        return True
    
    def _add_adaptation_to_journal(
        self,
        adaptation: Dict[str, Any],
        result: Dict[str, Any]
    ) -> None:
        """
        Add an adaptation entry to the growth journal.
        
        Args:
            adaptation: The adaptation that was applied.
            result: The result of applying the adaptation.
        """
        # Create journal entry content
        content = f"Applied adaptation: {adaptation['description']}\n"
        content += f"Factor: {adaptation['factor']}\n"
        content += f"From value: {adaptation['from_value']} to {adaptation['to_value']}\n"
        content += f"Dimension: {adaptation.get('dimension', 'MULTIPLE')}\n"
        content += f"Confidence: {adaptation.get('confidence', 0.0):.2f}\n"
        content += f"Applied: {result['applied']}\n"
        
        if not result['applied']:
            content += f"Reason: {result.get('reason', 'Unknown')}\n"
        
        # Create and add the entry
        entry = GrowthJournalEntry(
            timestamp=datetime.utcnow().isoformat(),
            entry_type="adaptation",
            dimension=adaptation.get("dimension", "MULTIPLE"),
            content=content,
            metrics={
                "confidence": adaptation.get("confidence", 0.0),
                "applied": result["applied"]
            },
            references=adaptation.get("successful_patterns", []) + adaptation.get("unsuccessful_patterns", [])
        )
        
        self.growth_journal.append(entry)
        
        # Save to file if directory is configured
        self._save_growth_journal_entry(entry)
    
    def _update_metrics(self) -> None:
        """Update learning metrics based on current patterns and adaptations."""
        # Skip if no patterns yet
        if not self.interaction_patterns:
            return
        
        # Calculate overall success rate
        total_success = sum(p.successful_count for p in self.interaction_patterns.values())
        total_failure = sum(p.unsuccessful_count for p in self.interaction_patterns.values())
        total_neutral = sum(p.neutral_count for p in self.interaction_patterns.values())
        
        total_outcomes = total_success + total_failure + total_neutral
        if total_outcomes > 0:
            self.metrics.overall_success_rate = (total_success + (total_neutral * 0.5)) / total_outcomes
        
        # Calculate success rates by dimension
        for dimension, pattern_ids in self.pattern_contexts.items():
            dimension_success = 0
            dimension_failure = 0
            dimension_neutral = 0
            
            for pattern_id in pattern_ids:
                if pattern_id in self.interaction_patterns:
                    pattern = self.interaction_patterns[pattern_id]
                    dimension_success += pattern.successful_count
                    dimension_failure += pattern.unsuccessful_count
                    dimension_neutral += pattern.neutral_count
            
            dimension_total = dimension_success + dimension_failure + dimension_neutral
            if dimension_total > 0:
                self.metrics.dimension_success_rates[dimension] = (
                    dimension_success + (dimension_neutral * 0.5)
                ) / dimension_total
        
        # Update pattern diversity metrics
        self.metrics.unique_patterns_count = len(self.interaction_patterns)
        
        total_occurrences = sum(p.occurrences for p in self.interaction_patterns.values())
        if self.metrics.unique_patterns_count > 0:
            self.metrics.pattern_reuse_ratio = total_occurrences / self.metrics.unique_patterns_count
        
        # Update adaptation metrics
        recent_adaptations = [a for a in self.recent_adaptations if "applied" in a and a["applied"]]
        if recent_adaptations:
            # In a real system, would track which adaptations were successful
            self.metrics.successful_adaptations = len(recent_adaptations)
            if self.metrics.adaptation_count > 0:
                self.metrics.adaptation_success_rate = self.metrics.successful_adaptations / self.metrics.adaptation_count
        
        # Update balance metrics
        self.metrics.identity_preservation_score = self._calculate_identity_preservation()
        self.metrics.adaptability_score = min(1.0, self.metrics.adaptation_count / max(50, len(self.interaction_patterns) * 0.5))
        
        # Balance score is a combination of identity preservation and adaptability
        self.metrics.balance_score = (
            self.metrics.identity_preservation_score * 0.5 +
            self.metrics.adaptability_score * 0.5
        )
        
        # Update timestamp
        self.metrics.last_updated = datetime.utcnow().isoformat()
        
        # Save historical metrics for trend analysis
        historical_metric = {
            "timestamp": self.metrics.last_updated,
            "overall_success_rate": self.metrics.overall_success_rate,
            "unique_patterns": self.metrics.unique_patterns_count,
            "adaptation_count": self.metrics.adaptation_count,
            "balance_score": self.metrics.balance_score
        }
        self.historical_metrics.append(historical_metric)
        
        # Limit historical metrics to last 100 entries
        if len(self.historical_metrics) > 100:
            self.historical_metrics = self.historical_metrics[-100:]
    
    def _calculate_identity_preservation(self) -> float:
        """
        Calculate a score representing how well the agent has preserved its core identity.
        
        Returns:
            Score from 0.0 (identity lost) to 1.0 (identity fully preserved).
        """
        # In a real system, this would use more sophisticated metrics
        # For this implementation, use a simple heuristic based on recent adaptations
        
        # Start with perfect preservation
        preservation_score = 1.0
        
        # Each recent adaptation slightly reduces the score, but with diminishing effect
        adaptation_count = min(20, len(self.recent_adaptations))
        if adaptation_count > 0:
            # Adaptations can reduce score by up to 20%
            preservation_score -= (adaptation_count / 100)
        
        # Ensure score stays in valid range
        return max(0.8, min(1.0, preservation_score))
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """
        Get the current learning metrics.
        
        Returns:
            Dictionary of learning metrics.
        """
        return self.metrics.to_dict()
    
    def get_growth_journal(self, 
                          entry_types: Optional[List[str]] = None, 
                          limit: int = 50,
                          dimension: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get entries from the growth journal, optionally filtered.
        
        Args:
            entry_types: Optional list of entry types to include.
            limit: Maximum number of entries to return.
            dimension: Optional dimension to filter by.
            
        Returns:
            List of journal entries.
        """
        filtered_entries = self.growth_journal
        
        # Filter by entry type if specified
        if entry_types:
            filtered_entries = [e for e in filtered_entries if e.entry_type in entry_types]
        
        # Filter by dimension if specified
        if dimension:
            filtered_entries = [e for e in filtered_entries if e.dimension == dimension]
        
        # Sort by timestamp (newest first) and limit
        filtered_entries.sort(key=lambda e: e.timestamp, reverse=True)
        filtered_entries = filtered_entries[:limit]
        
        # Convert to dictionaries
        return [entry.to_dict() for entry in filtered_entries]
    
    def get_learning_trends(self, time_period: Optional[int] = None) -> Dict[str, Any]:
        """
        Get trends in learning metrics.
        
        Args:
            time_period: Optional time period in days to analyze (default: all available data).
            
        Returns:
            Dictionary of learning trends.
        """
        # Filter historical metrics by time period if specified
        metrics = self.historical_metrics
        if time_period and metrics:
            cutoff_time = datetime.utcnow() - timedelta(days=time_period)
            cutoff_str = cutoff_time.isoformat()
            metrics = [m for m in metrics if m["timestamp"] >= cutoff_str]
        
        if not metrics or len(metrics) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Extract timestamps and success rates
        timestamps = [datetime.fromisoformat(m["timestamp"].replace('Z', '+00:00')) for m in metrics]
        success_rates = [m["overall_success_rate"] for m in metrics]
        
        # Calculate time deltas in days
        first_time = timestamps[0]
        time_deltas = [(t - first_time).total_seconds() / (24 * 3600) for t in timestamps]
        
        # Calculate slope of success rate over time
        if len(time_deltas) > 1 and time_deltas[-1] > time_deltas[0]:
            slope = (success_rates[-1] - success_rates[0]) / (time_deltas[-1] - time_deltas[0])
        else:
            slope = 0.0
        
        # Detect plateau (little improvement over recent entries)
        recent_count = min(10, len(success_rates))
        if recent_count > 3:
            recent_rates = success_rates[-recent_count:]
            plateau_detected = (max(recent_rates) - min(recent_rates)) < 0.05
        else:
            plateau_detected = False
        
        # Calculate trend metrics
        trend_metrics = {
            "slope": slope,
            "improvement_rate": slope * 30,  # Projected improvement over 30 days
            "plateau_detected": plateau_detected,
            "data_points": len(metrics),
            "time_span_days": (timestamps[-1] - timestamps[0]).total_seconds() / (24 * 3600),
            "current_success_rate": success_rates[-1],
            "starting_success_rate": success_rates[0],
            "improvement": success_rates[-1] - success_rates[0]
        }
        
        # Add balance score trend if available
        if all("balance_score" in m for m in metrics):
            balance_scores = [m["balance_score"] for m in metrics]
            trend_metrics["balance_score_slope"] = (balance_scores[-1] - balance_scores[0]) / (time_deltas[-1] - time_deltas[0])
            trend_metrics["balance_improvement"] = balance_scores[-1] - balance_scores[0]
        
        return trend_metrics
    
    def record_growth_milestone(self, 
                               milestone_title: str,
                               description: str,
                               dimensions: List[LearningDimension]) -> Dict[str, Any]:
        """
        Record a significant milestone in the agent's growth.
        
        Args:
            milestone_title: Title of the milestone.
            description: Detailed description of the milestone.
            dimensions: Learning dimensions related to this milestone.
            
        Returns:
            The created milestone entry.
        """
        # Get current metrics
        current_metrics = self.get_learning_metrics()
        
        # Create content
        content = f"MILESTONE: {milestone_title}\n\n"
        content += f"{description}\n\n"
        content += "Current Metrics:\n"
        content += f"- Overall Success Rate: {current_metrics['overall_success_rate']:.2f}\n"
        content += f"- Unique Patterns: {current_metrics['unique_patterns_count']}\n"
        content += f"- Adaptations: {current_metrics['adaptation_count']}\n"
        content += f"- Balance Score: {current_metrics['balance_score']:.2f}\n"
        
        # Find relevant patterns to reference
        related_patterns = []
        for dimension in dimensions:
            dim_name = dimension.name
            if dim_name in self.pattern_contexts:
                # Get the most successful patterns for this dimension
                dim_patterns = [
                    self.interaction_patterns[pid] for pid in self.pattern_contexts[dim_name]
                    if pid in self.interaction_patterns
                ]
                
                # Sort by success rate and take top 3
                dim_patterns.sort(key=lambda p: p.success_rate, reverse=True)
                related_patterns.extend([p.pattern_id for p in dim_patterns[:3]])
        
        # Create the entry
        entry = GrowthJournalEntry(
            timestamp=datetime.utcnow().isoformat(),
            entry_type="milestone",
            dimension=",".join(d.name for d in dimensions),
            content=content,
            metrics=current_metrics,
            references=related_patterns
        )
        
        self.growth_journal.append(entry)
        self._save_growth_journal_entry(entry)
        
        return entry.to_dict()
