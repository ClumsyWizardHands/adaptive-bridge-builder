#!/usr/bin/env python3
"""
Strategic Adaptation Module

This module provides functionality for adapting communication strategies based on
historical performance data and learning patterns while ensuring alignment with
core principles. It serves as a central component of the continuous evolution system.
"""

import logging
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum, auto
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
import statistics
from collections import Counter, defaultdict
import copy

from principle_engine import PrincipleEngine
from learning_system import (
    LearningSystem, LearningDimension, OutcomeType, 
    InteractionPattern, AdaptationLevel
)
from communication_style import (
    CommunicationStyle, EmotionalTone, FormalityLevel
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("StrategicAdaptation")

class StrategyComponentType(Enum):
    """Types of communication strategy components."""
    FORMALITY_LEVEL = auto()
    DETAIL_LEVEL = auto()
    EMOTIONAL_TONE = auto()
    RESPONSE_TIME = auto()
    COMMUNICATION_STRUCTURE = auto()
    VOCABULARY_COMPLEXITY = auto()
    FEEDBACK_FREQUENCY = auto()
    METAPHOR_USAGE = auto()
    TECHNICAL_DENSITY = auto()

@dataclass
class StrategyEvaluation:
    """
    Evaluation of a communication strategy.
    """
    strategy_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    effectiveness_score: float = 0.5  # 0.0 to 1.0
    component_scores: Dict[str, float] = field(default_factory=dict)
    target_audience: Optional[str] = None
    context_factors: Dict[str, Any] = field(default_factory=dict)
    principle_alignment: Dict[str, float] = field(default_factory=dict)
    historic_trend: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "strategy_id": self.strategy_id,
            "timestamp": self.timestamp,
            "effectiveness_score": self.effectiveness_score,
            "component_scores": self.component_scores,
            "target_audience": self.target_audience,
            "context_factors": self.context_factors,
            "principle_alignment": self.principle_alignment,
            "historic_trend": self.historic_trend
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyEvaluation':
        """Create from dictionary representation."""
        return cls(
            strategy_id=data["strategy_id"],
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            effectiveness_score=data.get("effectiveness_score", 0.5),
            component_scores=data.get("component_scores", {}),
            target_audience=data.get("target_audience"),
            context_factors=data.get("context_factors", {}),
            principle_alignment=data.get("principle_alignment", {}),
            historic_trend=data.get("historic_trend", [])
        )

@dataclass
class CommunicationStrategy:
    """
    A comprehensive communication strategy with various components
    that can be adapted based on performance data.
    """
    strategy_id: str
    name: str
    description: str
    target_audiences: List[str]
    style_parameters: Dict[str, Any]
    formality_level: Union[FormalityLevel, str]
    emotional_tone: Union[EmotionalTone, str]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    adaptations_history: List[Dict[str, Any]] = field(default_factory=list)
    core_values: List[str] = field(default_factory=list)
    immutable_aspects: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "description": self.description,
            "target_audiences": self.target_audiences,
            "style_parameters": self.style_parameters,
            "formality_level": self.formality_level.value if isinstance(self.formality_level, FormalityLevel) else self.formality_level,
            "emotional_tone": self.emotional_tone.value if isinstance(self.emotional_tone, EmotionalTone) else self.emotional_tone,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "performance_history": self.performance_history,
            "adaptations_history": self.adaptations_history,
            "core_values": self.core_values,
            "immutable_aspects": self.immutable_aspects
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommunicationStrategy':
        """Create from dictionary representation."""
        # Convert string enums to actual enum values if needed
        formality_level = data.get("formality_level")
        if isinstance(formality_level, str):
            try:
                formality_level = FormalityLevel[formality_level]
            except:
                pass
                
        emotional_tone = data.get("emotional_tone")
        if isinstance(emotional_tone, str):
            try:
                emotional_tone = EmotionalTone[emotional_tone]
            except:
                pass
                
        return cls(
            strategy_id=data["strategy_id"],
            name=data["name"],
            description=data["description"],
            target_audiences=data.get("target_audiences", []),
            style_parameters=data.get("style_parameters", {}),
            formality_level=formality_level,
            emotional_tone=emotional_tone,
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
            performance_history=data.get("performance_history", []),
            adaptations_history=data.get("adaptations_history", []),
            core_values=data.get("core_values", []),
            immutable_aspects=data.get("immutable_aspects", [])
        )

def adapt_strategy(
    strategy: CommunicationStrategy,
    learning_system: LearningSystem,
    principle_engine: Optional[PrincipleEngine] = None,
    context: Optional[Dict[str, Any]] = None,
    adaptation_level: AdaptationLevel = AdaptationLevel.MODERATE
) -> Tuple[CommunicationStrategy, Dict[str, Any]]:
    """
    Evaluates and adapts a communication strategy based on historical performance
    data while maintaining alignment with core values and principles.
    
    Args:
        strategy: The current communication strategy to evaluate and adapt
        learning_system: LearningSystem containing historical performance data
        principle_engine: Optional PrincipleEngine for principle alignment checks
        context: Optional context information for adaptation decisions
        adaptation_level: Level of adaptation to apply (from NONE to COMPLETE)
        
    Returns:
        Tuple of (adapted_strategy, adaptation_details) where:
        - adapted_strategy is the modified communication strategy
        - adaptation_details contains information about the changes made
    """
    if adaptation_level == AdaptationLevel.NONE:
        return strategy, {"adapted": False, "reason": "Adaptation level set to NONE"}
    
    context = context or {}
    
    # Step 1: Evaluate current strategy effectiveness
    evaluation = _evaluate_strategy(strategy, learning_system, principle_engine)
    
    # Step 2: Generate adaptation plan based on evaluation
    adaptation_plan = _generate_adaptation_plan(strategy, evaluation, adaptation_level)
    
    # If no adaptations are needed or possible, return original strategy
    if not adaptation_plan["adaptations"]:
        return strategy, {
            "adapted": False,
            "reason": adaptation_plan.get("reason", "No adaptations needed"),
            "evaluation": evaluation.to_dict()
        }
    
    # Step 3: Apply adaptations to create new strategy
    adapted_strategy = _apply_adaptations(strategy, adaptation_plan["adaptations"], principle_engine)
    
    # Step 4: Validate the adapted strategy against principles
    if principle_engine:
        validation_result = _validate_strategy_principles(adapted_strategy, principle_engine)
        if not validation_result["valid"]:
            # If invalid, revert problematic adaptations
            adaptations = adaptation_plan["adaptations"]
            problematic_components = validation_result["problematic_components"]
            
            logger.warning(f"Reverting problematic adaptations: {problematic_components}")
            # Filter out problematic adaptations
            filtered_adaptations = [
                a for a in adaptations 
                if a["component"] not in problematic_components
            ]
            
            # Re-apply remaining adaptations
            if filtered_adaptations:
                adapted_strategy = _apply_adaptations(strategy, filtered_adaptations, principle_engine)
            else:
                # If all adaptations are problematic, return original
                return strategy, {
                    "adapted": False,
                    "reason": "All adaptations violated principles",
                    "evaluation": evaluation.to_dict(),
                    "validation_result": validation_result
                }
    
    # Step 5: Record the adaptation in the strategy's history
    timestamp = datetime.now(timezone.utc).isoformat()
    adaptation_record = {
        "timestamp": timestamp,
        "adaptation_level": adaptation_level.value,
        "before_evaluation": evaluation.to_dict(),
        "adaptations": adaptation_plan["adaptations"],
        "context": context
    }
    
    adapted_strategy.adaptations_history.append(adaptation_record)
    adapted_strategy.updated_at = timestamp
    
    # Step 6: Return the adapted strategy with details
    return adapted_strategy, {
        "adapted": True,
        "adaptation_level": adaptation_level.value,
        "evaluation": evaluation.to_dict(),
        "changes": adaptation_plan["adaptations"],
        "timestamp": timestamp
    }

def _evaluate_strategy(
    strategy: CommunicationStrategy,
    learning_system: LearningSystem,
    principle_engine: Optional[PrincipleEngine] = None
) -> StrategyEvaluation:
    """
    Evaluate the effectiveness of a communication strategy based on historical data.
    
    Args:
        strategy: The communication strategy to evaluate
        learning_system: LearningSystem containing historical performance data
        principle_engine: Optional PrincipleEngine for principle alignment
        
    Returns:
        StrategyEvaluation with effectiveness scores
    """
    # Initialize evaluation
    evaluation = StrategyEvaluation(
        strategy_id=strategy.strategy_id,
        target_audience=strategy.target_audiences[0] if strategy.target_audiences else None
    )
    
    # Extract relevant patterns from learning system
    communication_patterns = _extract_relevant_patterns(strategy, learning_system)
    
    # If no relevant patterns found, return default evaluation
    if not communication_patterns:
        logger.info(f"No relevant patterns found for strategy {strategy.name}")
        return evaluation
    
    # Calculate overall effectiveness score
    success_rates = [pattern.success_rate for pattern in communication_patterns]
    if success_rates:
        # Weight by confidence
        confidence_weights = [pattern.confidence for pattern in communication_patterns]
        weighted_rates = [
            rate * weight for rate, weight in zip(success_rates, confidence_weights)
        ]
        total_weight = sum(confidence_weights)
        if total_weight > 0:
            evaluation.effectiveness_score = sum(weighted_rates) / total_weight
    
    # Calculate component-specific scores
    component_scores = defaultdict(list)
    components_to_evaluate = [
        "formality_level", "detail_level", "emotional_tone", 
        "response_time", "technical_density"
    ]
    
    # Extract component metrics from patterns
    for pattern in communication_patterns:
        for component in components_to_evaluate:
            component_key = f"style_{component}"
            if component_key in pattern.context:
                component_value = pattern.context[component_key]
                # Record component with success rate
                component_scores[component].append({
                    "value": component_value,
                    "success_rate": pattern.success_rate,
                    "confidence": pattern.confidence
                })
    
    # Calculate score for each component
    for component, scores in component_scores.items():
        # Group by component value
        value_scores = defaultdict(list)
        for entry in scores:
            value_scores[entry["value"]].append((entry["success_rate"], entry["confidence"]))
        
        # Find best performing value for each component
        best_value = None
        best_score = 0.0
        
        for value, entries in value_scores.items():
            rates, weights = zip(*entries)
            weighted_score = sum(r * w for r, w in zip(rates, weights)) / sum(weights) if sum(weights) > 0 else 0
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_value = value
        
        if best_value is not None:
            evaluation.component_scores[component] = {
                "best_value": best_value,
                "score": best_score
            }
    
    # If principle_engine available, calculate principle alignment
    if principle_engine:
        # Convert strategy to a format that principle_engine can evaluate
        strategy_data = {
            "type": "communication_strategy",
            "content": strategy.to_dict()
        }
        
        # Get applicable principles
        principles = principle_engine.get_applicable_principles(strategy_data, {})
        
        # Check alignment with principles
        for principle in principles:
            alignment_score = principle_engine.check_principle_alignment(
                strategy_data, principle, {}
            )
            evaluation.principle_alignment[principle["name"]] = alignment_score
    
    # Add historical trend
    if strategy.performance_history:
        evaluation.historic_trend = [
            entry.get("effectiveness_score", 0.5) 
            for entry in strategy.performance_history[-10:]  # Last 10 entries
        ]
    
    return evaluation

def _extract_relevant_patterns(
    strategy: CommunicationStrategy,
    learning_system: LearningSystem
) -> List[InteractionPattern]:
    """
    Extract interaction patterns relevant to the given communication strategy.
    
    Args:
        strategy: The communication strategy
        learning_system: The learning system containing patterns
        
    Returns:
        List of relevant interaction patterns
    """
    relevant_patterns = []
    
    # Define relevance criteria based on strategy properties
    relevant_criteria = {
        "formality_level": strategy.formality_level.name if isinstance(strategy.formality_level, FormalityLevel) else strategy.formality_level,
        "emotional_tone": strategy.emotional_tone.name if isinstance(strategy.emotional_tone, EmotionalTone) else strategy.emotional_tone,
    }
    
    # Add any style parameters present in the strategy
    for param, value in strategy.style_parameters.items():
        if isinstance(value, (int, float, str, bool)):
            relevant_criteria[param] = value
    
    # Find patterns that match the criteria
    for pattern_id, pattern in learning_system.interaction_patterns.items():
        # Check for pattern relevance based on context
        relevance_score = 0
        for key, value in relevant_criteria.items():
            context_key = f"style_{key}"
            if context_key in pattern.context and pattern.context[context_key] == value:
                relevance_score += 1
        
        # Include patterns with at least some relevance
        if relevance_score > 0:
            relevant_patterns.append(pattern)
    
    return relevant_patterns

def _generate_adaptation_plan(
    strategy: CommunicationStrategy,
    evaluation: StrategyEvaluation,
    adaptation_level: AdaptationLevel
) -> Dict[str, Any]:
    """
    Generate a plan for adapting the communication strategy based on evaluation.
    
    Args:
        strategy: The current communication strategy
        evaluation: Evaluation of the strategy
        adaptation_level: Level of adaptation to apply
        
    Returns:
        Dictionary with adaptations to apply
    """
    # Initialize adaptation plan
    plan = {
        "adaptations": [],
        "rationale": []
    }
    
    # If strategy is performing well (>0.8), minimal adaptations needed
    if evaluation.effectiveness_score > 0.8:
        # Only make minor tweaks if adaptation level is high enough
        if adaptation_level in [AdaptationLevel.SIGNIFICANT, AdaptationLevel.COMPLETE]:
            plan["rationale"].append("Strategy performing well, only minor optimizations needed")
        else:
            plan["rationale"].append("Strategy performing well, no adaptations needed")
            return plan
    
    # Define immutable aspects from strategy
    immutable_aspects = set(strategy.immutable_aspects)
    
    # Determine which components can be adapted based on adaptation level
    adaptable_components = []
    
    if adaptation_level == AdaptationLevel.MINIMAL:
        # Only adapt non-core parameters with clear improvements
        adaptable_components = ["detail_level", "response_time"]
    elif adaptation_level == AdaptationLevel.MODERATE:
        # Adapt most parameters except core emotional/formality aspects
        adaptable_components = [
            "detail_level", "response_time", "vocabulary_complexity", 
            "technical_density", "feedback_frequency"
        ]
    elif adaptation_level == AdaptationLevel.SIGNIFICANT:
        # Adapt almost everything except immutable aspects
        adaptable_components = [
            "detail_level", "response_time", "vocabulary_complexity", 
            "technical_density", "feedback_frequency", "emotional_tone",
            "formality_level", "communication_structure"
        ]
    elif adaptation_level == AdaptationLevel.COMPLETE:
        # Adapt everything except explicitly immutable aspects
        adaptable_components = [
            "detail_level", "response_time", "vocabulary_complexity",
            "technical_density", "feedback_frequency", "emotional_tone",
            "formality_level", "communication_structure", "metaphor_usage"
        ]
    
    # Filter out immutable aspects
    adaptable_components = [c for c in adaptable_components if c not in immutable_aspects]
    
    # Check component scores to identify improvement opportunities
    for component, data in evaluation.component_scores.items():
        if component not in adaptable_components:
            continue
            
        current_value = None
        
        # Get current component value from strategy
        if component == "formality_level":
            current_value = strategy.formality_level.name if isinstance(strategy.formality_level, FormalityLevel) else strategy.formality_level
        elif component == "emotional_tone":
            current_value = strategy.emotional_tone.name if isinstance(strategy.emotional_tone, EmotionalTone) else strategy.emotional_tone
        else:
            current_value = strategy.style_parameters.get(component)
        
        best_value = data.get("best_value")
        
        # If current value is different from best value and best value performance is good
        if best_value and current_value != best_value and data.get("score", 0) > 0.6:
            plan["adaptations"].append({
                "component": component,
                "from_value": current_value,
                "to_value": best_value,
                "confidence": data.get("score", 0.5),
                "rationale": f"Historical data suggests '{best_value}' performs better than '{current_value}'"
            })
            plan["rationale"].append(f"Adapting {component} based on historical performance")
    
    # Check historical trend for additional insights
    if evaluation.historic_trend and len(evaluation.historic_trend) >= 3:
        # Check if performance is declining
        if evaluation.historic_trend[-1] < evaluation.historic_trend[-3]:
            plan["rationale"].append("Performance trend is declining, more aggressive adaptation may be needed")
            
            # If not already adapting formality and it's adaptable, consider it
            if "formality_level" in adaptable_components and not any(a["component"] == "formality_level" for a in plan["adaptations"]):
                # Adjust formality based on current level
                current_formality = strategy.formality_level
                if isinstance(current_formality, str):
                    try:
                        current_formality = FormalityLevel[current_formality]
                    except:
                        current_formality = FormalityLevel.NEUTRAL
                
                # Suggest formality adjustment
                if current_formality == FormalityLevel.VERY_FORMAL:
                    new_formality = FormalityLevel.FORMAL
                elif current_formality == FormalityLevel.FORMAL:
                    new_formality = FormalityLevel.NEUTRAL
                elif current_formality == FormalityLevel.CASUAL:
                    new_formality = FormalityLevel.NEUTRAL
                elif current_formality == FormalityLevel.VERY_CASUAL:
                    new_formality = FormalityLevel.CASUAL
                else:  # NEUTRAL
                    # Try something different based on audience
                    audience = strategy.target_audiences[0] if strategy.target_audiences else None
                    if audience and "technical" in audience.lower():
                        new_formality = FormalityLevel.FORMAL
                    else:
                        new_formality = FormalityLevel.CASUAL
                
                plan["adaptations"].append({
                    "component": "formality_level",
                    "from_value": current_formality.name if isinstance(current_formality, FormalityLevel) else current_formality,
                    "to_value": new_formality.name,
                    "confidence": 0.6,
                    "rationale": "Adjusting formality level to address declining performance trend"
                })
    
    # If no adaptations were generated but adaptation is needed
    if not plan["adaptations"] and evaluation.effectiveness_score < 0.6:
        plan["rationale"].append("Strategy performing poorly but no clear improvement patterns identified")
        
        # Suggest general improvements based on common patterns
        if "detail_level" in adaptable_components:
            plan["adaptations"].append({
                "component": "detail_level",
                "from_value": strategy.style_parameters.get("detail_level", "MEDIUM"),
                "to_value": "HIGH" if evaluation.effectiveness_score < 0.4 else "MEDIUM_HIGH",
                "confidence": 0.5,
                "rationale": "Increasing detail level to provide more comprehensive information"
            })
        
        if "response_time" in adaptable_components:
            plan["adaptations"].append({
                "component": "response_time",
                "from_value": strategy.style_parameters.get("response_time", "NORMAL"),
                "to_value": "FAST",
                "confidence": 0.5,
                "rationale": "Decreasing response time to improve engagement"
            })
    
    return plan

def _apply_adaptations(
    strategy: CommunicationStrategy,
    adaptations: List[Dict[str, Any]],
    principle_engine: Optional[PrincipleEngine] = None
) -> CommunicationStrategy:
    """
    Apply adaptations to a communication strategy.
    
    Args:
        strategy: The communication strategy to adapt
        adaptations: List of adaptations to apply
        principle_engine: Optional PrincipleEngine for validation
        
    Returns:
        The adapted communication strategy
    """
    # Create a copy of the strategy to modify
    adapted_strategy = copy.deepcopy(strategy)
    
    # Apply each adaptation
    for adaptation in adaptations:
        component = adaptation["component"]
        to_value = adaptation["to_value"]
        
        # Apply based on component type
        if component == "formality_level":
            try:
                adapted_strategy.formality_level = FormalityLevel[to_value]
            except:
                adapted_strategy.formality_level = to_value
        elif component == "emotional_tone":
            try:
                adapted_strategy.emotional_tone = EmotionalTone[to_value]
            except:
                adapted_strategy.emotional_tone = to_value
        else:
            # For other components, update style parameters
            adapted_strategy.style_parameters[component] = to_value
    
    # Update description to reflect adaptations
    if adaptations:
        adapted_strategy.description = f"{strategy.description} (Adapted: {', '.join([a['component'] for a in adaptations])})"
    
    return adapted_strategy

def _validate_strategy_principles(
    strategy: CommunicationStrategy,
    principle_engine: PrincipleEngine
) -> Dict[str, Any]:
    """
    Validate a strategy against principles.
    
    Args:
        strategy: The strategy to validate
        principle_engine: The principle engine for validation
        
    Returns:
        Validation result dictionary
    """
    # Convert strategy to a format that principle_engine can evaluate
    strategy_data = {
        "type": "communication_strategy",
        "content": strategy.to_dict()
    }
    
    # Get applicable principles
    principles = principle_engine.get_applicable_principles(strategy_data, {})
    
    # Check alignment with each principle
    problematic_components = []
    alignment_scores = {}
    
    for principle in principles:
        alignment_score = principle_engine.check_principle_alignment(
            strategy_data, principle, {}
        )
        alignment_scores[principle["name"]] = alignment_score
        
        # If alignment is poor, identify problematic components
        if alignment_score < 0.6:
            # Check which components might be causing the issue
            for component in ["formality_level", "emotional_tone"]:
                # This is a simplified check - a real implementation would be more sophisticated
                if component not in problematic_components:
                    problematic_components.append(component)
            
            # Also check style parameters that might relate to this principle
            for param, value in strategy.style_parameters.items():
                if param not in problematic_components:
                    problematic_components.append(param)
    
    return {
        "valid": not problematic_components,
        "alignment_scores": alignment_scores,
        "problematic_components": problematic_components
    }

def integrate_with_continuous_evolution(
    continuous_evolution_system,
    strategy: CommunicationStrategy
) -> Tuple[CommunicationStrategy, Dict[str, Any]]:
    """
    Integrate strategy adaptation with the continuous evolution system.
    
    Args:
        continuous_evolution_system: ContinuousEvolutionSystem instance
        strategy: Communication strategy to adapt
        
    Returns:
        Tuple of (adapted_strategy, adaptation_details)
    """
    # Get required components from ContinuousEvolutionSystem
    learning_system = continuous_evolution_system.learning_system
    principle_engine = continuous_evolution_system.principle_engine
    
    # Determine adaptation level based on system state
    adaptation_level = AdaptationLevel.MODERATE
    
    # If recent reflection showed declining performance, increase adaptation
    if hasattr(continuous_evolution_system, 'last_reflection') and continuous_evolution_system.last_reflection:
        # This is a simplification - real implementation would analyze reflection results
        adaptation_level = AdaptationLevel.SIGNIFICANT
    
    # Adapt the strategy
    adapted_strategy, adaptation_details = adapt_strategy(
        strategy, 
        learning_system, 
        principle_engine,
        adaptation_level=adaptation_level
    )
    
    # If strategy was adapted, record it in the growth journal
    if adaptation_details.get("adapted", False):
        # Create journal entry
        entry_content = f"Adapted communication strategy: {strategy.name}\n"
        entry_content += f"Adaptation level: {adaptation_level.name}\n"
        entry_content += f"Previous effectiveness score: {adaptation_details['evaluation']['effectiveness_score']:.2f}\n"
        entry_content += f"Changes applied: {len(adaptation_details['changes'])}\n\n"
        
        for change in adaptation_details['changes']:
            entry_content += f"- {change['component']}: {change['from_value']} -> {change['to_value']}\n"
            entry_content += f"  Rationale: {change['rationale']}\n"
        
        # Add to growth journal
        if hasattr(continuous_evolution_system, 'learning_system') and hasattr(continuous_evolution_system.learning_system, 'growth_journal'):
            entry = continuous_evolution_system.learning_system.__class__.GrowthJournalEntry(
                timestamp=datetime.now(timezone.utc).isoformat(),
                entry_type="strategy_adaptation",
                dimension="COMMUNICATION_ADAPTATION",
                content=entry_content,
                metrics={
                    "before_effectiveness": adaptation_details['evaluation']['effectiveness_score'],
                    "adaptation_level": adaptation_level.value,
                    "changes_count": len(adaptation_details['changes'])
                },
                references=[strategy.strategy_id]
            )
            
            continuous_evolution_system.learning_system.growth_journal.append(entry)
            
            # Save entry if method exists
            if hasattr(continuous_evolution_system.learning_system, '_save_growth_journal_entry'):
                continuous_evolution_system.learning_system._save_growth_journal_entry(entry)
    
    return adapted_strategy, adaptation_details