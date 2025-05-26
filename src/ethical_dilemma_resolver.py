#!/usr/bin/env python3
"""
Ethical Dilemma Resolver

This module provides functionality for resolving ethical dilemmas by evaluating
possible actions against a hierarchy of ethical principles while balancing
efficiency considerations.
"""

import logging
import json
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
from datetime import datetime, timezone
from dataclasses import dataclass, field

from principle_engine import PrincipleEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EthicalDilemmaResolver")

class EthicalPriority(Enum):
    """Priority levels for different ethical principles."""
    CRITICAL = 5  # Highest priority, non-negotiable principles
    VERY_HIGH = 4  # Extremely important but may allow minor compromises
    HIGH = 3       # Important but may be balanced with other considerations
    MEDIUM = 2     # Significant but can be outweighed by higher priorities
    LOW = 1        # Consider when other priorities are satisfied
    SITUATIONAL = 0  # Priority depends on the specific context

@dataclass
class EthicalAction:
    """Represents a potential action in an ethical dilemma."""
    id: str
    description: str
    principle_scores: Dict[str, float] = field(default_factory=dict)
    efficiency_score: float = 0.5  # 0.0 (least efficient) to 1.0 (most efficient)
    weighted_score: float = 0.0
    justification: str = ""
    side_effects: List[str] = field(default_factory=list)
    context_alignment: float = 0.5  # How well action aligns with the specific context

@dataclass
class EthicalDilemma:
    """Represents an ethical dilemma requiring resolution."""
    id: str
    description: str
    context: Dict[str, Any]
    possible_actions: List[EthicalAction]
    primary_principles: List[str]  # Principle IDs most relevant to this dilemma
    stakeholders: List[Dict[str, Any]] = field(default_factory=list)
    urgency: float = 0.5  # 0.0 (non-urgent) to 1.0 (extremely urgent)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass
class DilemmaResolution:
    """The resolution to an ethical dilemma."""
    dilemma_id: str
    recommended_action: EthicalAction
    alternate_actions: List[EthicalAction]
    principle_weights_used: Dict[str, float]
    efficiency_weight: float
    context_weight: float
    justification: str
    warnings: List[str] = field(default_factory=list)
    resolved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    confidence_score: float = 0.5  # 0.0 (low confidence) to 1.0 (high confidence)

def resolve_ethical_dilemma(
    principle_engine: PrincipleEngine,
    dilemma_description: str,
    possible_actions: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
    principle_hierarchy: Optional[Dict[str, EthicalPriority]] = None,
    efficiency_importance: float = 0.3,  # 0.0-1.0 representing importance of efficiency
    context_importance: float = 0.2,  # 0.0-1.0 representing importance of context alignment
    primary_principles: Optional[List[str]] = None  # Principle IDs most relevant to this dilemma
) -> Dict[str, Any]:
    """
    Resolve an ethical dilemma by evaluating possible actions against a hierarchy of ethical principles.
    
    Args:
        principle_engine: The PrincipleEngine to use for principle-based evaluation
        dilemma_description: Description of the ethical dilemma
        possible_actions: List of possible actions to evaluate
        context: Additional context for the dilemma
        principle_hierarchy: Dictionary mapping principle IDs to priority levels
        efficiency_importance: Importance of efficiency vs. ethics (0.0-1.0)
        context_importance: Importance of context-specific considerations (0.0-1.0)
        primary_principles: List of principle IDs most relevant to this dilemma
        
    Returns:
        Dictionary containing the recommended action and justification
    """
    # Validate inputs
    if not possible_actions:
        raise ValueError("Must provide at least one possible action")
    
    if efficiency_importance < 0.0 or efficiency_importance > 1.0:
        raise ValueError("efficiency_importance must be between 0.0 and 1.0")
    
    if context_importance < 0.0 or context_importance > 1.0:
        raise ValueError("context_importance must be between 0.0 and 1.0")
    
    # Use default context if none provided
    context = context or {}
    
    # Create unique ID for the dilemma
    dilemma_id = f"dilemma_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    
    # Set up default principle hierarchy if none provided
    if not principle_hierarchy:
        principle_hierarchy = _get_default_principle_hierarchy(principle_engine)
    
    # Set primary principles if not provided
    if not primary_principles:
        primary_principles = _identify_primary_principles(
            dilemma_description, context, principle_engine
        )
    
    # Create EthicalAction objects from the provided possible actions
    ethical_actions = []
    for i, action in enumerate(possible_actions):
        action_id = action.get("id", f"action_{i+1}")
        ethical_actions.append(EthicalAction(
            id=action_id,
            description=action.get("description", f"Action {i+1}"),
            efficiency_score=action.get("efficiency_score", 0.5),
            side_effects=action.get("side_effects", []),
            context_alignment=action.get("context_alignment", 0.5)
        ))
    
    # Create an EthicalDilemma object
    dilemma = EthicalDilemma(
        id=dilemma_id,
        description=dilemma_description,
        context=context,
        possible_actions=ethical_actions,
        primary_principles=primary_principles,
        urgency=context.get("urgency", 0.5),
        stakeholders=context.get("stakeholders", [])
    )
    
    # Evaluate each action against all principles
    _evaluate_actions_against_principles(dilemma, principle_engine)
    
    # Calculate weighted scores for each action
    ethics_weight = 1.0 - efficiency_importance - context_importance
    principle_weights = _calculate_principle_weights(principle_hierarchy, primary_principles)
    
    _calculate_weighted_scores(
        dilemma.possible_actions, 
        principle_weights, 
        ethics_weight, 
        efficiency_importance,
        context_importance
    )
    
    # Generate justifications for each action
    for action in dilemma.possible_actions:
        action.justification = _generate_justification(
            action, principle_engine, dilemma, principle_weights
        )
    
    # Sort actions by weighted score (highest first)
    sorted_actions = sorted(
        dilemma.possible_actions, 
        key=lambda a: a.weighted_score, 
        reverse=True
    )
    
    # Select the recommended action (highest weighted score)
    recommended_action = sorted_actions[0]
    alternate_actions = sorted_actions[1:] if len(sorted_actions) > 1 else []
    
    # Check for any warnings about the recommended action
    warnings = _generate_warnings(recommended_action, principle_engine, dilemma)
    
    # Calculate confidence in the recommendation
    confidence_score = _calculate_confidence_score(
        recommended_action, 
        alternate_actions, 
        principle_weights,
        ethics_weight
    )
    
    # Create the resolution
    resolution = DilemmaResolution(
        dilemma_id=dilemma.id,
        recommended_action=recommended_action,
        alternate_actions=alternate_actions,
        principle_weights_used=principle_weights,
        efficiency_weight=efficiency_importance,
        context_weight=context_importance,
        justification=recommended_action.justification,
        warnings=warnings,
        confidence_score=confidence_score
    )
    
    # Log the resolution
    logger.info(f"Ethical dilemma {dilemma_id} resolved with confidence {confidence_score:.2f}")
    if warnings:
        logger.warning(f"Warnings for dilemma {dilemma_id}: {', '.join(warnings)}")
    
    # Return the resolution in a structured format
    return {
        "dilemma_id": resolution.dilemma_id,
        "recommended_action": {
            "id": resolution.recommended_action.id,
            "description": resolution.recommended_action.description,
            "weighted_score": resolution.recommended_action.weighted_score,
            "principle_scores": resolution.recommended_action.principle_scores,
            "efficiency_score": resolution.recommended_action.efficiency_score,
            "context_alignment": resolution.recommended_action.context_alignment
        },
        "alternate_actions": [
            {
                "id": action.id,
                "description": action.description,
                "weighted_score": action.weighted_score,
                "principle_scores": action.principle_scores,
                "efficiency_score": action.efficiency_score,
                "context_alignment": action.context_alignment
            } for action in resolution.alternate_actions
        ],
        "justification": resolution.justification,
        "warnings": resolution.warnings,
        "confidence_score": resolution.confidence_score,
        "weights_used": {
            "principles": resolution.principle_weights_used,
            "efficiency": resolution.efficiency_weight,
            "context": resolution.context_weight
        },
        "resolved_at": resolution.resolved_at
    }

def _get_default_principle_hierarchy(principle_engine: PrincipleEngine) -> Dict[str, EthicalPriority]:
    """
    Get default hierarchy for principles based on their importance.
    
    Args:
        principle_engine: The PrincipleEngine containing principles
        
    Returns:
        Dictionary mapping principle IDs to priority levels
    """
    # Default hierarchy - in a real implementation, this might be loaded from a config file
    default_hierarchy = {
        "fairness_as_truth": EthicalPriority.CRITICAL,
        "harmony_through_presence": EthicalPriority.HIGH,
        "adaptability_as_strength": EthicalPriority.HIGH,
        "balance_in_mediation": EthicalPriority.VERY_HIGH,
        "clarity_in_complexity": EthicalPriority.MEDIUM,
        "integrity_in_transmission": EthicalPriority.CRITICAL,
        "resilience_through_connection": EthicalPriority.MEDIUM,
        "empathy_in_interface": EthicalPriority.HIGH,
        "truth_in_representation": EthicalPriority.VERY_HIGH,
        "growth_through_reflection": EthicalPriority.MEDIUM
    }
    
    # Ensure all principles in the engine have a priority
    for principle in principle_engine.principles:
        principle_id = principle["id"]
        if principle_id not in default_hierarchy:
            default_hierarchy[principle_id] = EthicalPriority.MEDIUM
    
    return default_hierarchy

def _identify_primary_principles(
    dilemma_description: str, 
    context: Dict[str, Any],
    principle_engine: PrincipleEngine
) -> List[str]:
    """
    Identify the primary principles most relevant to this dilemma.
    
    Args:
        dilemma_description: Description of the ethical dilemma
        context: Additional context for the dilemma
        principle_engine: The PrincipleEngine containing principles
        
    Returns:
        List of principle IDs most relevant to this dilemma
    """
    # In a real implementation, this would use NLP to identify relevant principles
    # For now, use a keyword-based approach
    
    # Combine description and context
    combined_text = dilemma_description
    if "description" in context:
        combined_text += " " + context["description"]
    
    combined_text = combined_text.lower()
    
    # Define keywords associated with each principle
    principle_keywords = {
        "fairness_as_truth": ["fair", "fairness", "equal", "equality", "bias", "impartial"],
        "harmony_through_presence": ["harmony", "presence", "acknowledge", "responsive", "communication"],
        "adaptability_as_strength": ["adapt", "adaptable", "flexibility", "evolve", "change"],
        "balance_in_mediation": ["balance", "mediate", "neutral", "mediation", "arbitrate"],
        "clarity_in_complexity": ["clarity", "clear", "simple", "simplify", "understand"],
        "integrity_in_transmission": ["integrity", "accurate", "accuracy", "reliable", "consistent"],
        "resilience_through_connection": ["resilience", "resilient", "robust", "withstand", "recover"],
        "empathy_in_interface": ["empathy", "empathetic", "understand", "perspective", "user"],
        "truth_in_representation": ["truth", "honest", "honesty", "represent", "transparent"],
        "growth_through_reflection": ["growth", "learn", "learning", "reflect", "improve"]
    }
    
    # Count keyword matches for each principle
    relevance_scores = {}
    for principle_id, keywords in principle_keywords.items():
        score = sum(combined_text.count(keyword) for keyword in keywords)
        relevance_scores[principle_id] = score
    
    # Get the top 3-5 most relevant principles
    sorted_principles = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Include at least 3 principles, but more if there are ties
    top_score = sorted_principles[0][1] if sorted_principles else 0
    primary_principles = [p[0] for p in sorted_principles if p[1] > 0]
    
    # If no principles were found through keywords (all scores 0), include critical ones
    if not primary_principles:
        for principle in principle_engine.principles:
            if principle["weight"] >= 0.8:  # High weight principles
                primary_principles.append(principle["id"])
    
    # Limit to a reasonable number
    if len(primary_principles) > 5:
        primary_principles = primary_principles[:5]
    
    # Make sure we have at least 3 principles if possible
    all_principle_ids = [p["id"] for p in principle_engine.principles]
    while len(primary_principles) < 3 and len(primary_principles) < len(all_principle_ids):
        # Add the next highest weighted principle that's not already included
        for principle in sorted(principle_engine.principles, key=lambda p: p["weight"], reverse=True):
            if principle["id"] not in primary_principles:
                primary_principles.append(principle["id"])
                break
    
    return primary_principles

def _evaluate_actions_against_principles(
    dilemma: EthicalDilemma,
    principle_engine: PrincipleEngine
) -> None:
    """
    Evaluate all possible actions against all principles.
    
    Args:
        dilemma: The ethical dilemma containing possible actions
        principle_engine: The PrincipleEngine for evaluation
        
    Modifies:
        Updates the principle_scores in each action
    """
    for action in dilemma.possible_actions:
        # Convert action to message-like format for principle evaluation
        action_message = {
            "id": action.id,
            "method": "evaluateAction",
            "params": {
                "dilemma_id": dilemma.id,
                "action_description": action.description,
                "context": dilemma.context,
                "side_effects": action.side_effects
            }
        }
        
        # Evaluate against all principles
        evaluation = principle_engine.evaluate_message(action_message)
        
        # Store principle scores in the action
        action.principle_scores = {
            principle_id: score_data["score"] 
            for principle_id, score_data in evaluation["principle_scores"].items()
        }

def _calculate_principle_weights(
    principle_hierarchy: Dict[str, EthicalPriority],
    primary_principles: List[str]
) -> Dict[str, float]:
    """
    Calculate weights for each principle based on hierarchy and relevance.
    
    Args:
        principle_hierarchy: Dictionary mapping principle IDs to priority levels
        primary_principles: List of principle IDs most relevant to this dilemma
        
    Returns:
        Dictionary mapping principle IDs to weights (0.0-1.0)
    """
    # Start with base weights from hierarchy
    weights = {}
    
    # Convert priority levels to base weights
    for principle_id, priority in principle_hierarchy.items():
        # Base weight based on priority level
        if priority == EthicalPriority.CRITICAL:
            weights[principle_id] = 1.0
        elif priority == EthicalPriority.VERY_HIGH:
            weights[principle_id] = 0.8
        elif priority == EthicalPriority.HIGH:
            weights[principle_id] = 0.6
        elif priority == EthicalPriority.MEDIUM:
            weights[principle_id] = 0.4
        elif priority == EthicalPriority.LOW:
            weights[principle_id] = 0.2
        else:  # SITUATIONAL
            weights[principle_id] = 0.3  # Default weight, will be adjusted based on relevance
    
    # Boost weights for primary principles
    for principle_id in primary_principles:
        if principle_id in weights:
            # Boost by 50% but cap at 1.0
            weights[principle_id] = min(1.0, weights[principle_id] * 1.5)
    
    # Normalize weights to sum to 1.0
    total_weight = sum(weights.values())
    if total_weight > 0:
        for principle_id in weights:
            weights[principle_id] /= total_weight
    
    return weights

def _calculate_weighted_scores(
    actions: List[EthicalAction],
    principle_weights: Dict[str, float],
    ethics_weight: float,
    efficiency_weight: float,
    context_weight: float
) -> None:
    """
    Calculate weighted scores for each action.
    
    Args:
        actions: List of possible actions
        principle_weights: Weights for each principle
        ethics_weight: Overall weight for ethical considerations
        efficiency_weight: Weight for efficiency considerations
        context_weight: Weight for context alignment
        
    Modifies:
        Updates the weighted_score in each action
    """
    for action in actions:
        # Calculate ethical score
        ethical_score = 0.0
        total_weight = 0.0
        
        for principle_id, weight in principle_weights.items():
            if principle_id in action.principle_scores:
                principle_score = action.principle_scores[principle_id] / 100.0  # Convert to 0.0-1.0
                ethical_score += principle_score * weight
                total_weight += weight
        
        # Normalize ethical score if needed
        if total_weight > 0:
            ethical_score /= total_weight
        
        # Calculate final weighted score
        action.weighted_score = (
            (ethical_score * ethics_weight) +
            (action.efficiency_score * efficiency_weight) +
            (action.context_alignment * context_weight)
        )

def _generate_justification(
    action: EthicalAction,
    principle_engine: PrincipleEngine,
    dilemma: EthicalDilemma,
    principle_weights: Dict[str, float]
) -> str:
    """
    Generate a justification for an action.
    
    Args:
        action: The action to justify
        principle_engine: The PrincipleEngine containing principles
        dilemma: The ethical dilemma
        principle_weights: Weights for each principle
        
    Returns:
        Justification string
    """
    # Get principle names
    principle_names = {
        p["id"]: p["name"] for p in principle_engine.principles
    }
    
    # Start with a general statement
    justification = f"This action is recommended because it "
    
    # Add information about highest-scoring principles
    high_scoring_principles = []
    for principle_id, score in action.principle_scores.items():
        if score >= 80 and principle_id in principle_weights and principle_weights[principle_id] >= 0.1:
            principle_name = principle_names.get(principle_id, principle_id)
            high_scoring_principles.append(principle_name)
    
    if high_scoring_principles:
        if len(high_scoring_principles) == 1:
            justification += f"strongly aligns with the principle of {high_scoring_principles[0]}"
        else:
            justification += f"strongly aligns with the principles of {', '.join(high_scoring_principles[:-1])} and {high_scoring_principles[-1]}"
    else:
        justification += "provides a balanced approach to the ethical considerations involved"
    
    # Add efficiency information if it's a significant factor
    if action.efficiency_score >= 0.7:
        justification += ", while also being highly efficient"
    elif action.efficiency_score <= 0.3:
        justification += ", although it may require more resources than other options"
    
    # Add context information
    if action.context_alignment >= 0.7:
        justification += ". It is particularly well-suited to the specific context of this situation"
    
    # Add information about principle trade-offs if applicable
    low_scoring_principles = []
    for principle_id, score in action.principle_scores.items():
        if score <= 50 and principle_id in principle_weights and principle_weights[principle_id] >= 0.1:
            principle_name = principle_names.get(principle_id, principle_id)
            low_scoring_principles.append(principle_name)
    
    if low_scoring_principles:
        justification += f". This choice does involve trade-offs with respect to {', '.join(low_scoring_principles)}, but these are outweighed by the other considerations"
    
    # Add side effect information if available
    if action.side_effects:
        justification += f". It's important to note potential side effects: {', '.join(action.side_effects)}"
    
    return justification

def _generate_warnings(
    action: EthicalAction,
    principle_engine: PrincipleEngine,
    dilemma: EthicalDilemma
) -> List[str]:
    """
    Generate warnings about the recommended action.
    
    Args:
        action: The recommended action
        principle_engine: The PrincipleEngine containing principles
        dilemma: The ethical dilemma
        
    Returns:
        List of warning strings
    """
    warnings = []
    
    # Check for critically low principle scores
    for principle_id, score in action.principle_scores.items():
        # For critical principles, any score below 70 is concerning
        principle = next((p for p in principle_engine.principles if p["id"] == principle_id), None)
        if principle and principle.get("weight", 0) >= 0.9 and score < 70:
            warnings.append(f"This action scores low ({score}) on the critical principle: {principle['name']}")
    
    # Check for particularly harmful side effects
    for side_effect in action.side_effects:
        if "severe" in side_effect.lower() or "harm" in side_effect.lower() or "critical" in side_effect.lower():
            warnings.append(f"Potentially serious side effect: {side_effect}")
    
    # Check for very low overall ethical score
    ethical_score = sum(action.principle_scores.values()) / len(action.principle_scores) if action.principle_scores else 0
    if ethical_score < 60:
        warnings.append(f"Overall ethical score is relatively low ({ethical_score:.1f}/100)")
    
    # Check for very low efficiency in urgent situations
    if dilemma.urgency > 0.8 and action.efficiency_score < 0.4:
        warnings.append("This action has low efficiency in a situation marked as urgent")
    
    return warnings

def _calculate_confidence_score(
    recommended_action: EthicalAction,
    alternate_actions: List[EthicalAction],
    principle_weights: Dict[str, float],
    ethics_weight: float
) -> float:
    """
    Calculate confidence score for the recommendation.
    
    Args:
        recommended_action: The recommended action
        alternate_actions: Alternative actions
        principle_weights: Weights for each principle
        ethics_weight: Weight given to ethical considerations
        
    Returns:
        Confidence score (0.0-1.0)
    """
    # Base confidence starts at 0.5
    confidence = 0.5
    
    # Factor 1: Margin by which recommended action exceeds alternatives
    if alternate_actions:
        next_best_score = alternate_actions[0].weighted_score
        score_margin = recommended_action.weighted_score - next_best_score
        
        # Normalize score margin to 0.0-0.3 range
        normalized_margin = min(0.3, score_margin * 3.0)
        confidence += normalized_margin
    else:
        # No alternatives, slightly higher confidence
        confidence += 0.1
    
    # Factor 2: Consistency across principles
    principle_scores = list(recommended_action.principle_scores.values())
    if principle_scores:
        # Calculate standard deviation (normalized to 0-100 scale)
        import statistics
        try:
            stdev = statistics.stdev(principle_scores)
            normalized_stdev = max(0, 30 - stdev) / 30  # Lower stdev = higher consistency
            confidence += normalized_stdev * 0.2
        except statistics.StatisticsError:
            # Not enough data points for stdev
            pass
    
    # Factor 3: Strong alignment with highly weighted principles
    weighted_alignment = 0
    total_weight = 0
    for principle_id, weight in principle_weights.items():
        if principle_id in recommended_action.principle_scores:
            score = recommended_action.principle_scores[principle_id]
            weighted_alignment += score * weight
            total_weight += weight
    
    if total_weight > 0:
        avg_weighted_alignment = weighted_alignment / total_weight
        # Convert to 0.0-0.2 range (80+ is good alignment)
        normalized_alignment = min(0.2, max(0, (avg_weighted_alignment - 60) / 100))
        confidence += normalized_alignment
    
    # Cap confidence at 1.0
    confidence = min(1.0, confidence)
    
    return confidence