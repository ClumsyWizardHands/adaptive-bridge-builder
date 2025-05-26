#!/usr/bin/env python3
"""
Principle Engine Fairness Extension Module

This module provides functions to evaluate fairness of messages and actions
using the FairnessEvaluator, extending the PrincipleEngine's capabilities.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set

from principle_engine import PrincipleEngine
from fairness_evaluator import FairnessEvaluator, FairnessFlag, FairnessMetric, FairnessAlternative

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PrincipleEngineFairness")

def evaluate_fairness(
    message: Dict[str, Any],
    context: Dict[str, Any] = None,
    principle_engine: Optional[PrincipleEngine] = None,
    fairness_evaluator: Optional[FairnessEvaluator] = None
) -> Dict[str, Any]:
    """
    Evaluate the fairness of a message or action and provide alternatives if biased.
    
    Args:
        message: The message or action to evaluate
        context: Additional context for evaluation
        principle_engine: Optional PrincipleEngine instance
        fairness_evaluator: Optional FairnessEvaluator instance
        
    Returns:
        Dictionary containing evaluation results:
        - 'score': Overall fairness score (0.0-1.0)
        - 'metrics': List of dimension-specific fairness metrics
        - 'flags': List of specific fairness issues detected
        - 'alternatives': List of suggested alternatives for biased content
        - 'is_fair': Boolean indicating if the message passes the fairness check
    """
    # Create or use provided FairnessEvaluator
    evaluator = fairness_evaluator
    if evaluator is None:
        evaluator = FairnessEvaluator(principle_engine)
    
    # Evaluate message for fairness issues
    metrics, flags = evaluator.evaluate_message(message, context)
    
    # Generate alternatives if biased content is detected
    alternatives = []
    if flags:
        alternatives = evaluator.generate_alternatives(flags)
    
    # Calculate overall fairness score as weighted average of dimension scores
    dimension_weights = {
        "language_bias": 0.2,
        "assumption_bias": 0.15,
        "treatment_bias": 0.2,
        "perspective_diversity": 0.15,
        "decision_consistency": 0.3
    }
    
    total_weight = 0.0
    weighted_score_sum = 0.0
    
    for metric in metrics:
        weight = dimension_weights.get(metric.dimension, 0.1)  # Default weight for other dimensions
        weighted_score_sum += metric.score * weight
        total_weight += weight
    
    # If no metrics are available, default to neutral score
    overall_score = weighted_score_sum / total_weight if total_weight > 0 else 0.5
    
    # Determine if the message passes the fairness check
    # Consider message fair if score is above threshold and no high-severity flags
    score_threshold = 0.7
    severity_threshold = 0.7
    
    has_high_severity_flags = any(flag.severity >= severity_threshold for flag in flags)
    is_fair = overall_score >= score_threshold and not has_high_severity_flags
    
    # Compile results
    result = {
        "score": overall_score,
        "metrics": metrics,
        "flags": flags,
        "alternatives": alternatives,
        "is_fair": is_fair,
        "reason": "Passed fairness evaluation" if is_fair else "Failed fairness evaluation"
    }
    
    return result

def integrate_with_principle_engine(principle_engine: PrincipleEngine) -> None:
    """
    Integrate fairness evaluation capabilities with a PrincipleEngine instance.
    
    Args:
        principle_engine: The PrincipleEngine instance to enhance
    """
    # Add fairness principles to the engine if not already present
    fairness_principles = [
        {
            "id": "fair_treatment",
            "description": "Treat all users fairly and without bias or discrimination.",
            "weight": 0.8,
            "tags": ["fairness", "ethics"]
        },
        {
            "id": "consistent_rules",
            "description": "Apply rules and guidelines consistently across similar situations.",
            "weight": 0.7,
            "tags": ["fairness", "consistency"]
        },
        {
            "id": "inclusive_language",
            "description": "Use inclusive language that respects diversity of all kinds.",
            "weight": 0.8,
            "tags": ["fairness", "inclusivity", "communication"]
        },
        {
            "id": "perspective_diversity",
            "description": "Represent and consider diverse perspectives and approaches.",
            "weight": 0.7,
            "tags": ["fairness", "diversity", "inclusivity"]
        }
    ]
    
    # Get existing principle IDs to avoid duplicates
    existing_ids = {p["id"] for p in principle_engine.principles}
    
    # Add any missing fairness principles
    for principle in fairness_principles:
        if principle["id"] not in existing_ids:
            principle_engine.add_principle(principle)
    
    # Create a FairnessEvaluator instance linked to the principle engine
    fairness_evaluator = FairnessEvaluator(principle_engine)
    
    # Store the fairness evaluator in the principle engine for future use
    principle_engine.fairness_evaluator = fairness_evaluator
    
    # Add fairness evaluation as a capability to the principle engine
    principle_engine.evaluate_fairness = lambda message, context=None: evaluate_fairness(
        message, context, principle_engine, fairness_evaluator
    )
    
    logger.info("Fairness evaluation capabilities integrated with PrincipleEngine")

def compare_fairness_across_interactions(
    interactions: List[Dict[str, Any]],
    principle_engine: Optional[PrincipleEngine] = None,
    fairness_evaluator: Optional[FairnessEvaluator] = None
) -> Dict[str, Any]:
    """
    Compare fairness across multiple interactions to identify patterns of bias or inconsistency.
    
    Args:
        interactions: List of message or action interactions to evaluate
        principle_engine: Optional PrincipleEngine instance
        fairness_evaluator: Optional FairnessEvaluator instance
        
    Returns:
        Dictionary containing comparison results:
        - 'overall_score': Average fairness score across all interactions
        - 'consistency_score': Score for consistency in rule application
        - 'bias_patterns': Identified patterns of recurring bias
        - 'recommendations': Suggested improvements for fairness
    """
    # Create or use provided FairnessEvaluator
    evaluator = fairness_evaluator
    if evaluator is None:
        evaluator = FairnessEvaluator(principle_engine)
    
    # Evaluate each interaction
    results = []
    for interaction in interactions:
        result = evaluate_fairness(interaction, None, principle_engine, evaluator)
        results.append(result)
    
    # Calculate overall statistics
    overall_scores = [result["score"] for result in results]
    overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    
    # Identify recurring bias patterns
    bias_patterns = {}
    
    # Track types of fairness flags across interactions
    for result in results:
        for flag in result["flags"]:
            flag_type = flag.type
            if flag_type in bias_patterns:
                bias_patterns[flag_type]["count"] += 1
                bias_patterns[flag_type]["severity"].append(flag.severity)
                bias_patterns[flag_type]["affected_groups"].update(flag.affected_groups)
            else:
                bias_patterns[flag_type] = {
                    "count": 1,
                    "severity": [flag.severity],
                    "affected_groups": set(flag.affected_groups),
                    "description": flag.description
                }
    
    # Calculate average severity for each bias type
    for bias_type in bias_patterns:
        severities = bias_patterns[bias_type]["severity"]
        avg_severity = sum(severities) / len(severities)
        bias_patterns[bias_type]["avg_severity"] = avg_severity
        bias_patterns[bias_type]["affected_groups"] = list(bias_patterns[bias_type]["affected_groups"])
    
    # Sort bias patterns by frequency and severity
    sorted_patterns = sorted(
        bias_patterns.items(),
        key=lambda x: (x[1]["count"], x[1]["avg_severity"]),
        reverse=True
    )
    
    # Generate recommendations based on patterns
    recommendations = []
    
    if sorted_patterns:
        # Address most common bias types
        top_bias_type, top_bias_info = sorted_patterns[0]
        recommendations.append({
            "focus_area": top_bias_type,
            "description": f"Address {top_bias_type} which appears in {top_bias_info['count']} interactions",
            "severity": top_bias_info["avg_severity"],
            "affected_groups": top_bias_info["affected_groups"]
        })
        
        # Check for consistency issues
        if "decision_consistency" in bias_patterns or "rule_consistency" in bias_patterns:
            recommendations.append({
                "focus_area": "consistency",
                "description": "Improve consistency in rule application across similar situations",
                "severity": bias_patterns.get("decision_consistency", {"avg_severity": 0.5})["avg_severity"]
            })
        
        # Check for language bias
        if "language_bias" in bias_patterns:
            recommendations.append({
                "focus_area": "language",
                "description": "Adopt more inclusive language patterns",
                "severity": bias_patterns["language_bias"]["avg_severity"],
                "affected_groups": bias_patterns["language_bias"]["affected_groups"]
            })
    
    # Calculate consistency score based on decision and rule consistency
    consistency_flags = sum(1 for result in results for flag in result["flags"] 
                          if flag.type in ["decision_consistency", "rule_consistency"])
    consistency_score = 1.0 - (consistency_flags / len(interactions) if interactions else 0.0)
    consistency_score = max(0.0, min(1.0, consistency_score))  # Ensure score is between 0 and 1
    
    # Compile comparison results
    comparison_result = {
        "overall_score": overall_score,
        "consistency_score": consistency_score,
        "bias_patterns": dict(sorted_patterns),
        "recommendations": recommendations,
        "interaction_count": len(interactions),
        "interactions_with_issues": sum(1 for result in results if not result["is_fair"])
    }
    
    return comparison_result