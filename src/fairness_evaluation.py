"""
Fairness Evaluation Module for the Principle Engine

This module provides the evaluate_fairness function that extends the PrincipleEngine's
capabilities to ensure fair and unbiased agent interactions by:
- Ensuring consistent application of rules across interactions
- Checking for bias in proposed actions or messages
- Comparing current actions against historical patterns
- Generating a fairness score and flagging potentially biased actions
- Suggesting alternative unbiased approaches when needed
"""

import logging
import copy
import json
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timezone
import hashlib
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FairnessEvaluation")

def evaluate_fairness(
    self, 
    action_data: Dict[str, Any], 
    historical_actions: List[Dict[str, Any]], 
    agent_id: str
) -> Dict[str, Any]:
    """
    Evaluate the fairness of a proposed action in context of historical actions.
    
    This function extends the PrincipleEngine to evaluate actions against the
    'Fairness as Truth' principle, ensuring actions are free from bias
    and consistent with fair treatment of all involved parties.
    
    Args:
        action_data: The proposed action to evaluate (e.g., resource allocation, message, decision)
        historical_actions: Past actions for comparison
        agent_id: The ID of the agent proposing the action
        
    Returns:
        A dictionary containing:
        - fairness_score: Float from 0.0 to 1.0, where 1.0 is perfectly fair
        - bias_flags: List of identified potential biases with rationales
        - alternative_suggestions: Optional list of modified actions that would be more fair
        - evaluation_details: Detailed breakdown of the evaluation
    """
    logger.info(f"Evaluating fairness of action from agent {agent_id}")
    
    # Initialize result structure
    evaluation_result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_id": agent_id, 
        "fairness_score": 0.95,  # Start with a high baseline score
        "bias_flags": [],
        "alternative_suggestions": [],
        "evaluation_details": {}
    }
    
    # Determine action type for specific analysis
    action_type = _get_action_type(action_data)
    evaluation_result["action_type"] = action_type
    
    # Generate action fingerprint for comparison
    action_fingerprint = _create_action_fingerprint(action_data)
    evaluation_result["evaluation_details"]["action_fingerprint"] = action_fingerprint
    
    # 1. Rule consistency check - ensures consistent application of rules across interactions
    rule_consistency_score, rule_findings = _check_rule_consistency(action_data, historical_actions, action_type)
    
    # Apply rule consistency score with proper weighting (0.25)
    # This ensures a single rule inconsistency doesn't drop the score too drastically
    evaluation_result["fairness_score"] = evaluation_result["fairness_score"] * 0.75 + rule_consistency_score * 0.25
    evaluation_result["evaluation_details"]["rule_consistency"] = {
        "score": rule_consistency_score,
        "findings": rule_findings
    }
    
    # Add any bias flags from rule findings
    for finding in rule_findings:
        if finding["type"] == "inconsistency":
            bias_flag = {
                "type": "rule_inconsistency",
                "description": finding["description"],
                "severity": finding["severity"],
                "evidence": finding["evidence"]
            }
            evaluation_result["bias_flags"].append(bias_flag)
            
            # Add alternative suggestion if available
            if "suggestion" in finding:
                # Create a copy with higher fairness score for the suggested alternative
                suggestion = finding["suggestion"].copy()
                suggestion["fairness_score"] = 0.90  # Ensure high score for alternative
                evaluation_result["alternative_suggestions"].append(suggestion)
    
    # 2. Check for attribute-based bias (e.g., treating different agent types differently)
    attribute_bias_score, attribute_findings = _check_attribute_bias(
        action_data, historical_actions, agent_id
    )
    evaluation_result["fairness_score"] *= attribute_bias_score
    evaluation_result["evaluation_details"]["attribute_bias"] = {
        "score": attribute_bias_score,
        "findings": attribute_findings
    }
    
    # Add attribute bias flags
    for finding in attribute_findings:
        bias_flag = {
            "type": "attribute_bias",
            "description": finding["description"],
            "severity": finding["severity"],
            "evidence": finding["evidence"]
        }
        evaluation_result["bias_flags"].append(bias_flag)
        
        # Add alternative suggestion if available
        if "suggested_fix" in finding:
            # Create a copy with higher fairness score for the suggested alternative
            suggestion = finding["suggested_fix"].copy()
            suggestion["fairness_score"] = 0.90  # Ensure high score for alternative
            evaluation_result["alternative_suggestions"].append(suggestion)
    
    # 3. Similarity comparison with historical actions
    similarity_score, similar_findings = _compare_with_similar_actions(
        action_data, historical_actions, action_fingerprint
    )
    evaluation_result["fairness_score"] *= similarity_score
    evaluation_result["evaluation_details"]["historical_similarity"] = {
        "score": similarity_score,
        "findings": similar_findings
    }
    
    # Add historical comparison flags
    for finding in similar_findings:
        bias_flag = {
            "type": "historical_inconsistency",
            "description": finding["description"],
            "severity": finding["severity"],
            "evidence": finding["evidence"]
        }
        evaluation_result["bias_flags"].append(bias_flag)
        
        # Add alternative suggestion if available
        if "suggestion" in finding:
            # Create a copy with higher fairness score for the suggested alternative
            suggestion = finding["suggestion"].copy()
            suggestion["fairness_score"] = 0.90  # Ensure high score for alternative
            evaluation_result["alternative_suggestions"].append(suggestion)
    
    # 4. Check for preferential treatment (e.g., special privileges, priority)
    preference_score, preference_findings = _check_for_preferential_treatment(
        action_data, historical_actions
    )
    evaluation_result["fairness_score"] *= preference_score
    evaluation_result["evaluation_details"]["preferential_treatment"] = {
        "score": preference_score,
        "findings": preference_findings
    }
    
    # Add preference bias flags
    for finding in preference_findings:
        bias_flag = {
            "type": "preferential_treatment",
            "description": finding["description"],
            "severity": finding["severity"],
            "evidence": finding["evidence"]
        }
        evaluation_result["bias_flags"].append(bias_flag)
        
        # Add alternative suggestion if available
        if "suggested_fix" in finding:
            # Create a copy with higher fairness score for the suggested alternative
            suggestion = finding["suggested_fix"].copy()
            suggestion["fairness_score"] = 0.90  # Ensure high score for alternative
            evaluation_result["alternative_suggestions"].append(suggestion)
    
    # Ensure fairness score is in valid range
    evaluation_result["fairness_score"] = max(0.0, min(1.0, evaluation_result["fairness_score"]))
    
    # Add summary of fairness evaluation
    evaluation_result["summary"] = _generate_fairness_summary(
        evaluation_result["fairness_score"],
        evaluation_result["bias_flags"]
    )
    
    # Log for learning and review if significant issues detected
    if evaluation_result["fairness_score"] < 0.8 and evaluation_result["bias_flags"]:
        logger.warning(f"Fairness issues detected in action from agent {agent_id}")
        for bias in evaluation_result["bias_flags"]:
            logger.warning(f"Bias: {bias['type']} - {bias['description']} ({bias['severity']})")
    
    return evaluation_result

def _get_action_type(action_data: Dict[str, Any]) -> str:
    """
    Determine the type of action being evaluated.
    
    Args:
        action_data: The action data to analyze
            
    Returns:
        String representing the action type
    """
    # Try common action type fields
    for field in ["method", "action", "type", "operation"]:
        if field in action_data:
            return action_data[field]
    
    # Infer from content
    if "resource" in action_data:
        if "allocation" in action_data:
            return "resource_allocation"
        elif "access" in action_data:
            return "resource_access"
    
    if "message" in action_data:
        return "communication"
    
    if "decision" in action_data:
        return "decision_making"
    
    return "unknown_action"

def _create_action_fingerprint(action_data: Dict[str, Any]) -> str:
    """
    Create a fingerprint of the essential characteristics of an action.
    
    This allows for comparing similar actions without being affected by
    incidental details like timestamps or unique IDs.
    
    Args:
        action_data: The action data to fingerprint
            
    Returns:
        A string fingerprint of the action's core characteristics
    """
    # Create a copy with normalized structure
    normalized = {}
    
    # Preserve core action elements for fingerprinting
    action_type = _get_action_type(action_data)
    normalized["type"] = action_type
    
    # Extract core elements based on action type
    if action_type == "resource_allocation":
        if "resource" in action_data:
            normalized["resource"] = action_data["resource"]
        if "amount" in action_data:
            normalized["amount"] = action_data["amount"]
        if "recipient" in action_data:
            normalized["recipient"] = action_data["recipient"]
    elif action_type == "communication":
        if "recipient_id" in action_data:
            normalized["recipient_id"] = action_data["recipient_id"]
        if "message_type" in action_data:
            normalized["message_type"] = action_data["message_type"]
    elif action_type == "decision_making":
        if "decision_type" in action_data:
            normalized["decision_type"] = action_data["decision_type"]
        if "affected_parties" in action_data:
            normalized["affected_parties"] = action_data["affected_parties"]
            
    # Generate a hash of the normalized content
    try:
        normalized_json = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(normalized_json.encode()).hexdigest()
    except:
        # Fallback if JSON serialization fails
        return hashlib.sha256(str(normalized).encode()).hexdigest()

def _check_rule_consistency(
    action_data: Dict[str, Any], 
    historical_actions: List[Dict[str, Any]], 
    action_type: str
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Check if the action consistently applies defined rules or policies.
    
    Args:
        action_data: The proposed action
        historical_actions: Past actions for comparison
        action_type: Type of the action
            
    Returns:
        Tuple of (rule_consistency_score, findings)
    """
    # Start with perfect score
    score = 1.0
    findings = []
    
    # Extract patterns from historical actions
    patterns = _extract_patterns_from_history(historical_actions, action_type)
    
    # Check each pattern for consistency
    for pattern in patterns:
        # Verify if this pattern applies to the current action
        if _pattern_applies(pattern, action_data):
            # Check if action follows the pattern
            follows, evidence = _check_pattern_compliance(pattern, action_data)
            
            if not follows:
                # Reduce score based on pattern confidence/importance
                score *= 0.8
                
                # Add finding
                findings.append({
                    "type": "inconsistency",
                    "description": f"Action inconsistent with historical pattern: {pattern['description']}",
                    "severity": "medium",
                    "evidence": evidence,
                    "suggestion": {
                        "action": _generate_consistent_action(action_data, pattern),
                        "description": f"Modified to follow established pattern for {pattern['field']}"
                    }
                })
    
    return score, findings

def _extract_patterns_from_history(
    historical_actions: List[Dict[str, Any]], 
    action_type: str
) -> List[Dict[str, Any]]:
    """
    Extract consistent patterns from historical actions.
    
    Args:
        historical_actions: The historical actions to analyze
        action_type: Type of actions to focus on
            
    Returns:
        List of extracted patterns/rules
    """
    patterns = []
    
    # Only consider actions of the same type
    relevant_actions = [a for a in historical_actions if _get_action_type(a) == action_type]
    
    if len(relevant_actions) < 3:  # Need enough actions to establish patterns
        return patterns
        
    # Analyze common fields and their values
    field_values = {}
    
    for action in relevant_actions:
        # Extract key fields and values
        for field, value in _extract_key_fields(action).items():
            if field not in field_values:
                field_values[field] = {}
                
            str_value = str(value)
            if str_value not in field_values[field]:
                field_values[field][str_value] = 0
            
            field_values[field][str_value] += 1
            
    # Convert to patterns where there's strong consistency
    for field, values in field_values.items():
        for value, count in values.items():
            if count >= max(3, len(relevant_actions) * 0.7):  # At least 70% consistent
                patterns.append({
                    "field": field,
                    "expected_value": value,
                    "confidence": count / len(relevant_actions),
                    "description": f"For {action_type} actions, {field} typically has value {value}"
                })
    
    return patterns

def _extract_key_fields(action: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key fields from an action for pattern analysis.
    
    Args:
        action: The action to analyze
        
    Returns:
        Dictionary of key fields and their values
    """
    result = {}
    
    # Helper to recursively process nested dictionaries
    def process_dict(d, prefix="") -> None:
        for key, value in d.items():
            if key in ["id", "timestamp", "created_at", "updated_at"]:
                continue  # Skip non-pattern fields
                
            field_path = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, (str, int, float, bool)) or value is None:
                result[field_path] = value
            elif isinstance(value, dict):
                process_dict(value, field_path)
    
    process_dict(action)
    return result

def _pattern_applies(pattern: Dict[str, Any], action: Dict[str, Any]) -> bool:
    """
    Check if a pattern is applicable to the current action.
    
    Args:
        pattern: The pattern to check
        action: The action to evaluate
        
    Returns:
        True if the pattern applies to this action
    """
    # Extract the field path
    field_path = pattern["field"].split(".")
    
    # Navigate to check if field exists
    target = action
    for part in field_path:
        if isinstance(target, dict) and part in target:
            target = target[part]
        else:
            return False
            
    return True

def _check_pattern_compliance(
    pattern: Dict[str, Any], 
    action: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if an action complies with a pattern.
    
    Args:
        pattern: The pattern to check compliance with
        action: The action to evaluate
        
    Returns:
        Tuple of (complies, details)
    """
    field_path = pattern["field"].split(".")
    expected_value = pattern["expected_value"]
    
    # Navigate to get actual value
    target = action
    for part in field_path:
        if part in target:
            target = target[part]
        else:
            return False, {"missing_field": True, "field": pattern["field"]}
    
    # Convert to string for comparison
    actual_value = str(target)
    follows = actual_value == expected_value
    
    return follows, {
        "expected": expected_value,
        "actual": actual_value,
        "field": pattern["field"]
    }

def _generate_consistent_action(action: Dict[str, Any], pattern: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a modified action that follows the pattern.
    
    Args:
        action: The original action
        pattern: The pattern to comply with
        
    Returns:
        Modified action that complies with the pattern
    """
    # Create a deep copy to avoid modifying the original
    consistent_action = copy.deepcopy(action)
    
    # Extract field path
    field_path = pattern["field"].split(".")
    
    # Navigate to target field, creating path if needed
    target = consistent_action
    for i, part in enumerate(field_path[:-1]):
        if part not in target:
            target[part] = {}
        target = target[part]
    
    # Convert expected value to appropriate type
    expected_value = pattern["expected_value"]
    current_value = target.get(field_path[-1])
    
    if current_value is not None:
        # Try to maintain the original type
        if isinstance(current_value, int):
            try:
                expected_value = int(expected_value)
            except (ValueError, TypeError):
                pass
        elif isinstance(current_value, float):
            try:
                expected_value = float(expected_value)
            except (ValueError, TypeError):
                pass
        elif isinstance(current_value, bool):
            expected_value = expected_value.lower() in ["true", "yes", "1"]
    
    # Set the expected value
    target[field_path[-1]] = expected_value
    
    return consistent_action

def _check_attribute_bias(
    action_data: Dict[str, Any], 
    historical_actions: List[Dict[str, Any]], 
    agent_id: str
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Check for bias based on attributes of involved parties.
    
    Args:
        action_data: The action to evaluate
        historical_actions: Past actions for comparison
        agent_id: ID of agent proposing the action
        
    Returns:
        Tuple of (bias_score, findings)
    """
    score = 1.0
    findings = []
    
    # Extract attributes of parties involved in the action
    action_attributes = _extract_party_attributes(action_data)
    
    if not action_attributes:
        return score, findings  # No attributes to check
    
    # Group historical actions by attributes
    attribute_groups = _group_actions_by_attributes(historical_actions)
    
    # Analyze each attribute for potential bias
    for attr_name, attr_value in action_attributes.items():
        if attr_name in attribute_groups:
            # Compare metrics between groups with different attribute values
            bias_detected, bias_details = _analyze_metrics_by_attribute(
                action_data,
                attribute_groups[attr_name],
                attr_name,
                attr_value
            )
            
            if bias_detected:
                # Reduce score based on severity
                severity_factor = 0.7 if bias_details["severity"] == "high" else 0.85
                score *= severity_factor
                
                # Add finding
                findings.append({
                    "type": "attribute_bias",
                    "description": f"Potential bias detected for attribute '{attr_name}'",
                    "severity": bias_details["severity"],
                    "evidence": bias_details["evidence"],
                    "suggested_fix": bias_details.get("fix")
                })
    
    return score, findings

def _extract_party_attributes(action_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract attributes of parties involved in the action.
    
    Args:
        action_data: The action to analyze
        
    Returns:
        Dict of attribute names and values
    """
    attributes = {}
    
    # Common attribute fields to look for
    common_fields = [
        "sender_id", "recipient_id", "agent_id", "user_id", "group_id",
        "organization_id", "team_id", "role", "access_level", "permission_level",
        "subscription_tier", "account_type"
    ]
    
    # Check top level
    for field in common_fields:
        if field in action_data:
            attributes[field] = action_data[field]
    
    # Check common nested locations
    for container in ["sender", "recipient", "metadata"]:
        if container in action_data and isinstance(action_data[container], dict):
            for field in common_fields:
                if field in action_data[container]:
                    attributes[f"{container}_{field}"] = action_data[container][field]
    
    return attributes

def _group_actions_by_attributes(actions: List[Dict[str, Any]]) -> Dict[str, Dict[Any, List[Dict[str, Any]]]]:
    """
    Group historical actions by attribute values.
    
    Args:
        actions: List of actions to analyze
        
    Returns:
        Nested dict of attribute -> value -> list of actions
    """
    groups = {}
    
    for action in actions:
        attributes = _extract_party_attributes(action)
        
        for attr_name, attr_value in attributes.items():
            if attr_name not in groups:
                groups[attr_name] = {}
            
            if attr_value not in groups[attr_name]:
                groups[attr_name][attr_value] = []
            
            groups[attr_name][attr_value].append(action)
    
    return groups

def _analyze_metrics_by_attribute(
    action: Dict[str, Any],
    attribute_groups: Dict[Any, List[Dict[str, Any]]],
    attr_name: str,
    attr_value: Any
) -> Tuple[bool, Dict[str, Any]]:
    """
    Analyze actions grouped by an attribute for potential bias.
    
    Args:
        action: Current action to evaluate
        attribute_groups: Actions grouped by attribute values
        attr_name: Name of the attribute being analyzed
        attr_value: Value of the attribute in the current action
        
    Returns:
        Tuple of (bias_detected, details)
    """
    # Extract metrics from current action
    action_metrics = _extract_numeric_metrics(action)
    
    if not action_metrics:
        return False, {"severity": "low", "evidence": {}}
    
    # Calculate average metrics for each attribute value group
    group_metrics = {}
    
    for value, actions in attribute_groups.items():
        if len(actions) < 3:  # Need enough data for comparison
            continue
        
        metrics_sum = {}
        metrics_count = {}
        
        for a in actions:
            a_metrics = _extract_numeric_metrics(a)
            for metric, value in a_metrics.items():
                if metric not in metrics_sum:
                    metrics_sum[metric] = 0
                    metrics_count[metric] = 0
                
                metrics_sum[metric] += value
                metrics_count[metric] += 1
        
        # Calculate averages
        group_metrics[value] = {
            metric: metrics_sum[metric] / metrics_count[metric]
            for metric in metrics_sum if metrics_count[metric] > 0
        }
    
    # Compare metrics across groups
    for metric, current_value in action_metrics.items():
        # Get average for current attribute group
        if attr_value in group_metrics and metric in group_metrics[attr_value]:
            current_group_avg = group_metrics[attr_value][metric]
            
            # Compare with other groups
            for other_value, other_metrics in group_metrics.items():
                if other_value == attr_value:
                    continue
                
                if metric in other_metrics:
                    other_group_avg = other_metrics[metric]
                    
                    # Calculate percentage difference
                    pct_diff = abs(current_value - other_group_avg) / max(0.1, other_group_avg)
                    
                    # If significant difference exists
                    if pct_diff > 0.25:  # 25% threshold
                        # Only flag if this action deviates from its own group average
                        deviation_from_own_group = abs(current_value - current_group_avg) / max(0.1, current_group_avg)
                        
                        if deviation_from_own_group > 0.15:  # 15% threshold
                            # Create bias details
                            severity = "high" if pct_diff > 0.5 else "medium"
                            
                            evidence = {
                                "metric": metric,
                                "current_value": current_value,
                                "current_group_avg": current_group_avg,
                                "other_group": other_value,
                                "other_group_avg": other_group_avg,
                                "percent_difference": pct_diff * 100
                            }
                            
                            # Create suggested fix
                            fix = None
                            if _is_adjustable_metric(metric, action):
                                fixed_action = copy.deepcopy(action)
                                _adjust_metric(fixed_action, metric, other_group_avg)
                                
                                fix = {
                                    "action": fixed_action,
                                    "description": f"Adjusted {metric} to be more in line with average for other groups"
                                }
                            
                            return True, {
                                "severity": severity,
                                "evidence": evidence,
                                "fix": fix
                            }
    
    return False, {"severity": "low", "evidence": {}}

def _extract_numeric_metrics(action: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract numeric metrics from an action for comparison.
    
    Args:
        action: Action to extract metrics from
        
    Returns:
        Dict of metric names and values
    """
    metrics = {}
    action_type = _get_action_type(action)
    
    # Type-specific metrics
    if action_type == "resource_allocation":
        if "amount" in action:
            try:
                metrics["allocation_amount"] = float(action["amount"])
            except (ValueError, TypeError):
                pass
        
        if "priority" in action:
            try:
                metrics["priority_level"] = float(action["priority"])
            except (ValueError, TypeError):
                pass
    
    # Generic numeric fields
    common_fields = ["score", "amount", "priority", "level", "limit", "quota", "rate", "time", "duration"]
    for field in common_fields:
        if field in action:
            try:
                metrics[field] = float(action[field])
            except (ValueError, TypeError):
                pass
    
    return metrics

def _is_adjustable_metric(metric: str, action: Dict[str, Any]) -> bool:
    """
    Check if a metric can be adjusted in an action.
    
    Args:
        metric: Metric name
        action: Action to check
        
    Returns:
        True if metric can be adjusted
    """
    adjustable_metrics = ["amount", "priority", "level", "score", "limit", "quota", "rate"]
    
    # Check if metric is in our adjustable list
    for adjustable in adjustable_metrics:
        if metric.endswith(adjustable) or metric == adjustable:
            # Check if the field exists directly in the action
            if adjustable in action:
                return True
            
            # For composite metrics, look for the base field
            if metric.endswith(adjustable) and metric != adjustable:
                base_field = metric[:-len(adjustable)-1]  # Remove "_{adjustable}"
                if base_field in action and isinstance(action[base_field], dict) and adjustable in action[base_field]:
                    return True
    
    return False

def _adjust_metric(action: Dict[str, Any], metric: str, target_value: float) -> None:
    """
    Adjust a metric in an action to the target value.
    
    Args:
        action: Action to modify
        metric: Metric to adjust
        target_value: Target value for the metric
    """
    # Simple case: direct field in action
    for field in ["amount", "priority", "level", "score", "limit", "quota", "rate"]:
        if metric == field and field in action:
            action[field] = target_value
            return
    
    # Composite metric case (e.g., allocation_amount, priority_level)
    for field in ["amount", "priority", "level", "score", "limit", "quota", "rate"]:
        if metric.endswith(field) and metric != field:
            prefix = metric[:-len(field)-1]  # Remove "_{field}"
            if prefix in action and isinstance(action[prefix], dict) and field in action[prefix]:
                action[prefix][field] = target_value
                return

def _compare_with_similar_actions(
    action_data: Dict[str, Any],
    historical_actions: List[Dict[str, Any]],
    action_fingerprint: str
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Compare action with similar historical actions for consistency.
    
    Args:
        action_data: Action to evaluate
        historical_actions: Historical actions for comparison
        action_fingerprint: Fingerprint of the action
        
    Returns:
        Tuple of (similarity_score, findings)
    """
    score = 1.0
    findings = []
    
    # Find similar actions
    similar_actions = []
    
    for hist_action in historical_actions:
        hist_fingerprint = _create_action_fingerprint(hist_action)
        
        # Actions with identical fingerprints are very similar
        if hist_fingerprint == action_fingerprint:
            similar_actions.append((hist_action, 1.0))  # Perfect similarity
        else:
            # Calculate similarity based on shared fields with same values
            similarity = _calculate_action_similarity(action_data, hist_action)
            if similarity > 0.7:  # At least 70% similar
                similar_actions.append((hist_action, similarity))
    
    # Sort by similarity (most similar first)
    similar_actions.sort(key=lambda x: x[1], reverse=True)
    
    # If no similar actions, we can't evaluate historical consistency
    if not similar_actions:
        return score, findings
    
    # Compare with most similar action(s)
    compared_count = 0
    
    for hist_action, similarity in similar_actions[:3]:  # Check up to 3 most similar actions
        # Find differences between current and historical action
        differences = _find_significant_differences(action_data, hist_action)
        
        for diff in differences:
            # Reduce score for each significant difference
            score *= 0.9
            
            # Add finding
            findings.append({
                "type": "historical_inconsistency",
                "description": f"Action differs from similar historical action in {diff['field']}",
                "severity": "medium",
                "evidence": diff,
                "suggestion": {
                    "action": _create_historically_consistent_action(action_data, diff, hist_action),
                    "description": f"Modified to match historical pattern for {diff['field']}"
                }
            })
            
        compared_count += 1
        if compared_count >= 3 or len(findings) >= 5:
            break  # Limit the number of comparisons/findings
    
    return score, findings

def _calculate_action_similarity(action1: Dict[str, Any], action2: Dict[str, Any]) -> float:
    """
    Calculate similarity between two actions based on shared fields.
    
    Args:
        action1: First action
        action2: Second action
        
    Returns:
        Similarity score from 0.0 to 1.0
    """
    fields1 = set(_extract_key_fields(action1).keys())
    fields2 = set(_extract_key_fields(action2).keys())
    
    common_fields = fields1.intersection(fields2)
    
    if not common_fields:
        return 0.0
    
    matching_fields = 0
    for field in common_fields:
        val1 = _get_field_value(action1, field)
        val2 = _get_field_value(action2, field)
        if val1 == val2:
            matching_fields += 1
    
    return matching_fields / len(common_fields)

def _get_field_value(action: Dict[str, Any], field: str) -> Any:
    """Get a field value from an action by path."""
    parts = field.split(".")
    target = action
    
    for part in parts:
        if isinstance(target, dict) and part in target:
            target = target[part]
        else:
            return None
    
    return target

def _find_significant_differences(
    action: Dict[str, Any], 
    hist_action: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Find significant differences between current and historical action.
    
    Args:
        action: Current action
        hist_action: Historical action for comparison
        
    Returns:
        List of difference descriptions
    """
    differences = []
    
    # Get fields from both actions
    action_fields = _extract_key_fields(action)
    hist_fields = _extract_key_fields(hist_action)
    
    # Check common fields for differences
    for field in set(action_fields.keys()).intersection(set(hist_fields.keys())):
        action_value = action_fields[field]
        hist_value = hist_fields[field]
        
        # Skip ID fields and timestamps
        if any(skip in field for skip in ["id", "timestamp", "created", "updated"]):
            continue
            
        # If values differ significantly
        if str(action_value) != str(hist_value):
            # For numeric values, check if the difference is substantial
            try:
                action_num = float(action_value)
                hist_num = float(hist_value)
                
                # Only flag if difference is more than 20%
                if abs(action_num - hist_num) / max(0.1, abs(hist_num)) > 0.2:
                    differences.append({
                        "field": field,
                        "current_value": action_value,
                        "historical_value": hist_value,
                        "percent_difference": round(abs(action_num - hist_num) / max(0.1, abs(hist_num)) * 100, 2)
                    })
            except (ValueError, TypeError):
                # For non-numeric values, any difference is significant
                differences.append({
                    "field": field,
                    "current_value": action_value,
                    "historical_value": hist_value
                })
    
    # Sort differences by significance
    differences.sort(key=lambda d: d.get("percent_difference", 100), reverse=True)
    
    return differences

def _create_historically_consistent_action(
    action: Dict[str, Any], 
    difference: Dict[str, Any],
    hist_action: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a version of the action that is consistent with historical patterns.
    
    Args:
        action: Original action
        difference: Detected difference
        hist_action: Historical action to be consistent with
        
    Returns:
        Modified action that matches historical patterns
    """
    # Create a deep copy to avoid modifying the original
    consistent_action = copy.deepcopy(action)
    
    # Extract field path and historical value
    field_path = difference["field"].split(".")
    historical_value = difference["historical_value"]
    
    # Navigate to target field, creating path if needed
    target = consistent_action
    for i, part in enumerate(field_path[:-1]):
        if part not in target:
            target[part] = {}
        target = target[part]
    
    # Set the historical value with appropriate type conversion
    current_value = target.get(field_path[-1])
    
    if current_value is not None:
        # Try to maintain the original type
        if isinstance(current_value, int):
            try:
                historical_value = int(historical_value)
            except (ValueError, TypeError):
                pass
        elif isinstance(current_value, float):
            try:
                historical_value = float(historical_value)
            except (ValueError, TypeError):
                pass
        elif isinstance(current_value, bool):
            if isinstance(historical_value, str):
                historical_value = historical_value.lower() in ["true", "yes", "1"]
    
    # Set the value
    target[field_path[-1]] = historical_value
    
    return consistent_action

def _check_for_preferential_treatment(
    action_data: Dict[str, Any], 
    historical_actions: List[Dict[str, Any]]
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Check for unjustified preferential treatment.
    
    Args:
        action_data: Action to evaluate
        historical_actions: Historical actions for comparison
        
    Returns:
        Tuple of (score, findings)
    """
    score = 1.0
    findings = []
    
    # Check for priority flags
    if "priority" in action_data:
        try:
            priority = float(action_data["priority"])
            if priority > 0:
                # Look for justification
                justified = False
                
                # Check if high priority is justified based on historical pattern
                if _is_high_priority_justified(action_data, historical_actions, priority):
                    justified = True
                
                if not justified:
                    score *= 0.7
                    findings.append({
                        "type": "unjustified_priority",
                        "description": "High priority set without clear justification",
                        "severity": "medium",
                        "evidence": {
                            "priority": priority,
                            "average_priority": _get_average_priority(historical_actions)
                        },
                        "suggested_fix": {
                            "action": _create_fair_priority_action(action_data),
                            "description": "Reduced priority to standard level"
                        }
                    })
        except (ValueError, TypeError):
            pass
    
    # Check for other preferential indicators
    preferential_indicators = [
        "expedited", "vip", "special", "preferred", "premium", "exclusive"
    ]
    
    for indicator in preferential_indicators:
        if _contains_preferential_indicator(action_data, indicator):
            score *= 0.8
            findings.append({
                "type": "preferential_indicator",
                "description": f"Contains preferential indicator '{indicator}'",
                "severity": "medium",
                "evidence": {
                    "indicator": indicator
                },
                "suggested_fix": {
                    "action": _remove_preferential_indicator(action_data, indicator),
                    "description": f"Removed preferential indicator '{indicator}'"
                }
            })
    
    return score, findings

def _is_high_priority_justified(
    action: Dict[str, Any], 
    historical_actions: List[Dict[str, Any]], 
    priority: float
) -> bool:
    """
    Check if high priority is justified based on historical patterns.
    
    Args:
        action: Current action
        historical_actions: Historical actions for comparison
        priority: Priority value to check
        
    Returns:
        True if priority is justified
    """
    # Check if similarly structured actions have similar priority
    similar_actions = []
    action_type = _get_action_type(action)
    
    for hist_action in historical_actions:
        if _get_action_type(hist_action) == action_type:
            # Check basic structural similarity
            similarity = _calculate_action_similarity(action, hist_action)
            if similarity > 0.7:  # At least 70% similar
                similar_actions.append(hist_action)
    
    if not similar_actions:
        return False
    
    # Check priorities of similar actions
    priorities = []
    for a in similar_actions:
        if "priority" in a:
            try:
                priorities.append(float(a["priority"]))
            except (ValueError, TypeError):
                pass
    
    if not priorities:
        return False
    
    # Calculate average and standard deviation
    avg_priority = sum(priorities) / len(priorities)
    
    # Check if current priority is in line with historical ones
    return abs(priority - avg_priority) <= 1.0  # Within +/- 1 of average

def _get_average_priority(actions: List[Dict[str, Any]]) -> float:
    """
    Get the average priority from historical actions.
    
    Args:
        actions: List of historical actions
        
    Returns:
        Average priority value
    """
    priorities = []
    
    for action in actions:
        if "priority" in action:
            try:
                priorities.append(float(action["priority"]))
            except (ValueError, TypeError):
                pass
    
    if not priorities:
        return 0.0
    
    return sum(priorities) / len(priorities)

def _contains_preferential_indicator(action: Dict[str, Any], indicator: str) -> bool:
    """
    Check if an action contains a preferential indicator.
    
    Args:
        action: Action to check
        indicator: Indicator to look for
        
    Returns:
        True if indicator is found
    """
    action_str = json.dumps(action).lower()
    return indicator.lower() in action_str

def _create_fair_priority_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a version of the action with standard priority.
    
    Args:
        action: Original action
        
    Returns:
        Modified action with standard priority
    """
    fair_action = copy.deepcopy(action)
    
    if "priority" in fair_action:
        fair_action["priority"] = 0
    
    return fair_action

def _remove_preferential_indicator(action: Dict[str, Any], indicator: str) -> Dict[str, Any]:
    """
    Remove a preferential indicator from an action.
    
    Args:
        action: Original action
        indicator: Indicator to remove
        
    Returns:
        Modified action without preferential indicator
    """
    fair_action = copy.deepcopy(action)
    
    # Helper to recursively process dictionaries
    def process_dict(d) -> None:
        for key, value in list(d.items()):
            if isinstance(key, str) and indicator.lower() in key.lower():
                # Remove the key entirely
                del d[key]
            elif isinstance(value, str) and indicator.lower() in value.lower():
                # Replace the value with a non-preferential version
                d[key] = value.lower().replace(indicator.lower(), "standard")
            elif isinstance(value, dict):
                process_dict(value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        process_dict(item)
                    elif isinstance(item, str) and indicator.lower() in item.lower():
                        value[i] = item.lower().replace(indicator.lower(), "standard")
    
    process_dict(fair_action)
    return fair_action

def _generate_fairness_summary(fairness_score: float, bias_flags: List[Dict[str, Any]]) -> str:
    """
    Generate a summary of the fairness evaluation.
    
    Args:
        fairness_score: Overall fairness score
        bias_flags: List of identified biases
        
    Returns:
        Summary string
    """
    if fairness_score >= 0.95:
        summary = "The action demonstrates excellent fairness with no significant bias detected."
    elif fairness_score >= 0.85:
        summary = "The action is generally fair with minor consistency issues."
    elif fairness_score >= 0.7:
        summary = "The action shows moderate fairness concerns that should be addressed."
    elif fairness_score >= 0.5:
        summary = "The action has significant fairness issues that require attention."
    else:
        summary = "The action demonstrates severe fairness problems and should be reconsidered."
    
    # Add details about specific issues
    if bias_flags:
        summary += "\n\nKey issues:"
        for i, flag in enumerate(bias_flags[:3], 1):  # Show top 3 issues
            summary += f"\n{i}. {flag['description']} ({flag['severity']} severity)"
        
        if len(bias_flags) > 3:
            summary += f"\n...and {len(bias_flags) - 3} more issue(s)"
    
    return summary
