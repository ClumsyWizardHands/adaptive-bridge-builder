"""
Fairness Evaluation Extension for the Principle Engine

This module extends the PrincipleEngine with a fairness evaluation function that
analyzes proposed actions for bias and fairness compared to historical actions.
"""

import logging
import copy
import json
from typing import Dict, List, Any, Tuple
from datetime import datetime

from principle_engine import PrincipleEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PrincipleEngineFairness")

def extend_principle_engine():
    """
    Extends the PrincipleEngine class with the evaluate_fairness method.
    Call this function to add the fairness evaluation capability to PrincipleEngine.
    """
    if not hasattr(PrincipleEngine, 'evaluate_fairness'):
        PrincipleEngine.evaluate_fairness = evaluate_fairness
        logger.info("PrincipleEngine successfully extended with fairness evaluation capabilities")

def evaluate_fairness(
    self, 
    action_data: Dict[str, Any], 
    historical_actions: List[Dict[str, Any]], 
    agent_id: str
) -> Dict[str, Any]:
    """
    Evaluate the fairness of a proposed action in context of historical actions.
    
    This method extends the PrincipleEngine to evaluate actions against the
    'Fairness as a Fundamental Truth' principle, ensuring actions are free from bias
    and consistent with fair treatment of all involved parties.
    
    Args:
        action_data: The proposed action to evaluate (e.g., resource allocation, decision, communication)
        historical_actions: Past actions for comparison from SessionManager or logs
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
        "timestamp": datetime.utcnow().isoformat(),
        "agent_id": agent_id, 
        "fairness_score": 1.0,  # Start with perfect score
        "bias_flags": [],
        "alternative_suggestions": [],
        "evaluation_details": {}
    }
    
    # Determine action type for specific analysis
    action_type = _get_action_type(action_data)
    evaluation_result["action_type"] = action_type
    
    # 1. Rule consistency check - ensures consistent application of rules or policies
    rule_consistency_score, rule_findings = _check_rule_consistency(action_data, historical_actions, action_type)
    evaluation_result["fairness_score"] *= rule_consistency_score
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
                evaluation_result["alternative_suggestions"].append(finding["suggestion"])
    
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
            evaluation_result["alternative_suggestions"].append(finding["suggested_fix"])
    
    # 3. Check for preferential treatment (e.g., special privileges, priority)
    preference_score, preference_findings = _check_for_preferential_treatment(action_data, historical_actions)
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
            evaluation_result["alternative_suggestions"].append(finding["suggested_fix"])
    
    # Ensure fairness score is in valid range
    evaluation_result["fairness_score"] = max(0.0, min(1.0, evaluation_result["fairness_score"]))
    
    # Log for learning and review if significant issues detected
    if evaluation_result["fairness_score"] < 0.8 and evaluation_result["bias_flags"]:
        logger.warning(f"Fairness issues detected in action from agent {agent_id}")
        for bias in evaluation_result["bias_flags"]:
            logger.warning(f"Bias: {bias['type']} - {bias['description']} ({bias['severity']})")
    
    return evaluation_result

# Helper functions for fairness evaluation
def _get_action_type(action_data: Dict[str, Any]) -> str:
    """Determine the type of action being evaluated."""
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

def _check_rule_consistency(
    action_data: Dict[str, Any], 
    historical_actions: List[Dict[str, Any]], 
    action_type: str
) -> Tuple[float, List[Dict[str, Any]]]:
    """Check if the action consistently applies rules or policies from historical actions."""
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
    """Extract consistent patterns from historical actions."""
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
    """Extract key fields from an action for pattern analysis."""
    result = {}
    
    # Helper to recursively process nested dictionaries
    def process_dict(d, prefix=""):
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
    """Check if a pattern is applicable to the current action."""
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
    """Check if an action complies with a pattern."""
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
    """Generate a modified action that follows the pattern."""
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
    """Check for bias based on attributes of involved parties."""
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
    """Extract attributes of parties involved in the action."""
    attributes = {}
    
    # Common attribute fields to look for
    common_fields = [
        "sender_id", "recipient_id", "agent_id", "user_id", "group_id",
        "organization_id", "team_id", "role", "access_level", "permission_level"
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
    """Group historical actions by attribute values."""
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
    """Analyze actions grouped by an attribute for potential bias."""
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
            for metric in metrics_sum
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
                                    "description": f"Adjusted {metric} to be more in line with average for {other_value}"
                                }
                            
                            return True, {
                                "severity": severity,
                                "evidence": evidence,
                                "fix": fix
                            }
    
    return False, {"severity": "low", "evidence": {}}

def _extract_numeric_metrics(action: Dict[str, Any]) -> Dict[str, float]:
    """Extract numeric metrics from an action for comparison."""
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
    common_fields = ["score", "amount", "priority", "level", "limit", "quota", "rate"]
    for field in common_fields:
        if field in action:
            try:
                metrics[field] = float(action[field])
            except (ValueError, TypeError):
                pass
    
    return metrics

def _is_adjustable_metric(metric: str, action: Dict[str, Any]) -> bool:
    """Check if a metric can be adjusted in an action."""
    adjustable_metrics = ["amount", "priority", "level", "score", "limit", "quota", "rate"]
    
    # Check if metric is in our adjustable list
    for adjustable in adjustable_metrics:
        if metric.endswith(adjustable) and adjustable in action:
            return True
    
    return False

def _adjust_metric(action: Dict[str, Any], metric: str, target_value: float) -> None:
    """Adjust a metric in an action to the target value."""
    # Extract the field name from the metric
    field = None
    for potential_field in ["amount", "priority", "level", "score", "limit", "quota", "rate"]:
        if metric.endswith(potential_field) and potential_field in action:
            field = potential_field
            break
    
    if field:
        action[field] = target_value

def _check_for_preferential_treatment(
    action_data: Dict[str, Any], 
    historical_actions: List[Dict[str, Any]]
) -> Tuple[float, List[Dict[str, Any]]]:
    """Check for unjustified preferential treatment."""
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
        "expedited", "vip", "special", "preferred", "premium"
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
    """Check if high priority is justified based on historical patterns."""
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
    """Get the average priority from historical actions."""
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

def _calculate_action_similarity(action1: Dict[str, Any], action2: Dict[str, Any]) -> float:
    """Calculate similarity between two actions based on shared fields."""
    fields1 = set(_extract_key_fields(action1).keys())
    fields2 = set(_extract_key_fields(action2).keys())
    
    common_fields = fields1.intersection(fields2)
    
    if not common_fields:
        return 0.0
    
    matching_fields = 0
    for field in common_fields:
        if _get_field_value(action1, field) == _get_field_value(action2, field):
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

def _contains_preferential_indicator(action: Dict[str, Any], indicator: str) -> bool:
    """Check if an action contains a preferential indicator."""
    action_str = json.dumps(action).lower()
    return indicator.lower() in action_str

def _create_fair_priority_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """Create a version of the action with standard priority."""
    fair_action = copy.deepcopy(action)
    
    if "priority" in fair_action:
        fair_action["priority"] = 0
    
    return fair_action

def _remove_preferential_indicator(action: Dict[str, Any], indicator: str) -> Dict[str, Any]:
    """Remove a preferential indicator from an action."""
    fair_action = copy.deepcopy(action)
    
    # Helper to recursively process dictionaries
    def process_dict(d):
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
