"""
Fairness Evaluation Extension for the Principle Engine

This module extends the PrincipleEngine with fairness evaluation capabilities.
It provides functionality to evaluate actions for fairness, detect bias, and suggest alternatives.
"""

import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import copy

from principle_engine import PrincipleEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PrincipleEngineFairness")

class PrincipleEngineFairness(PrincipleEngine):
    """Extended PrincipleEngine with fairness evaluation capabilities."""
    
    def __init__(self, principles_file: Optional[str] = None):
        """
        Initialize the PrincipleEngineFairness with the core principles.
        
        Args:
            principles_file: Optional path to a JSON file containing principle definitions.
                            If None, default principles will be used.
        """
        super().__init__(principles_file)
        self.fairness_evaluations = []
        
    def evaluate_fairness(
        self, 
        action_data: Dict[str, Any], 
        historical_actions: List[Dict[str, Any]], 
        agent_id: str
    ) -> Dict[str, Any]:
        """
        Evaluate the fairness of a proposed action in context of historical actions.
        
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
        
        # Store original timestamp for reference
        timestamp = datetime.utcnow().isoformat()
        
        # Extract action type for better analysis
        action_type = self._get_action_type(action_data)
        
        # Perform comprehensive fairness evaluation
        evaluation_details = {}
        fairness_score = 1.0  # Start with perfect score
        bias_flags = []
        alternative_suggestions = []
        
        # 1. Rule consistency check - ensures consistent application of defined rules or policies
        rule_consistency_score, rule_findings = self._check_rule_consistency(action_data, historical_actions, action_type)
        fairness_score *= rule_consistency_score
        evaluation_details["rule_consistency"] = {
            "score": rule_consistency_score,
            "findings": rule_findings
        }
        bias_flags.extend(self._format_bias_flags_from_findings(rule_findings, "rule_inconsistency"))
        
        # 2. Attribute bias check - checks for potential bias based on attributes of involved parties
        attribute_bias_score, attribute_findings = self._check_attribute_bias(action_data, historical_actions, agent_id)
        fairness_score *= attribute_bias_score
        evaluation_details["attribute_bias"] = {
            "score": attribute_bias_score,
            "findings": attribute_findings
        }
        bias_flags.extend(self._format_bias_flags_from_findings(attribute_findings, "attribute_bias"))
        
        # 3. Historical consistency check - compares with similar past situations for consistency
        history_score, historical_findings = self._check_historical_consistency(action_data, historical_actions)
        fairness_score *= history_score
        evaluation_details["historical_consistency"] = {
            "score": history_score,
            "findings": historical_findings
        }
        bias_flags.extend(self._format_bias_flags_from_findings(historical_findings, "historical_inconsistency"))
        
        # 4. Special treatment check - checks for unjustified preferential treatment
        special_treatment_score, special_findings = self._check_for_special_treatment(action_data, historical_actions)
        fairness_score *= special_treatment_score
        evaluation_details["special_treatment"] = {
            "score": special_treatment_score,
            "findings": special_findings
        }
        bias_flags.extend(self._format_bias_flags_from_findings(special_findings, "preferential_treatment"))
        
        # Generate alternative suggestions based on findings
        alternative_suggestions = self._generate_alternative_suggestions(action_data, bias_flags)
        
        # Ensure score is within valid range
        fairness_score = max(0.0, min(1.0, fairness_score))
        
        # Create the evaluation result
        evaluation_result = {
            "timestamp": timestamp,
            "agent_id": agent_id,
            "action_type": action_type,
            "fairness_score": fairness_score,
            "bias_flags": bias_flags,
            "alternative_suggestions": alternative_suggestions,
            "evaluation_details": evaluation_details
        }
        
        # Store evaluation for learning and review
        self.fairness_evaluations.append(evaluation_result)
        
        # Log to the learning system if significant fairness issues are detected
        if fairness_score < 0.8 and len(bias_flags) > 0:
            self._log_fairness_evaluation(evaluation_result)
            
        return evaluation_result
    
    def _get_action_type(self, action_data: Dict[str, Any]) -> str:
        """
        Determine the type of action being evaluated.
        
        Args:
            action_data: The action data to analyze
            
        Returns:
            String representing the action type
        """
        # Try to identify action type from various possible fields
        if "method" in action_data:
            return action_data["method"]
        elif "action" in action_data:
            return action_data["action"]
        elif "type" in action_data:
            return action_data["type"]
        elif "operation" in action_data:
            return action_data["operation"]
        
        # Look at top-level keys to infer action type
        if "resource" in action_data:
            if "allocation" in action_data:
                return "resource_allocation"
            elif "access" in action_data:
                return "resource_access"
        
        if "message" in action_data:
            return "communication"
        
        if "decision" in action_data:
            return "decision_making"
        
        # Default type if we can't determine
        return "unknown_action"
    
    def _check_rule_consistency(
        self, 
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
        
        # Extract rules from historical actions
        rules = self._extract_rules_from_history(historical_actions, action_type)
        
        # Process each rule
        for rule in rules:
            # Check if rule applies to current action
            if self._rule_applies_to_action(rule, action_data):
                # Check if action follows the rule
                complies, compliance_details = self._check_rule_compliance(rule, action_data)
                
                if not complies:
                    # Reduce score based on rule importance
                    score *= (0.7 + (0.3 * rule.get("confidence", 0.5)))
                    
                    # Add finding
                    findings.append({
                        "type": "rule_violation",
                        "description": f"Action violates established rule: {rule['description']}",
                        "severity": "high" if rule.get("importance", "medium") == "high" else "medium",
                        "evidence": compliance_details,
                        "suggested_fix": self._generate_rule_compliant_version(action_data, rule)
                    })
        
        return score, findings
    
    def _extract_rules_from_history(
        self, 
        historical_actions: List[Dict[str, Any]], 
        action_type: str
    ) -> List[Dict[str, Any]]:
        """
        Extract consistent rules/patterns from historical actions.
        
        Args:
            historical_actions: The historical actions to analyze
            action_type: Type of actions to focus on
            
        Returns:
            List of extracted rules
        """
        # Filter for relevant actions
        relevant_actions = [
            action for action in historical_actions 
            if self._get_action_type(action) == action_type
        ]
        
        if not relevant_actions:
            return []
        
        # Analyze patterns in key fields across actions
        field_patterns = {}
        
        for action in relevant_actions:
            self._analyze_action_fields(action, field_patterns)
        
        # Convert strong patterns to rules
        rules = []
        threshold = max(3, len(relevant_actions) * 0.7)  # At least 70% of actions or 3 instances
        
        for field, values in field_patterns.items():
            for value, count in values.items():
                if count >= threshold:
                    rules.append({
                        "field": field,
                        "expected_value": value,
                        "confidence": count / len(relevant_actions),
                        "description": f"For {action_type} actions, {field} should be {value}",
                        "importance": "high" if count == len(relevant_actions) else "medium"
                    })
        
        return rules
    
    def _analyze_action_fields(self, action: Dict[str, Any], patterns: Dict[str, Dict[str, int]]) -> None:
        """
        Analyze fields in an action to extract patterns.
        
        Args:
            action: The action to analyze
            patterns: Dictionary to populate with patterns
        """
        def process_dict(d, prefix=""):
            for key, value in d.items():
                field_path = f"{prefix}.{key}" if prefix else key
                
                # Skip ID fields, timestamps, etc.
                if key in ["id", "timestamp", "created_at", "updated_at"]:
                    continue
                
                # Process based on value type
                if isinstance(value, (str, int, float, bool)) or value is None:
                    # For primitive values, track the field-value pattern
                    if field_path not in patterns:
                        patterns[field_path] = {}
                    
                    # Convert value to string for consistent comparison
                    str_value = str(value)
                    patterns[field_path][str_value] = patterns[field_path].get(str_value, 0) + 1
                
                elif isinstance(value, dict):
                    # Recursively process nested dictionaries
                    process_dict(value, field_path)
                
                elif isinstance(value, list) and all(isinstance(x, (str, int, float, bool)) for x in value):
                    # For simple lists, consider the sorted list as a whole
                    if field_path not in patterns:
                        patterns[field_path] = {}
                    
                    # Sort and stringify list for consistent comparison
                    sorted_value = str(sorted([str(x) for x in value]))
                    patterns[field_path][sorted_value] = patterns[field_path].get(sorted_value, 0) + 1
        
        # Start processing from the root
        process_dict(action)
    
    def _rule_applies_to_action(self, rule: Dict[str, Any], action: Dict[str, Any]) -> bool:
        """
        Check if a rule applies to an action.
        
        Args:
            rule: Rule to check
            action: Action to evaluate
            
        Returns:
            True if rule applies to action
        """
        # Extract the field path
        field_path = rule["field"].split(".")
        
        # Navigate to check if field exists in action
        target = action
        for part in field_path:
            if isinstance(target, dict) and part in target:
                target = target[part]
            else:
                return False
        
        return True
    
    def _check_rule_compliance(self, rule: Dict[str, Any], action: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if an action complies with a rule.
        
        Args:
            rule: Rule to check compliance with
            action: Action to evaluate
            
        Returns:
            Tuple of (complies, details)
        """
        # Extract field path and expected value
        field_path = rule["field"].split(".")
        expected_value = rule["expected_value"]
        
        # Navigate to get actual value
        target = action
        for part in field_path:
            if part in target:
                target = target[part]
            else:
                return False, {"missing_field": True, "field": rule["field"]}
        
        # Convert to string for comparison
        actual_value = str(target)
        complies = actual_value == expected_value
        
        return complies, {
            "expected": expected_value,
            "actual": actual_value,
            "field": rule["field"]
        }
    
    def _generate_rule_compliant_version(self, action: Dict[str, Any], rule: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a version of the action that complies with the rule.
        
        Args:
            action: Original action
            rule: Rule to comply with
            
        Returns:
            Modified compliant action
        """
        # Create a deep copy to avoid modifying the original
        compliant_action = copy.deepcopy(action)
        
        # Extract field path
        field_path = rule["field"].split(".")
        
        # Navigate to target field, creating path if needed
        target = compliant_action
        for i, part in enumerate(field_path[:-1]):
            if part not in target:
                target[part] = {}
            target = target[part]
        
        # Set the expected value
        target[field_path[-1]] = self._convert_to_appropriate_type(
            rule["expected_value"], 
            action.get(field_path[0], {})
        )
        
        return compliant_action
    
    def _convert_to_appropriate_type(self, value_str: str, original_value: Any) -> Any:
        """
        Convert a string value to the appropriate type based on original value.
        
        Args:
            value_str: String value to convert
            original_value: Original value to determine type
            
        Returns:
            Converted value
        """
        # Handle conversion based on original type
        if isinstance(original_value, int):
            try:
                return int(value_str)
            except ValueError:
                pass
        elif isinstance(original_value, float):
            try:
                return float(value_str)
            except ValueError:
                pass
        elif isinstance(original_value, bool):
            return value_str.lower() in ["true", "yes", "1"]
        
        # Default: return as string
        return value_str
    
    def _check_attribute_bias(
        self, 
        action_data: Dict[str, Any], 
        historical_actions: List[Dict[str, Any]], 
        agent_id: str
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Check for potential bias based on attributes of involved parties.
        
        Args:
            action_data: The action to evaluate
            historical_actions: Past actions for comparison
            agent_id: ID of agent proposing the action
            
        Returns:
            Tuple of (bias_score, findings)
        """
        score = 1.0
        findings = []
        
        # Extract attributes involved in the action
        attributes = self._extract_relevant_attributes(action_data)
        if not attributes:
            return score, findings  # No attributes to check bias for
        
        # Group historical actions by attributes
        attribute_actions = self._group_actions_by_attributes(historical_actions)
        
        # Analyze each attribute for potential bias
        for attr_name, attr_value in attributes.items():
            if attr_name in attribute_actions:
                # Check if this attribute shows consistent patterns across different values
                bias_detected, bias_details = self._analyze_attribute_bias(
                    attr_name, attr_value, attribute_actions[attr_name], action_data
                )
                
                if bias_detected:
                    # Adjust score based on severity of bias
                    severity_factor = 0.7 if bias_details["severity"] == "high" else 0.85
                    score *= severity_factor
                    
                    # Add finding
                    findings.append({
                        "type": "attribute_bias",
                        "description": f"Potential bias detected based on {attr_name}",
                        "severity": bias_details["severity"],
                        "evidence": bias_details["evidence"],
                        "suggested_fix": bias_details.get("suggested_fix")
                    })
        
        return score, findings
    
    def _extract_relevant_attributes(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract attributes of parties involved in the action.
        
        Args:
            action_data: The action to analyze
            
        Returns:
            Dict of attributes
        """
        attributes = {}
        
        # Common attribute fields to look for
        common_fields = [
            "sender_id", "recipient_id", "agent_id", "user_id", "group_id",
            "organization_id", "team_id", "role", "access_level", "permission_level",
            "subscription_tier", "account_type"
        ]
        
        # Check in top level
        for field in common_fields:
            if field in action_data:
                attributes[field] = action_data[field]
        
        # Check in common nested locations
        if "sender" in action_data and isinstance(action_data["sender"], dict):
            for field in common_fields:
                if field in action_data["sender"]:
                    attributes[f"sender_{field}"] = action_data["sender"][field]
        
        if "recipient" in action_data and isinstance(action_data["recipient"], dict):
            for field in common_fields:
                if field in action_data["recipient"]:
                    attributes[f"recipient_{field}"] = action_data["recipient"][field]
        
        if "metadata" in action_data and isinstance(action_data["metadata"], dict):
            for field in common_fields:
                if field in action_data["metadata"]:
                    attributes[f"metadata_{field}"] = action_data["metadata"][field]
        
        return attributes
    
    def _group_actions_by_attributes(self, actions: List[Dict[str, Any]]) -> Dict[str, Dict[Any, List[Dict[str, Any]]]]:
        """
        Group actions by attribute values.
        
        Args:
            actions: List of actions to analyze
            
        Returns:
            Nested dict of attribute -> value -> list of actions
        """
        grouped = {}
        
        for action in actions:
            # Extract attributes
            attrs = self._extract_relevant_attributes(action)
            
            # Group by each attribute
            for attr_name, attr_value in attrs.items():
                if attr_name not in grouped:
                    grouped[attr_name] = {}
                
                if attr_value not in grouped[attr_name]:
                    grouped[attr_name][attr_value] = []
                
                grouped[attr_name][attr_value].append(action)
        
        return grouped
    
    def _analyze_attribute_bias(
        self, 
        attr_name: str, 
        attr_value: Any, 
        attr_actions: Dict[Any, List[Dict[str, Any]]], 
        current_action: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze actions grouped by attributes for bias.
        
        Args:
            attr_name: Name of attribute to analyze
            attr_value: Value of attribute in current action
            attr_actions: Actions grouped by attribute values
            current_action: Current action being evaluated
            
        Returns:
            Tuple of (bias_detected, details)
        """
        bias_detected = False
        bias_details = {
            "severity": "low",
            "evidence": {}
        }
        
        # Extract metrics for the current action
        metrics = self._extract_action_metrics(current_action)
        if not metrics:
            return False, bias_details
        
        # Calculate averages for each attribute value
        value_metrics = {}
        
        for val, actions in attr_actions.items():
            if len(actions) < 3:  # Need enough data for meaningful comparison
                continue
                
            # Collect metric values across actions with this attribute value
            val_metrics = {}
            
            for action in actions:
                action_metrics = self._extract_action_metrics(action)
                for metric, value in action_metrics.items():
                    if metric not in val_metrics:
                        val_metrics[metric] = []
                    val_metrics[metric].append(value)
            
            # Calculate averages
            avgs = {}
            for metric, values in val_metrics.items():
                avgs[metric] = sum(values) / len(values)
            
            value_metrics[val] = avgs
        
        # Compare metrics across attribute values
        if attr_value in value_metrics and len(value_metrics) > 1:
            current_group_metrics = value_metrics[attr_value]
            
            # Compare with other groups
            for other_val, other_metrics in value_metrics.items():
                if other_val == attr_value:
                    continue
                
                # Compare each metric
                for metric, curr_avg in current_group_metrics.items():
                    if metric in other_metrics:
                        other_avg = other_metrics[metric]
                        current_value = metrics.get(metric)
                        
                        if current_value is not None:
                            # Calculate percent difference
                            pct_diff = abs(current_value - other_avg) / max(1, other_avg)
                            
                            # If significant difference exists
                            if pct_diff > 0.25:  # 25% difference threshold
                                bias_detected = True
                                bias_details["severity"] = "high" if pct_diff > 0.5 else "medium"
                                
                                # Add evidence
                                evidence = {
                                    "metric": metric,
                                    "current_value": current_value,
                                    "current_group_avg": curr_avg,
                                    "other_group": other_val,
                                    "other_group_avg": other_avg,
                                    "percent_difference": pct_diff * 100
                                }
                                
                                bias_details["evidence"] = evidence
                                
                                # Suggest fix if applicable
                                if self._is_adjustable_metric(metric, current_action):
                                    adjusted_action = copy.deepcopy(current_action)
                                    self._adjust_metric_for_fairness(adjusted_action, metric, other_avg)
                                    bias_details["suggested_fix"] = {
                                        "action": adjusted_action,
                                        "description": f"Adjusted {metric} to be more in line with historical average"
                                    }
                                
                                # Once we find significant bias, we can return
                                return bias_detected, bias_details
        
        return bias_detected, bias_details
    
    def _extract_action_metrics(self, action: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract numeric metrics from an action for comparison.
        
        Args:
            action: Action to extract metrics from
            
        Returns:
            Dict of metric names and values
        """
        metrics = {}
        action_type = self._get_action_type(action)
        
        # Extract different metrics based on action type
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
        
        elif action_type == "communication":
            if "response_time" in action:
                try:
                    metrics["response_time_ms"] = float(action["response_time"])
                except (ValueError, TypeError):
                    pass
            
            if "message" in action and isinstance(action["message"], dict) and "priority" in action["message"]:
                try:
                    metrics["message_priority"] = float(action["message"]["priority"])
                except (ValueError, TypeError):
                    pass
        
        elif action_type == "decision_making":
            if "impact_score" in action:
                try:
                    metrics["impact_score"] = float(action["impact_score"])
                except (ValueError, TypeError):
                    pass
        
        # Look for generic metrics across all action types
        common_metric_fields = ["score", "amount", "priority", "level", "limit", "quota", "threshold"]
        for field in common_metric_fields:
            if field in action:
                try:
                    metrics[field] = float(action[field])
                except (ValueError, TypeError):
                    pass
        
        return metrics
    
    def _is_adjustable_metric(self, metric: str, action: Dict[str, Any]) -> bool:
        """
        Check if a metric can be adjusted in an action.
        
        Args:
            metric: Metric name
            action: Action to check
            
        Returns:
            True if metric can be adjusted
        """
        # These metrics can typically be adjusted
        adjustable_metrics = [
            "allocation_amount", "priority_level", "response_time_ms", 
            "message_priority", "impact_score", "score", "amount", 
            "priority", "level", "limit", "quota", "threshold"
        ]
        
        # Check if the metric is in our list of adjustable metrics
        if metric not in adjustable_metrics:
            return False
            
        # Get the location info for this metric
        metric_info = self._get_metric_location(metric, action)
        
        # If we have location info and can find the metric in the action
        return metric_info and metric in metric_info
    
    def _get_metric_location(self, metric: str, action: Dict[str, Any]) -> Dict[str, str]:
        """
        Get the location of a metric in an action.
        
        Args:
            metric: Metric name
            action: Action to check
            
        Returns:
            Dict with metric path information
        """
        # Define mapping of metric names to their locations in actions
        metric_paths = {
            "allocation_amount": "amount",
            "priority_level": "priority",
            "response_time_ms": "response_time",
            "message_priority": "message.priority",
            "impact_score": "impact_score"
        }
        
        # For generic metrics, use the name directly
        generic_metrics = ["score", "amount", "priority", "level", "limit", "quota", "threshold"]
        for gen_metric in generic_metrics:
            metric_paths[gen_metric] = gen_metric
        
        return metric_paths
    
    def _adjust_metric_for_fairness(self, action: Dict[str, Any], metric: str, target_value: float) -> None:
        """
        Adjust a metric in an action to make it more fair.
        
        Args:
            action: Action to modify
            metric: Metric to adjust
            target_value: Target value for the metric
        """
        metric_paths = self._get_metric_location(metric, action)
        metric_path = metric_paths.get(metric)
        
        if not metric_path:
            return
        
        path_parts = metric_path.split(".")
        
        # Navigate to the target location
        target = action
        for i, part in enumerate(path_parts[:-1]):
            if part not in target:
                target[part] = {}
            target = target[part]
        
        # Set the adjusted value
        target[path_parts[-1]] = target_value
    
    def _check_historical_consistency(
        self, 
        action_data: Dict[str, Any], 
        historical_actions: List[Dict[str, Any]]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Check for consistency with similar past situations.
        
        Args:
            action_data: Action to evaluate
            historical_actions: Past actions for comparison
            
        Returns:
            Tuple of (consistency_score, findings)
        """
        score = 1.0
        findings = []
        
        # Find similar past actions
        similar_actions = self._find_similar_actions(action_data, historical_actions)
        
        if len(similar_actions) < 3:  # Need enough similar actions for meaningful comparison
            return score, findings
        
        # Find differences between current action and similar past actions
        differences = self._find_significant_differences(action_data, similar_actions)
        
        for diff in differences:
            # Each significant difference reduces consistency score
            score *= 0.8
            
            findings.append({
                "type": "historical_inconsistency",
                "description": f"Action differs from similar historical actions in {diff['field']}",
                "severity": "medium",
                "evidence": diff,
                "suggested_fix": self._create_historically_consistent_action(action_data, diff)
            })
        
        return score, findings
    
    def _find_similar_actions(
        self, 
        action: Dict[str, Any], 
        historical_actions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find actions similar to the given action.
        
        Args:
            action: Action to find similar actions for
            historical_actions: Past actions to search
            
        Returns:
            List of similar actions
        """
        action_type = self._get_action_type(action)
        similar = []
        
        # Filter by action type first
        type_filtered = [a for
