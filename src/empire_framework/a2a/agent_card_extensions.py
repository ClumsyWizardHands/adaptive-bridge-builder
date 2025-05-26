"""
Empire Framework A2A Agent Card Extensions

This module provides extensions to the A2A Protocol Agent Card standard
specifically for Empire Framework components. It allows agents to advertise
their capabilities for working with Empire components and exchanging them
through the A2A Protocol.
"""

import json
from typing import Dict, List, Any, Optional, Union

class EmpireAgentCardExtensions:
    """
    A utility class for extending agent cards with Empire Framework capabilities.
    
    This class adds specialized capabilities to agent cards that enable them
    to work with Empire Framework components through the A2A Protocol.
    """
    
    @staticmethod
    def add_empire_component_capabilities(agent_card: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add Empire Framework component capabilities to an agent card.
        
        Args:
            agent_card: The agent card to extend
            
        Returns:
            The extended agent card
        """
        # Create a deep copy of the agent card
        extended_card = json.loads(json.dumps(agent_card))
        
        # Add Empire Framework component capabilities
        if "capabilities" not in extended_card:
            extended_card["capabilities"] = []
            
        # Add component exchange capability
        extended_card["capabilities"].append({
            "name": "empire_component_exchange",
            "description": "Exchange Empire Framework components using A2A Protocol",
            "methods": [
                "getComponent", 
                "getComponents", 
                "updateComponent", 
                "createComponent", 
                "deleteComponent", 
                "validateComponent"
            ],
            "parameters": {
                "component_id": "string - ID of the component to operate on",
                "component_type": "string - Type of the component",
                "component_data": "object - Component data",
                "filters": "object - Filters for component queries",
                "schema_id": "string - ID of schema for validation"
            }
        })
        
        # Add component streaming capability
        extended_card["capabilities"].append({
            "name": "empire_component_streaming",
            "description": "Stream Empire Framework components using Server-Sent Events",
            "methods": [
                "streamComponent", 
                "streamComponents", 
                "streamComponentChanges"
            ],
            "parameters": {
                "component_id": "string - ID of the component to stream",
                "interval": "number - Update interval in seconds",
                "max_updates": "number - Maximum number of updates to stream",
                "include_changes_only": "boolean - Whether to include only changes"
            }
        })
        
        # Add component relationship capability
        extended_card["capabilities"].append({
            "name": "empire_component_relationships",
            "description": "Manage relationships between Empire Framework components",
            "methods": [
                "getComponentRelations", 
                "modifyComponentRelations"
            ],
            "parameters": {
                "component_id": "string - ID of the component",
                "relation_types": "array - Types of relationships to include",
                "operation": "string - Operation to perform",
                "relationship_data": "object - Relationship data"
            }
        })
        
        # Add component task capability
        extended_card["capabilities"].append({
            "name": "empire_component_tasks",
            "description": "Create and manage tasks for asynchronous component operations",
            "methods": [
                "createComponentTask", 
                "getTaskStatus", 
                "cancelTask"
            ],
            "parameters": {
                "task_type": "string - Type of task to create",
                "component_ids": "array - IDs of components involved",
                "task_data": "object - Task-specific data",
                "priority": "string - Task priority level",
                "task_id": "string - ID of the task to check or cancel"
            }
        })
        
        # Update agent card communication section
        if "communication" not in extended_card:
            extended_card["communication"] = {}
            
        if "protocols" not in extended_card["communication"]:
            extended_card["communication"]["protocols"] = []
            
        # Add A2A protocol if not already present
        if "a2a" not in extended_card["communication"]["protocols"]:
            extended_card["communication"]["protocols"].append("a2a")
            
        # Add JSON-RPC protocol if not already present
        if "json-rpc-2.0" not in extended_card["communication"]["protocols"]:
            extended_card["communication"]["protocols"].append("json-rpc-2.0")
            
        # Add SSE protocol if not already present
        if "server-sent-events" not in extended_card["communication"]["protocols"]:
            extended_card["communication"]["protocols"].append("server-sent-events")
            
        # Add supported formats if not present
        if "formats" not in extended_card["communication"]:
            extended_card["communication"]["formats"] = []
            
        # Add JSON format if not already present
        if "json" not in extended_card["communication"]["formats"]:
            extended_card["communication"]["formats"].append("json")
            
        # Add the Empire Framework Extensions section
        if "empire_framework" not in extended_card:
            extended_card["empire_framework"] = {}
            
        # Add component types supported
        extended_card["empire_framework"]["supported_component_types"] = [
            "Principle",
            "Mean",
            "End",
            "Identity",
            "Resentment",
            "Emotion"
        ]
        
        # Add operation modes supported
        extended_card["empire_framework"]["supported_operations"] = {
            "sync_operations": ["query", "retrieve", "validate"],
            "async_operations": ["batch", "transform", "analyze", "import", "export"],
            "streaming_operations": ["updates", "changes", "batches"]
        }
        
        # Add schema validation capabilities
        extended_card["empire_framework"]["schema_validation"] = {
            "supported": True,
            "schemas": [
                "core_empire_component_schema",
                "principles_schema",
                "means_schema",
                "ends_schema", 
                "identity_schema",
                "resentments_schema",
                "emotions_schema"
            ]
        }
        
        return extended_card
    
    @staticmethod
    def add_principle_engine_capabilities(agent_card: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add principle engine capabilities to an agent card.
        
        Args:
            agent_card: The agent card to extend
            
        Returns:
            The extended agent card
        """
        # Create a deep copy of the agent card
        extended_card = json.loads(json.dumps(agent_card))
        
        # Add principle engine capability
        if "capabilities" not in extended_card:
            extended_card["capabilities"] = []
            
        extended_card["capabilities"].append({
            "name": "principle_engine",
            "description": "Evaluate messages and actions against principles",
            "methods": [
                "evaluateMessage", 
                "evaluateAction", 
                "getPrinciples", 
                "addPrinciple",
                "resolvePrincipleConflict"
            ],
            "parameters": {
                "message": "string/object - Message or action to evaluate",
                "principles": "array - Specific principles to evaluate against",
                "context": "object - Additional context for evaluation",
                "principle_data": "object - Data for new principle",
                "conflict_resolution_strategy": "string - Strategy for resolving conflicts"
            }
        })
        
        # Add principle engine extensions to Empire Framework section
        if "empire_framework" not in extended_card:
            extended_card["empire_framework"] = {}
            
        extended_card["empire_framework"]["principle_engine"] = {
            "supported_evaluation_modes": [
                "individual_message",
                "conversation_context",
                "action_evaluation",
                "decision_making"
            ],
            "conflict_resolution_strategies": [
                "priority_based",
                "context_weighted",
                "compromise",
                "escalation"
            ],
            "evaluation_response_formats": [
                "simple_score",
                "detailed_analysis",
                "principle_alignment_breakdown",
                "suggested_alternatives"
            ]
        }
        
        return extended_card
    
    @staticmethod
    def add_emotional_intelligence_capabilities(agent_card: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add emotional intelligence capabilities to an agent card.
        
        Args:
            agent_card: The agent card to extend
            
        Returns:
            The extended agent card
        """
        # Create a deep copy of the agent card
        extended_card = json.loads(json.dumps(agent_card))
        
        # Add emotional intelligence capability
        if "capabilities" not in extended_card:
            extended_card["capabilities"] = []
            
        extended_card["capabilities"].append({
            "name": "emotional_intelligence",
            "description": "Emotional analysis and appropriate response generation",
            "methods": [
                "detectEmotions", 
                "generateEmotionalResponse", 
                "getEmotionalContext",
                "updateEmotionalContext"
            ],
            "parameters": {
                "message": "string - Message to analyze",
                "history": "array - Previous messages for context",
                "emotion_type": "string - Type of emotion to generate",
                "intensity": "number - Intensity level for emotional response",
                "adaption_level": "string - How much to adapt to detected emotions"
            }
        })
        
        # Add emotional intelligence extensions to Empire Framework section
        if "empire_framework" not in extended_card:
            extended_card["empire_framework"] = {}
            
        extended_card["empire_framework"]["emotional_intelligence"] = {
            "detectable_emotions": [
                "joy", "trust", "fear", "surprise", 
                "sadness", "disgust", "anger", "anticipation",
                "interest", "confusion", "frustration", "satisfaction"
            ],
            "response_styles": [
                "empathetic", "neutral", "solution-focused", 
                "encouraging", "reflective", "validating"
            ],
            "context_tracking": {
                "conversation_level": True,
                "user_profile_level": True,
                "long_term_patterns": True
            }
        }
        
        return extended_card
    
    @staticmethod
    def add_fairness_evaluation_capabilities(agent_card: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add fairness evaluation capabilities to an agent card.
        
        Args:
            agent_card: The agent card to extend
            
        Returns:
            The extended agent card
        """
        # Create a deep copy of the agent card
        extended_card = json.loads(json.dumps(agent_card))
        
        # Add fairness evaluation capability
        if "capabilities" not in extended_card:
            extended_card["capabilities"] = []
            
        extended_card["capabilities"].append({
            "name": "fairness_evaluation",
            "description": "Evaluate content for fairness across multiple dimensions",
            "methods": [
                "evaluateFairness", 
                "generateAlternatives", 
                "getFairnessMetrics",
                "compareFairness"
            ],
            "parameters": {
                "content": "string/object - Content to evaluate",
                "dimensions": "array - Fairness dimensions to evaluate",
                "context": "object - Additional context for evaluation",
                "flags": "array - Specific fairness issues to address",
                "comparison_id": "string - ID of content to compare with"
            }
        })
        
        # Add fairness evaluation extensions to Empire Framework section
        if "empire_framework" not in extended_card:
            extended_card["empire_framework"] = {}
            
        extended_card["empire_framework"]["fairness_evaluation"] = {
            "evaluation_dimensions": [
                "language_bias", "assumption_bias", "treatment_bias",
                "perspective_diversity", "decision_consistency",
                "rule_application_consistency", "process_visibility",
                "language_complexity"
            ],
            "alternative_generation_methods": [
                "bias_mitigation", "perspective_enrichment",
                "consistency_improvement", "transparency_enhancement"
            ],
            "metric_tracking": {
                "historical_comparison": True,
                "domain_benchmarking": True,
                "continuous_improvement": True
            }
        }
        
        return extended_card

def extend_agent_card_with_empire_capabilities(agent_card: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add all Empire Framework capabilities to an agent card.
    
    Args:
        agent_card: The agent card to extend
        
    Returns:
        The fully extended agent card
    """
    extended_card = EmpireAgentCardExtensions.add_empire_component_capabilities(agent_card)
    extended_card = EmpireAgentCardExtensions.add_principle_engine_capabilities(extended_card)
    extended_card = EmpireAgentCardExtensions.add_emotional_intelligence_capabilities(extended_card)
    extended_card = EmpireAgentCardExtensions.add_fairness_evaluation_capabilities(extended_card)
    
    # Add version and last updated information
    if "empire_framework" in extended_card:
        extended_card["empire_framework"]["version"] = "1.0.0"
        extended_card["empire_framework"]["last_updated"] = "2025-05-18"
    
    return extended_card
