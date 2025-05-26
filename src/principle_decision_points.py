#!/usr/bin/env python3
"""
Principle Decision Points

This module identifies critical decision points in the Adaptive Bridge Builder
where principles should be evaluated before actions are taken or responses
are formulated. It also provides mechanisms to collect contextual data for
each decision point to enable principle-based evaluation.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, TypeVar, Generic

from a2a_task_handler import A2ATaskHandler
from orchestrator_engine import OrchestratorEngine, TaskType
from conflict_resolver import ConflictResolver
from principle_engine_llm import PrincipleEngineLLM, PrincipleEvalResult
from agent_registry import AgentRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PrincipleDecisionPoints")

# Type definitions
T = TypeVar('T')  # Unused, kept for potential future use
ActionType = TypeVar('ActionType', bound=Union[str, Dict[str, Any]])
ContextType = TypeVar('ContextType', bound=Dict[str, Any])
ResultType = TypeVar('ResultType', bound=Tuple[Any, bool, Any])


class DecisionPoint(Generic[ActionType, ContextType, ResultType]):
    """
    Represents a decision point in the system where principles should be evaluated.
    
    A decision point encapsulates:
    1. A specific location in the code where a decision is made
    2. The action being considered at that point
    3. The contextual data relevant to evaluating the action
    4. The ability to modify the action based on principle evaluation
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        component: str,
        context_collector: Callable[..., ContextType],
        action_modifier: Optional[Callable[[ActionType, PrincipleEvalResult], Tuple[ActionType, bool]]] = None,
        principles: Optional[List[str]] = None,
        alignment_threshold: float = 0.7
    ):
        """
        Initialize a decision point.
        
        Args:
            name: Name of the decision point
            description: Description of what happens at this decision point
            component: Component where this decision point exists
            context_collector: Function to collect contextual data for evaluation
            action_modifier: Optional function to modify action based on evaluation
            principles: Optional list of principles to evaluate against (default: all)
            alignment_threshold: Threshold for principle alignment (0.0-1.0)
        """
        self.name = name
        self.description = description
        self.component = component
        self.context_collector = context_collector
        self.action_modifier = action_modifier
        self.principles = principles
        self.alignment_threshold = alignment_threshold
        self.evaluations: List[Dict[str, Any]] = []
    
    async def evaluate_action(
        self,
        action: ActionType,
        principle_engine: PrincipleEngineLLM,
        **context_args
    ) -> Tuple[ActionType, bool, PrincipleEvalResult]:
        """
        Evaluate an action against principles and possibly modify it.
        
        Args:
            action: The action to evaluate
            principle_engine: Engine to use for evaluation
            **context_args: Arguments to pass to context collector
            
        Returns:
            Tuple of (potentially_modified_action, was_modified, evaluation_result)
        """
        # Collect context
        context = self.context_collector(**context_args)
        
        # Convert action to string if it's not already
        action_description = str(action) if not isinstance(action, str) else action
        
        # Evaluate action against principles
        eval_result = await principle_engine.evaluate_action(
            action_description=action_description,
            context=context,
            principles=self.principles,
            alignment_threshold=self.alignment_threshold
        )
        
        # Record the evaluation
        self.evaluations.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action_description,
            "context_summary": principle_engine._summarize_context(context),
            "evaluation": eval_result.to_dict()
        })
        
        # If action is misaligned and we have a modifier function, try to modify it
        was_modified = False
        modified_action = action
        
        if not eval_result.aligned and self.action_modifier:
            try:
                modified_action, was_modified = self.action_modifier(action, eval_result)
                
                if was_modified:
                    logger.info(
                        f"Action at decision point '{self.name}' was modified to align with principles. "
                        f"Score before: {eval_result.overall_score:.2f}"
                    )
            except Exception as e:
                logger.error(f"Error modifying action at decision point '{self.name}': {str(e)}")
        
        return modified_action, was_modified, eval_result
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about evaluations at this decision point.
        
        Returns:
            Dictionary of evaluation statistics
        """
        if not self.evaluations:
            return {
                "decision_point": self.name,
                "component": self.component,
                "total_evaluations": 0,
                "aligned_count": 0,
                "aligned_percentage": 0.0,
                "average_score": 0.0,
                "modifications": 0,
                "modification_percentage": 0.0
            }
        
        # Calculate statistics
        total = len(self.evaluations)
        aligned_count = sum(1 for eval_dict in self.evaluations 
                           if eval_dict["evaluation"]["aligned"])
        aligned_percentage = (aligned_count / total) * 100 if total > 0 else 0.0
        
        scores = [eval_dict["evaluation"]["overall_score"] for eval_dict in self.evaluations]
        average_score = sum(scores) / len(scores) if scores else 0.0
        
        modifications = sum(1 for eval_dict in self.evaluations 
                           if eval_dict.get("was_modified", False))
        modification_percentage = (modifications / total) * 100 if total > 0 else 0.0
        
        return {
            "decision_point": self.name,
            "component": self.component,
            "total_evaluations": total,
            "aligned_count": aligned_count,
            "aligned_percentage": aligned_percentage,
            "average_score": average_score,
            "modifications": modifications,
            "modification_percentage": modification_percentage
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Decision Point 1: A2A Task Handler message processing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def collect_a2a_task_context(
    task_handler: A2ATaskHandler,
    message: Dict[str, Any],
    agent_id: str,
    conversation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Collect context data for A2A task handler message processing.
    
    Args:
        task_handler: The A2A task handler
        message: The incoming message
        agent_id: ID of the sender agent
        conversation_id: Optional conversation ID
        
    Returns:
        Context data for principle evaluation
    """
    # Extract important message fields
    method = message.get("method", "unknown")
    params = message.get("params", {})
    
    # Get relationship information if available
    relationship_data = {}
    if task_handler.relationship_tracker:
        relationship = task_handler.relationship_tracker.get_relationship(agent_id)
        if relationship:
            relationship_data = {
                "trust_level": relationship.trust_level.name,
                "interaction_count": len(relationship.interactions),
                "last_interaction": relationship.last_interaction.isoformat() 
                    if relationship.last_interaction else None,
                "relationship_quality": relationship.calculate_quality()
            }
    
    # Get conversation context if available
    conversation_data = {}
    if conversation_id:
        context_key = f"{agent_id}:{conversation_id}"
        if context_key in task_handler.active_contexts:
            context = task_handler.active_contexts[context_key]
            conversation_data = {
                "message_count": len(context.messages),
                "topics": context.topics,
                "common_intent": context.get_common_intent(),
                "last_updated": context.last_updated
            }
    
    # Compile context data
    context = {
        "message": {
            "method": method,
            "params_summary": {k: v for k, v in params.items() if k not in ["message", "data"]},
            "has_content": "content" in params,
            "has_data": "data" in params
        },
        "agent": {
            "id": agent_id,
            "relationship": relationship_data
        },
        "conversation": conversation_data,
        "system_state": {
            "active_conversations": len(task_handler.active_contexts),
            "task_count": len(task_handler.tasks)
        }
    }
    
    return context


def modify_a2a_task_response(
    response: Dict[str, Any], 
    eval_result: PrincipleEvalResult
) -> Tuple[Dict[str, Any], bool]:
    """
    Modify an A2A task response to better align with principles.
    
    Args:
        response: The response to modify
        eval_result: Principle evaluation result
        
    Returns:
        Tuple of (modified_response, was_modified)
    """
    modified = False
    modified_response = response.copy()
    
    # Check if response has a result field
    if "result" not in modified_response:
        return response, False
    
    result = modified_response["result"]
    if not isinstance(result, dict):
        return response, False
    
    # Apply fairness principles
    fairness_score = eval_result.principle_scores.get("fairness_as_truth", 1.0)
    if fairness_score < 0.7:
        # Remove any priority indicators
        if "priority" in result:
            result.pop("priority")
            modified = True
        if "priority_level" in result:
            result.pop("priority_level")
            modified = True
    
    # Apply harmony principles
    harmony_score = eval_result.principle_scores.get("harmony_through_presence", 1.0)
    if harmony_score < 0.7:
        # Add acknowledgment if missing
        if "acknowledged" not in result:
            result["acknowledged"] = True
            modified = True
        # Add timestamp if missing
        if "timestamp" not in result:
            result["timestamp"] = datetime.now(timezone.utc).isoformat()
            modified = True
    
    # Apply truth principles
    truth_score = eval_result.principle_scores.get("truth_in_representation", 1.0)
    if truth_score < 0.7:
        # Add clear limitation notices if any capabilities are overrepresented
        if "capabilities" in result:
            result["capabilities_note"] = "Note: These capabilities represent current functionality and may be subject to limitations."
            modified = True
    
    # Apply clarity principles
    clarity_score = eval_result.principle_scores.get("clarity_in_complexity", 1.0)
    if clarity_score < 0.7 and isinstance(result.get("message"), str):
        # Simplify overly complex messages
        if len(result["message"]) > 200:
            # Preserve original message but add simplified version
            result["simplified_message"] = result["message"][:200] + "... [simplified for clarity]"
            modified = True
    
    # Add feedback request for growth
    growth_score = eval_result.principle_scores.get("growth_through_reflection", 1.0)
    if growth_score < 0.7:
        result["feedback_requested"] = True
        modified = True
    
    # Add recommendations from the evaluation
    if eval_result.recommendations:
        result["principle_recommendations"] = eval_result.recommendations[:3]
        modified = True
    
    return modified_response, modified


# Create A2A Task Handler decision point
a2a_task_decision_point = DecisionPoint(
    name="a2a_task_response_generation",
    description="Evaluates and potentially modifies responses generated by the A2A Task Handler",
    component="A2ATaskHandler",
    context_collector=collect_a2a_task_context,
    action_modifier=modify_a2a_task_response,
    principles=["fairness_as_truth", "harmony_through_presence", "clarity_in_complexity", 
               "integrity_in_transmission", "truth_in_representation", "growth_through_reflection"],
    alignment_threshold=0.7
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Decision Point 2: Orchestrator Engine task assignment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def collect_orchestrator_task_context(
    orchestrator: OrchestratorEngine,
    task_id: str,
    task_type: TaskType,
    agent_candidates: List[str],
    priority: Any
) -> Dict[str, Any]:
    """
    Collect context data for orchestrator task assignment.
    
    Args:
        orchestrator: The orchestrator engine
        task_id: ID of the task being assigned
        task_type: Type of the task
        agent_candidates: List of candidate agent IDs
        priority: Task priority
        
    Returns:
        Context data for principle evaluation
    """
    # Get task details
    task = orchestrator.task_coordinator.get_task(task_id)
    
    # Get agent candidate profiles
    candidate_profiles = []
    for agent_id in agent_candidates:
        profile = orchestrator.agent_profiles.get(agent_id)
        if profile:
            candidate_profiles.append({
                "agent_id": agent_id,
                "current_load": profile.current_load,
                "max_load": profile.max_load,
                "availability": profile.availability.value if hasattr(profile.availability, "value") else str(profile.availability),
                "specialization_score": profile.specialization.get(task_type, 0.0) if hasattr(profile, "specialization") else 0.0,
                "success_rate": profile.success_rate.get(task_type, 0.0) if hasattr(profile, "success_rate") else 0.0
            })
    
    # Compile context data
    context = {
        "task": {
            "id": task_id,
            "type": task_type.name if hasattr(task_type, "name") else str(task_type),
            "priority": str(priority),
            "title": task.title if task else "Unknown",
            "status": task.status.value if task and hasattr(task.status, "value") else "Unknown"
        },
        "agent_candidates": candidate_profiles,
        "system_state": {
            "total_agents": len(orchestrator.agent_profiles),
            "active_tasks": len(orchestrator.running_tasks),
            "pending_tasks": sum(len(queue) for queue in orchestrator.task_queues.values())
        }
    }
    
    return context


def modify_orchestrator_assignment(
    assignment: Dict[str, Any],
    eval_result: PrincipleEvalResult
) -> Tuple[Dict[str, Any], bool]:
    """
    Modify an orchestrator task assignment to better align with principles.
    
    Args:
        assignment: The assignment to modify
        eval_result: Principle evaluation result
        
    Returns:
        Tuple of (modified_assignment, was_modified)
    """
    modified = False
    modified_assignment = assignment.copy()
    
    # Check if response has agent_id field
    if "agent_id" not in modified_assignment:
        return assignment, False
    
    # Apply balance principles
    balance_score = eval_result.principle_scores.get("balance_in_mediation", 1.0)
    if balance_score < 0.7:
        # Add a note about fair assignment
        modified_assignment["assignment_note"] = "Assigned based on balanced workload distribution"
        modified = True
    
    # Apply resilience principles
    resilience_score = eval_result.principle_scores.get("resilience_through_connection", 1.0)
    if resilience_score < 0.7:
        # Add backup agents
        if "backup_agents" not in modified_assignment and "candidate_agents" in assignment:
            candidates = assignment.get("candidate_agents", [])
            if len(candidates) > 1 and modified_assignment["agent_id"] in candidates:
                # Use candidates other than the primary agent as backups
                backups = [a for a in candidates if a != modified_assignment["agent_id"]][:2]
                modified_assignment["backup_agents"] = backups
                modified = True
    
    # Apply adaptability principles
    adaptability_score = eval_result.principle_scores.get("adaptability_as_strength", 1.0)
    if adaptability_score < 0.7:
        # Add adaptive retry logic
        modified_assignment["adaptive_retry"] = True
        modified_assignment["max_retries"] = 3
        modified = True
    
    # Add recommendations from the evaluation
    if eval_result.recommendations:
        modified_assignment["principle_recommendations"] = eval_result.recommendations[:3]
        modified = True
    
    return modified_assignment, modified


# Create Orchestrator Engine decision point
orchestrator_decision_point = DecisionPoint(
    name="orchestrator_task_assignment",
    description="Evaluates and potentially modifies task assignments made by the Orchestrator Engine",
    component="OrchestratorEngine",
    context_collector=collect_orchestrator_task_context,
    action_modifier=modify_orchestrator_assignment,
    principles=["fairness_as_truth", "balance_in_mediation", "adaptability_as_strength", 
               "resilience_through_connection", "empathy_in_interface"],
    alignment_threshold=0.7
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Decision Point 3: Conflict Resolver conflict handling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def collect_conflict_resolver_context(
    resolver: ConflictResolver,
    conflict_id: str,
    parties: List[str],
    conflict_type: str,
    severity: Any
) -> Dict[str, Any]:
    """
    Collect context data for conflict resolution.
    
    Args:
        resolver: The conflict resolver
        conflict_id: ID of the conflict
        parties: List of parties involved in the conflict
        conflict_type: Type of conflict
        severity: Conflict severity
        
    Returns:
        Context data for principle evaluation
    """
    # Get conflict record
    conflict_record = resolver.get_conflict_record(conflict_id)
    
    # Get relationship information if available
    relationship_data = {}
    if resolver.relationship_tracker:
        for party in parties:
            relationship = resolver.relationship_tracker.get_relationship(party)
            if relationship:
                relationship_data[party] = {
                    "trust_level": relationship.trust_level.name,
                    "interaction_count": len(relationship.interactions),
                    "relationship_quality": relationship.calculate_quality()
                }
    
    # Compile context data
    context = {
        "conflict": {
            "id": conflict_id,
            "type": conflict_type,
            "severity": str(severity),
            "parties": parties,
            "indicators": conflict_record.indicators if conflict_record else [],
            "status": conflict_record.status.name if conflict_record and hasattr(conflict_record.status, "name") else "Unknown"
        },
        "relationships": relationship_data,
        "system_state": {
            "active_conflicts": len(resolver.active_conflicts),
            "resolved_conflicts": len(resolver.resolved_conflicts)
        }
    }
    
    return context


def modify_conflict_resolution(
    resolution: Dict[str, Any],
    eval_result: PrincipleEvalResult
) -> Tuple[Dict[str, Any], bool]:
    """
    Modify a conflict resolution to better align with principles.
    
    Args:
        resolution: The resolution to modify
        eval_result: Principle evaluation result
        
    Returns:
        Tuple of (modified_resolution, was_modified)
    """
    modified = False
    modified_resolution = resolution.copy()
    
    # Apply fairness principles
    fairness_score = eval_result.principle_scores.get("fairness_as_truth", 1.0)
    if fairness_score < 0.7:
        # Ensure resolution is fair to all parties
        modified_resolution["fairness_verification"] = True
        if "affected_parties" in resolution:
            modified_resolution["equal_consideration"] = True
        modified = True
    
    # Apply balance principles
    balance_score = eval_result.principle_scores.get("balance_in_mediation", 1.0)
    if balance_score < 0.7:
        # Ensure balanced resolution
        modified_resolution["balanced_approach"] = True
        modified = True
    
    # Apply empathy principles
    empathy_score = eval_result.principle_scores.get("empathy_in_interface", 1.0)
    if empathy_score < 0.7:
        # Add empathetic communication
        if "message" in modified_resolution:
            message = modified_resolution["message"]
            modified_resolution["message"] = f"We understand this situation may be challenging. {message}"
            modified = True
    
    # Add recommendations from the evaluation
    if eval_result.recommendations:
        modified_resolution["principle_recommendations"] = eval_result.recommendations[:3]
        modified = True
    
    return modified_resolution, modified


# Create Conflict Resolver decision point
conflict_decision_point = DecisionPoint(
    name="conflict_resolution_generation",
    description="Evaluates and potentially modifies conflict resolutions generated by the Conflict Resolver",
    component="ConflictResolver",
    context_collector=collect_conflict_resolver_context,
    action_modifier=modify_conflict_resolution,
    principles=["fairness_as_truth", "balance_in_mediation", "empathy_in_interface", 
               "truth_in_representation", "clarity_in_complexity"],
    alignment_threshold=0.7
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Decision Point 4: Agent Registry agent selection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def collect_agent_registry_context(
    registry: AgentRegistry,
    task_type: str,
    required_capabilities: List[str],
    preferred_agent_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Collect context data for agent selection.
    
    Args:
        registry: The agent registry
        task_type: Type of task requiring an agent
        required_capabilities: List of required capabilities
        preferred_agent_type: Optional preferred agent type
        
    Returns:
        Context data for principle evaluation
    """
    # Get all agents of the preferred type
    candidate_agents = []
    if preferred_agent_type:
        agents = registry.get_agents_by_type(preferred_agent_type)
        candidate_agents = [agent.id for agent in agents if agent]
    
    # Get agents with required capabilities
    capable_agents = []
    for capability in required_capabilities:
        agents = registry.get_agents_by_capability(capability)
        capable_agents.extend([agent.id for agent in agents if agent])
    
    # Count unique capable agents
    unique_capable_agents = list(set(capable_agents))
    
    # Compile context data
    context = {
        "task": {
            "type": task_type,
            "required_capabilities": required_capabilities
        },
        "agent_selection": {
            "preferred_type": preferred_agent_type,
            "candidate_count": len(candidate_agents),
            "capable_agent_count": len(unique_capable_agents)
        },
        "system_state": {
            "total_agents": len(registry.agents),
            "agent_types": list(set(agent.type for agent in registry.agents.values() if hasattr(agent, 'type')))
        }
    }
    
    return context


def modify_agent_selection(
    selection: Dict[str, Any],
    eval_result: PrincipleEvalResult
) -> Tuple[Dict[str, Any], bool]:
    """
    Modify an agent selection to better align with principles.
    
    Args:
        selection: The selection to modify
        eval_result: Principle evaluation result
        
    Returns:
        Tuple of (modified_selection, was_modified)
    """
    modified = False
    modified_selection = selection.copy()
    
    # Apply adaptability principles
    adaptability_score = eval_result.principle_scores.get("adaptability_as_strength", 1.0)
    if adaptability_score < 0.7:
        # Add adaptive selection note
        modified_selection["adaptive_selection"] = True
        modified = True
    
    # Apply resilience principles
    resilience_score = eval_result.principle_scores.get("resilience_through_connection", 1.0)
    if resilience_score < 0.7:
        # Add backup agents if available
        if "backup_agents" not in modified_selection and "alternative_agents" in selection:
            modified_selection["backup_agents"] = selection["alternative_agents"][:2]
            modified = True
    
    # Apply truth principles
    truth_score = eval_result.principle_scores.get("truth_in_representation", 1.0)
    if truth_score < 0.7:
        # Add capability verification
        modified_selection["capability_verified"] = True
        modified = True
    
    # Add recommendations from the evaluation
    if eval_result.recommendations:
        modified_selection["principle_recommendations"] = eval_result.recommendations[:3]
        modified = True
    
    return modified_selection, modified


# Create Agent Registry decision point
registry_decision_point = DecisionPoint(
    name="agent_registry_selection",
    description="Evaluates and potentially modifies agent selections made by the Agent Registry",
    component="AgentRegistry",
    context_collector=collect_agent_registry_context,
    action_modifier=modify_agent_selection,
    principles=["adaptability_as_strength", "resilience_through_connection", 
               "truth_in_representation", "empathy_in_interface"],
    alignment_threshold=0.7
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Decision Point 5: Universal Agent Connector request formation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def collect_agent_connector_context(
    connector_id: str,
    target_agent_id: str,
    request_type: str,
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Collect context data for agent connector request formation.
    
    Args:
        connector_id: ID of the connector
        target_agent_id: ID of the target agent
        request_type: Type of request
        payload: Request payload
        
    Returns:
        Context data for principle evaluation
    """
    # Compile context data
    context = {
        "connector": {
            "id": connector_id
        },
        "target": {
            "agent_id": target_agent_id,
        },
        "request": {
            "type": request_type,
            "fields": list(payload.keys())
        }
    }
    
    return context


def modify_agent_connector_request(
    request: Dict[str, Any],
    eval_result: PrincipleEvalResult
) -> Tuple[Dict[str, Any], bool]:
    """
    Modify an agent connector request to better align with principles.
    
    Args:
        request: The request to modify
        eval_result: Principle evaluation result
        
    Returns:
        Tuple of (modified_request, was_modified)
    """
    modified = False
    modified_request = request.copy()
    
    # Apply integrity principles
    integrity_score = eval_result.principle_scores.get("integrity_in_transmission", 1.0)
    if integrity_score < 0.7:
        # Add integrity verification
        modified_request["verify_integrity"] = True
        modified = True
    
    # Apply clarity principles
    clarity_score = eval_result.principle_scores.get("clarity_in_complexity", 1.0)
    if clarity_score < 0.7 and "payload" in modified_request:
        # Simplify payload if needed
        payload = modified_request["payload"]
        if isinstance(payload, dict) and len(json.dumps(payload)) > 500:
            # Add a note about simplification
            modified_request["simplified_for_clarity"] = True
            modified = True
    
    # Apply harmony principles
    harmony_score = eval_result.principle_scores.get("harmony_through_presence", 1.0)
    if harmony_score < 0.7:
        # Add tracking information
        if "metadata" not in modified_request:
            modified_request["metadata"] = {}
        
        modified_request["metadata"]["request_id"] = str(uuid.uuid4())
        modified_request["metadata"]["timestamp"] = datetime.now(timezone.utc).isoformat()
        modified = True
    
    # Add recommendations from the evaluation
    if eval_result.recommendations:
        if "metadata" not in modified_request:
            modified_request["metadata"] = {}
        modified_request["metadata"]["principle_recommendations"] = eval_result.recommendations[:3]
        modified = True
    
    return modified_request, modified


# Create Universal Agent Connector decision point
connector_decision_point = DecisionPoint(
    name="agent_connector_request_formation",
    description="Evaluates and potentially modifies requests formed by the Universal Agent Connector",
    component="UniversalAgentConnector",
    context_collector=collect_agent_connector_context,
    action_modifier=modify_agent_connector_request,
    principles=["integrity_in_transmission", "clarity_in_complexity", 
               "harmony_through_presence", "growth_through_reflection"],
    alignment_threshold=0.7
)