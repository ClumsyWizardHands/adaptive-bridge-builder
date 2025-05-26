#!/usr/bin/env python3
"""
Principle Decision Points Example

This module demonstrates how to use the DecisionPoint class to integrate
principle-based evaluation and modification into the codebase. It shows
how to connect decision points with the PrincipleEngineLLM and how to 
log and audit principled decisions.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from a2a_task_handler import A2ATaskHandler
from orchestrator_engine import OrchestratorEngine
from conflict_resolver import ConflictResolver
from agent_registry import AgentRegistry
from principle_engine_llm import PrincipleEngineLLM
from principle_decision_points import (
    a2a_task_decision_point,
    orchestrator_decision_point,
    conflict_decision_point,
    registry_decision_point,
    connector_decision_point
)
from universal_agent_connector import UniversalAgentConnector
from llm_adapter_interface import BaseLLMAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("principle_decisions.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PrincipleDecisionPoints")


class PrincipleDecisionAuditor:
    """
    Manages the auditing and logging of principle-based decisions.
    
    This class tracks all principle evaluations, modifications, and their outcomes,
    providing tools for analysis and improvement of principle-based decision making.
    """
    
    def __init__(self, log_file: str = "principle_audit.jsonl") -> None:
        """
        Initialize the auditor.
        
        Args:
            log_file: Path to the log file for audit records
        """
        self.log_file = log_file
        self.evaluation_count = 0
        self.modification_count = 0
        self.decision_points = {}
    
    def record_decision(
        self,
        decision_point_name: str,
        component: str,
        action_description: str,
        context_summary: Dict[str, Any],
        evaluation_result: Dict[str, Any],
        was_modified: bool,
        modification_details: Optional[Dict[str, Any]] = None
    ):
        """
        Record a principle-based decision for auditing.
        
        Args:
            decision_point_name: Name of the decision point
            component: Component where the decision was made
            action_description: Description of the action evaluated
            context_summary: Summary of the context for evaluation
            evaluation_result: Results of the principle evaluation
            was_modified: Whether the action was modified
            modification_details: Details of any modifications made
        """
        # Increment counters
        self.evaluation_count = self.evaluation_count + 1
        if was_modified:
            self.modification_count = self.modification_count + 1
        
        # Ensure decision point is tracked
        if decision_point_name not in self.decision_points:
            self.decision_points = {**self.decision_points, decision_point_name: {}
                "component": component,
                "evaluation_count": 0,
                "modification_count": 0,
                "aligned_count": 0,
                "misaligned_count": 0
            }
        
        # Update decision point stats
        dp_stats = self.decision_points[decision_point_name]
        dp_stats["evaluation_count"] += 1
        if was_modified:
            dp_stats["modification_count"] += 1
        
        if evaluation_result.get("aligned", False):
            dp_stats["aligned_count"] += 1
        else:
            dp_stats["misaligned_count"] += 1
        
        # Prepare log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision_point": decision_point_name,
            "component": component,
            "action_description": action_description,
            "context_summary": context_summary,
            "evaluation_result": evaluation_result,
            "was_modified": was_modified
        }
        
        if modification_details:
            log_entry["modification_details"] = modification_details
        
        # Write to log file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Log to regular log
        if was_modified:
            logger.info(
                f"Principle-based modification at {decision_point_name}: "
                f"Overall score: {evaluation_result.get('overall_score', 0):.2f}, "
                f"Modified: Yes"
            )
        elif not evaluation_result.get("aligned", True):
            logger.warning(
                f"Principle misalignment at {decision_point_name}: "
                f"Overall score: {evaluation_result.get('overall_score', 0):.2f}, "
                f"Not modified"
            )
        else:
            logger.debug(
                f"Principle alignment at {decision_point_name}: "
                f"Overall score: {evaluation_result.get('overall_score', 0):.2f}"
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics for principle-based decisions.
        
        Returns:
            Dictionary of decision statistics
        """
        # Calculate overall statistics
        aligned_count = sum(dp["aligned_count"] for dp in self.decision_points.values())
        misaligned_count = sum(dp["misaligned_count"] for dp in self.decision_points.values())
        alignment_rate = aligned_count / self.evaluation_count if self.evaluation_count > 0 else 0.0
        modification_rate = self.modification_count / misaligned_count if misaligned_count > 0 else 0.0
        
        # Return statistics dictionary
        return {
            "evaluation_count": self.evaluation_count,
            "aligned_count": aligned_count,
            "misaligned_count": misaligned_count,
            "alignment_rate": alignment_rate,
            "modification_count": self.modification_count,
            "modification_rate": modification_rate,
            "decision_points": {
                name: {
                    "component": stats["component"],
                    "evaluation_count": stats["evaluation_count"],
                    "aligned_count": stats["aligned_count"],
                    "misaligned_count": stats["misaligned_count"],
                    "alignment_rate": stats["aligned_count"] / stats["evaluation_count"] 
                        if stats["evaluation_count"] > 0 else 0.0,
                    "modification_count": stats["modification_count"],
                    "modification_rate": stats["modification_count"] / stats["misaligned_count"]
                        if stats["misaligned_count"] > 0 else 0.0
                }
                for name, stats in self.decision_points.items()
            }
        }


class PrincipleEnabledSystem:
    """
    A system with integrated principle-based decision making.
    
    This class demonstrates how to integrate the DecisionPoint framework
    with various components of the Adaptive Bridge Builder.
    """
    
    def __init__(self, llm_adapter: BaseLLMAdapter) -> None:
        """
        Initialize the system.
        
        Args:
            llm_adapter: LLM adapter for principle evaluation
        """
        # Create components
        self.principle_engine = PrincipleEngineLLM(llm_adapter)
        self.auditor = PrincipleDecisionAuditor()
        
        # Initialize component instances (would be properly initialized in a real system)
        self.a2a_handler = A2ATaskHandler(None, None, None)
        self.orchestrator = OrchestratorEngine(None, None)
        self.conflict_resolver = ConflictResolver(None)
        self.agent_registry = AgentRegistry()
        self.agent_connector = UniversalAgentConnector(None)
        
        # Set up decision points with their respective components
        self.decision_points = {
            "a2a_task_response": a2a_task_decision_point,
            "orchestrator_assignment": orchestrator_decision_point, 
            "conflict_resolution": conflict_decision_point,
            "agent_selection": registry_decision_point,
            "connector_request": connector_decision_point
        }
        
        logger.info("PrincipleEnabledSystem initialized with principle-based decision points")
    
    async def handle_a2a_task_request(self, message: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
        """
        Handle an A2A task request with principle-based evaluation.
        
        Args:
            message: The incoming message
            agent_id: ID of the sender agent
            
        Returns:
            The response (potentially modified based on principles)
        """
        logger.info(f"Processing A2A task request from agent {agent_id}")
        
        # Generate initial response
        initial_response = {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {
                "status": "success",
                "message": "Task processed successfully",
                "data": {"task_id": "task-123"},
                "capabilities": ["text_processing", "information_retrieval", "task_execution"]
            }
        }
        
        # Evaluate against principles
        decision_point = self.decision_points["a2a_task_response"]
        modified_response, was_modified, eval_result = await decision_point.evaluate_action(
            action=initial_response,
            principle_engine=self.principle_engine,
            task_handler=self.a2a_handler,
            message=message,
            agent_id=agent_id
        )
        
        # Record decision for auditing
        self.auditor.record_decision(
            decision_point_name=decision_point.name,
            component=decision_point.component,
            action_description=f"A2A response to {message.get('method')} request",
            context_summary={"agent_id": agent_id, "method": message.get("method")},
            evaluation_result=eval_result.to_dict(),
            was_modified=was_modified,
            modification_details={
                "original": initial_response.get("result", {}),
                "modified": modified_response.get("result", {})
            } if was_modified else None
        )
        
        return modified_response
    
    async def assign_task(self, task_id: str, task_type: str, candidates: List[str], priority: int) -> Dict[str, Any]:
        """
        Assign a task to an agent with principle-based evaluation.
        
        Args:
            task_id: ID of the task
            task_type: Type of the task
            candidates: List of candidate agent IDs
            priority: Task priority
            
        Returns:
            The assignment (potentially modified based on principles)
        """
        logger.info(f"Assigning task {task_id} of type {task_type}")
        
        # Generate initial assignment
        initial_assignment = {
            "task_id": task_id,
            "agent_id": candidates[0] if candidates else None,
            "candidate_agents": candidates,
            "priority": priority,
            "assignment_time": datetime.now(timezone.utc).isoformat()
        }
        
        # Evaluate against principles
        decision_point = self.decision_points["orchestrator_assignment"]
        modified_assignment, was_modified, eval_result = await decision_point.evaluate_action(
            action=initial_assignment,
            principle_engine=self.principle_engine,
            orchestrator=self.orchestrator,
            task_id=task_id,
            task_type=task_type,
            agent_candidates=candidates,
            priority=priority
        )
        
        # Record decision for auditing
        self.auditor.record_decision(
            decision_point_name=decision_point.name,
            component=decision_point.component,
            action_description=f"Task assignment for {task_id}",
            context_summary={"task_type": str(task_type), "candidates": candidates},
            evaluation_result=eval_result.to_dict(),
            was_modified=was_modified,
            modification_details={
                "original": initial_assignment,
                "modified": modified_assignment
            } if was_modified else None
        )
        
        return modified_assignment
    
    async def resolve_conflict(self, conflict_id: str, parties: List[str], conflict_type: str, severity: str) -> Dict[str, Any]:
        """
        Resolve a conflict with principle-based evaluation.
        
        Args:
            conflict_id: ID of the conflict
            parties: List of parties involved in the conflict
            conflict_type: Type of conflict
            severity: Conflict severity
            
        Returns:
            The resolution (potentially modified based on principles)
        """
        logger.info(f"Resolving conflict {conflict_id} between {', '.join(parties)}")
        
        # Generate initial resolution
        initial_resolution = {
            "conflict_id": conflict_id,
            "resolution_type": "mediation",
            "affected_parties": parties,
            "resolution_steps": [
                "Identify core issues",
                "Address primary concerns",
                "Establish new communication protocol"
            ],
            "message": "The conflict has been analyzed and a resolution strategy has been developed."
        }
        
        # Evaluate against principles
        decision_point = self.decision_points["conflict_resolution"]
        modified_resolution, was_modified, eval_result = await decision_point.evaluate_action(
            action=initial_resolution,
            principle_engine=self.principle_engine,
            resolver=self.conflict_resolver,
            conflict_id=conflict_id,
            parties=parties,
            conflict_type=conflict_type,
            severity=severity
        )
        
        # Record decision for auditing
        self.auditor.record_decision(
            decision_point_name=decision_point.name,
            component=decision_point.component,
            action_description=f"Conflict resolution for {conflict_id}",
            context_summary={"conflict_type": conflict_type, "parties": parties, "severity": severity},
            evaluation_result=eval_result.to_dict(),
            was_modified=was_modified,
            modification_details={
                "original": initial_resolution,
                "modified": modified_resolution
            } if was_modified else None
        )
        
        return modified_resolution
    
    async def select_agent(self, task_type: str, required_capabilities: List[str], preferred_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Select an agent for a task with principle-based evaluation.
        
        Args:
            task_type: Type of task requiring an agent
            required_capabilities: List of required capabilities
            preferred_type: Optional preferred agent type
            
        Returns:
            The selection (potentially modified based on principles)
        """
        logger.info(f"Selecting agent for task type {task_type}")
        
        # Generate initial selection
        suitable_agents = ["agent1", "agent2", "agent3"]  # Placeholder, would be determined by actual logic
        initial_selection = {
            "agent_id": suitable_agents[0] if suitable_agents else None,
            "alternative_agents": suitable_agents[1:] if len(suitable_agents) > 1 else [],
            "task_type": task_type,
            "capabilities_matched": required_capabilities,
            "selection_confidence": 0.85
        }
        
        # Evaluate against principles
        decision_point = self.decision_points["agent_selection"]
        modified_selection, was_modified, eval_result = await decision_point.evaluate_action(
            action=initial_selection,
            principle_engine=self.principle_engine,
            registry=self.agent_registry,
            task_type=task_type,
            required_capabilities=required_capabilities,
            preferred_agent_type=preferred_type
        )
        
        # Record decision for auditing
        self.auditor.record_decision(
            decision_point_name=decision_point.name,
            component=decision_point.component,
            action_description=f"Agent selection for {task_type}",
            context_summary={"required_capabilities": required_capabilities, "preferred_type": preferred_type},
            evaluation_result=eval_result.to_dict(),
            was_modified=was_modified,
            modification_details={
                "original": initial_selection,
                "modified": modified_selection
            } if was_modified else None
        )
        
        return modified_selection
    
    async def form_agent_request(self, target_agent_id: str, request_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Form a request to an agent with principle-based evaluation.
        
        Args:
            target_agent_id: ID of the target agent
            request_type: Type of request
            payload: Request payload
            
        Returns:
            The request (potentially modified based on principles)
        """
        logger.info(f"Forming {request_type} request to agent {target_agent_id}")
        
        # Generate initial request
        initial_request = {
            "target_agent_id": target_agent_id,
            "request_type": request_type,
            "payload": payload,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Evaluate against principles
        decision_point = self.decision_points["connector_request"]
        modified_request, was_modified, eval_result = await decision_point.evaluate_action(
            action=initial_request,
            principle_engine=self.principle_engine,
            connector_id="connector-123",
            target_agent_id=target_agent_id,
            request_type=request_type,
            payload=payload
        )
        
        # Record decision for auditing
        self.auditor.record_decision(
            decision_point_name=decision_point.name,
            component=decision_point.component,
            action_description=f"{request_type} request to {target_agent_id}",
            context_summary={"request_type": request_type, "payload_keys": list(payload.keys())},
            evaluation_result=eval_result.to_dict(),
            was_modified=was_modified,
            modification_details={
                "original": {k: v for k, v in initial_request.items() if k != "payload"},
                "modified": {k: v for k, v in modified_request.items() if k != "payload"}
            } if was_modified else None
        )
        
        return modified_request


# Example usage of the framework
async def main() -> None:
    # This would be a proper LLM adapter in a real implementation
    mock_llm_adapter = type('MockLLMAdapter', (BaseLLMAdapter,), {
        'send_request': lambda self, prompt, **kwargs: {"content": "Principle evaluation response"},
        'process_response': lambda self, response: response
    })()
    
    # Create the principle-enabled system
    system = PrincipleEnabledSystem(mock_llm_adapter)
    
    # Example: Process an A2A task request
    a2a_response = await system.handle_a2a_task_request(
        message={
            "jsonrpc": "2.0",
            "id": "req-456",
            "method": "processTask",
            "params": {
                "task": "data_analysis",
                "data": {"sources": ["source1", "source2"]}
            }
        },
        agent_id="agent-789"
    )
    print(f"A2A Task Response: {json.dumps(a2a_response, indent=2)}")
    
    # Example: Assign a task
    assignment = await system.assign_task(
        task_id="task-123",
        task_type="data_processing",
        candidates=["agent1", "agent2", "agent3"],
        priority=2
    )
    print(f"Task Assignment: {json.dumps(assignment, indent=2)}")
    
    # Example: Resolve a conflict
    resolution = await system.resolve_conflict(
        conflict_id="conflict-456",
        parties=["agent1", "agent2"],
        conflict_type="resource_contention",
        severity="medium"
    )
    print(f"Conflict Resolution: {json.dumps(resolution, indent=2)}")
    
    # Example: Select an agent
    selection = await system.select_agent(
        task_type="information_retrieval",
        required_capabilities=["web_search", "content_extraction", "summarization"],
        preferred_type="search_specialist"
    )
    print(f"Agent Selection: {json.dumps(selection, indent=2)}")
    
    # Example: Form an agent request
    request = await system.form_agent_request(
        target_agent_id="agent-456",
        request_type="data_query",
        payload={
            "query": "recent market trends",
            "filters": {"industry": "technology", "timeframe": "last_month"},
            "format": "summary"
        }
    )
    print(f"Agent Request: {json.dumps(request, indent=2)}")
    
    # Print audit statistics
    print(f"Principle Decision Audit: {json.dumps(system.auditor.get_statistics(), indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
