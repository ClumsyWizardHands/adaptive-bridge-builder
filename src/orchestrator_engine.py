"""
Orchestrator Engine

This module provides advanced coordination capabilities for managing complex tasks across
multiple specialized agents. It extends the collaborative task handling system with
sophisticated task decomposition, dependency management, scheduling, and error recovery.

The OrchestratorEngine enables the Adaptive Bridge Builder to efficiently distribute
work among specialized agents while maintaining the "Harmony Through Presence" principle.
"""

import json
import uuid
import time
import logging
import heapq
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union
from enum import Enum, auto
import threading
import copy
from dataclasses import dataclass

# Import related modules
from collaborative_task_handler import Task, TaskStatus, TaskPriority, TaskCoordinator
from communication_adapter import CommunicationAdapter
from content_handler import ContentHandler, ContentFormat
from principle_engine import PrincipleEngine
from agent_card import AgentCard
from emotional_intelligence import EmotionalIntelligence
from relationship_tracker import RelationshipTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("OrchestratorEngine")


class TaskType(Enum):
    """Types of tasks that can be orchestrated."""
    ANALYSIS = auto()          # Data or content analysis
    GENERATION = auto()        # Content generation
    TRANSFORMATION = auto()    # Content transformation or conversion
    EXTRACTION = auto()        # Information extraction
    VALIDATION = auto()        # Validation or verification
    AGGREGATION = auto()       # Combining multiple results
    DECISION = auto()          # Making or recommending decisions
    ORCHESTRATION = auto()     # Subtask delegation and coordination
    COMMUNICATION = auto()     # External or inter-agent communication
    RESEARCH = auto()          # Information gathering
    EXECUTION = auto()         # Executing actions or code
    MONITORING = auto()        # Observing processes or events
    RECOVERY = auto()          # Recovering from failures
    NEGOTIATION = auto()       # Coordinating between differing viewpoints
    OTHER = auto()             # Other task types


class AgentRole(Enum):
    """Common agent roles in multi-agent orchestration."""
    COORDINATOR = "coordinator"      # Oversees multi-agent interactions
    ANALYZER = "analyzer"            # Analyzes content or data
    GENERATOR = "generator"          # Generates content
    TRANSFORMER = "transformer"      # Transforms or converts content
    VALIDATOR = "validator"          # Validates outputs or processes
    COMMUNICATOR = "communicator"    # Handles external communication
    RESEARCHER = "researcher"        # Gathers information
    EXECUTOR = "executor"            # Executes actions or code
    MONITOR = "monitor"              # Observes and reports on processes
    SPECIALIST = "specialist"        # Domain-specific expert
    GENERALIST = "generalist"        # Broad capability agent


class AgentAvailability(Enum):
    """Availability states for agents."""
    AVAILABLE = "available"          # Fully available for task assignment
    BUSY = "busy"                    # Currently occupied but can accept future tasks
    LIMITED = "limited"              # Available with restrictions (time/resource)
    UNAVAILABLE = "unavailable"      # Not available for new tasks
    UNKNOWN = "unknown"              # Availability not determined


class DependencyType(Enum):
    """Types of dependencies between tasks."""
    SEQUENTIAL = "sequential"        # Task must complete entirely before dependent starts
    PARALLEL = "parallel"            # Tasks can run concurrently
    CONDITIONAL = "conditional"      # Dependency based on meeting specific conditions
    PARTIAL = "partial"              # Task can start with partial results from dependency
    ITERATIVE = "iterative"          # Dependent tasks provide feedback loop
    RESOURCE = "resource"            # Tasks compete for same resources


class TaskDecompositionStrategy(Enum):
    """Strategies for decomposing complex tasks."""
    FUNCTIONAL = "functional"        # Decompose by function/capability
    SEQUENTIAL = "sequential"        # Decompose into sequential steps
    PARALLEL = "parallel"            # Decompose into parallel components
    HIERARCHICAL = "hierarchical"    # Decompose into hierarchical structure
    DOMAIN = "domain"                # Decompose by domain/subject area
    CAPABILITY = "capability"        # Decompose by required agent capabilities
    BALANCED = "balanced"            # Balance load across available agents
    PRIORITY = "priority"            # Prioritize critical components first


@dataclass
class AgentProfile:
    """Information about an agent, its capabilities, and current status."""
    agent_id: str
    roles: List[AgentRole]
    capabilities: List[str]
    specialization: Dict[TaskType, float]  # Task type to proficiency (0.0-1.0)
    current_load: int = 0
    max_load: int = 5
    availability: AgentAvailability = AgentAvailability.AVAILABLE
    task_history: List[str] = None
    success_rate: Dict[TaskType, float] = None
    response_time: Dict[TaskType, float] = None  # Average response time in seconds
    last_active: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> Any:
        if self.task_history is None:
            self.task_history = []
        if self.success_rate is None:
            self.success_rate = {task_type: 1.0 for task_type in TaskType}
        if self.response_time is None:
            self.response_time = {task_type: 1.0 for task_type in TaskType}
        if self.metadata is None:
            self.metadata = {}

    def update_availability(self) -> AgentAvailability:
        """Update and return the agent's availability based on current load."""
        if self.current_load >= self.max_load:
            self.availability = AgentAvailability.BUSY
        elif self.current_load > 0:
            self.availability = AgentAvailability.LIMITED
        else:
            self.availability = AgentAvailability.AVAILABLE
        return self.availability

    def can_accept_task(self, task_type: TaskType) -> bool:
        """Check if the agent can accept a new task of the given type."""
        return (self.availability != AgentAvailability.UNAVAILABLE and 
                self.current_load < self.max_load and
                task_type in self.specialization)

    def get_suitability_score(self, task_type: TaskType, priority: TaskPriority) -> float:
        """
        Calculate how suitable this agent is for a task of the given type and priority.
        
        Returns:
            Suitability score (higher is better)
        """
        if not self.can_accept_task(task_type):
            return 0.0
            
        # Base score is specialization level
        score = self.specialization.get(task_type, 0.1)
        
        # Adjust for current load
        load_factor = 1.0 - (self.current_load / self.max_load)
        
        # Adjust for success rate
        success_factor = self.success_rate.get(task_type, 0.5)
        
        # Adjust for response time (faster is better)
        avg_time = self.response_time.get(task_type, 10.0)
        time_factor = 1.0 / (1.0 + avg_time/10.0)  # Normalize
        
        # Adjust for priority
        priority_multiplier = {
            TaskPriority.CRITICAL: 1.5,
            TaskPriority.HIGH: 1.2, 
            TaskPriority.MEDIUM: 1.0,
            TaskPriority.LOW: 0.8,
            TaskPriority.BACKGROUND: 0.6
        }.get(priority, 1.0)
        
        # Calculate final score
        final_score = score * load_factor * success_factor * time_factor * priority_multiplier
        
        return final_score


class DecomposedTask:
    """
    Represents a task that has been decomposed into subtasks based on a strategy.
    This serves as a container for the original task and its decomposed components.
    """
    
    def __init__(
        self,
        original_task: Task,
        strategy: TaskDecompositionStrategy,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a decomposed task container.
        
        Args:
            original_task: The original complex task before decomposition
            strategy: The decomposition strategy used
            metadata: Additional metadata about the decomposition
        """
        self.original_task_id = original_task.task_id
        self.original_title = original_task.title
        self.original_description = original_task.description
        self.strategy = strategy
        self.subtasks: Dict[str, Task] = {}  # task_id -> Task
        self.subtask_outputs: Dict[str, Any] = {}  # task_id -> output
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at
        self.status = TaskStatus.CREATED
        self.progress = 0.0
        self.metadata = metadata or {}
        
        # Dependency graph representation
        self.dependency_graph: Dict[str, List[Tuple[str, DependencyType]]] = {}
        
        # Rehydrate the original task
        self.original_task = original_task
        
        logger.info(f"Created decomposed task for {self.original_task_id} using {strategy.value} strategy")
    
    def add_subtask(self, subtask: Task, dependencies: Optional[List[Tuple[str, DependencyType]]] = None) -> None:
        """
        Add a subtask to the decomposition.
        
        Args:
            subtask: The subtask to add
            dependencies: List of (task_id, dependency_type) tuples
        """
        self.subtasks = {**self.subtasks, subtask.task_id: subtask}
        
        # Update dependency graph
        if dependencies:
            self.dependency_graph = {**self.dependency_graph, subtask.task_id: dependencies.copy()}
            
            # Also update the Task object's dependencies list
            subtask.dependencies = [dep_id for dep_id, _ in dependencies]
        else:
            self.dependency_graph = {**self.dependency_graph, subtask.task_id: []}
            
        self.updated_at = datetime.now(timezone.utc).isoformat()
    
    def update_progress(self) -> float:
        """
        Update the overall progress based on all subtasks.
        
        Returns:
            Current progress (0.0-1.0)
        """
        if not self.subtasks:
            self.progress = 0.0
            return self.progress
            
        # Calculate progress as average of all subtasks
        total_progress = sum(subtask.progress for subtask in self.subtasks.values())
        self.progress = total_progress / len(self.subtasks)
        
        # Update status based on progress and subtask statuses
        if all(subtask.status == TaskStatus.COMPLETED for subtask in self.subtasks.values()):
            self.status = TaskStatus.COMPLETED
        elif any(subtask.status == TaskStatus.FAILED for subtask in self.subtasks.values()):
            # Only consider the whole task failed if a non-recoverable subtask fails
            critical_failures = [
                subtask for subtask in self.subtasks.values()
                if subtask.status == TaskStatus.FAILED and 
                not subtask.metadata.get("recoverable", False)
            ]
            if critical_failures:
                self.status = TaskStatus.FAILED
        elif any(subtask.status in [TaskStatus.IN_PROGRESS, TaskStatus.ASSIGNED] 
                for subtask in self.subtasks.values()):
            self.status = TaskStatus.IN_PROGRESS
            
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return self.progress
    
    def get_ready_subtasks(self) -> List[Task]:
        """
        Get subtasks that are ready to be executed (dependencies satisfied).
        
        Returns:
            List of ready-to-run subtasks
        """
        ready_tasks = []
        
        for task_id, subtask in self.subtasks.items():
            # Skip tasks that are already in progress, completed, or failed
            if subtask.status not in [TaskStatus.CREATED, TaskStatus.ASSIGNED, TaskStatus.BLOCKED]:
                continue
                
            # Check dependencies from dependency graph
            dependencies = self.dependency_graph.get(task_id, [])
            can_start = True
            
            for dep_id, dep_type in dependencies:
                dep_task = self.subtasks.get(dep_id)
                if not dep_task:
                    logger.warning(f"Dependency {dep_id} for task {task_id} not found")
                    can_start = False
                    break
                    
                # Check based on dependency type
                if dep_type == DependencyType.SEQUENTIAL:
                    # Require complete dependency
                    if dep_task.status != TaskStatus.COMPLETED:
                        can_start = False
                        break
                elif dep_type == DependencyType.PARALLEL:
                    # Can start in parallel, no blocking
                    pass
                elif dep_type == DependencyType.CONDITIONAL:
                    # Check conditions in metadata
                    condition_met = False
                    conditions = subtask.metadata.get("dependency_conditions", {}).get(dep_id)
                    if conditions:
                        # Evaluate conditions based on dependency task state
                        try:
                            if conditions.get("status") and dep_task.status.value != conditions["status"]:
                                can_start = False
                                break
                            if conditions.get("min_progress") and dep_task.progress < conditions["min_progress"]:
                                can_start = False
                                break
                        except (KeyError, AttributeError) as e:
                            logger.error(f"Error evaluating conditions for {task_id}: {str(e)}")
                            can_start = False
                            break
                    else:
                        # No conditions defined, require completion
                        if dep_task.status != TaskStatus.COMPLETED:
                            can_start = False
                            break
                elif dep_type == DependencyType.PARTIAL:
                    # Check if partial results are available
                    if dep_task.status not in [TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED]:
                        can_start = False
                        break
                    if dep_task.progress < subtask.metadata.get("required_progress", 0.5):
                        can_start = False
                        break
                elif dep_type == DependencyType.ITERATIVE:
                    # Iterative dependencies just need to be started
                    if dep_task.status not in [TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED]:
                        can_start = False
                        break
                elif dep_type == DependencyType.RESOURCE:
                    # Resource dependencies - check if resource is available
                    resource_id = subtask.metadata.get("required_resource")
                    if resource_id and resource_id in self.metadata.get("busy_resources", []):
                        can_start = False
                        break
            
            if can_start:
                ready_tasks.append(subtask)
                
        return ready_tasks
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get all results from completed subtasks.
        
        Returns:
            Dictionary of task_id -> result
        """
        results = {}
        for task_id, subtask in self.subtasks.items():
            if subtask.status == TaskStatus.COMPLETED:
                for agent_id, result_info in subtask.results.items():
                    if task_id not in results:
                        results[task_id] = {}
                    results[task_id][agent_id] = result_info["data"]
                    
        return results
    
    def get_critical_path(self) -> List[str]:
        """
        Get the critical path of subtasks (longest dependency chain).
        
        Returns:
            List of task IDs representing the critical path
        """
        # Calculate earliest possible start time for each task
        earliest_start: Dict[str, int] = {}
        
        # Topological sort
        visited: Set[str] = set()
        temp_mark: Set[str] = set()
        order: List[str] = []
        
        def visit(task_id) -> None:
            if task_id in visited:
                return
            if task_id in temp_mark:
                # Cycle detected
                logger.warning(f"Cycle detected in dependency graph at task {task_id}")
                return
                
            temp_mark.add(task_id)
            
            for dep_id, _ in self.dependency_graph.get(task_id, []):
                if dep_id in self.subtasks:
                    visit(dep_id)
                    
            temp_mark.remove(task_id)
            visited.add(task_id)
            order.append(task_id)
            
        # Perform topological sort
        for task_id in self.subtasks:
            if task_id not in visited:
                visit(task_id)
                
        # Calculate earliest start and finish times
        earliest_start = {task_id: 0 for task_id in self.subtasks}
        earliest_finish = {task_id: 0 for task_id in self.subtasks}
        
        for task_id in reversed(order):
            est = 0
            for dep_id, dep_type in self.dependency_graph.get(task_id, []):
                if dep_id in self.subtasks and dep_type in [DependencyType.SEQUENTIAL, DependencyType.CONDITIONAL]:
                    est = max(est, earliest_finish.get(dep_id, 0))
                    
            earliest_start[task_id] = est
            
            # Estimate task duration (could be more sophisticated)
            task_duration = self.subtasks[task_id].metadata.get("estimated_duration", 1)
            earliest_finish[task_id] = est + task_duration
            
        # Calculate latest start and finish times
        latest_finish = {}
        latest_start = {}
        
        # Find the maximum finish time
        max_finish = max(earliest_finish.values()) if earliest_finish else 0
        
        # Initialize latest finish for all tasks to the max finish time
        for task_id in self.subtasks:
            latest_finish[task_id] = max_finish
            
        # Calculate latest start time for each task
        for task_id in order:
            task_duration = self.subtasks[task_id].metadata.get("estimated_duration", 1)
            
            # Latest start time
            lst = latest_finish[task_id] - task_duration
            latest_start[task_id] = lst
            
            # Update latest finish time for dependencies
            for dep_id, dep_type in self.dependency_graph.get(task_id, []):
                if (dep_id in self.subtasks and 
                    dep_type in [DependencyType.SEQUENTIAL, DependencyType.CONDITIONAL]):
                    latest_finish[dep_id] = min(latest_finish[dep_id], lst)
                    
        # Tasks with zero slack (latest start - earliest start) are on critical path
        critical_path = []
        for task_id in self.subtasks:
            if round(latest_start[task_id] - earliest_start[task_id], 2) == 0:
                critical_path.append(task_id)
                
        # Sort critical path by earliest start time
        critical_path.sort(key=lambda task_id: earliest_start[task_id])
        
        return critical_path
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary representation of this decomposed task
        """
        return {
            "original_task_id": self.original_task_id,
            "original_title": self.original_title,
            "original_description": self.original_description,
            "strategy": self.strategy.value,
            "subtasks": {task_id: task.to_dict() for task_id, task in self.subtasks.items()},
            "dependency_graph": {
                task_id: [(dep_id, dep_type.value) for dep_id, dep_type in deps]
                for task_id, deps in self.dependency_graph.items()
            },
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status.value,
            "progress": self.progress,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], task_coordinator: 'TaskCoordinator') -> 'DecomposedTask':
        """
        Create from dictionary representation.
        
        Args:
            data: Dictionary data
            task_coordinator: TaskCoordinator to retrieve original and subtasks
            
        Returns:
            Reconstructed DecomposedTask
        """
        # Get original task
        original_task = task_coordinator.get_task(data["original_task_id"])
        if not original_task:
            raise ValueError(f"Original task {data['original_task_id']} not found")
            
        # Create decomposed task
        strategy = TaskDecompositionStrategy(data["strategy"])
        decomposed = cls(
            original_task=original_task,
            strategy=strategy,
            metadata=data.get("metadata", {})
        )
        
        # Set timestamps and status
        decomposed.created_at = data["created_at"]
        decomposed.updated_at = data["updated_at"]
        decomposed.status = TaskStatus(data["status"])
        decomposed.progress = data["progress"]
        
        # Get subtasks and rebuild dependency graph
        for task_id, task_data in data["subtasks"].items():
            subtask = task_coordinator.get_task(task_id)
            if subtask:
                decomposed.subtasks[task_id] = subtask
                
        # Rebuild dependency graph
        for task_id, deps in data["dependency_graph"].items():
            decomposed.dependency_graph[task_id] = [
                (dep_id, DependencyType(dep_type)) for dep_id, dep_type in deps
            ]
            
        return decomposed


class OrchestratorEngine:
    """
    Advanced orchestration engine for coordinating tasks across multiple specialized agents.
    
    The OrchestratorEngine extends TaskCoordinator with advanced task decomposition,
    dependency management, agent selection, scheduling, and error recovery capabilities.
    It implements the "Harmony Through Presence" principle by ensuring balanced workload
    distribution, active status updates, and responsive orchestration.
    """
    
    def __init__(
        self,
        agent_id: str,
        communication_adapter: Optional[CommunicationAdapter] = None,
        content_handler: Optional[ContentHandler] = None,
        principle_engine: Optional[PrincipleEngine] = None,
        emotional_intelligence: Optional[EmotionalIntelligence] = None,
        relationship_tracker: Optional[RelationshipTracker] = None,
        storage_dir: str = "data/orchestration"
    ):
        """
        Initialize the orchestrator engine.
        
        Args:
            agent_id: ID of the orchestrator agent
            communication_adapter: Adapter for agent communication
            content_handler: Handler for content format conversion
            principle_engine: Engine for principle-based reasoning
            emotional_intelligence: Module for emotional intelligence
            relationship_tracker: Tracker for agent relationships
            storage_dir: Directory for storing orchestration data
        """
        self.agent_id = agent_id
        self.communication_adapter = communication_adapter
        self.content_handler = content_handler
        self.principle_engine = principle_engine
        self.emotional_intelligence = emotional_intelligence
        self.relationship_tracker = relationship_tracker
        self.storage_dir = storage_dir
        
        # Create underlying task coordinator
        self.task_coordinator = TaskCoordinator(
            agent_id=agent_id,
            communication_adapter=communication_adapter,
            content_handler=content_handler,
            storage_dir=storage_dir
        )
        
        # Agent profiles and availability
        self.agent_profiles: Dict[str, AgentProfile] = {}
        
        # Decomposed tasks
        self.decomposed_tasks: Dict[str, DecomposedTask] = {}
        
        # Task queues for scheduling
        self.task_queues: Dict[TaskPriority, List[Tuple[float, str]]] = {
            priority: [] for priority in TaskPriority
        }
        
        # Scheduled tasks
        self.scheduled_tasks: Dict[str, datetime] = {}
        
        # Running tasks and their agents
        self.running_tasks: Dict[str, List[str]] = {}
        
        # Recovery strategies for failed tasks
        self.recovery_strategies: Dict[str, Dict[str, Any]] = {}
        
        # Locks
        self.agent_lock = threading.Lock()
        self.decomposition_lock = threading.Lock()
        self.schedule_lock = threading.Lock()
        self.task_lock = threading.Lock()
        
        # Scheduling thread
        self.scheduler_thread = None
        self.scheduler_running = False
        
        # Error tracking and pattern recognition
        self.error_patterns: Dict[str, Dict[str, Any]] = {}
        
        # For the Harmony Through Presence principle
        self.last_status_update = datetime.now(timezone.utc)
        self.status_update_frequency = timedelta(seconds=30)
        self.harmony_metrics = {
            "workload_distribution": 0.0,  # 0.0 (imbalanced) to 1.0 (perfectly balanced)
            "communication_responsiveness": 0.0,  # 0.0 (unresponsive) to 1.0 (highly responsive)
            "principle_alignment": 0.0,  # 0.0 (misaligned) to 1.0 (fully aligned)
            "error_recovery_rate": 0.0,  # 0.0 (poor) to 1.0 (excellent)
            "task_completion_rate": 0.0  # 0.0 (low) to 1.0 (high)
        }
        
        logger.info(f"OrchestratorEngine initialized for agent {agent_id}")
    
    def register_agent(
        self,
        agent_id: str,
        roles: List[AgentRole],
        capabilities: List[str],
        specialization: Dict[TaskType, float],
        max_load: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register an agent with the orchestrator.
        
        Args:
            agent_id: ID of the agent
            roles: Roles the agent can fulfill
            capabilities: Specific capabilities the agent has
            specialization: Mapping of task types to proficiency (0.0-1.0)
            max_load: Maximum concurrent tasks the agent can handle
            metadata: Additional metadata about the agent
        """
        with self.agent_lock:
            profile = AgentProfile(
                agent_id=agent_id,
                roles=roles,
                capabilities=capabilities,
                specialization=specialization,
                max_load=max_load,
                metadata=metadata or {}
            )
            self.agent_profiles = {**self.agent_profiles, agent_id: profile}
            
            # Update task coordinator agent capabilities
            self.task_coordinator.update_agent_capabilities(agent_id, capabilities)
            
            logger.info(f"Registered agent {agent_id} with {len(capabilities)} capabilities")
    
    def update_agent_status(
        self,
        agent_id: str,
        current_load: Optional[int] = None,
        availability: Optional[AgentAvailability] = None,
        success_rate: Optional[Dict[TaskType, float]] = None,
        response_time: Optional[Dict[TaskType, float]] = None
    ) -> bool:
        """
        Update an agent's status information.
        
        Args:
            agent_id: ID of the agent
            current_load: Current task load
            availability: Current availability
            success_rate: Updated success rates by task type
            response_time: Updated response times by task type
            
        Returns:
            Whether the update was successful
        """
        with self.agent_lock:
            profile = self.agent_profiles.get(agent_id)
            if not profile:
                logger.error(f"Agent {agent_id} not found for status update")
                return False
                
            # Update fields if provided
            if current_load is not None:
                profile.current_load = current_load
                
            if availability is not None:
                profile.availability = availability
            else:
                profile.update_availability()
                
            if success_rate:
                for task_type, rate in success_rate.items():
                    profile.success_rate[task_type] = rate
                    
            if response_time:
                for task_type, time_value in response_time.items():
                    profile.response_time[task_type] = time_value
                    
            profile.last_active = datetime.now(timezone.utc).isoformat()
            
            return True
    
    def decompose_task(
        self,
        task: Task,
        strategy: TaskDecompositionStrategy,
        agent_ids: Optional[List[str]] = None,
        task_types: Optional[Dict[str, TaskType]] = None,
        estimated_durations: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[DecomposedTask]:
        """
        Decompose a complex task into subtasks based on the specified strategy.
        
        Args:
            task: The task to decompose
            strategy: Strategy to use for decomposition
            agent_ids: Optional list of agent IDs to consider
            task_types: Optional mapping of subtask IDs to task types
            estimated_durations: Optional mapping of subtask IDs to estimated durations
            metadata: Additional metadata for the decomposition
            
        Returns:
            The decomposed task or None if decomposition failed
        """
        with self.decomposition_lock:
            # Create decomposed task container
            decomposed = DecomposedTask(
                original_task=task,
                strategy=strategy,
                metadata=metadata or {}
            )
            
            # Apply decomposition strategy
            if strategy == TaskDecompositionStrategy.FUNCTIONAL:
                success = self._decompose_by_function(decomposed, agent_ids, task_types)
            elif strategy == TaskDecompositionStrategy.SEQUENTIAL:
                success = self._decompose_sequentially(decomposed, agent_ids, task_types)
            elif strategy == TaskDecompositionStrategy.PARALLEL:
                success = self._decompose_in_parallel(decomposed, agent_ids, task_types)
            elif strategy == TaskDecompositionStrategy.HIERARCHICAL:
                success = self._decompose_hierarchically(decomposed, agent_ids, task_types)
            elif strategy == TaskDecompositionStrategy.DOMAIN:
                success = self._decompose_by_domain(decomposed, agent_ids, task_types)
            elif strategy == TaskDecompositionStrategy.CAPABILITY:
                success = self._decompose_by_capability(decomposed, agent_ids, task_types)
            elif strategy == TaskDecompositionStrategy.BALANCED:
                success = self._decompose_for_balance(decomposed, agent_ids, task_types)
            elif strategy == TaskDecompositionStrategy.PRIORITY:
                success = self._decompose_by_priority
