"""
Project Orchestrator Extension

This module extends the OrchestratorEngine with advanced project management capabilities
for handling complex, multi-stage projects requiring various agent capabilities.

The ProjectOrchestrator implements sophisticated project planning with milestones,
dependencies, and timelines, allocates tasks to appropriate agents, tracks progress,
adjusts plans when issues arise, and provides regular status updates to stakeholders.
It embodies the "Practicality Aligned with Conscience" principle in managing resources.
"""

import json
import uuid
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from enum import Enum, auto
import threading
import copy
from dataclasses import dataclass, field

# Import related modules
from orchestrator_engine import (
    OrchestratorEngine, TaskType, AgentRole, AgentAvailability, 
    DependencyType, TaskDecompositionStrategy, DecomposedTask, AgentProfile
)
from collaborative_task_handler import Task, TaskStatus, TaskPriority
from principle_engine import PrincipleEngine
from communication_adapter import CommunicationAdapter
from relationship_tracker import RelationshipTracker
from content_handler import ContentHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ProjectOrchestrator")


class MilestoneStatus(Enum):
    """Status of project milestones."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    DELAYED = "delayed"
    AT_RISK = "at_risk"
    BLOCKED = "blocked"
    CANCELED = "canceled"


class ResourceType(Enum):
    """Types of resources that can be allocated to projects."""
    AGENT = "agent"              # AI agent resource
    COMPUTE = "compute"          # Computational resource
    STORAGE = "storage"          # Data storage resource
    API_ACCESS = "api_access"    # External API access
    DATABASE = "database"        # Database access
    HUMAN = "human"              # Human resource (for review, input, etc.)
    TIME = "time"                # Time allocation
    CUSTOM = "custom"            # Custom resource type


@dataclass
class Resource:
    """Represents a resource that can be allocated to project tasks."""
    resource_id: str
    resource_type: ResourceType
    name: str
    capacity: float = 1.0                # Total capacity (1.0 = 100%)
    allocated: float = 0.0               # Currently allocated capacity
    cost_per_unit: float = 0.0           # Cost per unit of resource
    tags: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def available_capacity(self) -> float:
        """Get the available (unallocated) capacity of the resource."""
        return max(0.0, self.capacity - self.allocated)
    
    def utilization_percentage(self) -> float:
        """Get the current utilization percentage of the resource."""
        if self.capacity <= 0:
            return 100.0
        return (self.allocated / self.capacity) * 100.0
    
    def can_allocate(self, amount: float) -> bool:
        """Check if the requested amount can be allocated."""
        return self.available_capacity() >= amount
    
    def allocate(self, amount: float) -> bool:
        """
        Allocate the specified amount of the resource.
        
        Args:
            amount: Amount to allocate
            
        Returns:
            True if allocation was successful, False otherwise
        """
        if not self.can_allocate(amount):
            return False
            
        self.allocated += amount
        return True
    
    def release(self, amount: float) -> bool:
        """
        Release the specified amount of the resource.
        
        Args:
            amount: Amount to release
            
        Returns:
            True if release was successful, False otherwise
        """
        if amount > self.allocated:
            return False
            
        self.allocated -= amount
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type.value,
            "name": self.name,
            "capacity": self.capacity,
            "allocated": self.allocated,
            "cost_per_unit": self.cost_per_unit,
            "tags": self.tags,
            "constraints": self.constraints,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Resource':
        """Create from dictionary representation."""
        return cls(
            resource_id=data["resource_id"],
            resource_type=ResourceType(data["resource_type"]),
            name=data["name"],
            capacity=data["capacity"],
            allocated=data["allocated"],
            cost_per_unit=data.get("cost_per_unit", 0.0),
            tags=data.get("tags", []),
            constraints=data.get("constraints", {}),
            metadata=data.get("metadata", {})
        )


@dataclass
class Milestone:
    """Represents a project milestone with associated tasks and dependencies."""
    milestone_id: str
    name: str
    description: str
    target_date: str  # ISO format date
    status: MilestoneStatus = MilestoneStatus.NOT_STARTED
    task_ids: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Other milestone IDs
    progress: float = 0.0  # 0.0 to 1.0
    completion_criteria: Dict[str, Any] = field(default_factory=dict)
    stakeholders: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "milestone_id": self.milestone_id,
            "name": self.name,
            "description": self.description,
            "target_date": self.target_date,
            "status": self.status.value,
            "task_ids": self.task_ids,
            "dependencies": self.dependencies,
            "progress": self.progress,
            "completion_criteria": self.completion_criteria,
            "stakeholders": self.stakeholders,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Milestone':
        """Create from dictionary representation."""
        return cls(
            milestone_id=data["milestone_id"],
            name=data["name"],
            description=data["description"],
            target_date=data["target_date"],
            status=MilestoneStatus(data["status"]),
            task_ids=data.get("task_ids", []),
            dependencies=data.get("dependencies", []),
            progress=data.get("progress", 0.0),
            completion_criteria=data.get("completion_criteria", {}),
            stakeholders=data.get("stakeholders", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class ScheduleEvent:
    """Represents a scheduled event in the project timeline."""
    event_id: str
    title: str
    description: str
    event_type: str  # "milestone", "task", "meeting", "review", etc.
    start_time: str  # ISO format date-time
    end_time: str  # ISO format date-time
    related_ids: List[str] = field(default_factory=list)  # Task/milestone IDs
    participants: List[str] = field(default_factory=list)  # Agent/stakeholder IDs
    status: str = "scheduled"  # scheduled, in_progress, completed, canceled
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_id": self.event_id,
            "title": self.title,
            "description": self.description,
            "event_type": self.event_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "related_ids": self.related_ids,
            "participants": self.participants,
            "status": self.status,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScheduleEvent':
        """Create from dictionary representation."""
        return cls(
            event_id=data["event_id"],
            title=data["title"],
            description=data["description"],
            event_type=data["event_type"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            related_ids=data.get("related_ids", []),
            participants=data.get("participants", []),
            status=data.get("status", "scheduled"),
            metadata=data.get("metadata", {})
        )


@dataclass
class ProjectIssue:
    """Represents an issue or risk identified during project execution."""
    issue_id: str
    title: str
    description: str
    severity: str  # "critical", "high", "medium", "low"
    status: str  # "open", "in_progress", "resolved", "closed"
    created_at: str  # ISO format date-time
    updated_at: str  # ISO format date-time
    related_ids: List[str] = field(default_factory=list)  # Task/milestone IDs
    assigned_to: List[str] = field(default_factory=list)  # Agent/stakeholder IDs
    resolution_plan: Optional[str] = None
    resolved_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "issue_id": self.issue_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "related_ids": self.related_ids,
            "assigned_to": self.assigned_to,
            "resolution_plan": self.resolution_plan,
            "resolved_at": self.resolved_at,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectIssue':
        """Create from dictionary representation."""
        return cls(
            issue_id=data["issue_id"],
            title=data["title"],
            description=data["description"],
            severity=data["severity"],
            status=data["status"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            related_ids=data.get("related_ids", []),
            assigned_to=data.get("assigned_to", []),
            resolution_plan=data.get("resolution_plan"),
            resolved_at=data.get("resolved_at"),
            metadata=data.get("metadata", {})
        )


@dataclass
class StatusUpdate:
    """Represents a project status update for stakeholders."""
    update_id: str
    title: str
    summary: str
    created_at: str  # ISO format date-time
    update_type: str  # "regular", "milestone", "issue", "risk"
    project_health: str  # "on_track", "at_risk", "delayed", "blocked"
    accomplishments: List[Dict[str, Any]] = field(default_factory=list)
    current_focus: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    next_steps: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    stakeholders: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "update_id": self.update_id,
            "title": self.title,
            "summary": self.summary,
            "created_at": self.created_at,
            "update_type": self.update_type,
            "project_health": self.project_health,
            "accomplishments": self.accomplishments,
            "current_focus": self.current_focus,
            "issues": self.issues,
            "next_steps": self.next_steps,
            "metrics": self.metrics,
            "stakeholders": self.stakeholders,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StatusUpdate':
        """Create from dictionary representation."""
        return cls(
            update_id=data["update_id"],
            title=data["title"],
            summary=data["summary"],
            created_at=data["created_at"],
            update_type=data["update_type"],
            project_health=data["project_health"],
            accomplishments=data.get("accomplishments", []),
            current_focus=data.get("current_focus", []),
            issues=data.get("issues", []),
            next_steps=data.get("next_steps", []),
            metrics=data.get("metrics", {}),
            stakeholders=data.get("stakeholders", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class Project:
    """
    Represents a complex project with milestones, tasks, resources, and timeline.
    """
    project_id: str
    name: str
    description: str
    created_at: str  # ISO format date-time
    updated_at: str  # ISO format date-time
    start_date: str  # ISO format date
    end_date: str  # ISO format date
    status: str = "planning"  # planning, active, on_hold, completed, canceled
    owner_id: Optional[str] = None
    milestones: Dict[str, Milestone] = field(default_factory=dict)
    decomposed_tasks: Dict[str, str] = field(default_factory=dict)  # Map of task_id -> decomposed_task_id
    resources: Dict[str, Resource] = field(default_factory=dict)
    schedule: Dict[str, ScheduleEvent] = field(default_factory=dict)
    issues: Dict[str, ProjectIssue] = field(default_factory=dict)
    status_updates: Dict[str, StatusUpdate] = field(default_factory=dict)
    stakeholders: List[Dict[str, Any]] = field(default_factory=list)
    completion_percentage: float = 0.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_milestone(self, milestone: Milestone) -> None:
        """Add a milestone to the project."""
        self.milestones[milestone.milestone_id] = milestone
        self.updated_at = datetime.utcnow().isoformat()
    
    def update_progress(self) -> float:
        """
        Update the overall project progress based on milestone progress.
        
        Returns:
            Updated project completion percentage (0.0-1.0)
        """
        if not self.milestones:
            self.completion_percentage = 0.0
            return self.completion_percentage
        
        # Calculate weighted progress
        total_progress = 0.0
        num_milestones = len(self.milestones)
        
        for milestone in self.milestones.values():
            # For simplicity, each milestone has equal weight
            # Could be enhanced with milestone weight property
            total_progress += milestone.progress
        
        self.completion_percentage = total_progress / num_milestones
        self.updated_at = datetime.utcnow().isoformat()
        
        return self.completion_percentage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "project_id": self.project_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "status": self.status,
            "owner_id": self.owner_id,
            "milestones": {
                milestone_id: milestone.to_dict() 
                for milestone_id, milestone in self.milestones.items()
            },
            "decomposed_tasks": self.decomposed_tasks,
            "resources": {
                resource_id: resource.to_dict() 
                for resource_id, resource in self.resources.items()
            },
            "schedule": {
                event_id: event.to_dict() 
                for event_id, event in self.schedule.items()
            },
            "issues": {
                issue_id: issue.to_dict() 
                for issue_id, issue in self.issues.items()
            },
            "status_updates": {
                update_id: update.to_dict() 
                for update_id, update in self.status_updates.items()
            },
            "stakeholders": self.stakeholders,
            "completion_percentage": self.completion_percentage,
            "tags": self.tags,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Project':
        """Create from dictionary representation."""
        project = cls(
            project_id=data["project_id"],
            name=data["name"],
            description=data["description"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            start_date=data["start_date"],
            end_date=data["end_date"],
            status=data["status"],
            owner_id=data.get("owner_id"),
            decomposed_tasks=data.get("decomposed_tasks", {}),
            stakeholders=data.get("stakeholders", []),
            completion_percentage=data.get("completion_percentage", 0.0),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )
        
        # Load complex nested objects
        for milestone_id, milestone_data in data.get("milestones", {}).items():
            project.milestones[milestone_id] = Milestone.from_dict(milestone_data)
            
        for resource_id, resource_data in data.get("resources", {}).items():
            project.resources[resource_id] = Resource.from_dict(resource_data)
            
        for event_id, event_data in data.get("schedule", {}).items():
            project.schedule[event_id] = ScheduleEvent.from_dict(event_data)
            
        for issue_id, issue_data in data.get("issues", {}).items():
            project.issues[issue_id] = ProjectIssue.from_dict(issue_data)
            
        for update_id, update_data in data.get("status_updates", {}).items():
            project.status_updates[update_id] = StatusUpdate.from_dict(update_data)
            
        return project


class ResourceConflictStrategy(Enum):
    """Strategies for resolving resource conflicts."""
    PRIORITIZE = "prioritize"        # Prioritize higher priority tasks
    RESCHEDULE = "reschedule"        # Reschedule lower priority tasks
    SPLIT = "split"                  # Split resource between tasks
    SUBSTITUTE = "substitute"        # Find substitute resource
    REDUCE = "reduce"                # Reduce allocation to fit constraints
    NEGOTIATE = "negotiate"          # Negotiate with stakeholders
    ESCALATE = "escalate"            # Escalate to project owner


class ProjectOrchestrator:
    """
    Advanced project orchestration extension that manages complex, multi-stage projects
    requiring various agent capabilities.
    
    The ProjectOrchestrator extends OrchestratorEngine with project management capabilities
    including project planning with milestones and timelines, resource management,
    stakeholder communications, and adaptive planning when issues arise.
    """
    
    def __init__(
        self,
        agent_id: str,
        orchestrator_engine: Optional[OrchestratorEngine] = None,
        communication_adapter: Optional[CommunicationAdapter] = None,
        content_handler: Optional[ContentHandler] = None,
        principle_engine: Optional[PrincipleEngine] = None,
        relationship_tracker: Optional[RelationshipTracker] = None,
        storage_dir: str = "data/projects"
    ):
        """
        Initialize the project orchestrator.
        
        Args:
            agent_id: ID of the orchestrator agent
            orchestrator_engine: Existing OrchestratorEngine or None to create new
            communication_adapter: Adapter for agent communication
            content_handler: Handler for content format conversion
            principle_engine: Engine for principle-based reasoning
            relationship_tracker: Tracker for agent relationships
            storage_dir: Directory for storing project data
        """
        self.agent_id = agent_id
        self.storage_dir = storage_dir
        self.communication_adapter = communication_adapter
        self.content_handler = content_handler
        self.principle_engine = principle_engine
        self.relationship_tracker = relationship_tracker
        
        # Create or use existing orchestrator engine
        if orchestrator_engine:
            self.orchestrator_engine = orchestrator_engine
        else:
            self.orchestrator_engine = OrchestratorEngine(
                agent_id=agent_id,
                communication_adapter=communication_adapter,
                content_handler=content_handler,
                principle_engine=principle_engine,
                relationship_tracker=relationship_tracker,
                storage_dir=f"{storage_dir}/orchestration"
            )
        
        # Projects
        self.projects: Dict[str, Project] = {}
        
        # Resource pool (shared across projects)
        self.global_resources: Dict[str, Resource] = {}
        
        # Status update schedules
        self.status_update_schedules: Dict[str, Dict[str, Any]] = {}
        
        # Locks
        self.project_lock = threading.Lock()
        self.resource_lock = threading.Lock()
        
        # Schedule management
        self.schedule_thread = None
        self.schedule_running = False
        
        # Metrics tracking
        self.project_metrics: Dict[str, Dict[str, Any]] = {}
        
        # The practicality aligned with conscience principle implementation
        self.conscience_metrics = {
            "resource_efficiency": 0.0,  # 0.0 (inefficient) to 1.0 (optimal)
            "ethical_resource_use": 0.0,  # 0.0 (concerning) to 1.0 (exemplary)
            "stakeholder_welfare": 0.0,  # 0.0 (neglected) to 1.0 (prioritized)
            "sustainability": 0.0,  # 0.0 (unsustainable) to 1.0 (sustainable)
            "balance_score": 0.0  # 0.0 (imbalanced) to 1.0 (well balanced)
        }
        
        logger.info(f"ProjectOrchestrator initialized for agent {agent_id}")
    
    def create_project(
        self,
        name: str,
        description: str,
        start_date: str,
        end_date: str,
        owner_id: Optional[str] = None,
        stakeholders: Optional[List[Dict[str, Any]]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Project:
        """
        Create a new project.
        
        Args:
            name: Project name
            description: Project description
            start_date: Project start date (ISO format date)
            end_date: Project end date (ISO format date)
            owner_id: ID of the project owner
            stakeholders: List of stakeholder information
            tags: List of project tags
            metadata: Additional metadata about the project
            
        Returns:
            The created project
        """
        project_id = f"project-{str(uuid.uuid4())}"
        now = datetime.utcnow().isoformat()
        
        with self.project_lock:
            project = Project(
                project_id=project_id,
                name=name,
                description=description,
                created_at=now,
                updated_at=now,
                start_date=start_date,
                end_date=end_date,
                status="planning",
                owner_id=owner_id or self.agent_id,
                stakeholders=stakeholders or [],
                tags=tags or [],
                metadata=metadata or {}
            )
            
            self.projects[project_id] = project
            
            # Initialize project metrics
            self.project_metrics[project_id] = {
                "created_at": now,
                "milestone_completion_rate": 0.0,
                "resource_utilization": 0.0,
                "issue_resolution_rate": 0.0,
                "schedule_adherence": 0.0,
                "stakeholder_satisfaction": 0.0,
                "conscience_alignment": 0.0,
                "task_completion_trend": [],
                "resource_efficiency_trend": []
            }
            
            logger.info(f"Created project '{name}' with ID {project_id}")
            
            return project
    
    def add_project_milestone(
        self,
        project_id: str,
        name: str,
        description: str,
        target_date: str,
        dependencies: Optional[List[str]] = None,
        completion_criteria: Optional[Dict[str, Any]] = None,
        stakeholders: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Milestone]:
        """
        Add a milestone to a project.
        
        Args:
            project_id: Project ID
            name: Milestone name
            description: Milestone description
            target_date: Target completion date (ISO format date)
            dependencies: List of dependent milestone IDs
            completion_criteria: Criteria for milestone completion
            stakeholders: List of stakeholder IDs interested in this milestone
            metadata: Additional metadata about the milestone
            
        Returns:
            The created milestone, or None if project not found
        """
        with self.project_lock:
            project = self.projects.get(project_id)
            if not project:
                logger.error(f"Project {project_id} not found")
                return None
            
            milestone_id = f"milestone-{str(uuid.uuid4())}"
            milestone = Milestone(
                milestone_id=milestone_id,
                name=name,
                description=description,
                target_date=target_date,
                status=MilestoneStatus.NOT_STARTED,
                dependencies=dependencies or [],
                completion_criteria=completion_criteria or {},
                stakeholders=stakeholders or [],
                metadata=metadata or {}
            )
            
            project.add_milestone(milestone)
            logger.info(f"Added milestone '{name}' to project {project_id}")
            
            return milestone
    
    def create_project_task(
        self,
        project_id: str,
        milestone_id: str,
        title: str,
        description: str,
        task_type: TaskType,
        required_capabilities: List[str],
        priority: TaskPriority = TaskPriority.MEDIUM,
        dependencies: Optional[List[str]] = None,
        estimated_duration: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Task]:
        """
        Create a task within a project milestone.
        
        Args:
            project_id: Project ID
            milestone_id: Milestone ID
            title: Task title
            description: Task description
            task_type: Type of task
            required_capabilities: List of required capabilities
            priority: Task priority
            dependencies: List of dependent task IDs
            estimated_duration: Estimated duration in minutes
            metadata: Additional metadata about the task
            
        Returns:
            The created task, or None if project or milestone not found
        """
        with self.project_lock:
            project = self.projects.get(project_id)
            if not project:
                logger.error(f"Project {project_id} not found")
                return None
            
            milestone = project.milestones.get(milestone_id)
            if not milestone:
                logger.error(f"Milestone {milestone_id} not found in project {project_id}")
                return None
            
            # Create task using orchestrator's task coordinator
            task = self.orchestrator_engine.task_coordinator.create_task(
                title=title,
                description=description,
                required_capabilities=required_capabilities,
                priority=priority,
                dependencies=dependencies or [],
                metadata={
                    **(metadata or {}),
                    "project_id": project_id,
                    "milestone_id": milestone_id,
                    "task_type": task_type.name,
                    "estimated_duration": estimated_duration
                }
            )
            
            # Add task to milestone
            milestone.task_ids.append(task.task_id)
            project.updated_at = datetime.utcnow().isoformat()
            
            logger.info(f"Created task '{title}' in milestone {milestone_id} of project {project_id}")
            
            return task
    
    def decompose_project_tasks(
        self,
        project_id: str,
        milestone_id: Optional[str] = None,
        strategy: TaskDecompositionStrategy = TaskDecompositionStrategy.FUNCTIONAL
    ) -> Dict[str, str]:
        """
        Decompose tasks within a project or specific milestone into subtasks.
        
        Args:
            project_id: Project ID
            milestone_id: Optional milestone ID (if None, decompose all project tasks)
            strategy: Strategy to use for decomposition
            
        Returns:
            Dictionary mapping original task IDs to decomposed task IDs
        """
        with self.project_lock:
            project = self.projects.get(project_id)
            if not project:
                logger.error(f"Project {project_id} not found")
                return {}
            
            decomposed_map = {}
            
            if milestone_id:
                # Decompose tasks for specific milestone
                milestone = project.milestones.get(milestone_id)
                if not milestone:
                    logger.error(f"Milestone {milestone_id} not found in project {project_id}")
                    return {}
                
                task_ids = milestone.task_ids
            else:
                # Decompose all tasks across all milestones
                task_ids = []
                for milestone in project.milestones.values():
                    task_ids.extend(milestone.task_ids)
            
            # Decompose each task
            for task_id in task_ids:
                task = self.orchestrator_engine.task_coordinator.get_task(task_id)
                if not task:
                    logger.warning(f"Task {task_id} not found, skipping decomposition")
                    continue
