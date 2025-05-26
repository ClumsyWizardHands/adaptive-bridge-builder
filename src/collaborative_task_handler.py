"""
Collaborative Task Handler

This module provides functionality for coordinating collaborative tasks among multiple agents.
It manages task delegation, result aggregation, and inter-agent communication for complex
multi-step workflows.
"""

import json
import uuid
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from enum import Enum
import threading

# Import related modules
from communication_adapter import CommunicationAdapter
from content_handler import ContentHandler, ContentFormat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CollaborativeTaskHandler")


class TaskStatus(Enum):
    """Status states for collaborative tasks."""
    CREATED = "created"              # Task has been created but not started
    ASSIGNED = "assigned"            # Task has been assigned to agents
    IN_PROGRESS = "in_progress"      # Task is currently being worked on
    BLOCKED = "blocked"              # Task is blocked waiting for dependencies
    COMPLETED = "completed"          # Task has been completed successfully
    FAILED = "failed"                # Task has failed
    CANCELLED = "cancelled"          # Task has been cancelled


class TaskPriority(Enum):
    """Priority levels for tasks."""
    CRITICAL = "critical"            # Highest priority
    HIGH = "high"                    # High priority
    MEDIUM = "medium"                # Medium priority (default)
    LOW = "low"                      # Low priority
    BACKGROUND = "background"        # Lowest priority, process when resources available


class Task:
    """
    Represents a collaborative task that can be assigned to one or more agents.
    
    Tasks can have dependencies on other tasks, be broken down into subtasks,
    and require specific agent capabilities.
    """
    
    def __init__(
        self,
        task_id: str,
        title: str,
        description: str,
        creator_id: str,
        required_capabilities: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        deadline: Optional[str] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new task.
        
        Args:
            task_id: Unique identifier for the task
            title: Short title describing the task
            description: Detailed description of the task
            creator_id: ID of the agent creating the task
            required_capabilities: List of capabilities required to complete the task
            dependencies: List of task IDs that must be completed before this task
            deadline: ISO format datetime string deadline for the task
            priority: Priority level of the task
            metadata: Additional metadata for the task
        """
        self.task_id = task_id
        self.title = title
        self.description = description
        self.creator_id = creator_id
        self.required_capabilities = required_capabilities or []
        self.dependencies = dependencies or []
        self.deadline = deadline
        self.priority = priority
        self.metadata = metadata or {}
        
        # Task state
        self.status = TaskStatus.CREATED
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at
        self.started_at = None
        self.completed_at = None
        
        # Assignment information
        self.assigned_agents: List[str] = []
        self.progress: float = 0.0  # 0.0 to 1.0
        
        # Results and subtasks
        self.results: Dict[str, Any] = {}
        self.subtasks: Dict[str, 'Task'] = {}
        self.parent_task_id: Optional[str] = None
        
        # Communication history
        self.messages: List[Dict[str, Any]] = []
        
        # Error tracking
        self.errors: List[Dict[str, Any]] = []
        
        logger.info(f"Created task {task_id}: {title}")
    
    def assign_agent(self, agent_id: str) -> bool:
        """
        Assign an agent to this task.
        
        Args:
            agent_id: ID of the agent to assign
            
        Returns:
            Whether the assignment was successful
        """
        if agent_id in self.assigned_agents:
            return False  # Already assigned
            
        self.assigned_agents = [*self.assigned_agents, agent_id]
        self.status = TaskStatus.ASSIGNED
        self.updated_at = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"Assigned agent {agent_id} to task {self.task_id}")
        return True
    
    def start(self) -> bool:
        """
        Start the task execution.
        
        Returns:
            Whether the task was successfully started
        """
        # Check if task can be started
        if self.status != TaskStatus.ASSIGNED:
            return False
            
        # Check if all dependencies are complete
        # Note: This would typically be checked externally by the TaskCoordinator
        
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.started_at
        
        logger.info(f"Started task {self.task_id}")
        return True
    
    def update_progress(self, progress: float) -> None:
        """
        Update the progress of the task.
        
        Args:
            progress: Progress value between 0.0 and 1.0
        """
        self.progress = max(0.0, min(1.0, progress))
        self.updated_at = datetime.now(timezone.utc).isoformat()
        
        logger.debug(f"Updated task {self.task_id} progress to {self.progress:.2f}")
    
    def add_result(self, agent_id: str, result: Any) -> None:
        """
        Add a result from an agent.
        
        Args:
            agent_id: ID of the agent providing the result
            result: The result data
        """
        self.results = {**self.results, agent_id: {
            "data": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }}
        self.updated_at = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"Added result from agent {agent_id} to task {self.task_id}")
    
    def complete(self) -> bool:
        """
        Mark the task as completed.
        
        Returns:
            Whether the task was successfully completed
        """
        if self.status != TaskStatus.IN_PROGRESS:
            return False
            
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.completed_at
        self.progress = 1.0
        
        logger.info(f"Completed task {self.task_id}")
        return True
    
    def fail(self, error_message: str) -> None:
        """
        Mark the task as failed.
        
        Args:
            error_message: Description of the error
        """
        self.status = TaskStatus.FAILED
        self.updated_at = datetime.now(timezone.utc).isoformat()
        
        error = {
            "message": error_message,
            "timestamp": self.updated_at
        }
        self.errors = [*self.errors, error]
        
        logger.error(f"Task {self.task_id} failed: {error_message}")
    
    def cancel(self) -> bool:
        """
        Cancel the task.
        
        Returns:
            Whether the task was successfully cancelled
        """
        # Only allow cancellation if not already completed or failed
        if self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
            
        self.status = TaskStatus.CANCELLED
        self.updated_at = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"Cancelled task {self.task_id}")
        return True
    
    def add_subtask(self, subtask: 'Task') -> None:
        """
        Add a subtask to this task.
        
        Args:
            subtask: The subtask to add
        """
        subtask.parent_task_id = self.task_id
        self.subtasks = {**self.subtasks, subtask.task_id: subtask}
        self.updated_at = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"Added subtask {subtask.task_id} to task {self.task_id}")
    
    def add_message(self, sender_id: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a message to the task's communication history.
        
        Args:
            sender_id: ID of the agent sending the message
            content: Content of the message
            metadata: Additional metadata about the message
        """
        message = {
            "sender_id": sender_id,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }
        
        self.messages = [*self.messages, message]
        self.updated_at = message["timestamp"]
        
        logger.debug(f"Added message from {sender_id} to task {self.task_id}")
    
    def get_status_update(self) -> Dict[str, Any]:
        """
        Get a status update for the task.
        
        Returns:
            Status update information
        """
        # Calculate overall subtask progress
        subtask_progress = 0.0
        if self.subtasks:
            subtask_sum = sum(subtask.progress for subtask in self.subtasks.values())
            subtask_progress = subtask_sum / len(self.subtasks)
            
        # The task's overall progress is a combination of its own progress and its subtasks
        overall_progress = self.progress
        if self.subtasks:
            overall_progress = (self.progress + subtask_progress) / 2
            
        update = {
            "task_id": self.task_id,
            "title": self.title,
            "status": self.status.value,
            "progress": overall_progress,
            "assigned_agents": self.assigned_agents,
            "updated_at": self.updated_at,
            "subtask_count": len(self.subtasks),
            "result_count": len(self.results),
            "has_errors": len(self.errors) > 0
        }
        
        return update
    
    def to_dict(self, include_subtasks: bool = False) -> Dict[str, Any]:
        """
        Convert the task to a dictionary.
        
        Args:
            include_subtasks: Whether to include subtask data
            
        Returns:
            Dictionary representation of the task
        """
        task_dict = {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "creator_id": self.creator_id,
            "required_capabilities": self.required_capabilities,
            "dependencies": self.dependencies,
            "deadline": self.deadline,
            "priority": self.priority.value,
            "metadata": self.metadata,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "assigned_agents": self.assigned_agents,
            "progress": self.progress,
            "results": self.results,
            "parent_task_id": self.parent_task_id,
            "messages": self.messages,
            "errors": self.errors
        }
        
        if include_subtasks:
            task_dict["subtasks"] = {
                task_id: subtask.to_dict(include_subtasks=True)
                for task_id, subtask in self.subtasks.items()
            }
        else:
            task_dict["subtask_ids"] = list(self.subtasks.keys())
            
        return task_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """
        Create a Task from a dictionary.
        
        Args:
            data: Dictionary containing task data
            
        Returns:
            Task object
        """
        # Create task with required fields
        task = cls(
            task_id=data["task_id"],
            title=data["title"],
            description=data["description"],
            creator_id=data["creator_id"],
            required_capabilities=data.get("required_capabilities", []),
            dependencies=data.get("dependencies", []),
            deadline=data.get("deadline"),
            priority=TaskPriority(data.get("priority", TaskPriority.MEDIUM.value)),
            metadata=data.get("metadata", {})
        )
        
        # Set state fields
        task.status = TaskStatus(data.get("status", TaskStatus.CREATED.value))
        task.created_at = data.get("created_at", task.created_at)
        task.updated_at = data.get("updated_at", task.updated_at)
        task.started_at = data.get("started_at")
        task.completed_at = data.get("completed_at")
        
        # Set assignment information
        task.assigned_agents = data.get("assigned_agents", [])
        task.progress = data.get("progress", 0.0)
        
        # Set results and parent task
        task.results = data.get("results", {})
        task.parent_task_id = data.get("parent_task_id")
        
        # Set messages and errors
        task.messages = data.get("messages", [])
        task.errors = data.get("errors", [])
        
        # Load subtasks if present
        if "subtasks" in data:
            for subtask_data in data["subtasks"].values():
                subtask = cls.from_dict(subtask_data)
                task.subtasks[subtask.task_id] = subtask
                
        return task


class TaskCoordinator:
    """
    Coordinates collaborative tasks among multiple agents.
    
    This class manages task creation, assignment, status tracking, and
    result aggregation for multi-agent collaboration.
    """
    
    def __init__(
        self,
        agent_id: str,
        communication_adapter: Optional[CommunicationAdapter] = None,
        content_handler: Optional[ContentHandler] = None,
        storage_dir: str = "data/tasks"
    ):
        """
        Initialize the task coordinator.
        
        Args:
            agent_id: ID of the agent using this coordinator
            communication_adapter: Adapter for agent communication
            content_handler: Handler for content format conversion
            storage_dir: Directory for storing task data
        """
        self.agent_id = agent_id
        self.communication_adapter = communication_adapter
        self.content_handler = content_handler
        self.storage_dir = storage_dir
        
        # Task storage
        self.tasks: Dict[str, Task] = {}
        self.task_lock = threading.Lock()
        
        # Dependency tracking
        self.dependency_graph: Dict[str, Set[str]] = {}  # task_id -> set of dependent task_ids
        
        # Agent capability cache
        self.agent_capabilities: Dict[str, List[str]] = {}
        
        # Task completion handlers
        self.completion_handlers: Dict[str, Callable[[Task], None]] = {}
        
        logger.info(f"TaskCoordinator initialized for agent {agent_id}")
    
    def create_task(
        self,
        title: str,
        description: str,
        required_capabilities: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        deadline: Optional[str] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None
    ) -> Task:
        """
        Create a new collaborative task.
        
        Args:
            title: Short title describing the task
            description: Detailed description of the task
            required_capabilities: List of capabilities required to complete the task
            dependencies: List of task IDs that must be completed before this task
            deadline: ISO format datetime string deadline for the task
            priority: Priority level of the task
            metadata: Additional metadata for the task
            task_id: Optional custom task ID
            
        Returns:
            The created task
        """
        # Generate task ID if not provided
        if not task_id:
            task_id = f"task-{uuid.uuid4().hex}"
            
        # Create task
        task = Task(
            task_id=task_id,
            title=title,
            description=description,
            creator_id=self.agent_id,
            required_capabilities=required_capabilities,
            dependencies=dependencies,
            deadline=deadline,
            priority=priority,
            metadata=metadata
        )
        
        # Store task
        with self.task_lock:
            self.tasks = {**self.tasks, task_id: task}
            
            # Update dependency graph
            if dependencies:
                for dep_id in dependencies:
                    if dep_id not in self.dependency_graph:
                        self.dependency_graph = {**self.dependency_graph, dep_id: set()}
                    self.dependency_graph[dep_id].add(task_id)
                    
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task if found, None otherwise
        """
        return self.tasks.get(task_id)
    
    def assign_task(
        self,
        task_id: str,
        agent_ids: List[str]
    ) -> bool:
        """
        Assign a task to one or more agents.
        
        Args:
            task_id: ID of the task to assign
            agent_ids: List of agent IDs to assign the task to
            
        Returns:
            Whether the assignment was successful
        """
        task = self.get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found for assignment")
            return False
            
        # Check if task can be assigned
        if task.status not in [TaskStatus.CREATED, TaskStatus.ASSIGNED]:
            logger.error(f"Task {task_id} cannot be assigned (status: {task.status.value})")
            return False
            
        # Check if dependencies are satisfied
        for dep_id in task.dependencies:
            dep_task = self.get_task(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                logger.error(f"Task {task_id} has unsatisfied dependency: {dep_id}")
                return False
                
        # Check agent capabilities if required
        if task.required_capabilities:
            for agent_id in agent_ids:
                if not self._agent_has_capabilities(agent_id, task.required_capabilities):
                    logger.error(f"Agent {agent_id} lacks required capabilities for task {task_id}")
                    return False
                    
        # Assign task to agents
        for agent_id in agent_ids:
            task.assign_agent(agent_id)
            
        return True
    
    def start_task(self, task_id: str) -> bool:
        """
        Start a task that has been assigned.
        
        Args:
            task_id: ID of the task to start
            
        Returns:
            Whether the task was successfully started
        """
        task = self.get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found for starting")
            return False
            
        # Check if task can be started
        if task.status != TaskStatus.ASSIGNED:
            logger.error(f"Task {task_id} cannot be started (status: {task.status.value})")
            return False
            
        # Check if dependencies are satisfied
        for dep_id in task.dependencies:
            dep_task = self.get_task(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                logger.error(f"Task {task_id} has unsatisfied dependency: {dep_id}")
                task.status = TaskStatus.BLOCKED
                return False
                
        # Start the task
        return task.start()
    
    def update_task_progress(self, task_id: str, progress: float, agent_id: str) -> bool:
        """
        Update the progress of a task.
        
        Args:
            task_id: ID of the task
            progress: Progress value between 0.0 and 1.0
            agent_id: ID of the agent updating the progress
            
        Returns:
            Whether the update was successful
        """
        task = self.get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found for progress update")
            return False
            
        # Check if agent is assigned to the task
        if agent_id not in task.assigned_agents:
            logger.error(f"Agent {agent_id} not assigned to task {task_id}")
            return False
            
        # Update progress
        task.update_progress(progress)
        return True
    
    def add_task_result(self, task_id: str, agent_id: str, result: Any) -> bool:
        """
        Add a result to a task.
        
        Args:
            task_id: ID of the task
            agent_id: ID of the agent providing the result
            result: The result data
            
        Returns:
            Whether the result was added successfully
        """
        task = self.get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found for adding result")
            return False
            
        # Check if agent is assigned to the task
        if agent_id not in task.assigned_agents:
            logger.error(f"Agent {agent_id} not assigned to task {task_id}")
            return False
            
        # Add result
        task.add_result(agent_id, result)
        return True
    
    def complete_task(self, task_id: str, agent_id: str) -> bool:
        """
        Mark a task as completed.
        
        Args:
            task_id: ID of the task
            agent_id: ID of the agent completing the task
            
        Returns:
            Whether the task was successfully completed
        """
        task = self.get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found for completion")
            return False
            
        # Check if agent is assigned to the task
        if agent_id not in task.assigned_agents:
            logger.error(f"Agent {agent_id} not assigned to task {task_id}")
            return False
            
        # Complete the task
        success = task.complete()
        if not success:
            return False
            
        # Notify dependent tasks
        self._process_completed_task(task)
        
        # Call completion handler if registered
        if task_id in self.completion_handlers:
            try:
                self.completion_handlers[task_id](task)
            except Exception as e:
                logger.error(f"Error in completion handler for task {task_id}: {str(e)}")
                
        return True
    
    def fail_task(self, task_id: str, error_message: str, agent_id: Optional[str] = None) -> bool:
        """
        Mark a task as failed.
        
        Args:
            task_id: ID of the task
            error_message: Description of the error
            agent_id: Optional ID of the agent reporting the failure
            
        Returns:
            Whether the task was successfully marked as failed
        """
        task = self.get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found for failure")
            return False
            
        # Check if agent is assigned to the task if specified
        if agent_id and agent_id not in task.assigned_agents:
            logger.error(f"Agent {agent_id} not assigned to task {task_id}")
            return False
            
        # Mark as failed
        task.fail(error_message)
        return True
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Whether the task was successfully cancelled
        """
        task = self.get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found for cancellation")
            return False
            
        # Cancel the task
        return task.cancel()
    
    def add_subtask(
        self,
        parent_task_id: str,
        title: str,
        description: str,
        required_capabilities: Optional[List[str]] = None,
        priority: Optional[TaskPriority] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Task]:
        """
        Add a subtask to an existing task.
        
        Args:
            parent_task_id: ID of the parent task
            title: Short title describing the subtask
            description: Detailed description of the subtask
            required_capabilities: List of capabilities required for the subtask
            priority: Priority level (defaults to parent's priority)
            metadata: Additional metadata for the subtask
            
        Returns:
            The created subtask or None if parent not found
        """
        parent_task = self.get_task(parent_task_id)
        if not parent_task:
            logger.error(f"Parent task {parent_task_id} not found for adding subtask")
            return None
            
        # Use parent's priority if not specified
        if priority is None:
            priority = parent_task.priority
            
        # Inherit parent's deadline
        deadline = parent_task.deadline
        
        # Create subtask with auto-generated ID
        subtask = self.create_task(
            title=title,
            description=description,
            required_capabilities=required_capabilities,
            dependencies=None,  # Subtasks don't have external dependencies
            deadline=deadline,
            priority=priority,
            metadata=metadata
        )
        
        # Add to parent
        parent_task.add_subtask(subtask)
        
        return subtask
    
    def add_task_message(
        self,
        task_id: str,
        sender_id: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a message to a task's communication history.
        
        Args:
            task_id: ID of the task
            sender_id: ID of the agent sending the message
            content: Content of the message
            metadata: Additional metadata about the message
            
        Returns:
            Whether the message was added successfully
        """
        task = self.get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found for adding message")
            return False
            
        # Add message
        task.add_message(sender_id, content, metadata)
        return True
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """
        Get all tasks with a specific status.
        
        Args:
            status: Status to filter tasks by
            
        Returns:
            List of tasks with the specified status
        """
        return [task for task in self.tasks.values() if task.status == status]
    
    def get_tasks_by_agent(self, agent_id: str) -> List[Task]:
        """
        Get all tasks assigned to a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of tasks assigned to the agent
        """
        return [task for task in self.tasks.values() if agent_id in task.assigned_agents]
    
    def get_dependent_tasks(self, task_id: str) -> List[Task]:
        """
        Get all tasks that depend on a specific task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            List of dependent tasks
        """
        dependent_ids = self.dependency_graph.get(task_id, set())
        return [self.tasks[dep_id] for dep_id in dependent_ids if dep_id in self.tasks]
    
    def register_completion_handler(
        self,
        task_id: str,
        handler: Callable[[Task], None]
    ) -> None:
        """
        Register a handler to be called when a task is completed.
        
        Args:
            task_id: ID of the task
            handler: Function to call with the completed task
        """
        self.completion_handlers = {**self.completion_handlers, task_id: handler}
    
    def suggest_agents_for_task(self, task_id: str) -> List[str]:
        """
        Suggest agents that are suitable for a task based on capabilities.
        
        Args:
            task_id: ID of the task
            
        Returns:
            List of agent IDs that meet the capability requirements
        """
        task = self.get_task(task_id)
        if not task or not task.required_capabilities:
            return []
            
        # Find agents with the required capabilities
        suitable_agents = []
        for agent_id, capabilities in self.agent_capabilities.items():
            if self._agent_has_capabilities(agent_id, task.required_capabilities):
                suitable_agents.append(agent_id)
                
        return suitable_agents
    
    def update_agent_capabilities(self, agent_id: str, capabilities: List[str]) -> None:
        """
        Update the cached capabilities for an agent.
        
        Args:
            agent_id: ID of the agent
            capabilities: List of capabilities the agent has
        """
        self.agent_capabilities = {**self.agent_capabilities, agent_id: capabilities}
    
    def get_task_dependency_chain(self, task_id: str) -> List[str]:
        """
        Get the chain of task dependencies for a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            List of task IDs in dependency order (ancestors first)
        """
        task = self.get_task(task_id)
        if not task:
            return []
            
        # Build dependency chain
        chain = []
        visited = set()
        
        def visit(t_id) -> None:
            if t_id in visited:
                return
            visited.add(t_id)
            
            t = self.get_task(t_id)
            if t:
                for dep_id in t.dependencies:
                    visit(dep_id)
                chain.append(t_id)
                
        visit(task_id)
        return chain[:-1]  # Exclude the task itself
    
    def export_task_graph(self) -> Dict[str, Any]:
        """
        Export the task dependency graph.
        
        Returns:
            Graph representation of tasks and their dependencies
        """
        nodes = []
        edges = []
        
        # Create nodes for each task
        for task_id, task in self.tasks.items():
            nodes.append({
                "id": task_id,
                "label": task.title,
                "status": task.status.value,
                "progress": task.progress
            })
            
            # Create edges for dependencies
            for dep_id in task.dependencies:
                if dep_id in self.tasks:
                    edges.append({
                        "source": dep_id,
                        "target": task_id,
                        "type": "dependency"
                    })
                    
            # Create edges for subtasks
            for subtask_id in task.subtasks:
                edges.append({
                    "source": task_id,
                    "target": subtask_id,
                    "type": "subtask"
                })
                
        # Create the graph object
        graph = {
            "nodes": nodes,
            "edges": edges
        }
        
        return graph
    
    def _agent_has_capabilities(self, agent_id: str, required_capabilities: List[str]) -> bool:
        """
        Check if an agent has all the required capabilities.
        
        Args:
            agent_id: ID of the agent
            required_capabilities: List of capabilities required
            
        Returns:
            Whether the agent has all the required capabilities
        """
        # Get agent capabilities from cache
        agent_caps = self.agent_capabilities.get(agent_id, [])
        
        # Check if agent has all required capabilities
        return all(cap in agent_caps for cap in required_capabilities)
    
    def _process_completed_task(self, task: Task) -> None:
        """
        Process a completed task and update dependent tasks.
        
        Args:
            task: The completed task
        """
        # Get dependent tasks
        dependent_tasks = self.get_dependent_tasks(task.task_id)
        
        for dep_task in dependent_tasks:
            # Check if all dependencies are now satisfied
            all_deps_satisfied = True
            for dep_id in dep_task.dependencies:
                dep = self.get_task(dep_id)
                if not dep or dep.status != TaskStatus.COMPLETED:
                    all_deps_satisfied = False
                    break
                    
            # If task was blocked and all dependencies are now satisfied,
            # update its status to ASSIGNED if it has assigned agents
            if dep_task.status == TaskStatus.BLOCKED and all_deps_satisfied:
                if dep_task.assigned_agents:
                    dep_task.status = TaskStatus.ASSIGNED
                    dep_task.updated_at = datetime.now(timezone.utc).isoformat()
                    logger.info(f"Task {dep_task.task_id} unblocked after dependency {task.task_id} completed")
                else:
                    # No agents assigned yet, revert to CREATED
                    dep_task.status = TaskStatus.CREATED
                    dep_task.updated_at = datetime.now(timezone.utc).isoformat()
