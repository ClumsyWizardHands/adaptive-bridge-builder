"""
Empire Component Task Handler

This module implements a task handler for Empire Framework components,
enabling asynchronous processing of component operations through the A2A Protocol.
It handles tasks like component creation, updates, validation, and relationship management.
"""

import json
import uuid
import logging
import asyncio
import time
from typing import Any, Awaitable, Callable, Coroutine, Dict, List, Optional, Union
from datetime import datetime, timezone
from enum import Enum
import traceback

# Import Empire Framework modules
from .message_structures import (
    ComponentTaskTypes, 
    TaskStatus, 
    ComponentTaskDefinition,
    ComponentMessage
)

from ..registry.component_registry import ComponentRegistry
from ..validation.schema_validator import SchemaValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ComponentTaskHandler")

class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"

class Task:
    """
    Represents a component task for asynchronous processing.
    
    This class encapsulates all the information needed to execute
    a task related to Empire Framework components.
    """
    
    def __init__(
        self,
        task_id: str,
        task_type: str,
        component_ids: List[str],
        task_data: Dict[str, Any],
        priority: str = TaskPriority.MEDIUM.value,
        created_by: Optional[str] = None
    ):
        """
        Initialize a component task.
        
        Args:
            task_id: Unique ID for the task
            task_type: Type of task (use ComponentTaskTypes constants)
            component_ids: IDs of components involved in the task
            task_data: Data needed to perform the task
            priority: Task priority level
            created_by: ID of the agent that created the task
        """
        self.task_id = task_id
        self.task_type = task_type
        self.component_ids = component_ids
        self.task_data = task_data
        self.priority = priority
        self.created_by = created_by
        
        # Status tracking
        self.status = TaskStatus.PENDING
        self.created_at = datetime.utcnow().isoformat()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.progress: float = 0.0
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[Dict[str, Any]] = None
        self.cancellation_requested: bool = False
        self.cancellation_reason: Optional[str] = None
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert task to a dictionary.
        
        Returns:
            Dictionary representation of the task
        """
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "component_ids": self.component_ids,
            "task_data": self.task_data,
            "priority": self.priority,
            "created_by": self.created_by,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "cancellation_requested": self.cancellation_requested,
            "cancellation_reason": self.cancellation_reason
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """
        Create a task from a dictionary.
        
        Args:
            data: Dictionary with task data
            
        Returns:
            Task instance
        """
        task = cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            component_ids=data["component_ids"],
            task_data=data["task_data"],
            priority=data.get("priority", TaskPriority.MEDIUM.value),
            created_by=data.get("created_by")
        )
        
        # Load status tracking fields
        task.status = data.get("status", TaskStatus.PENDING)
        task.created_at = data.get("created_at", task.created_at)
        task.started_at = data.get("started_at")
        task.completed_at = data.get("completed_at")
        task.progress = data.get("progress", 0.0)
        task.result = data.get("result")
        task.error = data.get("error")
        task.cancellation_requested = data.get("cancellation_requested", False)
        task.cancellation_reason = data.get("cancellation_reason")
        
        return task

class ComponentTaskHandler:
    """
    Handler for asynchronous Empire Framework component tasks.
    
    This class processes component-related tasks like creation, updates,
    validation, and relationship management in an asynchronous manner,
    with support for priority-based execution and task cancellation.
    """
    
    def __init__(
        self,
        registry: Optional[ComponentRegistry] = None,
        validator: Optional[SchemaValidator] = None,
        max_concurrent_tasks: int = 10
    ):
        """
        Initialize the component task handler.
        
        Args:
            registry: Optional component registry to use
            validator: Optional schema validator to use
            max_concurrent_tasks: Maximum number of concurrent tasks
        """
        self._background_tasks: List[asyncio.Task] = []
        self.registry = registry or ComponentRegistry()
        self.validator = validator or SchemaValidator()
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Task storage
        self.tasks: Dict[str, Task] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # Handlers for different task types
        self.task_handlers: Dict[str, Callable[[Task], Awaitable[Dict[str, Any]]]] = {
            ComponentTaskTypes.CREATION: self._handle_creation_task,
            ComponentTaskTypes.UPDATE: self._handle_update_task,
            ComponentTaskTypes.DELETION: self._handle_deletion_task,
            ComponentTaskTypes.VALIDATION: self._handle_validation_task,
            ComponentTaskTypes.TRANSFORMATION: self._handle_transformation_task,
            ComponentTaskTypes.RELATIONSHIP_UPDATE: self._handle_relationship_update_task,
            ComponentTaskTypes.BATCH_OPERATION: self._handle_batch_operation_task,
            ComponentTaskTypes.EXPORT: self._handle_export_task,
            ComponentTaskTypes.IMPORT: self._handle_import_task,
            ComponentTaskTypes.ANALYSIS: self._handle_analysis_task
        }
        
        logger.info(f"Component task handler initialized with {max_concurrent_tasks} concurrent tasks limit")
    
    async def start_processing(self) -> None:
        """
        Start processing tasks from the queue.
        
        This method starts worker tasks to process the queue.
        """
        workers = []
        for _ in range(self.max_concurrent_tasks):
            worker = asyncio.create_task(self._process_task_queue())
            workers.append(worker)
            
        logger.info(f"Started {len(workers)} task processing workers")
        
        return workers
    
    async def _process_task_queue(self) -> None:
        """
        Process tasks from the queue continuously.
        
        This method runs as a worker to take tasks from the queue
        and process them according to their priority.
        """
        while True:
            try:
                # Get task with highest priority
                priority, task_id = await self.task_queue.get()
                
                # Skip if task was cancelled
                if task_id not in self.tasks or self.tasks[task_id].cancellation_requested:
                    self.task_queue.task_done()
                    continue
                    
                # Get the task
                task = self.tasks[task_id]
                
                # Update task status
                task.status = TaskStatus.PROCESSING
                task.started_at = datetime.utcnow().isoformat()
                
                # Process the task
                asyncio_task = asyncio.create_task(self._execute_task(task))
                self.running_tasks = {**self.running_tasks, task_id: asyncio_task}
                
                # Wait for task to complete
                await asyncio_task
                
                # Mark queue item as done
                self.task_queue.task_done()
                
                # Remove from running tasks
                if task_id in self.running_tasks:
                    self.running_tasks = {k: v for k, v in self.running_tasks.items() if k != task_id}
                    
            except Exception as e:
                logger.error(f"Error in task queue processor: {str(e)}")
                logger.debug(traceback.format_exc())
                
                # Mark queue item as done even on error
                self.task_queue.task_done()
    
    async def _execute_task(self, task: Task) -> Coroutine[Any, Any, None]:
        """
        Execute a specific task.
        
        Args:
            task: Task to execute
        """
        logger.info(f"Executing task {task.task_id} of type {task.task_type}")
        
        try:
            # Check if task was cancelled
            if task.cancellation_requested:
                task.status = TaskStatus.CANCELLED
                logger.info(f"Task {task.task_id} was cancelled before execution")
                return
                
            # Get appropriate handler for task type
            handler = self.task_handlers.get(task.task_type)
            
            if not handler:
                raise ValueError(f"No handler for task type: {task.task_type}")
                
            # Execute the handler
            result = await handler(task)
            
            # Update task with result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow().isoformat()
            task.progress = 1.0
            task.result = result
            
            logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            # Handle errors
            error_message = str(e)
            stack_trace = traceback.format_exc()
            
            logger.error(f"Error executing task {task.task_id}: {error_message}")
            logger.debug(f"Stack trace: {stack_trace}")
            
            # Update task with error
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow().isoformat()
            task.error = {
                "message": error_message,
                "stack_trace": stack_trace
            }
    
    async def create_task(
        self,
        task_type: str,
        component_ids: List[str],
        task_data: Dict[str, Any],
        priority: str = TaskPriority.MEDIUM.value,
        created_by: Optional[str] = None,
        task_id: Optional[str] = None
    ) -> str:
        """
        Create and queue a new task.
        
        Args:
            task_type: Type of task to create
            component_ids: IDs of components involved
            task_data: Data needed for the task
            priority: Priority level for execution
            created_by: ID of the agent creating the task
            task_id: Optional specific task ID
            
        Returns:
            ID of the created task
        """
        # Generate task ID if not provided
        if not task_id:
            task_id = f"task-{uuid.uuid4().hex}"
            
        # Create task
        task = Task(
            task_id=task_id,
            task_type=task_type,
            component_ids=component_ids,
            task_data=task_data,
            priority=priority,
            created_by=created_by
        )
        
        # Store task
        self.tasks = {**self.tasks, task_id: task}
        
        # Determine priority number (lower number = higher priority)
        priority_num = {
            TaskPriority.CRITICAL.value: 0,
            TaskPriority.HIGH.value: 1,
            TaskPriority.MEDIUM.value: 2,
            TaskPriority.LOW.value: 3,
            TaskPriority.BACKGROUND.value: 4
        }.get(priority, 2)  # Default to medium priority
        
        # Add to queue with priority
        await self.task_queue.put((priority_num, task_id))
        
        logger.info(f"Created task {task_id} of type {task_type} with priority {priority}")
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task data or None if not found
        """
        if task_id in self.tasks:
            return self.tasks[task_id].to_dict()
        return None
    
    def list_tasks(
        self,
        status: Optional[str] = None,
        task_type: Optional[str] = None,
        component_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List tasks with optional filters.
        
        Args:
            status: Optional status to filter by
            task_type: Optional task type to filter by
            component_id: Optional component ID to filter by
            limit: Optional maximum number of tasks to return
            
        Returns:
            List of matching tasks
        """
        results = []
        
        for task in self.tasks.values():
            # Apply filters
            if status and task.status != status:
                continue
                
            if task_type and task.task_type != task_type:
                continue
                
            if component_id and component_id not in task.component_ids:
                continue
                
            # Add to results
            results.append(task.to_dict())
            
            # Check limit
            if limit is not None and len(results) >= limit:
                break
                
        # Sort by creation time, newest first
        results.sort(key=lambda t: t["created_at"], reverse=True)
        
        return results
    
    async def cancel_task(
        self,
        task_id: str,
        reason: Optional[str] = None
    ) -> bool:
        """
        Cancel a pending or running task.
        
        Args:
            task_id: ID of the task to cancel
            reason: Optional reason for cancellation
            
        Returns:
            True if the task was cancelled
        """
        if task_id not in self.tasks:
            return False
            
        task = self.tasks[task_id]
        
        # Mark as cancellation requested
        task.cancellation_requested = True
        task.cancellation_reason = reason
        
        # If task is running, cancel it
        if task_id in self.running_tasks:
            running_task = self.running_tasks[task_id]
            running_task.cancel()
            
            try:
                await running_task
            except asyncio.CancelledError:
                pass
                
            # Remove from running tasks
            self.running_tasks = {k: v for k, v in self.running_tasks.items() if k != task_id}
            
        # Update status if not already completed or failed
        if task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.utcnow().isoformat()
            
        logger.info(f"Task {task_id} cancelled: {reason or 'No reason provided'}")
        
        return True
    
    async def _handle_creation_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle a component creation task.
        
        Args:
            task: Task to handle
            
        Returns:
            Result of task execution
        """
        # Extract task data
        component_data = task.task_data.get("component_data", {})
        component_type = task.task_data.get("component_type")
        
        # Validate data
        if not component_type:
            raise ValueError("Missing component_type in task data")
            
        if not component_data:
            raise ValueError("Missing component_data in task data")
            
        # Generate component ID if not provided
        if "id" not in component_data:
            component_data["id"] = f"{component_type}-{uuid.uuid4().hex[:8]}"
            
        # Add timestamp if not present
        if "created_at" not in component_data:
            component_data["created_at"] = datetime.utcnow().isoformat()
            
        if "last_modified_date" not in component_data:
            component_data["last_modified_date"] = datetime.utcnow().isoformat()
            
        # Add type if not present
        if "component_type" not in component_data:
            component_data["component_type"] = component_type
            
        # Validate schema if validator exists
        if self.validator:
            validation_result = self.validator.validate_component(component_data, component_type)
            if not validation_result["valid"]:
                raise ValueError(f"Invalid component data: {validation_result['errors']}")
                
        # Register component
        self.registry.register_component(component_data)
        
        # Update progress
        task.progress = 1.0
        
        # Return result
        return {
            "component_id": component_data["id"],
            "component_type": component_type,
            "success": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_update_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle a component update task.
        
        Args:
            task: Task to handle
            
        Returns:
            Result of task execution
        """
        # Extract task data
        updates = task.task_data.get("updates", {})
        component_id = task.component_ids[0] if task.component_ids else None
        
        # Validate data
        if not component_id:
            raise ValueError("Missing component_id in task")
            
        if not updates:
            raise ValueError("Missing updates in task data")
            
        # Get current component
        component = self.registry.get_component(component_id)
        if not component:
            raise ValueError(f"Component not found: {component_id}")
            
        # Apply updates
        updated_component = {**component, **updates}
        
        # Update last modified date
        updated_component["last_modified_date"] = datetime.utcnow().isoformat()
        
        # Validate schema if validator exists
        if self.validator:
            component_type = updated_component.get("component_type") or component.get("type")
            validation_result = self.validator.validate_component(updated_component, component_type)
            if not validation_result["valid"]:
                raise ValueError(f"Invalid component updates: {validation_result['errors']}")
                
        # Update component
        self.registry.update_component(component_id, updated_component)
        
        # Update progress
        task.progress = 1.0
        
        # Return result
        return {
            "component_id": component_id,
            "updated_fields": list(updates.keys()),
            "success": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_deletion_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle a component deletion task.
        
        Args:
            task: Task to handle
            
        Returns:
            Result of task execution
        """
        # Extract task data
        component_id = task.component_ids[0] if task.component_ids else None
        hard_delete = task.task_data.get("hard_delete", False)
        
        # Validate data
        if not component_id:
            raise ValueError("Missing component_id in task")
            
        # Delete component
        success = self.registry.remove_component(component_id, permanent=hard_delete)
        if not success:
            raise ValueError(f"Failed to delete component: {component_id}")
            
        # Update progress
        task.progress = 1.0
        
        # Return result
        return {
            "component_id": component_id,
            "hard_delete": hard_delete,
            "success": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_validation_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle a component validation task.
        
        Args:
            task: Task to handle
            
        Returns:
            Result of task execution
        """
        # Extract task data
        component_data = task.task_data.get("component_data", {})
        schema_id = task.task_data.get("schema_id")
        
        # Validate data
        if not component_data:
            raise ValueError("Missing component_data in task data")
            
        # Get component type
        component_type = component_data.get("component_type") or component_data.get("type")
        
        if not component_type and not schema_id:
            raise ValueError("Missing component_type and schema_id")
            
        # Validate schema
        if not self.validator:
            raise ValueError("Schema validator not available")
            
        validation_result = self.validator.validate_component(
            component_data, 
            component_type,
            schema_id=schema_id
        )
        
        # Update progress
        task.progress = 1.0
        
        # Return result
        return {
            "valid": validation_result["valid"],
            "errors": validation_result.get("errors", []),
            "warnings": validation_result.get("warnings", []),
            "schema_id": schema_id or f"{component_type}_schema",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_transformation_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle a component transformation task.
        
        Args:
            task: Task to handle
            
        Returns:
            Result of task execution
        """
        # Extract task data
        component_id = task.component_ids[0] if task.component_ids else None
        transformation_type = task.task_data.get("transformation_type")
        transformation_params = task.task_data.get("transformation_params", {})
        
        # Validate data
        if not component_id:
            raise ValueError("Missing component_id in task")
            
        if not transformation_type:
            raise ValueError("Missing transformation_type in task data")
            
        # Get component
        component = self.registry.get_component(component_id)
        if not component:
            raise ValueError(f"Component not found: {component_id}")
            
        # Apply transformation based on type
        transformed_component = await self._apply_transformation(
            component,
            transformation_type,
            transformation_params
        )
        
        # Update progress
        task.progress = 1.0
        
        # Return result
        return {
            "component_id": component_id,
            "transformation_type": transformation_type,
            "transformed_component": transformed_component,
            "success": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _apply_transformation(
        self,
        component: Dict[str, Any],
        transformation_type: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply a transformation to a component.
        
        Args:
            component: Component to transform
            transformation_type: Type of transformation
            params: Transformation parameters
            
        Returns:
            Transformed component
        """
        # Create a deep copy to avoid modifying the original
        result = json.loads(json.dumps(component))
        
        # Apply transformation based on type
        if transformation_type == "restructure":
            # Restructure component fields
            field_mappings = params.get("field_mappings", {})
            for old_field, new_field in field_mappings.items():
                if old_field in result:
                    result[new_field] = result.pop(old_field)
                    
        elif transformation_type == "augment":
            # Add additional fields
            augmentations = params.get("augmentations", {})
            for field, value in augmentations.items():
                result[field] = value
                
        elif transformation_type == "filter":
            # Keep only specified fields
            fields_to_keep = params.get("fields", [])
            result = {k: v for k, v in result.items() if k in fields_to_keep}
            
        elif transformation_type == "convert_format":
            # Convert to a different format
            target_format = params.get("target_format")
            
            if target_format == "simplified":
                # Create a simplified version
                result = {
                    "id": component.get("id"),
                    "type": component.get("component_type") or component.get("type"),
                    "name": component.get("name") or component.get("component_name", "Unnamed"),
                    "description": component.get("description") or component.get("description_text", "")
                }
                
        else:
            raise ValueError(f"Unknown transformation type: {transformation_type}")
            
        return result
    
    async def _handle_relationship_update_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle a relationship update task.
        
        Args:
            task: Task to handle
            
        Returns:
            Result of task execution
        """
        # Extract task data
        source_id = task.component_ids[0] if len(task.component_ids) > 0 else None
        target_id = task.component_ids[1] if len(task.component_ids) > 1 else None
        operation = task.task_data.get("operation")
        relationship_data = task.task_data.get("relationship_data", {})
        
        # Validate data
        if not source_id:
            raise ValueError("Missing source component ID in task")
            
        if not target_id:
            raise ValueError("Missing target component ID in task")
            
        if not operation:
            raise ValueError("Missing operation in task data")
            
        # Perform operation
        if operation == "add":
            # Add relationship
            relationship_type = relationship_data.get("type")
            if not relationship_type:
                raise ValueError("Missing relationship type in relationship data")
                
            strength = relationship_data.get("strength", 0.5)
            
            success = self.registry.add_relationship(
                source_id,
                target_id,
                relationship_type,
                strength
            )
            
            if not success:
                raise ValueError(f"Failed to add relationship between {source_id} and {target_id}")
                
        elif operation == "remove":
            # Remove relationship
            success = self.registry.remove_relationship(source_id, target_id)
            
            if not success:
                raise ValueError(f"Failed to remove relationship between {source_id} and {target_id}")
                
        elif operation == "update":
            # Update relationship
            relationship_type = relationship_data.get("type")
            strength = relationship_data.get("strength")
            
            # Get current relationship
            current = self.registry.get_relationship(source_id, target_id)
            if not current:
                raise ValueError(f"Relationship not found between {source_id} and {target_id}")
                
            # Apply updates
            updated_type = relationship_type or current.get("type")
            updated_strength = strength if strength is not None else current.get("strength", 0.5)
            
            # Remove old relationship
            self.registry.remove_relationship(source_id, target_id)
            
            # Add updated relationship
            success = self.registry.add_relationship(
                source_id,
                target_id,
                updated_type,
                updated_strength
            )
            
            if not success:
                raise ValueError(f"Failed to update relationship between {source_id} and {target_id}")
                
        else:
            raise ValueError(f"Unknown relationship operation: {operation}")
            
        # Update progress
        task.progress = 1.0
        
        # Return result
        return {
            "source_id": source_id,
            "target_id": target_id,
            "operation": operation,
            "success": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_batch_operation_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle a batch operation task.
        
        Args:
            task: Task to handle
            
        Returns:
            Result of task execution
        """
        # Extract task data
        operation_type = task.task_data.get("operation_type")
        components = task.task_data.get("components", [])
        
        # Validate data
        if not operation_type:
            raise ValueError("Missing operation_type in task data")
            
        if not components:
            raise ValueError("Missing components in task data")
            
        # Process based on operation type
        results = []
        failures = []
        
        if operation_type == "create_batch":
            # Create multiple components
            for i, component_data in enumerate(components):
                try:
                    # Update progress incrementally
                    task.progress = i / len(components)
                    
                    # Generate component ID if not provided
                    if "id" not in component_data:
                        component_type = component_data.get("component_type") or component_data.get("type", "component")
                        component_data["id"] = f"{component_type}-{uuid.uuid4().hex[:8]}"
                        
                    # Add timestamps if not present
                    if "created_at" not in component_data:
                        component_data["created_at"] = datetime.utcnow().isoformat()
                        
                    if "last_modified_date" not in component_data:
                        component_data["last_modified_date"] = datetime.utcnow().isoformat()
                        
                    # Register component
                    self.registry.register_component(component_data)
                    
                    # Add success result
                    results.append({
                        "component_id": component_data["id"],
                        "success": True
                    })
                    
                except Exception as e:
                    # Add failure result
                    failures.append({
                        "component_data": component_data,
                        "error": str(e)
                    })
                    
        elif operation_type == "update_batch":
            # Update multiple components
            for i, update_data in enumerate(components):
                try:
                    # Update progress incrementally
                    task.progress = i / len(components)
                    
                    component_id = update_data.get("id")
                    updates = update_data.get("updates", {})
                    
                    if not component_id:
                        raise ValueError("Missing id in component update data")
                        
                    if not updates:
                        raise ValueError("Missing updates in component update data")
                        
                    # Get current component
                    component = self.registry.get_component(component_id)
                    if not component:
                        raise ValueError(f"Component not found: {component_id}")
                        
                    # Apply updates
                    updated_component = {**component, **updates}
                    
                    # Update last modified date
                    updated_component["last_modified_date"] = datetime.utcnow().isoformat()
                    
                    # Update component
                    self.registry.update_component(component_id, updated_component)
                    
                    # Add success result
                    results.append({
                        "component_id": component_id,
                        "updated_fields": list(updates.keys()),
                        "success": True
                    })
                    
                except Exception as e:
                    # Add failure result
                    failures.append({
                        "component_id": update_data.get("id"),
                        "error": str(e)
                    })
                    
        elif operation_type == "delete_batch":
            # Delete multiple components
            for i, component_id in enumerate(components):
                try:
                    # Update progress incrementally
                    task.progress = i / len(components)
                    
                    # Delete component
                    hard_delete = task.task_data.get("hard_delete", False)
                    success = self.registry.remove_component(component_id, permanent=hard_delete)
                    
                    if not success:
                        raise ValueError(f"Failed to delete component: {component_id}")
                        
                    # Add success result
                    results.append({
                        "component_id": component_id,
                        "success": True
                    })
                    
                except Exception as e:
                    # Add failure result
                    failures.append({
                        "component_id": component_id,
                        "error": str(e)
                    })
        
        else:
            raise ValueError(f"Unknown batch operation type: {operation_type}")
            
        # Update progress
        task.progress = 1.0
        
        # Return result
        return {
            "operation_type": operation_type,
            "success_count": len(results),
            "failure_count": len(failures),
            "results": results,
            "failures": failures,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_export_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle a component export task.
        
        Args:
            task: Task to handle
            
        Returns:
            Result of task execution
        """
        # Extract task data
        export_format = task.task_data.get("format", "json")
        component_ids = task.component_ids
        include_relationships = task.task_data.get("include_relationships", False)
        
        # Validate data
        if not component_ids:
            raise ValueError("Missing component_ids in task")
            
        # Get components
        components = []
        for component_id in component_ids:
            component = self.registry.get_component(component_id)
            if not component:
                raise ValueError(f"Component not found: {component_id}")
                
            components.append(component)
            
        # Get relationships if requested
        relationships = []
        if include_relationships:
            for component_id in component_ids:
                # Get related components
                related = self.registry.get_related_components(component_id)
                
                # For each related component, get the relationship
                for related_component in related:
                    relationship = self.registry.get_relationship(
                        component_id, 
                        related_component["id"]
                    )
                    
                    if relationship:
                        relationships.append({
                            "source_id": component_id,
                            "target_id": related_component["id"],
                            "type": relationship.get("type", "related"),
                            "strength": relationship.get("strength", 0.5)
                        })
        
        # Format export based on requested format
        export_data = {}
        
        if export_format == "json":
            # JSON format
            export_data = {
                "components": components,
                "relationships": relationships if include_relationships else []
            }
            
        elif export_format == "simplified":
            # Simplified format with essential fields only
            simplified_components = []
            
            for component in components:
                simplified = {
                    "id": component.get("id"),
                    "type": component.get("component_type") or component.get("type"),
                    "name": component.get("name") or component.get("component_name", "Unnamed"),
                    "description": component.get("description") or component.get("description_text", "")
                }
                
                simplified_components.append(simplified)
                
            export_data = {
                "components": simplified_components,
                "relationships": relationships if include_relationships else []
            }
            
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
            
        # Update progress
        task.progress = 1.0
        
        # Return result
        return {
            "export_format": export_format,
            "component_count": len(components),
            "relationship_count": len(relationships),
            "export_data": export_data,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_import_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle a component import task.
        
        Args:
            task: Task to handle
            
        Returns:
            Result of task execution
        """
        # Extract task data
        import_data = task.task_data.get("import_data", {})
        overwrite_existing = task.task_data.get("overwrite_existing", False)
        
        # Validate data
        if not import_data:
            raise ValueError("Missing import_data in task data")
            
        # Get components and relationships
        components = import_data.get("components", [])
        relationships = import_data.get("relationships", [])
        
        # Import components
        imported_component_ids = []
        skipped_component_ids = []
        failed_component_imports = []
        
        for i, component in enumerate(components):
            try:
                # Update progress
                task.progress = i / (len(components) + len(relationships))
                
                # Check if component exists
                component_id = component.get("id")
                existing = None
                
                if component_id:
                    existing = self.registry.get_component(component_id)
                    
                if existing and not overwrite_existing:
                    # Skip if exists and not overwriting
                    skipped_component_ids.append(component_id)
                    continue
                    
                # Import component
                self.registry.register_component(component)
                imported_component_ids.append(component_id)
                
            except Exception as e:
                # Add to failures
                failed_component_imports.append({
                    "component": component,
                    "error": str(e)
                })
        
        # Import relationships
        imported_relationship_count = 0
        failed_relationship_imports = []
        
        for i, relationship in enumerate(relationships):
            try:
                # Update progress
                progress_base = len(components) / (len(components) + len(relationships))
                task.progress = progress_base + (i / (len(components) + len(relationships)))
                
                # Extract relationship data
                source_id = relationship.get("source_id")
                target_id = relationship.get("target_id")
                rel_type = relationship.get("type", "related")
                strength = relationship.get("strength", 0.5)
                
                # Check if components exist
                source = self.registry.get_component(source_id)
                target = self.registry.get_component(target_id)
                
                if not source or not target:
                    raise ValueError(f"Source or target component not found: {source_id} -> {target_id}")
                    
                # Import relationship
                self.registry.add_relationship(
                    source_id,
                    target_id,
                    rel_type,
                    strength
                )
                
                imported_relationship_count += 1
                
            except Exception as e:
                # Add to failures
                failed_relationship_imports.append({
                    "relationship": relationship,
                    "error": str(e)
                })
        
        # Update progress
        task.progress = 1.0
        
        # Return result
        return {
            "imported_component_count": len(imported_component_ids),
            "skipped_component_count": len(skipped_component_ids),
            "failed_component_count": len(failed_component_imports),
            "imported_relationship_count": imported_relationship_count,
            "failed_relationship_count": len(failed_relationship_imports),
            "imported_component_ids": imported_component_ids,
            "skipped_component_ids": skipped_component_ids,
            "failed_component_imports": failed_component_imports,
            "failed_relationship_imports": failed_relationship_imports,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_analysis_task(self, task: Task) -> Dict[str, Any]:
        """
        Handle a component analysis task.
        
        Args:
            task: Task to handle
            
        Returns:
            Result of task execution
        """
        # Extract task data
        analysis_type = task.task_data.get("analysis_type")
        component_ids = task.component_ids
        
        # Validate data
        if not analysis_type:
            raise ValueError("Missing analysis_type in task data")
            
        if not component_ids:
            raise ValueError("Missing component_ids in task")
            
        # Perform analysis based on type
        analysis_result = {}
        
        if analysis_type == "dependency":
            # Analyze component dependencies
            analysis_result = await self._analyze_dependencies(component_ids)
            
        elif analysis_type == "complexity":
            # Analyze component complexity
            analysis_result = await self._analyze_complexity(component_ids)
            
        elif analysis_type == "impact":
            # Analyze impact of changes to components
            analysis_result = await self._analyze_impact(component_ids)
            
        elif analysis_type == "consistency":
            # Analyze consistency across components
            analysis_result = await self._analyze_consistency(component_ids)
            
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
            
        # Update progress
        task.progress = 1.0
        
        # Return result
        return {
            "analysis_type": analysis_type,
            "component_count": len(component_ids),
            "analysis_result": analysis_result,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _analyze_dependencies(self, component_ids: List[str]) -> Dict[str, Any]:
        """
        Analyze dependencies between components.
        
        Args:
            component_ids: IDs of components to analyze
            
        Returns:
            Analysis results
        """
        # Get components
        components = {}
        for component_id in component_ids:
            component = self.registry.get_component(component_id)
            if component:
                components[component_id] = component
        
        # Analyze dependencies
        dependencies = {}
        reverse_dependencies = {}
        
        for component_id in components:
            # Get related components
            related = self.registry.get_related_components(component_id)
            
            # Add to dependencies
            dependencies[component_id] = []
            
            for related_component in related:
                related_id = related_component["id"]
                
                # Skip if not in our list of components
                if related_id not in components:
                    continue
                    
                # Get relationship
                relationship = self.registry.get_relationship(component_id, related_id)
                
                if relationship:
                    # Add to dependencies
                    dependencies[component_id].append({
                        "component_id": related_id,
                        "relationship_type": relationship.get("type", "related"),
                        "strength": relationship.get("strength", 0.5)
                    })
                    
                    # Add to reverse dependencies
                    if related_id not in reverse_dependencies:
                        reverse_dependencies[related_id] = []
                        
                    reverse_dependencies[related_id].append({
                        "component_id": component_id,
                        "relationship_type": relationship.get("type", "related"),
                        "strength": relationship.get("strength", 0.5)
                    })
        
        # Calculate dependency metrics
        dependency_counts = {}
        for component_id, deps in dependencies.items():
            dependency_counts[component_id] = len(deps)
            
        reverse_dependency_counts = {}
        for component_id, deps in reverse_dependencies.items():
            reverse_dependency_counts[component_id] = len(deps)
            
        # Identify high-dependency components
        high_dependency_threshold = 3  # Arbitrary threshold
        high_dependency_components = [
            component_id for component_id, count in dependency_counts.items()
            if count >= high_dependency_threshold
        ]
        
        high_reverse_dependency_components = [
            component_id for component_id, count in reverse_dependency_counts.items()
            if count >= high_dependency_threshold
        ]
        
        return {
            "dependencies": dependencies,
            "reverse_dependencies": reverse_dependencies,
            "dependency_counts": dependency_counts,
            "reverse_dependency_counts": reverse_dependency_counts,
            "high_dependency_components": high_dependency_components,
            "high_reverse_dependency_components": high_reverse_dependency_components
        }
    
    async def _analyze_complexity(self, component_ids: List[str]) -> Dict[str, Any]:
        """
        Analyze complexity of components.
        
        Args:
            component_ids: IDs of components to analyze
            
        Returns:
            Analysis results
        """
        # Get components
        components = {}
        for component_id in component_ids:
            component = self.registry.get_component(component_id)
            if component:
                components[component_id] = component
        
        # Analyze complexity
        complexity_scores = {}
        complexity_factors = {}
        
        for component_id, component in components.items():
            # Calculate complexity based on various factors
            factors = {}
            
            # Factor 1: Number of fields
            field_count = len(component)
            factors["field_count"] = field_count
            
            # Factor 2: Depth of nested structures
            max_depth = self._calculate_max_depth(component)
            factors["max_depth"] = max_depth
            
            # Factor 3: Number of relationships
            related = self.registry.get_related_components(component_id)
            relationship_count = len(related)
            factors["relationship_count"] = relationship_count
            
            # Calculate overall complexity score
            # This is a simplified calculation - could be more sophisticated
            complexity_score = (
                field_count * 0.3 +
                max_depth * 0.5 +
                relationship_count * 0.2
            )
            
            # Store results
            complexity_scores[component_id] = complexity_score
            complexity_factors[component_id] = factors
        
        # Calculate complexity statistics
        if complexity_scores:
            avg_complexity = sum(complexity_scores.values()) / len(complexity_scores)
            max_complexity = max(complexity_scores.values())
            min_complexity = min(complexity_scores.values())
        else:
            avg_complexity = 0
            max_complexity = 0
            min_complexity = 0
            
        # Identify high complexity components
        high_complexity_threshold = avg_complexity * 1.5  # 50% above average
        high_complexity_components = [
            component_id for component_id, score in complexity_scores.items()
            if score >= high_complexity_threshold
        ]
        
        return {
            "complexity_scores": complexity_scores,
            "complexity_factors": complexity_factors,
            "average_complexity": avg_complexity,
            "max_complexity": max_complexity,
            "min_complexity": min_complexity,
            "high_complexity_components": high_complexity_components
        }
    
    def _calculate_max_depth(self, obj: Any, current_depth: int = 0) -> int:
        """
        Calculate the maximum nesting depth of an object.
        
        Args:
            obj: Object to analyze
            current_depth: Current depth in the recursion
            
        Returns:
            Maximum nesting depth
        """
        if isinstance(obj, dict):
            if not obj:
                return current_depth
                
            return max(
                self._calculate_max_depth(value, current_depth + 1)
                for value in obj.values()
            )
        elif isinstance(obj, list):
            if not obj:
                return current_depth
                
            return max(
                self._calculate_max_depth(item, current_depth + 1)
                for item in obj
            )
        else:
            return current_depth
    
    async def _analyze_impact(self, component_ids: List[str]) -> Dict[str, Any]:
        """
        Analyze impact of changes to components.
        
        Args:
            component_ids: IDs of components to analyze
            
        Returns:
            Analysis results
        """
        # This is a simplified impact analysis
        # In a real implementation, this would be more sophisticated
        
        # Get dependency analysis
        dependency_analysis = await self._analyze_dependencies(component_ids)
        
        # Calculate impact scores based on reverse dependencies
        impact_scores = {}
        for component_id in component_ids:
            # Get reverse dependencies
            reverse_deps = dependency_analysis["reverse_dependencies"].get(component_id, [])
            
            # Calculate impact score
            # This is simplistic - a real implementation would consider more factors
            impact_score = len(reverse_deps) * 10  # Simple scaling
            
            impact_scores[component_id] = impact_score
        
        # Identify high impact components
        high_impact_threshold = 20  # Arbitrary threshold
        high_impact_components = [
            component_id for component_id, score in impact_scores.items()
            if score >= high_impact_threshold
        ]
        
        return {
            "impact_scores": impact_scores,
            "high_impact_components": high_impact_components,
            "impact_details": {
                component_id: {
                    "reverse_dependencies": dependency_analysis["reverse_dependencies"].get(component_id, [])
                }
                for component_id in component_ids
            }
        }
    
    async def _analyze_consistency(self, component_ids: List[str]) -> Dict[str, Any]:
        """
        Analyze consistency across components.
        
        Args:
            component_ids: IDs of components to analyze
            
        Returns:
            Analysis results
        """
        # Get components
        components = {}
        for component_id in component_ids:
            component = self.registry.get_component(component_id)
            if component:
                components[component_id] = component
        
        # Group components by type
        components_by_type = {}
        for component_id, component in components.items():
            component_type = component.get("component_type") or component.get("type", "unknown")
            
            if component_type not in components_by_type:
                components_by_type[component_type] = []
                
            components_by_type[component_type].append({
                "id": component_id,
                "component": component
            })
        
        # Analyze consistency within each type
        consistency_analysis = {}
        
        for component_type, type_components in components_by_type.items():
            # Skip if only one component of this type
            if len(type_components) < 2:
                continue
                
            # Analyze field consistency
            field_presence = {}
            
            # First pass: collect all fields
            all_fields = set()
            for component_info in type_components:
                component = component_info["component"]
                all_fields.update(component.keys())
            
            # Second pass: check field presence
            for field in all_fields:
                field_presence[field] = []
                
                for component_info in type_components:
                    component = component_info["component"]
                    if field in component:
                        field_presence[field].append(component_info["id"])
            
            # Calculate consistency score
            # Percentage of components that have each field
            field_consistency = {}
            
            for field, present_in in field_presence.items():
                consistency = len(present_in) / len(type_components)
                field_consistency[field] = consistency
            
            # Calculate average consistency
            if field_consistency:
                avg_consistency = sum(field_consistency.values()) / len(field_consistency)
            else:
                avg_consistency = 1.0
                
            # Identify inconsistent fields
            inconsistency_threshold = 0.5  # Arbitrary threshold
            inconsistent_fields = [
                field for field, consistency in field_consistency.items()
                if consistency <= inconsistency_threshold
            ]
            
            # Store results
            consistency_analysis[component_type] = {
                "component_count": len(type_components),
                "field_presence": field_presence,
                "field_consistency": field_consistency,
                "average_consistency": avg_consistency,
                "inconsistent_fields": inconsistent_fields
            }
        
        return consistency_analysis
