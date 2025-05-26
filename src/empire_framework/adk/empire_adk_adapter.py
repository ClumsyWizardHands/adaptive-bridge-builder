"""
Empire Framework ADK Adapter

This module provides integration between the Empire Framework and the Agent Development Kit (ADK),
enabling ADK-based agents to utilize Empire components, principles, and capabilities.
"""

import json
import uuid
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from datetime import datetime, timezone

# Empire Framework imports
from empire_framework.registry.component_registry import ComponentRegistry
from empire_framework.validation.schema_validator import SchemaValidator
from empire_framework.a2a.component_task_handler import ComponentTaskHandler, ComponentTaskTypes, Task
from empire_framework.a2a.message_structures import ComponentMethods

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EmpireADKAdapter")

class EmpireEventType:
    """Event types for Empire Framework components in ADK."""
    COMPONENT_CREATED = "empire.component.created"
    COMPONENT_UPDATED = "empire.component.updated"
    COMPONENT_DELETED = "empire.component.deleted"
    COMPONENT_VALIDATED = "empire.component.validated"
    RELATIONSHIP_CREATED = "empire.relationship.created"
    RELATIONSHIP_UPDATED = "empire.relationship.updated"
    RELATIONSHIP_DELETED = "empire.relationship.deleted"
    PRINCIPLE_APPLIED = "empire.principle.applied"
    PRINCIPLE_CONFLICT = "empire.principle.conflict"
    TASK_CREATED = "empire.task.created"
    TASK_COMPLETED = "empire.task.completed"
    TASK_FAILED = "empire.task.failed"

class EmpireADKAdapter:
    """
    Adapter for integrating Empire Framework with the Agent Development Kit.
    
    This adapter provides:
    1. Event handling for Empire component operations
    2. Function calls for Empire operations
    3. State management for Empire components
    4. Artifact management for Empire outputs
    5. Tool definitions for use in ADK workflows
    """
    
    def __init__(
        self,
        registry: Optional[ComponentRegistry] = None,
        validator: Optional[SchemaValidator] = None,
        task_handler: Optional[ComponentTaskHandler] = None,
        max_concurrent_tasks: int = 10,
        event_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None
    ):
        """
        Initialize the Empire ADK Adapter.
        
        Args:
            registry: Optional component registry to use
            validator: Optional schema validator to use
            task_handler: Optional task handler to use
            max_concurrent_tasks: Maximum number of concurrent tasks
            event_callback: Callback function for emitting events
        """
        self.registry = registry or ComponentRegistry()
        self.validator = validator or SchemaValidator()
        self.task_handler = task_handler or ComponentTaskHandler(
            registry=self.registry,
            validator=self.validator,
            max_concurrent_tasks=max_concurrent_tasks
        )
        self.event_callback = event_callback
        
        # Start task processing if task handler provided or created
        self._task_processors = None
        self._start_task_processing()
        
        # Component state tracking
        self.component_states: Dict[str, Dict[str, Any]] = {}
        self.registered_artifacts: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Empire ADK Adapter initialized")
    
    async def _start_task_processing(self) -> None:
        """Start the task processors if not already running."""
        if self._task_processors is None and self.task_handler is not None:
            self._task_processors = await self.task_handler.start_processing()
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Emit an event to the ADK event system.
        
        Args:
            event_type: The type of event
            data: The event data
        """
        if self.event_callback:
            await self.event_callback(event_type, data)
            logger.debug(f"Emitted event: {event_type}")
    
    # ====== ADK Function Calls ======
    
    async def get_component(self, component_id: str) -> Dict[str, Any]:
        """
        Get a component by ID.
        
        Args:
            component_id: ID of the component to retrieve
            
        Returns:
            The component data
            
        Raises:
            ValueError: If the component is not found
        """
        component = self.registry.get_component(component_id)
        if not component:
            raise ValueError(f"Component not found: {component_id}")
            
        # Track in state
        self.component_states = {**self.component_states, component_id: {}
            "last_accessed": datetime.utcnow().isoformat(),
            "access_count": self.component_states.get(component_id, {}).get("access_count", 0) + 1
        }
        
        return component
    
    async def get_components(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get components matching filters.
        
        Args:
            filters: Optional filters to apply
            limit: Optional maximum number of components to return
            offset: Optional offset for pagination
            
        Returns:
            List of matching components
        """
        # Get components from registry
        components = self.registry.get_components(filters)
        
        # Apply pagination
        if offset is not None:
            components = components[offset:]
            
        if limit is not None:
            components = components[:limit]
            
        # Track in state
        for component in components:
            component_id = component.get("id")
            if component_id:
                self.component_states = {**self.component_states, component_id: {}
                    "last_accessed": datetime.utcnow().isoformat(),
                    "access_count": self.component_states.get(component_id, {}).get("access_count", 0) + 1
                }
        
        return components
    
    async def create_component(
        self,
        component_data: Dict[str, Any],
        component_type: str,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Create a new component.
        
        Args:
            component_data: The component data
            component_type: The type of component to create
            validate: Whether to validate the component data
            
        Returns:
            The created component
            
        Raises:
            ValueError: If validation fails
        """
        # Generate ID if not provided
        if "id" not in component_data:
            component_data["id"] = f"{component_type}-{uuid.uuid4().hex[:8]}"
            
        # Add timestamps
        if "created_at" not in component_data:
            component_data["created_at"] = datetime.utcnow().isoformat()
            
        if "last_modified_date" not in component_data:
            component_data["last_modified_date"] = datetime.utcnow().isoformat()
            
        # Add type if not present
        if "type" not in component_data and "component_type" not in component_data:
            component_data["type"] = component_type
            
        # Validate if requested
        if validate and self.validator:
            validation_result = self.validator.validate_component(component_data, component_type)
            if not validation_result["valid"]:
                raise ValueError(f"Invalid component data: {validation_result['errors']}")
                
        # Register component
        self.registry.register_component(component_data)
        component_id = component_data["id"]
        
        # Track in state
        self.component_states = {**self.component_states, component_id: {}
            "created_at": datetime.utcnow().isoformat(),
            "last_modified": datetime.utcnow().isoformat(),
            "access_count": 1
        }
        
        # Emit event
        await self._emit_event(
            EmpireEventType.COMPONENT_CREATED,
            {
                "component_id": component_id,
                "component_type": component_type,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return component_data
    
    async def update_component(
        self,
        component_id: str,
        updates: Dict[str, Any],
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Update an existing component.
        
        Args:
            component_id: ID of the component to update
            updates: Changes to apply to the component
            validate: Whether to validate the updated component
            
        Returns:
            The updated component
            
        Raises:
            ValueError: If the component is not found or validation fails
        """
        # Get current component
        component = self.registry.get_component(component_id)
        if not component:
            raise ValueError(f"Component not found: {component_id}")
            
        # Apply updates
        updated_component = {**component, **updates}
        
        # Update last modified date
        updated_component["last_modified_date"] = datetime.utcnow().isoformat()
        
        # Validate if requested
        if validate and self.validator:
            component_type = updated_component.get("component_type") or updated_component.get("type")
            validation_result = self.validator.validate_component(updated_component, component_type)
            if not validation_result["valid"]:
                raise ValueError(f"Invalid component updates: {validation_result['errors']}")
                
        # Update component
        self.registry.update_component(component_id, updated_component)
        
        # Track in state
        self.component_states = {**self.component_states, component_id: {}
            **self.component_states.get(component_id, {}),
            "last_modified": datetime.utcnow().isoformat(),
            "modification_count": self.component_states.get(component_id, {}).get("modification_count", 0) + 1
        }
        
        # Emit event
        await self._emit_event(
            EmpireEventType.COMPONENT_UPDATED,
            {
                "component_id": component_id,
                "updated_fields": list(updates.keys()),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return updated_component
    
    async def delete_component(
        self,
        component_id: str,
        hard_delete: bool = False
    ) -> Dict[str, Any]:
        """
        Delete a component.
        
        Args:
            component_id: ID of the component to delete
            hard_delete: Whether to permanently delete the component
            
        Returns:
            Status information
            
        Raises:
            ValueError: If the component is not found
        """
        # Get component first for return
        component = self.registry.get_component(component_id)
        if not component:
            raise ValueError(f"Component not found: {component_id}")
            
        # Delete component
        success = self.registry.remove_component(component_id, permanent=hard_delete)
        if not success:
            raise ValueError(f"Failed to delete component: {component_id}")
            
        # Track in state
        if hard_delete:
            if component_id in self.component_states:
                self.component_states = {k: v for k, v in self.component_states.items() if k != component_id}
        else:
            self.component_states = {**self.component_states, component_id: {}
                **self.component_states.get(component_id, {}),
                "deleted_at": datetime.utcnow().isoformat(),
                "is_deleted": True
            }
        
        # Emit event
        await self._emit_event(
            EmpireEventType.COMPONENT_DELETED,
            {
                "component_id": component_id,
                "hard_delete": hard_delete,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return {
            "component_id": component_id,
            "success": True,
            "hard_delete": hard_delete,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def validate_component(
        self,
        component_data: Dict[str, Any],
        schema_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate a component against a schema.
        
        Args:
            component_data: The component data to validate
            schema_id: Optional ID of the schema to validate against
            
        Returns:
            Validation results
            
        Raises:
            ValueError: If validation fails or no validator is available
        """
        if not self.validator:
            raise ValueError("No validator available")
            
        # Get component type from data
        component_type = component_data.get("component_type") or component_data.get("type")
        
        # Validate
        validation_result = self.validator.validate_component(
            component_data,
            component_type,
            schema_id=schema_id
        )
        
        # Emit event
        component_id = component_data.get("id", "unknown")
        await self._emit_event(
            EmpireEventType.COMPONENT_VALIDATED,
            {
                "component_id": component_id,
                "valid": validation_result["valid"],
                "error_count": len(validation_result.get("errors", [])),
                "warning_count": len(validation_result.get("warnings", [])),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return validation_result
    
    async def get_related_components(
        self,
        component_id: str,
        relation_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get components related to a specific component.
        
        Args:
            component_id: ID of the component
            relation_types: Optional list of relation types to include
            
        Returns:
            List of related components
            
        Raises:
            ValueError: If the component is not found
        """
        # Check if component exists
        if not self.registry.get_component(component_id):
            raise ValueError(f"Component not found: {component_id}")
            
        # Get related components
        related = self.registry.get_related_components(
            component_id,
            relation_types=relation_types
        )
        
        # Track in state
        for component in related:
            related_id = component.get("id")
            if related_id:
                self.component_states = {**self.component_states, related_id: {}
                    "last_accessed": datetime.utcnow().isoformat(),
                    "access_count": self.component_states.get(related_id, {}).get("access_count", 0) + 1
                }
        
        return related
    
    async def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        strength: float = 0.5
    ) -> Dict[str, Any]:
        """
        Add a relationship between components.
        
        Args:
            source_id: ID of the source component
            target_id: ID of the target component
            relationship_type: Type of relationship
            strength: Strength of the relationship (0.0-1.0)
            
        Returns:
            Relationship information
            
        Raises:
            ValueError: If a component is not found
        """
        # Check if components exist
        source = self.registry.get_component(source_id)
        if not source:
            raise ValueError(f"Source component not found: {source_id}")
            
        target = self.registry.get_component(target_id)
        if not target:
            raise ValueError(f"Target component not found: {target_id}")
            
        # Add relationship
        success = self.registry.add_relationship(
            source_id,
            target_id,
            relationship_type,
            strength
        )
        
        if not success:
            raise ValueError(f"Failed to add relationship between {source_id} and {target_id}")
            
        # Emit event
        await self._emit_event(
            EmpireEventType.RELATIONSHIP_CREATED,
            {
                "source_id": source_id,
                "target_id": target_id,
                "relationship_type": relationship_type,
                "strength": strength,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return {
            "source_id": source_id,
            "target_id": target_id,
            "relationship_type": relationship_type,
            "strength": strength,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def remove_relationship(
        self,
        source_id: str,
        target_id: str
    ) -> Dict[str, Any]:
        """
        Remove a relationship between components.
        
        Args:
            source_id: ID of the source component
            target_id: ID of the target component
            
        Returns:
            Status information
            
        Raises:
            ValueError: If the relationship is not found
        """
        # Remove relationship
        success = self.registry.remove_relationship(source_id, target_id)
        
        if not success:
            raise ValueError(f"Failed to remove relationship between {source_id} and {target_id}")
            
        # Emit event
        await self._emit_event(
            EmpireEventType.RELATIONSHIP_DELETED,
            {
                "source_id": source_id,
                "target_id": target_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return {
            "source_id": source_id,
            "target_id": target_id,
            "success": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # ====== Asynchronous Tasks ======
    
    async def create_task(
        self,
        task_type: str,
        component_ids: List[str],
        task_data: Dict[str, Any],
        priority: str = "medium"
    ) -> str:
        """
        Create an asynchronous task.
        
        Args:
            task_type: Type of task to create
            component_ids: IDs of components involved
            task_data: Data needed for the task
            priority: Priority level
            
        Returns:
            ID of the created task
            
        Raises:
            ValueError: If the task handler is not available
        """
        if not self.task_handler:
            raise ValueError("Task handler not available")
            
        # Create task
        task_id = await self.task_handler.create_task(
            task_type=task_type,
            component_ids=component_ids,
            task_data=task_data,
            priority=priority
        )
        
        # Emit event
        await self._emit_event(
            EmpireEventType.TASK_CREATED,
            {
                "task_id": task_id,
                "task_type": task_type,
                "component_ids": component_ids,
                "priority": priority,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of an asynchronous task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task status information
            
        Raises:
            ValueError: If the task is not found
        """
        if not self.task_handler:
            raise ValueError("Task handler not available")
            
        # Get task status
        status = self.task_handler.get_task(task_id)
        
        if not status:
            raise ValueError(f"Task not found: {task_id}")
            
        # Check if task completed or failed since last check
        if status["status"] == "completed" and status.get("result"):
            # Emit completed event
            await self._emit_event(
                EmpireEventType.TASK_COMPLETED,
                {
                    "task_id": task_id,
                    "result": status["result"],
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        elif status["status"] == "failed" and status.get("error"):
            # Emit failed event
            await self._emit_event(
                EmpireEventType.TASK_FAILED,
                {
                    "task_id": task_id,
                    "error": status["error"],
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        return status
    
    async def cancel_task(
        self,
        task_id: str,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cancel an asynchronous task.
        
        Args:
            task_id: ID of the task to cancel
            reason: Optional reason for cancellation
            
        Returns:
            Cancellation status
            
        Raises:
            ValueError: If the task is not found
        """
        if not self.task_handler:
            raise ValueError("Task handler not available")
            
        # Cancel task
        success = await self.task_handler.cancel_task(task_id, reason)
        
        if not success:
            raise ValueError(f"Failed to cancel task: {task_id}")
            
        return {
            "task_id": task_id,
            "success": True,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # ====== Artifact Management ======
    
    async def register_artifact(
        self,
        artifact_type: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        component_id: Optional[str] = None,
        artifact_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Register an artifact from Empire Framework processing.
        
        Args:
            artifact_type: Type of artifact
            content: The artifact content
            metadata: Optional metadata about the artifact
            component_id: Optional ID of related component
            artifact_id: Optional specific artifact ID
            
        Returns:
            Artifact registration information
        """
        # Generate artifact ID if not provided
        if not artifact_id:
            artifact_id = f"artifact-{uuid.uuid4().hex}"
            
        # Create artifact record
        artifact = {
            "id": artifact_id,
            "type": artifact_type,
            "content": content,
            "metadata": metadata or {},
            "component_id": component_id,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Store artifact
        self.registered_artifacts = {**self.registered_artifacts, artifact_id: artifact}
        
        return {
            "artifact_id": artifact_id,
            "type": artifact_type,
            "component_id": component_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_artifact(self, artifact_id: str) -> Dict[str, Any]:
        """
        Get a registered artifact.
        
        Args:
            artifact_id: ID of the artifact
            
        Returns:
            The artifact data
            
        Raises:
            ValueError: If the artifact is not found
        """
        if artifact_id not in self.registered_artifacts:
            raise ValueError(f"Artifact not found: {artifact_id}")
            
        return self.registered_artifacts[artifact_id]
    
    async def list_artifacts(
        self,
        artifact_type: Optional[str] = None,
        component_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List registered artifacts with optional filters.
        
        Args:
            artifact_type: Optional type to filter by
            component_id: Optional component ID to filter by
            
        Returns:
            List of matching artifacts
        """
        artifacts = []
        
        for artifact in self.registered_artifacts.values():
            # Apply filters
            if artifact_type and artifact["type"] != artifact_type:
                continue
                
            if component_id and artifact.get("component_id") != component_id:
                continue
                
            # Include in results (without content to reduce size)
            artifact_copy = artifact.copy()
            artifact_copy.pop("content", None)
            artifacts.append(artifact_copy)
            
        return artifacts
    
    # ====== Principle Application ======
    
    async def apply_principle(
        self,
        principle_id: str,
        target_data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply an Empire Framework principle to evaluate data.
        
        Args:
            principle_id: ID of the principle to apply
            target_data: Data to evaluate against the principle
            context: Optional additional context
            
        Returns:
            Principle application results
            
        Raises:
            ValueError: If the principle is not found
        """
        # Get principle
        principle = self.registry.get_component(principle_id)
        if not principle:
            raise ValueError(f"Principle not found: {principle_id}")
            
        # Get basic principle properties
        name = principle.get("name", "Unnamed Principle")
        description = principle.get("description", "")
        
        # Simple evaluation (in a real implementation, this would use the principle engine)
        # This is just a placeholder for the actual implementation
        evaluation_result = {
            "principle_id": principle_id,
            "principle_name": name,
            "target_analyzed": str(target_data)[:100] + "..." if len(str(target_data)) > 100 else str(target_data),
            "alignment_score": 0.75,  # Placeholder score
            "insights": [
                f"Evaluated against principle: {name}",
                f"Principle context: {description}"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Register as artifact
        artifact_id = f"principle-evaluation-{uuid.uuid4().hex[:8]}"
        await self.register_artifact(
            artifact_type="principle_evaluation",
            content=evaluation_result,
            metadata={
                "principle_id": principle_id,
                "principle_name": name
            },
            component_id=principle_id,
            artifact_id=artifact_id
        )
        
        # Emit event
        await self._emit_event(
            EmpireEventType.PRINCIPLE_APPLIED,
            {
                "principle_id": principle_id,
                "principle_name": name,
                "alignment_score": evaluation_result["alignment_score"],
                "artifact_id": artifact_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return evaluation_result


# ====== ADK Tool Definitions ======

def get_empire_adk_tools(adapter: EmpireADKAdapter) -> Dict[str, Dict[str, Any]]:
    """
    Get Empire Framework tool definitions for ADK.
    
    Args:
        adapter: The Empire ADK adapter
        
    Returns:
        Dictionary of tool definitions
    """
    tools = {
        "empire_get_component": {
            "description": "Get an Empire Framework component by ID",
            "parameters": {
                "component_id": {
                    "type": "string",
                    "description": "ID of the component to retrieve"
                }
            },
            "function": adapter.get_component
        },
        "empire_get_components": {
            "description": "Get Empire Framework components matching filters",
            "parameters": {
                "filters": {
                    "type": "object",
                    "description": "Filters to apply to the query",
                    "required": False
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of components to return",
                    "required": False
                },
                "offset": {
                    "type": "integer",
                    "description": "Offset for pagination",
                    "required": False
                }
            },
            "function": adapter.get_components
        },
        "empire_create_component": {
            "description": "Create a new Empire Framework component",
            "parameters": {
                "component_data": {
                    "type": "object",
                    "description": "Component data to create"
                },
                "component_type": {
                    "type": "string",
                    "description": "Type of component to create"
                },
                "validate": {
                    "type": "boolean",
                    "description": "Whether to validate the component data",
                    "required": False
                }
            },
            "function": adapter.create_component
        },
        "empire_update_component": {
            "description": "Update an existing Empire Framework component",
            "parameters": {
                "component_id": {
                    "type": "string",
                    "description": "ID of the component to update"
                },
                "updates": {
                    "type": "object",
                    "description": "Changes to apply to the component"
                },
                "validate": {
                    "type": "boolean",
                    "description": "Whether to validate the updated component",
                    "required": False
                }
            },
            "function": adapter.update_component
        },
        "empire_delete_component": {
            "description": "Delete an Empire Framework component",
            "parameters": {
                "component_id": {
                    "type": "string",
                    "description": "ID of the component to delete"
                },
                "hard_delete": {
                    "type": "boolean",
                    "description": "Whether to permanently delete the component",
                    "required": False
                }
            },
            "function": adapter.delete_component
        },
        "empire_validate_component": {
            "description": "Validate an Empire Framework component against a schema",
            "parameters": {
                "component_data": {
                    "type": "object",
                    "description": "Component data to validate"
                },
                "schema_id": {
                    "type": "string",
                    "description": "ID of the schema to validate against",
                    "required": False
                }
            },
            "function": adapter.validate_component
        },
        "empire_get_related_components": {
            "description": "Get components related to a specific component",
            "parameters": {
                "component_id": {
                    "type": "string",
                    "description": "ID of the component"
                },
                "relation_types": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Types of relationships to include",
                    "required": False
                }
            },
            "function": adapter.get_related_components
        },
        "empire_add_relationship": {
            "description": "Add a relationship between Empire Framework components",
            "parameters": {
                "source_id": {
                    "type": "string",
                    "description": "ID of the source component"
                },
                "target_id": {
                    "type": "string",
                    "description": "ID of the target component"
                },
                "relationship_type": {
                    "type": "string",
                    "description": "Type of relationship"
                },
                "strength": {
                    "type": "number",
                    "description": "Strength of the relationship (0.0-1.0)",
                    "required": False
                }
            },
            "function": adapter.add_relationship
        },
        "empire_remove_relationship": {
            "description": "Remove a relationship between Empire Framework components",
            "parameters": {
                "source_id": {
                    "type": "string",
                    "description": "ID of the source component"
                },
                "target_id": {
                    "type": "string",
                    "description": "ID of the target component"
                }
            },
            "function": adapter.remove_relationship
        },
        "empire_create_task": {
            "description": "Create an asynchronous task for Empire Framework operations",
            "parameters": {
                "task_type": {
                    "type": "string",
                    "description": "Type of task to create"
                },
                "component_ids": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "IDs of components involved in the task"
                },
                "task_data": {
                    "type": "object",
                    "description": "Data needed for the task"
                },
                "priority": {
                    "type": "string",
                    "description": "Priority level for execution",
                    "enum": ["critical", "high", "medium", "low", "background"],
                    "required": False
                }
            },
            "function": adapter.create_task
        },
        "empire_get_task_status": {
            "description": "Get the status of an asynchronous Empire Framework task",
            "parameters": {
                "task_id": {
                    "type": "string",
                    "description": "ID of the task to check"
                }
            },
            "function": adapter.get_task_status
        },
        "empire_cancel_task": {
            "description": "Cancel an asynchronous Empire Framework task",
            "parameters": {
                "task_id": {
                    "type": "string",
                    "description": "ID of the task to cancel"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for cancellation",
                    "required": False
                }
            },
            "function": adapter.cancel_task
        },
        "empire_apply_principle": {
            "description": "Apply an Empire Framework principle to evaluate data",
            "parameters": {
                "principle_id": {
                    "type": "string",
                    "description": "ID of the principle to apply"
                },
                "target_data": {
                    "type": "string",
                    "description": "Data to evaluate against the principle"
                },
                "context": {
                    "type": "object",
                    "description": "Additional context for evaluation",
                    "required": False
                }
            },
            "function": adapter.apply_principle
        },
        "empire_register_artifact": {
            "description": "Register an artifact from Empire Framework processing",
            "parameters": {
                "artifact_type": {
                    "type": "string",
                    "description": "Type of artifact"
                },
                "content": {
                    "type": "object",
                    "description": "The artifact content"
                },
                "metadata": {
                    "type": "object",
                    "description": "Metadata about the artifact",
                    "required": False
                },
                "component_id": {
                    "type": "string",
                    "description": "ID of related component",
                    "required": False
                }
            },
            "function": adapter.register_artifact
        },
        "empire_get_artifact": {
            "description": "Get a registered Empire Framework artifact",
            "parameters": {
                "artifact_id": {
                    "type": "string",
                    "description": "ID of the artifact"
                }
            },
            "function": adapter.get_artifact
        },
        "empire_list_artifacts": {
            "description": "List registered Empire Framework artifacts",
            "parameters": {
                "artifact_type": {
                    "type": "string",
                    "description": "Type to filter by",
                    "required": False
                },
                "component_id": {
                    "type": "string",
                    "description": "Component ID to filter by",
                    "required": False
                }
            },
            "function": adapter.list_artifacts
        }
    }
    
    return tools


class EmpireADKStateManager:
    """
    State manager for Empire Framework components in ADK.
    
    This class manages the state of Empire components across agent workflows,
    enabling persistence, versioning, and context management.
    """
    
    def __init__(self, adapter: EmpireADKAdapter) -> None:
        """
        Initialize the Empire ADK state manager.
        
        Args:
            adapter: The Empire ADK adapter
        """
        self.adapter = adapter
        self.component_states = adapter.component_states
        self.registered_artifacts = adapter.registered_artifacts
        self.workflow_contexts: Dict[str, Dict[str, Any]] = {}
        
    async def persist_workflow_state(
        self,
        workflow_id: str,
        state_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Persist the state of a workflow.
        
        Args:
            workflow_id: ID of the workflow
            state_data: State data to persist
            
        Returns:
            Persisted state information
        """
        # Create a timestamp for the state
        timestamp = datetime.utcnow().isoformat()
        
        # Create the state record
        state_record = {
            "workflow_id": workflow_id,
            "timestamp": timestamp,
            "state_data": state_data,
            "component_references": list(state_data.get("component_references", {}).keys()),
            "artifact_references": list(state_data.get("artifact_references", {}).keys())
        }
        
        # Store in workflow contexts
        self.workflow_contexts = {**self.workflow_contexts, workflow_id: state_record}
        
        return {
            "workflow_id": workflow_id,
            "state_persisted": True,
            "timestamp": timestamp
        }
    
    async def get_workflow_state(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get the state of a workflow.
        
        Args:
            workflow_id: ID of the workflow
            
        Returns:
            The workflow state
            
        Raises:
            ValueError: If the workflow state is not found
        """
        if workflow_id not in self.workflow_contexts:
            raise ValueError(f"Workflow state not found: {workflow_id}")
            
        return self.workflow_contexts[workflow_id]
    
    async def get_component_state(self, component_id: str) -> Dict[str, Any]:
        """
        Get the state of a component.
        
        Args:
            component_id: ID of the component
            
        Returns:
            The component state
            
        Raises:
            ValueError: If the component state is not found
        """
        if component_id not in self.component_states:
            raise ValueError(f"Component state not found: {component_id}")
            
        return self.component_states[component_id]
    
    async def get_artifact_state(self, artifact_id: str) -> Dict[str, Any]:
        """
        Get the state of an artifact.
        
        Args:
            artifact_id: ID of the artifact
            
        Returns:
            The artifact state
            
        Raises:
            ValueError: If the artifact is not found
        """
        if artifact_id not in self.adapter.registered_artifacts:
            raise ValueError(f"Artifact not found: {artifact_id}")
            
        artifact = self.adapter.registered_artifacts[artifact_id]
        
        # Return metadata without content to keep response size small
        state = {
            "id": artifact["id"],
            "type": artifact["type"],
            "created_at": artifact["created_at"],
            "metadata": artifact["metadata"],
            "component_id": artifact.get("component_id")
        }
        
        return state


class EmpireWorkflowTools:
    """
    Tools for using Empire Framework components in ADK workflows.
    
    This class provides utility functions specifically designed for
    workflow agents to utilize Empire Framework principles and components.
    """
    
    def __init__(self, adapter: EmpireADKAdapter) -> None:
        """
        Initialize Empire workflow tools.
        
        Args:
            adapter: The Empire ADK adapter
        """
        self.adapter = adapter
        self.state_manager = EmpireADKStateManager(adapter)
    
    async def get_principle_components(self) -> List[Dict[str, Any]]:
        """
        Get all principle components.
        
        Returns:
            List of principle components
        """
        # Get components with type 'principle'
        filters = {"type": "principle"}
        principles = await self.adapter.get_components(filters=filters)
        
        return principles
    
    async def evaluate_with_principles(
        self,
        content: str,
        principle_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate content against Empire principles.
        
        Args:
            content: Content to evaluate
            principle_ids: Optional specific principle IDs to use
            
        Returns:
            Evaluation results
        """
        # Get principles to use
        if principle_ids:
            principles = []
            for principle_id in principle_ids:
                try:
                    principle = await self.adapter.get_component(principle_id)
                    principles.append(principle)
                except ValueError:
                    # Skip principles that don't exist
                    pass
        else:
            # Get all principles
            principles = await self.get_principle_components()
        
        # Evaluate against each principle
        evaluations = []
        for principle in principles:
            try:
                result = await self.adapter.apply_principle(
                    principle_id=principle["id"],
                    target_data=content
                )
                evaluations.append(result)
            except ValueError:
                # Skip principles that can't be applied
                pass
        
        # Calculate overall alignment
        if evaluations:
            overall_score = sum(e["alignment_score"] for e in evaluations) / len(evaluations)
        else:
            overall_score = 0.0
        
        return {
            "overall_alignment": overall_score,
            "principle_count": len(evaluations),
            "evaluations": evaluations,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def create_component_from_template(
        self,
        template_id: str,
        custom_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a component from a template.
        
        Args:
            template_id: ID of the template component
            custom_values: Custom values to override in the template
            
        Returns:
            The created component
            
        Raises:
            ValueError: If the template is not found
        """
        # Get template component
        template = await self.adapter.get_component(template_id)
        
        # Copy template
        component_data = template.copy()
        
        # Remove template-specific fields
        if "id" in component_data:
            del component_data["id"]
        if "created_at" in component_data:
            del component_data["created_at"]
        if "last_modified_date" in component_data:
            del component_data["last_modified_date"]
        
        # Apply custom values
        for key, value in custom_values.items():
            component_data[key] = value
        
        # Create the component
        component_type = component_data.get("component_type") or component_data.get("type", "component")
        created = await self.adapter.create_component(
            component_data=component_data,
            component_type=component_type
        )
        
        return created


def get_empire_workflow_tool_context(adapter: EmpireADKAdapter) -> Dict[str, Any]:
    """
    Get Empire Framework tool context for ADK workflows.
    
    Args:
        adapter: The Empire ADK adapter
        
    Returns:
        Tool context for ADK workflows
    """
    workflow_tools = EmpireWorkflowTools(adapter)
    
    # Create tool context
    context = {
        "empire_framework": {
            "principles": {
                "get_principles": workflow_tools.get_principle_components,
                "evaluate_with_principles": workflow_tools.evaluate_with_principles
            },
            "components": {
                "create_from_template": workflow_tools.create_component_from_template
            },
            "state": {
                "persist_workflow_state": workflow_tools.state_manager.persist_workflow_state,
                "get_workflow_state": workflow_tools.state_manager.get_workflow_state,
                "get_component_state": workflow_tools.state_manager.get_component_state,
                "get_artifact_state": workflow_tools.state_manager.get_artifact_state
            }
        }
    }
    
    return context
