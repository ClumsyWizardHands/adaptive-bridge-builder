"""
A2A Message Structures for Empire Framework Components

This module defines standardized JSON-RPC message structures for
exchanging Empire Framework components through the A2A Protocol.
These structures enable consistent component serialization, validation,
and exchange between agents.
"""

import json
import uuid
from typing import Dict, List, Any, Optional, Union, TypedDict
from datetime import datetime, timezone

# Type definitions for message structures
class ComponentIdentifier(TypedDict):
    """Component identifier for referencing in messages."""
    component_id: str
    component_type: str
    version: Optional[str]

class ComponentData(TypedDict):
    """Component data structure."""
    name: str
    description: Optional[str]
    content: Dict[str, Any]

class ComponentMetadata(TypedDict):
    """Metadata for component messages."""
    source: str
    timestamp: str
    version: str
    schema: Optional[str]
    author: Optional[str]

class ComponentMessage(TypedDict):
    """Complete component message structure."""
    jsonrpc: str
    method: str
    params: Dict[str, Any]
    id: str

class ComponentStreamParams(TypedDict):
    """Parameters for component streaming operations."""
    component_id: str
    interval: Optional[float]
    max_updates: Optional[int]
    include_changes_only: Optional[bool]

class ComponentTaskDefinition(TypedDict):
    """Definition of a component-related task."""
    task_id: str
    task_type: str
    component_ids: List[str]
    priority: Optional[str]
    created_at: str
    status: str
    metadata: Optional[Dict[str, Any]]

# Enum-like constants for message methods
class ComponentMethods:
    """Enum of standardized component method names."""
    GET_COMPONENT = "empire.getComponent"
    GET_COMPONENTS = "empire.getComponents"
    UPDATE_COMPONENT = "empire.updateComponent"
    CREATE_COMPONENT = "empire.createComponent"
    DELETE_COMPONENT = "empire.deleteComponent"
    VALIDATE_COMPONENT = "empire.validateComponent"
    STREAM_COMPONENT = "empire.streamComponent"
    STREAM_COMPONENTS = "empire.streamComponents"
    SEARCH_COMPONENTS = "empire.searchComponents"
    GET_COMPONENT_RELATIONS = "empire.getComponentRelations"
    MODIFY_COMPONENT_RELATIONS = "empire.modifyComponentRelations"
    CREATE_COMPONENT_TASK = "empire.createComponentTask"
    GET_TASK_STATUS = "empire.getTaskStatus"
    CANCEL_TASK = "empire.cancelTask"

# Enum-like constants for task types
class ComponentTaskTypes:
    """Enum of task types for component operations."""
    CREATION = "component_creation"
    UPDATE = "component_update"
    DELETION = "component_deletion"
    VALIDATION = "component_validation"
    TRANSFORMATION = "component_transformation"
    RELATIONSHIP_UPDATE = "relationship_update"
    BATCH_OPERATION = "batch_operation"
    EXPORT = "component_export"
    IMPORT = "component_import"
    ANALYSIS = "component_analysis"

# Enum-like constants for task status
class TaskStatus:
    """Enum of task status values."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class JsonRpcError(TypedDict):
    """JSON-RPC error structure."""
    code: int
    message: str
    data: Optional[Dict[str, Any]]

def create_component_message(
    method: str,
    params: Dict[str, Any],
    message_id: Optional[str] = None
) -> ComponentMessage:
    """
    Create a standardized JSON-RPC message for component operations.
    
    Args:
        method: The method to invoke
        params: The method parameters
        message_id: Optional message ID
        
    Returns:
        A complete ComponentMessage
    """
    if not message_id:
        message_id = str(uuid.uuid4())
        
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": message_id
    }

def create_get_component_message(
    component_id: str,
    include_metadata: bool = True,
    message_id: Optional[str] = None
) -> ComponentMessage:
    """
    Create a message to request a component by ID.
    
    Args:
        component_id: ID of the component to retrieve
        include_metadata: Whether to include metadata in the response
        message_id: Optional message ID
        
    Returns:
        A complete ComponentMessage
    """
    params = {
        "component_id": component_id,
        "include_metadata": include_metadata
    }
    
    return create_component_message(
        method=ComponentMethods.GET_COMPONENT,
        params=params,
        message_id=message_id
    )

def create_get_components_message(
    filters: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = None,
    message_id: Optional[str] = None
) -> ComponentMessage:
    """
    Create a message to request multiple components with filters.
    
    Args:
        filters: Optional filters to apply
        limit: Optional maximum number of components to return
        offset: Optional offset for pagination
        sort_by: Optional field to sort by
        sort_order: Optional sort order ('asc' or 'desc')
        message_id: Optional message ID
        
    Returns:
        A complete ComponentMessage
    """
    params = {}
    
    if filters:
        params["filters"] = filters
        
    if limit is not None:
        params["limit"] = limit
        
    if offset is not None:
        params["offset"] = offset
        
    if sort_by:
        params["sort_by"] = sort_by
        
    if sort_order:
        params["sort_order"] = sort_order
    
    return create_component_message(
        method=ComponentMethods.GET_COMPONENTS,
        params=params,
        message_id=message_id
    )

def create_update_component_message(
    component_id: str,
    updates: Dict[str, Any],
    message_id: Optional[str] = None
) -> ComponentMessage:
    """
    Create a message to update a component.
    
    Args:
        component_id: ID of the component to update
        updates: Changes to apply to the component
        message_id: Optional message ID
        
    Returns:
        A complete ComponentMessage
    """
    params = {
        "component_id": component_id,
        "updates": updates,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return create_component_message(
        method=ComponentMethods.UPDATE_COMPONENT,
        params=params,
        message_id=message_id
    )

def create_create_component_message(
    component_data: Dict[str, Any],
    component_type: str,
    message_id: Optional[str] = None
) -> ComponentMessage:
    """
    Create a message to create a new component.
    
    Args:
        component_data: Component data to create
        component_type: Type of component to create
        message_id: Optional message ID
        
    Returns:
        A complete ComponentMessage
    """
    params = {
        "component_data": component_data,
        "component_type": component_type,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return create_component_message(
        method=ComponentMethods.CREATE_COMPONENT,
        params=params,
        message_id=message_id
    )

def create_delete_component_message(
    component_id: str,
    hard_delete: bool = False,
    message_id: Optional[str] = None
) -> ComponentMessage:
    """
    Create a message to delete a component.
    
    Args:
        component_id: ID of the component to delete
        hard_delete: Whether to permanently delete the component
        message_id: Optional message ID
        
    Returns:
        A complete ComponentMessage
    """
    params = {
        "component_id": component_id,
        "hard_delete": hard_delete,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return create_component_message(
        method=ComponentMethods.DELETE_COMPONENT,
        params=params,
        message_id=message_id
    )

def create_validate_component_message(
    component_data: Dict[str, Any],
    schema_id: Optional[str] = None,
    message_id: Optional[str] = None
) -> ComponentMessage:
    """
    Create a message to validate a component against a schema.
    
    Args:
        component_data: Component data to validate
        schema_id: Optional ID of the schema to validate against
        message_id: Optional message ID
        
    Returns:
        A complete ComponentMessage
    """
    params = {
        "component_data": component_data
    }
    
    if schema_id:
        params["schema_id"] = schema_id
    
    return create_component_message(
        method=ComponentMethods.VALIDATE_COMPONENT,
        params=params,
        message_id=message_id
    )

def create_stream_component_message(
    component_id: str,
    interval: float = 1.0,
    max_updates: Optional[int] = None,
    include_changes_only: bool = False,
    message_id: Optional[str] = None
) -> ComponentMessage:
    """
    Create a message to stream component updates.
    
    Args:
        component_id: ID of the component to stream
        interval: Interval between updates in seconds
        max_updates: Maximum number of updates to stream
        include_changes_only: Whether to include only changes
        message_id: Optional message ID
        
    Returns:
        A complete ComponentMessage
    """
    params = {
        "component_id": component_id,
        "interval": interval,
        "include_changes_only": include_changes_only
    }
    
    if max_updates is not None:
        params["max_updates"] = max_updates
    
    return create_component_message(
        method=ComponentMethods.STREAM_COMPONENT,
        params=params,
        message_id=message_id
    )

def create_search_components_message(
    query: str,
    component_types: Optional[List[str]] = None,
    search_fields: Optional[List[str]] = None,
    limit: Optional[int] = None,
    message_id: Optional[str] = None
) -> ComponentMessage:
    """
    Create a message to search for components.
    
    Args:
        query: Search query string
        component_types: Optional list of component types to search
        search_fields: Optional list of fields to search
        limit: Optional maximum number of results to return
        message_id: Optional message ID
        
    Returns:
        A complete ComponentMessage
    """
    params = {
        "query": query
    }
    
    if component_types:
        params["component_types"] = component_types
        
    if search_fields:
        params["search_fields"] = search_fields
        
    if limit is not None:
        params["limit"] = limit
    
    return create_component_message(
        method=ComponentMethods.SEARCH_COMPONENTS,
        params=params,
        message_id=message_id
    )

def create_component_relations_message(
    component_id: str,
    relation_types: Optional[List[str]] = None,
    include_components: bool = False,
    message_id: Optional[str] = None
) -> ComponentMessage:
    """
    Create a message to get component relationships.
    
    Args:
        component_id: ID of the component
        relation_types: Optional list of relation types to include
        include_components: Whether to include the related components
        message_id: Optional message ID
        
    Returns:
        A complete ComponentMessage
    """
    params = {
        "component_id": component_id,
        "include_components": include_components
    }
    
    if relation_types:
        params["relation_types"] = relation_types
    
    return create_component_message(
        method=ComponentMethods.GET_COMPONENT_RELATIONS,
        params=params,
        message_id=message_id
    )

def create_task_message(
    task_type: str,
    component_ids: List[str],
    task_data: Dict[str, Any],
    priority: Optional[str] = None,
    message_id: Optional[str] = None
) -> ComponentMessage:
    """
    Create a message to create a component task.
    
    Args:
        task_type: Type of task to create
        component_ids: IDs of components involved in the task
        task_data: Task-specific data
        priority: Optional task priority
        message_id: Optional message ID
        
    Returns:
        A complete ComponentMessage
    """
    params = {
        "task_type": task_type,
        "component_ids": component_ids,
        "task_data": task_data,
        "created_at": datetime.utcnow().isoformat()
    }
    
    if priority:
        params["priority"] = priority
    
    return create_component_message(
        method=ComponentMethods.CREATE_COMPONENT_TASK,
        params=params,
        message_id=message_id
    )

def create_get_task_status_message(
    task_id: str,
    include_details: bool = True,
    message_id: Optional[str] = None
) -> ComponentMessage:
    """
    Create a message to get task status.
    
    Args:
        task_id: ID of the task to check
        include_details: Whether to include task details
        message_id: Optional message ID
        
    Returns:
        A complete ComponentMessage
    """
    params = {
        "task_id": task_id,
        "include_details": include_details
    }
    
    return create_component_message(
        method=ComponentMethods.GET_TASK_STATUS,
        params=params,
        message_id=message_id
    )

def create_cancel_task_message(
    task_id: str,
    reason: Optional[str] = None,
    message_id: Optional[str] = None
) -> ComponentMessage:
    """
    Create a message to cancel a task.
    
    Args:
        task_id: ID of the task to cancel
        reason: Optional reason for cancellation
        message_id: Optional message ID
        
    Returns:
        A complete ComponentMessage
    """
    params = {
        "task_id": task_id
    }
    
    if reason:
        params["reason"] = reason
    
    return create_component_message(
        method=ComponentMethods.CANCEL_TASK,
        params=params,
        message_id=message_id
    )

def create_error_response(
    request_id: str,
    error_code: int,
    error_message: str,
    error_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized JSON-RPC error response.
    
    Args:
        request_id: ID of the original request
        error_code: Error code
        error_message: Error message
        error_data: Optional additional error data
        
    Returns:
        A JSON-RPC error response
    """
    error: JsonRpcError = {
        "code": error_code,
        "message": error_message,
        "data": error_data
    }
    
    return {
        "jsonrpc": "2.0",
        "error": error,
        "id": request_id
    }

def create_success_response(
    request_id: str,
    result: Any
) -> Dict[str, Any]:
    """
    Create a standardized JSON-RPC success response.
    
    Args:
        request_id: ID of the original request
        result: Result data
        
    Returns:
        A JSON-RPC success response
    """
    return {
        "jsonrpc": "2.0",
        "result": result,
        "id": request_id
    }

def parse_component_message(message_json: str) -> ComponentMessage:
    """
    Parse a JSON string into a component message.
    
    Args:
        message_json: JSON string to parse
        
    Returns:
        A ComponentMessage
        
    Raises:
        ValueError: If message is not a valid ComponentMessage
    """
    try:
        message = json.loads(message_json)
        
        # Validate required fields
        if "jsonrpc" not in message or message["jsonrpc"] != "2.0":
            raise ValueError("Invalid jsonrpc version")
            
        if "method" not in message:
            raise ValueError("Missing method")
            
        if "id" not in message:
            raise ValueError("Missing message ID")
            
        if "params" not in message or not isinstance(message["params"], dict):
            raise ValueError("Missing or invalid params")
        
        return message
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {str(e)}")
