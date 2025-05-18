"""
A2A API Handlers for Empire Framework Components

This module provides API handlers for exposing Empire Framework components 
through the A2A Protocol, allowing external agents to retrieve and interact
with components via standardized endpoints.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union

# Import the A2A adapter
from empire_framework.a2a.a2a_adapter import A2AAdapter

# Import component registry for accessing components
from empire_framework.registry.component_registry import ComponentRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("A2AHandlers")

class EmpireA2AHandlers:
    """
    Handlers for Empire Framework A2A endpoints.
    
    This class provides methods to handle A2A requests for Empire components,
    enabling agent-to-agent communication around Empire Framework concepts.
    """
    
    def __init__(self, registry: Optional[ComponentRegistry] = None):
        """
        Initialize the A2A handlers.
        
        Args:
            registry: Optional component registry to use. If None, a new registry will be created.
        """
        self.registry = registry or ComponentRegistry()
        self.adapter = A2AAdapter(validation_enabled=True)
        logger.info("Empire A2A Handlers initialized")
    
    def get_components(self, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle the empire.getComponents A2A method.
        
        This method returns Empire components matching the provided filters.
        
        Args:
            filters: Optional filters to apply when searching for components.
                    Supported filters include:
                    - type: Component type (e.g., "principle", "means", "ends")
                    - tags: List of tags to match
                    - created_after: ISO timestamp
                    - modified_after: ISO timestamp
                    - limit: Maximum number of components to return
                    
        Returns:
            A2A batch message containing matching components
        """
        filters = filters or {}
        logger.info(f"Handling empire.getComponents with filters: {filters}")
        
        try:
            # Query the registry
            components = self.registry.query_components(
                type_filter=filters.get("type"),
                tags=filters.get("tags"),
                created_after=filters.get("created_after"),
                modified_after=filters.get("modified_after"),
                limit=filters.get("limit")
            )
            
            # Convert to A2A batch
            batch_metadata = {
                "source": "empire_framework",
                "filter_applied": filters,
                "count": len(components)
            }
            
            a2a_response = self.adapter.components_to_a2a_batch(components, batch_metadata)
            
            logger.info(f"Returning {len(components)} components in response to getComponents")
            return a2a_response
            
        except Exception as e:
            logger.error(f"Error handling getComponents: {str(e)}")
            return self._create_error_response(
                code=-32000, 
                message="Error retrieving components", 
                data={"error": str(e)}
            )
    
    def get_component_by_id(self, component_id: str) -> Dict[str, Any]:
        """
        Handle the empire.getComponentById A2A method.
        
        This method returns a specific Empire component by ID.
        
        Args:
            component_id: ID of the component to retrieve
            
        Returns:
            A2A message containing the requested component
        """
        logger.info(f"Handling empire.getComponentById for component: {component_id}")
        
        try:
            # Get the component from registry
            component = self.registry.get_component(component_id)
            
            if not component:
                logger.warning(f"Component not found: {component_id}")
                return self._create_error_response(
                    code=-32001,
                    message="Component not found",
                    data={"component_id": component_id}
                )
            
            # Convert to A2A message
            a2a_response = self.adapter.component_to_a2a_message(component)
            
            logger.info(f"Returning component {component_id}")
            return a2a_response
            
        except Exception as e:
            logger.error(f"Error handling getComponentById: {str(e)}")
            return self._create_error_response(
                code=-32000,
                message="Error retrieving component",
                data={"error": str(e), "component_id": component_id}
            )
    
    def get_related_components(self, component_id: str, relationship_types: List[str] = None) -> Dict[str, Any]:
        """
        Handle the empire.getRelatedComponents A2A method.
        
        This method returns components related to the specified component.
        
        Args:
            component_id: ID of the component to find relations for
            relationship_types: Optional list of relationship types to filter by
                               (e.g., "depends_on", "influences", "contradicts")
            
        Returns:
            A2A batch message containing related components
        """
        logger.info(f"Handling empire.getRelatedComponents for component: {component_id}")
        relationship_types = relationship_types or []
        
        try:
            # Check if component exists
            component = self.registry.get_component(component_id)
            if not component:
                logger.warning(f"Component not found: {component_id}")
                return self._create_error_response(
                    code=-32001,
                    message="Component not found",
                    data={"component_id": component_id}
                )
            
            # Get related components
            related_components = self.registry.get_related_components(
                component_id, 
                relationship_types=relationship_types
            )
            
            # Add relationship information to metadata
            relationships = []
            for rel_comp in related_components:
                # Registry should also provide relationship info
                rel_info = self.registry.get_relationship_info(component_id, rel_comp["id"])
                relationships.append({
                    "component_id": rel_comp["id"],
                    "relationship_type": rel_info.get("type", "related_to"),
                    "relationship_strength": rel_info.get("strength", 0.5)
                })
            
            # Convert to A2A batch
            batch_metadata = {
                "source": "empire_framework",
                "root_component_id": component_id,
                "relationship_types": relationship_types,
                "count": len(related_components),
                "relationships": relationships
            }
            
            a2a_response = self.adapter.components_to_a2a_batch(related_components, batch_metadata)
            
            logger.info(f"Returning {len(related_components)} related components")
            return a2a_response
            
        except Exception as e:
            logger.error(f"Error handling getRelatedComponents: {str(e)}")
            return self._create_error_response(
                code=-32000,
                message="Error retrieving related components",
                data={"error": str(e), "component_id": component_id}
            )
    
    def get_component_parts(self, component_id: str, parts: List[str] = None) -> Dict[str, Any]:
        """
        Handle the empire.getComponentParts A2A method.
        
        This method returns specific parts of a component for granular access.
        
        Args:
            component_id: ID of the component to retrieve parts from
            parts: Optional list of specific parts to retrieve 
                  (e.g., "identity", "principle", "means")
            
        Returns:
            A2A parts message containing the requested component parts
        """
        logger.info(f"Handling empire.getComponentParts for component: {component_id}")
        
        try:
            # Get the component from registry
            component = self.registry.get_component(component_id)
            
            if not component:
                logger.warning(f"Component not found: {component_id}")
                return self._create_error_response(
                    code=-32001,
                    message="Component not found",
                    data={"component_id": component_id}
                )
            
            # Convert to A2A parts message
            a2a_parts = self.adapter.component_to_a2a_parts(component)
            
            # Filter to only the requested parts if specified
            if parts:
                # Extract the full parts dictionary
                full_parts = a2a_parts["content"]["parts"]
                
                # Create filtered parts dictionary
                filtered_parts = {}
                for part_name in parts:
                    if part_name in full_parts:
                        filtered_parts[part_name] = full_parts[part_name]
                
                # Replace with filtered parts
                a2a_parts["content"]["parts"] = filtered_parts
                a2a_parts["metadata"]["requested_parts"] = parts
            
            logger.info(f"Returning parts for component {component_id}")
            return a2a_parts
            
        except Exception as e:
            logger.error(f"Error handling getComponentParts: {str(e)}")
            return self._create_error_response(
                code=-32000,
                message="Error retrieving component parts",
                data={"error": str(e), "component_id": component_id}
            )
    
    def _create_error_response(self, code: int, message: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create an A2A error response.
        
        Args:
            code: Error code
            message: Error message
            data: Optional additional error data
            
        Returns:
            A2A error message
        """
        error_response = {
            "a2a_version": "1.0",
            "message_type": "empire.error",
            "content": {
                "error": {
                    "code": code,
                    "message": message
                }
            },
            "metadata": {
                "source": "empire_framework"
            }
        }
        
        if data:
            error_response["content"]["error"]["data"] = data
            
        return error_response
    
    def dispatch(self, a2a_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatch an A2A request to the appropriate handler.
        
        This method parses the A2A request and calls the appropriate handler method.
        
        Args:
            a2a_request: The A2A request to handle
            
        Returns:
            A2A response
        """
        try:
            # Validate request
            if "message_type" not in a2a_request:
                return self._create_error_response(
                    code=-32600, 
                    message="Invalid request: missing message_type"
                )
                
            if "content" not in a2a_request:
                return self._create_error_response(
                    code=-32600,
                    message="Invalid request: missing content"
                )
            
            # Extract message type and params
            message_type = a2a_request["message_type"]
            content = a2a_request["content"]
            
            # Dispatch to appropriate handler
            if message_type == "empire.getComponents":
                filters = content.get("filters", {})
                return self.get_components(filters)
                
            elif message_type == "empire.getComponentById":
                component_id = content.get("component_id")
                if not component_id:
                    return self._create_error_response(
                        code=-32602,
                        message="Invalid params: missing component_id"
                    )
                return self.get_component_by_id(component_id)
                
            elif message_type == "empire.getRelatedComponents":
                component_id = content.get("component_id")
                if not component_id:
                    return self._create_error_response(
                        code=-32602,
                        message="Invalid params: missing component_id"
                    )
                relationship_types = content.get("relationship_types", [])
                return self.get_related_components(component_id, relationship_types)
                
            elif message_type == "empire.getComponentParts":
                component_id = content.get("component_id")
                if not component_id:
                    return self._create_error_response(
                        code=-32602,
                        message="Invalid params: missing component_id"
                    )
                parts = content.get("parts", [])
                return self.get_component_parts(component_id, parts)
                
            else:
                return self._create_error_response(
                    code=-32601,
                    message=f"Method not found: {message_type}",
                    data={"available_methods": [
                        "empire.getComponents",
                        "empire.getComponentById",
                        "empire.getRelatedComponents",
                        "empire.getComponentParts"
                    ]}
                )
                
        except Exception as e:
            logger.error(f"Error dispatching A2A request: {str(e)}")
            return self._create_error_response(
                code=-32603,
                message="Internal error",
                data={"error": str(e)}
            )
