"""
A2A Adapter for Empire Framework

This module provides the functionality to convert Empire Framework components 
to A2A Messages/Parts format and vice versa, enabling integration between 
the Empire Framework and the A2A Protocol for agent-to-agent communications.
"""

import json
import logging
import copy
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("A2AAdapter")

class A2AAdapter:
    """
    Adapter class that handles conversion between Empire Components and A2A Messages.
    
    This adapter enables Empire Framework components to be exposed through the A2A Protocol,
    allowing components to be exchanged between agents that understand the A2A specification.
    """
    
    def __init__(self, validation_enabled: bool = True):
        """
        Initialize the A2A adapter.
        
        Args:
            validation_enabled: Whether to validate components during conversion
        """
        self.validation_enabled = validation_enabled
        logger.info("A2A Adapter initialized with validation %s", 
                   "enabled" if validation_enabled else "disabled")
    
    def component_to_a2a_message(
        self, 
        component: Dict[str, Any], 
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Convert an Empire component to an A2A message.
        
        Args:
            component: The Empire component to convert
            include_metadata: Whether to include component metadata in the message
            
        Returns:
            A2A message representation of the component
            
        Raises:
            ValueError: If component validation fails and validation is enabled
        """
        # Validate component if enabled
        if self.validation_enabled:
            self._validate_component(component)
        
        # Create A2A message structure
        a2a_message = {
            "a2a_version": "1.0",
            "message_type": "empire.component", 
            "content": {
                "component_id": component.get("id", "unknown"),
                "component_type": component.get("type", "unknown"),
                "component_data": self._process_component_data(component)
            }
        }
        
        # Add metadata if needed
        if include_metadata:
            a2a_message["metadata"] = {
                "source": "empire_framework",
                "component_version": component.get("version", "1.0"),
                "schema": component.get("schema", None),
                "timestamp": component.get("last_updated", None)
            }
        
        return a2a_message
    
    def a2a_message_to_component(
        self, 
        a2a_message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert an A2A message to an Empire component.
        
        Args:
            a2a_message: The A2A message to convert
            
        Returns:
            Empire component representation of the A2A message
            
        Raises:
            ValueError: If A2A message validation fails
            ValueError: If resulting component validation fails and validation is enabled
        """
        # Validate A2A message
        self._validate_a2a_message(a2a_message)
        
        # Extract content and metadata
        content = a2a_message.get("content", {})
        metadata = a2a_message.get("metadata", {})
        
        # Create Empire component
        component = {
            "id": content.get("component_id", "unknown"),
            "type": content.get("component_type", "unknown")
        }
        
        # Add component data
        component_data = content.get("component_data", {})
        for key, value in component_data.items():
            component[key] = value
        
        # Add metadata if available
        if "component_version" in metadata:
            component["version"] = metadata["component_version"]
        
        if "schema" in metadata:
            component["schema"] = metadata["schema"]
        
        if "timestamp" in metadata:
            component["last_updated"] = metadata["timestamp"]
        
        # Validate the resulting component if enabled
        if self.validation_enabled:
            self._validate_component(component)
        
        return component
    
    def components_to_a2a_batch(
        self, 
        components: List[Dict[str, Any]], 
        batch_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Convert multiple Empire components to an A2A batch message.
        
        Args:
            components: List of Empire components to convert
            batch_metadata: Optional metadata for the batch
            
        Returns:
            A2A batch message containing the components
        """
        # Convert each component to A2A message
        component_messages = []
        for component in components:
            try:
                # Convert without separate metadata as we'll include it in the parts
                a2a_message = self.component_to_a2a_message(component, include_metadata=False)
                component_messages.append(a2a_message["content"])
            except Exception as e:
                logger.warning(f"Failed to convert component {component.get('id', 'unknown')}: {str(e)}")
        
        # Create batch message
        batch_message = {
            "a2a_version": "1.0",
            "message_type": "empire.component_batch",
            "content": {
                "component_count": len(component_messages),
                "components": component_messages
            }
        }
        
        # Add metadata
        if batch_metadata:
            batch_message["metadata"] = batch_metadata
        else:
            batch_message["metadata"] = {
                "source": "empire_framework",
                "batch_timestamp": None  # Caller could add a timestamp here
            }
        
        return batch_message
    
    def a2a_batch_to_components(
        self, 
        a2a_batch: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Convert an A2A batch message to multiple Empire components.
        
        Args:
            a2a_batch: A2A batch message to convert
            
        Returns:
            List of Empire components
        """
        # Validate batch message
        self._validate_a2a_batch(a2a_batch)
        
        # Extract components from batch
        components = []
        batch_content = a2a_batch.get("content", {})
        component_items = batch_content.get("components", [])
        
        for component_data in component_items:
            try:
                # Create synthetic A2A message to use existing conversion
                a2a_message = {
                    "a2a_version": "1.0",
                    "message_type": "empire.component",
                    "content": component_data,
                    "metadata": a2a_batch.get("metadata", {})
                }
                
                component = self.a2a_message_to_component(a2a_message)
                components.append(component)
            except Exception as e:
                logger.warning(f"Failed to convert component: {str(e)}")
        
        return components
    
    def component_to_a2a_parts(
        self, 
        component: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert an Empire component to A2A parts format for more granular access.
        
        This allows accessing individual aspects of a component through the A2A protocol.
        
        Args:
            component: The Empire component to convert
            
        Returns:
            A2A message with parts representation of the component
        """
        # Validate component if enabled
        if self.validation_enabled:
            self._validate_component(component)
        
        # Extract parts based on component type
        parts = {}
        
        # Common parts for all component types
        parts["identity"] = self._extract_identity_part(component)
        
        # Type-specific parts
        component_type = component.get("type", "unknown")
        if component_type == "principle":
            parts["principle"] = self._extract_principle_part(component)
        elif component_type == "means":
            parts["means"] = self._extract_means_part(component)
        elif component_type == "ends":
            parts["ends"] = self._extract_ends_part(component)
        elif component_type == "resentment":
            parts["resentment"] = self._extract_resentment_part(component)
        elif component_type == "emotion":
            parts["emotion"] = self._extract_emotion_part(component)
        
        # Create A2A message with parts
        a2a_message = {
            "a2a_version": "1.0",
            "message_type": "empire.component_parts",
            "content": {
                "component_id": component.get("id", "unknown"),
                "component_type": component_type,
                "parts": parts
            },
            "metadata": {
                "source": "empire_framework",
                "component_version": component.get("version", "1.0")
            }
        }
        
        return a2a_message
    
    def a2a_parts_to_component(
        self, 
        a2a_parts_message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert an A2A parts message to an Empire component.
        
        Args:
            a2a_parts_message: A2A parts message to convert
            
        Returns:
            Empire component reconstructed from parts
        """
        # Validate parts message
        self._validate_a2a_parts_message(a2a_parts_message)
        
        # Extract content and parts
        content = a2a_parts_message.get("content", {})
        parts = content.get("parts", {})
        
        # Create base component
        component = {
            "id": content.get("component_id", "unknown"),
            "type": content.get("component_type", "unknown"),
            "version": a2a_parts_message.get("metadata", {}).get("component_version", "1.0")
        }
        
        # Merge identity part
        if "identity" in parts:
            self._merge_identity_part(component, parts["identity"])
        
        # Merge type-specific parts
        component_type = component["type"]
        if component_type == "principle" and "principle" in parts:
            self._merge_principle_part(component, parts["principle"])
        elif component_type == "means" and "means" in parts:
            self._merge_means_part(component, parts["means"])
        elif component_type == "ends" and "ends" in parts:
            self._merge_ends_part(component, parts["ends"])
        elif component_type == "resentment" and "resentment" in parts:
            self._merge_resentment_part(component, parts["resentment"])
        elif component_type == "emotion" and "emotion" in parts:
            self._merge_emotion_part(component, parts["emotion"])
        
        # Validate the resulting component if enabled
        if self.validation_enabled:
            self._validate_component(component)
        
        return component
    
    def _process_component_data(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Process and extract the core data from a component."""
        # Create a copy to avoid modifying the original
        data = copy.deepcopy(component)
        
        # Remove metadata fields
        for field in ["id", "type", "version", "schema", "last_updated"]:
            if field in data:
                del data[field]
                
        return data
    
    def _validate_component(self, component: Dict[str, Any]) -> None:
        """
        Validate an Empire component.
        
        Args:
            component: The component to validate
            
        Raises:
            ValueError: If component validation fails
        """
        # Check required fields
        required_fields = ["id", "type"]
        for field in required_fields:
            if field not in component:
                raise ValueError(f"Missing required field '{field}' in component")
        
        # Type-specific validation
        component_type = component.get("type")
        if component_type == "principle":
            self._validate_principle_component(component)
        elif component_type == "means":
            self._validate_means_component(component)
        elif component_type == "ends":
            self._validate_ends_component(component)
        elif component_type == "resentment":
            self._validate_resentment_component(component)
        elif component_type == "emotion":
            self._validate_emotion_component(component)
    
    def _validate_principle_component(self, component: Dict[str, Any]) -> None:
        """Validate a principle component."""
        required_fields = ["name", "description"]
        for field in required_fields:
            if field not in component:
                raise ValueError(f"Missing required field '{field}' in principle component")
    
    def _validate_means_component(self, component: Dict[str, Any]) -> None:
        """Validate a means component."""
        required_fields = ["name", "capabilities"]
        for field in required_fields:
            if field not in component:
                raise ValueError(f"Missing required field '{field}' in means component")
    
    def _validate_ends_component(self, component: Dict[str, Any]) -> None:
        """Validate an ends component."""
        required_fields = ["goal", "success_criteria"]
        for field in required_fields:
            if field not in component:
                raise ValueError(f"Missing required field '{field}' in ends component")
    
    def _validate_resentment_component(self, component: Dict[str, Any]) -> None:
        """Validate a resentment component."""
        required_fields = ["trigger", "response"]
        for field in required_fields:
            if field not in component:
                raise ValueError(f"Missing required field '{field}' in resentment component")
    
    def _validate_emotion_component(self, component: Dict[str, Any]) -> None:
        """Validate an emotion component."""
        required_fields = ["emotion_type", "intensity"]
        for field in required_fields:
            if field not in component:
                raise ValueError(f"Missing required field '{field}' in emotion component")
    
    def _validate_a2a_message(self, message: Dict[str, Any]) -> None:
        """
        Validate an A2A message.
        
        Args:
            message: The A2A message to validate
            
        Raises:
            ValueError: If message validation fails
        """
        # Check required fields
        if "a2a_version" not in message:
            raise ValueError("Missing 'a2a_version' in A2A message")
            
        if "message_type" not in message:
            raise ValueError("Missing 'message_type' in A2A message")
        
        if message.get("message_type") != "empire.component":
            raise ValueError(f"Expected message_type 'empire.component', got '{message.get('message_type')}'")
            
        if "content" not in message:
            raise ValueError("Missing 'content' in A2A message")
            
        # Check content fields
        content = message.get("content", {})
        if "component_id" not in content:
            raise ValueError("Missing 'component_id' in message content")
            
        if "component_type" not in content:
            raise ValueError("Missing 'component_type' in message content")
            
        if "component_data" not in content:
            raise ValueError("Missing 'component_data' in message content")
    
    def _validate_a2a_batch(self, batch: Dict[str, Any]) -> None:
        """
        Validate an A2A batch message.
        
        Args:
            batch: The A2A batch message to validate
            
        Raises:
            ValueError: If batch validation fails
        """
        # Check required fields
        if "a2a_version" not in batch:
            raise ValueError("Missing 'a2a_version' in A2A batch message")
            
        if "message_type" not in batch:
            raise ValueError("Missing 'message_type' in A2A batch message")
        
        if batch.get("message_type") != "empire.component_batch":
            raise ValueError(f"Expected message_type 'empire.component_batch', got '{batch.get('message_type')}'")
            
        if "content" not in batch:
            raise ValueError("Missing 'content' in A2A batch message")
            
        # Check content fields
        content = batch.get("content", {})
        if "component_count" not in content:
            raise ValueError("Missing 'component_count' in batch content")
            
        if "components" not in content:
            raise ValueError("Missing 'components' in batch content")
            
        # Check components array
        components = content.get("components", [])
        if not isinstance(components, list):
            raise ValueError("'components' should be an array")
            
        if len(components) != content.get("component_count", 0):
            raise ValueError("Component count mismatch in batch message")
    
    def _validate_a2a_parts_message(self, message: Dict[str, Any]) -> None:
        """
        Validate an A2A parts message.
        
        Args:
            message: The A2A parts message to validate
            
        Raises:
            ValueError: If message validation fails
        """
        # Check required fields
        if "a2a_version" not in message:
            raise ValueError("Missing 'a2a_version' in A2A parts message")
            
        if "message_type" not in message:
            raise ValueError("Missing 'message_type' in A2A parts message")
        
        if message.get("message_type") != "empire.component_parts":
            raise ValueError(f"Expected message_type 'empire.component_parts', got '{message.get('message_type')}'")
            
        if "content" not in message:
            raise ValueError("Missing 'content' in A2A parts message")
            
        # Check content fields
        content = message.get("content", {})
        if "component_id" not in content:
            raise ValueError("Missing 'component_id' in parts message content")
            
        if "component_type" not in content:
            raise ValueError("Missing 'component_type' in parts message content")
            
        if "parts" not in content:
            raise ValueError("Missing 'parts' in parts message content")
            
        # Ensure parts is a dictionary
        parts = content.get("parts", {})
        if not isinstance(parts, dict):
            raise ValueError("'parts' should be a dictionary")
    
    def _extract_identity_part(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the identity part from a component."""
        identity = {
            "id": component.get("id"),
            "type": component.get("type"),
            "version": component.get("version", "1.0"),
            "name": component.get("name", "Unnamed Component")
        }
        
        # Add description if available
        if "description" in component:
            identity["description"] = component["description"]
            
        return identity
    
    def _extract_principle_part(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the principle part from a principle component."""
        principle = {
            "name": component.get("name"),
            "description": component.get("description"),
            "importance": component.get("importance", "medium")
        }
        
        # Add optional fields
        for field in ["example", "rationale", "evaluation_criteria"]:
            if field in component:
                principle[field] = component[field]
                
        return principle
    
    def _extract_means_part(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the means part from a means component."""
        means = {
            "name": component.get("name"),
            "capabilities": component.get("capabilities", [])
        }
        
        # Add optional fields
        for field in ["limitations", "resources", "efficiency", "adaptability"]:
            if field in component:
                means[field] = component[field]
                
        return means
    
    def _extract_ends_part(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the ends part from an ends component."""
        ends = {
            "goal": component.get("goal"),
            "success_criteria": component.get("success_criteria", [])
        }
        
        # Add optional fields
        for field in ["priority", "timeline", "dependencies", "metrics"]:
            if field in component:
                ends[field] = component[field]
                
        return ends
    
    def _extract_resentment_part(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the resentment part from a resentment component."""
        resentment = {
            "trigger": component.get("trigger"),
            "response": component.get("response")
        }
        
        # Add optional fields
        for field in ["intensity", "justification", "resolution_path"]:
            if field in component:
                resentment[field] = component[field]
                
        return resentment
    
    def _extract_emotion_part(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the emotion part from an emotion component."""
        emotion = {
            "emotion_type": component.get("emotion_type"),
            "intensity": component.get("intensity")
        }
        
        # Add optional fields
        for field in ["trigger", "duration", "associated_principle", "physical_manifestation"]:
            if field in component:
                emotion[field] = component[field]
                
        return emotion
    
    def _merge_identity_part(self, component: Dict[str, Any], identity_part: Dict[str, Any]) -> None:
        """Merge identity part into a component."""
        component["id"] = identity_part.get("id", component.get("id", "unknown"))
        component["type"] = identity_part.get("type", component.get("type", "unknown"))
        component["version"] = identity_part.get("version", component.get("version", "1.0"))
        component["name"] = identity_part.get("name", component.get("name", "Unnamed Component"))
        
        if "description" in identity_part:
            component["description"] = identity_part["description"]
    
    def _merge_principle_part(self, component: Dict[str, Any], principle_part: Dict[str, Any]) -> None:
        """Merge principle part into a component."""
        component["name"] = principle_part.get("name", component.get("name"))
        component["description"] = principle_part.get("description", component.get("description"))
        component["importance"] = principle_part.get("importance", component.get("importance", "medium"))
        
        # Add optional fields
        for field in ["example", "rationale", "evaluation_criteria"]:
            if field in principle_part:
                component[field] = principle_part[field]
    
    def _merge_means_part(self, component: Dict[str, Any], means_part: Dict[str, Any]) -> None:
        """Merge means part into a component."""
        component["name"] = means_part.get("name", component.get("name"))
        component["capabilities"] = means_part.get("capabilities", component.get("capabilities", []))
        
        # Add optional fields
        for field in ["limitations", "resources", "efficiency", "adaptability"]:
            if field in means_part:
                component[field] = means_part[field]
    
    def _merge_ends_part(self, component: Dict[str, Any], ends_part: Dict[str, Any]) -> None:
        """Merge ends part into a component."""
        component["goal"] = ends_part.get("goal", component.get("goal"))
        component["success_criteria"] = ends_part.get("success_criteria", component.get("success_criteria", []))
        
        # Add optional fields
        for field in ["priority", "timeline", "dependencies", "metrics"]:
            if field in ends_part:
                component[field] = ends_part[field]
    
    def _merge_resentment_part(self, component: Dict[str, Any], resentment_part: Dict[str, Any]) -> None:
        """Merge resentment part into a component."""
        component["trigger"] = resentment_part.get("trigger", component.get("trigger"))
        component["response"] = resentment_part.get("response", component.get("response"))
        
        # Add optional fields
        for field in ["intensity", "justification", "resolution_path"]:
            if field in resentment_part:
                component[field] = resentment_part[field]
    
    def _merge_emotion_part(self, component: Dict[str, Any], emotion_part: Dict[str, Any]) -> None:
        """Merge emotion part into a component."""
        component["emotion_type"] = emotion_part.get("emotion_type", component.get("emotion_type"))
        component["intensity"] = emotion_part.get("intensity", component.get("intensity"))
        
        # Add optional fields
        for field in ["trigger", "duration", "associated_principle", "physical_manifestation"]:
            if field in emotion_part:
                component[field] = emotion_part[field]
