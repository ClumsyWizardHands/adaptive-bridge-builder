"""
Integration Tests for Empire Framework A2A Integration

This module contains tests for validating the integration between the Empire Framework
and the A2A Protocol, ensuring components can be properly exposed and exchanged.
"""

import unittest
import json
import copy
from typing import Dict, List, Any

# Import Empire Framework components
from empire_framework.a2a.a2a_adapter import A2AAdapter
from api.a2a.handlers import EmpireA2AHandlers
from empire_framework.registry.component_registry import ComponentRegistry

class TestA2AIntegration(unittest.TestCase):
    """Test suite for Empire Framework A2A integration."""
    
    def setUp(self) -> None:
        """Set up test environment."""
        # Create a new adapter and handlers
        self.adapter = A2AAdapter(validation_enabled=True)
        
        # Create a registry with some test components
        self.registry = ComponentRegistry()
        
        # Add test components to registry
        self._add_test_components_to_registry()
        
        # Create handlers with our test registry
        self.handlers = EmpireA2AHandlers(registry=self.registry)
    
    def _add_test_components_to_registry(self) -> None:
        """Add test components to the registry for testing."""
        # Principle component
        principle = {
            "id": "principle-001",
            "type": "principle",
            "version": "1.0",
            "name": "Fairness as Truth",
            "description": "Equal treatment of all agents regardless of source",
            "importance": "high",
            "example": "When agents send requests, each is evaluated with the same criteria",
            "evaluation_criteria": [
                "Message is processed using standardized validation",
                "No prioritization based on sender identity"
            ],
            "tags": ["fairness", "core_principle"]
        }
        
        # Means component
        means = {
            "id": "means-001",
            "type": "means",
            "version": "1.0",
            "name": "Adaptive Communication",
            "capabilities": [
                "Protocol translation",
                "Message routing",
                "Format adaptation"
            ],
            "limitations": [
                "Requires clear message structure",
                "Limited to supported protocols"
            ],
            "tags": ["communication", "adaptation"]
        }
        
        # Ends component
        ends = {
            "id": "ends-001",
            "type": "ends",
            "version": "1.0",
            "goal": "Seamless agent interoperability",
            "success_criteria": [
                "Messages successfully translated between all supported protocols",
                "No information loss during translation"
            ],
            "priority": "high",
            "tags": ["interoperability", "goal"]
        }
        
        # Add components to registry
        self.registry.register_component(principle)
        self.registry.register_component(means)
        self.registry.register_component(ends)
        
        # Add relationships
        self.registry.add_relationship("means-001", "principle-001", "implements", 0.8)
        self.registry.add_relationship("ends-001", "means-001", "requires", 0.9)
    
    def test_component_to_a2a_message_conversion(self) -> None:
        """Test conversion of Empire component to A2A message."""
        # Get a test component
        component = self.registry.get_component("principle-001")
        
        # Convert to A2A message
        a2a_message = self.adapter.component_to_a2a_message(component)
        
        # Verify conversion
        self.assertEqual(a2a_message["a2a_version"], "1.0")
        self.assertEqual(a2a_message["message_type"], "empire.component")
        self.assertEqual(a2a_message["content"]["component_id"], "principle-001")
        self.assertEqual(a2a_message["content"]["component_type"], "principle")
        self.assertEqual(a2a_message["content"]["component_data"]["name"], "Fairness as Truth")
        self.assertEqual(a2a_message["metadata"]["source"], "empire_framework")
    
    def test_a2a_message_to_component_conversion(self) -> None:
        """Test conversion of A2A message back to Empire component."""
        # Create a test A2A message
        a2a_message = {
            "a2a_version": "1.0",
            "message_type": "empire.component",
            "content": {
                "component_id": "test-component-001",
                "component_type": "principle",
                "component_data": {
                    "name": "Test Principle",
                    "description": "A test principle for conversion testing",
                    "importance": "medium"
                }
            },
            "metadata": {
                "source": "test_suite",
                "component_version": "1.0"
            }
        }
        
        # Convert to component
        component = self.adapter.a2a_message_to_component(a2a_message)
        
        # Verify conversion
        self.assertEqual(component["id"], "test-component-001")
        self.assertEqual(component["type"], "principle")
        self.assertEqual(component["name"], "Test Principle")
        self.assertEqual(component["description"], "A test principle for conversion testing")
        self.assertEqual(component["version"], "1.0")
    
    def test_component_parts_conversion(self) -> None:
        """Test conversion of component to A2A parts and back."""
        # Get a test component
        component = self.registry.get_component("means-001")
        
        # Convert to A2A parts
        a2a_parts = self.adapter.component_to_a2a_parts(component)
        
        # Verify parts
        self.assertEqual(a2a_parts["content"]["component_id"], "means-001")
        self.assertTrue("identity" in a2a_parts["content"]["parts"])
        self.assertTrue("means" in a2a_parts["content"]["parts"])
        self.assertEqual(a2a_parts["content"]["parts"]["means"]["capabilities"][0], "Protocol translation")
        
        # Convert back to component
        reconstructed = self.adapter.a2a_parts_to_component(a2a_parts)
        
        # Verify reconstruction
        self.assertEqual(reconstructed["id"], component["id"])
        self.assertEqual(reconstructed["name"], component["name"])
        self.assertEqual(reconstructed["capabilities"], component["capabilities"])
    
    def test_get_components_handler(self) -> None:
        """Test the getComponents handler."""
        # Create a test request
        request = {
            "a2a_version": "1.0",
            "message_type": "empire.getComponents",
            "content": {
                "filters": {
                    "type": "principle"
                }
            }
        }
        
        # Get response from handler
        response = self.handlers.dispatch(request)
        
        # Verify response
        self.assertEqual(response["message_type"], "empire.component_batch")
        self.assertEqual(response["content"]["component_count"], 1)
        self.assertEqual(response["content"]["components"][0]["component_id"], "principle-001")
    
    def test_get_component_by_id_handler(self) -> None:
        """Test the getComponentById handler."""
        # Create a test request
        request = {
            "a2a_version": "1.0",
            "message_type": "empire.getComponentById",
            "content": {
                "component_id": "ends-001"
            }
        }
        
        # Get response from handler
        response = self.handlers.dispatch(request)
        
        # Verify response
        self.assertEqual(response["message_type"], "empire.component")
        self.assertEqual(response["content"]["component_id"], "ends-001")
        self.assertEqual(response["content"]["component_data"]["goal"], "Seamless agent interoperability")
    
    def test_get_related_components_handler(self) -> None:
        """Test the getRelatedComponents handler."""
        # Create a test request
        request = {
            "a2a_version": "1.0",
            "message_type": "empire.getRelatedComponents",
            "content": {
                "component_id": "means-001",
                "relationship_types": ["implements"]
            }
        }
        
        # Get response from handler
        response = self.handlers.dispatch(request)
        
        # Verify response
        self.assertEqual(response["message_type"], "empire.component_batch")
        self.assertTrue(response["content"]["component_count"] > 0)
        # Check that one of the components is the principle that means-001 implements
        found_principle = False
        for component in response["content"]["components"]:
            if component["component_id"] == "principle-001":
                found_principle = True
                break
        self.assertTrue(found_principle, "Related principle component not found")
    
    def test_get_component_parts_handler(self) -> None:
        """Test the getComponentParts handler."""
        # Create a test request
        request = {
            "a2a_version": "1.0",
            "message_type": "empire.getComponentParts",
            "content": {
                "component_id": "principle-001",
                "parts": ["identity", "principle"]
            }
        }
        
        # Get response from handler
        response = self.handlers.dispatch(request)
        
        # Verify response
        self.assertEqual(response["message_type"], "empire.component_parts")
        self.assertEqual(response["content"]["component_id"], "principle-001")
        self.assertTrue("identity" in response["content"]["parts"])
        self.assertTrue("principle" in response["content"]["parts"])
        self.assertEqual(response["content"]["parts"]["principle"]["name"], "Fairness as Truth")
    
    def test_error_handling(self) -> None:
        """Test error handling in the A2A handlers."""
        # Test missing component
        request = {
            "a2a_version": "1.0",
            "message_type": "empire.getComponentById",
            "content": {
                "component_id": "nonexistent-id"
            }
        }
        
        response = self.handlers.dispatch(request)
        
        # Verify error response
        self.assertEqual(response["message_type"], "empire.error")
        self.assertEqual(response["content"]["error"]["code"], -32001)
        self.assertEqual(response["content"]["error"]["data"]["component_id"], "nonexistent-id")
        
        # Test invalid request
        request = {
            "a2a_version": "1.0",
            "message_type": "empire.invalidMethod",
            "content": {}
        }
        
        response = self.handlers.dispatch(request)
        
        # Verify error response
        self.assertEqual(response["message_type"], "empire.error")
        self.assertEqual(response["content"]["error"]["code"], -32601)
        self.assertTrue("available_methods" in response["content"]["error"]["data"])
    
    def test_components_batch_conversion(self) -> None:
        """Test batch conversion of components to A2A message and back."""
        # Get multiple components
        components = [
            self.registry.get_component("principle-001"),
            self.registry.get_component("means-001"),
            self.registry.get_component("ends-001")
        ]
        
        # Convert to A2A batch
        batch = self.adapter.components_to_a2a_batch(components)
        
        # Verify batch
        self.assertEqual(batch["message_type"], "empire.component_batch")
        self.assertEqual(batch["content"]["component_count"], 3)
        
        # Convert back to components
        reconstructed = self.adapter.a2a_batch_to_components(batch)
        
        # Verify reconstruction
        self.assertEqual(len(reconstructed), 3)
        self.assertEqual(reconstructed[0]["id"], components[0]["id"])
        self.assertEqual(reconstructed[1]["id"], components[1]["id"])
        self.assertEqual(reconstructed[2]["id"], components[2]["id"])

if __name__ == "__main__":
    unittest.main()
