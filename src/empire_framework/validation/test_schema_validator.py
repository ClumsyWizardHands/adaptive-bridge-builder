"""
Unit tests for the Empire Framework Schema Validator.

This module contains tests for the schema validation functionality,
covering validation of all component types, error handling, and caching behavior.
"""

import unittest
import json
import uuid
import datetime
from datetime import timezone
import tempfile
import os
from pathlib import Path
from typing import Dict, Any

# Ensure src directory is in the path so we can import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from .schema_validator import (
    validate_component,
    validate_component_by_type,
    check_component,
    load_schema,
    component_type_to_schema_file,
    ValidationError,
    InvalidComponentTypeError,
    SchemaNotFoundError,
    ComponentValidationError,
    SCHEMA_DIR,
    _schema_cache
)

# Import helper functions from validator_example for creating test components
from .validator_example import (
    create_valid_end,
    create_valid_mean,
    create_valid_principle,
    create_valid_identity,
    create_valid_resentment,
    create_valid_emotion,
    create_invalid_end,
    create_invalid_principle
)


class SchemaValidatorTests(unittest.TestCase):
    """Tests for schema validation functionality."""

    def setUp(self) -> None:
        """Set up the test environment before each test method runs."""
        # Clear the schema cache before each test
        _schema_cache.clear()

    def test_valid_components(self) -> None:
        """Test validation of valid components of each type."""
        # Test each component type
        components = [
            create_valid_end(),
            create_valid_mean(),
            create_valid_principle(),
            create_valid_identity(),
            create_valid_resentment(),
            create_valid_emotion()
        ]
        
        for component in components:
            component_type = component['component_type']
            with self.subTest(component_type=component_type):
                is_valid, errors = validate_component(component)
                self.assertTrue(is_valid, f"Valid {component_type} component failed validation")
                self.assertEqual(0, len(errors), f"Valid {component_type} component had validation errors")

    def test_invalid_components(self) -> None:
        """Test validation of invalid components."""
        # Test invalid End
        is_valid, errors = validate_component(create_invalid_end())
        self.assertFalse(is_valid, "Invalid End component passed validation")
        self.assertGreater(len(errors), 0, "Invalid End component had no validation errors")
        
        # Test invalid Principle
        is_valid, errors = validate_component(create_invalid_principle())
        self.assertFalse(is_valid, "Invalid Principle component passed validation")
        self.assertGreater(len(errors), 0, "Invalid Principle component had no validation errors")

    def test_missing_component_type(self) -> None:
        """Test validation when component_type is missing."""
        component = {
            "id": str(uuid.uuid4()),
            "component_name": "Missing Type Component",
            "description_text": "This component is missing its component_type field.",
            "version": "1.0.0",
            "creation_date": datetime.datetime.now().isoformat(),
            "last_modified_date": datetime.datetime.now().isoformat(),
            "relationships": []
        }
        
        is_valid, errors = validate_component(component)
        
        self.assertFalse(is_valid, "Component with missing type passed validation")
        self.assertEqual(1, len(errors), "Expected exactly one validation error")
        self.assertEqual("root", errors[0]['path'], "Error path should be 'root'")
        self.assertIn("component_type", errors[0]['message'], "Error message should mention component_type")

    def test_invalid_component_type(self) -> None:
        """Test validation when component_type is invalid."""
        component = {
            "id": str(uuid.uuid4()),
            "component_name": "Invalid Type Component",
            "component_type": "InvalidType",
            "description_text": "This component has an invalid component_type.",
            "version": "1.0.0",
            "creation_date": datetime.datetime.now().isoformat(),
            "last_modified_date": datetime.datetime.now().isoformat(),
            "relationships": []
        }
        
        with self.assertRaises(InvalidComponentTypeError) as context:
            validate_component(component)
        
        self.assertIn("InvalidType", str(context.exception), "Exception message should include the invalid type")
        self.assertIn("End", context.exception.valid_types, "Valid types should include 'End'")

    def test_validate_component_by_type(self) -> None:
        """Test validating a component with an explicit type."""
        # Create a component without a type field
        component = {
            "id": str(uuid.uuid4()),
            "component_name": "Explicit Type Component",
            "description_text": "This component is validated with an explicit type.",
            "version": "1.0.0",
            "creation_date": datetime.datetime.now().isoformat(),
            "last_modified_date": datetime.datetime.now().isoformat(),
            "relationships": [
                {
                    "target_component_id": str(uuid.uuid4()),
                    "relationship_type": "SUPPORTS"
                }
            ],
            # Add required field for Principle
            "principle_statement": "This is a principle statement."
        }
        
        # Should be valid as a Principle
        is_valid, errors = validate_component_by_type(component, "Principle")
        self.assertTrue(is_valid, "Component should be valid as a Principle")
        
        # Should be invalid as an End (missing required fields)
        is_valid, errors = validate_component_by_type(component, "End")
        self.assertFalse(is_valid, "Component should be invalid as an End")
        found_error = False
        for error in errors:
            if error.get('path') == 'root' and 'target_outcome_description' in str(error.get('message', '')):
                found_error = True
                break
        self.assertTrue(found_error, "Should have error about missing target_outcome_description")

    def test_check_component(self) -> None:
        """Test the check_component function that raises exceptions."""
        # Valid component should not raise an exception
        try:
            check_component(create_valid_end())
        except Exception as e:
            self.fail(f"check_component raised {type(e).__name__} unexpectedly!")
        
        # Invalid component should raise ComponentValidationError
        with self.assertRaises(ComponentValidationError) as context:
            check_component(create_invalid_end())
        
        # Check that the exception contains useful information
        exception = context.exception
        self.assertEqual("End", exception.component_type, "Exception should capture component type")
        self.assertGreater(len(exception.errors), 0, "Exception should contain validation errors")

    def test_schema_loading_and_caching(self) -> None:
        """Test schema loading and caching behavior."""
        # First, clear the cache
        _schema_cache.clear()
        
        # First load should not be cached
        schema1 = load_schema("ends_schema.json")
        self.assertIsInstance(schema1, dict, "Schema should be loaded as a dictionary")
        
        # Second load should retrieve from cache
        schema2 = load_schema("ends_schema.json")
        self.assertIs(schema1, schema2, "Second load should return cached schema")
        
        # Test handling of non-existent schema
        with self.assertRaises(SchemaNotFoundError):
            load_schema("non_existent_schema.json")

    def test_component_type_to_schema_file(self) -> None:
        """Test the component_type_to_schema_file function."""
        # Test valid component types
        self.assertEqual("ends_schema.json", component_type_to_schema_file("End"))
        self.assertEqual("means_schema.json", component_type_to_schema_file("Mean"))
        self.assertEqual("principles_schema.json", component_type_to_schema_file("Principle"))
        
        # Test invalid component type
        with self.assertRaises(InvalidComponentTypeError):
            component_type_to_schema_file("InvalidType")


if __name__ == '__main__':
    unittest.main()
