import jsonschema
"""
Schema Validator for Empire Framework Components

This module provides validation utilities for Empire Framework components,
ensuring they conform to their respective JSON schemas.
"""

import os
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from functools import lru_cache
import jsonschema


# Custom Exception Classes
class ValidationError(Exception):
    """Base class for all validation errors."""
    pass


class InvalidComponentTypeError(ValidationError):
    """Raised when an invalid component type is provided."""
    def __init__(self, component_type: str, valid_types: List[str]) -> None:
        self.component_type = component_type
        self.valid_types = valid_types
        message = f"Invalid component type: '{component_type}'. Valid types are: {', '.join(valid_types)}"
        super().__init__(message)


class SchemaNotFoundError(ValidationError):
    """Raised when a schema file cannot be found."""
    def __init__(self, schema_path: str) -> None:
        self.schema_path = schema_path
        message = f"Schema file not found: '{schema_path}'"
        super().__init__(message)


class ComponentValidationError(ValidationError):
    """Raised when component validation fails."""
    def __init__(self, 
                 component_id: Optional[str], 
                 component_type: str, 
                 errors: List[Dict[str, Any]]):
        self.component_id = component_id
        self.component_type = component_type
        self.errors = errors
        
        id_str = f" (ID: {component_id})" if component_id else ""
        message = f"Validation failed for {component_type} component{id_str}. Found {len(errors)} validation errors."
        super().__init__(message)


# Constants
VALID_COMPONENT_TYPES = ["End", "Mean", "Principle", "Identity", "Resentment", "Emotion"]
SCHEMA_DIR = Path("resources/empire_framework_schemas")


# Schema Mapping Function
def component_type_to_schema_file(component_type: str) -> str:
    """
    Maps a component type to its schema filename.
    
    Args:
        component_type: The type of component (e.g., 'End', 'Mean', 'Principle')
        
    Returns:
        The filename of the corresponding schema
        
    Raises:
        InvalidComponentTypeError: If the component type is not recognized
    """
    type_to_file = {
        "End": "ends_schema.json",
        "Mean": "means_schema.json",
        "Principle": "principles_schema.json",
        "Identity": "identity_schema.json",
        "Resentment": "resentments_schema.json",
        "Emotion": "emotions_schema.json"
    }
    
    if component_type not in type_to_file:
        raise InvalidComponentTypeError(component_type, list(type_to_file.keys()))
    
    return type_to_file[component_type]


# Schema Loading with Caching
_schema_cache: Dict[str, Dict[str, Any]] = {}

@lru_cache(maxsize=32)
def load_schema(schema_path: str) -> Dict[str, Any]:
    """
    Loads a JSON schema from file with caching.
    
    Args:
        schema_path: Path to the schema file
        
    Returns:
        The loaded schema as a dictionary
        
    Raises:
        SchemaNotFoundError: If the schema file cannot be found
        ValueError: If the schema is not valid JSON
    """
    if schema_path in _schema_cache:
        return _schema_cache[schema_path]
    
    full_path = SCHEMA_DIR / schema_path
    
    if not full_path.exists():
        raise SchemaNotFoundError(str(full_path))
    
    try:
        with open(full_path, 'r') as f:
            schema = json.load(f)
            _schema_cache[schema_path] = schema
            return schema
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in schema file {schema_path}: {str(e)}")


def get_schema_for_component_type(component_type: str) -> Dict[str, Any]:
    """
    Gets the appropriate schema for a given component type.
    
    Args:
        component_type: The type of component
        
    Returns:
        The loaded schema for the component type
        
    Raises:
        InvalidComponentTypeError: If the component type is not recognized
        SchemaNotFoundError: If the schema file cannot be found
    """
    schema_file = component_type_to_schema_file(component_type)
    return load_schema(schema_file)


def resolve_schema_references(schema: Dict[str, Any], base_dir: Path = SCHEMA_DIR) -> Dict[str, Any]:
    """
    Resolves $ref references in a schema by loading and embedding the referenced schemas.
    
    Args:
        schema: The schema that may contain references
        base_dir: Base directory for resolving relative references
        
    Returns:
        A schema with all references resolved
    """
    schema_copy = schema.copy()
    resolver = jsonschema.RefResolver(base_uri=f"file://{base_dir.absolute()}/", referrer=schema_copy)
    return schema_copy


# Validation Error Formatting
def format_validation_error(error: jsonschema.exceptions.ValidationError) -> Dict[str, Any]:
    """
    Formats a jsonschema ValidationError into a more user-friendly format.
    
    Args:
        error: The validation error from jsonschema
        
    Returns:
        A dictionary with structured error information
    """
    path = "/".join(str(p) for p in error.path) if error.path else "root"
    
    # Extract expected and actual values if available
    expected = None
    actual = None
    
    if error.validator == 'type':
        expected = error.validator_value
        actual = type(error.instance).__name__ if error.instance is not None else 'None'
    elif error.validator == 'enum':
        expected = error.validator_value
        actual = error.instance
    elif error.validator == 'required':
        expected = error.validator_value
        actual = list(error.instance.keys()) if isinstance(error.instance, dict) else 'missing field'
    elif error.validator == 'pattern':
        expected = error.validator_value
        actual = error.instance
    elif error.validator == 'format':
        expected = error.validator_value
        actual = error.instance
    elif error.validator in ('minimum', 'maximum', 'minLength', 'maxLength', 'minItems', 'maxItems'):
        expected = error.validator_value
        actual = error.instance
    
    return {
        'path': path,
        'message': error.message,
        'error_type': error.validator,
        'expected': expected,
        'actual': actual,
        'schema_path': '/'.join(str(p) for p in error.schema_path) if error.schema_path else None
    }


# Main Validation Functions
def validate_component_by_type(component_data: Dict[str, Any], component_type: str) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Validates a component against its schema based on the provided type.
    
    Args:
        component_data: The component data to validate
        component_type: The type of component to validate against
        
    Returns:
        Tuple of (is_valid, error_list)
        
    Raises:
        InvalidComponentTypeError: If the component type is not recognized
        SchemaNotFoundError: If the schema file cannot be found
    """
    if component_type not in VALID_COMPONENT_TYPES:
        raise InvalidComponentTypeError(component_type, VALID_COMPONENT_TYPES)
    
    schema = get_schema_for_component_type(component_type)
    errors = []
    
    try:
        jsonschema.validate(component_data, schema)
        return True, []
    except jsonschema.exceptions.ValidationError as e:
        # Collect all validation errors
        error_details = []
        
        if hasattr(e, 'context'):
            # Multiple validation errors
            for suberror in sorted(e.context, key=lambda e: str(e.path)):
                error_details.append(format_validation_error(suberror))
        else:
            # Single validation error
            error_details.append(format_validation_error(e))
        
        return False, error_details


def validate_component(component_data: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Validates a component against its schema by inferring the component type.
    
    Args:
        component_data: The component data to validate
        
    Returns:
        Tuple of (is_valid, error_list)
        
    Raises:
        InvalidComponentTypeError: If the component type is not recognized or missing
        SchemaNotFoundError: If the schema file cannot be found
    """
    # Check for component_type field
    if 'component_type' not in component_data:
        return False, [{
            'path': 'root',
            'message': "Missing required field 'component_type'",
            'error_type': 'required',
            'expected': ['component_type'],
            'actual': list(component_data.keys()),
            'schema_path': 'required'
        }]
    
    component_type = component_data['component_type']
    
    # Validate against the appropriate schema
    return validate_component_by_type(component_data, component_type)


def check_component(component_data: Dict[str, Any]) -> None:
    """
    Validates a component and raises an exception if validation fails.
    
    This is a convenience wrapper around validate_component that raises
    a ComponentValidationError when validation fails.
    
    Args:
        component_data: The component data to validate
        
    Raises:
        InvalidComponentTypeError: If the component type is not recognized or missing
        SchemaNotFoundError: If the schema file cannot be found
        ComponentValidationError: If the component fails validation
    """
    is_valid, errors = validate_component(component_data)
    
    if not is_valid:
        component_id = component_data.get('id', None)
        component_type = component_data.get('component_type', 'Unknown')
        raise ComponentValidationError(component_id, component_type, errors)


class SchemaValidator:
    """
    A class wrapper for the schema validation functions.
    
    This class provides an object-oriented interface to the schema validation
    functionality, making it easier to use in contexts that expect a validator
    instance.
    """
    
    def __init__(self) -> None:
        """Initialize the SchemaValidator."""
        pass
    
    def validate_component(
        self, 
        component_data: Dict[str, Any], 
        component_type: Optional[str] = None,
        schema_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate a component against its schema.
        
        Args:
            component_data: The component data to validate
            component_type: Optional explicit component type to validate against
            schema_id: Optional specific schema ID to use
            
        Returns:
            Dictionary with 'valid' boolean and 'errors' list
        """
        # If component_type is provided, use validate_component_by_type
        if component_type:
            is_valid, errors = validate_component_by_type(component_data, component_type)
        else:
            # Otherwise use the general validate_component function
            is_valid, errors = validate_component(component_data)
        
        return {
            "valid": is_valid,
            "errors": errors,
            "warnings": []  # Could be extended to include warnings
        }
    
    def check_component(self, component_data: Dict[str, Any]) -> None:
        """
        Validate a component and raise exception if invalid.
        
        Args:
            component_data: The component data to validate
            
        Raises:
            ComponentValidationError: If validation fails
        """
        check_component(component_data)
    
    def validate_by_type(
        self, 
        component_data: Dict[str, Any], 
        component_type: str
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate a component against a specific type schema.
        
        Args:
            component_data: The component data to validate
            component_type: The component type to validate against
            
        Returns:
            Tuple of (is_valid, error_list)
        """
        return validate_component_by_type(component_data, component_type)
    
    def get_valid_component_types(self) -> List[str]:
        """
        Get list of valid component types.
        
        Returns:
            List of valid component type names
        """
        return VALID_COMPONENT_TYPES.copy()