"""
Empire Framework Validation Package

This package provides validation utilities for Empire Framework components, ensuring
that all component data structures conform to their respective JSON schemas.
"""

from .schema_validator import (
    validate_component,
    validate_component_by_type,
    component_type_to_schema_file,
    ValidationError,
    InvalidComponentTypeError,
    SchemaNotFoundError,
)

__all__ = [
    'validate_component',
    'validate_component_by_type',
    'component_type_to_schema_file',
    'ValidationError',
    'InvalidComponentTypeError',
    'SchemaNotFoundError',
]
