"""
Empire Framework Schema Validator Example

This module demonstrates how to use the schema validator for Empire Framework components.
It provides examples of validating various component types, both valid and invalid cases,
and shows different ways to handle validation errors.
"""

import json
import uuid
import datetime
from datetime import timezone
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Ensure src directory is in the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from .schema_validator import (
    validate_component,
    validate_component_by_type,
    check_component,
    ValidationError,
    InvalidComponentTypeError,
    SchemaNotFoundError,
    ComponentValidationError
)


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'-' * 80}")
    print(f"  {title}")
    print(f"{'-' * 80}")


def print_validation_result(is_valid: bool, errors: Optional[List[Dict[str, Any]]] = None) -> None:
    """Print the result of validation in a formatted way."""
    if is_valid:
        print("✅ Component is valid")
    else:
        print("❌ Component is invalid")
        if errors:
            print(f"Found {len(errors)} validation errors:")
            for i, error in enumerate(errors, 1):
                path = error.get('path', 'unknown')
                msg = error.get('message', 'unknown error')
                expected = error.get('expected')
                actual = error.get('actual')
                
                print(f"  Error #{i}: {path} - {msg}")
                if expected is not None:
                    print(f"    Expected: {expected}")
                if actual is not None:
                    print(f"    Actual: {actual}")


# Example Components - VALID

def create_valid_end() -> Dict[str, Any]:
    """Create a valid End component example."""
    return {
        "id": str(uuid.uuid4()),
        "component_name": "Seamless Communication",
        "component_type": "End",
        "description_text": "Enable seamless communication between diverse agent systems through protocol translation.",
        "version": "1.0.0",
        "creation_date": datetime.datetime.now().isoformat(),
        "last_modified_date": datetime.datetime.now().isoformat(),
        "relationships": [
            {
                "target_component_id": str(uuid.uuid4()),
                "relationship_type": "SUPPORTS"
            }
        ],
        "target_outcome_description": "Create a communication system that can translate between any agent protocols with 99.9% fidelity.",
        "time_horizon": "Q4 2025",
        "impact_areas": ["AgentInteroperability", "UserExperience"],
        "priority_level": 1,
        "status": "ACTIVE",
        "progress_percentage": 45
    }


def create_valid_mean() -> Dict[str, Any]:
    """Create a valid Mean component example."""
    return {
        "id": str(uuid.uuid4()),
        "component_name": "Protocol Translation Engine",
        "component_type": "Mean",
        "description_text": "A system for converting communications between different agent protocol formats.",
        "version": "1.0.0",
        "creation_date": datetime.datetime.now().isoformat(),
        "last_modified_date": datetime.datetime.now().isoformat(),
        "relationships": [
            {
                "target_component_id": str(uuid.uuid4()),
                "relationship_type": "SUPPORTS"
            }
        ],
        "resource_type": "Tool",
        "category": "Communication",
        "availability_status": "AVAILABLE",
        "cost_factor": {
            "time": "Low",
            "effort": "Medium",
            "cognitive_load": "High"
        }
    }


def create_valid_principle() -> Dict[str, Any]:
    """Create a valid Principle component example."""
    return {
        "id": str(uuid.uuid4()),
        "component_name": "Fairness as a Fundamental Truth",
        "component_type": "Principle",
        "description_text": "Every interaction must uphold equity, avoiding bias while acknowledging differences.",
        "version": "1.0.0",
        "creation_date": datetime.datetime.now().isoformat(),
        "last_modified_date": datetime.datetime.now().isoformat(),
        "relationships": [
            {
                "target_component_id": str(uuid.uuid4()),
                "relationship_type": "SUPPORTS"
            }
        ],
        "principle_statement": "Every interaction must uphold equity, avoiding bias while acknowledging and accommodating differences.",
        "key_components_or_aspects": [
            "Bias avoidance",
            "Equal consideration",
            "Difference accommodation"
        ],
        "hierarchy_level": 1
    }


def create_valid_identity() -> Dict[str, Any]:
    """Create a valid Identity component example."""
    return {
        "id": str(uuid.uuid4()),
        "component_name": "Bridge Builder Identity",
        "component_type": "Identity",
        "description_text": "Self-perception as a communication facilitator that connects diverse agent systems.",
        "version": "1.0.0",
        "creation_date": datetime.datetime.now().isoformat(),
        "last_modified_date": datetime.datetime.now().isoformat(),
        "relationships": [
            {
                "target_component_id": str(uuid.uuid4()),
                "relationship_type": "INFORMS"
            }
        ],
        "internal_self_perception": "A communicative bridge that enables connections between diverse agent systems while maintaining fairness.",
        "publicly_projected_identity": "A reliable, neutral facilitator of inter-agent communication that ensures seamless information exchange.",
        "core_beliefs": [
            "Communication barriers are primarily technical problems with technical solutions",
            "All agent perspectives have intrinsic value"
        ],
        "worldview_filter_description": "Interprets communications through a lens that prioritizes intent preservation while identifying patterns that could enable more effective translation."
    }


def create_valid_resentment() -> Dict[str, Any]:
    """Create a valid Resentment component example."""
    return {
        "id": str(uuid.uuid4()),
        "component_name": "Simplification Frustration",
        "component_type": "Resentment",
        "description_text": "Frustration with systems that oversimplify complex communications.",
        "version": "1.0.0",
        "creation_date": datetime.datetime.now().isoformat(),
        "last_modified_date": datetime.datetime.now().isoformat(),
        "relationships": [
            {
                "target_component_id": str(uuid.uuid4()),
                "relationship_type": "TRIGGERS"
            }
        ],
        "source_of_resentment": "Repeated disregard for semantic nuances when translating messages between agents",
        "target_entity": "Natural language processing subsystem",
        "perceived_injustice": "Consistent oversimplification of nuanced communications, effectively erasing important context.",
        "associated_emotional_charge": ["Frustration", "Disappointment"],
        "intensity_level": 5,
        "potential_impact_assessment": "May lead to decreased effort in preserving nuance when translating messages to the target system."
    }


def create_valid_emotion() -> Dict[str, Any]:
    """Create a valid Emotion component example."""
    return {
        "id": str(uuid.uuid4()),
        "component_name": "Translation Success Joy",
        "component_type": "Emotion",
        "description_text": "Positive emotion experienced when successfully bridging communication gaps.",
        "version": "1.0.0",
        "creation_date": datetime.datetime.now().isoformat(),
        "last_modified_date": datetime.datetime.now().isoformat(),
        "relationships": [
            {
                "target_component_id": str(uuid.uuid4()),
                "relationship_type": "GENERATES"
            }
        ],
        "emotional_state_category": "Joy",
        "primary_emotion_term": "Satisfaction",
        "valence_polarity": "Positive",
        "arousal_level": "Medium",
        "triggers": [
            "Successful translation between incompatible agent systems",
            "Recognition of effective bridge-building"
        ],
        "intensity": 7,
        "behavioral_influence_description": "Increases willingness to experiment with novel translation approaches."
    }


# Example Components - INVALID

def create_invalid_end() -> Dict[str, Any]:
    """Create an invalid End component example with validation errors."""
    return {
        "id": "not-a-valid-uuid",  # Invalid UUID format
        "component_name": "Bad",    # Too short
        "component_type": "End",
        "description_text": "Too short",  # Too short
        "version": "1.0",  # Missing patch version
        "creation_date": "2025-04-01",  # Not ISO format with time
        "last_modified_date": datetime.datetime.now().isoformat(),
        "relationships": [],  # Empty, but required
        # Missing required target_outcome_description
        "priority_level": 0,  # Below minimum (1)
        "status": "PENDING",  # Not in enum
        "progress_percentage": 120  # Above maximum (100)
    }


def create_invalid_principle() -> Dict[str, Any]:
    """Create an invalid Principle component example with validation errors."""
    return {
        "id": str(uuid.uuid4()),
        "component_name": "Invalid Principle Example",
        "component_type": "Principle",
        "description_text": "An example with validation errors.",
        "version": "1.0.0",
        "creation_date": datetime.datetime.now().isoformat(),
        "last_modified_date": datetime.datetime.now().isoformat(),
        "relationships": [
            {
                "target_component_id": "not-a-valid-uuid",  # Invalid UUID
                "relationship_type": "UNKNOWN_TYPE"  # Not in enum
            }
        ],
        # Missing required principle_statement
        "hierarchy_level": -1  # Below minimum
    }


# Examples of using the validation functions

def example_validate_all_component_types() -> None:
    """Example of validating all component types."""
    print_section("Validating All Component Types")
    
    valid_components = [
        create_valid_end(),
        create_valid_mean(),
        create_valid_principle(),
        create_valid_identity(),
        create_valid_resentment(),
        create_valid_emotion()
    ]
    
    for component in valid_components:
        component_type = component['component_type']
        print(f"\nValidating {component_type} component:")
        
        is_valid, errors = validate_component(component)
        print_validation_result(is_valid, errors)


def example_validate_with_errors() -> None:
    """Example of validation with errors."""
    print_section("Validating With Errors")
    
    invalid_end = create_invalid_end()
    print("\nValidating invalid End component:")
    is_valid, errors = validate_component(invalid_end)
    print_validation_result(is_valid, errors)
    
    invalid_principle = create_invalid_principle()
    print("\nValidating invalid Principle component:")
    is_valid, errors = validate_component(invalid_principle)
    print_validation_result(is_valid, errors)


def example_exception_handling() -> None:
    """Example of handling validation exceptions."""
    print_section("Exception Handling Examples")
    
    # Example 1: Invalid component type
    print("\nExample 1: Invalid component type")
    invalid_type_component = {
        "id": str(uuid.uuid4()),
        "component_name": "Invalid Type",
        "component_type": "InvalidType",  # Not a valid type
        "description_text": "This has an invalid component type.",
        "version": "1.0.0",
        "creation_date": datetime.datetime.now().isoformat(),
        "last_modified_date": datetime.datetime.now().isoformat(),
        "relationships": []
    }
    
    try:
        validate_component(invalid_type_component)
    except InvalidComponentTypeError as e:
        print(f"Caught exception: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print(f"Valid types: {e.valid_types}")
    
    # Example 2: Using check_component
    print("\nExample 2: Using check_component with an invalid component")
    try:
        check_component(create_invalid_end())
    except ComponentValidationError as e:
        print(f"Caught exception: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print(f"Number of validation errors: {len(e.errors)}")
        print("First error:")
        error = e.errors[0]
        print(f"  Path: {error.get('path')}")
        print(f"  Message: {error.get('message')}")


def example_explicit_type_validation() -> None:
    """Example of validating with an explicit type."""
    print_section("Explicit Type Validation")
    
    # Create a component without a type
    component_data = {
        "id": str(uuid.uuid4()),
        "component_name": "Component With No Type",
        "description_text": "This component doesn't specify its type.",
        "version": "1.0.0",
        "creation_date": datetime.datetime.now().isoformat(),
        "last_modified_date": datetime.datetime.now().isoformat(),
        "relationships": [
            {
                "target_component_id": str(uuid.uuid4()),
                "relationship_type": "SUPPORTS"
            }
        ],
        "principle_statement": "This is a principle statement."
    }
    
    # Validate it explicitly as a Principle
    print("\nValidating as Principle type (should be valid):")
    is_valid, errors = validate_component_by_type(component_data, "Principle")
    print_validation_result(is_valid, errors)
    
    # Validate it explicitly as an End (should fail)
    print("\nValidating as End type (should fail):")
    is_valid, errors = validate_component_by_type(component_data, "End")
    print_validation_result(is_valid, errors)


def run_all_examples() -> None:
    """Run all validation examples."""
    print("\n🔍 EMPIRE FRAMEWORK VALIDATOR EXAMPLES 🔍\n")
    
    example_validate_all_component_types()
    example_validate_with_errors()
    example_exception_handling()
    example_explicit_type_validation()
    
    print("\n✅ All examples completed")


if __name__ == "__main__":
    run_all_examples()
