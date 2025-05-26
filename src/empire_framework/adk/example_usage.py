"""
Empire Framework ADK Integration Example

This example demonstrates how to use the Empire Framework ADK adapter
to integrate Empire components with the Agent Development Kit.
"""

import asyncio
import json
from typing import Dict, List, Any
from datetime import datetime, timezone

# Import the Empire ADK adapter
from .empire_adk_adapter import (
    EmpireADKAdapter,
    EmpireEventType,
    get_empire_adk_tools,
    get_empire_workflow_tool_context
)

# Import Empire Framework components
from empire_framework.registry.component_registry import ComponentRegistry
from empire_framework.validation.schema_validator import SchemaValidator
from empire_framework.a2a.component_task_handler import ComponentTaskHandler

async def event_handler(event_type: str, data: Dict[str, Any]) -> None:
    """Example event handler for Empire events."""
    print(f"EVENT: {event_type}")
    print(f"DATA: {json.dumps(data, indent=2)}")
    print("---")


async def main() -> None:
    """Main example function."""
    print("=== Empire Framework ADK Integration Example ===\n")
    
    # Create the Empire ADK adapter
    adapter = EmpireADKAdapter(
        registry=ComponentRegistry(),
        validator=SchemaValidator(),
        task_handler=ComponentTaskHandler(),
        event_callback=event_handler
    )
    
    # Get ADK tools from the adapter
    tools = get_empire_adk_tools(adapter)
    print(f"Available Empire ADK tools: {', '.join(tools.keys())}\n")
    
    # Get workflow tool context
    workflow_context = get_empire_workflow_tool_context(adapter)
    print(f"Workflow tool context: {json.dumps(workflow_context.keys(), indent=2)}\n")
    
    # Example 1: Create a principle component
    print("--- Example 1: Create a Principle Component ---")
    principle = await adapter.create_component(
        component_data={
            "name": "Adaptability as Strength",
            "description": "Ability to evolve and respond to changing needs",
            "importance": "high",
            "example": "When encountering a new message format, the system analyzes patterns and adjusts processing accordingly",
            "evaluation_criteria": [
                "Speed of adaptation to new patterns",
                "Quality of adapted response",
                "Maintenance of core functionality during adaptation"
            ]
        },
        component_type="principle"
    )
    print(f"Created principle: {principle['id']}")
    
    # Example 2: Create a means component
    print("\n--- Example 2: Create a Means Component ---")
    means = await adapter.create_component(
        component_data={
            "name": "Dynamic Protocol Adaptation",
            "capabilities": [
                "Protocol translation",
                "Format conversion",
                "Message routing"
            ],
            "limitations": [
                "Requires clear message structure",
                "Limited to supported protocols"
            ]
        },
        component_type="means"
    )
    print(f"Created means: {means['id']}")
    
    # Example 3: Create a relationship between components
    print("\n--- Example 3: Create Component Relationship ---")
    relationship = await adapter.add_relationship(
        source_id=means["id"],
        target_id=principle["id"],
        relationship_type="implements",
        strength=0.8
    )
    print(f"Created relationship: {means['id']} implements {principle['id']}")
    
    # Example 4: Get related components
    print("\n--- Example 4: Get Related Components ---")
    related = await adapter.get_related_components(means["id"])
    print(f"Components related to {means['id']}: {len(related)}")
    for component in related:
        print(f"  - {component['id']}: {component['name']}")
    
    # Example 5: Create an asynchronous task
    print("\n--- Example 5: Create Asynchronous Task ---")
    task_id = await adapter.create_task(
        task_type="component_analysis",
        component_ids=[principle["id"], means["id"]],
        task_data={
            "analysis_type": "dependency",
            "depth": 2
        }
    )
    print(f"Created task: {task_id}")
    
    # Wait a moment for task to begin processing
    await asyncio.sleep(1)
    
    # Check task status
    task_status = await adapter.get_task_status(task_id)
    print(f"Task status: {task_status['status']}")
    
    # Example 6: Register an artifact
    print("\n--- Example 6: Register Artifact ---")
    artifact = await adapter.register_artifact(
        artifact_type="analysis_result",
        content={
            "component_count": 2,
            "relationship_count": 1,
            "analysis_summary": "Components form a cohesive implementation structure"
        },
        metadata={
            "analysis_type": "dependency",
            "component_ids": [principle["id"], means["id"]]
        },
        component_id=principle["id"]
    )
    print(f"Registered artifact: {artifact['artifact_id']}")
    
    # Example 7: List artifacts
    print("\n--- Example 7: List Artifacts ---")
    artifacts = await adapter.list_artifacts()
    print(f"Available artifacts: {len(artifacts)}")
    for a in artifacts:
        print(f"  - {a['id']} ({a['type']})")
    
    # Example 8: Apply principle to evaluate data
    print("\n--- Example 8: Apply Principle to Evaluate Data ---")
    evaluation = await adapter.apply_principle(
        principle_id=principle["id"],
        target_data="The system should dynamically adapt to new message formats as they are encountered."
    )
    print(f"Principle evaluation score: {evaluation['alignment_score']}")
    for insight in evaluation["insights"]:
        print(f"  - {insight}")
    
    # Example 9: Use workflow tools to evaluate with multiple principles
    print("\n--- Example 9: Use Workflow Tools ---")
    workflow_tools = workflow_context["empire_framework"]["principles"]
    
    # Get all principles
    principles = await workflow_tools["get_principles"]()
    print(f"Found {len(principles)} principles")
    
    # Evaluate with all principles
    evaluation = await workflow_tools["evaluate_with_principles"](
        content="Systems should prioritize adaptability while maintaining core functionality."
    )
    print(f"Overall alignment score: {evaluation['overall_alignment']}")
    print(f"Evaluated against {evaluation['principle_count']} principles")
    
    print("\n=== End of Empire Framework ADK Integration Example ===")


if __name__ == "__main__":
    asyncio.run(main())
