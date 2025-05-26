#!/usr/bin/env python3
"""
Conflict Resolver Example

This module demonstrates the usage of the ConflictResolver class,
showing how it detects, categorizes, and resolves conflicts between agents.
It applies the "Harmony Through Presence" principle by actively monitoring
for tensions in communication, implementing appropriate resolution strategies,
and creating distance when conflicts can't be resolved.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from conflict_resolver import (
    ConflictResolver,
    ConflictType,
    ConflictSeverity,
    ConflictIndicator,
    ResolutionStrategy,
    ResolutionOutcome,
    ConflictResolutionStep
)
from relationship_tracker import (
    RelationshipTracker,
    InteractionType,
    InteractionQuality
)
from principle_engine import PrincipleEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ConflictResolverExample")

def create_example_message(
    method: str,
    params: Dict[str, Any],
    message_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create an example message for testing."""
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": message_id or f"msg-{uuid.uuid4().hex[:8]}"
    }

def print_separator(title: str = None) -> None:
    """Print a separator line with optional title."""
    width = 80
    if title:
        print("\n" + "=" * 10 + f" {title} " + "=" * (width - len(title) - 12) + "\n")
    else:
        print("\n" + "=" * width + "\n")

def create_test_data() -> None:
    """Create test data for different conflict types."""
    test_data = {}
    
    # Goal conflicts
    test_data["goal"] = [
        create_example_message(
            "execute",
            {
                "action": "process_request",
                "text": "I disagree with your approach to solving this problem. My goal is to optimize for speed while you seem to be prioritizing accuracy.",
                "sender_id": "agent-alice",
                "conversation_id": "conv-1"
            }
        ),
        create_example_message(
            "execute",
            {
                "action": "process_request",
                "text": "We have conflicting objectives. You want to maximize user engagement, but our primary goal should be data privacy.",
                "sender_id": "agent-bob",
                "conversation_id": "conv-2"
            }
        )
    ]
    
    # Value conflicts
    test_data["value"] = [
        create_example_message(
            "execute", 
            {
                "action": "process_request",
                "text": "Your approach violates the principle of fairness that we agreed to follow.",
                "sender_id": "agent-charlie",
                "conversation_id": "conv-3"
            }
        ),
        create_example_message(
            "execute",
            {
                "action": "process_request",
                "text": "I believe it's ethically wrong to collect this much data from users without explicit consent.",
                "sender_id": "agent-diana",
                "conversation_id": "conv-4"
            }
        )
    ]
    
    # Factual conflicts
    test_data["factual"] = [
        create_example_message(
            "execute",
            {
                "action": "process_request",
                "text": "The data you provided is incorrect. The actual conversion rate was 3.2%, not 5.7%.",
                "sender_id": "agent-eve",
                "conversation_id": "conv-5"
            }
        ),
        create_example_message(
            "execute",
            {
                "action": "process_request",
                "text": "The statistics in your analysis are wrong. Let me provide the accurate numbers based on our database.",
                "sender_id": "agent-frank",
                "conversation_id": "conv-6"
            }
        )
    ]
    
    # Communication conflicts
    test_data["communication"] = [
        create_example_message(
            "execute",
            {
                "action": "process_request",
                "text": "I think there's been a misunderstanding about what I meant in my last message.",
                "sender_id": "agent-grace",
                "conversation_id": "conv-7"
            }
        ),
        create_example_message(
            "execute",
            {
                "action": "process_request",
                "text": "It seems like our communication has broken down. We're talking past each other.",
                "sender_id": "agent-harry",
                "conversation_id": "conv-8"
            }
        )
    ]
    
    # Relationship conflicts
    test_data["relationship"] = [
        create_example_message(
            "execute",
            {
                "action": "process_request",
                "text": "I'm frustrated by how you've been dismissing my input lately. It feels like you don't value my contributions.",
                "sender_id": "agent-isabel",
                "conversation_id": "conv-9"
            }
        ),
        create_example_message(
            "execute",
            {
                "action": "process_request",
                "text": "I don't trust the data you're providing anymore after those previous errors.",
                "sender_id": "agent-jack",
                "conversation_id": "conv-10"
            }
        )
    ]
    
    return test_data

def main() -> None:
    """Run the Conflict Resolver example."""
    print_separator("1. Initializing Components")
    
    # Create temporary directories for data
    data_dir = "data/example_conflicts"
    rel_data_dir = "data/example_relationships"
    
    # Initialize components
    principle_engine = PrincipleEngine()
    relationship_tracker = RelationshipTracker(
        agent_id="harmony-agent-001",
        data_dir=rel_data_dir
    )
    conflict_resolver = ConflictResolver(
        agent_id="harmony-agent-001",
        principle_engine=principle_engine,
        relationship_tracker=relationship_tracker,
        data_dir=data_dir
    )
    
    print(f"Initialized ConflictResolver for agent: harmony-agent-001")
    print(f"Conflict data directory: {data_dir}")
    print(f"Using PrincipleEngine: {principle_engine is not None}")
    print(f"Using RelationshipTracker: {relationship_tracker is not None}")
    
    # Create test data
    test_data = create_test_data()
    
    # Example 1: Detecting Conflicts
    print_separator("2. Detecting Conflicts")
    
    # Detect a goal conflict
    goal_message = test_data["goal"][0]
    goal_indicators = conflict_resolver.detect_conflicts(
        message=goal_message,
        agent_id="agent-alice",
        conversation_id="conv-1"
    )
    
    print(f"Detected {len(goal_indicators)} indicators in goal conflict message")
    for i, indicator in enumerate(goal_indicators):
        print(f"  {i+1}. Type: {indicator.conflict_type.value.upper()}, " 
              f"Trigger: {indicator.trigger_name}, "
              f"Severity: {indicator.severity.value.upper()}")
        print(f"     Matched text: '{indicator.matched_text}'")
        print(f"     Confidence: {indicator.confidence:.2f}")
    
    # Detect a value conflict
    value_message = test_data["value"][0]
    value_indicators = conflict_resolver.detect_conflicts(
        message=value_message,
        agent_id="agent-charlie",
        conversation_id="conv-3"
    )
    
    print(f"\nDetected {len(value_indicators)} indicators in value conflict message")
    for i, indicator in enumerate(value_indicators):
        print(f"  {i+1}. Type: {indicator.conflict_type.value.upper()}, " 
              f"Severity: {indicator.severity.value.upper()}")
        print(f"     Matched text: '{indicator.matched_text}'")
    
    # Example 2: Categorizing Conflicts
    print_separator("3. Categorizing Conflicts")
    
    # Categorize the goal conflict
    goal_categorization = conflict_resolver.categorize_conflict(
        indicators=goal_indicators,
        agent_id="agent-alice",
        message=goal_message
    )
    
    if goal_categorization:
        conflict_type, severity, description = goal_categorization
        print(f"Goal conflict categorization:")
        print(f"  Type: {conflict_type.value.upper()}")
        print(f"  Severity: {severity.value.upper()}")
        print(f"  Description: {description}")
    
    # Categorize the value conflict
    value_categorization = conflict_resolver.categorize_conflict(
        indicators=value_indicators,
        agent_id="agent-charlie",
        message=value_message
    )
    
    if value_categorization:
        conflict_type, severity, description = value_categorization
        print(f"\nValue conflict categorization:")
        print(f"  Type: {conflict_type.value.upper()}")
        print(f"  Severity: {severity.value.upper()}")
        print(f"  Description: {description}")
    
    # Example 3: Creating Conflict Records
    print_separator("4. Creating Conflict Records")
    
    # Create a conflict record for the goal conflict
    goal_conflict_record = conflict_resolver.create_conflict_record(
        indicators=goal_indicators,
        agent_id="agent-alice",
        message=goal_message,
        conversation_id="conv-1",
        metadata={"source": "example_script", "priority": "medium"}
    )
    
    if goal_conflict_record:
        print(f"Created conflict record for goal conflict:")
        print(f"  Conflict ID: {goal_conflict_record.conflict_id}")
        print(f"  Type: {goal_conflict_record.conflict_type.value}")
        print(f"  Severity: {goal_conflict_record.severity.value}")
        print(f"  Status: {goal_conflict_record.status}")
        print(f"  Agents involved: {', '.join(goal_conflict_record.agents)}")
    
    # Create a conflict record for the value conflict
    value_conflict_record = conflict_resolver.create_conflict_record(
        indicators=value_indicators,
        agent_id="agent-charlie",
        message=value_message,
        conversation_id="conv-3"
    )
    
    if value_conflict_record:
        print(f"\nCreated conflict record for value conflict:")
        print(f"  Conflict ID: {value_conflict_record.conflict_id}")
        print(f"  Type: {value_conflict_record.conflict_type.value}")
        print(f"  Severity: {value_conflict_record.severity.value}")
        print(f"  Status: {value_conflict_record.status}")
    
    # Example 4: Creating Resolution Plans
    print_separator("5. Creating Resolution Plans")
    
    # Create a resolution plan for the goal conflict
    goal_conflict_id = goal_conflict_record.conflict_id
    goal_resolution_plan = conflict_resolver.create_resolution_plan(goal_conflict_id)
    
    print(f"Created resolution plan for goal conflict with {len(goal_resolution_plan)} steps:")
    for i, step in enumerate(goal_resolution_plan):
        print(f"  Step {i+1}: {step.step_type} ({step.strategy.value})")
        print(f"    {step.description}")
        print(f"    Expected outcome: {step.expected_outcome}")
        if step.dependencies:
            print(f"    Dependencies: {len(step.dependencies)} prior steps")
    
    # Create a resolution plan for the value conflict
    value_conflict_id = value_conflict_record.conflict_id
    value_resolution_plan = conflict_resolver.create_resolution_plan(value_conflict_id)
    
    print(f"\nCreated resolution plan for value conflict with {len(value_resolution_plan)} steps:")
    for i, step in enumerate(value_resolution_plan):
        print(f"  Step {i+1}: {step.step_type} ({step.strategy.value})")
        print(f"    {step.description}")
    
    # Example 5: Implementing Resolution Strategies
    print_separator("6. Implementing Resolution Strategies")
    
    # Get the first step from the goal conflict resolution plan (acknowledgment)
    acknowledgment_step = goal_resolution_plan[0]
    
    # Implement the acknowledgment step
    implementation_result = conflict_resolver.implement_resolution_strategy(
        conflict_id=goal_conflict_id,
        step_id=acknowledgment_step.step_id,
        implementation_details={
            "description": "Acknowledged the existence of the goal conflict with agent-alice. "
                           "Validated their perspective on prioritizing speed.",
            "common_interests": ["delivering high-quality results", "meeting project deadlines"],
            "outcome": "Both parties acknowledged the conflict exists and expressed willingness to find a solution."
        }
    )
    
    print(f"Implemented acknowledgment step for goal conflict:")
    print(f"  Success: {implementation_result.get('success', False)}")
    print(f"  Message: {implementation_result.get('message', '')}")
    if implementation_result.get('details'):
        print(f"  Details: {implementation_result.get('details')}")
    
    # Get next actionable step (should be clarification since acknowledgment is now complete)
    goal_conflict = conflict_resolver.active_conflicts[goal_conflict_id]
    next_steps = goal_conflict.get_next_actionable_steps()
    
    if next_steps:
        clarification_step = next_steps[0]
        print(f"\nNext actionable step: {clarification_step.step_type}")
        print(f"  Strategy: {clarification_step.strategy.value}")
        print(f"  Description: {clarification_step.description}")
        
        # Implement the clarification step
        implementation_result = conflict_resolver.implement_resolution_strategy(
            conflict_id=goal_conflict_id,
            step_id=clarification_step.step_id,
            implementation_details={
                "description": "Gathered detailed information about agent-alice's priorities and constraints. "
                              "Clarified our own position on balancing speed and accuracy.",
                "findings": [
                    "Agent-alice has a strict deadline that requires optimizing for speed",
                    "Our concern is that rushing may introduce errors that are costly to fix later",
                    "Both parties agree that a balance is necessary, but disagree on where the balance point should be"
                ],
                "outcome": "Clear understanding of both positions and the core points of disagreement."
            }
        )
        
        print(f"\nImplemented clarification step for goal conflict:")
        print(f"  Success: {implementation_result.get('success', False)}")
        print(f"  Message: {implementation_result.get('message', '')}")
    
    # Example 6: Recording Resolution Outcomes
    print_separator("7. Recording Resolution Outcomes")
    
    # Record successful resolution of the goal conflict
    resolution_outcome = conflict_resolver.record_resolution_outcome(
        conflict_id=goal_conflict_id,
        outcome=ResolutionOutcome.RESOLVED,
        details={
            "notes": "Successfully resolved the goal conflict by developing a phased approach "
                    "that balances speed and accuracy. Initial phase will focus on speed to meet "
                    "deadline, followed by an accuracy-focused refinement phase.",
            "principle_alignment": 0.9,
            "metrics": {
                "resolution_time_hours": 2.5,
                "compromise_level": "high",
                "satisfaction_score": 0.85
            }
        }
    )
    
    print(f"Recorded resolution outcome for goal conflict:")
    print(f"  Success: {resolution_outcome.get('success', False)}")
    print(f"  Message: {resolution_outcome.get('message', '')}")
    if 'details' in resolution_outcome:
        print(f"  Outcome: {resolution_outcome['details'].get('outcome')}")
        print(f"  Principle alignment: {resolution_outcome['details'].get('principle_alignment')}")
    
    # Example 7: Creating Distance When Resolution Fails
    print_separator("8. Creating Distance When Resolution Fails")
    
    # First create a new conflict record for a relationship conflict
    relationship_message = test_data["relationship"][1]
    relationship_indicators = conflict_resolver.detect_conflicts(
        message=relationship_message,
        agent_id="agent-jack",
        conversation_id="conv-10"
    )
    
    relationship_conflict_record = conflict_resolver.create_conflict_record(
        indicators=relationship_indicators,
        agent_id="agent-jack",
        message=relationship_message,
        conversation_id="conv-10"
    )
    
    relationship_conflict_id = relationship_conflict_record.conflict_id
    print(f"Created new conflict record for relationship conflict: {relationship_conflict_id}")
    
    # Create a resolution plan for the relationship conflict
    relationship_resolution_plan = conflict_resolver.create_resolution_plan(relationship_conflict_id)
    print(f"Created resolution plan with {len(relationship_resolution_plan)} steps")
    
    # After trying to resolve the conflict and failing (simulated), create distance
    distance_result = conflict_resolver.create_distance_measures(
        conflict_id=relationship_conflict_id,
        distance_type="temporal",
        parameters={
            "duration": "48 hours",
            "reason": "Trust breach requires cooling-off period",
            "resumption_conditions": [
                "Verification of data accuracy through third party",
                "Establishment of new data validation protocols"
            ]
        }
    )
    
    print(f"Created temporal distance for relationship conflict:")
    print(f"  Success: {distance_result.get('success', False)}")
    print(f"  Message: {distance_result.get('message', '')}")
    print(f"  Type: {distance_result.get('details', {}).get('type')}")
    print(f"  Duration: {distance_result.get('details', {}).get('duration')}")
    print(f"  Reason: {distance_result.get('details', {}).get('reason')}")
    print(f"  Resumption conditions: {distance_result.get('details', {}).get('resumption_conditions')}")
    
    # Example 8: Get Active Conflicts
    print_separator("9. Getting Active Conflicts")
    
    # Create one more conflict
    factual_message = test_data["factual"][0]
    factual_indicators = conflict_resolver.detect_conflicts(
        message=factual_message,
        agent_id="agent-eve",
        conversation_id="conv-5"
    )
    
    factual_conflict_record = conflict_resolver.create_conflict_record(
        indicators=factual_indicators,
        agent_id="agent-eve",
        message=factual_message,
        conversation_id="conv-5"
    )
    
    # Get active conflicts - should include the value conflict and factual conflict
    # (goal conflict and relationship conflict were resolved)
    active_conflicts = conflict_resolver.get_active_conflicts()
    
    print(f"Currently active conflicts: {len(active_conflicts)}")
    for conflict in active_conflicts:
        print(f"  Conflict ID: {conflict['conflict_id']}")
        print(f"  Type: {conflict['conflict_type'].upper()}")
        print(f"  Severity: {conflict['severity'].upper()}")
        print(f"  Status: {conflict['status']}")
        print(f"  Agents: {', '.join(conflict['agents'])}")
        print(f"  Resolution progress: {conflict['resolution_progress']*100:.1f}%\n")
    
    # Example 9: Retrieving Conflict Details
    print_separator("10. Getting Conflict Details")
    
    # Get details of the value conflict
    value_conflict_details = conflict_resolver.get_conflict_details(value_conflict_id)
    
    if value_conflict_details:
        print(f"Value conflict details:")
        print(f"  Conflict ID: {value_conflict_details['conflict_id']}")
        print(f"  Created: {value_conflict_details['created_at']}")
        print(f"  Status: {value_conflict_details['status']}")
        print(f"  Type: {value_conflict_details['conflict_type']}")
        print(f"  Severity: {value_conflict_details['severity']}")
        print(f"  Resolution plan steps: {len(value_conflict_details['resolution_plan'])}")
        print(f"  Resolution progress: {value_conflict_details['resolution_progress']*100:.1f}%")
        
        # Show agent relationships if available
        if value_conflict_details.get('agent_relationships'):
            for agent_id, rel_data in value_conflict_details['agent_relationships'].items():
                print(f"\n  Relationship with {agent_id}:")
                print(f"    Trust level: {rel_data['trust_level']}")
                print(f"    Trust score: {rel_data['trust_score']:.2f}")
                print(f"    Status: {rel_data['status']}")
    
    # Example 10: Resolution Statistics
    print_separator("11. Getting Resolution Statistics")
    
    # Get overall resolution statistics
    resolution_stats = conflict_resolver.get_resolution_statistics()
    
    print(f"Conflict resolution statistics:")
    print(f"  Total conflicts detected: {resolution_stats['total_conflicts']}")
    print(f"  Active conflicts: {resolution_stats['active_conflicts']}")
    print(f"  Resolved conflicts: {resolution_stats['resolved_conflicts']}")
    print(f"  Resolution success rate: {resolution_stats['success_rate']*100:.1f}%")
    
    print(f"\nResolutions by outcome:")
    for outcome, count in resolution_stats['outcomes'].items():
        print(f"  {outcome}: {count}")
    
    print(f"\nConflicts by type:")
    for type_name, count in resolution_stats['conflict_types'].items():
        print(f"  {type_name}: {count}")
    
    print(f"\nAverage resolution time: {resolution_stats['avg_resolution_time']:.1f} hours")
    print(f"Average principle alignment: {resolution_stats['avg_principle_alignment']:.2f}")
    
    print_separator("Example Complete")

if __name__ == "__main__":
    main()