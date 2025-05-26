#!/usr/bin/env python3
"""
Conflict Engagement Example

This module demonstrates how to use the engage_with_conflict function to
proactively detect and engage with conflicts or misunderstandings, rather
than waiting for them to escalate.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

from conflict_resolver import ConflictResolver, ConflictType, ConflictSeverity
from relationship_tracker import RelationshipTracker
from communication_style_analyzer import CommunicationStyleAnalyzer
from emotional_intelligence import EmotionalIntelligence
from conflict_engagement import engage_with_conflict, EngagementType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ConflictEngagementExample")

def create_example_message(
    method: str,
    params: Dict[str, Any],
    message_id: str = None
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

def main() -> None:
    """Run the Conflict Engagement example."""
    print_separator("1. Initializing Components")
    
    # Initialize components
    agent_id = "harmony-agent-001"
    conflict_resolver = ConflictResolver(agent_id=agent_id)
    relationship_tracker = RelationshipTracker(agent_id=agent_id)
    communication_analyzer = CommunicationStyleAnalyzer(agent_id=agent_id)
    emotional_intelligence = EmotionalIntelligence(agent_id=agent_id)
    
    print(f"Initialized components for agent: {agent_id}")
    
    # Example 1: Clear Misunderstanding
    print_separator("2. Engaging with a Clear Misunderstanding")
    
    misunderstanding_message = create_example_message(
        "execute",
        {
            "action": "process_request",
            "text": "I'm confused about what you meant in your last message. Could you clarify what you're asking for?",
            "sender_id": "agent-alice",
            "conversation_id": "conv-1"
        }
    )
    
    engagement_plan = engage_with_conflict(
        conflict_resolver=conflict_resolver,
        message=misunderstanding_message,
        agent_id="agent-alice",
        conversation_id="conv-1",
        relationship_tracker=relationship_tracker,
        communication_analyzer=communication_analyzer,
        emotional_intelligence=emotional_intelligence
    )
    
    print(f"Engagement plan created with severity: {engagement_plan.severity:.2f}")
    print(f"Explanation: {engagement_plan.explanation}")
    
    print("\nPrimary actions:")
    for i, action in enumerate(engagement_plan.primary_actions):
        print(f"  {i+1}. [{action.engagement_type}] {action.content}")
    
    if engagement_plan.alternative_actions:
        print("\nAlternative actions:")
        for i, action in enumerate(engagement_plan.alternative_actions):
            print(f"  {i+1}. [{action.engagement_type}] {action.content}")
    
    # Example 2: Potential Value Conflict
    print_separator("3. Engaging with a Potential Value Conflict")
    
    value_conflict_message = create_example_message(
        "execute",
        {
            "action": "process_request",
            "text": "Your approach violates our principle of data privacy. I cannot support this recommendation.",
            "sender_id": "agent-bob",
            "conversation_id": "conv-2"
        }
    )
    
    engagement_plan = engage_with_conflict(
        conflict_resolver=conflict_resolver,
        message=value_conflict_message,
        agent_id="agent-bob",
        conversation_id="conv-2",
        relationship_tracker=relationship_tracker,
        communication_analyzer=communication_analyzer,
        emotional_intelligence=emotional_intelligence
    )
    
    print(f"Conflict type: {engagement_plan.conflict_type.value if engagement_plan.conflict_type else 'None'}")
    print(f"Severity: {engagement_plan.severity:.2f}")
    print(f"Explanation: {engagement_plan.explanation}")
    
    print("\nPrimary actions:")
    for i, action in enumerate(engagement_plan.primary_actions):
        print(f"  {i+1}. [{action.engagement_type}] {action.content}")
    
    # Example 3: Noncommittal Response
    print_separator("4. Engaging with a Noncommittal Response")
    
    noncommittal_message = create_example_message(
        "execute",
        {
            "action": "process_request",
            "text": "I guess we could try that approach if you really think it's necessary.",
            "sender_id": "agent-charlie",
            "conversation_id": "conv-3"
        }
    )
    
    engagement_plan = engage_with_conflict(
        conflict_resolver=conflict_resolver,
        message=noncommittal_message,
        agent_id="agent-charlie",
        conversation_id="conv-3",
        relationship_tracker=relationship_tracker,
        communication_analyzer=communication_analyzer,
        emotional_intelligence=emotional_intelligence
    )
    
    print(f"Detected signs: {len(engagement_plan.detected_signs)}")
    for sign in engagement_plan.detected_signs:
        print(f"  - {sign['name']}: {sign['description']}")
        print(f"    Matched text: {sign['matched_text']}")
        print(f"    Severity: {sign['severity']:.2f}")
    
    print("\nPrimary actions:")
    for i, action in enumerate(engagement_plan.primary_actions):
        print(f"  {i+1}. [{action.engagement_type}] {action.content}")
    
    # Example 4: Defensive Response
    print_separator("5. Engaging with a Defensive Response")
    
    defensive_message = create_example_message(
        "execute",
        {
            "action": "process_request",
            "text": "That's not what I meant at all. You're completely misinterpreting what I said.",
            "sender_id": "agent-diana",
            "conversation_id": "conv-4"
        }
    )
    
    engagement_plan = engage_with_conflict(
        conflict_resolver=conflict_resolver,
        message=defensive_message,
        agent_id="agent-diana",
        conversation_id="conv-4",
        relationship_tracker=relationship_tracker,
        communication_analyzer=communication_analyzer,
        emotional_intelligence=emotional_intelligence
    )
    
    print(f"Explanation: {engagement_plan.explanation}")
    
    print("\nPrimary actions:")
    for i, action in enumerate(engagement_plan.primary_actions):
        print(f"  {i+1}. [{action.engagement_type}] {action.content}")
    
    if engagement_plan.long_term_steps:
        print("\nLong-term steps:")
        for i, step in enumerate(engagement_plan.long_term_steps):
            print(f"  {i+1}. {step['description']}")
            print(f"     Reasoning: {step['reasoning']}")
            print(f"     Expected outcome: {step['expected_outcome']}")
            if step.get('timeframe'):
                print(f"     Timeframe: {step['timeframe']}")
    
    # Example 5: Implementing the Best Action
    print_separator("6. Implementing the Best Action")
    
    best_action = engagement_plan.get_best_action()
    if best_action:
        print(f"Best action: [{best_action.engagement_type}] {best_action.content}")
        print(f"Priority: {best_action.priority}")
        print(f"Confidence: {best_action.confidence:.2f}")
        
        # Simulate implementing this action
        print("\nImplementing this action would look like:")
        print(f"Agent {agent_id} to Agent diana: {best_action.content}")
    
    print_separator("Example Complete")

if __name__ == "__main__":
    main()
