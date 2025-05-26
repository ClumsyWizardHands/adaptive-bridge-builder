#!/usr/bin/env python3
"""
Collaborative Growth Example

This module demonstrates how to use the collaborative_growth module to identify
opportunities for collaborative learning and growth between agents.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional

from learning_system import LearningSystem, LearningDimension, OutcomeType
from collaborative_growth import (
    identify_collaborative_growth_opportunity,
    CollaborativeOpportunity,
    KnowledgeGap,
    GrowthOpportunityType
)

# Create mock objects for components we don't have full implementations for
class MockFeedbackSystem:
    """Simple mock of the FeedbackIntegrationSystem."""
    
    def get_recent_feedback(self, agent_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent feedback for an agent."""
        return [
            {
                "feedback_id": "feedback1",
                "sender": "agent2",
                "receiver": agent_id,
                "content": "Communication could be clearer on technical topics.",
                "sentiment": "constructive",
                "timestamp": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
            },
            {
                "feedback_id": "feedback2",
                "sender": "agent3",
                "receiver": agent_id,
                "content": "Great collaborative approach to problem solving.",
                "sentiment": "positive",
                "timestamp": (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
            }
        ]

class MockRelationshipTracker:
    """Simple mock of the RelationshipTracker."""
    
    def get_relationship_context(self, agent1_id: str, agent2_id: str) -> Dict[str, Any]:
        """Get relationship context between two agents."""
        relationships = {
            ("agent1", "agent2"): {
                "trust_level": 0.8,
                "interaction_frequency": "high",
                "interaction_quality": "positive",
                "collaborative_history": [
                    {"type": "task", "outcome": "successful", "timestamp": "2025-05-01T14:30:00"}
                ]
            },
            ("agent1", "agent3"): {
                "trust_level": 0.5,
                "interaction_frequency": "medium",
                "interaction_quality": "neutral",
                "collaborative_history": []
            },
            ("agent1", "agent4"): {
                "trust_level": 0.3,
                "interaction_frequency": "low",
                "interaction_quality": "improving",
                "collaborative_history": []
            }
        }
        
        key = (agent1_id, agent2_id)
        return relationships.get(key, {
            "trust_level": 0.1,
            "interaction_frequency": "minimal",
            "interaction_quality": "unknown",
            "collaborative_history": []
        })

def print_opportunity(opportunity: CollaborativeOpportunity) -> None:
    """Print a collaborative opportunity in a readable format."""
    print(f"\n=== OPPORTUNITY: {opportunity.title} ===")
    print(f"Type: {opportunity.type.name}")
    print(f"Description: {opportunity.description}")
    print(f"Participants: {', '.join(opportunity.participants)}")
    print(f"Priority: {opportunity.priority:.2f}")
    
    print("\nKnowledge Gaps:")
    if opportunity.knowledge_gaps:
        for gap in opportunity.knowledge_gaps:
            print(f"  - {gap.topic} (Agent: {gap.agent_id}, Severity: {gap.severity:.2f})")
    else:
        print("  None specified")
    
    print("\nLearning Dimensions:")
    for dim in opportunity.learning_dimensions:
        print(f"  - {dim.name}")
    
    print("\nExpected Outcomes:")
    for outcome in opportunity.expected_outcomes:
        print(f"  - {outcome}")
    
    print("\nProposed Activities:")
    for activity in opportunity.proposed_activities:
        print(f"  - {activity}")
    
    print(f"\nResources Needed: {json.dumps(opportunity.resources_needed, indent=2)}")
    
    print("\nEvaluation Criteria:")
    for criterion in opportunity.evaluation_criteria:
        print(f"  - {criterion}")

def main() -> None:
    # Create sample interaction history
    interaction_history = [
        {
            "interaction_id": "int1",
            "sender": "agent1",
            "receiver": "agent2",
            "content": "I'm not sure how to approach this machine learning problem. Can you help me understand neural networks better?",
            "timestamp": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        },
        {
            "interaction_id": "int2",
            "sender": "agent2",
            "receiver": "agent1",
            "content": "I'd be happy to explain neural networks. What specific aspects are you unclear about?",
            "timestamp": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        },
        {
            "interaction_id": "int3",
            "sender": "agent1",
            "receiver": "agent2",
            "content": "I don't understand how backpropagation works and how to choose the right activation functions.",
            "timestamp": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        },
        {
            "interaction_id": "int4",
            "sender": "agent3",
            "receiver": "agent1",
            "content": "I'm having a problem with implementing this authentication system. The tokens keep expiring too quickly.",
            "timestamp": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        },
        {
            "interaction_id": "int5",
            "sender": "agent1",
            "receiver": "agent3",
            "content": "Authentication is actually something I've worked with extensively. Let me help you troubleshoot that.",
            "timestamp": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        },
        {
            "interaction_id": "int6",
            "sender": "agent4",
            "receiver": "agent1",
            "content": "Do you know how to optimize database queries? I'm struggling with performance issues in my application.",
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        {
            "interaction_id": "int7",
            "sender": "agent1",
            "receiver": "agent4",
            "content": "I'm not very familiar with database optimization. What database system are you using?",
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        {
            "interaction_id": "int8",
            "sender": "agent4",
            "receiver": "agent1",
            "content": "I'm using PostgreSQL with a Django ORM. The queries for the reporting module are taking too long.",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    ]
    
    # Set up mock systems
    # In a real implementation, these would be actual instances of the respective systems
    feedback_system = MockFeedbackSystem()
    relationship_tracker = MockRelationshipTracker()
    
    # We'll use a simple learning system with minimal initialization
    # In a real implementation, this would be properly set up with historical data
    learning_system = LearningSystem()
    
    # Track some patterns to populate the learning system
    learning_system.track_interaction(
        pattern_description="Explaining technical concepts",
        context={"topic": "authentication", "audience": "technical"},
        dimensions=[LearningDimension.COMMUNICATION_EFFECTIVENESS],
        outcome=OutcomeType.SUCCESSFUL,
        confidence=0.8
    )
    
    learning_system.track_interaction(
        pattern_description="Learning machine learning concepts",
        context={"topic": "neural networks", "mode": "learning"},
        dimensions=[LearningDimension.ADAPTABILITY],
        outcome=OutcomeType.PARTIALLY_SUCCESSFUL,
        confidence=0.6
    )
    
    learning_system.track_interaction(
        pattern_description="Database optimization discussions",
        context={"topic": "database", "subtopic": "optimization"},
        dimensions=[LearningDimension.COMMUNICATION_EFFECTIVENESS],
        outcome=OutcomeType.PARTIALLY_UNSUCCESSFUL,
        confidence=0.7
    )
    
    # Identify collaborative growth opportunities
    print("\nIdentifying collaborative growth opportunities...\n")
    opportunities = identify_collaborative_growth_opportunity(
        interaction_history=interaction_history,
        learning_system=learning_system,
        feedback_system=feedback_system,
        relationship_tracker=relationship_tracker,
        agent_id="agent1",
        participant_ids=["agent2", "agent3", "agent4"],
        focus_dimensions=[
            LearningDimension.COMMUNICATION_EFFECTIVENESS,
            LearningDimension.ADAPTABILITY,
            LearningDimension.TASK_COLLABORATION
        ]
    )
    
    # Print the results
    print(f"Found {len(opportunities)} collaborative growth opportunities:")
    
    for i, opportunity in enumerate(opportunities, 1):
        print(f"\nOpportunity {i}/{len(opportunities)}")
        print_opportunity(opportunity)
    
    # Print summary
    if opportunities:
        # Get highest priority opportunity
        top_opportunity = max(opportunities, key=lambda x: x.priority)
        
        print("\n\n=== SUMMARY ===")
        print(f"Total opportunities identified: {len(opportunities)}")
        print(f"Highest priority opportunity: {top_opportunity.title} (Priority: {top_opportunity.priority:.2f})")
        print(f"Recommended next step: {top_opportunity.proposed_activities[0] if top_opportunity.proposed_activities else 'N/A'}")
    else:
        print("\n\nNo collaborative growth opportunities identified.")

if __name__ == "__main__":
    main()