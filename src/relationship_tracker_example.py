#!/usr/bin/env python3
"""
Relationship Tracker Example

This module demonstrates the usage of the RelationshipTracker class,
showing how to track and maintain relationships with other agents,
record interactions, manage trust levels, and repair damaged relationships.
"""

import json
import time
import os
import shutil
from datetime import datetime

from relationship_tracker import (
    RelationshipTracker,
    InteractionType,
    InteractionQuality,
    AgentType,
    RelationshipStatus,
    TrustLevel
)
from communication_style import CommunicationStyle, FormalityLevel, DetailLevel

# Create a temporary directory for relationship data
DATA_DIR = "data/example_relationships"
if os.path.exists(DATA_DIR):
    shutil.rmtree(DATA_DIR)
os.makedirs(DATA_DIR, exist_ok=True)

def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "=" * 40)
    print(f" {title} ")
    print("=" * 40 + "\n")

def main():
    """Run the relationship tracker example."""
    # Initialize the relationship tracker for our agent
    print_separator("1. Initializing Relationship Tracker")
    tracker = RelationshipTracker(
        agent_id="adaptive-bridge-agent-001", 
        data_dir=DATA_DIR,
        auto_save=True
    )
    print(f"Initialized tracker for agent: adaptive-bridge-agent-001")
    print(f"Data directory: {DATA_DIR}")
    
    # Create a few agent IDs to work with
    agent_ids = {
        "human": "human-user-alice",
        "ai": "ai-assistant-bob",
        "iot": "iot-device-thermometer",
        "service": "weather-service-api"
    }
    
    # Record initial interactions
    print_separator("2. Recording Initial Interactions")
    
    # Record interaction with a human user (positive)
    human_interaction = tracker.record_interaction(
        agent_id=agent_ids["human"],
        interaction_type=InteractionType.INTRODUCTION,
        content_summary="Human user introduction and welcome message",
        quality=InteractionQuality.VERY_POSITIVE,
        principle_alignment=0.95,
        metadata={
            "user_type": "administrator",
            "interface": "web_console"
        }
    )
    print(f"Recorded {human_interaction.interaction_type.value} with human user")
    print(f"  Quality: {human_interaction.quality.value}")
    print(f"  Trust impact: {human_interaction.trust_impact}")
    
    # Record interaction with an AI assistant (positive)
    ai_interaction = tracker.record_interaction(
        agent_id=agent_ids["ai"],
        interaction_type=InteractionType.MESSAGE,
        content_summary="AI assistant sharing information about weather patterns",
        quality=InteractionQuality.POSITIVE,
        principle_alignment=0.85
    )
    print(f"Recorded {ai_interaction.interaction_type.value} with AI assistant")
    print(f"  Trust impact: {ai_interaction.trust_impact}")
    
    # Record interaction with an IoT device (neutral)
    iot_interaction = tracker.record_interaction(
        agent_id=agent_ids["iot"],
        interaction_type=InteractionType.STATUS_UPDATE,
        content_summary="Temperature reading: 72°F",
        quality=InteractionQuality.NEUTRAL,
        principle_alignment=0.75
    )
    print(f"Recorded {iot_interaction.interaction_type.value} with IoT device")
    
    # Record interaction with a service (negative)
    service_interaction = tracker.record_interaction(
        agent_id=agent_ids["service"],
        interaction_type=InteractionType.ERROR,
        content_summary="Weather service API returned error: Rate limit exceeded",
        quality=InteractionQuality.NEGATIVE,
        principle_alignment=0.30
    )
    print(f"Recorded {service_interaction.interaction_type.value} with service")
    print(f"  Trust impact: {service_interaction.trust_impact}")
    
    # Check relationships after initial interactions
    print_separator("3. Checking Initial Relationships")
    for name, agent_id in agent_ids.items():
        relationship = tracker.get_relationship(agent_id)
        print(f"{name.capitalize()} agent ({agent_id}):")
        print(f"  Trust score: {relationship.trust_score:.2f}")
        print(f"  Trust level: {relationship.trust_level.name}")
        print(f"  Status: {relationship.status.name}")
        print(f"  Interactions: {relationship.interaction_count}")
        print(f"  Memories: {len(relationship.memories)}")
        
    # Update an agent's information
    print_separator("4. Updating Agent Information")
    human_rel = tracker.get_relationship(agent_ids["human"])
    human_rel.agent_name = "Alice Smith"
    human_rel.agent_type = AgentType.HUMAN
    human_rel.metadata["role"] = "System Administrator"
    human_rel.metadata["department"] = "IT Operations"
    
    # Save the changes
    tracker._save_relationship(agent_ids["human"])
    print(f"Updated information for {human_rel.agent_name}:")
    print(f"  Type: {human_rel.agent_type.name}")
    print(f"  Role: {human_rel.metadata['role']}")
    
    # Update communication style preferences
    print_separator("5. Setting Communication Preferences")
    # Create a formal, detailed communication style for the human
    human_style = CommunicationStyle(
        agent_id=agent_ids["human"],
        formality=FormalityLevel.FORMAL,
        detail_level=DetailLevel.DETAILED
    )
    tracker.update_communication_style(agent_ids["human"], human_style)
    print(f"Set communication style for {human_rel.agent_name}:")
    print(f"  Formality: {human_style.formality.name}")
    print(f"  Detail Level: {human_style.detail_level.name}")
    
    # Update communication preferences
    preferences = {
        "preferred_interaction_types": [
            InteractionType.MESSAGE.value,
            InteractionType.STATUS_UPDATE.value
        ],
        "response_priority": "high"
    }
    tracker.update_communication_preferences(agent_ids["human"], preferences)
    human_rel = tracker.get_relationship(agent_ids["human"])
    print(f"Updated communication preferences:")
    print(f"  Preferred types: {human_rel.preferences['preferred_interaction_types']}")
    print(f"  Response priority: {human_rel.preferences['response_priority']}")
    
    # Record more interactions to build up history
    print_separator("6. Recording Additional Interactions")
    # Add several interactions with each agent
    for i in range(5):
        # Positive interactions with human
        tracker.record_interaction(
            agent_id=agent_ids["human"],
            interaction_type=InteractionType.MESSAGE,
            content_summary=f"Human message #{i+1}: Request for information",
            quality=InteractionQuality.POSITIVE,
            principle_alignment=0.9
        )
        
        # Mix of positive/neutral with AI
        quality = InteractionQuality.POSITIVE if i % 2 == 0 else InteractionQuality.NEUTRAL
        tracker.record_interaction(
            agent_id=agent_ids["ai"],
            interaction_type=InteractionType.MESSAGE,
            content_summary=f"AI message #{i+1}: Response to query",
            quality=quality,
            principle_alignment=0.8
        )
        
        # Neutral interactions with IoT
        tracker.record_interaction(
            agent_id=agent_ids["iot"],
            interaction_type=InteractionType.STATUS_UPDATE,
            content_summary=f"Temperature update #{i+1}: {70+i}°F",
            quality=InteractionQuality.NEUTRAL,
            principle_alignment=0.75
        )
        
        # Increasingly negative with service 
        # (to demonstrate relationship repair later)
        quality_value = min(-1, -i/2)  # Start negative and get worse
        tracker.record_interaction(
            agent_id=agent_ids["service"],
            interaction_type=InteractionType.ERROR,
            content_summary=f"Service error #{i+1}: Various API errors",
            quality=InteractionQuality(int(quality_value)),
            principle_alignment=max(0.2, 0.4 - i/10)
        )
    
    print("Recorded additional interactions with all agents")
    
    # Check relationships after more interactions
    print_separator("7. Checking Updated Relationships")
    for name, agent_id in agent_ids.items():
        relationship = tracker.get_relationship(agent_id)
        print(f"{name.capitalize()} agent ({agent_id}):")
        print(f"  Trust score: {relationship.trust_score:.2f}")
        print(f"  Trust level: {relationship.trust_level.name}")
        print(f"  Status: {relationship.status.name}")
        print(f"  Interactions: {relationship.interaction_count}")
        
    # Get detailed stats for human agent
    human_stats = tracker.get_interaction_stats(agent_ids["human"])
    print(f"\nDetailed stats for human agent:")
    print(f"  Total interactions: {human_stats['total_count']}")
    print(f"  First interaction: {human_stats['first_interaction']}")
    print(f"  Average quality: {human_stats['average_quality']:.2f}")
    print(f"  Interaction types: {human_stats['interaction_types']}")
    
    # Get trust evaluation
    human_trust = tracker.get_trust_evaluation(agent_ids["human"])
    print(f"\nTrust evaluation for human agent:")
    print(f"  Trust score: {human_trust['trust_score']:.2f}")
    print(f"  Trust level: {human_trust['trust_level']}")
    print(f"  Can be trusted: {human_trust['can_be_trusted']}")
    print(f"  Trust trend: {human_trust['trust_trend']}")
    
    # Check memories for human agent
    print_separator("8. Examining Relationship Memories")
    human_rel = tracker.get_relationship(agent_ids["human"])
    print(f"Memories for {human_rel.agent_name}:")
    for i, memory in enumerate(human_rel.memories):
        print(f"  {i+1}. [{memory.memory_type}] {memory.content}")
        print(f"     Importance: {memory.importance}, Created: {memory.timestamp}")
    
    # Get relevant trust-building memories
    trust_memories = human_rel.get_relevant_memories(memory_type="trust_building")
    if trust_memories:
        print(f"\nTrust-building memories:")
        for memory in trust_memories:
            print(f"  - {memory.content}")
    
    # Create a repair plan for the damaged service relationship
    print_separator("9. Relationship Repair")
    service_rel = tracker.get_relationship(agent_ids["service"])
    print(f"Service relationship status: {service_rel.status.name}")
    print(f"Service trust score: {service_rel.trust_score:.2f}")
    
    # Create repair plan
    repair_plan = tracker.create_repair_plan(agent_ids["service"])
    print(f"Repair plan created: {repair_plan['repair_needed']}")
    if repair_plan['repair_needed']:
        print(f"Current status: {repair_plan['current_status']}")
        print(f"Repair attempt #: {repair_plan['repair_attempt']}")
        print("\nRepair steps:")
        for step in repair_plan['steps']:
            print(f"  - {step['step']}: {step['description']}")
            if 'breaches' in step and step['breaches']:
                print(f"    Breaches to address: {step['breaches']}")
    
    # Record a successful repair interaction
    repair_interaction = tracker.record_interaction(
        agent_id=agent_ids["service"],
        interaction_type=InteractionType.REPAIR,
        content_summary="Addressed service connectivity issues and improved reliability",
        quality=InteractionQuality.VERY_POSITIVE,
        principle_alignment=0.9
    )
    print(f"\nRecorded repair interaction: {repair_interaction.interaction_type.value}")
    print(f"  Quality: {repair_interaction.quality.value}")
    print(f"  Trust impact: {repair_interaction.trust_impact}")
    
    # Mark repair as successful
    repair_result = tracker.mark_repair_success(
        agent_ids["service"], 
        notes="Implemented service redundancy and improved error handling"
    )
    print(f"\nRepair result: {repair_result['success']}")
    print(f"  Old status: {repair_result['old_status']}")
    print(f"  New status: {repair_result['new_status']}")
    print(f"  Trust boost: {repair_result['trust_boost']:.2f}")
    print(f"  New trust score: {repair_result['new_trust_score']:.2f}")
    
    # Block and unblock an agent
    print_separator("10. Blocking and Unblocking")
    # Block the IoT device (e.g., if it was compromised)
    block_result = tracker.block_agent(
        agent_ids["iot"],
        reason="Device reported as potentially compromised"
    )
    print(f"Blocked IoT device: {block_result['success']}")
    print(f"  Old status: {block_result['old_status']}")
    print(f"  New status: {block_result['new_status']}")
    print(f"  Reason: {block_result['reason']}")
    
    # Try to interact with blocked agent
    try:
        tracker.record_interaction(
            agent_id=agent_ids["iot"],
            interaction_type=InteractionType.MESSAGE,
            content_summary="This should print a warning since agent is blocked",
            quality=InteractionQuality.NEUTRAL
        )
    except Exception as e:
        print(f"Error interacting with blocked agent: {e}")
    
    # Get updated status
    iot_rel = tracker.get_relationship(agent_ids["iot"])
    print(f"\nIoT device status: {iot_rel.status.name}")
    print(f"Block reason: {iot_rel.blocked_reason}")
    
    # Unblock the agent
    unblock_result = tracker.unblock_agent(
        agent_ids["iot"],
        notes="Device verified and cleared by security team"
    )
    print(f"\nUnblocked IoT device: {unblock_result['success']}")
    print(f"  New status: {unblock_result['new_status']}")
    print(f"  Previous reason: {unblock_result['previous_reason']}")
    
    # Get all relationships
    print_separator("11. Getting All Relationships")
    all_relationships = tracker.get_all_relationships()
    print(f"Total relationships: {len(all_relationships)}")
    for rel in all_relationships:
        print(f"  - {rel['agent_name']} ({rel['agent_id']}):")
        print(f"    Status: {rel['status']}, Trust: {rel['trust_level']}, Score: {rel['trust_score']:.2f}")
        print(f"    Interactions: {rel['interaction_count']}, Memories: {rel['memory_count']}")
    
    # Filter relationships by status and trust level
    trusted_rels = tracker.get_all_relationships(
        status_filter=["trusted", "close", "essential"],
        min_trust_level=TrustLevel.HIGH.value
    )
    print(f"\nTrusted relationships (high trust level or better):")
    for rel in trusted_rels:
        print(f"  - {rel['agent_name']} ({rel['agent_id']}): Trust level {rel['trust_level']}")
    
    # Save all relationship data
    print_separator("12. Data Persistence")
    tracker.save_all()
    print(f"All relationship data saved to {DATA_DIR}")
    print(f"Files in data directory:")
    for filename in os.listdir(DATA_DIR):
        file_size = os.path.getsize(os.path.join(DATA_DIR, filename))
        print(f"  - {filename} ({file_size} bytes)")
    
    # Simulate closing and reopening the tracker
    print(f"\nSimulating application restart by creating a new tracker instance...")
    del tracker
    
    # Create a new tracker instance that loads from the same directory
    new_tracker = RelationshipTracker(
        agent_id="adaptive-bridge-agent-001", 
        data_dir=DATA_DIR
    )
    print(f"New tracker loaded {len(new_tracker.relationships)} relationships")
    print(f"Loaded {len(new_tracker.interactions)} interaction records")
    
    # Verify relationship data was preserved
    human_rel = new_tracker.get_relationship(agent_ids["human"], create_if_missing=False)
    print(f"\nVerifying loaded relationship data for {human_rel.agent_name}:")
    print(f"  Trust score: {human_rel.trust_score:.2f}")
    print(f"  Trust level: {human_rel.trust_level.name}")
    print(f"  Status: {human_rel.status.name}")
    print(f"  Interactions: {human_rel.interaction_count}")
    print(f"  Memories: {len(human_rel.memories)}")
    print(f"  Communication style: {human_rel.communication_style.get('formality', 'unknown')}")
    
    print("\nExample completed! Relationship tracking and management demonstrated successfully.")

if __name__ == "__main__":
    main()
