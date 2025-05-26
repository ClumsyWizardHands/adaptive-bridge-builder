#!/usr/bin/env python3
"""
Demonstration Script for the Adaptive Bridge Builder Agent

This script demonstrates the core functionality of the Adaptive Bridge Builder agent
by simulating communication between the bridge agent and an external agent.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Import the Adaptive Bridge Builder
from adaptive_bridge_builder import AdaptiveBridgeBuilder

def print_divider(title=None) -> None:
    """Print a divider line with an optional title."""
    width = 80
    if title:
        print(f"\n{'=' * 10} {title} {'=' * (width - 12 - len(title))}")
    else:
        print("\n" + "=" * width)

def print_json(data) -> None:
    """Pretty print JSON data."""
    print(json.dumps(data, indent=2))

def main() -> None:
    """Main demonstration function."""
    print_divider("ADAPTIVE BRIDGE BUILDER DEMO")
    print("Initializing agents and demonstrating communication...")
    
    # Initialize the Bridge Builder agent
    print("\nInitializing Adaptive Bridge Builder agent...")
    bridge_agent = AdaptiveBridgeBuilder(agent_card_path="agent_card.json")
    
    # Get the agent card
    print("\nRetrieving Bridge Agent card...")
    bridge_card = bridge_agent.get_agent_card()
    print_divider("BRIDGE AGENT CARD")
    print_json(bridge_card)
    
    # Create a simulated external agent message
    conversation_id = str(uuid.uuid4())
    message_id = f"msg-{str(uuid.uuid4())}"
    
    print_divider("SIMULATING EXTERNAL AGENT -> BRIDGE")
    print("Creating message from External Agent to Bridge Agent...")
    
    # Create a message asking about capabilities
    external_message = {
        "jsonrpc": "2.0",
        "method": "getAgentCard",
        "params": {
            "conversation_id": conversation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        "id": message_id
    }
    
    print("\nSending message to Bridge Agent:")
    print_json(external_message)
    
    # Process the message
    bridge_response = bridge_agent.process_message(external_message)
    
    print("\nResponse from Bridge Agent:")
    print_json(bridge_response)
    
    # Simulate an echo message
    print_divider("SIMULATING BRIDGE -> EXTERNAL ECHO")
    
    # Create an echo message
    echo_message_id = f"msg-{str(uuid.uuid4())}"
    echo_message = {
        "jsonrpc": "2.0",
        "method": "echo",
        "params": {
            "conversation_id": conversation_id,
            "content": "Hello from the Bridge Agent! Can you help me route this message?",
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        "id": echo_message_id
    }
    
    print("\nSending echo message to Bridge Agent (simulating processing):")
    print_json(echo_message)
    
    # Process the echo message
    echo_response = bridge_agent.process_message(echo_message)
    
    print("\nResponse from Bridge Agent (to be sent to External Agent):")
    print_json(echo_response)
    
    # Simulate a routing request
    print_divider("SIMULATING ROUTING REQUEST")
    
    # Create a routing message
    route_message_id = f"msg-{str(uuid.uuid4())}"
    route_message = {
        "jsonrpc": "2.0",
        "method": "route",
        "params": {
            "conversation_id": conversation_id,
            "destination": "target-agent-id-123",
            "content": "This message needs to be routed to another agent",
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        "id": route_message_id
    }
    
    print("\nSending routing message to Bridge Agent:")
    print_json(route_message)
    
    # Process the routing message
    route_response = bridge_agent.process_message(route_message)
    
    print("\nRouting response from Bridge Agent:")
    print_json(route_response)
    
    # Simulate a protocol translation request
    print_divider("SIMULATING PROTOCOL TRANSLATION")
    
    # Create a translation message
    translate_message_id = f"msg-{str(uuid.uuid4())}"
    translate_message = {
        "jsonrpc": "2.0",
        "method": "translateProtocol",
        "params": {
            "conversation_id": conversation_id,
            "source_protocol": "a2a",
            "target_protocol": "openai-assistant",
            "content": "This message needs to be translated to another protocol format",
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        "id": translate_message_id
    }
    
    print("\nSending translation message to Bridge Agent:")
    print_json(translate_message)
    
    # Process the translation message
    translate_response = bridge_agent.process_message(translate_message)
    
    print("\nTranslation response from Bridge Agent:")
    print_json(translate_response)
    
    # Summary
    print_divider("DEMONSTRATION SUMMARY")
    print("The Adaptive Bridge Builder agent has successfully demonstrated:")
    print("  1. Agent Card retrieval")
    print("  2. Echo message processing")
    print("  3. Message routing capability")
    print("  4. Protocol translation capability")
    print("\nEach of these features is essential for facilitating communication")
    print("between diverse agent systems while maintaining principles of")
    print("fairness, harmony, and adaptability.")
    
    print_divider("END OF DEMONSTRATION")

if __name__ == "__main__":
    main()