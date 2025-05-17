#!/usr/bin/env python3
"""
Example Usage of the Adaptive Bridge Builder Agent

This script demonstrates how to instantiate and interact with the
Adaptive Bridge Builder agent using the A2A Protocol and JSON-RPC 2.0.
"""

import json
import uuid
from adaptive_bridge_builder import AdaptiveBridgeBuilder

def print_response(title, response):
    """Print a formatted JSON-RPC response."""
    print(f"\n{'-' * 40}")
    print(f"{title}:")
    print(f"{'-' * 40}")
    print(json.dumps(response, indent=2))
    print(f"{'-' * 40}\n")

def main():
    """Main demonstration function."""
    print("Initializing Adaptive Bridge Builder Agent...")
    
    # Create a new instance of the Adaptive Bridge Builder
    agent = AdaptiveBridgeBuilder()
    
    # Generate a unique conversation ID for this session
    conversation_id = str(uuid.uuid4())
    print(f"Conversation ID: {conversation_id}")
    
    # Example 1: Get Agent Card
    print("\nExample 1: Retrieving Agent Card")
    agent_card_request = {
        "jsonrpc": "2.0",
        "method": "getAgentCard",
        "params": {
            "conversation_id": conversation_id
        },
        "id": "request-1"
    }
    response = agent.process_message(agent_card_request)
    print_response("Agent Card Response", response)
    
    # Example 2: Echo Test
    print("\nExample 2: Echo Test")
    echo_request = {
        "jsonrpc": "2.0",
        "method": "echo",
        "params": {
            "conversation_id": conversation_id,
            "message": "Hello, Adaptive Bridge Builder!",
            "timestamp": "2025-05-16T14:45:30Z"
        },
        "id": "request-2"
    }
    response = agent.process_message(echo_request)
    print_response("Echo Response", response)
    
    # Example 3: Route Message
    print("\nExample 3: Route Message")
    route_request = {
        "jsonrpc": "2.0",
        "method": "route",
        "params": {
            "conversation_id": conversation_id,
            "destination": "target-agent-001",
            "message": {
                "jsonrpc": "2.0",
                "method": "processData",
                "params": {
                    "data": {
                        "type": "sensor_reading",
                        "value": 23.5,
                        "unit": "celsius"
                    }
                },
                "id": "nested-request-1"
            }
        },
        "id": "request-3"
    }
    response = agent.process_message(route_request)
    print_response("Route Response", response)
    
    # Example 4: Protocol Translation
    print("\nExample 4: Protocol Translation")
    translate_request = {
        "jsonrpc": "2.0",
        "method": "translateProtocol",
        "params": {
            "conversation_id": conversation_id,
            "source_protocol": "json-rpc-2.0",
            "target_protocol": "a2a",
            "message": {
                "jsonrpc": "2.0",
                "method": "getData",
                "params": {"id": "dataset-123"},
                "id": "nested-request-2"
            }
        },
        "id": "request-4"
    }
    response = agent.process_message(translate_request)
    print_response("Protocol Translation Response", response)
    
    # Example 5: Invalid Method
    print("\nExample 5: Invalid Method")
    invalid_request = {
        "jsonrpc": "2.0",
        "method": "nonExistentMethod",
        "params": {
            "conversation_id": conversation_id
        },
        "id": "request-5"
    }
    response = agent.process_message(invalid_request)
    print_response("Invalid Method Response", response)
    
    # Example 6: Invalid JSON-RPC Format
    print("\nExample 6: Invalid JSON-RPC Format")
    invalid_jsonrpc = {
        "method": "getAgentCard",
        "params": {},
        "id": "request-6"
        # Missing jsonrpc field
    }
    response = agent.process_message(invalid_jsonrpc)
    print_response("Invalid JSON-RPC Response", response)
    
    # Summary
    print("\nSummary of Conversation")
    print(f"Total messages processed: {agent.message_counter}")
    print(f"Active conversations: {len(agent.active_conversations)}")
    if conversation_id in agent.active_conversations:
        conv_data = agent.active_conversations[conversation_id]
        print(f"Current conversation messages: {conv_data['message_count']}")
        print(f"Conversation started at: {conv_data['started_at']}")
        print(f"Last activity: {conv_data['last_activity']}")

if __name__ == "__main__":
    main()
