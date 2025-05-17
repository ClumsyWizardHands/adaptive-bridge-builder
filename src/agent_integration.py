#!/usr/bin/env python3
"""
Agent Integration Script

This script demonstrates how to initialize and establish communication between
the Adaptive Bridge Builder agent and an external AI agent using A2A Protocol.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

# Import our Adaptive Bridge Builder (using relative import from current directory)
from adaptive_bridge_builder import AdaptiveBridgeBuilder

# Placeholder for external AI agent (replace with your actual agent import)
class ExternalAIAgent:
    """
    Placeholder class for the external AI agent.
    Replace this with your actual AI agent implementation.
    """
    def __init__(self, agent_id: Optional[str] = None):
        """Initialize the external AI agent."""
        self.agent_id = agent_id or str(uuid.uuid4())
        self.created_at = datetime.utcnow().isoformat()
        print(f"External AI Agent initialized with ID: {self.agent_id}")
    
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming messages.
        Replace this with your agent's actual message processing logic.
        """
        print(f"External agent received message: {json.dumps(message, indent=2)}")
        
        # Simple echo response for demonstration
        if message.get('method') == 'echo':
            return {
                "jsonrpc": "2.0",
                "id": message.get('id'),
                "result": {
                    "status": "success",
                    "timestamp": datetime.utcnow().isoformat(),
                    "content": message.get('params', {}).get('content', "No content provided"),
                    "agent_id": self.agent_id
                }
            }
        
        # Default response
        return {
            "jsonrpc": "2.0",
            "id": message.get('id'),
            "result": {
                "status": "acknowledged",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Message received by external agent"
            }
        }

# Communication functions

def send_a2a_message(
    sender, 
    receiver, 
    content: Any, 
    method: str = "echo", 
    conversation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Send an A2A Protocol message from sender to receiver.
    
    Args:
        sender: Sending agent
        receiver: Receiving agent
        content: Message content
        method: JSON-RPC method to call
        conversation_id: Optional conversation ID for tracking
        
    Returns:
        Receiver's response
    """
    # Generate message ID and conversation ID if needed
    msg_id = f"msg-{str(uuid.uuid4())}"
    if conversation_id is None:
        conversation_id = f"convo-{str(uuid.uuid4())}"
    
    # Create A2A message
    a2a_message = {
        "jsonrpc": "2.0",
        "method": method,
        "params": {
            "conversation_id": conversation_id,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        },
        "id": msg_id
    }
    
    # Send message to receiver
    response = receiver.process_message(a2a_message)
    return response

def main():
    # Step 1: Initialize the Adaptive Bridge Builder with explicit path to agent_card.json
    bridge_agent = AdaptiveBridgeBuilder(agent_card_path="agent_card.json")
    agent_card = bridge_agent.get_agent_card()
    print(f"Bridge Agent initialized with ID: {agent_card['agent_id']}")
    print(f"Bridge Agent name: {agent_card['name']}")
    print(f"Bridge Agent capabilities: {[cap['name'] for cap in agent_card['capabilities']]}")
    
    # Step 2: Initialize the External AI Agent
    # Replace ExternalAIAgent with your actual agent class
    external_agent = ExternalAIAgent()
    
    # Step 3: Demonstrate basic communication
    print("\n--- Testing Bridge Agent to External Agent ---")
    message_content = "Hello from Bridge Agent!"
    response = send_a2a_message(
        bridge_agent, 
        external_agent, 
        message_content
    )
    print(f"Response from External Agent: {json.dumps(response, indent=2)}")
    
    print("\n--- Testing External Agent to Bridge Agent ---")
    message_content = "Hello from External Agent!"
    response = send_a2a_message(
        external_agent, 
        bridge_agent, 
        message_content
    )
    print(f"Response from Bridge Agent: {json.dumps(response, indent=2)}")
    
    # Step 4: Get Agent Card from Bridge
    print("\n--- Requesting Agent Card from Bridge ---")
    response = send_a2a_message(
        external_agent,
        bridge_agent,
        {},
        method="getAgentCard"
    )
    print(f"Agent Card Response: {json.dumps(response, indent=2)}")
    
    print("\nIntegration demonstration complete.")

if __name__ == "__main__":
    main()
