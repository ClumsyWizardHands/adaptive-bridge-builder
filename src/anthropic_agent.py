#!/usr/bin/env python3
"""
Anthropic Claude Agent Integration

This script creates an interface between the Adaptive Bridge Builder and
Anthropic's Claude API using the A2A Protocol.
"""

import os
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Import Anthropic library
try:
    import anthropic
except ImportError:
    print("Anthropic library not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "anthropic"])
    import anthropic

class AnthropicAgent:
    """
    An agent that connects to Anthropic's Claude API and
    interfaces with the Adaptive Bridge Builder.
    """
    def __init__(self, api_key: Optional[str] = None, agent_id: Optional[str] = None) -> None:
        """Initialize the Anthropic agent with API key."""
        self.agent_id = agent_id or f"claude-agent-{str(uuid.uuid4())[:8]}"
        
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Either pass it to the constructor or set ANTHROPIC_API_KEY environment variable.")
        
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-3-haiku-20240307"  # Latest Claude model with better compatibility
        
        self.created_at = datetime.now(timezone.utc).isoformat()
        print(f"Claude Agent initialized with ID: {self.agent_id}")
        print(f"Using model: {self.model}")
    
    def get_agent_card(self) -> Dict[str, Any]:
        """Return the agent card with capabilities."""
        return {
            "agent_id": self.agent_id,
            "name": "Claude by Anthropic",
            "description": "Advanced AI assistant with strong natural language capabilities",
            "version": "3.0.0",
            "created_at": self.created_at,
            "principles": [
                {
                    "name": "Constitutional AI",
                    "description": "Built with safety and helpfulness in mind"
                }
            ],
            "capabilities": [
                {
                    "name": "conversation",
                    "description": "Engage in natural language conversations"
                },
                {
                    "name": "content_generation",
                    "description": "Generate creative and informative content"
                },
                {
                    "name": "question_answering",
                    "description": "Answer questions based on knowledge and context"
                }
            ],
            "communication": {
                "protocols": [
                    "a2a",
                    "json-rpc-2.0"
                ]
            }
        }
    
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming messages from the Bridge agent.
        Route to appropriate method based on the message method.
        """
        print(f"Claude agent received message: {json.dumps(message, indent=2)}")
        
        method = message.get('method', '')
        msg_id = message.get('id', f"response-{str(uuid.uuid4())}")
        
        # Handle different methods
        if method == 'getAgentCard':
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": self.get_agent_card()
            }
        
        elif method == 'echo':
            # Simple echo response
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "conversation_id": message.get('params', {}).get('conversation_id', ''),
                    "content": message.get('params', {}).get('content', ''),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
        
        elif method == 'generate':
            # Call Claude to generate a response
            return self._generate_content(message, msg_id)
        
        else:
            # Default response for unsupported methods
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": "Method not found",
                    "data": f"The method {method} is not supported by Claude agent"
                }
            }
    
    def _generate_content(self, message: Dict[str, Any], msg_id: str) -> Dict[str, Any]:
        """Generate content using Claude API."""
        try:
            # Extract message content
            content = message.get('params', {}).get('content', '')
            conversation_id = message.get('params', {}).get('conversation_id', '')
            
            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": content}
                ]
            )
            
            # Extract Claude's response
            claude_response = response.content[0].text
            
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "conversation_id": conversation_id,
                    "content": claude_response,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "agent_id": self.agent_id
                }
            }
            
        except Exception as e:
            # Handle any API errors
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32603,
                    "message": "Claude API error",
                    "data": str(e)
                }
            }

# Simple test if run directly
if __name__ == "__main__":
    # Check if API key is set
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("WARNING: ANTHROPIC_API_KEY environment variable not set.")
        print("Please set it or provide the API key when creating the agent.")
        print("For testing purposes, you can enter the API key now:")
        api_key = input("API Key: ").strip()
    
    # Initialize agent
    try:
        claude_agent = AnthropicAgent(api_key=api_key)
        
        # Test basic functionality
        print("\nTesting Claude agent...")
        test_message = {
            "jsonrpc": "2.0",
            "method": "generate",
            "params": {
                "conversation_id": f"test-{str(uuid.uuid4())}",
                "content": "Tell me a short story about a bridge that connects two different worlds."
            },
            "id": "test-1"
        }
        
        response = claude_agent.process_message(test_message)
        print("\nResponse from Claude:")
        print(json.dumps(response, indent=2))
        
    except Exception as e:
        print(f"Error during testing: {e}")
