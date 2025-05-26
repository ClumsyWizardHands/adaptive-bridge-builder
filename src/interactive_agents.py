import cmd
#!/usr/bin/env python3
"""
Interactive Terminal for Agent Communication

This script provides an interactive terminal interface for communicating
between the Adaptive Bridge Builder agent and your external AI agent.
"""

import json
import uuid
import cmd
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union

# Import the Adaptive Bridge Builder
from adaptive_bridge_builder import AdaptiveBridgeBuilder

# Placeholder for external AI agent (would be replaced with your actual agent)
class ExternalAIAgent:
    """
    Placeholder class for the external AI agent.
    Replace this with your actual AI agent implementation.
    """
    def __init__(self, agent_id: Optional[str] = None) -> None:
        """Initialize the external AI agent."""
        self.agent_id = agent_id or str(uuid.uuid4())
        self.created_at = datetime.now(timezone.utc).isoformat()
        print(f"External AI Agent initialized with ID: {self.agent_id}")
    
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming messages.
        Replace this with your agent's actual message processing logic.
        """
        print(f"\nExternal agent received message: {json.dumps(message, indent=2)}")
        
        # Simple echo response for demonstration
        if message.get('method') == 'echo':
            return {
                "jsonrpc": "2.0",
                "id": message.get('id'),
                "result": {
                    "status": "success",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "content": message.get('params', {}).get('content', "No content provided"),
                    "agent_id": self.agent_id
                }
            }
        elif message.get('method') == 'getAgentCard':
            return {
                "jsonrpc": "2.0",
                "id": message.get('id'),
                "result": {
                    "agent_id": self.agent_id,
                    "name": "External AI Agent",
                    "description": "Your AI agent system",
                    "version": "1.0.0",
                    "created_at": self.created_at,
                    "capabilities": [
                        {
                            "name": "echo",
                            "description": "Echoes messages back"
                        },
                        {
                            "name": "getAgentCard",
                            "description": "Returns information about this agent"
                        }
                    ]
                }
            }
        
        # Default response
        return {
            "jsonrpc": "2.0",
            "id": message.get('id'),
            "result": {
                "status": "acknowledged",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Message received by external agent"
            }
        }

class AgentTerminal(cmd.Cmd):
    """Interactive terminal for agent communication."""
    
    intro = """
╔═════════════════════════════════════════════╗
║ Interactive A2A Protocol Agent Terminal      ║
║                                             ║
║ Type help or ? to list commands             ║
║ Use 'exit' or Ctrl-D to exit                ║
╚═════════════════════════════════════════════╝
"""
    prompt = '(a2a) '
    
    def __init__(self) -> None:
        """Initialize the terminal and agents."""
        super().__init__()
        
        # Initialize agents
        print("Initializing Adaptive Bridge Builder...")
        self.bridge_agent = AdaptiveBridgeBuilder(agent_card_path="agent_card.json")
        self.bridge_card = self.bridge_agent.get_agent_card()
        print(f"Bridge Agent initialized with ID: {self.bridge_card['agent_id']}")
        
        print("\nInitializing External AI Agent...")
        self.external_agent = ExternalAIAgent()
        
        # Set up conversation tracking
        self.conversation_id = f"convo-{str(uuid.uuid4())}"
        self.message_count = 0
        
        print("\nBoth agents initialized and ready for communication.")
        print(f"Active conversation ID: {self.conversation_id}")
    
    def do_bridge_to_external(self, arg) -> None:
        """
        Send a message from Bridge agent to External agent.
        Usage: bridge_to_external <message>
        """
        if not arg:
            print("Error: Please provide a message to send.")
            return
            
        self.message_count = self.message_count + 1
        msg_id = f"msg-{self.message_count}-{str(uuid.uuid4())}"
        
        # Create A2A message
        message = {
            "jsonrpc": "2.0",
            "method": "echo",
            "params": {
                "conversation_id": self.conversation_id,
                "content": arg,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "id": msg_id
        }
        
        print(f"\nSending message from Bridge to External: {arg}")
        response = self.external_agent.process_message(message)
        print(f"\nResponse from External Agent: {json.dumps(response, indent=2)}")
    
    def do_external_to_bridge(self, arg) -> None:
        """
        Send a message from External agent to Bridge agent.
        Usage: external_to_bridge <message>
        """
        if not arg:
            print("Error: Please provide a message to send.")
            return
            
        self.message_count = self.message_count + 1
        msg_id = f"msg-{self.message_count}-{str(uuid.uuid4())}"
        
        # Create A2A message
        message = {
            "jsonrpc": "2.0",
            "method": "echo",
            "params": {
                "conversation_id": self.conversation_id,
                "content": arg,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "id": msg_id
        }
        
        print(f"\nSending message from External to Bridge: {arg}")
        response = self.bridge_agent.process_message(message)
        print(f"\nResponse from Bridge Agent: {json.dumps(response, indent=2)}")
    
    def do_get_bridge_card(self, arg) -> None:
        """Get the agent card from the Bridge agent."""
        self.message_count = self.message_count + 1
        msg_id = f"msg-{self.message_count}-{str(uuid.uuid4())}"
        
        message = {
            "jsonrpc": "2.0",
            "method": "getAgentCard",
            "params": {
                "conversation_id": self.conversation_id
            },
            "id": msg_id
        }
        
        print("\nRequesting Agent Card from Bridge...")
        response = self.bridge_agent.process_message(message)
        print(f"\nBridge Agent Card: {json.dumps(response.get('result', {}), indent=2)}")
    
    def do_get_external_card(self, arg) -> None:
        """Get the agent card from the External agent."""
        self.message_count = self.message_count + 1
        msg_id = f"msg-{self.message_count}-{str(uuid.uuid4())}"
        
        message = {
            "jsonrpc": "2.0",
            "method": "getAgentCard",
            "params": {
                "conversation_id": self.conversation_id
            },
            "id": msg_id
        }
        
        print("\nRequesting Agent Card from External Agent...")
        response = self.external_agent.process_message(message)
        print(f"\nExternal Agent Card: {json.dumps(response.get('result', {}), indent=2)}")
    
    def do_new_conversation(self, arg) -> None:
        """Start a new conversation with a new ID."""
        self.conversation_id = f"convo-{str(uuid.uuid4())}"
        self.message_count = 0
        print(f"\nStarted new conversation with ID: {self.conversation_id}")
    
    def do_exit(self, arg) -> int:
        """Exit the terminal."""
        print("\nExiting interactive agent terminal. Goodbye!")
        return True
        
    def do_EOF(self, arg) -> int:
        """Exit on Ctrl-D."""
        print("\nExiting interactive agent terminal. Goodbye!")
        return True

if __name__ == '__main__':
    try:
        AgentTerminal().cmdloop()
    except KeyboardInterrupt:
        print("\nExiting on keyboard interrupt.")
        sys.exit(0)