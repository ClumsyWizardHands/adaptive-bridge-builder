import cmd
#!/usr/bin/env python3
"""
Interactive Bridge Agent Terminal

A simplified interactive terminal interface for communicating with the
Adaptive Bridge Builder agent, with easy-to-use commands.
"""

import json
import uuid
import cmd
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union

# Import the Adaptive Bridge Builder
from adaptive_bridge_builder import AdaptiveBridgeBuilder

class BridgeTerminal(cmd.Cmd):
    """Interactive terminal for the Adaptive Bridge Builder agent."""
    
    intro = """
╔══════════════════════════════════════════════════════╗
║ Adaptive Bridge Builder Interactive Terminal         ║
║                                                      ║
║ Type:                                                ║
║   'card'    - View the Bridge agent's card           ║
║   'send'    - Send a message to the Bridge agent     ║
║   'route'   - Route a message through the Bridge     ║
║   'translate' - Translate between protocols          ║
║   'help'    - Show available commands                ║
║   'exit'    - Exit the terminal                      ║
╚══════════════════════════════════════════════════════╝
"""
    prompt = '(bridge) '
    
    def __init__(self) -> None:
        """Initialize the terminal and bridge agent."""
        super().__init__()
        
        # Initialize bridge agent
        print("Initializing Adaptive Bridge Builder...")
        self.bridge_agent = AdaptiveBridgeBuilder(agent_card_path="agent_card.json")
        self.conversation_id = f"convo-{str(uuid.uuid4())}"
        self.message_count = 0
        
        # Get agent card for initial verification
        self.bridge_card = self.bridge_agent.get_agent_card()
        print(f"Bridge Agent initialized with ID: {self.bridge_card['agent_id']}")
        print(f"Active conversation ID: {self.conversation_id}\n")
    
    def do_card(self, arg) -> None:
        """Display the bridge agent's card."""
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
        
        print("\nRetrieving Bridge Agent card...")
        response = self.bridge_agent.process_message(message)
        print("\nBridge Agent Card:")
        print(json.dumps(response.get('result', {}), indent=2))
    
    def do_send(self, arg) -> None:
        """Send a message to the bridge agent."""
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
        
        print(f"\nSending message to Bridge: {arg}")
        response = self.bridge_agent.process_message(message)
        print("\nResponse from Bridge Agent:")
        print(json.dumps(response, indent=2))
    
    def do_route(self, arg) -> None:
        """Route a message through the bridge."""
        if not arg:
            print("Error: Please provide a message and destination in format: <destination> <message>")
            return
            
        try:
            parts = arg.split(' ', 1)
            if len(parts) < 2:
                print("Error: Please provide both destination and message.")
                return
                
            destination = parts[0]
            content = parts[1]
                
            self.message_count = self.message_count + 1
            msg_id = f"msg-{self.message_count}-{str(uuid.uuid4())}"
            
            # Create routing message
            message = {
                "jsonrpc": "2.0",
                "method": "route",
                "params": {
                    "conversation_id": self.conversation_id,
                    "destination": destination,
                    "content": content,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                "id": msg_id
            }
            
            print(f"\nRouting message to {destination}: {content}")
            response = self.bridge_agent.process_message(message)
            print("\nRouting response from Bridge Agent:")
            print(json.dumps(response, indent=2))
        except Exception as e:
            print(f"Error: {e}")
    
    def do_translate(self, arg) -> None:
        """Translate a message between protocols."""
        if not arg:
            print("Error: Please provide source protocol, target protocol, and message in format: <source> <target> <message>")
            return
            
        try:
            parts = arg.split(' ', 2)
            if len(parts) < 3:
                print("Error: Please provide source protocol, target protocol, and message.")
                return
                
            source_protocol = parts[0]
            target_protocol = parts[1]
            content = parts[2]
                
            self.message_count = self.message_count + 1
            msg_id = f"msg-{self.message_count}-{str(uuid.uuid4())}"
            
            # Create translation message
            message = {
                "jsonrpc": "2.0",
                "method": "translateProtocol",
                "params": {
                    "conversation_id": self.conversation_id,
                    "source_protocol": source_protocol,
                    "target_protocol": target_protocol,
                    "content": content,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                "id": msg_id
            }
            
            print(f"\nTranslating message from {source_protocol} to {target_protocol}: {content}")
            response = self.bridge_agent.process_message(message)
            print("\nTranslation response from Bridge Agent:")
            print(json.dumps(response, indent=2))
        except Exception as e:
            print(f"Error: {e}")
    
    def do_new(self, arg) -> None:
        """Start a new conversation."""
        self.conversation_id = f"convo-{str(uuid.uuid4())}"
        self.message_count = 0
        print(f"\nStarted new conversation with ID: {self.conversation_id}")
    
    def do_exit(self, arg) -> int:
        """Exit the terminal."""
        print("\nExiting Adaptive Bridge Builder terminal. Goodbye!")
        return True
        
    def do_EOF(self, arg) -> int:
        """Exit on Ctrl-D."""
        print("\nExiting Adaptive Bridge Builder terminal. Goodbye!")
        return True

if __name__ == '__main__':
    try:
        BridgeTerminal().cmdloop()
    except KeyboardInterrupt:
        print("\nExiting on keyboard interrupt.")
        sys.exit(0)