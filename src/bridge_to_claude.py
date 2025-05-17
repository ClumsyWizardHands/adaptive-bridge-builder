#!/usr/bin/env python3
"""
Bridge to Claude Interactive Terminal

A simple interactive terminal that connects the Adaptive Bridge Builder
with Anthropic's Claude AI using the A2A Protocol.
"""

import os
import json
import uuid
import cmd
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Import the Adaptive Bridge Builder and Claude Agent
from adaptive_bridge_builder import AdaptiveBridgeBuilder
from anthropic_agent import AnthropicAgent

class BridgeToClaudeTerminal(cmd.Cmd):
    """Interactive terminal connecting Bridge and Claude."""
    
    intro = """
╔══════════════════════════════════════════════════════╗
║ Adaptive Bridge Builder to Claude Terminal           ║
║                                                      ║
║ Type:                                                ║
║   'bridge_card'   - View the Bridge agent's card     ║
║   'claude_card'   - View Claude's agent card         ║
║   'ask <message>' - Ask Claude a question            ║
║   'help'          - Show available commands          ║
║   'exit'          - Exit the terminal                ║
╚══════════════════════════════════════════════════════╝
"""
    prompt = '(bridge-claude) '
    
    def __init__(self, anthropic_api_key: str):
        """Initialize the terminal and both agents."""
        super().__init__()
        
        # Initialize bridge agent
        print("Initializing Adaptive Bridge Builder...")
        self.bridge_agent = AdaptiveBridgeBuilder(agent_card_path="agent_card.json")
        
        # Initialize Claude agent
        print("Initializing Claude agent...")
        self.claude_agent = AnthropicAgent(api_key=anthropic_api_key)
        
        # Setup conversation
        self.conversation_id = f"convo-{str(uuid.uuid4())}"
        self.message_count = 0
        
        # Get agent cards for initial verification
        self.bridge_card = self.bridge_agent.get_agent_card()
        print(f"Bridge Agent initialized with ID: {self.bridge_card['agent_id']}")
        print(f"Claude Agent initialized with ID: {self.claude_agent.agent_id}")
        print(f"Active conversation ID: {self.conversation_id}\n")
    
    def do_bridge_card(self, arg):
        """Display the bridge agent's card."""
        self.message_count += 1
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
    
    def do_claude_card(self, arg):
        """Display Claude's agent card."""
        self.message_count += 1
        msg_id = f"msg-{self.message_count}-{str(uuid.uuid4())}"
        
        message = {
            "jsonrpc": "2.0",
            "method": "getAgentCard",
            "params": {
                "conversation_id": self.conversation_id
            },
            "id": msg_id
        }
        
        print("\nRetrieving Claude Agent card...")
        response = self.claude_agent.process_message(message)
        print("\nClaude Agent Card:")
        print(json.dumps(response.get('result', {}), indent=2))
    
    def do_ask(self, arg):
        """Ask Claude a question through the Bridge."""
        if not arg:
            print("Error: Please provide a message to send to Claude.")
            return
            
        self.message_count += 1
        msg_id = f"msg-{self.message_count}-{str(uuid.uuid4())}"
        
        # Step 1: Create message to route through Bridge
        bridge_message = {
            "jsonrpc": "2.0",
            "method": "route",
            "params": {
                "conversation_id": self.conversation_id,
                "destination": self.claude_agent.agent_id,
                "content": arg,
                "timestamp": datetime.utcnow().isoformat()
            },
            "id": msg_id
        }
        
        print(f"\n1. Sending message to Bridge for routing: \"{arg}\"")
        bridge_response = self.bridge_agent.process_message(bridge_message)
        print("\n2. Bridge accepted message for routing:")
        print(json.dumps(bridge_response, indent=2))
        
        # Step 2: Create message for Claude to generate content
        self.message_count += 1
        claude_msg_id = f"msg-{self.message_count}-{str(uuid.uuid4())}"
        
        claude_message = {
            "jsonrpc": "2.0",
            "method": "generate",
            "params": {
                "conversation_id": self.conversation_id,
                "content": arg,
                "timestamp": datetime.utcnow().isoformat()
            },
            "id": claude_msg_id
        }
        
        print("\n3. Sending message to Claude for response generation...")
        claude_response = self.claude_agent.process_message(claude_message)
        
        # Step 3: Format and display Claude's response
        if "error" in claude_response:
            print("\nError from Claude:")
            print(json.dumps(claude_response.get("error", {}), indent=2))
        else:
            content = claude_response.get("result", {}).get("content", "No response")
            print("\n4. Response from Claude:")
            print("=" * 80)
            print(content)
            print("=" * 80)
    
    def do_exit(self, arg):
        """Exit the terminal."""
        print("\nExiting Bridge to Claude terminal. Goodbye!")
        return True
        
    def do_EOF(self, arg):
        """Exit on Ctrl-D."""
        print("\nExiting Bridge to Claude terminal. Goodbye!")
        return True

def main():
    """Main function to run the Bridge to Claude terminal."""
    # Get Anthropic API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    # Try to read from config file if environment variable is not set
    if not api_key:
        config_file = Path(".claude_config.json")
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
                    api_key = config.get("api_key")
                    if api_key:
                        print("Using API key from config file.")
            except Exception as e:
                print(f"Error reading config file: {e}")
    
    # If still no API key, ask the user
    if not api_key:
        print("Anthropic API key not found in environment or config file.")
        print("Please enter your Anthropic API key:")
        print("(Or run 'python src/setup_claude_key.py' first to store it securely)")
        api_key = input("API Key: ").strip()
        
        if not api_key:
            print("Error: API key is required.")
            sys.exit(1)
    
    try:
        # Install anthropic if not already installed
        try:
            import anthropic
        except ImportError:
            print("Installing anthropic package...")
            import subprocess
            subprocess.check_call(["pip", "install", "anthropic"])
            import anthropic
        
        # Start the terminal
        BridgeToClaudeTerminal(api_key).cmdloop()
    except KeyboardInterrupt:
        print("\nExiting on keyboard interrupt.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
