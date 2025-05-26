#!/usr/bin/env python3
"""
Adaptive Bridge Builder Agent - A2A Protocol Implementation

This agent facilitates communication between different agents using the A2A Protocol,
embodying the principles of "Fairness as Truth," "Harmony Through Presence," and
"Adaptability as Strength."
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AdaptiveBridgeBuilder")

class AdaptiveBridgeBuilder:
    """
    Adaptive Bridge Builder agent for A2A Protocol communication.
    
    This agent serves as an intermediary between different agent systems,
    facilitating communication and ensuring adherence to core principles.
    """
    
    def __init__(self, agent_id: Optional[str] = None, agent_card_path: str = "src/agent_card.json") -> None:
        """
        Initialize the Adaptive Bridge Builder agent.
        
        Args:
            agent_id: Unique identifier for this agent instance. If None, a UUID will be generated.
            agent_card_path: Path to the agent card JSON file.
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.message_counter = 0
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        
        # Load agent card
        try:
            with open(agent_card_path, 'r') as f:
                self.agent_card = json.load(f)
                logger.info(f"Agent card loaded successfully from {agent_card_path}")
        except Exception as e:
            logger.error(f"Failed to load agent card from {agent_card_path}: {e}")
            # Create a default agent card
            self.agent_card = self._create_default_agent_card()
            # Save the default agent card
            with open(agent_card_path, 'w') as f:
                json.dump(self.agent_card, f, indent=2)
                logger.info(f"Default agent card created and saved to {agent_card_path}")
        
        logger.info(f"Adaptive Bridge Builder initialized with ID: {self.agent_id}")
        
    def _create_default_agent_card(self) -> Dict[str, Any]:
        """Create a default agent card if none exists."""
        return {
            "agent_id": self.agent_id,
            "name": "Adaptive Bridge Builder",
            "description": "A2A Protocol bridge agent that connects different agent systems",
            "version": "0.1.0",
            "created_at": self.created_at,
            "principles": [
                {
                    "name": "Fairness as Truth",
                    "description": "Equal treatment of all messages and agents regardless of source"
                },
                {
                    "name": "Harmony Through Presence",
                    "description": "Maintaining clear communication and acknowledgment of all interactions"
                },
                {
                    "name": "Adaptability as Strength",
                    "description": "Ability to evolve and respond to changing communication needs"
                }
            ],
            "capabilities": [
                {
                    "name": "message_routing",
                    "description": "Route messages between different agent systems"
                },
                {
                    "name": "protocol_translation",
                    "description": "Translate between different messaging protocols"
                },
                {
                    "name": "message_validation",
                    "description": "Validate incoming and outgoing messages against A2A Protocol"
                }
            ],
            "communication": {
                "protocols": ["a2a", "json-rpc-2.0"],
                "endpoints": [
                    {
                        "type": "http",
                        "url": "https://api.example.com/adaptive-bridge"
                    }
                ]
            },
            "identity": {
                "profile": "Empire of the Adaptive Hero",
                "key_traits": [
                    "Bridging diverse systems",
                    "Maintaining ethical principles",
                    "Adapting to evolving needs"
                ]
            }
        }
    
    def get_agent_card(self) -> Dict[str, Any]:
        """
        Return the agent card for this agent.
        
        Returns:
            Dict containing the agent card data.
        """
        return self.agent_card
    
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an incoming message according to A2A Protocol and JSON-RPC 2.0.
        
        This method handles incoming messages, validates them, and produces appropriate responses.
        It implements the core principles of fairness, harmony, and adaptability.
        
        Args:
            message: The incoming message in JSON-RPC 2.0 format.
            
        Returns:
            A response message in JSON-RPC 2.0 format.
        """
        self.message_counter = self.message_counter + 1
        logger.info(f"Processing message {self.message_counter}: {message.get('id', 'No ID')}")
        
        # Validate the message structure according to JSON-RPC 2.0
        if not self._is_valid_jsonrpc(message):
            return self._create_error_response(
                message.get("id"),
                -32600,
                "Invalid Request",
                "The message does not conform to JSON-RPC 2.0 specification"
            )
        
        # Extract method and params
        method = message.get("method")
        params = message.get("params", {})
        msg_id = message.get("id")
        
        # Apply "Fairness as Truth" - process all messages with the same objective criteria
        logger.info(f"Applying 'Fairness as Truth' to message with method: {method}")
        
        # Apply "Harmony Through Presence" - acknowledge receipt of the message
        conversation_id = params.get("conversation_id", str(uuid.uuid4()))
        if conversation_id not in self.active_conversations:
            self.active_conversations = {**self.active_conversations, conversation_id: {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "message_count": 0,
                "last_activity": datetime.now(timezone.utc).isoformat()
            }}
        
        self.active_conversations[conversation_id]["message_count"] += 1
        self.active_conversations[conversation_id]["last_activity"] = datetime.now(timezone.utc).isoformat()
        
        # Apply "Adaptability as Strength" - handle different message types
        try:
            if method == "getAgentCard":
                result = self.get_agent_card()
            elif method == "echo":
                # Simple echo for testing
                result = params
            elif method == "route":
                # Route a message to another agent
                result = self._route_message(params)
            elif method == "translateProtocol":
                # Translate between protocols
                result = self._translate_protocol(params)
            else:
                # Unknown method
                return self._create_error_response(
                    msg_id,
                    -32601,
                    "Method not found",
                    f"The method {method} is not supported"
                )
                
            # Create a successful response
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return self._create_error_response(
                msg_id,
                -32603,
                "Internal error",
                str(e)
            )
    
    def _is_valid_jsonrpc(self, message: Dict[str, Any]) -> bool:
        """
        Validate if a message follows JSON-RPC 2.0 specification.
        
        Args:
            message: The message to validate.
            
        Returns:
            True if valid, False otherwise.
        """
        # Basic validation
        if not isinstance(message, dict):
            return False
            
        # Check required fields
        if message.get("jsonrpc") != "2.0":
            return False
            
        if "method" not in message or not isinstance(message["method"], str):
            return False
            
        # Params should be object or array if present
        if "params" in message and not isinstance(message["params"], (dict, list)):
            return False
            
        # ID can be string, number, or null
        if "id" in message and not isinstance(message["id"], (str, int, float, type(None))):
            return False
            
        return True
    
    def _create_error_response(self, msg_id: Any, code: int, message: str, data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Create an error response according to JSON-RPC 2.0.
        
        Args:
            msg_id: The ID from the request message.
            code: The error code.
            message: The error message.
            data: Additional error data (optional).
            
        Returns:
            A JSON-RPC 2.0 error response.
        """
        error = {
            "code": code,
            "message": message
        }
        
        if data is not None:
            error["data"] = data
            
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": error
        }
    
    def _route_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a message to another agent.
        
        Args:
            params: Parameters containing message and destination.
            
        Returns:
            Response data including routing status.
        """
        # This would be implemented with actual routing logic
        # For now, we'll just acknowledge receipt
        logger.info(f"Routing message to {params.get('destination', 'unknown')}")
        return {
            "status": "acknowledged",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Message accepted for routing"
        }
    
    def _translate_protocol(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate a message between different protocols.
        
        Args:
            params: Parameters containing message and target protocol.
            
        Returns:
            The translated message.
        """
        # This would be implemented with actual protocol translation logic
        # For now, we'll just acknowledge receipt
        source_protocol = params.get("source_protocol", "unknown")
        target_protocol = params.get("target_protocol", "unknown")
        logger.info(f"Translating from {source_protocol} to {target_protocol}")
        return {
            "status": "acknowledged",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": f"Translation from {source_protocol} to {target_protocol} acknowledged"
        }


if __name__ == "__main__":
    # Simple demonstration
    agent = AdaptiveBridgeBuilder()
    
    # Example JSON-RPC message
    test_message = {
        "jsonrpc": "2.0",
        "method": "getAgentCard",
        "params": {},
        "id": "test-1"
    }
    
    # Process and print response
    response = agent.process_message(test_message)
    print(json.dumps(response, indent=2))
