"""
Universal Agent Connector LLM Integration

This module extends the UniversalAgentConnector to work with LLM adapters.
It provides a bridge between the LLM adapters and the A2A protocol, allowing
LLMs to be used as agents in the ecosystem.
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

from universal_agent_connector import (
    UniversalAgentConnector,
    AgentFramework,
    AgentMessage,
    MessageType,
    ProtocolType,
    ConnectionStatus
)
from llm_adapter_interface import (
    BaseLLMAdapter,
    AuthenticationError,
    RequestError,
    ResponseError,
    RateLimitError
)
from agent_registry_llm_integration import LLMAgentRegistry
from llm_key_manager import LLMKeyManager


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("universal_agent_connector_llm")


class LLMConnector(UniversalAgentConnector):
    """
    A connector that bridges LLM adapters with the A2A protocol.
    
    This connector allows LLMs to be used as agents in the ecosystem,
    handling the translation between A2A messages and LLM API calls.
    """
    
    def __init__(
        self,
        llm_registry: LLMAgentRegistry,
        adapter_provider: str = "openai",
        model_name: Optional[str] = None,
        system_message: str = "You are a helpful AI assistant.",
        **kwargs
    ):
        """
        Initialize the LLM connector.
        
        Args:
            llm_registry: Registry of LLM adapters
            adapter_provider: Provider name for the LLM adapter to use
            model_name: Optional model name to use
            system_message: System message to use for the LLM
            **kwargs: Additional arguments for the UniversalAgentConnector
        """
        # Initialize the base class
        super().__init__(
            agent_framework=AgentFramework.CUSTOM,
            protocol_type=ProtocolType.CUSTOM,
            **kwargs
        )
        
        # LLM-specific attributes
        self.llm_registry = llm_registry
        self.adapter_provider = adapter_provider
        self.system_message = system_message
        
        # Get the LLM adapter
        self.llm_adapter = llm_registry.get_adapter_by_provider(adapter_provider)
        
        if not self.llm_adapter:
            raise ValueError(f"No LLM adapter found for provider: {adapter_provider}")
            
        # Override model name if provided
        if model_name:
            self.llm_adapter.model_name = model_name
            
        # Initialize conversational context
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Mark as connected if we have an adapter
        self.connection_status = ConnectionStatus.CONNECTED
    
    async def connect(self) -> bool:
        """
        Establish a connection to the LLM.
        
        For LLMs, this mainly validates the adapter is functioning.
        
        Returns:
            True if connected successfully
        """
        if not self.llm_adapter:
            self.connection_status = ConnectionStatus.FAILED
            return False
            
        # The adapter already validates the API key during initialization,
        # so we just need to update the status
        self.connection_status = ConnectionStatus.CONNECTED
        return True
    
    async def disconnect(self) -> bool:
        """
        Disconnect from the LLM.
        
        For LLMs, this just updates the connection status.
        
        Returns:
            True if disconnected successfully
        """
        self.connection_status = ConnectionStatus.DISCONNECTED
        return True
    
    async def send_message(self, message: AgentMessage) -> bool:
        """
        Send a message to the LLM.
        
        Translates an A2A message to an LLM API call.
        
        Args:
            message: The message to send
            
        Returns:
            True if sent successfully
        """
        try:
            # Extract message content
            conversation_id = message.metadata.get("conversation_id", str(uuid.uuid4()))
            content = message.content.get("text", "")
            
            # Initialize conversation history if this is a new conversation
            if conversation_id not in self.conversation_history:
                self.conversation_history = {**self.conversation_history, conversation_id: []}
            
            # Add user message to conversation history
            self.conversation_history[conversation_id].append({
                "role": "user",
                "content": content
            })
            
            # Send request to LLM
            response = await self.llm_adapter.send_request(
                prompt=content,
                system_message=self.system_message
            )
            
            # Process the response
            result = self.llm_adapter.process_response(response)
            
            # Add assistant message to conversation history
            self.conversation_history[conversation_id].append({
                "role": "assistant",
                "content": result
            })
            
            # Create a response message
            response_message = AgentMessage(
                id=str(uuid.uuid4()),
                type=MessageType.TASK_RESPONSE,
                content={"text": result},
                sender_id=self.llm_adapter.provider_name,
                recipient_id=message.sender_id,
                timestamp=datetime.now(),
                correlation_id=message.id,
                metadata={
                    "conversation_id": conversation_id,
                    "tokens": response.get("usage", {})
                }
            )
            
            # Add message to received queue for later retrieval
            await self._queue_received_message(response_message)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending message to LLM: {e}")
            
            # Create an error response
            error_message = AgentMessage(
                id=str(uuid.uuid4()),
                type=MessageType.ERROR,
                content={"error": str(e)},
                sender_id=self.llm_adapter.provider_name,
                recipient_id=message.sender_id,
                timestamp=datetime.now(),
                correlation_id=message.id,
                metadata={
                    "conversation_id": message.metadata.get("conversation_id", ""),
                    "error_type": type(e).__name__
                }
            )
            
            # Add error message to received queue
            await self._queue_received_message(error_message)
            
            return False
    
    async def receive_message(self) -> Optional[AgentMessage]:
        """
        Receive a message from the message queue.
        
        Returns:
            The next message or None if no messages are available
        """
        # Messages are queued by the send_message method
        return await self._get_next_message()
    
    async def _queue_received_message(self, message: AgentMessage) -> None:
        """
        Queue a message for later retrieval.
        
        Args:
            message: The message to queue
        """
        # In a full implementation, this would use a proper queue
        # For now, we'll use a simple list as a message queue
        if not hasattr(self, "_message_queue"):
            self._message_queue = []
            
        self._message_queue = [*self._message_queue, message]
    
    async def _get_next_message(self) -> Optional[AgentMessage]:
        """
        Get the next message from the queue.
        
        Returns:
            The next message or None if no messages are available
        """
        if not hasattr(self, "_message_queue") or not self._message_queue:
            return None
            
        return self._message_queue.pop(0)
    
    async def get_capabilities(self) -> List[Dict[str, Any]]:
        """
        Get the capabilities of the LLM.
        
        Returns:
            List of capability descriptions
        """
        # Get model info
        model_info = self.llm_adapter.get_model_info()
        
        # Convert model capabilities to agent capabilities
        capabilities = []
        
        # Basic text generation capability
        capabilities.append({
            "id": "text_generation",
            "name": "Text Generation",
            "description": f"Generate text using {self.llm_adapter.provider_name}'s {self.llm_adapter.model_name}",
            "parameters": {
                "prompt": {"type": "string", "description": "The input prompt"}
            }
        })
        
        # Add model-specific capabilities
        model_capabilities = model_info.get("capabilities", [])
        
        if "image_understanding" in model_capabilities or "image_processing" in model_capabilities:
            capabilities.append({
                "id": "image_understanding",
                "name": "Image Understanding",
                "description": f"Process and understand images using {self.llm_adapter.provider_name}'s {self.llm_adapter.model_name}",
                "parameters": {
                    "image_url": {"type": "string", "description": "URL of the image to process"},
                    "prompt": {"type": "string", "description": "Text prompt to guide image analysis"}
                }
            })
            
        if "function_calling" in model_capabilities:
            capabilities.append({
                "id": "function_calling",
                "name": "Function Calling",
                "description": f"Execute function calls using {self.llm_adapter.provider_name}'s {self.llm_adapter.model_name}",
                "parameters": {
                    "prompt": {"type": "string", "description": "The input prompt"},
                    "functions": {"type": "array", "description": "Function definitions"}
                }
            })
            
        return capabilities



    async def __aenter__(self):
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context and cleanup."""
        if hasattr(self, 'cleanup'):
            await self.cleanup()
        elif hasattr(self, 'close'):
            await self.close()
        return False
# Example usage
if __name__ == "__main__":
    import asyncio
    from agent_registry_llm_integration import setup_llm_registry
    
    async def run_example() -> Coroutine[Any, Any, None]:
        # Set up the LLM registry
        llm_registry = setup_llm_registry()
        
        # Get available LLM adapters
        llm_adapters = llm_registry.list_llm_adapters()
        
        if not llm_adapters:
            print("No LLM adapters registered. Please check your API keys.")
            return
            
        # Use the first available adapter
        provider_name = llm_adapters[0]
        
        # Create an LLM connector
        connector = LLMConnector(
            llm_registry=llm_registry,
            adapter_provider=provider_name,
            system_message="You are a helpful AI assistant specialized in explaining technical concepts."
        )
        
        # Connect to the LLM
        await connector.connect()
        
        # Create a test message
        test_message = AgentMessage(
            id=str(uuid.uuid4()),
            type=MessageType.TASK_REQUEST,
            content={"text": "Explain what an LLM adapter is and why it's useful, in 3 sentences."},
            sender_id="test_user",
            recipient_id=provider_name,
            timestamp=datetime.now(),
            metadata={"conversation_id": "test_conversation"}
        )
        
        # Send the message
        print(f"Sending message to {provider_name}...")
        await connector.send_message(test_message)
        
        # Receive the response
        response = await connector.receive_message()
        
        if response:
            print("\nResponse:")
            print(response.content.get("text", ""))
            
            # Show token usage if available
            tokens = response.metadata.get("tokens", {})
            if tokens:
                print(f"\nToken usage: {tokens.get('prompt_tokens', 0)} prompt, "
                      f"{tokens.get('completion_tokens', 0)} completion, "
                      f"{tokens.get('total_tokens', 0)} total")
        else:
            print("No response received")
        
        # Disconnect
        await connector.disconnect()
    
    asyncio.run(run_example())
