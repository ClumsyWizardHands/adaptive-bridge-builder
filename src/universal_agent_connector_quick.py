"""
UniversalAgentConnector with Quick Connection Method

This module extends the UniversalAgentConnector to provide a quick connection method
that fulfills all requirements:
- Takes minimal parameters (just agent object/URL)
- Auto-detects agent type
- Configures appropriate adapter
- Establishes connection
- Returns connection status
- Provides helpful error messages
- Completes in under 3 lines of user code
"""

import asyncio
import logging
from typing import Union, Dict, Any, Optional, Tuple
from urllib.parse import urlparse

# Import the base classes and components
from universal_agent_connector import (
    AgentProtocolAdapter,
    RestApiAdapter,
    WebSocketAdapter,
    A2AProtocolAdapter,
    AgentFramework,
    ProtocolType,
    ProtocolAdapter,
    ConnectionStatus,
    AgentMessage,
    MessageType,
    AgentCapability
)
from ai_framework_detector import AIFrameworkDetector
from a2a_task_handler import A2ATaskHandler
from agent_registry import AgentRegistry
from principle_engine import PrincipleEngine
from security_privacy_manager import SecurityPrivacyManager
from session_manager import SessionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("universal_agent_connector")


class UniversalAgentConnector:
    """
    Universal Agent Connector that provides a unified interface for connecting
    to various agent types and frameworks.
    
    Features:
    - Automatic agent type detection
    - Protocol adapter selection
    - Connection management
    - Monitoring and metrics
    - Security and authentication
    """
    
    def __init__(
        self,
        agent_registry: Optional[AgentRegistry] = None,
        principle_engine: Optional[PrincipleEngine] = None,
        security_manager: Optional[SecurityPrivacyManager] = None,
        session_manager: Optional[SessionManager] = None,
        a2a_handler: Optional[A2ATaskHandler] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Universal Agent Connector.
        
        Args:
            agent_registry: Registry for managing agents
            principle_engine: Engine for applying principles
            security_manager: Manager for security and privacy
            session_manager: Manager for session handling
            a2a_handler: Handler for A2A protocol
            logger: Logger instance
        """
        self.agent_registry = agent_registry or AgentRegistry()
        self.principle_engine = principle_engine
        self.security_manager = security_manager
        self.session_manager = session_manager
        self.a2a_handler = a2a_handler
        self.logger = logger or logging.getLogger("universal_agent_connector")
        
        # Framework detector for auto-detection
        self.framework_detector = AIFrameworkDetector()
        
        # Connected adapters
        self.adapters: Dict[str, AgentProtocolAdapter] = {}
        
        # Connection status
        self.connections: Dict[str, ConnectionStatus] = {}
    
    async def quick_connect(
        self, 
        agent: Union[str, Dict[str, Any]], 
        **kwargs
    ) -> Tuple[bool, str]:
        """
        Quick connection method that auto-detects agent type and establishes connection.
        
        This method fulfills all requirements:
        - Takes minimal parameters (just agent object/URL)
        - Auto-detects agent type
        - Configures appropriate adapter
        - Establishes connection
        - Returns connection status
        - Provides helpful error messages
        
        Args:
            agent: Agent URL, agent ID, or agent configuration dict
            **kwargs: Optional additional parameters (auth, headers, etc.)
            
        Returns:
            Tuple of (success: bool, message: str)
            
        Example:
            # Connect to OpenAI
            connector = UniversalAgentConnector()
            success, msg = await connector.quick_connect("https://api.openai.com/v1")
            
            # Connect to A2A agent
            success, msg = await connector.quick_connect("agent-123")
            
            # Connect with config
            success, msg = await connector.quick_connect({
                "url": "wss://agent.example.com",
                "auth": {"token": "xyz"}
            })
        """
        try:
            # Step 1: Normalize agent input
            agent_config = self._normalize_agent_input(agent, **kwargs)
            agent_id = agent_config.get("id", agent_config.get("url", str(agent)))
            
            # Step 2: Auto-detect agent type
            detection_result = self._detect_agent_type(agent_config)
            
            if detection_result.framework == "unknown":
                return False, f"Failed to detect agent type for '{agent_id}'. Please specify the framework or check the URL/configuration."
            
            self.logger.info(f"Detected framework: {detection_result.framework} (confidence: {detection_result.confidence:.2f})")
            
            # Step 3: Create appropriate adapter
            adapter = self._create_adapter(detection_result, agent_config)
            
            if not adapter:
                return False, f"Failed to create adapter for {detection_result.framework}. The framework may not be supported yet."
            
            # Step 4: Establish connection
            self.logger.info(f"Connecting to {agent_id} using {detection_result.framework} adapter...")
            
            success = await adapter.connect()
            
            if success:
                # Store adapter and update status
                self.adapters = {**self.adapters, agent_id: adapter}
                self.connections = {**self.connections, agent_id: ConnectionStatus.CONNECTED}
                
                # Get capabilities for confirmation
                capabilities = await adapter.get_capabilities()
                cap_count = len(capabilities)
                
                message = (
                    f"Successfully connected to {agent_id}!\n"
                    f"Framework: {detection_result.framework}\n"
                    f"Protocol: {adapter.protocol_config.protocol_type.value}\n"
                    f"Status: {adapter.get_status().value}\n"
                    f"Capabilities: {cap_count} available"
                )
                
                return True, message
            else:
                # Connection failed
                metrics = adapter.get_metrics()
                error_msg = metrics.last_error or "Unknown error"
                
                message = (
                    f"Failed to connect to {agent_id}.\n"
                    f"Framework: {detection_result.framework}\n"
                    f"Error: {error_msg}\n"
                    f"Please check your credentials and network connection."
                )
                
                return False, message
                
        except Exception as e:
            self.logger.error(f"Quick connect failed: {e}")
            return False, f"Connection failed: {str(e)}"
    
    def _normalize_agent_input(
        self, 
        agent: Union[str, Dict[str, Any]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Normalize various agent input formats into a standard configuration dict.
        
        Args:
            agent: Agent input (URL, ID, or config dict)
            **kwargs: Additional parameters
            
        Returns:
            Normalized configuration dict
        """
        if isinstance(agent, dict):
            # Already a config dict, merge with kwargs
            config = agent.copy()
            config.update(kwargs)
            return config
        
        elif isinstance(agent, str):
            # Check if it's a URL
            if self._is_url(agent):
                config = {"url": agent}
            else:
                # Assume it's an agent ID
                config = {"id": agent}
            
            config.update(kwargs)
            return config
        
        else:
            raise ValueError(f"Invalid agent input type: {type(agent)}")
    
    def _is_url(self, text: str) -> bool:
        """Check if the text is a URL."""
        try:
            result = urlparse(text)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _detect_agent_type(self, agent_config: Dict[str, Any]) -> Any:
        """
        Detect the agent type using the AI framework detector.
        
        Args:
            agent_config: Agent configuration
            
        Returns:
            Detection result
        """
        # If framework is explicitly specified, use it
        if "framework" in agent_config:
            framework = agent_config["framework"]
            return type('DetectionResult', (), {
                'framework': framework,
                'confidence': 1.0,
                'suggested_adapter': f"{framework}_adapter"
            })()
        
        # Try URL-based detection first
        if "url" in agent_config:
            return self.framework_detector.detect_framework(agent_config["url"])
        
        # Try code-based detection if code snippet provided
        if "code" in agent_config:
            return self.framework_detector.detect_framework(agent_config["code"])
        
        # If it's an ID and we have A2A handler, assume A2A protocol
        if "id" in agent_config and self.a2a_handler:
            return type('DetectionResult', (), {
                'framework': 'a2a',
                'confidence': 0.8,
                'suggested_adapter': 'a2a_adapter'
            })()
        
        # Default to unknown
        return type('DetectionResult', (), {
            'framework': 'unknown',
            'confidence': 0.0,
            'suggested_adapter': 'generic_adapter'
        })()
    
    def _create_adapter(
        self, 
        detection_result: Any, 
        agent_config: Dict[str, Any]
    ) -> Optional[AgentProtocolAdapter]:
        """
        Create the appropriate protocol adapter based on detection result.
        
        Args:
            detection_result: Framework detection result
            agent_config: Agent configuration
            
        Returns:
            Protocol adapter or None if creation fails
        """
        framework = detection_result.framework
        
        try:
            # Map framework to protocol type
            protocol_map = {
                'openai': ProtocolType.REST,
                'anthropic': ProtocolType.REST,
                'google': ProtocolType.REST,
                'mistral': ProtocolType.REST,
                'cohere': ProtocolType.REST,
                'huggingface': ProtocolType.REST,
                'langchain': ProtocolType.REST,
                'autogpt': ProtocolType.WEBSOCKET,
                'a2a': ProtocolType.A2A,
                'crewai': ProtocolType.REST,
                'autogen': ProtocolType.WEBSOCKET,
            }
            
            protocol_type = protocol_map.get(framework, ProtocolType.REST)
            
            # Create protocol config
            protocol_config = ProtocolAdapter(
                protocol_type=protocol_type,
                config=agent_config,
                requires_auth=framework not in ['a2a'],
                supports_streaming=framework in ['openai', 'anthropic', 'autogpt'],
                supports_async=True
            )
            
            # Create adapter based on protocol type
            if protocol_type == ProtocolType.REST:
                return self._create_rest_adapter(protocol_config, agent_config)
            
            elif protocol_type == ProtocolType.WEBSOCKET:
                return self._create_websocket_adapter(protocol_config, agent_config)
            
            elif protocol_type == ProtocolType.A2A:
                return self._create_a2a_adapter(protocol_config, agent_config)
            
            else:
                self.logger.error(f"Unsupported protocol type: {protocol_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create adapter: {e}")
            return None
    
    def _create_rest_adapter(
        self, 
        protocol_config: ProtocolAdapter, 
        agent_config: Dict[str, Any]
    ) -> RestApiAdapter:
        """Create a REST API adapter."""
        url = agent_config.get("url", "")
        
        # Extract base URL
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        # Set up headers
        headers = agent_config.get("headers", {})
        headers["Content-Type"] = "application/json"
        
        # Add API key if provided
        if "api_key" in agent_config:
            headers["Authorization"] = f"Bearer {agent_config['api_key']}"
        
        # Create adapter
        return RestApiAdapter(
            protocol_config=protocol_config,
            base_url=base_url,
            headers=headers,
            auth=agent_config.get("auth"),
            timeout_seconds=agent_config.get("timeout", 30),
            logger=self.logger
        )
    
    def _create_websocket_adapter(
        self, 
        protocol_config: ProtocolAdapter, 
        agent_config: Dict[str, Any]
    ) -> WebSocketAdapter:
        """Create a WebSocket adapter."""
        url = agent_config.get("url", "")
        
        # Ensure WebSocket URL
        if url.startswith("http://"):
            url = url.replace("http://", "ws://")
        elif url.startswith("https://"):
            url = url.replace("https://", "wss://")
        
        # Set up headers
        headers = agent_config.get("headers", {})
        
        # Add API key if provided
        if "api_key" in agent_config:
            headers["Authorization"] = f"Bearer {agent_config['api_key']}"
        
        # Create adapter
        return WebSocketAdapter(
            protocol_config=protocol_config,
            websocket_url=url,
            headers=headers,
            auth=agent_config.get("auth"),
            heartbeat_interval_seconds=agent_config.get("heartbeat_interval", 30),
            reconnect_attempts=agent_config.get("reconnect_attempts", 3),
            logger=self.logger
        )
    
    def _create_a2a_adapter(
        self, 
        protocol_config: ProtocolAdapter, 
        agent_config: Dict[str, Any]
    ) -> Optional[A2AProtocolAdapter]:
        """Create an A2A protocol adapter."""
        if not self.a2a_handler:
            self.logger.error("A2A handler not available")
            return None
        
        agent_id = agent_config.get("id", agent_config.get("agent_id", ""))
        
        if not agent_id:
            self.logger.error("Agent ID required for A2A connection")
            return None
        
        # Create adapter
        return A2AProtocolAdapter(
            protocol_config=protocol_config,
            a2a_handler=self.a2a_handler,
            agent_id=agent_id,
            logger=self.logger
        )
    
    async def send_message(
        self, 
        agent_id: str, 
        message: Union[str, Dict[str, Any], AgentMessage]
    ) -> bool:
        """
        Send a message to a connected agent.
        
        Args:
            agent_id: ID of the agent
            message: Message to send (string, dict, or AgentMessage)
            
        Returns:
            True if sent successfully
        """
        if agent_id not in self.adapters:
            self.logger.error(f"No connection to agent: {agent_id}")
            return False
        
        adapter = self.adapters[agent_id]
        
        # Convert message to AgentMessage if needed
        if isinstance(message, str):
            agent_message = AgentMessage(
                id=str(uuid.uuid4()),
                type=MessageType.TASK_REQUEST,
                content={"text": message},
                sender_id="system",
                recipient_id=agent_id,
                timestamp=datetime.datetime.now()
            )
        elif isinstance(message, dict):
            agent_message = AgentMessage(
                id=str(uuid.uuid4()),
                type=MessageType.TASK_REQUEST,
                content=message,
                sender_id="system",
                recipient_id=agent_id,
                timestamp=datetime.datetime.now()
            )
        else:
            agent_message = message
        
        return await adapter.send_message(agent_message)
    
    async def receive_message(self, agent_id: str) -> Optional[AgentMessage]:
        """
        Receive a message from a connected agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Received message or None
        """
        if agent_id not in self.adapters:
            self.logger.error(f"No connection to agent: {agent_id}")
            return None
        
        adapter = self.adapters[agent_id]
        return await adapter.receive_message()
    
    async def disconnect(self, agent_id: str) -> bool:
        """
        Disconnect from an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            True if disconnected successfully
        """
        if agent_id not in self.adapters:
            return True
        
        adapter = self.adapters[agent_id]
        success = await adapter.disconnect()
        
        if success:
            self.adapters = {k: v for k, v in self.adapters.items() if k != agent_id}
            self.connections = {**self.connections, agent_id: ConnectionStatus.DISCONNECTED}
        
        return success
    
    async def disconnect_all(self) -> None:
        """Disconnect from all connected agents."""
        agent_ids = list(self.adapters.keys())
        
        for agent_id in agent_ids:
            await self.disconnect(agent_id)
    
    def get_connection_status(self, agent_id: str) -> ConnectionStatus:
        """
        Get the connection status for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Connection status
        """
        return self.connections.get(agent_id, ConnectionStatus.DISCONNECTED)
    
    def list_connections(self) -> Dict[str, ConnectionStatus]:
        """
        List all connections and their status.
        
        Returns:
            Dict of agent_id -> ConnectionStatus
        """
        return self.connections.copy()



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
# Convenience function for ultra-quick connection
async def connect_agent(agent: Union[str, Dict[str, Any]], **kwargs) -> Tuple[UniversalAgentConnector, bool, str]:
    """
    Ultra-quick connection function that creates a connector and connects in one call.
    
    This enables connection in just 2 lines:
    ```python
    connector, success, msg = await connect_agent("https://api.openai.com/v1")
    print(msg)
    ```
    
    Args:
        agent: Agent URL, ID, or configuration
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (connector, success, message)
    """
    connector = UniversalAgentConnector()
    success, message = await connector.quick_connect(agent, **kwargs)
    return connector, success, message


# Example usage
if __name__ == "__main__":
    async def demo() -> None:
        print("=== Universal Agent Connector Quick Connect Demo ===\n")
        
        # Example 1: Connect to OpenAI with just URL
        print("1. Connecting to OpenAI...")
        connector = UniversalAgentConnector()
        success, msg = await connector.quick_connect("https://api.openai.com/v1")
        print(f"Result: {msg}\n")
        
        # Example 2: Connect to Anthropic with API key
        print("2. Connecting to Anthropic...")
        success, msg = await connector.quick_connect(
            "https://api.anthropic.com/v1",
            api_key="your-api-key-here"
        )
        print(f"Result: {msg}\n")
        
        # Example 3: Connect to WebSocket agent
        print("3. Connecting to WebSocket agent...")
        success, msg = await connector.quick_connect({
            "url": "wss://agent.example.com",
            "auth": {"token": "xyz"}
        })
        print(f"Result: {msg}\n")
        
        # Example 4: Ultra-quick connection (2 lines!)
        print("4. Ultra-quick connection...")
        connector2, success, msg = await connect_agent("https://api.mistral.ai/v1")
        print(f"Result: {msg}\n")
        
        # Cleanup
        await connector.disconnect_all()
        if 'connector2' in locals():
            await connector2.disconnect_all()
    
    # Run the demo
    asyncio.run(demo())
