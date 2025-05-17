"""
UniversalAgentConnector - A system for integrating various agent types regardless of their framework.

This component enables the Adaptive Bridge Builder to connect with agents built on different
frameworks (GPT, Claude, Gemini, etc.) by providing a unified interface, protocol adapters,
standardized communication patterns, monitoring capabilities, and reliability features.
The connector applies the 'Adaptability as a Form of Strength' principle to maximize compatibility.
"""
import abc
import asyncio
import datetime
import enum
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, TypeVar, Generic

# Import existing components that we will integrate with
from principle_engine import PrincipleEngine
from agent_registry import AgentRegistry
from agent_card import AgentCard
from a2a_task_handler import A2ATaskHandler
from communication_adapter import CommunicationAdapter
from security_privacy_manager import SecurityPrivacyManager
from session_manager import SessionManager


# Type variables for generic typing
T = TypeVar('T')  # Generic response type
U = TypeVar('U')  # Generic request type


class AgentFramework(enum.Enum):
    """Supported agent frameworks"""
    GPT = "gpt"                # OpenAI GPT models
    CLAUDE = "claude"          # Anthropic Claude models
    GEMINI = "gemini"          # Google Gemini models
    LLAMA = "llama"            # Meta Llama models
    MISTRAL = "mistral"        # Mistral AI models
    CUSTOM = "custom"          # Custom/proprietary models
    A2A = "a2a"                # Standard A2A protocol
    LEGACY = "legacy"          # Legacy systems with custom protocols
    HUMAN = "human"            # Human agents


class ProtocolType(enum.Enum):
    """Types of communication protocols supported"""
    A2A = "a2a"                # Standard Agent-to-Agent protocol
    REST = "rest"              # RESTful API
    WEBSOCKET = "websocket"    # WebSocket for real-time communication
    GRPC = "grpc"              # gRPC for high-performance communication
    GRAPHQL = "graphql"        # GraphQL for flexible API queries
    MESSAGE_QUEUE = "mq"       # Message Queue (RabbitMQ, Kafka, etc.)
    CUSTOM = "custom"          # Custom protocol
    VOICE = "voice"            # Voice-based protocol
    TEXT = "text"              # Simple text-based protocol


class MessageType(enum.Enum):
    """Types of messages exchanged between agents"""
    TASK_REQUEST = "task_request"         # Request to perform a task
    TASK_RESPONSE = "task_response"       # Response to a task request
    INFORMATION_REQUEST = "info_request"  # Request for information
    INFORMATION_RESPONSE = "info_response"  # Response to an information request
    NOTIFICATION = "notification"         # Notification message
    ERROR = "error"                       # Error message
    HEARTBEAT = "heartbeat"               # Heartbeat/keepalive message
    CAPABILITY_REQUEST = "capability_request"  # Request for agent capabilities
    CAPABILITY_RESPONSE = "capability_response"  # Response with agent capabilities
    AUTH_REQUEST = "auth_request"         # Authentication request
    AUTH_RESPONSE = "auth_response"       # Authentication response


class ConnectionStatus(enum.Enum):
    """Status of the connection to an agent"""
    DISCONNECTED = "disconnected"  # Not connected
    CONNECTING = "connecting"      # Connection in progress
    CONNECTED = "connected"        # Connected and ready
    DEGRADED = "degraded"          # Connected but with issues
    FAILED = "failed"              # Connection failed
    UNAUTHORIZED = "unauthorized"  # Not authorized to connect


@dataclass
class AgentCapability:
    """Represents a capability that an agent can perform"""
    id: str
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    example: Optional[str] = None
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)


@dataclass
class AgentMessage:
    """Message exchanged between agents"""
    id: str
    type: MessageType
    content: Dict[str, Any]
    sender_id: str
    recipient_id: str
    timestamp: datetime.datetime
    correlation_id: Optional[str] = None
    priority: int = 0
    expires_at: Optional[datetime.datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectionMetrics:
    """Metrics for monitoring the connection to an agent"""
    latency_ms: int = 0
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    last_request_time: Optional[datetime.datetime] = None
    last_response_time: Optional[datetime.datetime] = None
    average_response_time_ms: int = 0
    uptime_seconds: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime.datetime] = None


@dataclass
class ProtocolAdapter:
    """
    Configuration for a protocol adapter that handles communication
    with agents using a specific protocol
    """
    protocol_type: ProtocolType
    config: Dict[str, Any] = field(default_factory=dict)
    requires_auth: bool = False
    supports_streaming: bool = False
    supports_batch: bool = False
    supports_sync: bool = True
    supports_async: bool = True
    max_message_size_kb: int = 1024  # 1MB
    serialization_format: str = "json"  # json, protobuf, etc.


class AgentProtocolAdapter(abc.ABC):
    """
    Abstract base class for protocol adapters that handle communication 
    with agents using different protocols.
    
    This class defines the interface that all protocol adapters must implement.
    """
    
    def __init__(
        self, 
        protocol_config: ProtocolAdapter,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the protocol adapter with configuration"""
        self.protocol_config = protocol_config
        self.logger = logger or logging.getLogger(f"agent_protocol.{protocol_config.protocol_type.value}")
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.metrics = ConnectionMetrics()
        self.last_heartbeat = None
    
    @abc.abstractmethod
    async def connect(self) -> bool:
        """Establish a connection to the agent"""
        pass
    
    @abc.abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the agent"""
        pass
    
    @abc.abstractmethod
    async def send_message(self, message: AgentMessage) -> bool:
        """Send a message to the agent"""
        pass
    
    @abc.abstractmethod
    async def receive_message(self) -> Optional[AgentMessage]:
        """Receive a message from the agent"""
        pass
    
    @abc.abstractmethod
    async def get_capabilities(self) -> List[AgentCapability]:
        """Get the capabilities of the agent"""
        pass
    
    async def send_heartbeat(self) -> bool:
        """Send a heartbeat message to the agent to check if it's alive"""
        try:
            heartbeat_message = AgentMessage(
                id=str(uuid.uuid4()),
                type=MessageType.HEARTBEAT,
                content={},
                sender_id="system",
                recipient_id="agent",
                timestamp=datetime.datetime.now()
            )
            
            start_time = time.time()
            result = await self.send_message(heartbeat_message)
            end_time = time.time()
            
            if result:
                self.metrics.latency_ms = int((end_time - start_time) * 1000)
                self.last_heartbeat = datetime.datetime.now()
                
                if self.connection_status == ConnectionStatus.DEGRADED:
                    self.connection_status = ConnectionStatus.CONNECTED
                    self.logger.info(f"Connection restored: latency={self.metrics.latency_ms}ms")
            else:
                if self.connection_status == ConnectionStatus.CONNECTED:
                    self.connection_status = ConnectionStatus.DEGRADED
                    self.logger.warning("Connection degraded: heartbeat failed")
                    
            return result
        except Exception as e:
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.datetime.now()
            self.metrics.error_count += 1
            self.connection_status = ConnectionStatus.DEGRADED
            self.logger.error(f"Heartbeat failed: {e}")
            return False
    
    def get_status(self) -> ConnectionStatus:
        """Get the current connection status"""
        return self.connection_status
    
    def get_metrics(self) -> ConnectionMetrics:
        """Get metrics for the connection"""
        # Update uptime if connected
        if self.connection_status in [ConnectionStatus.CONNECTED, ConnectionStatus.DEGRADED]:
            self.metrics.uptime_seconds = int(time.time() - self.metrics.uptime_seconds)
            
        return self.metrics
    
    def _update_metrics(self, success: bool, response_time_ms: int) -> None:
        """Update metrics with the result of a request"""
        self.metrics.request_count += 1
        
        if success:
            self.metrics.success_count += 1
            self.metrics.last_response_time = datetime.datetime.now()
            
            # Update average response time
            if self.metrics.average_response_time_ms == 0:
                self.metrics.average_response_time_ms = response_time_ms
            else:
                self.metrics.average_response_time_ms = int(
                    (self.metrics.average_response_time_ms * (self.metrics.success_count - 1) + response_time_ms) / 
                    self.metrics.success_count
                )
        else:
            self.metrics.error_count += 1


class RestApiAdapter(AgentProtocolAdapter):
    """
    Protocol adapter for RESTful API communication.
    
    This adapter handles communication with agents that expose a RESTful API.
    """
    
    def __init__(
        self, 
        protocol_config: ProtocolAdapter,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        auth: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 30,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the REST API adapter"""
        super().__init__(protocol_config, logger)
        self.base_url = base_url
        self.headers = headers or {"Content-Type": "application/json"}
        self.auth = auth
        self.timeout = timeout_seconds
        self.session = None  # Will be initialized during connect()
    
    async def connect(self) -> bool:
        """
        Establish a connection to the agent by creating an HTTP session.
        
        For REST, this mainly involves setting up the session and validating
        that the API is accessible.
        """
        try:
            import aiohttp
            
            self.connection_status = ConnectionStatus.CONNECTING
            self.logger.info(f"Connecting to REST API at {self.base_url}")
            
            # Create a session
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            
            # Test the connection with a simple GET request
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    self.connection_status = ConnectionStatus.CONNECTED
                    self.metrics.uptime_seconds = int(time.time())
                    self.logger.info("Successfully connected to REST API")
                    return True
                else:
                    self.connection_status = ConnectionStatus.FAILED
                    self.metrics.last_error = f"Health check failed: {response.status}"
                    self.metrics.last_error_time = datetime.datetime.now()
                    self.logger.error(f"Failed to connect to REST API: {response.status}")
                    return False
                    
        except Exception as e:
            self.connection_status = ConnectionStatus.FAILED
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.datetime.now()
            self.logger.error(f"Failed to connect to REST API: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """
        Disconnect from the agent by closing the HTTP session.
        """
        try:
            if self.session:
                await self.session.close()
                self.session = None
                
            self.connection_status = ConnectionStatus.DISCONNECTED
            self.logger.info("Disconnected from REST API")
            return True
        except Exception as e:
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.datetime.now()
            self.logger.error(f"Failed to disconnect from REST API: {e}")
            return False
    
    async def send_message(self, message: AgentMessage) -> bool:
        """
        Send a message to the agent using a POST request.
        
        Maps the message to the appropriate REST endpoint based on message type.
        """
        if not self.session:
            self.logger.error("Cannot send message: Not connected to REST API")
            return False
            
        try:
            # Map message type to endpoint
            endpoint_map = {
                MessageType.TASK_REQUEST: "/tasks",
                MessageType.INFORMATION_REQUEST: "/info",
                MessageType.NOTIFICATION: "/notifications",
                MessageType.HEARTBEAT: "/health",
                MessageType.CAPABILITY_REQUEST: "/capabilities",
                MessageType.AUTH_REQUEST: "/auth"
            }
            
            endpoint = endpoint_map.get(message.type, "/messages")
            
            # Serialize message
            payload = {
                "id": message.id,
                "type": message.type.value,
                "content": message.content,
                "sender_id": message.sender_id,
                "recipient_id": message.recipient_id,
                "timestamp": message.timestamp.isoformat(),
                "correlation_id": message.correlation_id,
                "priority": message.priority,
                "expires_at": message.expires_at.isoformat() if message.expires_at else None,
                "metadata": message.metadata
            }
            
            # Record metrics
            self.metrics.last_request_time = datetime.datetime.now()
            start_time = time.time()
            
            # Send request
            async with self.session.post(
                f"{self.base_url}{endpoint}",
                json=payload
            ) as response:
                # Calculate response time
                end_time = time.time()
                response_time_ms = int((end_time - start_time) * 1000)
                
                # Update metrics
                success = 200 <= response.status < 300
                self._update_metrics(success, response_time_ms)
                
                if not success:
                    self.metrics.last_error = f"HTTP {response.status}: {await response.text()}"
                    self.metrics.last_error_time = datetime.datetime.now()
                    self.logger.error(f"Failed to send message: {self.metrics.last_error}")
                
                return success
                
        except Exception as e:
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.datetime.now()
            self.metrics.error_count += 1
            self.logger.error(f"Failed to send message: {e}")
            return False
    
    async def receive_message(self) -> Optional[AgentMessage]:
        """
        Receive a message from the agent using a GET request.
        
        For REST, this typically involves polling an endpoint for new messages.
        """
        if not self.session:
            self.logger.error("Cannot receive message: Not connected to REST API")
            return None
            
        try:
            # Record metrics
            start_time = time.time()
            
            # Send request to get messages
            async with self.session.get(
                f"{self.base_url}/messages"
            ) as response:
                # Calculate response time
                end_time = time.time()
                response_time_ms = int((end_time - start_time) * 1000)
                
                # Update metrics
                success = 200 <= response.status < 300
                self._update_metrics(success, response_time_ms)
                
                if not success:
                    self.metrics.last_error = f"HTTP {response.status}: {await response.text()}"
                    self.metrics.last_error_time = datetime.datetime.now()
                    self.logger.error(f"Failed to receive message: {self.metrics.last_error}")
                    return None
                
                # Parse response
                data = await response.json()
                
                # No messages
                if not data:
                    return None
                    
                # Convert to AgentMessage
                try:
                    return AgentMessage(
                        id=data["id"],
                        type=MessageType(data["type"]),
                        content=data["content"],
                        sender_id=data["sender_id"],
                        recipient_id=data["recipient_id"],
                        timestamp=datetime.datetime.fromisoformat(data["timestamp"]),
                        correlation_id=data.get("correlation_id"),
                        priority=data.get("priority", 0),
                        expires_at=datetime.datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
                        metadata=data.get("metadata", {})
                    )
                except Exception as e:
                    self.logger.error(f"Failed to parse message: {e}")
                    return None
                
        except Exception as e:
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.datetime.now()
            self.metrics.error_count += 1
            self.logger.error(f"Failed to receive message: {e}")
            return None
    
    async def get_capabilities(self) -> List[AgentCapability]:
        """
        Get the capabilities of the agent using a GET request.
        """
        if not self.session:
            self.logger.error("Cannot get capabilities: Not connected to REST API")
            return []
            
        try:
            # Record metrics
            start_time = time.time()
            
            # Send request to get capabilities
            async with self.session.get(
                f"{self.base_url}/capabilities"
            ) as response:
                # Calculate response time
                end_time = time.time()
                response_time_ms = int((end_time - start_time) * 1000)
                
                # Update metrics
                success = 200 <= response.status < 300
                self._update_metrics(success, response_time_ms)
                
                if not success:
                    self.metrics.last_error = f"HTTP {response.status}: {await response.text()}"
                    self.metrics.last_error_time = datetime.datetime.now()
                    self.logger.error(f"Failed to get capabilities: {self.metrics.last_error}")
                    return []
                
                # Parse response
                data = await response.json()
                
                # Convert to AgentCapability objects
                capabilities = []
                for cap_data in data:
                    capabilities.append(AgentCapability(
                        id=cap_data["id"],
                        name=cap_data["name"],
                        description=cap_data["description"],
                        parameters=cap_data.get("parameters", {}),
                        example=cap_data.get("example"),
                        version=cap_data.get("version", "1.0"),
                        tags=cap_data.get("tags", [])
                    ))
                
                return capabilities
                
        except Exception as e:
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.datetime.now()
            self.metrics.error_count += 1
            self.logger.error(f"Failed to get capabilities: {e}")
            return []


class WebSocketAdapter(AgentProtocolAdapter):
    """
    Protocol adapter for WebSocket communication.
    
    This adapter handles communication with agents that support WebSocket connections,
    which is particularly useful for real-time, bidirectional communication.
    """
    
    def __init__(
        self, 
        protocol_config: ProtocolAdapter,
        websocket_url: str,
        headers: Optional[Dict[str, str]] = None,
        auth: Optional[Dict[str, str]] = None,
        heartbeat_interval_seconds: int = 30,
        reconnect_attempts: int = 3,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the WebSocket adapter"""
        super().__init__(protocol_config, logger)
        self.websocket_url = websocket_url
        self.headers = headers or {}
        self.auth = auth
        self.heartbeat_interval = heartbeat_interval_seconds
        self.reconnect_attempts = reconnect_attempts
        self.ws = None  # WebSocket connection
        self.message_queue = asyncio.Queue()  # Queue for received messages
        self.heartbeat_task = None  # Task for sending heartbeats
        self.receive_task = None  # Task for receiving messages
    
    async def connect(self) -> bool:
        """
        Establish a connection to the agent using WebSocket.
        
        Sets up the connection and starts tasks for heartbeats and message receiving.
        """
        import websockets
        
        try:
            self.connection_status = ConnectionStatus.CONNECTING
            self.logger.info(f"Connecting to WebSocket at {self.websocket_url}")
            
            # Connect to WebSocket
            attempt = 0
            while attempt < self.reconnect_attempts:
                try:
                    self.ws = await websockets.connect(
                        self.websocket_url,
                        extra_headers=self.headers
                    )
                    break
                except Exception as e:
                    attempt += 1
                    if attempt >= self.reconnect_attempts:
                        raise e
                    self.logger.warning(f"Connection attempt {attempt} failed: {e}. Retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            # Send authentication if needed
            if self.auth:
                auth_message = AgentMessage(
                    id=str(uuid.uuid4()),
                    type=MessageType.AUTH_REQUEST,
                    content=self.auth,
                    sender_id="system",
                    recipient_id="agent",
                    timestamp=datetime.datetime.now()
                )
                
                await self.ws.send(json.dumps({
                    "id": auth_message.id,
                    "type": auth_message.type.value,
                    "content": auth_message.content,
                    "sender_id": auth_message.sender_id,
                    "recipient_id": auth_message.recipient_id,
                    "timestamp": auth_message.timestamp.isoformat()
                }))
                
                # Wait for auth response
                response_raw = await self.ws.recv()
                response = json.loads(response_raw)
                
                if response.get("type") != MessageType.AUTH_RESPONSE.value or response.get("content", {}).get("status") != "success":
                    self.connection_status = ConnectionStatus.UNAUTHORIZED
                    self.metrics.last_error = "Authentication failed"
                    self.metrics.last_error_time = datetime.datetime.now()
                    self.logger.error("Failed to authenticate with WebSocket")
                    await self.ws.close()
                    self.ws = None
                    return False
            
            # Start heartbeat task
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Start receive task
            self.receive_task = asyncio.create_task(self._receive_loop())
            
            self.connection_status = ConnectionStatus.CONNECTED
            self.metrics.uptime_seconds = int(time.time())
            self.logger.info("Successfully connected to WebSocket")
            return True
            
        except Exception as e:
            self.connection_status = ConnectionStatus.FAILED
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.datetime.now()
            self.logger.error(f"Failed to connect to WebSocket: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """
        Disconnect from the agent by closing the WebSocket connection
        and canceling any running tasks.
        """
        try:
            # Cancel tasks
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
                self.heartbeat_task = None
                
            if self.receive_task:
                self.receive_task.cancel()
                self.receive_task = None
            
            # Close WebSocket connection
            if self.ws:
                await self.ws.close()
                self.ws = None
                
            self.connection_status = ConnectionStatus.DISCONNECTED
            self.logger.info("Disconnected from WebSocket")
            return True
        except Exception as e:
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.datetime.now()
            self.logger.error(f"Failed to disconnect from WebSocket: {e}")
            return False
    
    async def send_message(self, message: AgentMessage) -> bool:
        """
        Send a message to the agent through the WebSocket connection.
        """
        if not self.ws:
            self.logger.error("Cannot send message: Not connected to WebSocket")
            return False
            
        try:
            # Serialize message
            payload = {
                "id": message.id,
                "type": message.type.value,
                "content": message.content,
                "sender_id": message.sender_id,
                "recipient_id": message.recipient_id,
                "timestamp": message.timestamp.isoformat(),
                "correlation_id": message.correlation_id,
                "priority": message.priority,
                "expires_at": message.expires_at.isoformat() if message.expires_at else None,
                "metadata": message.metadata
            }
            
            # Record metrics
            self.metrics.last_request_time = datetime.datetime.now()
            start_time = time.time()
            
            # Send message
            await self.ws.send(json.dumps(payload))
            
            # Calculate response time (for WebSocket, this is just the time to send)
            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)
            
            # Update metrics
            self._update_metrics(True, response_time_ms)
            
            return True
                
        except Exception as e:
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.datetime.now()
            self.metrics.error_count += 1
            self.logger.error(f"Failed to send message: {e}")
            
            # Check if connection was lost and try to reconnect
            if "connection is closed" in str(e).lower():
                self.connection_status = ConnectionStatus.DISCONNECTED
                self.logger.warning("WebSocket connection lost. Attempting to reconnect...")
                reconnect_success = await self.connect()
                if reconnect_success:
                    self.logger.info("Reconnected to WebSocket. Retrying message send...")
                    return await self.send_message(message)
            
            return False
    
    async def receive_message(self) -> Optional[AgentMessage]:
        """
        Receive a message from the agent.
        
        For WebSocket, this retrieves messages from the internal queue
        that is populated by the _receive_loop.
        """
        if not self.ws:
            self.logger.error("Cannot receive message: Not connected to WebSocket")
            return None
            
        try:
            # Get message from queue with a timeout
            message_data = await asyncio.wait_for(self.message_queue.get(), timeout=0.1)
            self.message_queue.task_done()
            
            # Convert to AgentMessage
            try:
                return AgentMessage(
                    id=message_data["id"],
                    type=MessageType(message_data["type"]),
                    content=message_data["content"],
                    sender_id=message_data["sender_id"],
                    recipient_id=message_data["recipient_id"],
                    timestamp=datetime.datetime.fromisoformat(message_data["timestamp"]),
                    correlation_id=message_data.get("correlation_id"),
                    priority=message_data.get("priority", 0),
                    expires_at=datetime.datetime.fromisoformat(message_data["expires_at"]) if message_data.get("expires_at") else None,
                    metadata=message_data.get("metadata", {})
                )
            except Exception as e:
                self.logger.error(f"Failed to parse message: {e}")
                return None
                
        except asyncio.TimeoutError:
            # No message available in the queue
            return None
        except Exception as e:
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.datetime.now()
            self.metrics.error_count += 1
            self.logger.error(f"Failed to receive message: {e}")
            return None
    
    async def get_capabilities(self) -> List[AgentCapability]:
        """
        Get the capabilities of the agent by sending a capability request message.
        """
        if not self.ws:
            self.logger.error("Cannot get capabilities: Not connected to WebSocket")
            return []
            
        try:
            # Create capability request message
            request_id = str(uuid.uuid4())
            capability_request = AgentMessage(
                id=request_id,
                type=MessageType.CAPABILITY_REQUEST,
                content={},
                sender_id="system",
                recipient_id="agent",
                timestamp=datetime.datetime.now()
            )
            
            # Send capability request
            await self.send_message(capability_request)
            
            # Wait for response with a timeout
            start_time = time.time()
            timeout = 10  # seconds
            while time.time() - start_time < timeout:
                message = await self.receive_message()
                
                if (message and 
                    message.type == MessageType.CAPABILITY_RESPONSE and 
                    message.correlation_id == request_id):
                    # Parse capabilities
                    capability_data = message.content.get("capabilities", [])
                    
                    # Convert to AgentCapability objects
                    capabilities = []
                    for cap_data in capability_data:
                        capabilities.append(AgentCapability(
                            id=cap_data["id"],
                            name=cap_data["name"],
                            description=cap_data["description"],
                            parameters=cap_data.get("parameters", {}),
                            example=cap_data.get("example"),
                            version=cap_data.get("version", "1.0"),
                            tags=cap_data.get("tags", [])
                        ))
                    
                    return capabilities
                
                # Short delay before checking again
                await asyncio.sleep(0.1)
            
            # Timeout reached, no response received
            self.metrics.last_error = "Timeout waiting for capability response"
            self.metrics.last_error_time = datetime.datetime.now()
            self.logger.error("Timeout waiting for capability response")
            return []
                
        except Exception as e:
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.datetime.now()
            self.metrics.error_count += 1
            self.logger.error(f"Failed to get capabilities: {e}")
            return []
            
    async def _heartbeat_loop(self) -> None:
        """
        Background task that sends periodic heartbeats to check the connection.
        """
        try:
            while True:
                # Wait for the next heartbeat interval
                await asyncio.sleep(self.heartbeat_interval)
                
                # Send heartbeat if connected
                if self.connection_status in [ConnectionStatus.CONNECTED, ConnectionStatus.DEGRADED]:
                    await self.send_heartbeat()
        except asyncio.CancelledError:
            # Task was canceled during disconnect
            self.logger.debug("Heartbeat loop canceled")
        except Exception as e:
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.datetime.now()
            self.logger.error(f"Error in heartbeat loop: {e}")
            
    async def _receive_loop(self) -> None:
        """
        Background task that receives messages from the WebSocket and puts them in the queue.
        """
        try:
            while True:
                if self.ws is None:
                    # Not connected, wait and try again
                    await asyncio.sleep(1)
                    continue
                    
                try:
                    # Wait for a message
                    message_raw = await self.ws.recv()
                    
                    # Parse message
                    message_data = json.loads(message_raw)
                    
                    # Put in queue
                    await self.message_queue.put(message_data)
                    
                except Exception as e:
                    self.metrics.last_error = str(e)
                    self.metrics.last_error_time = datetime.datetime.now()
                    self.metrics.error_count += 1
                    self.logger.error(f"Error receiving message: {e}")
                    
                    # If the connection was lost, try to reconnect
                    if "connection is closed" in str(e).lower():
                        self.connection_status = ConnectionStatus.DISCONNECTED
                        self.logger.warning("WebSocket connection lost in receive loop. Waiting for reconnect...")
                        await asyncio.sleep(5)  # Wait before trying again
                    else:
                        # Other error, short delay
                        await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            # Task was canceled during disconnect
            self.logger.debug("Receive loop canceled")
        except Exception as e:
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.datetime.now()
            self.logger.error(f"Error in receive loop: {e}")


class A2AProtocolAdapter(AgentProtocolAdapter):
    """
    Protocol adapter for the standard Agent-to-Agent (A2A) protocol.
    
    This adapter handles communication with agents that support the A2A protocol,
    which is the standard protocol used by the Adaptive Bridge Builder.
    """
    
    def __init__(
        self,
        protocol_config: ProtocolAdapter,
        a2a_handler: A2ATaskHandler,
        agent_id: str,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the A2A protocol adapter"""
        super().__init__(protocol_config, logger)
        self.a2a_handler = a2a_handler
        self.agent_id = agent_id
        self.capabilities_cache = None
        
    async def connect(self) -> bool:
        """
        Establish a connection to the agent using A2A protocol.
        
        For A2A, this mainly involves checking if the agent exists and is available.
        """
        try:
            self.connection_status = ConnectionStatus.CONNECTING
            self.logger.info(f"Connecting to A2A agent: {self.agent_id}")
            
            # Check if the agent exists and is available
            if await self.a2a_handler.check_agent_available(self.agent_id):
                self.connection_status = ConnectionStatus.CONNECTED
                self.metrics.uptime_seconds = int(time.time())
                self.logger.info(f"Successfully connected to A2A agent: {self.agent_id}")
                return True
            else:
                self.connection_status = ConnectionStatus.FAILED
                self.metrics.last_error = f"Agent not available: {self.agent_id}"
                self.metrics.last_error_time = datetime.datetime.now()
                self.logger.error(f"Failed to connect to A2A agent: {self.agent_id} (not available)")
                return False
                
        except Exception as e:
            self.connection_status = ConnectionStatus.FAILED
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.datetime.now()
            self.logger.error(f"Failed to connect to A2A agent: {e}")
            return False
            
    async def disconnect(self) -> bool:
        """
        Disconnect from the agent.
        
        For A2A, there's no persistent connection to close, so this just updates the status.
        """
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.logger.info(f"Disconnected from A2A agent: {self.agent_id}")
        return True
        
    async def send_message(self, message: AgentMessage) -> bool:
        """
        Send a message to the agent using the A2A protocol.
        
        Maps the message to the appropriate A2A method based on message type.
        """
        try:
            # Record metrics
            self.metrics.last_request_time = datetime.datetime.now()
            start_time = time.time()
            
            # Map message type to A2A method
            if message.type == MessageType.TASK_REQUEST:
                # For task requests, use the submitTask method
                task_content = message.content.copy()
                task_content.update({
                    "sender_id": message.sender_id,
                    "task_id": message.id
                })
                
                result = await self.a2a_handler.submit_task(
                    agent_id=self.agent_id,
                    task=task_content
                )
                
                success = bool(result)
                
            elif message.type == MessageType.INFORMATION_REQUEST:
                # For information requests, use the queryInfo method
                info_result = await self.a2a_handler.query_info(
                    agent_id=self.agent_id,
                    query=message.content.get("query", ""),
                    query_id=message.id
                )
                
                success = bool(info_result)
                
            elif message.type == MessageType.HEARTBEAT:
                # For heartbeats, check if the agent is available
                success = await self.a2a_handler.check_agent_available(self.agent_id)
                
            else:
                # For other message types, use the sendMessage method
                success = await self.a2a_handler.send_message(
                    recipient_id=self.agent_id,
                    message_content=message.content,
                    message_type=message.type.value,
                    message_id=message.id
                )
                
            # Calculate response time
            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)
            
            # Update metrics
            self._update_metrics(success, response_time_ms)
            
            if not success:
                self.metrics.last_error = "Failed to send message via A2A"
                self.metrics.last_error_time = datetime.datetime.now()
                self.logger.error(f"Failed to send message to A2A agent: {self.agent_id}")
            
            return success
                
        except Exception as e:
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.datetime.now()
            self.metrics.error_count += 1
            self.logger.error(f"Failed to send message to A2A agent: {e}")
            return False
            
    async def receive_message(self) -> Optional[AgentMessage]:
        """
        Receive a message from the agent using the A2A protocol.
        
        For A2A, this typically involves checking for pending messages.
        """
        try:
            # Record metrics
            start_time = time.time()
            
            # Check for pending messages
            pending_messages = await self.a2a_handler.get_pending_messages(
                agent_id=self.agent_id
            )
            
            # Calculate response time
            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)
            
            # Update metrics
            success = pending_messages is not None
            self._update_metrics(success, response_time_ms)
            
            if not success or not pending_messages:
                return None
                
            # Get the first pending message
            message_data = pending_messages[0]
            
            # Convert to AgentMessage
            try:
                return AgentMessage(
                    id=message_data.get("message_id", str(uuid.uuid4())),
                    type=MessageType(message_data.get("message_type", "notification")),
                    content=message_data.get("content", {}),
                    sender_id=message_data.get("sender_id", self.agent_id),
                    recipient_id=message_data.get("recipient_id", "system"),
                    timestamp=datetime.datetime.fromisoformat(message_data.get("timestamp", datetime.datetime.now().isoformat())),
                    correlation_id=message_data.get("correlation_id"),
                    priority=message_data.get("priority", 0),
                    expires_at=datetime.datetime.fromisoformat(message_data["expires_at"]) if message_data.get("expires_at") else None,
                    metadata=message_data.get("metadata", {})
                )
            except Exception as e:
                self.logger.error(f"Failed to parse A2A message: {e}")
                return None
                
        except Exception as e:
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.datetime.now()
            self.metrics.error_count += 1
            self.logger.error(f"Failed to receive message from A2A agent: {e}")
            return None
            
    async def get_capabilities(self) -> List[AgentCapability]:
        """
        Get the capabilities of the agent.
        
        For A2A, this involves querying the agent's capabilities.
        """
        # Use the cached capabilities if available
        if self.capabilities_cache:
            return self.capabilities_cache
            
        try:
            # Record metrics
            start_time = time.time()
            
            # Get capabilities
            capabilities_info = await self.a2a_handler.get_agent_capabilities(
                agent_id=self.agent_id
            )
            
            # Calculate response time
            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)
            
            # Update metrics
            success = capabilities_info is not None
            self._update_metrics(success, response_time_ms)
            
            if not success or not capabilities_info:
                return []
                
            # Convert to AgentCapability objects
            capabilities = []
            for cap_data in capabilities_info:
                capabilities.append(AgentCapability(
                    id=cap_data.get("id", str(uuid.uuid4())),
                    name=cap_data.get("name", "Unknown Capability"),
                    description=cap_data.get("description", ""),
                    parameters=cap_data.get("parameters", {}),
                    example=cap_data.get("example"),
                    version=cap_data.get("version", "1.0"),
                    tags=cap_data.get("tags", [])
                ))
            
            # Cache the capabilities
            self.capabilities_cache = capabilities
            
            return capabilities
                
        except Exception as e:
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.datetime.now()
            self.metrics.error_count += 1
            self.logger.error(f"Failed to get capabilities from A2A agent: {e}")
            return []
