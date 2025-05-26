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
from datetime import timezone
import enum
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, TypeVar, Generic, Type

# Import existing components that we will integrate with
from principle_engine import PrincipleEngine
from agent_registry import AgentRegistry
from agent_card import AgentCard
from a2a_task_handler import A2ATaskHandler
from communication_adapter import CommunicationAdapter
from security_privacy_manager import SecurityPrivacyManager
from session_manager import SessionManager


# Type variables for generic typing
T = TypeVar('T', bound=Dict[str, Any])  # Generic response type - currently unused
U = TypeVar('U', bound=Dict[str, Any])  # Generic request type - currently unused


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
                
                # Continue waiting for response
                await asyncio.sleep(0.1)
            
            # Timeout reached
            self.logger.warning(f"Timeout waiting for capability response")
            return []
            
        except Exception as e:
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.datetime.now()
            self.metrics.error_count += 1
            self.logger.error(f"Failed to get capabilities: {e}")
            return []
    
    async def _heartbeat_loop(self) -> None:
        """
        Background task that sends heartbeats at regular intervals.
        """
        while self.connection_status == ConnectionStatus.CONNECTED:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                await self.send_heartbeat()
            except asyncio.CancelledError:
                # Task was cancelled, exit gracefully
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                # Continue trying heartbeats even if one fails
    
    async def _receive_loop(self) -> None:
        """
        Background task that continuously receives messages from the WebSocket
        and puts them in the message queue.
        """
        while self.connection_status == ConnectionStatus.CONNECTED:
            try:
                # Receive message from WebSocket
                message_raw = await self.ws.recv()
                message_data = json.loads(message_raw)
                
                # Filter out heartbeat responses
                if message_data.get("type") != MessageType.HEARTBEAT.value:
                    # Add to queue
                    await self.message_queue.put(message_data)
                
            except asyncio.CancelledError:
                # Task was cancelled, exit gracefully
                break
            except Exception as e:
                self.logger.error(f"Error in receive loop: {e}")
                # Check if connection was lost
                if "connection is closed" in str(e).lower():
                    self.connection_status = ConnectionStatus.DISCONNECTED
                    break
                # Continue receiving even if one message fails


    async def cleanup(self) -> None:
        """Clean up background tasks."""
        if hasattr(self, '_background_tasks'):
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


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
# Create a class for managing connections
@dataclass
class Connection:
    """Represents a connection to an agent."""
    agent_id: str
    framework: AgentFramework
    protocol: ProtocolType
    adapter: AgentProtocolAdapter
    status: ConnectionStatus
    capabilities: List[AgentCapability] = field(default_factory=list)
    last_activity: Optional[datetime.datetime] = None
    created_at: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class UniversalAgentConnector:
    """
    Main class for connecting to agents regardless of their framework.
    
    This class manages connections to various types of agents and provides
    a unified interface for communication. It uses protocol adapters to handle
    different communication protocols and applies principle-driven adaptation
    to ensure compatibility.
    """
    
    def __init__(
        self,
        agent_id: str,
        principle_engine: Optional[PrincipleEngine] = None,
        security_manager: Optional[SecurityPrivacyManager] = None,
        session_manager: Optional[SessionManager] = None,
        agent_registry: Optional[AgentRegistry] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the UniversalAgentConnector.
        
        Args:
            agent_id: ID of the agent using this connector
            principle_engine: Optional PrincipleEngine for principle alignment
            security_manager: Optional SecurityPrivacyManager for security
            session_manager: Optional SessionManager for session management
            agent_registry: Optional AgentRegistry for agent discovery
            logger: Optional logger instance
        """
        self._background_tasks: List[asyncio.Task] = []
        self.agent_id = agent_id
        self.principle_engine = principle_engine or PrincipleEngine()
        self.security_manager = security_manager or SecurityPrivacyManager(agent_id)
        self.session_manager = session_manager or SessionManager()
        self.agent_registry = agent_registry or AgentRegistry()
        self.logger = logger or logging.getLogger(f"UniversalAgentConnector.{agent_id}")
        
        # Connection management
        self.connections: Dict[str, Connection] = {}
        self.adapters: Dict[ProtocolType, Type[AgentProtocolAdapter]] = {
            ProtocolType.REST: RestApiAdapter,
            ProtocolType.WEBSOCKET: WebSocketAdapter,
            # Add more adapters as implemented
        }
        
        # Monitoring
        self.start_time = datetime.datetime.now(timezone.utc)
        self.message_count = 0
        self.error_count = 0
        
        self.logger.info(f"UniversalAgentConnector initialized for agent {agent_id}")
    
    async def connect_to_agent(
        self,
        target_agent_id: str,
        framework: Optional[AgentFramework] = None,
        protocol: Optional[ProtocolType] = None,
        connection_params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Connect to another agent.
        
        Args:
            target_agent_id: ID of the agent to connect to
            framework: Optional framework of the target agent
            protocol: Optional protocol to use for connection
            connection_params: Optional connection parameters
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Check if already connected
            if target_agent_id in self.connections:
                existing_conn = self.connections[target_agent_id]
                if existing_conn.status == ConnectionStatus.CONNECTED:
                    self.logger.info(f"Already connected to agent {target_agent_id}")
                    return True
            
            # Get agent information from registry if not provided
            if not framework or not protocol:
                agent_info = await self._get_agent_info(target_agent_id)
                if not agent_info:
                    self.logger.error(f"Agent {target_agent_id} not found in registry")
                    return False
                    
                framework = framework or agent_info.get("framework", AgentFramework.CUSTOM)
                protocol = protocol or agent_info.get("protocol", ProtocolType.A2A)
                connection_params = connection_params or agent_info.get("connection_params", {})
            
            # Select appropriate adapter
            adapter_class = self.adapters.get(protocol)
            if not adapter_class:
                self.logger.error(f"No adapter available for protocol {protocol}")
                return False
            
            # Create protocol configuration
            protocol_config = ProtocolAdapter(
                protocol_type=protocol,
                config=connection_params.get("config", {}),
                requires_auth=connection_params.get("requires_auth", False),
                supports_streaming=connection_params.get("supports_streaming", False),
                supports_batch=connection_params.get("supports_batch", False),
                supports_sync=connection_params.get("supports_sync", True),
                supports_async=connection_params.get("supports_async", True)
            )
            
            # Create adapter instance
            adapter = self._create_adapter(adapter_class, protocol_config, connection_params)
            
            # Connect using the adapter
            success = await adapter.connect()
            
            if success:
                # Get capabilities
                capabilities = await adapter.get_capabilities()
                
                # Create connection record
                connection = Connection(
                    agent_id=target_agent_id,
                    framework=framework,
                    protocol=protocol,
                    adapter=adapter,
                    status=ConnectionStatus.CONNECTED,
                    capabilities=capabilities,
                    last_activity=datetime.datetime.now(timezone.utc),
                    metadata=connection_params.get("metadata", {})
                )
                
                self.connections = {**self.connections, target_agent_id: connection}
                self.logger.info(f"Successfully connected to agent {target_agent_id} using {protocol.value} protocol")
                
                # Start monitoring
                monitor_task = asyncio.create_task(self._monitor_connection(target_agent_id))
                
                return True
            else:
                self.logger.error(f"Failed to connect to agent {target_agent_id}")
                return False
                
        except Exception as e:
            self.error_count = self.error_count + 1
            self.logger.error(f"Error connecting to agent {target_agent_id}: {e}")
            return False
    
    async def disconnect_from_agent(self, target_agent_id: str) -> bool:
        """
        Disconnect from an agent.
        
        Args:
            target_agent_id: ID of the agent to disconnect from
            
        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            if target_agent_id not in self.connections:
                self.logger.warning(f"Not connected to agent {target_agent_id}")
                return True
                
            connection = self.connections[target_agent_id]
            
            # Disconnect using the adapter
            success = await connection.adapter.disconnect()
            
            if success:
                # Remove connection record
                self.connections = {k: v for k, v in self.connections.items() if k != target_agent_id}
                self.logger.info(f"Disconnected from agent {target_agent_id}")
                return True
            else:
                self.logger.error(f"Failed to disconnect from agent {target_agent_id}")
                return False
                
        except Exception as e:
            self.error_count = self.error_count + 1
            self.logger.error(f"Error disconnecting from agent {target_agent_id}: {e}")
            return False
    
    async def send_message(
        self,
        target_agent_id: str,
        message_type: MessageType,
        content: Dict[str, Any],
        priority: int = 0,
        correlation_id: Optional[str] = None,
        timeout_seconds: Optional[int] = None
    ) -> Optional[AgentMessage]:
        """
        Send a message to an agent.
        
        Args:
            target_agent_id: ID of the agent to send message to
            message_type: Type of message
            content: Message content
            priority: Message priority (0 = normal, higher = more important)
            correlation_id: Optional correlation ID for tracking
            timeout_seconds: Optional timeout for waiting for response
            
        Returns:
            Response message if applicable, None otherwise
        """
        try:
            # Check connection
            if target_agent_id not in self.connections:
                self.logger.error(f"Not connected to agent {target_agent_id}")
                return None
                
            connection = self.connections[target_agent_id]
            if connection.status != ConnectionStatus.CONNECTED:
                self.logger.error(f"Connection to agent {target_agent_id} is not active")
                return None
            
            # Apply security checks
            if self.security_manager:
                security_check = await self.security_manager.check_message_security(content)
                if not security_check.get("allowed", False):
                    self.logger.error(f"Security check failed: {security_check.get('reason', 'Unknown')}")
                    return None
            
            # Apply principle alignment
            if self.principle_engine:
                aligned_content = self.principle_engine.align_message(content)
            else:
                aligned_content = content
            
            # Create message
            message = AgentMessage(
                id=str(uuid.uuid4()),
                type=message_type,
                content=aligned_content,
                sender_id=self.agent_id,
                recipient_id=target_agent_id,
                timestamp=datetime.datetime.now(timezone.utc),
                correlation_id=correlation_id,
                priority=priority,
                expires_at=datetime.datetime.now(timezone.utc) + datetime.timedelta(seconds=timeout_seconds) if timeout_seconds else None
            )
            
            # Send message
            send_success = await connection.adapter.send_message(message)
            
            if send_success:
                self.message_count = self.message_count + 1
                connection.last_activity = datetime.datetime.now(timezone.utc)
                
                # Wait for response if it's a request type
                if message_type in [MessageType.TASK_REQUEST, MessageType.INFORMATION_REQUEST, MessageType.CAPABILITY_REQUEST]:
                    response = await self._wait_for_response(
                        target_agent_id,
                        message.id,
                        timeout_seconds or 30
                    )
                    return response
                else:
                    return message
            else:
                self.error_count = self.error_count + 1
                self.logger.error(f"Failed to send message to agent {target_agent_id}")
                return None
                
        except Exception as e:
            self.error_count = self.error_count + 1
            self.logger.error(f"Error sending message to agent {target_agent_id}: {e}")
            return None
    
    async def receive_messages(self, timeout_seconds: float = 0.1) -> List[AgentMessage]:
        """
        Receive messages from all connected agents.
        
        Args:
            timeout_seconds: Timeout for receiving messages
            
        Returns:
            List of received messages
        """
        messages = []
        
        # Check each connection for messages
        for agent_id, connection in self.connections.items():
            if connection.status != ConnectionStatus.CONNECTED:
                continue
                
            try:
                # Set a short timeout for each connection
                start_time = time.time()
                while time.time() - start_time < timeout_seconds:
                    message = await connection.adapter.receive_message()
                    
                    if message:
                        # Apply security checks
                        if self.security_manager:
                            security_check = await self.security_manager.check_message_security(message.content)
                            if not security_check.get("allowed", False):
                                self.logger.warning(f"Received message failed security check: {security_check.get('reason', 'Unknown')}")
                                continue
                        
                        messages.append(message)
                        connection.last_activity = datetime.datetime.now(timezone.utc)
                    else:
                        # No more messages from this agent
                        break
                        
            except Exception as e:
                self.error_count = self.error_count + 1
                self.logger.error(f"Error receiving message from agent {agent_id}: {e}")
        
        return messages
    
    def get_connected_agents(self) -> List[Dict[str, Any]]:
        """
        Get information about all connected agents.
        
        Returns:
            List of connected agent information
        """
        connected_agents = []
        
        for agent_id, connection in self.connections.items():
            agent_info = {
                "agent_id": agent_id,
                "framework": connection.framework.value,
                "protocol": connection.protocol.value,
                "status": connection.status.value,
                "capabilities": [
                    {
                        "id": cap.id,
                        "name": cap.name,
                        "description": cap.description
                    }
                    for cap in connection.capabilities
                ],
                "last_activity": connection.last_activity.isoformat() if connection.last_activity else None,
                "connected_since": connection.created_at.isoformat(),
                "metrics": connection.adapter.get_metrics().__dict__ if hasattr(connection.adapter, 'get_metrics') else {}
            }
            connected_agents.append(agent_info)
        
        return connected_agents
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the connector.
        
        Returns:
            Dictionary of statistics
        """
        uptime_seconds = (datetime.datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        stats = {
            "agent_id": self.agent_id,
            "uptime_seconds": int(uptime_seconds),
            "connected_agents": len([c for c in self.connections.values() if c.status == ConnectionStatus.CONNECTED]),
            "total_connections": len(self.connections),
            "messages_sent": self.message_count,
            "errors": self.error_count,
            "error_rate": self.error_count / max(1, self.message_count),
            "connections_by_framework": {},
            "connections_by_protocol": {},
            "connections_by_status": {}
        }
        
        # Count connections by framework, protocol, and status
        for connection in self.connections.values():
            # By framework
            framework_key = connection.framework.value
            stats["connections_by_framework"][framework_key] = stats["connections_by_framework"].get(framework_key, 0) + 1
            
            # By protocol
            protocol_key = connection.protocol.value
            stats["connections_by_protocol"][protocol_key] = stats["connections_by_protocol"].get(protocol_key, 0) + 1
            
            # By status
            status_key = connection.status.value
            stats["connections_by_status"][status_key] = stats["connections_by_status"].get(status_key, 0) + 1
        
        return stats
    
    async def _get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get agent information from the registry.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Agent information or None if not found
        """
        try:
            # Try to get from registry
            agent_card = self.agent_registry.get_agent(agent_id)
            
            if agent_card:
                return {
                    "framework": AgentFramework.A2A,  # Default to A2A
                    "protocol": ProtocolType.A2A,      # Default to A2A
                    "connection_params": {
                        "config": agent_card.get("connection", {}),
                        "requires_auth": agent_card.get("requires_auth", False),
                        "supports_streaming": agent_card.get("supports_streaming", False),
                        "supports_batch": agent_card.get("supports_batch", False)
                    }
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting agent info: {e}")
            return None
    
    def _create_adapter(
        self,
        adapter_class: Type[AgentProtocolAdapter],
        protocol_config: ProtocolAdapter,
        connection_params: Dict[str, Any]
    ) -> AgentProtocolAdapter:
        """
        Create an adapter instance with appropriate parameters.
        
        Args:
            adapter_class: Adapter class to instantiate
            protocol_config: Protocol configuration
            connection_params: Connection parameters
            
        Returns:
            Adapter instance
        """
        # Extract common parameters
        base_params = {
            "protocol_config": protocol_config,
            "logger": self.logger
        }
        
        # Add adapter-specific parameters
        if adapter_class == RestApiAdapter:
            base_params.update({
                "base_url": connection_params.get("base_url", "http://localhost:8080"),
                "headers": connection_params.get("headers"),
                "auth": connection_params.get("auth"),
                "timeout_seconds": connection_params.get("timeout_seconds", 30)
            })
        elif adapter_class == WebSocketAdapter:
            base_params.update({
                "websocket_url": connection_params.get("websocket_url", "ws://localhost:8080"),
                "headers": connection_params.get("headers"),
                "auth": connection_params.get("auth"),
                "heartbeat_interval_seconds": connection_params.get("heartbeat_interval_seconds", 30),
                "reconnect_attempts": connection_params.get("reconnect_attempts", 3)
            })
        
        return adapter_class(**base_params)
    
    async def _wait_for_response(
        self,
        agent_id: str,
        request_id: str,
        timeout_seconds: int
    ) -> Optional[AgentMessage]:
        """
        Wait for a response message with a specific correlation ID.
        
        Args:
            agent_id: ID of the agent to receive from
            request_id: Request ID to match in correlation_id
            timeout_seconds: Timeout in seconds
            
        Returns:
            Response message or None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            # Check for messages
            messages = await self.receive_messages(timeout_seconds=0.1)
            
            # Look for matching response
            for message in messages:
                if (message.sender_id == agent_id and 
                    message.correlation_id == request_id and
                    message.type in [MessageType.TASK_RESPONSE, MessageType.INFORMATION_RESPONSE, MessageType.CAPABILITY_RESPONSE]):
                    return message
            
            # Short sleep before next check
            await asyncio.sleep(0.1)
        
        self.logger.warning(f"Timeout waiting for response from agent {agent_id} (request_id={request_id})")
        return None
    
    async def _monitor_connection(self, agent_id: str) -> None:
        """
        Monitor a connection and send heartbeats.
        
        Args:
            agent_id: ID of the agent to monitor
        """
        while agent_id in self.connections:
            try:
                connection = self.connections.get(agent_id)
                if not connection:
                    break
                    
                # Send heartbeat
                if connection.status == ConnectionStatus.CONNECTED:
                    success = await connection.adapter.send_heartbeat()
                    
                    if not success:
                        # Mark as degraded
                        connection.status = ConnectionStatus.DEGRADED
                        self.logger.warning(f"Connection to agent {agent_id} degraded")
                
                # Wait before next heartbeat
                await asyncio.sleep(30)  # 30 seconds between heartbeats
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error monitoring connection to agent {agent_id}: {e}")
                await asyncio.sleep(30)  # Continue monitoring after error


    async def cleanup(self) -> None:
        """Clean up background tasks."""
        if hasattr(self, '_background_tasks'):
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


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
# Export main classes
__all__ = ["AgentFramework", "UniversalAgentConnector", "ProtocolType", "MessageType", 
          "ConnectionStatus", "AgentCapability", "AgentMessage", "Connection",
          "ConnectionMetrics", "ProtocolAdapter", "AgentProtocolAdapter",
          "RestApiAdapter", "WebSocketAdapter"]