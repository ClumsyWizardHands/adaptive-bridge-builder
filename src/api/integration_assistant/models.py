"""
Data Models for Integration Assistant API

Defines Pydantic models for API requests and responses.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from enum import Enum


class AgentFramework(str, Enum):
    """Supported agent frameworks"""
    LANGCHAIN = "langchain"
    AUTOGPT = "autogpt"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    LLAMAINDEX = "llamaindex"
    CREWAI = "crewai"
    AUTOGEN = "autogen"
    SUPERAGI = "superagi"
    BABYAGI = "babyagi"
    METAGPT = "metagpt"
    AGENTGPT = "agentgpt"
    A2A_PROTOCOL = "a2a_protocol"
    CUSTOM = "custom"


class ConnectionState(str, Enum):
    """WebSocket connection states"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    TESTING = "testing"


class AgentCapability(BaseModel):
    """Agent capability definition"""
    name: str = Field(..., description="Capability name")
    description: str = Field(..., description="Capability description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters schema")
    required: bool = Field(default=False, description="Whether this capability is required")


class AgentRegistration(BaseModel):
    """Agent registration request model"""
    name: str = Field(..., description="Agent name")
    framework: AgentFramework = Field(..., description="Agent framework type")
    version: str = Field(..., description="Agent version")
    description: Optional[str] = Field(None, description="Agent description")
    capabilities: List[AgentCapability] = Field(default_factory=list, description="Agent capabilities")
    endpoint_url: Optional[str] = Field(None, description="Agent endpoint URL")
    authentication: Optional[Dict[str, Any]] = Field(None, description="Authentication configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AgentRegistrationResponse(BaseModel):
    """Agent registration response model"""
    agent_id: str = Field(..., description="Unique agent identifier")
    registration_time: datetime = Field(..., description="Registration timestamp")
    status: str = Field(..., description="Registration status")
    connection_token: str = Field(..., description="Token for WebSocket connections")
    integration_code: Optional[str] = Field(None, description="Initial integration code")


class ConnectionStatus(BaseModel):
    """WebSocket connection status"""
    client_id: str = Field(..., description="Client identifier")
    agent_id: Optional[str] = Field(None, description="Associated agent ID")
    state: ConnectionState = Field(..., description="Connection state")
    connected_at: Optional[datetime] = Field(None, description="Connection timestamp")
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")
    error_message: Optional[str] = Field(None, description="Error message if any")


class IntegrationCodeRequest(BaseModel):
    """Request for integration code generation"""
    agent_id: str = Field(..., description="Agent identifier")
    framework: AgentFramework = Field(..., description="Target framework")
    language: str = Field(default="python", description="Programming language")
    include_examples: bool = Field(default=True, description="Include usage examples")
    custom_config: Optional[Dict[str, Any]] = Field(None, description="Custom configuration")


class IntegrationCodeResponse(BaseModel):
    """Integration code response"""
    agent_id: str = Field(..., description="Agent identifier")
    framework: AgentFramework = Field(..., description="Target framework")
    language: str = Field(..., description="Programming language")
    code: str = Field(..., description="Generated integration code")
    imports: List[str] = Field(default_factory=list, description="Required imports")
    dependencies: List[str] = Field(default_factory=list, description="Required dependencies")
    examples: Optional[List[str]] = Field(None, description="Usage examples")
    documentation_url: Optional[str] = Field(None, description="Documentation URL")


class ConnectionTestRequest(BaseModel):
    """Connection test request"""
    agent_id: str = Field(..., description="Agent identifier")
    test_payload: Optional[Dict[str, Any]] = Field(None, description="Test payload")
    timeout: int = Field(default=30, description="Test timeout in seconds")


class ConnectionTestResponse(BaseModel):
    """Connection test response"""
    agent_id: str = Field(..., description="Agent identifier")
    success: bool = Field(..., description="Test success status")
    response_time_ms: float = Field(..., description="Response time in milliseconds")
    test_results: Dict[str, Any] = Field(default_factory=dict, description="Detailed test results")
    error_message: Optional[str] = Field(None, description="Error message if test failed")


class WebSocketMessage(BaseModel):
    """WebSocket message structure"""
    type: str = Field(..., description="Message type")
    client_id: str = Field(..., description="Client identifier")
    agent_id: Optional[str] = Field(None, description="Agent identifier")
    data: Dict[str, Any] = Field(default_factory=dict, description="Message data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
