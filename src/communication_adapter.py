"""
Communication Adapter

This module provides adaptation layers for formatting messages based on receiving agents' capabilities.
It handles different communication protocols, message transformations, and capability-aware formatting.
"""

import json
import logging
import re
import base64
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set, Callable

from content_handler import ContentHandler, ContentFormat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CommunicationAdapter")


class AgentCapability(str, Enum):
    """Capabilities that agents may have for communication."""
    STRUCTURED_DATA = "structured_data"     # Can handle JSON, XML, etc.
    BINARY_DATA = "binary_data"             # Can handle binary attachments
    MARKDOWN = "markdown"                   # Can handle Markdown formatting
    HTML = "html"                           # Can handle HTML content
    CODE_SNIPPETS = "code_snippets"         # Can handle code blocks
    COMPRESSED_DATA = "compressed_data"     # Can handle compressed content
    STREAMING = "streaming"                 # Can receive streaming responses
    INTERRUPTION = "interruption"           # Can be interrupted during response
    MULTI_MODAL = "multi_modal"             # Can handle multiple content types
    SECURE_CHANNEL = "secure_channel"       # Can use encrypted communications


class CommunicationProtocol(str, Enum):
    """Communication protocols supported by the adapter."""
    DIRECT = "direct"                       # Direct message exchange
    REST_API = "rest_api"                   # RESTful API communication
    WEBSOCKET = "websocket"                 # WebSocket communication
    EVENT_BASED = "event_based"             # Event-based communication
    MESSAGE_QUEUE = "message_queue"         # Message queue communication
    RPC = "rpc"                             # Remote procedure call
    GRAPHQL = "graphql"                     # GraphQL API communication


class MessageTransformation(str, Enum):
    """Transformations that can be applied to messages."""
    NONE = "none"                           # No transformation
    SIMPLIFY = "simplify"                   # Simplify complex content
    STRUCTURE = "structure"                 # Add structure to unstructured content
    COMPRESS = "compress"                   # Compress content
    ENCRYPT = "encrypt"                     # Encrypt content
    CHUNK = "chunk"                         # Break into smaller chunks
    SUMMARIZE = "summarize"                 # Create a summary
    FORMAT_CONVERT = "format_convert"       # Convert between formats


class AgentProfile:
    """
    Represents the communication profile of an agent.
    
    Contains information about an agent's capabilities, preferred formats,
    and communication constraints.
    """
    
    def __init__(
        self,
        agent_id: str,
        capabilities: Optional[List[AgentCapability]] = None,
        preferred_format: Optional[ContentFormat] = None,
        max_message_size: Optional[int] = None,
        protocol: CommunicationProtocol = CommunicationProtocol.DIRECT,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new agent profile.
        
        Args:
            agent_id: ID of the agent
            capabilities: List of communication capabilities
            preferred_format: Preferred content format
            max_message_size: Maximum message size in bytes
            protocol: Communication protocol
            metadata: Additional metadata about the agent
        """
        self.agent_id = agent_id
        self.capabilities = set(capabilities or [])
        self.preferred_format = preferred_format or ContentFormat.TEXT
        self.max_message_size = max_message_size
        self.protocol = protocol
        self.metadata = metadata or {}
        
        logger.debug(f"Created agent profile for {agent_id}")
    
    def has_capability(self, capability: AgentCapability) -> bool:
        """
        Check if the agent has a specific capability.
        
        Args:
            capability: The capability to check
            
        Returns:
            Whether the agent has the capability
        """
        return capability in self.capabilities
    
    def can_handle_format(self, format_type: ContentFormat) -> bool:
        """
        Check if the agent can handle a specific content format.
        
        Args:
            format_type: The content format
            
        Returns:
            Whether the agent can handle the format
        """
        if format_type == self.preferred_format:
            return True
            
        format_capability_map = {
            ContentFormat.JSON: AgentCapability.STRUCTURED_DATA,
            ContentFormat.XML: AgentCapability.STRUCTURED_DATA,
            ContentFormat.HTML: AgentCapability.HTML,
            ContentFormat.MARKDOWN: AgentCapability.MARKDOWN,
            ContentFormat.BINARY: AgentCapability.BINARY_DATA,
            ContentFormat.BASE64: AgentCapability.BINARY_DATA
        }
        
        required_capability = format_capability_map.get(format_type)
        if required_capability:
            return self.has_capability(required_capability)
            
        # Default to true for TEXT format, assuming all agents can handle plain text
        return format_type == ContentFormat.TEXT
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the profile to a dictionary.
        
        Returns:
            Dictionary representation of the profile
        """
        return {
            "agent_id": self.agent_id,
            "capabilities": list(self.capabilities),
            "preferred_format": self.preferred_format.value,
            "max_message_size": self.max_message_size,
            "protocol": self.protocol.value,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentProfile':
        """
        Create an AgentProfile from a dictionary.
        
        Args:
            data: Dictionary containing profile data
            
        Returns:
            AgentProfile object
        """
        capabilities = [AgentCapability(cap) for cap in data.get("capabilities", [])]
        preferred_format = ContentFormat(data.get("preferred_format", ContentFormat.TEXT.value))
        
        return cls(
            agent_id=data["agent_id"],
            capabilities=capabilities,
            preferred_format=preferred_format,
            max_message_size=data.get("max_message_size"),
            protocol=CommunicationProtocol(data.get("protocol", CommunicationProtocol.DIRECT.value)),
            metadata=data.get("metadata", {})
        )


class CommunicationAdapter:
    """
    Adapts communication between agents with different capabilities.
    
    This class handles:
    1. Formatting messages based on receiving agent capabilities
    2. Converting content between different formats
    3. Transforming messages to meet constraints
    4. Managing communication protocols
    """
    
    def __init__(
        self,
        agent_id: str,
        content_handler: Optional[ContentHandler] = None
    ):
        """
        Initialize the communication adapter.
        
        Args:
            agent_id: ID of the agent using this adapter
            content_handler: Handler for content format conversion
        """
        self.agent_id = agent_id
        self.content_handler = content_handler or ContentHandler()
        
        # Agent profiles cache
        self.agent_profiles: Dict[str, AgentProfile] = {}
        
        # Protocol handlers
        self.protocol_handlers: Dict[CommunicationProtocol, Callable] = {
            CommunicationProtocol.DIRECT: self._handle_direct_communication,
            CommunicationProtocol.REST_API: self._handle_rest_api_communication,
            CommunicationProtocol.WEBSOCKET: self._handle_websocket_communication,
            CommunicationProtocol.EVENT_BASED: self._handle_event_based_communication,
            CommunicationProtocol.MESSAGE_QUEUE: self._handle_message_queue_communication,
            CommunicationProtocol.RPC: self._handle_rpc_communication,
            CommunicationProtocol.GRAPHQL: self._handle_graphql_communication
        }
        
        logger.info(f"CommunicationAdapter initialized for agent {agent_id}")
    
    def register_agent_profile(self, profile: AgentProfile) -> None:
        """
        Register or update an agent's profile.
        
        Args:
            profile: The agent profile to register
        """
        self.agent_profiles[profile.agent_id] = profile
        logger.info(f"Registered profile for agent {profile.agent_id}")
    
    def get_agent_profile(self, agent_id: str) -> Optional[AgentProfile]:
        """
        Get an agent's profile.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Agent profile if found, None otherwise
        """
        return self.agent_profiles.get(agent_id)
    
    def adapt_message(
        self,
        message: Any,
        recipient_id: str,
        transformations: Optional[List[MessageTransformation]] = None
    ) -> Dict[str, Any]:
        """
        Adapt a message for a specific recipient agent.
        
        Args:
            message: The message to adapt
            recipient_id: ID of the recipient agent
            transformations: Optional list of transformations to apply
            
        Returns:
            Adapted message
        """
        # Get recipient profile
        recipient_profile = self.get_agent_profile(recipient_id)
        
        if not recipient_profile:
            # Use default settings if profile not found
            logger.warning(f"No profile found for agent {recipient_id}, using default settings")
            recipient_profile = AgentProfile(agent_id=recipient_id)
            
        # Extract content and format
        if isinstance(message, dict):
            # Message is already structured
            structured_message = message
            if "content" not in structured_message:
                # Wrap content if needed
                structured_message = {"content": message}
        else:
            # Simple content, wrap in a message object
            structured_message = {"content": message}
            
        # Determine content format
        if "format" in structured_message:
            try:
                current_format = ContentFormat(structured_message["format"])
            except ValueError:
                current_format = self.content_handler.detect_format(structured_message["content"])
        else:
            current_format = self.content_handler.detect_format(structured_message["content"])
            
        # Get preferred format for recipient
        target_format = recipient_profile.preferred_format
        
        # Convert format if needed
        if current_format != target_format and recipient_profile.can_handle_format(target_format):
            content = structured_message.get("content")
            converted_content, success = self.content_handler.convert_content(
                content=content,
                from_format=current_format,
                to_format=target_format
            )
            
            if success:
                structured_message["content"] = converted_content
                structured_message["format"] = target_format.value
                logger.debug(f"Converted content from {current_format.value} to {target_format.value}")
            else:
                logger.warning(
                    f"Failed to convert from {current_format.value} to {target_format.value}, "
                    "keeping original format"
                )
                
        # Apply transformations
        if transformations:
            for transformation in transformations:
                structured_message = self._apply_transformation(
                    message=structured_message,
                    transformation=transformation,
                    recipient_profile=recipient_profile
                )
                
        # Apply protocol-specific formatting
        protocol = recipient_profile.protocol
        protocol_handler = self.protocol_handlers.get(protocol)
        
        if protocol_handler:
            structured_message = protocol_handler(structured_message, recipient_profile)
            
        # Add metadata
        structured_message["sender_id"] = self.agent_id
        structured_message["recipient_id"] = recipient_id
        structured_message["adapted"] = True
        
        return structured_message
    
    def check_compatibility(self, agent_id_1: str, agent_id_2: str) -> Dict[str, Any]:
        """
        Check compatibility between two agents.
        
        Args:
            agent_id_1: ID of the first agent
            agent_id_2: ID of the second agent
            
        Returns:
            Compatibility report
        """
        profile1 = self.get_agent_profile(agent_id_1)
        profile2 = self.get_agent_profile(agent_id_2)
        
        if not profile1 or not profile2:
            missing = []
            if not profile1:
                missing.append(agent_id_1)
            if not profile2:
                missing.append(agent_id_2)
                
            return {
                "compatible": False,
                "error": f"Missing profiles for agents: {', '.join(missing)}"
            }
            
        # Check for common capabilities
        common_capabilities = profile1.capabilities.intersection(profile2.capabilities)
        
        # Check format compatibility
        format1 = profile1.preferred_format
        format2 = profile2.preferred_format
        format_compatible = format1 == format2 or profile1.can_handle_format(format2) or profile2.can_handle_format(format1)
        
        # Check protocol compatibility
        protocol_compatible = profile1.protocol == profile2.protocol
        
        # Check size constraints
        size_compatible = True
        size_constraints = []
        
        if profile1.max_message_size and profile2.max_message_size:
            if profile1.max_message_size < profile2.max_message_size:
                size_constraints.append(f"{agent_id_1} has smaller message size limit than {agent_id_2}")
            elif profile2.max_message_size < profile1.max_message_size:
                size_constraints.append(f"{agent_id_2} has smaller message size limit than {agent_id_1}")
                
        # Determine overall compatibility
        compatible = bool(common_capabilities) and format_compatible
        
        return {
            "compatible": compatible,
            "common_capabilities": list(common_capabilities),
            "format_compatible": format_compatible,
            "protocol_compatible": protocol_compatible,
            "size_compatible": size_compatible,
            "size_constraints": size_constraints,
            "recommended_format": self._recommend_format(profile1, profile2).value,
            "adaptation_needed": not (format_compatible and protocol_compatible)
        }
    
    def can_handle_binary(self, agent_id: str) -> bool:
        """
        Check if an agent can handle binary data.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Whether the agent can handle binary data
        """
        profile = self.get_agent_profile(agent_id)
        if not profile:
            return False
            
        return AgentCapability.BINARY_DATA in profile.capabilities
    
    def can_handle_structured_data(self, agent_id: str) -> bool:
        """
        Check if an agent can handle structured data.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Whether the agent can handle structured data
        """
        profile = self.get_agent_profile(agent_id)
        if not profile:
            return False
            
        return AgentCapability.STRUCTURED_DATA in profile.capabilities
    
    def get_optimal_agents_for_task(
        self,
        required_capabilities: List[AgentCapability],
        count: int = 1
    ) -> List[str]:
        """
        Find the optimal agents for a task based on required capabilities.
        
        Args:
            required_capabilities: Capabilities required for the task
            count: Number of agents to return
            
        Returns:
            List of agent IDs
        """
        if not required_capabilities:
            # No specific requirements, return any agents up to count
            return list(self.agent_profiles.keys())[:count]
            
        # Score agents based on capabilities
        agent_scores = []
        for agent_id, profile in self.agent_profiles.items():
            # Count how many required capabilities this agent has
            matches = sum(1 for cap in required_capabilities if cap in profile.capabilities)
            if matches > 0:
                agent_scores.append((agent_id, matches))
                
        # Sort by score (highest first)
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top agents
        return [agent_id for agent_id, _ in agent_scores[:count]]
    
    def simplify_for_limited_agent(
        self,
        content: Any,
        recipient_id: str
    ) -> Any:
        """
        Simplify content for an agent with limited capabilities.
        
        Args:
            content: The content to simplify
            recipient_id: ID of the recipient agent
            
        Returns:
            Simplified content
        """
        profile = self.get_agent_profile(recipient_id)
        if not profile:
            return content
            
        # Determine content format
        format_type = self.content_handler.detect_format(content)
        
        # Choose simplification level based on agent capabilities
        if len(profile.capabilities) >= 3:
            # More capable agent, minimal simplification
            simplification_level = 3
        elif len(profile.capabilities) >= 1:
            # Moderately capable agent
            simplification_level = 2
        else:
            # Very limited agent
            simplification_level = 1
            
        # Simplify content based on its format
        return self.content_handler.simplify_content(
            content=content,
            format_type=format_type,
            target_complexity=simplification_level
        )
    
    def generate_compatibility_matrix(self, agent_ids: List[str]) -> Dict[str, Any]:
        """
        Generate a compatibility matrix for a group of agents.
        
        Args:
            agent_ids: List of agent IDs to include
            
        Returns:
            Compatibility matrix
        """
        matrix = {}
        
        for i, agent1 in enumerate(agent_ids):
            matrix[agent1] = {}
            for agent2 in agent_ids[i+1:]:
                compatibility = self.check_compatibility(agent1, agent2)
                matrix[agent1][agent2] = compatibility
                
        return matrix
    
    # --- Protocol Handlers ---
    
    def _handle_direct_communication(
        self,
        message: Dict[str, Any],
        recipient_profile: AgentProfile
    ) -> Dict[str, Any]:
        """Handle direct communication protocol."""
        # Direct communication needs no special handling
        return message
    
    def _handle_rest_api_communication(
        self,
        message: Dict[str, Any],
        recipient_profile: AgentProfile
    ) -> Dict[str, Any]:
        """Format message for REST API communication."""
        # Structure message as a REST API payload
        formatted = {
            "data": message.get("content"),
            "metadata": {
                "sender": self.agent_id,
                "timestamp": message.get("timestamp", self._get_timestamp()),
                "format": message.get("format", "text")
            }
        }
        
        # Add any headers from recipient metadata
        if "api_headers" in recipient_profile.metadata:
            formatted["headers"] = recipient_profile.metadata["api_headers"]
            
        # Add API key if needed
        if "api_key" in recipient_profile.metadata:
            formatted["auth"] = {
                "api_key": recipient_profile.metadata["api_key"]
            }
            
        return formatted
    
    def _handle_websocket_communication(
        self,
        message: Dict[str, Any],
        recipient_profile: AgentProfile
    ) -> Dict[str, Any]:
        """Format message for WebSocket communication."""
        # Structure message as a WebSocket event
        formatted = {
            "event": "message",
            "data": message.get("content"),
            "sender": self.agent_id,
            "recipient": recipient_profile.agent_id,
            "timestamp": message.get("timestamp", self._get_timestamp())
        }
        
        return formatted
    
    def _handle_event_based_communication(
        self,
        message: Dict[str, Any],
        recipient_profile: AgentProfile
    ) -> Dict[str, Any]:
        """Format message for event-based communication."""
        # Structure message as an event
        event_type = message.get("event_type", "default")
        
        formatted = {
            "event_type": event_type,
            "payload": message.get("content"),
            "metadata": {
                "sender": self.agent_id,
                "recipient": recipient_profile.agent_id,
                "timestamp": message.get("timestamp", self._get_timestamp()),
                "format": message.get("format", "text")
            }
        }
        
        return formatted
    
    def _handle_message_queue_communication(
        self,
        message: Dict[str, Any],
        recipient_profile: AgentProfile
    ) -> Dict[str, Any]:
        """Format message for message queue communication."""
        # Structure message for a message queue
        formatted = {
            "type": "message",
            "payload": message.get("content"),
            "properties": {
                "sender": self.agent_id,
                "recipient": recipient_profile.agent_id,
                "timestamp": message.get("timestamp", self._get_timestamp()),
                "format": message.get("format", "text"),
                "content_type": message.get("content_type", "text/plain")
            }
        }
        
        # Add routing information if available
        if "queue" in recipient_profile.metadata:
            formatted["queue"] = recipient_profile.metadata["queue"]
            
        return formatted
    
    def _handle_rpc_communication(
        self,
        message: Dict[str, Any],
        recipient_profile: AgentProfile
    ) -> Dict[str, Any]:
        """Format message for RPC communication."""
        # Structure message as an RPC call
        method = message.get("method", "process")
        
        formatted = {
            "jsonrpc": "2.0",
            "method": method,
            "params": {
                "content": message.get("content"),
                "sender": self.agent_id,
                "format": message.get("format", "text")
            },
            "id": message.get("id", self._generate_id())
        }
        
        return formatted
    
    def _handle_graphql_communication(
        self,
        message: Dict[str, Any],
        recipient_profile: AgentProfile
    ) -> Dict[str, Any]:
        """Format message for GraphQL communication."""
        # Structure message as a GraphQL operation
        operation = message.get("operation", "sendMessage")
        
        # Basic GraphQL query structure
        formatted = {
            "query": f"""
                mutation {operation}($input: MessageInput!) {{
                    {operation}(input: $input) {{
                        success
                        message
                        id
                    }}
                }}
            """,
            "variables": {
                "input": {
                    "content": message.get("content"),
                    "sender": self.agent_id,
                    "recipient": recipient_profile.agent_id,
                    "format": message.get("format", "text")
                }
            }
        }
        
        return formatted
    
    # --- Transformation Methods ---
    
    def _apply_transformation(
        self,
        message: Dict[str, Any],
        transformation: MessageTransformation,
        recipient_profile: AgentProfile
    ) -> Dict[str, Any]:
        """
        Apply a transformation to a message.
        
        Args:
            message: The message to transform
            transformation: The transformation to apply
            recipient_profile: Profile of the recipient
            
        Returns:
            Transformed message
        """
        content = message.get("content")
        format_type = ContentFormat(message.get("format", "text"))
        
        # Apply the specific transformation
        if transformation == MessageTransformation.NONE:
            # No transformation
            pass
            
        elif transformation == MessageTransformation.SIMPLIFY:
            # Simplify content
            if content is not None:
                simplified = self.content_handler.simplify_content(content, format_type)
                message["content"] = simplified
                message["transformed"] = "simplified"
                
        elif transformation == MessageTransformation.STRUCTURE:
            # Add structure to content
            if format_type == ContentFormat.TEXT and isinstance(content, str):
                # Try to convert text to structured format
                try:
                    # Look for JSON-like patterns and convert
                    if "{" in content and ":" in content:
                        # Extract key-value pairs
                        pattern = r'[\"\']?(\w+)[\"\']?\s*:\s*[\"\']?([^\"\',}]+)[\"\']?'
                        matches = re.findall(pattern, content)
                        
                        if matches:
                            structured = {key: value.strip() for key, value in matches}
                            message["content"] = structured
                            message["format"] = ContentFormat.JSON.value
                            message["transformed"] = "structured"
                except Exception as e:
                    logger.warning(f"Failed to structure content: {str(e)}")
                    
        elif transformation == MessageTransformation.COMPRESS:
            # Compress content
            if isinstance(content, str) and len(content) > 1000:
                # Simple compression: truncate long content and add summary
                truncated = content[:1000] + "..."
                message["content"] = truncated
                message["original_length"] = len(content)
                message["transformed"] = "compressed"
                
        elif transformation == MessageTransformation.ENCRYPT:
            # Simple "encryption" (for demonstration - not real encryption)
            if isinstance(content, str):
                # Base64 encoding as a stand-in for encryption
                encoded = base64.b64encode(content.encode('utf-8')).decode('utf-8')
                message["content"] = encoded
                message["format"] = ContentFormat.BASE64.value
                message["transformed"] = "encrypted"
                
        elif transformation == MessageTransformation.CHUNK:
            # Chunk content
            max_size = recipient_profile.max_message_size
            if max_size and isinstance(content, str) and len(content) > max_size:
                # Only keep first chunk in this message
                message["content"] = content[:max_size]
                message["chunked"] = True
                message["chunk_index"] = 0
                message["total_chunks"] = (len(content) + max_size - 1) // max_size
                message["transformed"] = "chunked"
                
        elif transformation == MessageTransformation.SUMMARIZE:
            # Create a summary (simplified implementation)
            if isinstance(content, str) and len(content) > 500:
                # Simple summary approach: first 100 chars + length information
                summary = content[:100] + f"... [Content length: {len(content)} chars]"
                message["content"] = summary
                message["has_full_content"] = True
                message["transformed"] = "summarized"
                
        elif transformation == MessageTransformation.FORMAT_CONVERT:
            # Format conversion handled earlier in adapt_message
            pass
            
        return message
    
    def _recommend_format(self, profile1: AgentProfile, profile2: AgentProfile) -> ContentFormat:
        """
        Recommend the best format for communication between two agents.
        
        Args:
            profile1: Profile of the first agent
            profile2: Profile of the second agent
            
        Returns:
            Recommended content format
        """
        # Check if either agent's preferred format is viable for both
        format1 = profile1.preferred_format
        format2 = profile2.preferred_format
        
        if profile2.can_handle_format(format1):
            return format1
            
        if profile1.can_handle_format(format2):
            return format2
            
        # Check for structured data capability
        if (AgentCapability.STRUCTURED_DATA in profile1.capabilities and
                AgentCapability.STRUCTURED_DATA in profile2.capabilities):
            return ContentFormat.JSON
            
        # Fallback to common formats in order of preference
        for format_type in [ContentFormat.TEXT, ContentFormat.MARKDOWN, ContentFormat.HTML]:
            if profile1.can_handle_format(format_type) and profile2.can_handle_format(format_type):
                return format_type
                
        # Last resort
        return ContentFormat.TEXT
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def _generate_id(self) -> str:
        """Generate a unique ID."""
        return f"msg-{uuid.uuid4().hex[:8]}"
