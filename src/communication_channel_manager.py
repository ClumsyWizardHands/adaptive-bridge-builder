import mimetypes
#!/usr/bin/env python3
"""
Communication Channel Manager

This module provides a unified interface for the agent to communicate through
various channels (email, chat, API, messaging platforms) while maintaining
a consistent identity and conversation context across all channels.
"""

import logging
import json
import time
import base64
import uuid
import mimetypes
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set, Callable, Tuple, BinaryIO
from dataclasses import dataclass, field

from communication_adapter import CommunicationAdapter
from communication_style import CommunicationStyle, FormalityLevel
from content_handler import ContentHandler, ContentFormat
from session_manager import SessionManager
from principle_engine import PrincipleEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CommunicationChannelManager")


class ChannelType(str, Enum):
    """Types of communication channels supported by the manager."""
    EMAIL = "email"
    CHAT = "chat"
    API = "api"
    SMS = "sms"
    MESSAGING_PLATFORM = "messaging_platform"
    VOICE = "voice"
    WEB_INTERFACE = "web_interface"
    CUSTOM = "custom"


class MessagePriority(str, Enum):
    """Priority levels for outgoing messages."""
    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class DeliveryStatus(str, Enum):
    """Status of message delivery."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"


class SecurityLevel(str, Enum):
    """Security levels for communication channels."""
    PUBLIC = "public"            # No authentication needed
    BASIC = "basic"              # Basic authentication
    ENCRYPTED = "encrypted"      # Encrypted communication
    SECURE = "secure"            # Full security (encryption, auth, verification)
    CONFIDENTIAL = "confidential"  # Highest security for sensitive information


@dataclass
class Attachment:
    """Represents a file attachment in a message."""
    filename: str
    content_type: str
    data: Union[bytes, str]  # Raw data or Base64 encoded string
    size: int
    description: Optional[str] = None
    
    @classmethod
    def from_file(cls, filepath: str, description: Optional[str] = None) -> 'Attachment':
        """Create an attachment from a file path."""
        import os
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        content_type, _ = mimetypes.guess_type(filepath)
        if content_type is None:
            content_type = "application/octet-stream"
            
        with open(filepath, 'rb') as f:
            data = f.read()
            
        return cls(
            filename=os.path.basename(filepath),
            content_type=content_type,
            data=data,
            size=len(data),
            description=description
        )
    
    def to_base64(self) -> str:
        """Convert the attachment data to base64 encoded string."""
        if isinstance(self.data, str):
            # Already a string, assume it's base64
            return self.data
            
        return base64.b64encode(self.data).decode('utf-8')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the attachment to a dictionary."""
        return {
            "filename": self.filename,
            "content_type": self.content_type,
            "data": self.to_base64(),
            "size": self.size,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Attachment':
        """Create an attachment from a dictionary."""
        return cls(
            filename=data["filename"],
            content_type=data["content_type"],
            data=data["data"],  # Keep as base64 string
            size=data["size"],
            description=data.get("description")
        )


@dataclass
class ChannelMessage:
    """
    Represents a message sent or received through a communication channel.
    
    This class provides a unified format for messages across different
    channel types, with metadata specific to each channel stored separately.
    """
    
    message_id: str
    channel_type: ChannelType
    sender_id: str
    recipient_id: str
    content: Any
    timestamp: float
    session_id: Optional[str] = None
    subject: Optional[str] = None
    priority: MessagePriority = MessagePriority.NORMAL
    attachments: List[Attachment] = field(default_factory=list)
    content_format: ContentFormat = ContentFormat.TEXT
    metadata: Dict[str, Any] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)  # IDs of related messages
    status: DeliveryStatus = DeliveryStatus.PENDING
    security_level: SecurityLevel = SecurityLevel.BASIC
    
    @classmethod
    def create(
        cls,
        channel_type: ChannelType,
        sender_id: str,
        recipient_id: str,
        content: Any,
        subject: Optional[str] = None,
        session_id: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        attachments: Optional[List[Attachment]] = None,
        content_format: ContentFormat = ContentFormat.TEXT,
        metadata: Optional[Dict[str, Any]] = None,
        references: Optional[List[str]] = None,
        security_level: SecurityLevel = SecurityLevel.BASIC
    ) -> 'ChannelMessage':
        """Create a new channel message with a generated ID and current timestamp."""
        return cls(
            message_id=f"msg-{uuid.uuid4().hex}",
            channel_type=channel_type,
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            subject=subject,
            timestamp=time.time(),
            session_id=session_id,
            priority=priority,
            attachments=attachments or [],
            content_format=content_format,
            metadata=metadata or {},
            references=references or [],
            status=DeliveryStatus.PENDING,
            security_level=security_level
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the channel message to a dictionary."""
        return {
            "message_id": self.message_id,
            "channel_type": self.channel_type.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "content": self.content,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "subject": self.subject,
            "priority": self.priority.value,
            "attachments": [attachment.to_dict() for attachment in self.attachments],
            "content_format": self.content_format.value,
            "metadata": self.metadata,
            "references": self.references,
            "status": self.status.value,
            "security_level": self.security_level.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChannelMessage':
        """Create a channel message from a dictionary."""
        # Convert string values to enums
        channel_type = ChannelType(data["channel_type"])
        priority = MessagePriority(data["priority"]) if "priority" in data else MessagePriority.NORMAL
        content_format = ContentFormat(data["content_format"]) if "content_format" in data else ContentFormat.TEXT
        status = DeliveryStatus(data["status"]) if "status" in data else DeliveryStatus.PENDING
        security_level = SecurityLevel(data["security_level"]) if "security_level" in data else SecurityLevel.BASIC
        
        # Process attachments
        attachments = []
        for attachment_data in data.get("attachments", []):
            attachments.append(Attachment.from_dict(attachment_data))
            
        return cls(
            message_id=data["message_id"],
            channel_type=channel_type,
            sender_id=data["sender_id"],
            recipient_id=data["recipient_id"],
            content=data["content"],
            timestamp=data["timestamp"],
            session_id=data.get("session_id"),
            subject=data.get("subject"),
            priority=priority,
            attachments=attachments,
            content_format=content_format,
            metadata=data.get("metadata", {}),
            references=data.get("references", []),
            status=status,
            security_level=security_level
        )
    
    def add_attachment(self, attachment: Attachment) -> None:
        """Add an attachment to the message."""
        self.attachments = [*self.attachments, attachment]
    
    def get_content_as_text(self) -> str:
        """Get the content as a text string, regardless of original format."""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, dict) or isinstance(self.content, list):
            return json.dumps(self.content, indent=2)
        else:
            return str(self.content)


class ChannelCapabilities:
    """
    Defines the capabilities and constraints of a communication channel.
    
    This class specifies what features a channel supports, such as rich
    text, attachments, delivery confirmations, and formatting options.
    """
    
    def __init__(
        self,
        channel_type: ChannelType,
        max_message_size: Optional[int] = None,
        supports_rich_text: bool = False,
        supports_attachments: bool = False,
        supports_delivery_confirmation: bool = False,
        supports_read_receipts: bool = False,
        supports_formatting: bool = False,
        supported_content_formats: Optional[List[ContentFormat]] = None,
        supports_threading: bool = False,
        is_real_time: bool = False,
        is_synchronous: bool = True,
        throttling_limits: Optional[Dict[str, Any]] = None,
        security_features: Optional[List[str]] = None
    ):
        """
        Initialize channel capabilities.
        
        Args:
            channel_type: The type of communication channel
            max_message_size: Maximum message size in bytes (if applicable)
            supports_rich_text: Whether the channel supports rich text formatting
            supports_attachments: Whether the channel supports file attachments
            supports_delivery_confirmation: Whether the channel provides delivery confirmations
            supports_read_receipts: Whether the channel supports read receipts
            supports_formatting: Whether the channel supports text formatting
            supported_content_formats: Content formats supported by the channel
            supports_threading: Whether the channel supports message threading
            is_real_time: Whether the channel is real-time
            is_synchronous: Whether communication is synchronous
            throttling_limits: Any rate limits on the channel
            security_features: Security features supported by the channel
        """
        self.channel_type = channel_type
        self.max_message_size = max_message_size
        self.supports_rich_text = supports_rich_text
        self.supports_attachments = supports_attachments
        self.supports_delivery_confirmation = supports_delivery_confirmation
        self.supports_read_receipts = supports_read_receipts
        self.supports_formatting = supports_formatting
        self.supported_content_formats = supported_content_formats or [ContentFormat.TEXT]
        self.supports_threading = supports_threading
        self.is_real_time = is_real_time
        self.is_synchronous = is_synchronous
        self.throttling_limits = throttling_limits or {}
        self.security_features = security_features or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert capabilities to a dictionary."""
        return {
            "channel_type": self.channel_type.value,
            "max_message_size": self.max_message_size,
            "supports_rich_text": self.supports_rich_text,
            "supports_attachments": self.supports_attachments,
            "supports_delivery_confirmation": self.supports_delivery_confirmation,
            "supports_read_receipts": self.supports_read_receipts,
            "supports_formatting": self.supports_formatting,
            "supported_content_formats": [fmt.value for fmt in self.supported_content_formats],
            "supports_threading": self.supports_threading,
            "is_real_time": self.is_real_time,
            "is_synchronous": self.is_synchronous,
            "throttling_limits": self.throttling_limits,
            "security_features": self.security_features
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChannelCapabilities':
        """Create capabilities from a dictionary."""
        channel_type = ChannelType(data["channel_type"])
        
        # Convert content format strings to enum values
        content_formats = []
        for fmt_str in data.get("supported_content_formats", ["text"]):
            try:
                content_formats.append(ContentFormat(fmt_str))
            except ValueError:
                logger.warning(f"Unknown content format: {fmt_str}")
                
        return cls(
            channel_type=channel_type,
            max_message_size=data.get("max_message_size"),
            supports_rich_text=data.get("supports_rich_text", False),
            supports_attachments=data.get("supports_attachments", False),
            supports_delivery_confirmation=data.get("supports_delivery_confirmation", False),
            supports_read_receipts=data.get("supports_read_receipts", False),
            supports_formatting=data.get("supports_formatting", False),
            supported_content_formats=content_formats,
            supports_threading=data.get("supports_threading", False),
            is_real_time=data.get("is_real_time", False),
            is_synchronous=data.get("is_synchronous", True),
            throttling_limits=data.get("throttling_limits", {}),
            security_features=data.get("security_features", [])
        )
    
    def can_handle_content_format(self, format_type: ContentFormat) -> bool:
        """Check if the channel can handle a specific content format."""
        return format_type in self.supported_content_formats
    
    def can_handle_attachment(self, attachment: Attachment) -> bool:
        """Check if the channel can handle a specific attachment."""
        if not self.supports_attachments:
            return False
            
        # Check if attachment size exceeds channel limit
        if self.max_message_size and attachment.size > self.max_message_size:
            return False
            
        # Additional checks could be implemented based on content type, etc.
        return True


class ChannelAdapter:
    """
    Abstract base class for channel-specific adapters.
    
    This class defines the interface that all channel adapters must implement
    to integrate with the CommunicationChannelManager.
    """
    
    def __init__(self, channel_type: ChannelType, channel_id: str) -> None:
        """
        Initialize the channel adapter.
        
        Args:
            channel_type: The type of communication channel
            channel_id: Unique identifier for this channel instance
        """
        self.channel_type = channel_type
        self.channel_id = channel_id
        self.capabilities = self._get_capabilities()
        
    def _get_capabilities(self) -> ChannelCapabilities:
        """
        Get the capabilities of this channel adapter.
        
        Returns:
            ChannelCapabilities object
        """
        raise NotImplementedError("Subclasses must implement _get_capabilities")
    
    async def send_message(self, message: ChannelMessage) -> DeliveryStatus:
        """
        Send a message through this channel.
        
        Args:
            message: The message to send
            
        Returns:
            Delivery status after attempted send
        """
        raise NotImplementedError("Subclasses must implement send_message")
    
    async def receive_message(self, raw_message: Any) -> ChannelMessage:
        """
        Process a received message from this channel.
        
        Args:
            raw_message: The raw message data from the channel
            
        Returns:
            Processed ChannelMessage
        """
        raise NotImplementedError("Subclasses must implement receive_message")
    
    async def format_message(self, message: ChannelMessage) -> Any:
        """
        Format a message for this specific channel.
        
        Args:
            message: The message to format
            
        Returns:
            Formatted message ready for this channel
        """
        raise NotImplementedError("Subclasses must implement format_message")
    
    async def check_message_status(self, message_id: str) -> DeliveryStatus:
        """
        Check the delivery status of a previously sent message.
        
        Args:
            message_id: ID of the message to check
            
        Returns:
            Current delivery status
        """
        raise NotImplementedError("Subclasses must implement check_message_status")
    
    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """
        Authenticate with the channel using provided credentials.
        
        Args:
            credentials: Authentication credentials
            
        Returns:
            Whether authentication was successful
        """
        raise NotImplementedError("Subclasses must implement authenticate")


class ChannelSecurityHandler:
    """
    Handles security operations for communication channels.
    
    This class provides authentication, encryption, and other security
    features for channel communications.
    """
    
    def __init__(self) -> None:
        """Initialize the channel security handler."""
        self.security_providers = {}
        
    def register_security_provider(self, channel_type: ChannelType, provider: Any) -> None:
        """
        Register a security provider for a channel type.
        
        Args:
            channel_type: The channel type
            provider: Security provider implementation
        """
        self.security_providers = {**self.security_providers, channel_type: provider}
        
    async def authenticate(
        self,
        channel_type: ChannelType,
        credentials: Dict[str, Any]
    ) -> bool:
        """
        Authenticate with a channel.
        
        Args:
            channel_type: The channel type
            credentials: Authentication credentials
            
        Returns:
            Whether authentication was successful
        """
        provider = self.security_providers.get(channel_type)
        if not provider:
            logger.warning(f"No security provider for channel type: {channel_type}")
            return False
            
        return await provider.authenticate(credentials)
    
    async def encrypt_message(
        self,
        channel_type: ChannelType,
        message: ChannelMessage
    ) -> ChannelMessage:
        """
        Encrypt a message for secure transmission.
        
        Args:
            channel_type: The channel type
            message: Message to encrypt
            
        Returns:
            Encrypted message
        """
        provider = self.security_providers.get(channel_type)
        if not provider or not hasattr(provider, 'encrypt_message'):
            # If no provider or provider doesn't support encryption,
            # return the original message
            return message
            
        return await provider.encrypt_message(message)
    
    async def decrypt_message(
        self,
        channel_type: ChannelType,
        message: ChannelMessage
    ) -> ChannelMessage:
        """
        Decrypt a received encrypted message.
        
        Args:
            channel_type: The channel type
            message: Message to decrypt
            
        Returns:
            Decrypted message
        """
        provider = self.security_providers.get(channel_type)
        if not provider or not hasattr(provider, 'decrypt_message'):
            # If no provider or provider doesn't support decryption,
            # return the original message
            return message
            
        return await provider.decrypt_message(message)
    
    def get_security_level(self, channel_type: ChannelType) -> SecurityLevel:
        """
        Get the security level for a channel type.
        
        Args:
            channel_type: The channel type
            
        Returns:
            Security level
        """
        provider = self.security_providers.get(channel_type)
        if not provider:
            return SecurityLevel.PUBLIC
            
        return getattr(provider, 'security_level', SecurityLevel.BASIC)


class CommunicationChannelManager:
    """
    Manages communication across multiple channels.
    
    This class provides a unified interface for sending and receiving messages
    across different communication channels while maintaining context and
    ensuring appropriate formatting and security.
    """
    
    def __init__(
        self,
        agent_id: str,
        session_manager: Optional[SessionManager] = None,
        communication_adapter: Optional[CommunicationAdapter] = None,
        content_handler: Optional[ContentHandler] = None,
        principle_engine: Optional[PrincipleEngine] = None
    ):
        """
        Initialize the communication channel manager.
        
        Args:
            agent_id: ID of the agent using this manager
            session_manager: Manager for conversation sessions
            communication_adapter: Adapter for message formatting
            content_handler: Handler for content format conversion
            principle_engine: Engine for principle-aligned communication
        """
        self.agent_id = agent_id
        self.session_manager = session_manager or SessionManager(agent_id=agent_id)
        self.communication_adapter = communication_adapter or CommunicationAdapter(agent_id=agent_id)
        self.content_handler = content_handler or ContentHandler()
        self.principle_engine = principle_engine
        
        # Initialize channel adapters, security handler and message store
        self.channel_adapters: Dict[str, ChannelAdapter] = {}
        self.security_handler = ChannelSecurityHandler()
        self.message_store: Dict[str, ChannelMessage] = {}
        
        # Channel-entity mapping (tracks which channels are used by which entities)
        self.entity_channels: Dict[str, Dict[ChannelType, str]] = {}
        
        # Default communication styles for different channel types
        self.channel_communication_styles: Dict[ChannelType, CommunicationStyle] = {}
        
        logger.info(f"CommunicationChannelManager initialized for agent {agent_id}")
    
    def register_channel_adapter(self, adapter: ChannelAdapter) -> None:
        """
        Register a channel adapter.
        
        Args:
            adapter: The channel adapter to register
        """
        self.channel_adapters = {**self.channel_adapters, adapter.channel_id: adapter}
        logger.info(f"Registered {adapter.channel_type} adapter with ID {adapter.channel_id}")
    
    def get_channel_adapter(self, channel_id: str) -> Optional[ChannelAdapter]:
        """
        Get a registered channel adapter by ID.
        
        Args:
            channel_id: ID of the channel adapter
            
        Returns:
            The channel adapter if found, None otherwise
        """
        return self.channel_adapters.get(channel_id)
    
    def get_adapters_by_type(self, channel_type: ChannelType) -> List[ChannelAdapter]:
        """
        Get all registered adapters of a specific type.
        
        Args:
            channel_type: Type of channel adapters to get
            
        Returns:
            List of matching channel adapters
        """
        return [
            adapter for adapter in self.channel_adapters.values()
            if adapter.channel_type == channel_type
        ]
    
    def register_entity_channel(
        self,
        entity_id: str,
        channel_type: ChannelType,
        channel_id: str
    ) -> None:
        """
        Register a channel used by a specific entity.
        
        Args:
            entity_id: ID of the entity (user, agent, etc.)
            channel_type: Type of communication channel
            channel_id: ID of the specific channel adapter
        """
        if entity_id not in self.entity_channels:
            self.entity_channels = {**self.entity_channels, entity_id: {}}
            
        self.entity_channels[entity_id][channel_type] = channel_id
        logger.debug(f"Registered {channel_type} for entity {entity_id}")
    
    def get_entity_channel(
        self,
        entity_id: str,
        channel_type: ChannelType
    ) -> Optional[str]:
        """
        Get the channel ID used by an entity for a specific channel type.
        
        Args:
            entity_id: ID of the entity
            channel_type: Type of communication channel
            
        Returns:
            Channel ID if found, None otherwise
        """
        if entity_id not in self.entity_channels:
            return None
            
        return self.entity_channels[entity_id].get(channel_type)
    
    def get_entity_channels(self, entity_id: str) -> Dict[ChannelType, str]:
        """
        Get all channels used by an entity.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Dictionary of channel types to channel IDs
        """
        return self.entity_channels.get(entity_id, {})
    
    def set_channel_communication_style(
        self,
        channel_type: ChannelType,
        style: CommunicationStyle
    ) -> None:
        """
        Set the default communication style for a channel type.
        
        Args:
            channel_type: Type of communication channel
            style: Communication style to use
        """
        self.channel_communication_styles = {**self.channel_communication_styles, channel_type: style}
        logger.debug(f"Set communication style for {channel_type}")
    
    def get_channel_communication_style(
        self,
        channel_type: ChannelType
    ) -> Optional[CommunicationStyle]:
        """
        Get the default communication style for a channel type.
        
        Args:
            channel_type: Type of communication channel
            
        Returns:
            Communication style if set, None otherwise
        """
        return self.channel_communication_styles.get(channel_type)
    
    async def format_for_channel(
        self,
        message: ChannelMessage,
        channel_id: str
    ) -> ChannelMessage:
        """
        Format a message for a specific channel.
        
        Args:
            message: Message to format
            channel_id: ID of the target channel
            
        Returns:
            Formatted message
        """
        adapter = self.get_channel_adapter(channel_id)
        if not adapter:
            logger.error(f"Channel adapter not found: {channel_id}")
            return message
            
        # Check if the content format is supported by the channel
        if not adapter.capabilities.can_handle_content_format(message.content_format):
            # Convert to a supported format
            supported_formats = adapter.capabilities.supported_content_formats
            if not supported_formats:
                logger.warning(f"No supported formats for channel {channel_id}")
                # Default to TEXT format
                target_format = ContentFormat.TEXT
            else:
                target_format = supported_formats[0]
                
            # Convert content
            content_str = message.get_content_as_text()
            converted_content, success = self.content_handler.convert_content(
                content=content_str,
                from_format=message.content_format,
                to_format=target_format
            )
            
            if success:
                message.content = converted_content
                message.content_format = target_format
                logger.debug(f"Converted content format from {message.content_format} to {target_format}")
            else:
                logger.warning(f"Failed to convert content format from {message.content_format} to {target_format}")
        
        # Check and process attachments
        if message.attachments and not adapter.capabilities.supports_attachments:
            # Channel doesn't support attachments, add references to them in the content
            content_str = message.get_content_as_text()
            attachment_list = "\n\nAttachments (not supported in this channel):\n"
            for attachment in message.attachments:
                attachment_list += f"- {attachment.filename} ({attachment.size} bytes) - {attachment.description or 'No description'}\n"
                
            message.content = content_str + attachment_list
            message.attachments = []  # Remove attachments
        
        # Apply channel-specific formatting
        formatted_message = await adapter.format_message(message)
        
        return formatted_message
    
    async def send_message(
        self,
        recipient_id: str,
        content: Any,
        channel_type: Optional[ChannelType] = None,
        subject: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        attachments: Optional[List[Attachment]] = None,
        content_format: ContentFormat = ContentFormat.TEXT,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        references: Optional[List[str]] = None,
        security_level: SecurityLevel = SecurityLevel.BASIC
    ) -> Optional[str]:
        """
        Send a message to a recipient through an appropriate channel.
        
        Args:
            recipient_id: ID of the recipient
            content: Message content
            channel_type: Preferred channel type (if None, best available will be used)
            subject: Message subject (for email, etc.)
            priority: Message priority
            attachments: File attachments
            content_format: Format of the message content
            session_id: ID of the conversation session
            metadata: Additional metadata
            references: IDs of related messages
            security_level: Required security level
            
        Returns:
            Message ID if sent successfully, None otherwise
        """
        # Determine the channel to use
        channel_id = None
        
        if channel_type:
            # Use the specified channel type
            channel_id = self.get_entity_channel(recipient_id, channel_type)
            if not channel_id:
                # Try to find any adapter of the requested type
                adapters = self.get_adapters_by_type(channel_type)
                if adapters:
                    channel_id = adapters[0].channel_id
        else:
            # Try to find the best channel for this entity
            entity_channels = self.get_entity_channels(recipient_id)
            if entity_channels:
                # Use the first available channel
                first_type = next(iter(entity_channels.keys()))
                channel_id = entity_channels[first_type]
        
        if not channel_id:
            logger.error(f"No suitable channel found for recipient {recipient_id}")
            return None
            
        adapter = self.get_channel_adapter(channel_id)
        if not adapter:
            logger.error(f"Channel adapter not found: {channel_id}")
            return None
            
        # Get or create session
        if not session_id:
            # Try to find an existing session for this entity
            entity_sessions = self.session_manager.get_sessions_by_participant(recipient_id)
            if entity_sessions:
                # Use the most recent session
                session_id = entity_sessions[0].session_id
            else:
                # Create a new session
                session = self.session_manager.create_session([self.agent_id, recipient_id])
                session_id = session.session_id
                
        # Create the message
        message = ChannelMessage.create(
            channel_type=adapter.channel_type,
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            content=content,
            subject=subject,
            session_id=session_id,
            priority=priority,
            attachments=attachments,
            content_format=content_format,
            metadata=metadata,
            references=references,
            security_level=security_level
        )
        
        # Apply communication style appropriate for the channel
        style = self.get_channel_communication_style(adapter.channel_type)
        if style and hasattr(self.communication_adapter, 'adapt_message'):
            # If we have a communication adapter with adaptation capabilities,
            # use it to format the message according to the communication style
            adapted_content = self.communication_adapter.adapt_message(
                message=message.content,
                recipient_id=recipient_id
            )
            message.content = adapted_content.get('content', message.content)
            
        # Format for the specific channel
        formatted_message = await self.format_for_channel(message, channel_id)
        
        # Apply security measures if needed
        if security_level != SecurityLevel.PUBLIC:
            formatted_message = await self.security_handler.encrypt_message(
                adapter.channel_type, formatted_message
            )
            
        # Send the message
        status = await adapter.send_message(formatted_message)
        
        # Update message status and store
        message.status = status
        self.message_store = {**self.message_store, message.message_id: message}
        
        # Add to the session
        self.session_manager.add_message_to_session(
            session_id=session_id,
            message_id=message.message_id,
            content=message.content,
            sender_id=message.sender_id,
            metadata={
                "channel_type": message.channel_type.value,
                "timestamp": message.timestamp,
                "status": message.status.value
            }
        )
        
        # Return the message ID if successful
        if status == DeliveryStatus.FAILED:
            logger.error(f"Failed to send message to {recipient_id}: {message.message_id}")
            return None
        
        return message.message_id
    
    async def receive_message(
        self,
        raw_message: Any,
        channel_id: str
    ) -> Optional[ChannelMessage]:
        """
        Process a received message from a channel.
        
        Args:
            raw_message: Raw message data
            channel_id: ID of the channel that received the message
            
        Returns:
            Processed ChannelMessage if successful, None otherwise
        """
        adapter = self.get_channel_adapter(channel_id)
        if not adapter:
            logger.error(f"Channel adapter not found: {channel_id}")
            return None
            
        # Process the message through the channel adapter
        try:
            message = await adapter.receive_message(raw_message)
        except Exception as e:
            logger.error(f"Error processing message from {channel_id}: {str(e)}")
            return None
            
        # Apply security measures if needed
        if message.security_level != SecurityLevel.PUBLIC:
            message = await self.security_handler.decrypt_message(
                adapter.channel_type, message
            )
            
        # Store the message
        self.message_store = {**self.message_store, message.message_id: message}
        
        # Get or create session
        if message.session_id:
            session_id = message.session_id
        else:
            # Try to find an existing session for this entity
            entity_sessions = self.session_manager.get_sessions_by_participant(message.sender_id)
            if entity_sessions:
                # Use the most recent session
                session_id = entity_sessions[0].session_id
            else:
                # Create a new session
                session = self.session_manager.create_session([self.agent_id, message.sender_id])
                session_id = session.session_id
                
            # Update the message with the session ID
            message.session_id = session_id
            
        # Add to the session
        self.session_manager.add_message_to_session(
            session_id=session_id,
            message_id=message.message_id,
            content=message.content,
            sender_id=message.sender_id,
            metadata={
                "channel_type": message.channel_type.value,
                "timestamp": message.timestamp,
                "status": message.status.value
            }
        )
        
        # Register the channel for this entity if we haven't already
        self.register_entity_channel(
            entity_id=message.sender_id,
            channel_type=message.channel_type,
            channel_id=channel_id
        )
        
        return message
    
    async def check_message_status(self, message_id: str) -> Optional[DeliveryStatus]:
        """
        Check the status of a previously sent message.
        
        Args:
            message_id: ID of the message
            
        Returns:
            Current delivery status if found, None otherwise
        """
        message = self.message_store.get(message_id)
        if not message:
            logger.warning(f"Message not found: {message_id}")
            return None
            
        adapter = None
        for channel_id in self.entity_channels.get(message.recipient_id, {}).values():
            adapter_candidate = self.get_channel_adapter(channel_id)
            if adapter_candidate and adapter_candidate.channel_type == message.channel_type:
                adapter = adapter_candidate
                break
                
        if not adapter:
            logger.warning(f"Channel adapter not found for message: {message_id}")
            return message.status
            
        # Check status through the adapter
        try:
            status = await adapter.check_message_status(message_id)
            
            # Update the stored message
            message.status = status
            self.message_store = {**self.message_store, message_id: message}
            
            return status
        except Exception as e:
            logger.error(f"Error checking message status: {str(e)}")
            return message.status
    
    def get_conversation_context(
        self,
        entity_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation context for an entity across all channels.
        
        Args:
            entity_id: ID of the entity
            limit: Maximum number of messages to include
            
        Returns:
            List of messages with most recent first
        """
        # Get sessions involving this entity
        entity_sessions = self.session_manager.get_sessions_by_participant(entity_id)
        if not entity_sessions:
            return []
            
        # Collect messages from all sessions
        all_messages = []
        for session in entity_sessions:
            session_messages = self.session_manager.get_session_messages(session.session_id)
            all_messages.extend(session_messages)
            
        # Sort by timestamp (most recent first)
        all_messages.sort(key=lambda msg: msg.get('timestamp', 0), reverse=True)
        
        # Apply limit if specified
        if limit:
            all_messages = all_messages[:limit]
            
        return all_messages
    
    def get_channel_history(
        self,
        entity_id: str,
        channel_type: ChannelType,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for an entity on a specific channel.
        
        Args:
            entity_id: ID of the entity
            channel_type: Type of channel
            limit: Maximum number of messages to include
            
        Returns:
            List of messages with most recent first
        """
        # Get all messages for this entity
        all_messages = self.get_conversation_context(entity_id)
        
        # Filter by channel type
        channel_messages = [
            msg for msg in all_messages
            if msg.get('metadata', {}).get('channel_type') == channel_type.value
        ]
        
        # Apply limit if specified
        if limit:
            channel_messages = channel_messages[:limit]
            
        return channel_messages
    
    async def switch_channel(
        self,
        entity_id: str,
        from_channel_type: ChannelType,
        to_channel_type: ChannelType,
        content: Optional[str] = None
    ) -> Optional[str]:
        """
        Switch communication with an entity from one channel to another.
        
        Args:
            entity_id: ID of the entity
            from_channel_type: Current channel type
            to_channel_type: New channel type
            content: Optional message to send on the new channel
            
        Returns:
            ID of the sent message if content provided, None otherwise
        """
        # Check if entity has the target channel registered
        target_channel_id = self.get_entity_channel(entity_id, to_channel_type)
        if not target_channel_id:
            logger.error(f"Entity {entity_id} does not have a registered {to_channel_type} channel")
            return None
            
        # Get the current session
        entity_sessions = self.session_manager.get_sessions_by_participant(entity_id)
        if not entity_sessions:
            logger.warning(f"No sessions found for entity {entity_id}")
            return None
            
        current_session_id = entity_sessions[0].session_id
        
        # Send a message on the new channel if content is provided
        if content:
            return await self.send_message(
                recipient_id=entity_id,
                content=content,
                channel_type=to_channel_type,
                session_id=current_session_id,
                metadata={"channel_switch": True, "previous_channel": from_channel_type.value}
            )
        
        return None
    
    def get_entity_preferred_channel(self, entity_id: str) -> Optional[ChannelType]:
        """
        Get the preferred channel type for an entity based on usage patterns.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Preferred channel type if found, None otherwise
        """
        # Get all messages for this entity
        all_messages = self.get_conversation_context(entity_id)
        if not all_messages:
            return None
            
        # Count channel usage
        channel_counts = {}
        for msg in all_messages:
            channel_type = msg.get('metadata', {}).get('channel_type')
            if channel_type:
                channel_counts[channel_type] = channel_counts.get(channel_type, 0) + 1
                
        # Get the most used channel
        if not channel_counts:
            return None
            
        most_used_channel = max(channel_counts.items(), key=lambda x: x[1])[0]
        try:
            return ChannelType(most_used_channel)
        except ValueError:
            logger.warning(f"Invalid channel type: {most_used_channel}")
            return None
    
    async def bulk_send(
        self,
        recipients: List[str],
        content: Any,
        channel_type: ChannelType,
        **kwargs
    ) -> Dict[str, Optional[str]]:
        """
        Send the same message to multiple recipients.
        
        Args:
            recipients: List of recipient IDs
            content: Message content
            channel_type: Channel type to use
            **kwargs: Additional parameters for send_message
            
        Returns:
            Dictionary mapping recipient IDs to message IDs (or None if failed)
        """
        results = {}
        for recipient_id in recipients:
            message_id = await self.send_message(
                recipient_id=recipient_id,
                content=content,
                channel_type=channel_type,
                **kwargs
            )
            results[recipient_id] = message_id
            
        return results