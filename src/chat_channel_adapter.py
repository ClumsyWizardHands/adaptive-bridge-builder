#!/usr/bin/env python3
"""
Chat Channel Adapter

This module provides an implementation of the ChannelAdapter interface
for real-time chat communication, allowing the agent to interact through
chat interfaces like messaging apps, web chat widgets, and chat platforms.
"""

import asyncio
import json
import logging
import time
import uuid
import websockets
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set, Awaitable

from communication_channel_manager import (
    ChannelAdapter, ChannelType, ChannelCapabilities, ChannelMessage,
    Attachment, DeliveryStatus, SecurityLevel, MessagePriority
)
from content_handler import ContentFormat
from communication_style import CommunicationStyle, FormalityLevel, DirectnessLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ChatChannelAdapter")


class ChatPlatform(str, Enum):
    """Supported chat platforms."""
    GENERIC = "generic"               # Generic chat interface
    SLACK = "slack"                   # Slack
    TEAMS = "teams"                   # Microsoft Teams
    DISCORD = "discord"               # Discord
    TELEGRAM = "telegram"             # Telegram
    WHATSAPP = "whatsapp"             # WhatsApp
    MESSENGER = "messenger"           # Facebook Messenger
    CUSTOM = "custom"                 # Custom chat platform


class ChatEvent(str, Enum):
    """Events that can occur in chat communication."""
    MESSAGE = "message"               # New message
    TYPING = "typing"                 # User is typing
    READ = "read"                     # Message was read
    JOIN = "join"                     # User joined the chat
    LEAVE = "leave"                   # User left the chat
    REACTION = "reaction"             # Reaction to a message
    EDIT = "edit"                     # Message was edited
    DELETE = "delete"                 # Message was deleted


class ChatFeature(str, Enum):
    """Features that may be supported by chat platforms."""
    TYPING_INDICATOR = "typing_indicator"  # Show when user is typing
    READ_RECEIPTS = "read_receipts"        # Show when message is read
    REACTIONS = "reactions"                # Allow emoji reactions
    THREADS = "threads"                    # Support threaded conversations
    RICH_MESSAGES = "rich_messages"        # Support for rich UI elements
    FILE_SHARING = "file_sharing"          # Support for file attachments
    USER_PRESENCE = "user_presence"        # Show user online/offline status
    EDITED_MESSAGES = "edited_messages"    # Support for editing messages
    MESSAGE_FORMATTING = "message_formatting"  # Support for text formatting


class RichMessageComponent(str, Enum):
    """Types of rich message components."""
    BUTTON = "button"                 # Clickable button
    QUICK_REPLY = "quick_reply"       # Quick reply option
    CARD = "card"                     # Rich card with image/title/text
    CAROUSEL = "carousel"             # Carousel of cards
    IMAGE = "image"                   # Image
    VIDEO = "video"                   # Video
    AUDIO = "audio"                   # Audio
    LOCATION = "location"             # Location data


class ChatConfig:
    """Configuration for chat communication."""
    
    def __init__(
        self,
        platform: ChatPlatform,
        connection_url: Optional[str] = None,
        api_key: Optional[str] = None,
        auth_token: Optional[str] = None,
        bot_id: Optional[str] = None,
        bot_name: Optional[str] = None,
        supported_features: Optional[List[ChatFeature]] = None,
        supported_rich_components: Optional[List[RichMessageComponent]] = None,
        max_message_length: Optional[int] = None,
        rate_limit: Optional[int] = None,
        custom_platform_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize chat configuration.
        
        Args:
            platform: Chat platform to use
            connection_url: WebSocket or API URL for connection
            api_key: API key for authentication
            auth_token: Authentication token
            bot_id: ID of the bot on the platform
            bot_name: Display name of the bot
            supported_features: List of supported chat features
            supported_rich_components: List of supported rich message components
            max_message_length: Maximum allowed message length
            rate_limit: Maximum messages per minute
            custom_platform_config: Additional platform-specific configuration
        """
        self.platform = platform
        self.connection_url = connection_url
        self.api_key = api_key
        self.auth_token = auth_token
        self.bot_id = bot_id
        self.bot_name = bot_name
        self.supported_features = supported_features or []
        self.supported_rich_components = supported_rich_components or []
        self.max_message_length = max_message_length
        self.rate_limit = rate_limit
        self.custom_platform_config = custom_platform_config or {}


class ChatChannelAdapter(ChannelAdapter):
    """
    Adapter for real-time chat communication.
    
    This adapter implements the ChannelAdapter interface for chat-based
    communication, handling message formatting, sending, and receiving
    through various chat platforms.
    """
    
    def __init__(
        self,
        channel_id: str,
        config: ChatConfig,
        agent_id: str,
        message_handler: Optional[Callable[[ChannelMessage], Any]] = None,
        event_handlers: Optional[Dict[ChatEvent, Callable]] = None,
        communication_style: Optional[CommunicationStyle] = None,
        rich_message_templates: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initialize the chat channel adapter.
        
        Args:
            channel_id: Unique identifier for this channel
            config: Chat configuration
            agent_id: ID of the agent using this adapter
            message_handler: Callback for handling incoming messages
            event_handlers: Callbacks for handling chat events
            communication_style: Communication style for chat messages
            rich_message_templates: Templates for rich message components
        """
        super().__init__(ChannelType.CHAT, channel_id)
        self.config = config
        self.agent_id = agent_id
        self.message_handler = message_handler
        self.event_handlers = event_handlers or {}
        self.communication_style = communication_style or self._get_default_communication_style()
        self.rich_message_templates = rich_message_templates or {}
        
        # Websocket connection
        self.websocket = None
        self.is_connected = False
        self.connection_task = None
        self.message_queue = asyncio.Queue()
        
        # Track active chats and users
        self.active_chats: Dict[str, Dict[str, Any]] = {}
        self.user_info: Dict[str, Dict[str, Any]] = {}
        
        # Track message delivery
        self.message_status: Dict[str, DeliveryStatus] = {}
        self.typing_users: Set[str] = set()
        
        logger.info(f"ChatChannelAdapter initialized for {config.platform.value}")
    
    def _get_capabilities(self) -> ChannelCapabilities:
        """Get the capabilities of this chat channel."""
        return ChannelCapabilities(
            channel_type=ChannelType.CHAT,
            max_message_size=self.config.max_message_length,
            supports_rich_text=ChatFeature.MESSAGE_FORMATTING in self.config.supported_features,
            supports_attachments=ChatFeature.FILE_SHARING in self.config.supported_features,
            supports_delivery_confirmation=ChatFeature.READ_RECEIPTS in self.config.supported_features,
            supports_read_receipts=ChatFeature.READ_RECEIPTS in self.config.supported_features,
            supports_formatting=ChatFeature.MESSAGE_FORMATTING in self.config.supported_features,
            supported_content_formats=[
                ContentFormat.TEXT,
                ContentFormat.MARKDOWN if ChatFeature.MESSAGE_FORMATTING in self.config.supported_features else None,
                ContentFormat.HTML if ChatFeature.MESSAGE_FORMATTING in self.config.supported_features else None
            ],
            supports_threading=ChatFeature.THREADS in self.config.supported_features,
            is_real_time=True,
            is_synchronous=True,
            throttling_limits={
                "max_messages_per_minute": self.config.rate_limit or 60
            },
            security_features=[
                "token_auth",
                "api_key"
            ]
        )
    
    def _get_default_communication_style(self) -> CommunicationStyle:
        """Get a default communication style for chat."""
        return CommunicationStyle(
            agent_id=self.agent_id,
            formality=FormalityLevel.CASUAL,
            directness=DirectnessLevel.DIRECT,
            prefers_emoji=True,
            vocabulary_level=0.4  # Slightly simpler vocabulary for chat
        )
    
    async def connect(self) -> bool:
        """
        Connect to the chat platform.
        
        Returns:
            Whether connection was successful
        """
        if self.is_connected:
            return True
            
        try:
            if self.config.connection_url:
                # Connect via WebSocket
                extra_headers = {}
                if self.config.auth_token:
                    extra_headers["Authorization"] = f"Bearer {self.config.auth_token}"
                if self.config.api_key:
                    extra_headers["X-API-Key"] = self.config.api_key
                    
                self.websocket = await websockets.connect(
                    self.config.connection_url,
                    extra_headers=extra_headers
                )
                
                # Set up connection task
                self.connection_task = asyncio.create_task(self._connection_handler())
                self.is_connected = True
                logger.info(f"Connected to {self.config.platform.value} chat")
                
                # Send authentication message if needed
                if self.config.platform != ChatPlatform.GENERIC:
                    await self._send_platform_auth()
                    
                return True
            else:
                logger.error("No connection URL provided")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to chat: {str(e)}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the chat platform."""
        if not self.is_connected:
            return
            
        try:
            # Cancel connection task
            if self.connection_task:
                self.connection_task.cancel()
                try:
                    await self.connection_task
                except asyncio.CancelledError:
                    pass
                self.connection_task = None
                
            # Close websocket
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
                
            self.is_connected = False
            logger.info(f"Disconnected from {self.config.platform.value} chat")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {str(e)}")
    
    async def _connection_handler(self) -> None:
        """Handle the websocket connection and messages."""
        try:
            # Start message processor
            processor_task = asyncio.create_task(self._process_outgoing_messages())
            
            # Process incoming messages
            async for message in self.websocket:
                try:
                    # Parse incoming message
                    if isinstance(message, str):
                        data = json.loads(message)
                    elif isinstance(message, bytes):
                        data = json.loads(message.decode('utf-8'))
                    else:
                        data = message
                        
                    # Determine event type
                    event_type = data.get("type", ChatEvent.MESSAGE)
                    
                    # Handle based on event type
                    if event_type == ChatEvent.MESSAGE:
                        # Convert to channel message
                        channel_message = await self.receive_message(data)
                        
                        # Call message handler if provided
                        if self.message_handler:
                            await self.message_handler(channel_message)
                            
                    else:
                        # Handle other event types
                        handler = self.event_handlers.get(event_type)
                        if handler:
                            await handler(data)
                            
                except Exception as e:
                    logger.error(f"Error processing incoming message: {str(e)}")
                    
        except asyncio.CancelledError:
            # Connection task was cancelled
            pass
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {str(e)}")
            
        finally:
            # Clean up
            if processor_task:
                processor_task.cancel()
                try:
                    await processor_task
                except asyncio.CancelledError:
                    pass
    
    async def _process_outgoing_messages(self) -> None:
        """Process the outgoing message queue."""
        try:
            while True:
                # Get message from queue
                message, future = await self.message_queue.get()
                
                try:
                    # Send the message
                    await self.websocket.send(json.dumps(message))
                    
                    # Set future result
                    if not future.done():
                        future.set_result(True)
                        
                except Exception as e:
                    logger.error(f"Error sending message: {str(e)}")
                    
                    # Set future exception
                    if not future.done():
                        future.set_exception(e)
                        
                # Mark task as done
                self.message_queue.task_done()
                
                # Rate limiting
                if self.config.rate_limit:
                    await asyncio.sleep(60 / self.config.rate_limit)
                    
        except asyncio.CancelledError:
            # Message processor was cancelled
            pass
    
    async def _send_platform_auth(self) -> None:
        """Send platform-specific authentication message."""
        if self.config.platform == ChatPlatform.SLACK:
            auth_message = {
                "type": "auth",
                "token": self.config.auth_token
            }
        elif self.config.platform == ChatPlatform.DISCORD:
            auth_message = {
                "op": 2,  # Discord IDENTIFY opcode
                "d": {
                    "token": self.config.auth_token,
                    "properties": {
                        "$os": "linux",
                        "$browser": "adaptive_bridge_builder",
                        "$device": "adaptive_bridge_builder"
                    }
                }
            }
        elif self.config.platform == ChatPlatform.TEAMS:
            auth_message = {
                "type": "auth",
                "tenant": self.config.custom_platform_config.get("tenant_id", ""),
                "token": self.config.auth_token
            }
        else:
            # Generic auth message
            auth_message = {
                "type": "auth",
                "token": self.config.auth_token,
                "api_key": self.config.api_key,
                "bot_id": self.config.bot_id
            }
            
        # Queue the auth message
        future = asyncio.Future()
        await self.message_queue.put((auth_message, future))
        
        # Wait for result
        try:
            await future
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
    
    async def _send_websocket_message(self, message: Dict[str, Any]) -> bool:
        """
        Send a message through the websocket.
        
        Args:
            message: Message to send
            
        Returns:
            Whether message was queued successfully
        """
        if not self.is_connected:
            logger.error("Not connected to chat")
            return False
            
        # Create future for tracking the result
        future = asyncio.Future()
        
        # Queue the message
        await self.message_queue.put((message, future))
        
        # Wait for result
        try:
            result = await future
            return result
        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")
            return False
    
    async def format_message(self, message: ChannelMessage) -> Dict[str, Any]:
        """
        Format a channel message for the chat platform.
        
        Args:
            message: Channel message to format
            
        Returns:
            Formatted chat message
        """
        platform = self.config.platform
        
        # Start with a basic structure
        formatted_message = {
            "id": message.message_id,
            "text": message.content if isinstance(message.content, str) else str(message.content)
        }
        
        # Check if message content needs to be truncated
        if self.config.max_message_length and len(formatted_message["text"]) > self.config.max_message_length:
            # Truncate to max length - 3 and add ellipsis
            formatted_message["text"] = formatted_message["text"][:self.config.max_message_length - 3] + "..."
            
        # Add recipient information
        if message.recipient_id:
            formatted_message["recipient"] = message.recipient_id
            
        # Add thread information if available
        if message.references and ChatFeature.THREADS in self.config.supported_features:
            formatted_message["thread_ts"] = message.references[0]
            
        # Handle platform-specific formatting
        if platform == ChatPlatform.SLACK:
            # Slack-specific formatting
            slack_message = {
                "id": message.message_id,
                "type": "message",
                "channel": message.recipient_id,
                "text": formatted_message["text"]
            }
            
            # Add thread_ts if available
            if "thread_ts" in formatted_message:
                slack_message["thread_ts"] = formatted_message["thread_ts"]
                
            # Convert attachments if any
            if message.attachments and ChatFeature.FILE_SHARING in self.config.supported_features:
                slack_message["attachments"] = []
                for attachment in message.attachments:
                    slack_attachment = {
                        "filename": attachment.filename,
                        "filetype": attachment.content_type.split("/")[1] if "/" in attachment.content_type else "binary",
                        "size": attachment.size
                    }
                    
                    # Include file data if available
                    if isinstance(attachment.data, str):
                        slack_attachment["url_private"] = attachment.data  # Assume it's a URL
                    
                    slack_message["attachments"].append(slack_attachment)
                    
            return slack_message
            
        elif platform == ChatPlatform.TEAMS:
            # Teams-specific formatting
            teams_message = {
                "type": "message",
                "id": message.message_id,
                "timestamp": datetime.now().isoformat(),
                "content": formatted_message["text"],
                "from": {
                    "id": self.config.bot_id,
                    "name": self.config.bot_name
                },
                "recipient": {
                    "id": message.recipient_id
                }
            }
            
            # Add thread reference if available
            if "thread_ts" in formatted_message:
                teams_message["replyToId"] = formatted_message["thread_ts"]
                
            return teams_message
            
        elif platform == ChatPlatform.DISCORD:
            # Discord-specific formatting
            discord_message = {
                "op": 0,  # Discord message opcode
                "d": {
                    "content": formatted_message["text"],
                    "tts": False,
                    "message_reference": None
                }
            }
            
            # Add channel ID
            discord_message["d"]["channel_id"] = message.recipient_id
            
            # Add message reference if replying to a thread
            if "thread_ts" in formatted_message:
                discord_message["d"]["message_reference"] = {
                    "message_id": formatted_message["thread_ts"]
                }
                
            return discord_message
            
        else:
            # Generic formatting for other platforms
            return formatted_message
    
    async def send_message(self, message: ChannelMessage) -> DeliveryStatus:
        """
        Send a message through the chat channel.
        
        Args:
            message: Channel message to send
            
        Returns:
            Delivery status after attempted send
        """
        try:
            # Ensure we're connected
            if not self.is_connected:
                success = await self.connect()
                if not success:
                    logger.error("Failed to connect to chat")
                    return DeliveryStatus.FAILED
                    
            # Format message for the platform
            formatted_message = await self.format_message(message)
            
            # Send typing indicator if supported
            if ChatFeature.TYPING_INDICATOR in self.config.supported_features:
                typing_message = {
                    "type": "typing",
                    "recipient": message.recipient_id
                }
                await self._send_websocket_message(typing_message)
                
                # Brief pause to simulate typing
                await asyncio.sleep(1)
                
            # Send the message
            success = await self._send_websocket_message(formatted_message)
            
            if success:
                # Update status
                self.message_status[message.message_id] = DeliveryStatus.SENT
                return DeliveryStatus.SENT
            else:
                self.message_status[message.message_id] = DeliveryStatus.FAILED
                return DeliveryStatus.FAILED
                
        except Exception as e:
            logger.error(f"Error sending chat message: {str(e)}")
            self.message_status[message.message_id] = DeliveryStatus.FAILED
            return DeliveryStatus.FAILED
    
    async def receive_message(self, raw_message: Any) -> ChannelMessage:
        """
        Process a received chat message.
        
        Args:
            raw_message: Raw chat message data
            
        Returns:
            Processed channel message
        """
        platform = self.config.platform
        
        # Parse the raw message based on platform
        if platform == ChatPlatform.SLACK:
            return await self._parse_slack_message(raw_message)
        elif platform == ChatPlatform.TEAMS:
            return await self._parse_teams_message(raw_message)
        elif platform == ChatPlatform.DISCORD:
            return await self._parse_discord_message(raw_message)
        else:
            # Generic parsing for other platforms
            return await self._parse_generic_message(raw_message)
    
    async def _parse_slack_message(self, raw_message: Dict[str, Any]) -> ChannelMessage:
        """Parse a Slack message into a channel message."""
        # Extract message fields
        message_id = raw_message.get("ts") or raw_message.get("id") or f"slack-{uuid.uuid4().hex}"
        sender_id = raw_message.get("user")
        content = raw_message.get("text", "")
        timestamp = float(raw_message.get("ts") or time.time())
        
        # Get chat/channel ID
        recipient_id = raw_message.get("channel")
        
        # Thread/parent message reference
        references = []
        if "thread_ts" in raw_message and raw_message["thread_ts"] != raw_message.get("ts"):
            references.append(raw_message["thread_ts"])
            
        # Handle attachments if any
        attachments = []
        for file in raw_message.get("files", []):
            attachment = Attachment(
                filename=file.get("name", "slack_file"),
                content_type=file.get("mimetype", "application/octet-stream"),
                data=file.get("url_private", ""),  # Store the URL
                size=file.get("size", 0),
                description=file.get("title")
            )
            attachments.append(attachment)
            
        # Create the channel message
        message = ChannelMessage(
            message_id=message_id,
            channel_type=ChannelType.CHAT,
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            timestamp=timestamp,
            attachments=attachments,
            references=references,
            status=DeliveryStatus.DELIVERED,
            content_format=ContentFormat.TEXT,
            metadata={
                "platform": self.config.platform.value,
                "raw_data": raw_message
            }
        )
        
        return message
    
    async def check_message_status(self, message_id: str) -> DeliveryStatus:
        """
        Check the delivery status of a previously sent message.
        
        Args:
            message_id: ID of the message to check
            
        Returns:
            Current delivery status
        """
        # Check cache
        if message_id in self.message_status:
            return self.message_status[message_id]
            
        # For chat platforms, we usually don't have a reliable way to check
        # status after the fact unless they have read receipts
        if ChatFeature.READ_RECEIPTS in self.config.supported_features:
            # Could implement platform-specific read receipt checking here
            pass
            
        # Default to the last known status or SENT
        return self.message_status.get(message_id, DeliveryStatus.SENT)
    
    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """
        Authenticate with the chat platform.
        
        Args:
            credentials: Authentication credentials
            
        Returns:
            Whether authentication was successful
        """
        # Use provided credentials to update config
        if "api_key" in credentials:
            self.config.api_key = credentials["api_key"]
        if "auth_token" in credentials:
            self.config.auth_token = credentials["auth_token"]
        if "connection_url" in credentials:
            self.config.connection_url = credentials["connection_url"]
            
        # Try to connect with the updated credentials
        if self.is_connected:
            await self.disconnect()
            
        return await self.connect()
    
    def create_rich_message(
        self,
        component_type: RichMessageComponent,
        content: Any,
        template_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a rich message component for the chat platform.
        
        Args:
            component_type: Type of rich message component
            content: Content for the component
            template_name: Optional template name to use
            
        Returns:
            Formatted rich message component
        """
        # Check if the platform supports rich messages
        if ChatFeature.RICH_MESSAGES not in self.config.supported_features:
            logger.warning(f"Platform {self.config.platform} does not support rich messages")
            # Return a simple text representation
            if isinstance(content, str):
                return {"text": content}
            else:
                return {"text": str(content)}
                
        # Check if the component type is supported
        if component_type not in self.config.supported_rich_components:
            logger.warning(f"Component type {component_type} not supported by platform {self.config.platform}")
            # Return a simple text representation
            if isinstance(content, str):
                return {"text": content}
            else:
                return {"text": str(content)}
                
        # Use template if provided
        if template_name and template_name in self.rich_message_templates:
            template = self.rich_message_templates[template_name]
            # Fill template with content
            filled_template = self._fill_template(template, content)
            return filled_template
            
        # Create component based on type
        if component_type == RichMessageComponent.BUTTON:
            return self._create_button(content)
        elif component_type == RichMessageComponent.QUICK_REPLY:
            return self._create_quick_reply(content)
        elif component_type == RichMessageComponent.CARD:
            return self._create_card(content)
        elif component_type == RichMessageComponent.CAROUSEL:
            return self._create_carousel(content)
        else:
            # Generic component
            return {
                "type": component_type.value,
                "content": content
            }
    
    def _fill_template(self, template: Dict[str, Any], content: Any) -> Dict[str, Any]:
        """
        Fill a template with content.
        
        Args:
            template: Template to fill
            content: Content to fill the template with
            
        Returns:
            Filled template
        """
        # Make a copy of the template
        result = dict(template)
        
        # Simple placeholder replacement for string values
        if isinstance(content, str):
            for key, value in result.items():
                if isinstance(value, str) and "{content}" in value:
                    result[key] = value.replace("{content}", content)
        # Complex content
        elif isinstance(content, dict):
            for key, value in result.items():
                if isinstance(value, str):
                    # Replace placeholders with content values
                    for content_key, content_value in content.items():
                        placeholder = f"{{{content_key}}}"
                        if placeholder in value:
                            result[key] = value.replace(placeholder, str(content_value))
                
        return result
    
    def _create_button(self, content: Any) -> Dict[str, Any]:
        """Create a button component."""
        if isinstance(content, str):
            return {
                "type": "button",
                "text": content,
                "value": content
            }
        elif isinstance(content, dict):
            return {
                "type": "button",
                "text": content.get("text", "Button"),
                "value": content.get("value", content.get("text", "Button")),
                "action": content.get("action", "postback")
            }
        else:
            return {
                "type": "button",
                "text": str(content),
                "value": str(content)
            }
    
    def _create_quick_reply(self, content: Any) -> Dict[str, Any]:
        """Create a quick reply component."""
        if isinstance(content, list):
            replies = []
            for item in content:
                if isinstance(item, str):
                    replies.append({
                        "content_type": "text",
                        "title": item,
                        "payload": item
                    })
                elif isinstance(item, dict):
                    replies.append({
                        "content_type": item.get("type", "text"),
                        "title": item.get("title", "Reply"),
                        "payload": item.get("payload", item.get("title", "Reply"))
                    })
            return {
                "type": "quick_replies",
                "replies": replies
            }
        else:
            return {
                "type": "quick_replies",
                "replies": [
                    {
                        "content_type": "text",
                        "title": str(content),
                        "payload": str(content)
                    }
                ]
            }
    
    def _create_card(self, content: Any) -> Dict[str, Any]:
        """Create a card component."""
        if isinstance(content, dict):
            return {
                "type": "card",
                "title": content.get("title", ""),
                "subtitle": content.get("subtitle", ""),
                "image_url": content.get("image_url", ""),
                "buttons": content.get("buttons", [])
            }
        else:
            return {
                "type": "card",
                "title": str(content),
                "subtitle": "",
                "image_url": "",
                "buttons": []
            }
    
    def _create_carousel(self, content: Any) -> Dict[str, Any]:
        """Create a carousel component."""
        if isinstance(content, list):
            items = []
            for item in content:
                if isinstance(item, dict):
                    items.append(self._create_card(item))
                else:
                    items.append(self._create_card(str(item)))
            return {
                "type": "carousel",
                "items": items
            }
        else:
            return {
                "type": "carousel",
                "items": [self._create_card(content)]
            }
    
    async def _parse_teams_message(self, raw_message: Dict[str, Any]) -> ChannelMessage:
        """Parse a Microsoft Teams message into a channel message."""
        # Extract message fields
        message_id = raw_message.get("id") or f"teams-{uuid.uuid4().hex}"
        sender_id = raw_message.get("from", {}).get("id")
        recipient_id = raw_message.get("recipient", {}).get("id") or raw_message.get("conversation", {}).get("id")
        content = raw_message.get("text") or raw_message.get("content", "")
        
        # Parse timestamp
        timestamp_str = raw_message.get("timestamp")
        if timestamp_str:
            try:
                dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                timestamp = dt.timestamp()
            except ValueError:
                timestamp = time.time()
        else:
            timestamp = time.time()
            
        # Thread/parent message reference
        references = []
        if "replyToId" in raw_message:
            references.append(raw_message["replyToId"])
            
        # Handle attachments if any
        attachments = []
        for attachment in raw_message.get("attachments", []):
            att = Attachment(
                filename=attachment.get("name", "teams_attachment"),
                content_type=attachment.get("contentType", "application/octet-stream"),
                data=attachment.get("contentUrl", ""),  # Store the URL
                size=0,  # Teams doesn't always provide size
                description=attachment.get("content")
            )
            attachments.append(att)
            
        # Create the channel message
        message = ChannelMessage(
            message_id=message_id,
            channel_type=ChannelType.CHAT,
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            timestamp=timestamp,
            attachments=attachments,
            references=references,
            status=DeliveryStatus.DELIVERED,
            content_format=ContentFormat.TEXT,
            metadata={
                "platform": "teams",
                "conversation_id": recipient_id,
                "raw_data": raw_message
            }
        )
        
        return message
    
    async def _parse_discord_message(self, raw_message: Dict[str, Any]) -> ChannelMessage:
        """Parse a Discord message into a channel message."""
        # Extract the data part if present
        if "d" in raw_message:
            data = raw_message["d"]
        else:
            data = raw_message
            
        # Extract message fields
        message_id = data.get("id") or f"discord-{uuid.uuid4().hex}"
        sender_id = data.get("author", {}).get("id") if "author" in data else data.get("user_id")
        content = data.get("content", "")
        timestamp_str = data.get("timestamp")
        
        # Parse timestamp
        if timestamp_str:
            try:
                dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                timestamp = dt.timestamp()
            except ValueError:
                timestamp = time.time()
        else:
            timestamp = time.time()
            
        # Get channel ID
        recipient_id = data.get("channel_id")
        
        # Thread/parent message reference
        references = []
        if "message_reference" in data and data["message_reference"]:
            ref_id = data["message_reference"].get("message_id")
            if ref_id:
                references.append(ref_id)
                
        # Handle attachments if any
        attachments = []
        for attachment in data.get("attachments", []):
            att = Attachment(
                filename=attachment.get("filename", "discord_attachment"),
                content_type=attachment.get("content_type", "application/octet-stream"),
                data=attachment.get("url", ""),  # Store the URL
                size=attachment.get("size", 0),
                description=None
            )
            attachments.append(att)
            
        # Create the channel message
        message = ChannelMessage(
            message_id=message_id,
            channel_type=ChannelType.CHAT,
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            timestamp=timestamp,
            attachments=attachments,
            references=references,
            status=DeliveryStatus.DELIVERED,
            content_format=ContentFormat.TEXT,
            metadata={
                "platform": "discord",
                "channel_id": recipient_id,
                "raw_data": raw_message
            }
        )
        
        return message
    
    async def _parse_generic_message(self, raw_message: Dict[str, Any]) -> ChannelMessage:
        """Parse a generic chat message into a channel message."""
        # Extract common fields
        message_id = raw_message.get("id") or f"chat-{uuid.uuid4().hex}"
        sender_id = raw_message.get("sender") or raw_message.get("user") or raw_message.get("from")
        recipient_id = raw_message.get("recipient") or raw_message.get("channel") or raw_message.get("to")
        content = raw_message.get("text") or raw_message.get("content") or raw_message.get("message", "")
        timestamp = raw_message.get("timestamp") or time.time()
        
        # Handle references/threading
        references = []
        thread_id = raw_message.get("thread_id") or raw_message.get("parent_id") or raw_message.get("in_reply_to")
        if thread_id:
            references.append(thread_id)
            
        # Extract attachments if any
        attachments = []
        for attachment in raw_message.get("attachments") or raw_message.get("files") or []:
            # Try to extract common fields
            filename = attachment.get("filename") or attachment.get("name", "attachment")
            content_type = attachment.get("content_type") or attachment.get("mimetype", "application/octet-stream")
            data = attachment.get("url") or attachment.get("data", "")
            size = attachment.get("size", 0)
            description = attachment.get("description") or attachment.get("title")
            
            att = Attachment(
                filename=filename,
                content_type=content_type,
                data=data,
                size=size,
                description=description
            )
            attachments.append(att)
            
        # Create the channel message
        message = ChannelMessage(
            message_id=message_id,
            channel_type=ChannelType.CHAT,
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            timestamp=timestamp,
            attachments=attachments,
            references=references,
            status=DeliveryStatus.DELIVERED,
            content_format=ContentFormat.TEXT,
            metadata={
                "platform": self.config.platform.value,
                "raw_data": raw_message
            }
        )
        
        return message
