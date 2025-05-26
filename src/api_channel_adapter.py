#!/usr/bin/env python3
"""
API Channel Adapter

This module provides an implementation of the ChannelAdapter interface
for API-based communication, allowing the agent to interact with external
systems via RESTful APIs, webhooks, and other API-based mechanisms.
"""

import asyncio
import json
import logging
import time
import uuid
import aiohttp
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin

from communication_channel_manager import (
    ChannelAdapter, ChannelType, ChannelCapabilities, ChannelMessage,
    Attachment, DeliveryStatus, SecurityLevel, MessagePriority
)
from content_handler import ContentFormat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ApiChannelAdapter")


class ApiAuthType(str, Enum):
    """Types of API authentication."""
    NONE = "none"                   # No authentication
    API_KEY = "api_key"             # API key authentication
    BASIC = "basic"                 # Basic authentication
    BEARER = "bearer"               # Bearer token authentication
    OAUTH = "oauth"                 # OAuth authentication
    CUSTOM = "custom"               # Custom authentication


class ApiMethod(str, Enum):
    """HTTP methods for API requests."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"


class ApiConfig:
    """Configuration for API communication."""
    
    def __init__(
        self,
        base_url: str,
        auth_type: ApiAuthType = ApiAuthType.NONE,
        auth_params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        webhook_url: Optional[str] = None
    ):
        """
        Initialize API configuration.
        
        Args:
            base_url: Base URL for API requests
            auth_type: Type of authentication to use
            auth_params: Authentication parameters
            headers: Default headers to include in all requests
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            webhook_url: URL for receiving webhook callbacks
        """
        self.base_url = base_url
        self.auth_type = auth_type
        self.auth_params = auth_params or {}
        self.headers = headers or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.webhook_url = webhook_url


class ApiChannelAdapter(ChannelAdapter):
    """
    Adapter for API-based communication.
    
    This adapter implements the ChannelAdapter interface for API-based
    communication, handling message formatting, sending, and receiving
    through REST APIs and webhooks.
    """
    
    def __init__(
        self,
        channel_id: str,
        config: ApiConfig,
        agent_id: str,
        endpoints: Optional[Dict[str, str]] = None,
        custom_formatters: Optional[Dict[str, Callable]] = None,
        client_session: Optional[aiohttp.ClientSession] = None
    ):
        """
        Initialize the API channel adapter.
        
        Args:
            channel_id: Unique identifier for this channel
            config: API configuration
            agent_id: ID of the agent using this adapter
            endpoints: Mapping of endpoint names to relative URLs
            custom_formatters: Custom message formatters for specific endpoints
            client_session: Optional aiohttp session for requests
        """
        super().__init__(ChannelType.API, channel_id)
        self.config = config
        self.agent_id = agent_id
        self.endpoints = endpoints or {
            "send": "/messages",
            "status": "/messages/{message_id}/status",
            "receive": "/messages/incoming"
        }
        self.custom_formatters = custom_formatters or {}
        self.message_status_cache: Dict[str, DeliveryStatus] = {}
        
        # Entity to API endpoint mapping
        self.entity_endpoints: Dict[str, Dict[str, str]] = {}
        
        # Create or use provided HTTP client session
        self.client_session = client_session
        self.is_session_owner = client_session is None
        
        logger.info(f"ApiChannelAdapter initialized for {config.base_url}")
    
    async def __aenter__(self) -> Any:
        """Async context manager entry."""
        if self.is_session_owner and self.client_session is None:
            self.client_session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Async context manager exit."""
        if self.is_session_owner and self.client_session is not None:
            await self.client_session.close()
            self.client_session = None
    
    def _get_capabilities(self) -> ChannelCapabilities:
        """Get the capabilities of this API channel."""
        return ChannelCapabilities(
            channel_type=ChannelType.API,
            max_message_size=10 * 1024 * 1024,  # 10 MB
            supports_rich_text=True,
            supports_attachments=True,
            supports_delivery_confirmation=True,
            supports_read_receipts=False,
            supports_formatting=True,
            supported_content_formats=[
                ContentFormat.JSON,
                ContentFormat.TEXT,
                ContentFormat.XML
            ],
            supports_threading=True,
            is_real_time=False,
            is_synchronous=True,
            throttling_limits={
                "max_requests_per_minute": 60,
                "max_parallel_requests": 10
            },
            security_features=[
                "tls",
                "api_key",
                "oauth",
                "bearer_token",
                "request_signing"
            ]
        )
    
    def register_entity_endpoint(
        self,
        entity_id: str,
        endpoint_name: str,
        endpoint_url: str
    ) -> None:
        """
        Register an API endpoint for a specific entity.
        
        Args:
            entity_id: ID of the entity
            endpoint_name: Name of the endpoint
            endpoint_url: URL of the endpoint
        """
        if entity_id not in self.entity_endpoints:
            self.entity_endpoints = {**self.entity_endpoints, entity_id: {}}
            
        self.entity_endpoints[entity_id][endpoint_name] = endpoint_url
        logger.debug(f"Registered {endpoint_name} endpoint for entity {entity_id}: {endpoint_url}")
    
    def get_entity_endpoint(
        self,
        entity_id: str,
        endpoint_name: str
    ) -> Optional[str]:
        """
        Get an API endpoint for a specific entity.
        
        Args:
            entity_id: ID of the entity
            endpoint_name: Name of the endpoint
            
        Returns:
            Endpoint URL if found, None otherwise
        """
        return self.entity_endpoints.get(entity_id, {}).get(
            endpoint_name,
            self.endpoints.get(endpoint_name)
        )
    
    async def _apply_auth(
        self,
        request_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply authentication to a request.
        
        Args:
            request_kwargs: Request keyword arguments
            
        Returns:
            Updated request keyword arguments
        """
        auth_type = self.config.auth_type
        auth_params = self.config.auth_params
        
        if auth_type == ApiAuthType.NONE:
            return request_kwargs
            
        if auth_type == ApiAuthType.API_KEY:
            # API key can be in header, query param, or cookie
            key_name = auth_params.get("key_name", "api_key")
            key_value = auth_params.get("key_value", "")
            key_location = auth_params.get("key_location", "header")
            
            if key_location == "header":
                headers = request_kwargs.get("headers", {})
                headers[key_name] = key_value
                request_kwargs["headers"] = headers
            elif key_location == "query":
                params = request_kwargs.get("params", {})
                params[key_name] = key_value
                request_kwargs["params"] = params
            elif key_location == "cookie":
                cookies = request_kwargs.get("cookies", {})
                cookies[key_name] = key_value
                request_kwargs["cookies"] = cookies
                
        elif auth_type == ApiAuthType.BASIC:
            # Basic authentication
            username = auth_params.get("username", "")
            password = auth_params.get("password", "")
            
            if "auth" not in request_kwargs:
                request_kwargs["auth"] = aiohttp.BasicAuth(username, password)
                
        elif auth_type == ApiAuthType.BEARER:
            # Bearer token authentication
            token = auth_params.get("token", "")
            headers = request_kwargs.get("headers", {})
            headers["Authorization"] = f"Bearer {token}"
            request_kwargs["headers"] = headers
            
        elif auth_type == ApiAuthType.OAUTH:
            # OAuth authentication - get token first if not provided
            token = auth_params.get("token")
            
            if not token and "token_url" in auth_params:
                # Request an OAuth token
                token_url = auth_params["token_url"]
                client_id = auth_params.get("client_id", "")
                client_secret = auth_params.get("client_secret", "")
                scope = auth_params.get("scope", "")
                
                # Prepare token request
                token_request = {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "grant_type": "client_credentials"
                }
                
                if scope:
                    token_request["scope"] = scope
                    
                try:
                    if self.client_session is None:
                        async with aiohttp.ClientSession() as session:
                            async with session.post(token_url, data=token_request) as response:
                                token_data = await response.json()
                                token = token_data.get("access_token")
                    else:
                        async with self.client_session.post(token_url, data=token_request) as response:
                            token_data = await response.json()
                            token = token_data.get("access_token")
                except Exception as e:
                    logger.error(f"Failed to obtain OAuth token: {str(e)}")
                    
            # Apply token if available
            if token:
                headers = request_kwargs.get("headers", {})
                headers["Authorization"] = f"Bearer {token}"
                request_kwargs["headers"] = headers
                
        elif auth_type == ApiAuthType.CUSTOM:
            # Custom authentication - use callback if provided
            auth_callback = auth_params.get("auth_callback")
            if auth_callback and callable(auth_callback):
                request_kwargs = await auth_callback(request_kwargs)
                
        return request_kwargs
    
    async def _make_request(
        self,
        method: ApiMethod,
        url: str,
        **kwargs
    ) -> Tuple[int, Any, Dict[str, str]]:
        """
        Make an HTTP request with retries and error handling.
        
        Args:
            method: HTTP method to use
            url: URL to request
            **kwargs: Additional request arguments
            
        Returns:
            Tuple of (status_code, response_data, response_headers)
        """
        # Apply authentication
        kwargs = await self._apply_auth(kwargs)
        
        # Apply default headers
        headers = dict(self.config.headers)
        headers.update(kwargs.pop("headers", {}))
        
        # Set timeout if not provided
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.config.timeout
            
        # Initialize retry counter
        retries = 0
        last_error = None
        
        # Create session if needed
        if self.client_session is None:
            self.client_session = aiohttp.ClientSession()
            self.is_session_owner = True
            
        # Make request with retries
        while retries <= self.config.max_retries:
            try:
                method_func = getattr(self.client_session, method.lower())
                async with method_func(url, headers=headers, **kwargs) as response:
                    # Check for rate limiting
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", self.config.retry_delay))
                        logger.warning(f"Rate limited. Waiting {retry_after} seconds before retry.")
                        await asyncio.sleep(retry_after)
                        retries += 1
                        continue
                        
                    # Check for server errors
                    if response.status >= 500:
                        logger.warning(f"Server error: {response.status}. Retrying...")
                        await asyncio.sleep(self.config.retry_delay * (2 ** retries))
                        retries += 1
                        continue
                        
                    # Get response data based on content type
                    content_type = response.headers.get("Content-Type", "")
                    
                    if "application/json" in content_type:
                        data = await response.json()
                    elif "text/" in content_type:
                        data = await response.text()
                    else:
                        data = await response.read()
                        
                    return response.status, data, dict(response.headers)
                    
            except aiohttp.ClientError as e:
                logger.error(f"Request error: {str(e)}")
                last_error = e
                
            # Exponential backoff
            await asyncio.sleep(self.config.retry_delay * (2 ** retries))
            retries += 1
            
        # If we get here, all retries failed
        logger.error(f"Failed after {retries} retries: {str(last_error)}")
        raise last_error or Exception("Request failed after all retries")
    
    async def format_message(self, message: ChannelMessage) -> Dict[str, Any]:
        """
        Format a channel message for API transmission.
        
        Args:
            message: Channel message to format
            
        Returns:
            Formatted API message
        """
        # Check for custom formatter for this entity
        entity_id = message.recipient_id
        endpoint = self.get_entity_endpoint(entity_id, "send")
        
        if endpoint in self.custom_formatters:
            return self.custom_formatters[endpoint](message)
            
        # Standard API message format
        formatted_message = {
            "message_id": message.message_id,
            "sender_id": message.sender_id,
            "recipient_id": message.recipient_id,
            "subject": message.subject,
            "timestamp": message.timestamp,
            "priority": message.priority.value,
            "content_format": message.content_format.value,
            "references": message.references,
            "metadata": message.metadata
        }
        
        # Format content based on content format
        if message.content_format == ContentFormat.JSON:
            if isinstance(message.content, str):
                try:
                    # Try to parse JSON string
                    formatted_message["content"] = json.loads(message.content)
                except json.JSONDecodeError:
                    # Treat as regular text if not valid JSON
                    formatted_message["content"] = message.content
                    formatted_message["content_format"] = ContentFormat.TEXT.value
            else:
                # Already a Python object, include directly
                formatted_message["content"] = message.content
        else:
            # Other formats - include as is
            formatted_message["content"] = message.content
            
        # Handle attachments
        if message.attachments:
            formatted_message["attachments"] = []
            
            for attachment in message.attachments:
                formatted_attachment = {
                    "filename": attachment.filename,
                    "content_type": attachment.content_type,
                    "size": attachment.size,
                    "description": attachment.description
                }
                
                # Add data as base64 if not already
                if isinstance(attachment.data, bytes):
                    import base64
                    formatted_attachment["data"] = base64.b64encode(attachment.data).decode('utf-8')
                else:
                    formatted_attachment["data"] = attachment.data
                    
                formatted_message["attachments"].append(formatted_attachment)
                
        return formatted_message
    
    async def send_message(self, message: ChannelMessage) -> DeliveryStatus:
        """
        Send a message through the API.
        
        Args:
            message: Channel message to send
            
        Returns:
            Delivery status after attempted send
        """
        try:
            # Determine endpoint for this recipient
            endpoint = self.get_entity_endpoint(message.recipient_id, "send")
            if not endpoint:
                logger.error(f"No 'send' endpoint configured for {message.recipient_id}")
                return DeliveryStatus.FAILED
                
            # Build full URL
            if endpoint.startswith(("http://", "https://")):
                url = endpoint
            else:
                url = urljoin(self.config.base_url, endpoint)
                
            # Format message
            api_message = await self.format_message(message)
            
            # Make request
            status_code, response_data, headers = await self._make_request(
                method=ApiMethod.POST,
                url=url,
                json=api_message
            )
            
            # Process response
            if 200 <= status_code < 300:
                # Success
                self.message_status_cache = {**self.message_status_cache, message.message_id: DeliveryStatus.SENT}
                
                # Check if response includes a status field
                if isinstance(response_data, dict) and "status" in response_data:
                    try:
                        status = DeliveryStatus(response_data["status"])
                        self.message_status_cache = {**self.message_status_cache, message.message_id: status}
                        return status
                    except ValueError:
                        # Invalid status value, ignore
                        pass
                        
                return DeliveryStatus.SENT
            else:
                # Error
                logger.error(f"Failed to send message: HTTP {status_code}, response: {response_data}")
                self.message_status_cache = {**self.message_status_cache, message.message_id: DeliveryStatus.FAILED}
                return DeliveryStatus.FAILED
                
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            self.message_status_cache = {**self.message_status_cache, message.message_id: DeliveryStatus.FAILED}
            return DeliveryStatus.FAILED
    
    async def receive_message(self, raw_message: Any) -> ChannelMessage:
        """
        Process a received API message.
        
        Args:
            raw_message: Raw API message
            
        Returns:
            Processed channel message
        """
        # Parse the raw message
        if isinstance(raw_message, bytes):
            try:
                message_data = json.loads(raw_message.decode('utf-8'))
            except json.JSONDecodeError:
                # Not JSON, treat as text
                message_data = {"content": raw_message.decode('utf-8')}
        elif isinstance(raw_message, str):
            try:
                message_data = json.loads(raw_message)
            except json.JSONDecodeError:
                # Not JSON, treat as text
                message_data = {"content": raw_message}
        else:
            # Assume it's already parsed
            message_data = raw_message
            
        # Extract message fields
        message_id = message_data.get("message_id", f"api-msg-{uuid.uuid4().hex}")
        sender_id = message_data.get("sender_id", "unknown")
        recipient_id = message_data.get("recipient_id", self.agent_id)
        subject = message_data.get("subject")
        content = message_data.get("content", "")
        timestamp = message_data.get("timestamp", time.time())
        session_id = message_data.get("session_id")
        references = message_data.get("references", [])
        metadata = message_data.get("metadata", {})
        
        # Determine content format
        content_format_str = message_data.get("content_format", "text")
        try:
            content_format = ContentFormat(content_format_str)
        except ValueError:
            content_format = ContentFormat.TEXT
            
        # Process attachments if any
        attachments = []
        for attachment_data in message_data.get("attachments", []):
            attachment = Attachment(
                filename=attachment_data.get("filename", "attachment.bin"),
                content_type=attachment_data.get("content_type", "application/octet-stream"),
                data=attachment_data.get("data", ""),
                size=attachment_data.get("size", 0),
                description=attachment_data.get("description")
            )
            attachments.append(attachment)
            
        # Create channel message
        message = ChannelMessage(
            message_id=message_id,
            channel_type=ChannelType.API,
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            timestamp=timestamp,
            session_id=session_id,
            subject=subject,
            attachments=attachments,
            content_format=content_format,
            metadata=metadata,
            references=references,
            status=DeliveryStatus.DELIVERED
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
        # Check local cache first
        if message_id in self.message_status_cache:
            return self.message_status_cache[message_id]
            
        # Determine endpoint
        endpoint = self.endpoints.get("status", "").replace("{message_id}", message_id)
        if not endpoint:
            logger.warning("No status endpoint configured")
            return DeliveryStatus.PENDING
            
        # Build full URL
        if endpoint.startswith(("http://", "https://")):
            url = endpoint
        else:
            url = urljoin(self.config.base_url, endpoint)
            
        try:
            # Make request
            status_code, response_data, headers = await self._make_request(
                method=ApiMethod.GET,
                url=url
            )
            
            # Process response
            if 200 <= status_code < 300:
                if isinstance(response_data, dict) and "status" in response_data:
                    try:
                        status = DeliveryStatus(response_data["status"])
                        self.message_status_cache = {**self.message_status_cache, message_id: status}
                        return status
                    except ValueError:
                        logger.warning(f"Invalid status value: {response_data['status']}")
                        
                # Default to delivered if we got a success response but no valid status
                return DeliveryStatus.DELIVERED
            else:
                logger.error(f"Failed to check message status: HTTP {status_code}")
                return DeliveryStatus.PENDING
                
        except Exception as e:
            logger.error(f"Error checking message status: {str(e)}")
            return DeliveryStatus.PENDING
    
    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """
        Authenticate with the API.
        
        Args:
            credentials: Authentication credentials
            
        Returns:
            Whether authentication was successful
        """
        # Get URL from credentials or use default
        auth_url = credentials.get("auth_url")
        if not auth_url:
            # Use base URL if no specific auth URL provided
            auth_url = self.config.base_url
            
        # Prepare auth request based on auth type
        auth_type = credentials.get("auth_type", self.config.auth_type)
        request_kwargs = {"headers": {}}
        
        if auth_type == ApiAuthType.API_KEY:
            key_name = credentials.get("key_name", "api_key")
            key_value = credentials.get("key_value", "")
            key_location = credentials.get("key_location", "header")
            
            if key_location == "header":
                request_kwargs["headers"][key_name] = key_value
            elif key_location == "query":
                request_kwargs["params"] = {key_name: key_value}
                
        elif auth_type == ApiAuthType.BASIC:
            username = credentials.get("username", "")
            password = credentials.get("password", "")
            request_kwargs["auth"] = aiohttp.BasicAuth(username, password)
            
        elif auth_type == ApiAuthType.BEARER:
            token = credentials.get("token", "")
            request_kwargs["headers"]["Authorization"] = f"Bearer {token}"
            
        elif auth_type == ApiAuthType.OAUTH:
            # For OAuth, we'll test the token endpoint
            token_url = credentials.get("token_url", "")
            if not token_url:
                logger.error("No token URL provided for OAuth authentication")
                return False
                
            client_id = credentials.get("client_id", "")
            client_secret = credentials.get("client_secret", "")
            
            # Use the token URL instead
            auth_url = token_url
            request_kwargs["data"] = {
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret
            }
            
        try:
            # Make request to verify authentication
            status_code, response_data, headers = await self._make_request(
                method=ApiMethod.GET if auth_type != ApiAuthType.OAUTH else ApiMethod.POST,
                url=auth_url,
                **request_kwargs
            )
            
            # 2xx status codes indicate success
            return 200 <= status_code < 300
            
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False
    
    def register_webhook_handler(self, handler: Callable[[Any], Awaitable[bool]]) -> None:
        """
        Register a callback for handling incoming webhook events.
        
        Args:
            handler: Async callback function that takes a webhook payload
                    and returns a boolean indicating success
        """
        self.webhook_handler = handler
        
    async def process_webhook(self, payload: Any) -> bool:
        """
        Process an incoming webhook payload.
        
        Args:
            payload: Webhook payload to process
            
        Returns:
            Whether the webhook was processed successfully
        """
        if not hasattr(self, 'webhook_handler'):
            logger.warning("No webhook handler registered")
            return False
            
        try:
            # Convert webhook payload to a channel message
            message = await self.receive_message(payload)
            
            # Call the webhook handler
            return await self.webhook_handler(message)
            
        except Exception as e:
            logger.error(f"Error processing webhook: {str(e)}")
            return False
