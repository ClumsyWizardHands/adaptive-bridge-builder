import imaplib
import mimetypes
"""
ApiGatewaySystem Email Extension

This module extends the ApiGatewaySystem to handle email communication,
enabling agents to read, send, and organize emails using A2A Protocol and
Empire Framework components.
"""

import asyncio
import base64
import email
import imaplib
import json
import logging
import mimetypes
import os
import re
import smtplib
import time
import uuid

from datetime import datetime, timezone
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.utils import parseaddr, formataddr, make_msgid, formatdate
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from enum import Enum

# Import ApiGatewaySystem components
from api_gateway_system import (
    ApiGatewaySystem, ApiConfig, EndpointConfig, AuthConfig, AuthType,
    DataFormat, HttpMethod, CacheConfig, RateLimitConfig, ErrorHandlingConfig,
    AuditLogEntry, LogLevel
)

# Import EmailChannelAdapter for email handling
from email_channel_adapter import (
    EmailConfig, EmailChannelAdapter, ChannelMessage, ChannelType,
    ContentFormat, Attachment, DeliveryStatus, SecurityLevel, MessagePriority
)

# Import Empire Framework components
from principle_engine import PrincipleEngine
from principle_engine_example import PrincipleEvaluationRequest
from empire_framework.a2a.component_task_handler import Task, ComponentTaskTypes, TaskStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ApiGatewaySystemEmail")


class EmailOperation(Enum):
    """Email operations supported by the system."""
    READ = "read"                 # Read emails from server
    SEND = "send"                 # Send an email
    SEARCH = "search"             # Search for emails
    LIST_FOLDERS = "list_folders" # List email folders
    CREATE_FOLDER = "create_folder" # Create email folder
    MOVE = "move"                 # Move email to folder
    DELETE = "delete"             # Delete email
    MARK_READ = "mark_read"       # Mark email as read
    MARK_UNREAD = "mark_unread"   # Mark email as unread
    FLAG = "flag"                 # Flag email
    UNFLAG = "unflag"             # Unflag email
    DOWNLOAD_ATTACHMENT = "download_attachment" # Download attachment


class EmailTaskTypes:
    """A2A Task types for email operations."""
    FETCH_EMAILS = "email.fetch"         # Fetch emails from server
    SEND_EMAIL = "email.send"            # Send an email
    SEARCH_EMAILS = "email.search"       # Search for emails
    ORGANIZE_EMAILS = "email.organize"   # Organize emails (move, flag, etc.)
    PROCESS_ATTACHMENTS = "email.attachments"  # Process email attachments
    ANALYZE_EMAILS = "email.analyze"     # Analyze email content
    COMPOSE_RESPONSE = "email.compose"   # Compose email response with principles


class EmailSecurityLevel(Enum):
    """Security levels for email content."""
    PUBLIC = "public"             # Public content
    INTERNAL = "internal"         # Internal only
    CONFIDENTIAL = "confidential" # Confidential
    SENSITIVE = "sensitive"       # Sensitive content
    RESTRICTED = "restricted"     # Restricted access


class EmailServiceAdapter:
    """
    Adapter for connecting email services to the ApiGatewaySystem.
    
    This adapter enables the ApiGatewaySystem to interact with email servers,
    handling email reading, sending, and organization through a standardized API.
    """
    
    def __init__(
        self,
        api_gateway: ApiGatewaySystem,
        email_config: EmailConfig,
        principle_engine: Optional[PrincipleEngine] = None,
        agent_id: str = "email-gateway-agent",
        security_level: EmailSecurityLevel = EmailSecurityLevel.INTERNAL
    ):
        """
        Initialize the email service adapter.
        
        Args:
            api_gateway: ApiGatewaySystem to extend
            email_config: Email configuration
            principle_engine: Optional principle engine for decision-making
            agent_id: ID of the agent using the adapter
            security_level: Default security level for emails
        """
        self._background_tasks: List[asyncio.Task] = []
        self.api_gateway = api_gateway
        self.email_config = email_config
        self.principle_engine = principle_engine
        self.agent_id = agent_id
        self.security_level = security_level
        
        # Initialize email channel adapter
        self.channel_adapter = EmailChannelAdapter(
            channel_id=f"email-channel-{uuid.uuid4().hex[:8]}",
            config=email_config,
            agent_id=agent_id
        )
        
        # Create email API configuration
        self._register_email_api()
        
        # Track operations for auditing
        self.operation_history: List[Dict[str, Any]] = []
        
        logger.info(f"EmailServiceAdapter initialized for {email_config.email_address}")
    
    def _register_email_api(self) -> None:
        """Register the email API with the ApiGatewaySystem."""
        # Create authentication configuration
        auth_config = AuthConfig(
            auth_type=AuthType.BASIC,
            credentials={
                "username": self.email_config.email_address,
                "password": self.email_config.password
            }
        )
        
        # Create endpoints for email operations
        endpoints = {
            "read": EndpointConfig(
                name="read_emails",
                url="/email/read",
                method=HttpMethod.GET,
                auth=auth_config,
                description="Read emails from server"
            ),
            "send": EndpointConfig(
                name="send_email",
                url="/email/send",
                method=HttpMethod.POST,
                auth=auth_config,
                description="Send an email"
            ),
            "search": EndpointConfig(
                name="search_emails",
                url="/email/search",
                method=HttpMethod.GET,
                auth=auth_config,
                description="Search for emails"
            ),
            "list_folders": EndpointConfig(
                name="list_folders",
                url="/email/folders",
                method=HttpMethod.GET,
                auth=auth_config,
                description="List email folders"
            ),
            "create_folder": EndpointConfig(
                name="create_folder",
                url="/email/folders",
                method=HttpMethod.POST,
                auth=auth_config,
                description="Create an email folder"
            ),
            "move": EndpointConfig(
                name="move_email",
                url="/email/move",
                method=HttpMethod.PUT,
                auth=auth_config,
                description="Move an email to a folder"
            ),
            "delete": EndpointConfig(
                name="delete_email",
                url="/email/delete",
                method=HttpMethod.DELETE,
                auth=auth_config,
                description="Delete an email"
            ),
            "mark_read": EndpointConfig(
                name="mark_read",
                url="/email/mark-read",
                method=HttpMethod.PUT,
                auth=auth_config,
                description="Mark an email as read"
            ),
            "mark_unread": EndpointConfig(
                name="mark_unread",
                url="/email/mark-unread",
                method=HttpMethod.PUT,
                auth=auth_config,
                description="Mark an email as unread"
            ),
            "flag": EndpointConfig(
                name="flag_email",
                url="/email/flag",
                method=HttpMethod.PUT,
                auth=auth_config,
                description="Flag an email"
            ),
            "unflag": EndpointConfig(
                name="unflag_email",
                url="/email/unflag",
                method=HttpMethod.PUT,
                auth=auth_config,
                description="Unflag an email"
            ),
            "download_attachment": EndpointConfig(
                name="download_attachment",
                url="/email/attachment",
                method=HttpMethod.GET,
                auth=auth_config,
                description="Download an email attachment"
            )
        }
        
        # Create API configuration
        email_api_config = ApiConfig(
            name="email_service",
            base_url=f"imap://{self.email_config.imap_server}:{self.email_config.imap_port}",
            auth=auth_config,
            endpoints=endpoints,
            description="Email service API",
            tags=["email", "communication"],
            version="1.0"
        )
        
        # Register the API with the gateway
        self.api_gateway.register_api(email_api_config)
        
        logger.info(f"Email API registered with API Gateway")
    
    async def authenticate(self) -> bool:
        """
        Authenticate with the email server.
        
        Returns:
            Whether authentication was successful
        """
        return await self.channel_adapter.authenticate({
            "email_address": self.email_config.email_address,
            "password": self.email_config.password,
            "smtp_server": self.email_config.smtp_server,
            "smtp_port": self.email_config.smtp_port,
            "use_ssl": self.email_config.use_ssl
        })
    
    async def execute_operation(
        self,
        operation: EmailOperation,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute an email operation.
        
        Args:
            operation: Email operation to execute
            params: Parameters for the operation
            
        Returns:
            Result of the operation
        """
        try:
            # Track operation start time for history
            start_time = time.time()
            
            # Authenticate if needed
            if not await self.authenticate():
                raise ValueError(f"Authentication failed for {self.email_config.email_address}")
            
            # Apply principle evaluation if available
            approved = True
            evaluation_result = None
            if self.principle_engine:
                approved, evaluation_result = await self._evaluate_with_principles(operation, params)
                
                if not approved:
                    logger.warning(f"Operation {operation.value} rejected by principle evaluation")
                    return {
                        "success": False,
                        "error": "Operation rejected by principle evaluation",
                        "principle_evaluation": evaluation_result
                    }
            
            # Execute operation
            result = await self._dispatch_operation(operation, params)
            
            # Track in operation history
            elapsed_time = time.time() - start_time
            self.operation_history.append({
                "operation": operation.value,
                "params": params,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": result.get("success", False),
                "duration_ms": int(elapsed_time * 1000),
                "principle_evaluation": evaluation_result
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing email operation {operation.value}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _dispatch_operation(
        self,
        operation: EmailOperation,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Dispatch an email operation to the appropriate handler.
        
        Args:
            operation: Email operation to execute
            params: Parameters for the operation
            
        Returns:
            Result of the operation
        """
        # Dispatch to the appropriate handler
        if operation == EmailOperation.READ:
            return await self._read_emails(params)
        elif operation == EmailOperation.SEND:
            return await self._send_email(params)
        elif operation == EmailOperation.SEARCH:
            return await self._search_emails(params)
        elif operation == EmailOperation.LIST_FOLDERS:
            return await self._list_folders(params)
        elif operation == EmailOperation.CREATE_FOLDER:
            return await self._create_folder(params)
        elif operation == EmailOperation.MOVE:
            return await self._move_email(params)
        elif operation == EmailOperation.DELETE:
            return await self._delete_email(params)
        elif operation == EmailOperation.MARK_READ:
            return await self._mark_email(params, read=True)
        elif operation == EmailOperation.MARK_UNREAD:
            return await self._mark_email(params, read=False)
        elif operation == EmailOperation.FLAG:
            return await self._flag_email(params, flag=True)
        elif operation == EmailOperation.UNFLAG:
            return await self._flag_email(params, flag=False)
        elif operation == EmailOperation.DOWNLOAD_ATTACHMENT:
            return await self._download_attachment(params)
        else:
            raise ValueError(f"Unsupported email operation: {operation}")
    
    async def _evaluate_with_principles(
        self,
        operation: EmailOperation,
        params: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate an operation against principles.
        
        Args:
            operation: Email operation to evaluate
            params: Parameters for the operation
            
        Returns:
            Tuple of (approved, evaluation_result)
        """
        if not self.principle_engine:
            return True, None
            
        # Create evaluation request
        request = PrincipleEvaluationRequest(
            content=f"Email operation: {operation.value}",
            context={
                "operation": operation.value,
                "parameters": params,
                "agent_id": self.agent_id,
                "email_address": self.email_config.email_address,
                "security_level": self.security_level.value,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
        # Evaluate against principles
        result = await self.principle_engine.evaluate_content(request)
        
        # Check if operation is approved
        approved = result.get("score", 0) >= 0.7  # Threshold for approval
        
        return approved, result
    
    # Email operation implementations
    
    async def _read_emails(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Read emails from the server.
        
        Args:
            params: Parameters for the operation
                - folder: Email folder to read from (default: "INBOX")
                - limit: Maximum number of emails to read
                - offset: Number of emails to skip
                - unread_only: Whether to read only unread emails
                
        Returns:
            Result of the operation with emails
        """
        try:
            # Extract parameters
            folder = params.get("folder", "INBOX")
            limit = params.get("limit", 10)
            offset = params.get("offset", 0)
            unread_only = params.get("unread_only", False)
            
            # Connect to IMAP server
            mail = await self._connect_to_imap()
            
            # Select the folder
            mail.select(folder)
            
            # Search for emails
            search_criteria = "UNSEEN" if unread_only else "ALL"
            status, data = mail.search(None, search_criteria)
            
            if status != "OK":
                raise ValueError(f"Failed to search for emails: {status}")
                
            # Get message IDs
            message_ids = data[0].split()
            
            # Apply pagination
            message_ids = message_ids[offset:offset+limit]
            
            # Fetch emails
            emails = []
            for msg_id in message_ids:
                status, data = mail.fetch(msg_id, "(RFC822)")
                
                if status != "OK":
                    logger.warning(f"Failed to fetch email {msg_id}")
                    continue
                    
                # Process email
                email_data = data[0][1]
                msg = email.message_from_bytes(email_data)
                
                # Create channel message
                channel_message = await self.channel_adapter.receive_message(msg)
                
                # Extract key information
                emails.append({
                    "message_id": channel_message.message_id,
                    "subject": channel_message.subject,
                    "sender": channel_message.sender_id,
                    "date": channel_message.metadata.get("received_date"),
                    "content": channel_message.content,
                    "content_format": channel_message.content_format.value,
                    "has_attachments": len(channel_message.attachments) > 0,
                    "attachment_count": len(channel_message.attachments),
                    "imap_id": msg_id.decode("utf-8")
                })
            
            # Close the connection
            mail.close()
            mail.logout()
            
            return {
                "success": True,
                "emails": emails,
                "total_count": len(message_ids),
                "folder": folder,
                "unread_only": unread_only
            }
            
        except Exception as e:
            logger.error(f"Error reading emails: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _send_email(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send an email.
        
        Args:
            params: Parameters for the operation
                - recipient_email: Email address of the recipient
                - subject: Email subject
                - content: Email content
                - content_format: Format of the content (TEXT, HTML, MARKDOWN)
                - attachments: Optional list of attachments
                - priority: Optional priority (NORMAL, HIGH, LOW, URGENT)
                
        Returns:
            Result of the operation
        """
        try:
            # Extract parameters
            recipient_email = params.get("recipient_email")
            subject = params.get("subject", "")
            content = params.get("content", "")
            content_format_str = params.get("content_format", "TEXT")
            attachments_data = params.get("attachments", [])
            priority_str = params.get("priority", "NORMAL")
            
            if not recipient_email:
                raise ValueError("Missing recipient_email parameter")
                
            # Convert string enums to actual enum values
            try:
                content_format = ContentFormat(content_format_str)
            except ValueError:
                content_format = ContentFormat.TEXT
                
            try:
                priority = MessagePriority(priority_str)
            except ValueError:
                priority = MessagePriority.NORMAL
            
            # Register recipient mapping if not already registered
            recipient_id = self.channel_adapter.get_entity_for_email(recipient_email)
            if not recipient_id:
                recipient_id = f"email-recipient-{uuid.uuid4().hex[:8]}"
                self.channel_adapter.register_email_mapping(recipient_email, recipient_id)
            
            # Process attachments
            attachments = []
            for attachment_data in attachments_data:
                filename = attachment_data.get("filename", f"attachment-{uuid.uuid4().hex[:8]}")
                content_type = attachment_data.get("content_type", "application/octet-stream")
                data = attachment_data.get("data", "")
                description = attachment_data.get("description")
                
                # Create attachment
                attachment = Attachment(
                    filename=filename,
                    content_type=content_type,
                    data=data,
                    size=len(data) if isinstance(data, bytes) else len(data.encode("utf-8")),
                    description=description
                )
                
                attachments.append(attachment)
            
            # Create channel message
            message = ChannelMessage.create(
                channel_type=ChannelType.EMAIL,
                sender_id=self.agent_id,
                recipient_id=recipient_id,
                content=content,
                subject=subject,
                attachments=attachments,
                content_format=content_format,
                priority=priority
            )
            
            # Send the message
            status = await self.channel_adapter.send_message(message)
            
            return {
                "success": status == DeliveryStatus.SENT,
                "message_id": message.message_id,
                "status": status.value,
                "recipient": recipient_email,
                "subject": subject
            }
            
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _search_emails(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for emails.
        
        Args:
            params: Parameters for the operation
                - folder: Email folder to search in (default: "INBOX")
                - query: Search query
                - criteria: Search criteria (FROM, TO, SUBJECT, BODY, ALL)
                - limit: Maximum number of emails to return
                
        Returns:
            Result of the operation with matching emails
        """
        try:
            # Extract parameters
            folder = params.get("folder", "INBOX")
            query = params.get("query", "")
            criteria = params.get("criteria", "ALL")
            limit = params.get("limit", 10)
            
            if not query:
                raise ValueError("Missing query parameter")
                
            # Connect to IMAP server
            mail = await self._connect_to_imap()
            
            # Select the folder
            mail.select(folder)
            
            # Prepare search criteria
            if criteria == "FROM":
                search_cmd = f'(FROM "{query}")'
            elif criteria == "TO":
                search_cmd = f'(TO "{query}")'
            elif criteria == "SUBJECT":
                search_cmd = f'(SUBJECT "{query}")'
            elif criteria == "BODY":
                search_cmd = f'(BODY "{query}")'
            else:
                # Search in all fields
                search_cmd = f'(OR OR OR (FROM "{query}") (TO "{query}") (SUBJECT "{query}") (BODY "{query}"))'
            
            # Search for emails
            status, data = mail.search(None, search_cmd)
            
            if status != "OK":
                raise ValueError(f"Failed to search for emails: {status}")
                
            # Get message IDs
            message_ids = data[0].split()
            
            # Limit results
            message_ids = message_ids[:limit]
            
            # Fetch emails
            emails = []
            for msg_id in message_ids:
                status, data = mail.fetch(msg_id, "(RFC822)")
                
                if status != "OK":
                    logger.warning(f"Failed to fetch email {msg_id}")
                    continue
                    
                # Process email
                email_data = data[0][1]
                msg = email.message_from_bytes(email_data)
                
                # Create channel message
                channel_message = await self.channel_adapter.receive_message(msg)
                
                # Extract key information
                emails.append({
                    "message_id": channel_message.message_id,
                    "subject": channel_message.subject,
                    "sender": channel_message.sender_id,
                    "date": channel_message.metadata.get("received_date"),
                    "content": channel_message.content,
                    "content_format": channel_message.content_format.value,
                    "has_attachments": len(channel_message.attachments) > 0,
                    "attachment_count": len(channel_message.attachments),
                    "imap_id": msg_id.decode("utf-8")
                })
            
            # Close the connection
            mail.close()
            mail.logout()
            
            return {
                "success": True,
                "emails": emails,
                "total_count": len(message_ids),
                "query": query,
                "criteria": criteria,
                "folder": folder
            }
            
        except Exception as e:
            logger.error(f"Error searching emails: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _list_folders(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        List email folders.
        
        Args:
            params: Parameters for the operation (none required)
                
        Returns:
            Result of the operation with folders
        """
        try:
            # Connect to IMAP server
            mail = await self._connect_to_imap()
            
            # List folders
            status, folder_data = mail.list()
            
            if status != "OK":
                raise ValueError(f"Failed to list folders: {status}")
                
            # Parse folders
            folders = []
            for folder_line in folder_data:
                folder_info = folder_line.decode("utf-8")
                
                # Extract folder name with regex
                match = re.search(r'"([^"]+)"$', folder_info)
                if match:
                    folder_name = match.group(1)
                    folders.append(folder_name)
                    
            # Logout
            mail.logout()
            
            return {
                "success": True,
                "folders": folders,
                "count": len(folders)
            }
            
        except Exception as e:
            logger.error(f"Error listing folders: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _create_folder(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an email folder.
        
        Args:
            params: Parameters for the operation
                - folder_name: Name of the folder to create
                
        Returns:
            Result of the operation
        """
        try:
            # Extract parameters
            folder_name = params.get("folder_name")
            
            if not folder_name:
                raise ValueError("Missing folder_name parameter")
                
            # Connect to IMAP server
            mail = await self._connect_to_imap()
            
            # Create folder
            status, data = mail.create(folder_name)
            
            if status != "OK":
                raise ValueError(f"Failed to create folder: {status}")
                
            # Logout
            mail.logout()
            
            return {
                "success": True,
                "folder_name": folder_name
            }
            
        except Exception as e:
            logger.error(f"Error creating folder: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _move_email(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move an email to a different folder.
        
        Args:
            params: Parameters for the operation
                - message_id: IMAP ID of the email to move
                - source_folder: Source folder
                - destination_folder: Destination folder
                
        Returns:
            Result of the operation
        """
        try:
            # Extract parameters
            message_id = params.get("message_id")
            source_folder = params.get("source_folder", "INBOX")
            destination_folder = params.get("destination_folder")
            
            if not message_id:
                raise ValueError("Missing message_id parameter")
                
            if not destination_folder:
                raise ValueError("Missing destination_folder parameter")
                
            # Connect to IMAP server
            mail = await self._connect_to_imap()
            
            # Select source folder
            mail.select(source_folder)
            
            # Copy message to destination folder
            status, data = mail.copy(message_id, destination_folder)
            
            if status != "OK":
                raise ValueError(f"Failed to copy message to destination folder: {status}")
                
            # Mark original as deleted
            mail.store(message_id, "+FLAGS", "\\Deleted")
            
            # Expunge to actually delete
            mail.expunge()
            
            # Logout
            mail.logout()
            
            return {
                "success": True,
                "message_id": message_id,
                "source_folder": source_folder,
                "destination_folder": destination_folder
            }
            
        except Exception as e:
            logger.error(f"Error moving email: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _delete_email(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delete an email.
        
        Args:
            params: Parameters for the operation
                - message_id: IMAP ID of the email to delete
                - folder: Folder containing the email
                
        Returns:
            Result of the operation
        """
        try:
            # Extract parameters
            message_id = params.get("message_id")
            folder = params.get("folder", "INBOX")
            
            if not message_id:
                raise ValueError("Missing message_id parameter")
                
            # Connect to IMAP server
            mail = await self._connect_to_imap()
            
            # Select folder
            mail.select(folder)
            
            # Mark as deleted
            mail.store(message_id, "+FLAGS", "\\Deleted")
            
            # Expunge to actually delete
            mail.expunge()
            
            # Logout
            mail.logout()
            
            return {
                "success": True,
                "message_id": message_id,
                "folder": folder
            }
            
        except Exception as e:
            logger.error(f"Error deleting email: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _mark_email(self, params: Dict[str, Any], read: bool) -> Dict[str, Any]:
        """
        Mark an email as read or unread.
        
        Args:
            params: Parameters for the operation
                - message_id: IMAP ID of the email to mark
                - folder: Folder containing the email
            read: Whether to mark as read (True) or unread (False)
                
        Returns:
            Result of the operation
        """
        try:
            # Extract parameters
            message_id = params.get("message_id")
            folder = params.get("folder", "INBOX")
            
            if not message_id:
                raise ValueError("Missing message_id parameter")
                
            # Connect to IMAP server
            mail = await self._connect_to_imap()
            
            # Select folder
            mail.select(folder)
            
            # Mark as read or unread
            if read:
                mail.store(message_id, "+FLAGS", "\\Seen")
            else:
                mail.store(message_id, "-FLAGS", "\\Seen")
                
            # Logout
            mail.logout()
            
            return {
                "success": True,
                "message_id": message_id,
                "folder": folder,
                "marked_as": "read" if read else "unread"
            }
            
        except Exception as e:
            logger.error(f"Error marking email as {'read' if read else 'unread'}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _flag_email(self, params: Dict[str, Any], flag: bool) -> Dict[str, Any]:
        """
        Flag or unflag an email.
        
        Args:
            params: Parameters for the operation
                - message_id: IMAP ID of the email to flag
                - folder: Folder containing the email
            flag: Whether to flag (True) or unflag (False)
                
        Returns:
            Result of the operation
        """
        try:
            # Extract parameters
            message_id = params.get("message_id")
            folder = params.get("folder", "INBOX")
            
            if not message_id:
                raise ValueError("Missing message_id parameter")
                
            # Connect to IMAP server
            mail = await self._connect_to_imap()
            
            # Select folder
            mail.select(folder)
            
            # Flag or unflag
            if flag:
                mail.store(message_id, "+FLAGS", "\\Flagged")
            else:
                mail.store(message_id, "-FLAGS", "\\Flagged")
                
            # Logout
            mail.logout()
            
            return {
                "success": True,
                "message_id": message_id,
                "folder": folder,
                "action": "flagged" if flag else "unflagged"
            }
            
        except Exception as e:
            logger.error(f"Error {'flagging' if flag else 'unflagging'} email: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _download_attachment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Download an email attachment.
        
        Args:
            params: Parameters for the operation
                - message_id: IMAP ID of the email
                - attachment_index: Index of the attachment to download
                - folder: Folder containing the email
                
        Returns:
            Result of the operation with attachment data
        """
        try:
            # Extract parameters
            message_id = params.get("message_id")
            attachment_index = params.get("attachment_index", 0)
            folder = params.get("folder", "INBOX")
            
            if not message_id:
                raise ValueError("Missing message_id parameter")
                
            # Connect to IMAP server
            mail = await self._connect_to_imap()
            
            # Select folder
            mail.select(folder)
            
            # Fetch email
            status, data = mail.fetch(message_id, "(RFC822)")
            
            if status != "OK":
                raise ValueError(f"Failed to fetch email {message_id}")
                
            # Process email
            email_data = data[0][1]
            msg = email.message_from_bytes(email_data)
            
            # Create channel message to parse attachments
            channel_message = await self.channel_adapter.receive_message(msg)
            
            # Check if attachment exists
            if attachment_index >= len(channel_message.attachments):
                raise ValueError(f"Attachment index {attachment_index} out of range")
                
            # Get attachment
            attachment = channel_message.attachments[attachment_index]
            
            # Logout
            mail.logout()
            
            # Convert binary data to base64 if needed
            if isinstance(attachment.data, bytes):
                attachment_data = base64.b64encode(attachment.data).decode("utf-8")
            else:
                attachment_data = attachment.data
            
            return {
                "success": True,
                "filename": attachment.filename,
                "content_type": attachment.content_type,
                "size": attachment.size,
                "data": attachment_data,
                "description": attachment.description
            }
            
        except Exception as e:
            logger.error(f"Error downloading attachment: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _connect_to_imap(self) -> None:
        """
        Connect to the IMAP server.
        
        Returns:
            IMAP connection
        """
        # Connect to server
        if self.email_config.use_ssl:
            mail = imaplib.IMAP4_SSL(self.email_config.imap_server, self.email_config.imap_port)
        else:
            mail = imaplib.IMAP4(self.email_config.imap_server, self.email_config.imap_port)
            
        # Login
        mail.login(self.email_config.email_address, self.email_config.password)
        
        return mail

    # A2A Task Handling

    async def create_email_task(
        self,
        task_type: str,
        task_data: Dict[str, Any],
        priority: str = "medium"
    ) -> str:
        """
        Create an A2A task for email operations.
        
        Args:
            task_type: Type of email task
            task_data: Data needed for the task
            priority: Task priority
            
        Returns:
            ID of the created task
        """
        # Create task
        task = Task(
            task_id=f"email-task-{uuid.uuid4().hex[:8]}",
            task_type=task_type,
            component_ids=[],
            task_data=task_data,
            priority=priority,
            created_by=self.agent_id
        )
        
        # Execute task (in reality you would use a task handler)
        create_email_task_task = asyncio.create_task(self._process_email_task(task))
        
        return task.task_id
    
    async def _process_email_task(self, task: Task) -> None:
        """
        Process an email task.
        
        Args:
            task: Task to process
        """
        try:
            # Mark as processing
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.now(timezone.utc).isoformat()
            
            # Process based on task type
            if task.task_type == EmailTaskTypes.FETCH_EMAILS:
                result = await self._process_fetch_emails_task(task)
            elif task.task_type == EmailTaskTypes.SEND_EMAIL:
                result = await self._process_send_email_task(task)
            elif task.task_type == EmailTaskTypes.SEARCH_EMAILS:
                result = await self._process_search_emails_task(task)
            elif task.task_type == EmailTaskTypes.ORGANIZE_EMAILS:
                result = await self._process_organize_emails_task(task)
            elif task.task_type == EmailTaskTypes.PROCESS_ATTACHMENTS:
                result = await self._process_attachments_task(task)
            elif task.task_type == EmailTaskTypes.ANALYZE_EMAILS:
                result = await self._process_analyze_emails_task(task)
            elif task.task_type == EmailTaskTypes.COMPOSE_RESPONSE:
                result = await self._process_compose_response_task(task)
            else:
                raise ValueError(f"Unknown email task type: {task.task_type}")
                
            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc).isoformat()
            task.progress = 1.0
            task.result = result
            
        except Exception as e:
            # Mark as failed
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now(timezone.utc).isoformat()
            task.error = {
                "message": str(e),
                "stack_trace": ""  # In a real implementation, you'd include the stack trace
            }
    
    async def _process_fetch_emails_task(self, task: Task) -> Dict[str, Any]:
        """
        Process a fetch emails task.
        
        Args:
            task: Task to process
            
        Returns:
            Task result
        """
        # Extract task data
        folder = task.task_data.get("folder", "INBOX")
        limit = task.task_data.get("limit", 10)
        offset = task.task_data.get("offset", 0)
        unread_only = task.task_data.get("unread_only", False)
        
        # Execute operation
        result = await self.execute_operation(
            EmailOperation.READ,
            {
                "folder": folder,
                "limit": limit,
                "offset": offset,
                "unread_only": unread_only
            }
        )
        
        return result
    
    async def _process_send_email_task(self, task: Task) -> Dict[str, Any]:
        """
        Process a send email task.
        
        Args:
            task: Task to process
            
        Returns:
            Task result
        """
        # Execute operation
        result = await self.execute_operation(
            EmailOperation.SEND,
            task.task_data
        )
        
        return result
    
    async def _process_search_emails_task(self, task: Task) -> Dict[str, Any]:
        """
        Process a search emails task.
        
        Args:
            task: Task to process
            
        Returns:
            Task result
        """
        # Execute operation
        result = await self.execute_operation(
            EmailOperation.SEARCH,
            task.task_data
        )
        
        return result
    
    async def _process_organize_emails_task(self, task: Task) -> Dict[str, Any]:
        """
        Process an organize emails task.
        
        Args:
            task: Task to process
            
        Returns:
            Task result
        """
        # Extract task data
        operation_type = task.task_data.get("operation_type")
        emails = task.task_data.get("emails", [])
        target_folder = task.task_data.get("target_folder")
        
        # Check required fields
        if not operation_type:
            raise ValueError("Missing operation_type in task data")
            
        if not emails:
            raise ValueError("Missing emails in task data")
            
        # Execute operations
        results = []
        for email_info in emails:
            message_id = email_info.get("message_id")
            folder = email_info.get("folder", "INBOX")
            
            if not message_id:
                continue
                
            # Execute based on operation type
            if operation_type == "move" and target_folder:
                # Move email
                result = await self.execute_operation(
                    EmailOperation.MOVE,
                    {
                        "message_id": message_id,
                        "source_folder": folder,
                        "destination_folder": target_folder
                    }
                )
            elif operation_type == "delete":
                # Delete email
                result = await self.execute_operation(
                    EmailOperation.DELETE,
                    {
                        "message_id": message_id,
                        "folder": folder
                    }
                )
            elif operation_type == "mark_read":
                # Mark as read
                result = await self.execute_operation(
                    EmailOperation.MARK_READ,
                    {
                        "message_id": message_id,
                        "folder": folder
                    }
                )
            elif operation_type == "mark_unread":
                # Mark as unread
                result = await self.execute_operation(
                    EmailOperation.MARK_UNREAD,
                    {
                        "message_id": message_id,
                        "folder": folder
                    }
                )
            elif operation_type == "flag":
                # Flag email
                result = await self.execute_operation(
                    EmailOperation.FLAG,
                    {
                        "message_id": message_id,
                        "folder": folder
                    }
                )
            elif operation_type == "unflag":
                # Unflag email
                result = await self.execute_operation(
                    EmailOperation.UNFLAG,
                    {
                        "message_id": message_id,
                        "folder": folder
                    }
                )
            else:
                result = {
                    "success": False,
                    "error": f"Unsupported operation type: {operation_type}"
                }
                
            # Add to results
            results.append({
                "message_id": message_id,
                "result": result
            })
            
            # Update progress
            task.progress = len(results) / len(emails)
            
        return {
            "operation_type": operation_type,
            "success_count": sum(1 for r in results if r["result"]["success"]),
            "failure_count": sum(1 for r in results if not r["result"]["success"]),
            "results": results
        }
    
    async def _process_attachments_task(self, task: Task) -> Dict[str, Any]:
        """
        Process an email attachments task.
        
        Args:
            task: Task to process
            
        Returns:
            Task result
        """
        # Extract task data
        email_id = task.task_data.get("email_id")
        folder = task.task_data.get("folder", "INBOX")
        
        if not email_id:
            raise ValueError("Missing email_id in task data")
            
        # Fetch email to get attachments
        email_result = await self.execute_operation(
            EmailOperation.READ,
            {
                "message_ids": [email_id],
                "folder": folder
            }
        )
        
        if not email_result["success"] or not email_result["emails"]:
            raise ValueError(f"Failed to fetch email {email_id}")
            
        email_data = email_result["emails"][0]
        
        # Check if email has attachments
        if not email_data["has_attachments"]:
            return {
                "success": True,
                "message": "No attachments found",
                "attachments": []
            }
            
        # Download each attachment
        attachments = []
        for i in range(email_data["attachment_count"]):
            result = await self.execute_operation(
                EmailOperation.DOWNLOAD_ATTACHMENT,
                {
                    "message_id": email_id,
                    "attachment_index": i,
                    "folder": folder
                }
            )
            
            if result["success"]:
                attachments.append({
                    "index": i,
                    "filename": result["filename"],
                    "content_type": result["content_type"],
                    "size": result["size"],
                    "data": result["data"]
                })
                
            # Update progress
            task.progress = (i + 1) / email_data["attachment_count"]
            
        return {
            "success": True,
            "email_id": email_id,
            "attachment_count": len(attachments),
            "attachments": attachments
        }
    
    async def _process_analyze_emails_task(self, task: Task) -> Dict[str, Any]:
        """
        Process an analyze emails task.
        
        Args:
            task: Task to process
            
        Returns:
            Task result
        """
        # Extract task data
        emails = task.task_data.get("emails", [])
        analysis_type = task.task_data.get("analysis_type", "content")
        
        if not emails:
            raise ValueError("Missing emails in task data")
            
        # Analyze emails
        analysis_results = []
        for i, email_info in enumerate(emails):
            # Extract email content
            content = email_info.get("content", "")
            subject = email_info.get("subject", "")
            sender = email_info.get("sender", "")
            
            # Perform analysis (in a real implementation, this would use more sophisticated techniques)
            if analysis_type == "content":
                # Simple content analysis
                result = {
                    "email_id": email_info.get("message_id"),
                    "length": len(content),
                    "word_count": len(content.split()),
                    "has_questions": "?" in content,
                    "has_links": "http" in content.lower(),
                    "sentiment": "neutral"  # Placeholder
                }
            elif analysis_type == "priority":
                # Priority determination
                priority = "normal"
                if any(word in subject.lower() for word in ["urgent", "important", "critical", "asap"]):
                    priority = "high"
                    
                result = {
                    "email_id": email_info.get("message_id"),
                    "priority": priority,
                    "urgency_keywords": [word for word in ["urgent", "important", "critical", "asap"] 
                                        if word in subject.lower() or word in content.lower()]
                }
            else:
                result = {
                    "email_id": email_info.get("message_id"),
                    "error": f"Unsupported analysis type: {analysis_type}"
                }
                
            analysis_results.append(result)
            
            # Update progress
            task.progress = (i + 1) / len(emails)
            
        return {
            "success": True,
            "analysis_type": analysis_type,
            "email_count": len(emails),
            "results": analysis_results
        }
    
    async def _process_compose_response_task(self, task: Task) -> Dict[str, Any]:
        """
        Process a compose response task.
        
        Args:
            task: Task to process
            
        Returns:
            Task result
        """
        # Extract task data
        email_info = task.task_data.get("email")
        response_type = task.task_data.get("response_type", "standard")
        custom_content = task.task_data.get("custom_content")
        
        if not email_info:
            raise ValueError("Missing email in task data")
            
        # Extract email details
        sender = email_info.get("sender")
        subject = email_info.get("subject", "")
        content = email_info.get("content", "")
        
        # Check if we have a sender
        if not sender:
            raise ValueError("Missing sender in email data")
            
        # Get sender email
        sender_email = self.channel_adapter.get_email_for_entity(sender)
        if not sender_email:
            raise ValueError(f"No email address found for sender {sender}")
            
        # Prepare response
        if custom_content:
            response_content = custom_content
        else:
            # Generate response based on type (in a real implementation, this would be more sophisticated)
            if response_type == "standard":
                response_content = f"Thank you for your message regarding '{subject}'."
            elif response_type == "acknowledgment":
                response_content = f"This is to acknowledge receipt of your message regarding '{subject}'."
            elif response_type == "inquiry":
                response_content = f"Thank you for your message. Regarding '{subject}', could you please provide more information?"
            else:
                response_content = f"Thank you for your message."
                
        # Apply principle evaluation if available
        evaluation = None
        if self.principle_engine:
            request = PrincipleEvaluationRequest(
                content=response_content,
                context={
                    "email_subject": subject,
                    "email_content": content,
                    "response_type": response_type,
                    "recipient": sender_email
                }
            )
            
            evaluation = await self.principle_engine.evaluate_content(request)
            
            # Adjust response if needed (in a real implementation, this would be more sophisticated)
            if evaluation.get("score", 1.0) < 0.7:
                response_content = f"Thank you for your message. I will get back to you soon with a more detailed response."
        
        # Compose response
        response_subject = f"Re: {subject}" if not subject.startswith("Re:") else subject
        
        # Send email
        result = await self.execute_operation(
            EmailOperation.SEND,
            {
                "recipient_email": sender_email,
                "subject": response_subject,
                "content": response_content,
                "content_format": "TEXT",
                "priority": "NORMAL"
            }
        )
        
        return {
            "success": result["success"],
            "message_id": result.get("message_id"),
            "recipient": sender_email,
            "subject": response_subject,
            "response_type": response_type,
            "principle_application": evaluation
        }