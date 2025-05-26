import html
import markdown
import mimetypes
#!/usr/bin/env python3
"""
Email Channel Adapter

This module provides an implementation of the ChannelAdapter interface
for email communication, allowing the agent to send and receive messages
via email while maintaining consistent identity and conversation context.
"""

import asyncio
import email
import logging
import mimetypes
import os
import smtplib
import uuid
from datetime import datetime, timezone
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.utils import parseaddr, formataddr, make_msgid, formatdate
from typing import Dict, List, Any, Optional, Union, Tuple

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
logger = logging.getLogger("EmailChannelAdapter")


class EmailConfig:
    """Configuration for email servers and authentication."""
    
    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        imap_server: str,
        imap_port: int,
        email_address: str,
        password: str,
        use_ssl: bool = True,
        display_name: Optional[str] = None
    ):
        """
        Initialize email configuration.
        
        Args:
            smtp_server: SMTP server hostname
            smtp_port: SMTP server port
            imap_server: IMAP server hostname
            imap_port: IMAP server port
            email_address: Email address to use
            password: Password for authentication
            use_ssl: Whether to use SSL/TLS
            display_name: Display name to use (defaults to email address)
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.imap_server = imap_server
        self.imap_port = imap_port
        self.email_address = email_address
        self.password = password
        self.use_ssl = use_ssl
        self.display_name = display_name or email_address


class EmailChannelAdapter(ChannelAdapter):
    """
    Adapter for email communication.
    
    This adapter implements the ChannelAdapter interface for email communication,
    handling message formatting, sending, receiving, and delivery tracking.
    """
    
    def __init__(
        self,
        channel_id: str,
        config: EmailConfig,
        agent_id: str,
        signature: Optional[str] = None
    ):
        """
        Initialize the email channel adapter.
        
        Args:
            channel_id: Unique identifier for this channel
            config: Email configuration
            agent_id: ID of the agent using this adapter
            signature: Optional email signature to append to messages
        """
        super().__init__(ChannelType.EMAIL, channel_id)
        self.config = config
        self.agent_id = agent_id
        self.signature = signature
        
        # Message ID tracking
        self.sent_messages: Dict[str, Dict[str, Any]] = {}
        
        # Email address to entity ID mapping
        self.email_to_entity: Dict[str, str] = {}
        self.entity_to_email: Dict[str, str] = {}
        
        logger.info(f"EmailChannelAdapter initialized for {config.email_address}")
    
    def _get_capabilities(self) -> ChannelCapabilities:
        """Get the capabilities of this email channel."""
        return ChannelCapabilities(
            channel_type=ChannelType.EMAIL,
            max_message_size=25 * 1024 * 1024,  # 25 MB (typical email limit)
            supports_rich_text=True,
            supports_attachments=True,
            supports_delivery_confirmation=True,
            supports_read_receipts=True,
            supports_formatting=True,
            supported_content_formats=[
                ContentFormat.TEXT,
                ContentFormat.HTML,
                ContentFormat.MARKDOWN
            ],
            supports_threading=True,
            is_real_time=False,
            is_synchronous=False,
            throttling_limits={
                "max_recipients_per_message": 100,
                "max_messages_per_minute": 30
            },
            security_features=[
                "tls",
                "smtp_auth",
                "spam_filtering"
            ]
        )
    
    def register_email_mapping(self, email_address: str, entity_id: str) -> None:
        """
        Register a mapping between an email address and an entity ID.
        
        Args:
            email_address: Email address
            entity_id: Entity ID to associate with the email address
        """
        self.email_to_entity = {**self.email_to_entity, email_address: entity_id}
        self.entity_to_email = {**self.entity_to_email, entity_id: email_address}
        logger.debug(f"Registered mapping: {email_address} -> {entity_id}")
    
    def get_entity_for_email(self, email_address: str) -> Optional[str]:
        """
        Get the entity ID associated with an email address.
        
        Args:
            email_address: Email address to look up
            
        Returns:
            Associated entity ID if found, None otherwise
        """
        return self.email_to_entity.get(email_address)
    
    def get_email_for_entity(self, entity_id: str) -> Optional[str]:
        """
        Get the email address associated with an entity ID.
        
        Args:
            entity_id: Entity ID to look up
            
        Returns:
            Associated email address if found, None otherwise
        """
        return self.entity_to_email.get(entity_id)
    
    async def format_message(self, message: ChannelMessage) -> MIMEMultipart:
        """
        Format a channel message as an email message.
        
        Args:
            message: Channel message to format
            
        Returns:
            Formatted email message
        """
        # Get recipient email address
        recipient_email = self.get_email_for_entity(message.recipient_id)
        if not recipient_email:
            raise ValueError(f"No email address found for entity {message.recipient_id}")
            
        # Create email message
        email_msg = MIMEMultipart('alternative')
        
        # Set headers
        email_msg['From'] = formataddr((self.config.display_name, self.config.email_address))
        email_msg['To'] = recipient_email
        email_msg['Subject'] = message.subject or "Message from Adaptive Bridge Builder"
        email_msg['Date'] = formatdate(localtime=True)
        email_msg['Message-ID'] = make_msgid(domain=self.config.email_address.split('@')[1])
        
        # Add references for threading
        if message.references:
            email_msg['References'] = ' '.join(message.references)
            email_msg['In-Reply-To'] = message.references[-1]
            
        # Set priority
        if message.priority == MessagePriority.URGENT:
            email_msg['X-Priority'] = '1'
        elif message.priority == MessagePriority.HIGH:
            email_msg['X-Priority'] = '2'
        elif message.priority == MessagePriority.LOW:
            email_msg['X-Priority'] = '5'
            
        # Process content based on format
        content = message.content
        if isinstance(content, dict) or isinstance(content, list):
            content = json.dumps(content, indent=2)
            
        # Add signature if configured
        if self.signature:
            if message.content_format == ContentFormat.HTML:
                content += f"<br><br>--<br>{self.signature}"
            else:
                content += f"\n\n--\n{self.signature}"
                
        # Add appropriate content parts
        if message.content_format == ContentFormat.HTML:
            # Add both HTML and plain text versions
            plain_text = html_to_text(content)
            email_msg.attach(MIMEText(plain_text, 'plain'))
            email_msg.attach(MIMEText(content, 'html'))
        elif message.content_format == ContentFormat.MARKDOWN:
            # Convert markdown to HTML and include both
            html_content = markdown_to_html(content)
            plain_text = content  # Markdown is readable as plain text
            email_msg.attach(MIMEText(plain_text, 'plain'))
            email_msg.attach(MIMEText(html_content, 'html'))
        else:
            # Plain text only
            email_msg.attach(MIMEText(content, 'plain'))
            
        # Add attachments
        if message.attachments:
            # Convert to multipart/mixed if we have attachments
            mixed_msg = MIMEMultipart('mixed')
            
            # Copy headers from the alternative message
            for key, value in email_msg.items():
                mixed_msg[key] = value
                
            # Attach the body from the alternative message
            mixed_msg.attach(email_msg)
            
            # Add each attachment
            for attachment in message.attachments:
                part = MIMEBase(*attachment.content_type.split('/', 1))
                
                # Load attachment data
                if isinstance(attachment.data, str):
                    # Assume base64 encoded
                    import base64
                    part.set_payload(base64.b64decode(attachment.data))
                else:
                    part.set_payload(attachment.data)
                    
                # Encode payload and add headers
                email.encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename="{attachment.filename}"'
                )
                if attachment.description:
                    part.add_header('Content-Description', attachment.description)
                    
                mixed_msg.attach(part)
                
            # Use the mixed message as our final message
            email_msg = mixed_msg
            
        # Store the mapping between our message ID and the email message ID
        self.sent_messages = {**self.sent_messages, message.message_id: {}
            'email_message_id': email_msg['Message-ID'],
            'recipient': recipient_email,
            'sent_time': datetime.now(),
            'status': DeliveryStatus.PENDING.value
        }
        
        return email_msg
    
    async def send_message(self, message: ChannelMessage) -> DeliveryStatus:
        """
        Send a message via email.
        
        Args:
            message: Channel message to send
            
        Returns:
            Delivery status after attempted send
        """
        try:
            # Format the message
            email_msg = await self.format_message(message)
            
            # Connect to SMTP server
            if self.config.use_ssl:
                smtp = smtplib.SMTP_SSL(self.config.smtp_server, self.config.smtp_port)
            else:
                smtp = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
                smtp.starttls()
                
            # Authenticate
            smtp.login(self.config.email_address, self.config.password)
            
            # Send message
            recipient_email = self.get_email_for_entity(message.recipient_id)
            smtp.send_message(email_msg, from_addr=self.config.email_address, to_addrs=[recipient_email])
            
            # Close connection
            smtp.quit()
            
            # Update status
            self.sent_messages[message.message_id]['status'] = DeliveryStatus.SENT.value
            
            logger.info(f"Message {message.message_id} sent to {recipient_email}")
            return DeliveryStatus.SENT
            
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            return DeliveryStatus.FAILED
    
    async def receive_message(self, raw_message: Any) -> ChannelMessage:
        """
        Process a received email message.
        
        Args:
            raw_message: Raw email message data
            
        Returns:
            Processed channel message
        """
        # Parse email message
        if isinstance(raw_message, bytes):
            email_msg = email.message_from_bytes(raw_message)
        else:
            email_msg = raw_message
            
        # Extract sender and check if we have a mapping
        sender_email = parseaddr(email_msg['From'])[1]
        sender_id = self.get_entity_for_email(sender_email)
        
        if not sender_id:
            # Create a new entity ID for this sender
            sender_id = f"email-{uuid.uuid4().hex[:8]}"
            self.register_email_mapping(sender_email, sender_id)
            
        # Extract subject
        subject = email_msg['Subject'] or "No Subject"
        
        # Extract references for threading
        references = []
        if email_msg['References']:
            references = email_msg['References'].split()
        elif email_msg['In-Reply-To']:
            references = [email_msg['In-Reply-To']]
            
        # Extract message content
        content, content_format = self._extract_content(email_msg)
        
        # Extract attachments
        attachments = await self._extract_attachments(email_msg)
        
        # Create channel message
        message = ChannelMessage.create(
            channel_type=ChannelType.EMAIL,
            sender_id=sender_id,
            recipient_id=self.agent_id,
            content=content,
            subject=subject,
            attachments=attachments,
            content_format=content_format,
            references=references,
            metadata={
                'email_message_id': email_msg['Message-ID'],
                'sender_email': sender_email,
                'received_date': email_msg['Date']
            }
        )
        
        # Set status to delivered
        message.status = DeliveryStatus.DELIVERED
        
        return message
    
    def _extract_content(self, email_msg) -> Tuple[str, ContentFormat]:
        """
        Extract content from an email message.
        
        Args:
            email_msg: Email message to extract content from
            
        Returns:
            Tuple of (content, content_format)
        """
        content = ""
        content_format = ContentFormat.TEXT
        
        # Check for HTML content first
        html_part = None
        text_part = None
        
        if email_msg.is_multipart():
            for part in email_msg.walk():
                content_type = part.get_content_type()
                
                if content_type == "text/html":
                    html_part = part
                elif content_type == "text/plain":
                    text_part = part
                    
        else:
            # Not multipart
            content_type = email_msg.get_content_type()
            if content_type == "text/html":
                html_part = email_msg
            elif content_type == "text/plain":
                text_part = email_msg
                
        # Prefer HTML if available
        if html_part:
            content = html_part.get_payload(decode=True).decode()
            content_format = ContentFormat.HTML
        elif text_part:
            content = text_part.get_payload(decode=True).decode()
            content_format = ContentFormat.TEXT
        else:
            # No text parts found
            content = "No text content available"
            content_format = ContentFormat.TEXT
            
        return content, content_format
    
    async def _extract_attachments(self, email_msg) -> List[Attachment]:
        """
        Extract attachments from an email message.
        
        Args:
            email_msg: Email message to extract attachments from
            
        Returns:
            List of attachments
        """
        attachments = []
        
        if not email_msg.is_multipart():
            return attachments
            
        for part in email_msg.walk():
            if part.get_content_maintype() == 'multipart':
                continue
                
            # Skip text/html parts (they're part of the message body)
            if part.get_content_type() in ["text/plain", "text/html"]:
                continue
                
            # Get filename
            filename = part.get_filename()
            if not filename:
                # Generate a filename if none provided
                ext = mimetypes.guess_extension(part.get_content_type()) or '.bin'
                filename = f"attachment-{uuid.uuid4().hex[:8]}{ext}"
                
            # Decode payload
            payload = part.get_payload(decode=True)
            
            # Create attachment
            attachment = Attachment(
                filename=filename,
                content_type=part.get_content_type(),
                data=payload,
                size=len(payload),
                description=part.get("Content-Description", None)
            )
            
            attachments.append(attachment)
            
        return attachments
    
    async def check_message_status(self, message_id: str) -> DeliveryStatus:
        """
        Check the delivery status of a previously sent message.
        
        Args:
            message_id: ID of the message to check
            
        Returns:
            Current delivery status
        """
        # Email doesn't provide reliable delivery status without additional
        # protocols like DSN (Delivery Status Notification) or MDN (Message
        # Disposition Notification). We'll just return the stored status.
        
        message_info = self.sent_messages.get(message_id)
        if not message_info:
            logger.warning(f"No information found for message {message_id}")
            return DeliveryStatus.PENDING
            
        # For now, just return the stored status
        try:
            return DeliveryStatus(message_info['status'])
        except ValueError:
            logger.warning(f"Invalid status value: {message_info['status']}")
            return DeliveryStatus.PENDING
    
    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """
        Authenticate with the email server.
        
        Args:
            credentials: Authentication credentials
            
        Returns:
            Whether authentication was successful
        """
        try:
            # Extract credentials
            email_address = credentials.get('email_address', self.config.email_address)
            password = credentials.get('password', self.config.password)
            smtp_server = credentials.get('smtp_server', self.config.smtp_server)
            smtp_port = credentials.get('smtp_port', self.config.smtp_port)
            use_ssl = credentials.get('use_ssl', self.config.use_ssl)
            
            # Try to connect and authenticate with SMTP server
            if use_ssl:
                smtp = smtplib.SMTP_SSL(smtp_server, smtp_port)
            else:
                smtp = smtplib.SMTP(smtp_server, smtp_port)
                smtp.starttls()
                
            # Authenticate
            smtp.login(email_address, password)
            
            # Close connection
            smtp.quit()
            
            logger.info(f"Successfully authenticated with {smtp_server}")
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False


# Helper functions

def html_to_text(html_content: str) -> str:
    """
    Convert HTML content to plain text.
    
    This is a simple implementation. For production use, consider
    using a dedicated library like beautiful_soup or html2text.
    
    Args:
        html_content: HTML content to convert
        
    Returns:
        Plain text version
    """
    import re
    
    # Remove style and script elements
    html_content = re.sub(r'<(style|script)[^>]*>.*?</\1>', '', html_content, flags=re.DOTALL)
    
    # Replace <br> with newline
    html_content = re.sub(r'<br[^>]*>', '\n', html_content)
    
    # Replace <p> with double newline
    html_content = re.sub(r'<p[^>]*>', '\n\n', html_content)
    
    # Remove all other HTML tags
    html_content = re.sub(r'<[^>]*>', '', html_content)
    
    # Replace multiple whitespace with single space
    html_content = re.sub(r'[ \t]+', ' ', html_content)
    
    # Replace multiple newlines with double newline
    html_content = re.sub(r'\n{3,}', '\n\n', html_content)
    
    # Decode HTML entities
    from html import unescape
    html_content = unescape(html_content)
    
    return html_content.strip()


def markdown_to_html(markdown_content: str) -> str:
    """
    Convert Markdown content to HTML.
    
    This is a placeholder implementation. For production use,
    consider using a dedicated library like markdown or commonmark.
    
    Args:
        markdown_content: Markdown content to convert
        
    Returns:
        HTML version
    """
    try:
        # Try to use markdown library if available
        import markdown
        return markdown.markdown(markdown_content)
    except ImportError:
        # Fallback to very basic conversion
        import re
        
        # Convert headers
        html_content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', markdown_content, flags=re.MULTILINE)
        html_content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html_content, flags=re.MULTILINE)
        html_content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html_content, flags=re.MULTILINE)
        
        # Convert bold and italic
        html_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_content)
        html_content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html_content)
        
        # Convert links
        html_content = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', html_content)
        
        # Convert paragraphs
        paragraphs = html_content.split('\n\n')
        html_content = ''.join([f'<p>{p}</p>' for p in paragraphs if p.strip()])
        
        return html_content