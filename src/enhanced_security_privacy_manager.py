"""
Enhanced Security & Privacy Manager for Empire Framework

This module provides a comprehensive security and privacy management system
that integrates with the Empire Framework and A2A Protocol. It handles secure
authentication, permission enforcement, privacy principles, data access logging,
and ethical decision-making through PrincipleEngine integration.
"""

import asyncio
import base64
import copy
import datetime
from datetime import timezone
import hashlib
import hmac
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable, TypeVar, Generic

# Import Empire Framework components
from principle_engine import PrincipleEngine
from principle_engine_example import PrincipleEvaluationRequest

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EnhancedSecurityPrivacyManager")


class SecurityLevel(Enum):
    """Security levels for data and operations."""
    PUBLIC = auto()      # Public data, accessible to anyone
    PROTECTED = auto()   # Protected data, requires basic authentication
    PRIVATE = auto()     # Private data, requires specific permissions
    SENSITIVE = auto()   # Sensitive data, requires elevated permissions
    CRITICAL = auto()    # Critical data, requires highest level of permission


class DataCategory(Enum):
    """Categories of data for privacy purposes."""
    BASIC = auto()       # Basic information (name, public profile, etc.)
    CONTACT = auto()     # Contact information (email, phone, etc.)
    DEMOGRAPHIC = auto() # Demographic information (age, gender, etc.)
    PREFERENCE = auto()  # User preferences
    BEHAVIORAL = auto()  # Behavioral data (usage patterns, etc.)
    LOCATION = auto()    # Location data
    BIOMETRIC = auto()   # Biometric data
    FINANCIAL = auto()   # Financial information
    HEALTH = auto()      # Health information
    CREDENTIAL = auto()  # Authentication credentials
    CONTENT = auto()     # User-generated content
    METADATA = auto()    # Metadata about other data


class OperationType(Enum):
    """Types of operations performed on data."""
    READ = auto()        # Reading data
    CREATE = auto()      # Creating new data
    UPDATE = auto()      # Updating existing data
    DELETE = auto()      # Deleting data
    SHARE = auto()       # Sharing data with others
    EXPORT = auto()      # Exporting data from the system
    AGGREGATE = auto()   # Aggregating data
    ANALYZE = auto()     # Analyzing data
    INFER = auto()       # Inferring information from data
    TRANSFER = auto()    # Transferring data to another system


class AccessDeniedReason(Enum):
    """Reasons for denying access to data or operations."""
    AUTHENTICATION_FAILURE = auto()  # Failed authentication
    INSUFFICIENT_PERMISSIONS = auto() # Insufficient permissions
    PRIVACY_VIOLATION = auto()       # Violation of privacy rules
    PRINCIPLE_VIOLATION = auto()     # Violation of principle evaluation
    RATE_LIMIT_EXCEEDED = auto()     # Rate limit exceeded
    SECURITY_POLICY = auto()         # Violation of security policy
    DATA_MINIMIZATION = auto()       # Violates data minimization principle
    PURPOSE_LIMITATION = auto()      # Outside the scope of specified purpose
    TOKEN_EXPIRED = auto()           # Authentication token expired
    CONTEXT_MISMATCH = auto()        # Request context doesn't match allowed context


@dataclass
class TokenData:
    """Secure token data for authentication."""
    token_id: str
    user_id: str
    created_at: datetime.datetime
    expires_at: datetime.datetime
    scopes: List[str] = field(default_factory=list)
    service_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    refresh_token: Optional[str] = None
    is_revoked: bool = False
    last_used: Optional[datetime.datetime] = None
    
    def is_expired(self) -> bool:
        """Check if the token has expired."""
        return datetime.datetime.now() > self.expires_at
    
    def has_scope(self, scope: str) -> bool:
        """Check if the token has a specific scope."""
        return scope in self.scopes or '*' in self.scopes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert token data to a dictionary."""
        return {
            "token_id": self.token_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "scopes": self.scopes,
            "service_name": self.service_name,
            "metadata": self.metadata,
            "is_revoked": self.is_revoked,
            "last_used": self.last_used.isoformat() if self.last_used else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TokenData':
        """Create token data from a dictionary."""
        return cls(
            token_id=data["token_id"],
            user_id=data["user_id"],
            created_at=datetime.datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.datetime.fromisoformat(data["expires_at"]),
            scopes=data.get("scopes", []),
            service_name=data.get("service_name"),
            metadata=data.get("metadata", {}),
            refresh_token=data.get("refresh_token"),
            is_revoked=data.get("is_revoked", False),
            last_used=datetime.datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None
        )


@dataclass
class Permission:
    """Permission definition for access control."""
    resource_type: str
    action: str
    resource_id: Optional[str] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def get_key(self) -> str:
        """Get a unique key for this permission."""
        resource_key = f"{self.resource_id}" if self.resource_id else "*"
        return f"{self.resource_type}:{resource_key}:{self.action}"
    
    def matches(self, resource_type: str, action: str, resource_id: Optional[str] = None) -> bool:
        """
        Check if this permission matches the requested access.
        
        Args:
            resource_type: Type of resource to access
            action: Action to perform
            resource_id: Specific resource ID or None for all
            
        Returns:
            Whether permission matches the requested access
        """
        if self.resource_type != resource_type and self.resource_type != "*":
            return False
            
        if self.action != action and self.action != "*":
            return False
            
        if resource_id and self.resource_id and self.resource_id != resource_id and self.resource_id != "*":
            return False
            
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert permission to a dictionary."""
        return {
            "resource_type": self.resource_type,
            "action": self.action,
            "resource_id": self.resource_id,
            "conditions": self.conditions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Permission':
        """Create permission from a dictionary."""
        return cls(
            resource_type=data["resource_type"],
            action=data["action"],
            resource_id=data.get("resource_id"),
            conditions=data.get("conditions", {})
        )


@dataclass
class PrivacyRule:
    """Privacy rule definition for enforcing privacy principles."""
    rule_id: str
    data_categories: List[DataCategory]
    operation_types: List[OperationType]
    security_level: SecurityLevel
    purpose_limitation: Optional[List[str]] = None
    retention_period: Optional[datetime.timedelta] = None
    requires_consent: bool = False
    requires_logging: bool = True
    anonymization_required: bool = False
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def applies_to(self, data_category: DataCategory, operation_type: OperationType) -> bool:
        """
        Check if this rule applies to the data category and operation.
        
        Args:
            data_category: Category of data being accessed
            operation_type: Type of operation being performed
            
        Returns:
            Whether the rule applies
        """
        return data_category in self.data_categories and operation_type in self.operation_types
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert privacy rule to a dictionary."""
        return {
            "rule_id": self.rule_id,
            "data_categories": [dc.name for dc in self.data_categories],
            "operation_types": [ot.name for ot in self.operation_types],
            "security_level": self.security_level.name,
            "purpose_limitation": self.purpose_limitation,
            "retention_period": self.retention_period.total_seconds() if self.retention_period else None,
            "requires_consent": self.requires_consent,
            "requires_logging": self.requires_logging,
            "anonymization_required": self.anonymization_required,
            "conditions": self.conditions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PrivacyRule':
        """Create privacy rule from a dictionary."""
        return cls(
            rule_id=data["rule_id"],
            data_categories=[DataCategory[dc] for dc in data["data_categories"]],
            operation_types=[OperationType[ot] for ot in data["operation_types"]],
            security_level=SecurityLevel[data["security_level"]],
            purpose_limitation=data.get("purpose_limitation"),
            retention_period=datetime.timedelta(seconds=data["retention_period"]) if data.get("retention_period") else None,
            requires_consent=data.get("requires_consent", False),
            requires_logging=data.get("requires_logging", True),
            anonymization_required=data.get("anonymization_required", False),
            conditions=data.get("conditions", {})
        )


@dataclass
class AccessLogEntry:
    """Log entry for data access."""
    entry_id: str
    timestamp: datetime.datetime
    user_id: str
    resource_type: str
    resource_id: Optional[str]
    action: str
    data_categories: List[DataCategory]
    operation_type: OperationType
    access_granted: bool
    reason_denied: Optional[AccessDeniedReason] = None
    client_info: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert access log entry to a dictionary."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "data_categories": [dc.name for dc in self.data_categories],
            "operation_type": self.operation_type.name,
            "access_granted": self.access_granted,
            "reason_denied": self.reason_denied.name if self.reason_denied else None,
            "client_info": self.client_info,
            "context": self.context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AccessLogEntry':
        """Create access log entry from a dictionary."""
        return cls(
            entry_id=data["entry_id"],
            timestamp=datetime.datetime.fromisoformat(data["timestamp"]),
            user_id=data["user_id"],
            resource_type=data["resource_type"],
            resource_id=data["resource_id"],
            action=data["action"],
            data_categories=[DataCategory[dc] for dc in data["data_categories"]],
            operation_type=OperationType[data["operation_type"]],
            access_granted=data["access_granted"],
            reason_denied=AccessDeniedReason[data["reason_denied"]] if data.get("reason_denied") else None,
            client_info=data.get("client_info", {}),
            context=data.get("context", {})
        )


@dataclass
class DataAccessRequest:
    """Request to access data."""
    request_id: str
    user_id: str
    resource_type: str
    resource_id: Optional[str]
    action: str
    data_categories: List[DataCategory]
    operation_type: OperationType
    purpose: str
    context: Dict[str, Any] = field(default_factory=dict)
    token_data: Optional[TokenData] = None
    client_info: Dict[str, Any] = field(default_factory=dict)
    requires_principle_evaluation: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert data access request to a dictionary."""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "data_categories": [dc.name for dc in self.data_categories],
            "operation_type": self.operation_type.name,
            "purpose": self.purpose,
            "context": self.context,
            "token_data": self.token_data.to_dict() if self.token_data else None,
            "client_info": self.client_info,
            "requires_principle_evaluation": self.requires_principle_evaluation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataAccessRequest':
        """Create data access request from a dictionary."""
        return cls(
            request_id=data["request_id"],
            user_id=data["user_id"],
            resource_type=data["resource_type"],
            resource_id=data["resource_id"],
            action=data["action"],
            data_categories=[DataCategory[dc] for dc in data["data_categories"]],
            operation_type=OperationType[data["operation_type"]],
            purpose=data["purpose"],
            context=data.get("context", {}),
            token_data=TokenData.from_dict(data["token_data"]) if data.get("token_data") else None,
            client_info=data.get("client_info", {}),
            requires_principle_evaluation=data.get("requires_principle_evaluation", False)
        )


@dataclass
class DataAccessResult:
    """Result of a data access request."""
    request_id: str
    access_granted: bool
    reason_denied: Optional[AccessDeniedReason] = None
    log_entry: Optional[AccessLogEntry] = None
    principle_evaluation: Optional[Dict[str, Any]] = None
    anonymized: bool = False
    privacy_warning: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert data access result to a dictionary."""
        return {
            "request_id": self.request_id,
            "access_granted": self.access_granted,
            "reason_denied": self.reason_denied.name if self.reason_denied else None,
            "log_entry": self.log_entry.to_dict() if self.log_entry else None,
            "principle_evaluation": self.principle_evaluation,
            "anonymized": self.anonymized,
            "privacy_warning": self.privacy_warning
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataAccessResult':
        """Create data access result from a dictionary."""
        return cls(
            request_id=data["request_id"],
            access_granted=data["access_granted"],
            reason_denied=AccessDeniedReason[data["reason_denied"]] if data.get("reason_denied") else None,
            log_entry=AccessLogEntry.from_dict(data["log_entry"]) if data.get("log_entry") else None,
            principle_evaluation=data.get("principle_evaluation"),
            anonymized=data.get("anonymized", False),
            privacy_warning=data.get("privacy_warning")
        )


class EnhancedSecurityPrivacyManager:
    """
    Comprehensive security and privacy management system for Empire Framework.
    
    This manager handles:
    1. Secure authentication token management
    2. User permission enforcement
    3. Privacy principle enforcement
    4. Access logging for sensitive data
    5. Data access visibility
    6. Integration with PrincipleEngine for ethical decisions
    """
    
    def __init__(
        self,
        token_secret: str,
        principle_engine: Optional[PrincipleEngine] = None,
        token_expiry: datetime.timedelta = datetime.timedelta(hours=24),
        access_log_retention: datetime.timedelta = datetime.timedelta(days=90),
        enable_detailed_logging: bool = True
    ):
        """
        Initialize the security and privacy manager.
        
        Args:
            token_secret: Secret key for token encryption
            principle_engine: PrincipleEngine for ethical evaluations
            token_expiry: Default expiry time for tokens
            access_log_retention: Retention period for access logs
            enable_detailed_logging: Whether to enable detailed logging
        """
        self.token_secret = token_secret
        self.principle_engine = principle_engine
        self.token_expiry = token_expiry
        self.access_log_retention = access_log_retention
        self.enable_detailed_logging = enable_detailed_logging
        
        # Initialize storage
        self.tokens: Dict[str, TokenData] = {}
        self.user_permissions: Dict[str, List[Permission]] = {}
        self.privacy_rules: List[PrivacyRule] = []
        self.access_logs: List[AccessLogEntry] = []
        
        # Initialize locks for thread safety
        self._token_lock = asyncio.Lock()
        self._permission_lock = asyncio.Lock()
        self._privacy_rule_lock = asyncio.Lock()
        self._access_log_lock = asyncio.Lock()
        
        # Setup default privacy rules
        self._setup_default_privacy_rules()
        
        logger.info("Enhanced Security & Privacy Manager initialized")
    
    async def create_token(
        self,
        user_id: str,
        scopes: List[str],
        service_name: Optional[str] = None,
        expiry: Optional[datetime.timedelta] = None,
        metadata: Dict[str, Any] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Create a new authentication token.
        
        Args:
            user_id: ID of the user
            scopes: List of permission scopes for the token
            service_name: Name of the service the token is for
            expiry: Expiry time for the token, or None for default
            metadata: Additional metadata for the token
            
        Returns:
            Tuple of (token, refresh_token) strings
        """
        if expiry is None:
            expiry = self.token_expiry
            
        if metadata is None:
            metadata = {}
            
        # Create token data
        token_id = str(uuid.uuid4())
        created_at = datetime.datetime.now()
        expires_at = created_at + expiry
        
        # Generate refresh token if expiry > 1 hour
        refresh_token = None
        if expiry > datetime.timedelta(hours=1):
            refresh_token = self._generate_secure_string()
        
        # Create token data object
        token_data = TokenData(
            token_id=token_id,
            user_id=user_id,
            created_at=created_at,
            expires_at=expires_at,
            scopes=scopes,
            service_name=service_name,
            metadata=metadata,
            refresh_token=refresh_token,
            last_used=created_at
        )
        
        # Generate token string
        token_payload = {
            "uid": user_id,
            "tid": token_id,
            "exp": int(expires_at.timestamp()),
            "scopes": scopes
        }
        
        if service_name:
            token_payload["svc"] = service_name
            
        token_string = self._sign_token_payload(token_payload)
        
        # Store token data
        async with self._token_lock:
            self.tokens = {**self.tokens, token_id: token_data}
        
        # Log token creation
        self._log_system_event(
            event_type="token_created",
            user_id=user_id,
            details={
                "token_id": token_id,
                "service_name": service_name,
                "scopes": scopes,
                "expires_at": expires_at.isoformat()
            }
        )
        
        return token_string, refresh_token
    
    async def validate_token(self, token_string: str) -> Optional[TokenData]:
        """
        Validate an authentication token.
        
        Args:
            token_string: Token string to validate
            
        Returns:
            Token data if valid, None if invalid
        """
        try:
            # Verify token signature and decode payload
            token_payload = self._verify_token_signature(token_string)
            if not token_payload:
                return None
                
            # Extract token ID and check if it exists
            token_id = token_payload.get("tid")
            if not token_id:
                return None
                
            # Get token data
            async with self._token_lock:
                token_data = self.tokens.get(token_id)
                
            if not token_data:
                return None
                
            # Check if token is revoked
            if token_data.is_revoked:
                return None
                
            # Check if token has expired
            if token_data.is_expired():
                return None
                
            # Update last used time
            token_data.last_used = datetime.datetime.now()
            
            return token_data
            
        except Exception as e:
            logger.error(f"Error validating token: {str(e)}")
            return None
    
    async def refresh_token(
        self,
        token_id: str,
        refresh_token: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Refresh an authentication token.
        
        Args:
            token_id: ID of the token to refresh
            refresh_token: Refresh token for verification
            
        Returns:
            Tuple of (new_token, new_refresh_token) or (None, None) if invalid
        """
        try:
            # Get token data
            async with self._token_lock:
                token_data = self.tokens.get(token_id)
                
            if not token_data:
                return None, None
                
            # Check if token is revoked
            if token_data.is_revoked:
                return None, None
                
            # Verify refresh token
            if not token_data.refresh_token or token_data.refresh_token != refresh_token:
                return None, None
                
            # Create new token with same parameters
            new_token, new_refresh_token = await self.create_token(
                user_id=token_data.user_id,
                scopes=token_data.scopes,
                service_name=token_data.service_name,
                expiry=self.token_expiry,
                metadata=token_data.metadata
            )
            
            # Revoke old token
            await self.revoke_token(token_id)
            
            return new_token, new_refresh_token
            
        except Exception as e:
            logger.error(f"Error refreshing token: {str(e)}")
            return None, None
    
    async def revoke_token(self, token_id: str) -> bool:
        """
        Revoke an authentication token.
        
        Args:
            token_id: ID of the token to revoke
            
        Returns:
            Whether the token was successfully revoked
        """
        try:
            # Get token data
            async with self._token_lock:
                token_data = self.tokens.get(token_id)
                
            if not token_data:
                return False
                
            # Mark token as revoked
            token_data.is_revoked = True
            
            # Log token revocation
            self._log_system_event(
                event_type="token_revoked",
                user_id=token_data.user_id,
                details={
                    "token_id": token_id,
                    "service_name": token_data.service_name,
                    "revoked_at": datetime.datetime.now().isoformat()
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error revoking token: {str(e)}")
            return False
    
    async def grant_permission(
        self,
        user_id: str,
        permission: Permission
    ) -> bool:
        """
        Grant a permission to a user.
        
        Args:
            user_id: ID of the user
            permission: Permission to grant
            
        Returns:
            Whether the permission was granted
        """
        try:
            async with self._permission_lock:
                # Initialize user permissions if needed
                if user_id not in self.user_permissions:
                    self.user_permissions = {**self.user_permissions, user_id: []}
                    
                # Check if permission already exists
                for existing_perm in self.user_permissions[user_id]:
                    if existing_perm.get_key() == permission.get_key():
                        # Update existing permission
                        existing_perm.conditions.update(permission.conditions)
                        return True
                        
                # Add new permission
                self.user_permissions[user_id].append(permission)
                
            # Log permission grant
            self._log_system_event(
                event_type="permission_granted",
                user_id=user_id,
                details={
                    "resource_type": permission.resource_type,
                    "action": permission.action,
                    "resource_id": permission.resource_id
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error granting permission: {str(e)}")
            return False
    
    async def revoke_permission(
        self,
        user_id: str,
        resource_type: str,
        action: str,
        resource_id: Optional[str] = None
    ) -> bool:
        """
        Revoke a permission from a user.
        
        Args:
            user_id: ID of the user
            resource_type: Type of resource
            action: Action to revoke
            resource_id: Specific resource ID or None for all
            
        Returns:
            Whether any permissions were revoked
        """
        try:
            revoked_any = False
            
            async with self._permission_lock:
                # Check if user has any permissions
                if user_id not in self.user_permissions:
                    return False
                    
                # Find matching permissions
                new_permissions = []
                for perm in self.user_permissions[user_id]:
                    if perm.matches(resource_type, action, resource_id):
                        revoked_any = True
                    else:
                        new_permissions.append(perm)
                        
                # Update user permissions
                self.user_permissions = {**self.user_permissions, user_id: new_permissions}
                
            if revoked_any:
                # Log permission revocation
                self._log_system_event(
                    event_type="permission_revoked",
                    user_id=user_id,
                    details={
                        "resource_type": resource_type,
                        "action": action,
                        "resource_id": resource_id
                    }
                )
                
            return revoked_any
            
        except Exception as e:
            logger.error(f"Error revoking permission: {str(e)}")
            return False
    
    async def has_permission(
        self,
        user_id: str,
        resource_type: str,
        action: str,
        resource_id: Optional[str] = None,
        context: Dict[str, Any] = None
    ) -> bool:
        """
        Check if a user has a specific permission.
        
        Args:
            user_id: ID of the user
            resource_type: Type of resource to access
            action: Action to perform
            resource_id: Specific resource ID or None for all
            context: Additional context for permission evaluation
            
        Returns:
            Whether the user has the permission
        """
        try:
            # Special case for system user
            if user_id == "system":
                return True
                
            async with self._permission_lock:
                # Check if user has any permissions
                if user_id not in self.user_permissions:
                    return False
                    
                # Check if any permission matches
                for perm in self.user_permissions[user_id]:
                    if perm.matches(resource_type, action, resource_id):
                        # Check conditions
                        if context and perm.conditions:
                            # Evaluate conditions (simple implementation)
                            # In a real-world scenario, this would be more sophisticated
                            for key, value in perm.conditions.items():
                                if key not in context or context[key] != value:
                                    return False
                                    
                        return True
                        
            return False
            
        except Exception as e:
            logger.error(f"Error checking permission: {str(e)}")
            return False
    
    async def add_privacy_rule(self, rule: PrivacyRule) -> bool:
        """
        Add a privacy rule.
        
        Args:
            rule: Privacy rule to add
            
        Returns:
            Whether the rule was added
        """
        try:
            async with self._privacy_rule_lock:
                # Check if rule with same ID already exists
                for i, existing_rule in enumerate(self.privacy_rules):
                    if existing_rule.rule_id == rule.rule_id:
                        # Replace existing rule
                        self.privacy_rules = {**self.privacy_rules, i: rule}
                        return True
                        
                # Add new rule
                self.privacy_rules = [*self.privacy_rules, rule]
                
            # Log rule addition
            self._log_system_event(
                event_type="privacy_rule_added",
                user_id="system",
                details={
                    "rule_id": rule.rule_id,
                    "data_categories": [dc.name for dc in rule.data_categories],
                    "operation_types": [ot.name for ot in rule.operation_types],
                    "security_level": rule.security_level.name
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding privacy rule: {str(e)}")
            return False
    
    def _setup_default_privacy_rules(self) -> None:
        """Setup default privacy rules for common scenarios."""
        # Rule for biometric data - highest security
        self.privacy_rules.append(PrivacyRule(
            rule_id="default_biometric",
            data_categories=[DataCategory.BIOMETRIC],
            operation_types=list(OperationType),
            security_level=SecurityLevel.CRITICAL,
            purpose_limitation=["authentication", "security"],
            retention_period=datetime.timedelta(days=30),
            requires_consent=True,
            requires_logging=True,
            anonymization_required=False
        ))
        
        # Rule for financial data
        self.privacy_rules.append(PrivacyRule(
            rule_id="default_financial",
            data_categories=[DataCategory.FINANCIAL],
            operation_types=list(OperationType),
            security_level=SecurityLevel.SENSITIVE,
            purpose_limitation=["transactions", "billing", "accounting"],
            retention_period=datetime.timedelta(days=2555),  # ~7 years
            requires_consent=True,
            requires_logging=True,
            anonymization_required=False
        ))
        
        # Rule for health data
        self.privacy_rules.append(PrivacyRule(
            rule_id="default_health",
            data_categories=[DataCategory.HEALTH],
            operation_types=list(OperationType),
            security_level=SecurityLevel.CRITICAL,
            purpose_limitation=["healthcare", "emergency"],
            retention_period=datetime.timedelta(days=3650),  # 10 years
            requires_consent=True,
            requires_logging=True,
            anonymization_required=False
        ))
        
        # Rule for basic data
        self.privacy_rules.append(PrivacyRule(
            rule_id="default_basic",
            data_categories=[DataCategory.BASIC],
            operation_types=[OperationType.READ, OperationType.UPDATE],
            security_level=SecurityLevel.PROTECTED,
            purpose_limitation=None,
            retention_period=None,
            requires_consent=False,
            requires_logging=False,
            anonymization_required=False
        ))
        
        # Rule for location data
        self.privacy_rules.append(PrivacyRule(
            rule_id="default_location",
            data_categories=[DataCategory.LOCATION],
            operation_types=list(OperationType),
            security_level=SecurityLevel.SENSITIVE,
            purpose_limitation=["services", "emergency", "analytics"],
            retention_period=datetime.timedelta(days=90),
            requires_consent=True,
            requires_logging=True,
            anonymization_required=True
        ))
    
    def _log_system_event(self, event_type: str, user_id: str, details: Dict[str, Any]) -> None:
        """
        Log a system event.
        
        Args:
            event_type: Type of event
            user_id: User ID associated with the event
            details: Event details
        """
        if self.enable_detailed_logging:
            logger.info(f"System Event: {event_type} | User: {user_id} | Details: {details}")
    
    def _generate_secure_string(self, length: int = 32) -> str:
        """Generate a secure random string."""
        import secrets
        import string
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def _sign_token_payload(self, payload: Dict[str, Any]) -> str:
        """Sign a token payload and return the token string."""
        # Simple implementation - in production, use proper JWT
        import json
        payload_json = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            self.token_secret.encode(),
            payload_json.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Create token: base64(payload).signature
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode().rstrip('=')
        return f"{payload_b64}.{signature}"
    
    def _verify_token_signature(self, token_string: str) -> Optional[Dict[str, Any]]:
        """Verify a token signature and return the payload if valid."""
        try:
            parts = token_string.split('.')
            if len(parts) != 2:
                return None
                
            payload_b64, signature = parts
            
            # Decode payload
            padding = 4 - (len(payload_b64) % 4)
            if padding != 4:
                payload_b64 += '=' * padding
                
            payload_json = base64.urlsafe_b64decode(payload_b64).decode()
            
            # Verify signature
            expected_signature = hmac.new(
                self.token_secret.encode(),
                payload_json.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                return None
                
            return json.loads(payload_json)
            
        except Exception as e:
            logger.error(f"Error verifying token: {str(e)}")
            return None
    
    async def evaluate_data_access(self, request: DataAccessRequest) -> DataAccessResult:
        """
        Evaluate a data access request against security and privacy policies.
        
        Args:
            request: Data access request to evaluate
            
        Returns:
            Data access result with decision and details
        """
        try:
            # Validate token if provided
            if request.token_data:
                if request.token_data.is_expired():
                    return DataAccessResult(
                        request_id=request.request_id,
                        access_granted=False,
                        reason_denied=AccessDeniedReason.TOKEN_EXPIRED
                    )
                    
                if request.token_data.is_revoked:
                    return DataAccessResult(
                        request_id=request.request_id,
                        access_granted=False,
                        reason_denied=AccessDeniedReason.AUTHENTICATION_FAILURE
                    )
            
            # Check permissions
            has_permission = await self.has_permission(
                user_id=request.user_id,
                resource_type=request.resource_type,
                action=request.action,
                resource_id=request.resource_id,
                context=request.context
            )
            
            if not has_permission:
                # Log access denial
                log_entry = AccessLogEntry(
                    entry_id=str(uuid.uuid4()),
                    timestamp=datetime.datetime.now(),
                    user_id=request.user_id,
                    resource_type=request.resource_type,
                    resource_id=request.resource_id,
                    action=request.action,
                    data_categories=request.data_categories,
                    operation_type=request.operation_type,
                    access_granted=False,
                    reason_denied=AccessDeniedReason.INSUFFICIENT_PERMISSIONS,
                    client_info=request.client_info,
                    context=request.context
                )
                
                async with self._access_log_lock:
                    self.access_logs = [*self.access_logs, log_entry]
                
                return DataAccessResult(
                    request_id=request.request_id,
                    access_granted=False,
                    reason_denied=AccessDeniedReason.INSUFFICIENT_PERMISSIONS,
                    log_entry=log_entry
                )
            
            # Check privacy rules
            applicable_rules = []
            for rule in self.privacy_rules:
                for category in request.data_categories:
                    if rule.applies_to(category, request.operation_type):
                        applicable_rules.append(rule)
                        break
            
            # Apply most restrictive rule
            if applicable_rules:
                most_restrictive = max(applicable_rules, key=lambda r: r.security_level.value)
                
                # Check purpose limitation
                if most_restrictive.purpose_limitation:
                    if request.purpose not in most_restrictive.purpose_limitation:
                        return DataAccessResult(
                            request_id=request.request_id,
                            access_granted=False,
                            reason_denied=AccessDeniedReason.PURPOSE_LIMITATION
                        )
                
                # Check if consent is required
                if most_restrictive.requires_consent:
                    # In a real implementation, check consent database
                    # For now, we'll assume consent is given if in context
                    if not request.context.get("consent_given", False):
                        return DataAccessResult(
                            request_id=request.request_id,
                            access_granted=False,
                            reason_denied=AccessDeniedReason.PRIVACY_VIOLATION,
                            privacy_warning="User consent required for this data access"
                        )
            
            # Principle evaluation if requested
            principle_evaluation = None
            if request.requires_principle_evaluation and self.principle_engine:
                # Create principle evaluation request
                eval_request = PrincipleEvaluationRequest(
                    action=f"{request.action} {request.resource_type}",
                    context={
                        "user_id": request.user_id,
                        "resource_type": request.resource_type,
                        "resource_id": request.resource_id,
                        "data_categories": [dc.name for dc in request.data_categories],
                        "operation_type": request.operation_type.name,
                        "purpose": request.purpose,
                        **request.context
                    },
                    requestor_id=request.user_id
                )
                
                # Evaluate with principle engine
                result = self.principle_engine.evaluate(eval_request)
                principle_evaluation = result
                
                if not result["approved"]:
                    return DataAccessResult(
                        request_id=request.request_id,
                        access_granted=False,
                        reason_denied=AccessDeniedReason.PRINCIPLE_VIOLATION,
                        principle_evaluation=principle_evaluation
                    )
            
            # Access granted - log success
            log_entry = AccessLogEntry(
                entry_id=str(uuid.uuid4()),
                timestamp=datetime.datetime.now(),
                user_id=request.user_id,
                resource_type=request.resource_type,
                resource_id=request.resource_id,
                action=request.action,
                data_categories=request.data_categories,
                operation_type=request.operation_type,
                access_granted=True,
                client_info=request.client_info,
                context=request.context
            )
            
            async with self._access_log_lock:
                self.access_logs = [*self.access_logs, log_entry]
            
            # Determine if anonymization is needed
            anonymized = False
            for rule in applicable_rules:
                if rule.anonymization_required:
                    anonymized = True
                    break
            
            return DataAccessResult(
                request_id=request.request_id,
                access_granted=True,
                log_entry=log_entry,
                principle_evaluation=principle_evaluation,
                anonymized=anonymized
            )
            
        except Exception as e:
            logger.error(f"Error evaluating data access: {str(e)}")
            return DataAccessResult(
                request_id=request.request_id,
                access_granted=False,
                reason_denied=AccessDeniedReason.SECURITY_POLICY
            )
    
    async def get_access_logs(
        self,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        limit: int = 100
    ) -> List[AccessLogEntry]:
        """
        Get access logs with optional filtering.
        
        Args:
            user_id: Filter by user ID
            resource_type: Filter by resource type
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of entries to return
            
        Returns:
            List of matching access log entries
        """
        async with self._access_log_lock:
            filtered_logs = []
            
            for log in reversed(self.access_logs):  # Most recent first
                # Apply filters
                if user_id and log.user_id != user_id:
                    continue
                if resource_type and log.resource_type != resource_type:
                    continue
                if start_time and log.timestamp < start_time:
                    continue
                if end_time and log.timestamp > end_time:
                    continue
                    
                filtered_logs.append(log)
                
                if len(filtered_logs) >= limit:
                    break
                    
            return filtered_logs
    
    async def cleanup_expired_data(self) -> None:
        """Clean up expired tokens and old access logs."""
        try:
            # Clean up expired tokens
            async with self._token_lock:
                expired_tokens = []
                for token_id, token_data in self.tokens.items():
                    if token_data.is_expired():
                        # Keep expired tokens for 24 hours for audit
                        if datetime.datetime.now() > token_data.expires_at + datetime.timedelta(hours=24):
                            expired_tokens.append(token_id)
                
                for token_id in expired_tokens:
                    self.tokens = {k: v for k, v in self.tokens.items() if k != token_id}
                
                if expired_tokens:
                    logger.info(f"Cleaned up {len(expired_tokens)} expired tokens")
            
            # Clean up old access logs
            async with self._access_log_lock:
                cutoff_time = datetime.datetime.now() - self.access_log_retention
                old_logs = []
                
                for i, log in enumerate(self.access_logs):
                    if log.timestamp < cutoff_time:
                        old_logs.append(i)
                
                # Remove from end to preserve indices
                for i in reversed(old_logs):
                    self.access_logs.pop(i)
                
                if old_logs:
                    logger.info(f"Cleaned up {len(old_logs)} old access logs")
                    
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


# Example usage functions
def example_basic_usage() -> None:
    """Example of basic security and privacy manager usage."""
    import asyncio
    
    async def main() -> None:
        # Initialize manager
        manager = EnhancedSecurityPrivacyManager(
            token_secret="your-secret-key-here",
            principle_engine=None  # Can integrate with PrincipleEngine
        )
        
        # Create a token
        token, refresh_token = await manager.create_token(
            user_id="user123",
            scopes=["read:profile", "write:profile"],
            service_name="web_app"
        )
        print(f"Created token: {token[:20]}...")
        
        # Grant permissions
        permission = Permission(
            resource_type="profile",
            action="read",
            resource_id="*"
        )
        await manager.grant_permission("user123", permission)
        
        # Evaluate data access request
        request = DataAccessRequest(
            request_id="req_001",
            user_id="user123",
            resource_type="profile",
            resource_id="profile_456",
            action="read",
            data_categories=[DataCategory.BASIC, DataCategory.CONTACT],
            operation_type=OperationType.READ,
            purpose="user_view",
            context={"consent_given": True}
        )
        
        result = await manager.evaluate_data_access(request)
        print(f"Access granted: {result.access_granted}")
        
    asyncio.run(main())


if __name__ == "__main__":
    example_basic_usage()