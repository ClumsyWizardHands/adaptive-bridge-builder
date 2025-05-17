"""
ApiGatewaySystem - A system allowing agents to connect with virtually any external API or service.

This component enables secure, standardized, and reliable interaction with external APIs 
by handling authentication, data transformation, rate limiting, caching, and audit logging.
The ApiGatewaySystem acts as a central access point for all external system interactions,
providing a consistent interface regardless of the underlying API details.
"""
import asyncio
import datetime
import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

# Rate limiting related imports
from collections import defaultdict, deque


class AuthType(Enum):
    """Types of authentication supported by the gateway"""
    NONE = "none"                      # No authentication
    API_KEY = "api_key"                # API key authentication
    BASIC = "basic"                    # Basic username/password auth
    BEARER = "bearer"                  # Bearer token auth (e.g., JWT)
    OAUTH1 = "oauth1"                  # OAuth 1.0
    OAUTH2 = "oauth2"                  # OAuth 2.0
    API_KEY_QUERY = "api_key_query"    # API key in query parameters
    CERTIFICATE = "certificate"        # Client certificate auth
    DIGEST = "digest"                  # Digest authentication
    CUSTOM = "custom"                  # Custom authentication method


class DataFormat(Enum):
    """Data formats supported for transformation"""
    JSON = "json"                      # JSON data
    XML = "xml"                        # XML data
    YAML = "yaml"                      # YAML data
    CSV = "csv"                        # CSV data
    TEXT = "text"                      # Plain text
    BINARY = "binary"                  # Binary data
    FORM = "form"                      # Form data
    MULTIPART = "multipart"            # Multipart form data
    PROTOBUF = "protobuf"              # Protocol Buffers
    AVRO = "avro"                      # Apache Avro
    CUSTOM = "custom"                  # Custom data format


class HttpMethod(Enum):
    """HTTP methods supported by the gateway"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class LogLevel(Enum):
    """Log levels for audit logging"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int = 60               # Max requests per minute
    burst_size: int = 10                        # Max concurrent requests
    per_endpoint: bool = False                  # Apply per endpoint vs global
    per_client: bool = True                     # Apply per client vs global
    retry_after_seconds: int = 60               # Wait time after limit hit
    custom_limits: Dict[str, int] = field(default_factory=dict)  # Custom limits for specific endpoints


@dataclass
class CacheConfig:
    """Configuration for response caching"""
    enabled: bool = True                        # Enable/disable caching
    ttl_seconds: int = 300                      # Default cache TTL (5 minutes)
    max_size_mb: int = 100                      # Max cache size
    cache_keys: List[str] = field(default_factory=list)  # Keys to use for cache lookup
    ignore_params: List[str] = field(default_factory=list)  # Query params to ignore in cache key
    custom_ttl: Dict[str, int] = field(default_factory=dict)  # Custom TTL for endpoints
    respect_cache_control: bool = True          # Respect cache-control headers
    respect_etag: bool = True                   # Respect ETag headers
    serialize_method: str = "pickle"            # Method to serialize cache data


@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling"""
    retry_count: int = 3                        # Number of retries
    retry_delay_ms: int = 1000                  # Delay between retries
    timeout_ms: int = 5000                      # Request timeout
    circuit_breaker_enabled: bool = True        # Enable circuit breaker
    circuit_breaker_threshold: int = 5          # Errors to trigger circuit open
    circuit_breaker_reset_seconds: int = 60     # Time before trying recovery
    fallback_response: Optional[Dict] = None    # Fallback when all else fails
    custom_error_mappings: Dict = field(default_factory=dict)  # Map error responses


@dataclass
class AuthConfig:
    """Authentication configuration"""
    auth_type: AuthType                         # Type of authentication
    credentials: Dict[str, str] = field(default_factory=dict)  # Credentials
    header_name: Optional[str] = None           # Header name for API key
    token_url: Optional[str] = None             # URL to obtain token
    auth_url: Optional[str] = None              # URL for authorization
    scope: Optional[str] = None                 # OAuth scope
    cert_path: Optional[str] = None             # Path to certificate
    refresh_token: Optional[str] = None         # Refresh token for OAuth
    token_expiry: Optional[datetime.datetime] = None  # When token expires
    custom_auth_handler: Optional[Callable] = None  # Custom auth function


@dataclass
class EndpointConfig:
    """Configuration for an API endpoint"""
    name: str                                   # Endpoint name
    url: str                                    # Endpoint URL
    method: HttpMethod                          # HTTP method
    auth: AuthConfig                            # Authentication configuration
    headers: Dict[str, str] = field(default_factory=dict)  # HTTP headers
    query_params: Dict[str, str] = field(default_factory=dict)  # Query parameters
    request_format: DataFormat = DataFormat.JSON  # Request data format
    response_format: DataFormat = DataFormat.JSON  # Response data format
    rate_limit: Optional[RateLimitConfig] = None  # Rate limiting config
    cache_config: Optional[CacheConfig] = None  # Caching config
    error_config: Optional[ErrorHandlingConfig] = None  # Error handling config
    transform_request: Optional[Callable] = None  # Request transform function
    transform_response: Optional[Callable] = None  # Response transform function
    timeout_ms: int = 30000                     # Timeout in milliseconds
    retry_count: int = 3                        # Number of retries
    webhook_mode: bool = False                  # If endpoint is a webhook
    description: str = ""                       # Endpoint description


@dataclass
class ApiConfig:
    """Configuration for an external API"""
    name: str                                   # API name
    base_url: str                               # Base URL
    auth: AuthConfig                            # Authentication configuration
    endpoints: Dict[str, EndpointConfig] = field(default_factory=dict)  # Endpoints
    headers: Dict[str, str] = field(default_factory=dict)  # Common HTTP headers
    global_rate_limit: Optional[RateLimitConfig] = None  # Global rate limit
    global_cache_config: Optional[CacheConfig] = None  # Global cache config
    global_error_config: Optional[ErrorHandlingConfig] = None  # Global error config
    request_format: DataFormat = DataFormat.JSON  # Default request format
    response_format: DataFormat = DataFormat.JSON  # Default response format
    description: str = ""                       # API description
    tags: List[str] = field(default_factory=list)  # Tags for categorization
    version: str = "1.0"                        # API version
    contact_info: Dict[str, str] = field(default_factory=dict)  # Contact information


@dataclass
class AuditLogEntry:
    """Entry in the audit log"""
    timestamp: datetime.datetime                # Time of the request
    api_name: str                               # API name
    endpoint_name: str                          # Endpoint name
    request_id: str                             # Unique request ID
    client_id: str                              # Client ID
    ip_address: Optional[str] = None            # Client IP address
    request_method: Optional[HttpMethod] = None  # HTTP method
    request_url: Optional[str] = None           # Full request URL
    request_headers: Optional[Dict] = None      # Request headers
    request_params: Optional[Dict] = None       # Request parameters
    request_body: Optional[Any] = None          # Request body
    response_status: Optional[int] = None       # Response status code
    response_headers: Optional[Dict] = None     # Response headers
    response_body: Optional[Any] = None         # Response body
    error_message: Optional[str] = None         # Error message if any
    latency_ms: Optional[int] = None            # Request latency
    cache_hit: Optional[bool] = None            # Whether cache was hit
    rate_limited: Optional[bool] = None         # Whether rate limited
    transformed: Optional[bool] = None          # Whether transformed


class RateLimiter:
    """
    Rate limiter for controlling request frequency to external APIs.
    
    This implementation uses a sliding window algorithm to track requests over time.
    It supports per-client, per-endpoint, and global rate limiting.
    """
    
    def __init__(self, config: RateLimitConfig):
        """Initialize the rate limiter with configuration"""
        self.config = config
        self.request_windows = defaultdict(deque)  # Maps client_id+endpoint -> timestamps
        self.tokens = defaultdict(lambda: config.burst_size)  # Maps client_id+endpoint -> tokens
        self.last_refill = defaultdict(time.time)  # Last time tokens were refilled
    
    def get_key(self, client_id: str, endpoint: str) -> str:
        """Get the key for rate limiting based on configuration"""
        if self.config.per_client and self.config.per_endpoint:
            return f"{client_id}:{endpoint}"
        elif self.config.per_client:
            return client_id
        elif self.config.per_endpoint:
            return endpoint
        else:
            return "global"
    
    async def check_rate_limit(self, client_id: str, endpoint: str) -> Tuple[bool, Optional[int]]:
        """
        Check if the request should be rate limited.
        
        Args:
            client_id: The ID of the client making the request
            endpoint: The endpoint being accessed
            
        Returns:
            Tuple containing:
                - Whether the request is allowed
                - Seconds to wait before next request (if limited)
        """
        key = self.get_key(client_id, endpoint)
        
        # Check for custom limit for this endpoint
        requests_per_minute = self.config.custom_limits.get(
            endpoint, self.config.requests_per_minute
        )
        
        # Use token bucket algorithm
        now = time.time()
        if now > self.last_refill[key] + 60:  # Refill every minute
            # Full refill if more than a minute has passed
            self.tokens[key] = self.config.burst_size
            self.last_refill[key] = now
        else:
            # Partial refill based on time elapsed
            elapsed = now - self.last_refill[key]
            refill = int(elapsed * (requests_per_minute / 60))
            if refill > 0:
                self.tokens[key] = min(
                    self.config.burst_size,
                    self.tokens[key] + refill
                )
                self.last_refill[key] = now
        
        # Check if we have tokens available
        if self.tokens[key] > 0:
            self.tokens[key] -= 1
            return True, None
        else:
            # Rate limited
            wait_time = int(60 - (now - self.last_refill[key]))
            return False, max(1, wait_time)
    
    def reset(self, client_id: str = None, endpoint: str = None) -> None:
        """
        Reset rate limiting for a client, endpoint, or globally.
        
        Args:
            client_id: Optional client ID to reset
            endpoint: Optional endpoint to reset
        """
        if client_id and endpoint:
            key = self.get_key(client_id, endpoint)
            if key in self.tokens:
                del self.tokens[key]
                del self.last_refill[key]
        elif client_id:
            for key in list(self.tokens.keys()):
                if key.startswith(f"{client_id}:") or key == client_id:
                    del self.tokens[key]
                    del self.last_refill[key]
        elif endpoint:
            for key in list(self.tokens.keys()):
                if key.endswith(f":{endpoint}") or key == endpoint:
                    del self.tokens[key]
                    del self.last_refill[key]
        else:
            # Reset all
            self.tokens.clear()
            self.last_refill.clear()


class CacheManager:
    """
    Cache manager for storing API responses.
    
    This implementation supports time-based expiration, size limits,
    and various cache key strategies.
    """
    
    def __init__(self, config: CacheConfig):
        """Initialize the cache manager with configuration"""
        self.config = config
        self.cache = {}  # Maps cache key -> (value, expiry, size)
        self.cache_size = 0  # Current cache size in bytes
        self.hits = 0
        self.misses = 0
    
    def _serialize(self, data: Any) -> bytes:
        """Serialize data based on the configured method"""
        if self.config.serialize_method == "pickle":
            import pickle
            return pickle.dumps(data)
        elif self.config.serialize_method == "json":
            return json.dumps(data).encode("utf-8")
        elif self.config.serialize_method == "marshal":
            import marshal
            return marshal.dumps(data)
        else:
            import pickle
            return pickle.dumps(data)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data based on the configured method"""
        if self.config.serialize_method == "pickle":
            import pickle
            return pickle.loads(data)
        elif self.config.serialize_method == "json":
            return json.loads(data.decode("utf-8"))
        elif self.config.serialize_method == "marshal":
            import marshal
            return marshal.loads(data)
        else:
            import pickle
            return pickle.loads(data)
    
    def _get_cache_key(self, endpoint: str, params: Dict, headers: Dict) -> str:
        """Generate a cache key based on configuration"""
        # Filter out ignored params
        filtered_params = {
            k: v for k, v in params.items() 
            if k not in self.config.ignore_params
        }
        
        # Prepare key components
        key_components = [endpoint]
        
        # Add specified cache key fields
        if self.config.cache_keys:
            for key in self.config.cache_keys:
                if key in filtered_params:
                    key_components.append(f"{key}={filtered_params[key]}")
        else:
            # Use all parameters if no specific keys provided
            for k, v in sorted(filtered_params.items()):
                key_components.append(f"{k}={v}")
        
        # Combine and hash
        key_str = ":".join(key_components)
        return hashlib.md5(key_str.encode("utf-8")).hexdigest()
    
    def _cleanup_cache(self) -> None:
        """Remove expired entries and enforce size limit"""
        # Remove expired entries
        now = time.time()
        expired_keys = [
            key for key, (_, expiry, _) in self.cache.items() 
            if expiry < now
        ]
        
        for key in expired_keys:
            _, _, size = self.cache[key]
            self.cache_size -= size
            del self.cache[key]
        
        # If still over size limit, remove oldest entries
        if self.cache_size > self.config.max_size_mb * 1024 * 1024:
            # Sort by expiry time (older first)
            sorted_items = sorted(
                self.cache.items(), 
                key=lambda x: x[1][1]
            )
            
            # Remove entries until under limit
            for key, (_, _, size) in sorted_items:
                if self.cache_size <= self.config.max_size_mb * 1024 * 1024:
                    break
                    
                self.cache_size -= size
                del self.cache[key]
    
    def get(self, endpoint: str, params: Dict, headers: Dict) -> Tuple[bool, Any, Dict]:
        """
        Get a cached response if available.
        
        Args:
            endpoint: The API endpoint
            params: Request parameters
            headers: Request headers
            
        Returns:
            Tuple containing:
                - Whether a cache hit occurred
                - The cached response (or None)
                - Extended headers with cache HIT/MISS info
        """
        if not self.config.enabled:
            self.misses += 1
            return False, None, {"X-Cache": "DISABLED"}
            
        cache_key = self._get_cache_key(endpoint, params, headers)
        
        # Check if we have a cached response
        if cache_key in self.cache:
            data, expiry, _ = self.cache[cache_key]
            
            # Check if expired
            if expiry < time.time():
                self.misses += 1
                return False, None, {"X-Cache": "EXPIRED"}
                
            # Valid cache hit
            self.hits += 1
            return True, self._deserialize(data), {"X-Cache": "HIT"}
        
        # Cache miss
        self.misses += 1
        return False, None, {"X-Cache": "MISS"}
    
    def set(self, endpoint: str, params: Dict, headers: Dict, response: Any) -> None:
        """
        Cache a response.
        
        Args:
            endpoint: The API endpoint
            params: Request parameters
            headers: Request headers
            response: Response to cache
        """
        if not self.config.enabled:
            return
            
        # Calculate custom TTL for endpoint
        ttl = self.config.custom_ttl.get(endpoint, self.config.ttl_seconds)
        
        # Check cache-control headers if configured
        if self.config.respect_cache_control and "cache-control" in headers:
            cache_control = headers["cache-control"]
            
            # Parse cache control directives
            if "no-cache" in cache_control or "no-store" in cache_control:
                return
                
            # Check max-age directive
            max_age_match = re.search(r"max-age=(\d+)", cache_control)
            if max_age_match:
                ttl = int(max_age_match.group(1))
        
        # Generate cache key and serialize data
        cache_key = self._get_cache_key(endpoint, params, headers)
        serialized_data = self._serialize(response)
        data_size = len(serialized_data)
        
        # Store in cache with expiry time
        self.cache[cache_key] = (
            serialized_data,
            time.time() + ttl,
            data_size
        )
        
        # Update cache size
        self.cache_size += data_size
        
        # Clean up if needed
        self._cleanup_cache()
    
    def invalidate(self, endpoint: Optional[str] = None, params: Optional[Dict] = None) -> None:
        """
        Invalidate cache entries.
        
        Args:
            endpoint: Optional endpoint to invalidate
            params: Optional parameters to invalidate
        """
        if endpoint and params:
            # Invalidate specific endpoint + params
            cache_key = self._get_cache_key(endpoint, params, {})
            if cache_key in self.cache:
                _, _, size = self.cache[cache_key]
                self.cache_size -= size
                del self.cache[cache_key]
        elif endpoint:
            # Invalidate all entries for an endpoint
            keys_to_remove = []
            for key, (_, _, size) in self.cache.items():
                if key.startswith(f"{endpoint}:"):
                    self.cache_size -= size
                    keys_to_remove.append(key)
                    
            for key in keys_to_remove:
                del self.cache[key]
        else:
            # Invalidate entire cache
            self.cache.clear()
            self.cache_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "enabled": self.config.enabled,
            "size_bytes": self.cache_size,
            "size_percent": (self.cache_size / (self.config.max_size_mb * 1024 * 1024)) * 100,
            "entries": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance with external APIs.
    
    Implements the Circuit Breaker pattern to prevent cascading failures
    when an external service is experiencing problems.
    """
    
    # Circuit states
    CLOSED = "CLOSED"       # Normal operation, requests allowed
    OPEN = "OPEN"           # Service considered unavailable, fast fail
    HALF_OPEN = "HALF_OPEN" # Testing if service is back, limited requests
    
    def __init__(self, config: ErrorHandlingConfig):
        """Initialize the circuit breaker with configuration"""
        self.config = config
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.success_count = 0
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: The function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            The function result if successful
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if not self.config.circuit_breaker_enabled:
            # Circuit breaker disabled, execute normally
            return await func(*args, **kwargs)
        
        if self.state == self.OPEN:
            # Check if it's time to move to half-open
            if (time.time() - self.last_failure_time) > self.config.circuit_breaker_reset_seconds:
                self.state = self.HALF_OPEN
                self.success_count = 0
            else:
                # Circuit is open, fast fail
                if self.config.fallback_response:
                    return self.config.fallback_response
                else:
                    raise Exception(f"Circuit breaker open: Service unavailable for {self.config.circuit_breaker_reset_seconds - (time.time() - self.last_failure_time):.1f} more seconds")
        
        # Circuit is closed or half-open, try the request
        try:
            result = await func(*args, **kwargs)
            
            # Success - move toward closed state
            if self.state == self.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= 2:  # Need 2 successful requests to close
                    self.state = self.CLOSED
                    self.failure_count = 0
            elif self.state == self.CLOSED:
                self.failure_count = 0
                
            return result
            
        except Exception as e:
            # Failure - record and possibly open circuit
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == self.HALF_OPEN or self.failure_count >= self.config.circuit_breaker_threshold:
                self.state = self.OPEN
                
            # Re-raise the exception
            if self.config.fallback_response and self.state == self.OPEN:
                return self.config.fallback_response
            else:
                raise e
    
    def reset(self) -> None:
        """Reset the circuit breaker to closed state"""
        self.state = self.CLOSED
        self.failure_count = 0
        self.success_count = 0
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the circuit breaker"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "enabled": self.config.circuit_breaker_enabled
        }


class AuditLogger:
    """
    Audit logger for tracking API interactions.
    
    Records detailed information about requests and responses
    for compliance, debugging, and monitoring purposes.
    """
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        log_level: LogLevel = LogLevel.INFO,
        max_log_size_mb: int = 100,
        max_log_files: int = 10,
        mask_sensitive_data: bool = True,
        sensitive_fields: List[str] = None
    ):
        """Initialize the audit logger"""
        self.log_level = log_level
        self.mask_sensitive_data = mask_sensitive_data
        self.sensitive_fields = sensitive_fields or [
            "password", "api_key", "secret", "token", "auth", "key", "credential",
            "access_token", "refresh_token", "authorization", "ssn", "credit_card"
        ]
        
        # Set up logger
        self.logger = logging.getLogger("api_gateway_audit")
        self.logger.setLevel(logging.getLevelName(log_level.value.upper()))
        
        # File handler if log file specified
        if log_file:
            import logging.handlers
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_log_size_mb * 1024 * 1024,
                backupCount=max_log_files
            )
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Always add a console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def _mask_sensitive_data(self, data: Any) -> Any:
        """Mask sensitive data in logs"""
        if not self.mask_sensitive_data or not data:
            return data
            
        if isinstance(data, dict):
            masked_data = {}
            for key, value in data.items():
                if any(field in key.lower() for field in self.sensitive_fields):
                    if isinstance(value, str):
                        if len(value) > 8:
                            masked_data[key] = value[:4] + "****" + value[-2:]
                        else:
                            masked_data[key] = "********"
                    else:
                        masked_data[key] = "********"
                else:
                    masked_data[key] = self._mask_sensitive_data(value)
            return masked_data
        elif isinstance(data, list):
            return [self._mask_sensitive_data(item) for item in data]
        else:
            return data
    
    def log(self, entry: AuditLogEntry) -> None:
        """Log an audit entry"""
        # Convert entry to JSON
        entry_dict = {
            "timestamp": entry.timestamp.isoformat(),
            "request_id": entry.request_id,
            "api_name": entry.api_name,
            "endpoint_name": entry.endpoint_name,
            "client_id": entry.client_id,
            "ip_address": entry.ip_address,
            "request_method": entry.request_method.value if entry.request_method else None,
            "request_url": entry.request_url,
            "request_headers": self._mask_sensitive_data(entry.request_headers),
            "request_params": self._mask_sensitive_data(entry.request_params),
            "request_body": self._mask_sensitive_data(entry.request_body),
            "response_status": entry.response_status,
            "response_headers": entry.response_headers,
            "response_body": entry.response_body,
            "error_message": entry.error_message,
            "latency_ms": entry.latency_ms,
            "cache_hit": entry.cache_hit,
            "rate_limited": entry.rate_limited,
            "transformed": entry.transformed
        }
        
        # Log at appropriate level
        if entry.error_message or (entry.response_status and entry.response_status >= 500):
            self.logger.error(json.dumps(entry_dict))
        elif entry.response_status and entry.response_status >= 400:
            self.logger.warning(json.dumps(entry_dict))
        else:
            self.logger.info(json.dumps(entry_dict))
            
        # Could also store logs in database or send to external system
    
    def search_logs(
        self, 
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        client_id: Optional[str] = None,
        api_name: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        min_status: Optional[int] = None,
        max_status: Optional[int] = None,
        request_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Search audit logs with various filters.
        
        Note: This is a placeholder implementation that would typically
        involve searching in log files or a database. The actual
        implementation would depend on how logs are stored.
        """
        # Placeholder for log search functionality
        return []


class DataTransformer:
    """
    Data transformer for converting between different formats and schemas.
    
    Handles transformations like JSON<->XML, data restructuring,
    field renaming, and custom transformations.
    """
    
    def __init__(self):
        """Initialize the data transformer"""
        # Registry of built-in transformers
        self.transformers = {
            "json_to_xml": self._json_to_xml,
            "xml_to_json": self._xml_to_json,
            "json_to_yaml": self._json_to_yaml,
            "yaml_to_json": self._yaml_to_json,
            "csv_to_json": self._csv_to_json,
            "json_to_csv": self._json_to_csv,
            "field_map": self._field_map,
            "filter_fields": self._filter_fields,
            "add_fields": self._add_fields,
            "rename_fields": self._rename_fields,
            "flatten_nested": self._flatten_nested,
            "nest_fields": self._nest_fields
        }
        
        # Custom transformers
        self.custom_transformers = {}
    
    def register_transformer(self, name: str, transformer_func: Callable) -> None:
        """Register a custom transformer function"""
        self.custom_transformers[name] = transformer_func
    
    def transform(
        self,
        data: Any,
        source_format: DataFormat,
        target_format: DataFormat,
        transformation_steps: List[Dict] = None
    ) -> Any:
        """
        Transform data from one format to another with optional transformation steps.
        
        Args:
            data: The data to transform
            source_format: The source data format
            target_format: The target data format
            transformation_steps: Optional list of transformation steps
            
        Returns:
            The transformed data
        """
        # First convert to internal format (JSON/dict)
        internal_data = self._to_internal_format(data, source_format)
        
        # Apply transformation steps if provided
        if transformation_steps:
            for step in transformation_steps:
                step_type = step.get("type")
                step_config = step.get("config", {})
                
                if step_type in self.transformers:
                    internal_data = self.transformers[step_type](internal_data, **step_config)
                elif step_type in self.custom_transformers:
                    internal_data = self.custom_transformers[step_type](internal_data, **step_config)
                else:
                    raise ValueError(f"Unknown transformation step: {step_type}")
        
        # Convert from internal format to target format
        return self._from_internal_format(internal_data, target_format)
    
    def _to_internal_format(self, data: Any, source_format: DataFormat) -> Dict:
        """Convert data from source format to internal format (JSON/dict)"""
        if source_format == DataFormat.JSON:
            if isinstance(data, str):
                return json.loads(data)
            return data
        elif source_format == DataFormat.XML:
            import xml.etree.ElementTree as ET
            import xmltodict
            
            if isinstance(data, str):
                return xmltodict.parse(data)
            elif isinstance(data, ET.Element):
                return xmltodict.parse(ET.tostring(data))
            return data
        elif source_format == DataFormat.YAML:
            import yaml
            
            if isinstance(data, str):
                return yaml.safe_load(data)
            return data
        elif source_format == DataFormat.CSV:
            import csv
            import io
            
            if isinstance(data, str):
                result = []
                f = io.StringIO(data)
                reader = csv.DictReader(f)
                for row in reader:
                    result.append(dict(row))
                return result
            return data
        elif source_format == DataFormat.TEXT:
            return {"text": data}
        else:
            # For other formats, assume already in internal format
            return data
    
    def _from_internal_format(self, data: Any, target_format: DataFormat) -> Any:
        """Convert data from internal format to target format"""
        if target_format == DataFormat.JSON:
            return json.dumps(data)
        elif target_format == DataFormat.XML:
            import dicttoxml
            
            return dicttoxml.dicttoxml(data).decode("utf-8")
        elif target_format == DataFormat.YAML:
            import yaml
            
            return yaml.dump(data)
        elif target_format == DataFormat.CSV:
            import csv
            import io
            
            if not isinstance(data, list):
                data = [data]
                
            output = io.StringIO()
            if data:
                writer = csv.DictWriter(output, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
                
            return output.getvalue()
        elif target_format == DataFormat.TEXT:
            if isinstance(data, dict) and "text" in data:
                return data["text"]
            return str(data)
        else:
            # For other formats, return as is
            return data
    
    # Transformation functions
    
    def _json_to_xml(self, data: Dict, root_name: str = "root", **kwargs) -> Dict:
        """Convert JSON to XML"""
        import dicttoxml
        
        xml = dicttoxml.dicttoxml(data, root=root_name, **kwargs)
        return {"xml": xml.decode("utf-8")}
    
    def _xml_to_json(self, data: Dict, **kwargs) -> Dict:
        """Convert XML to JSON"""
        if isinstance(data, dict) and "xml" in data:
            import xmltodict
            
            return xmltodict.parse(data["xml"])
        return data
    
    def _json_to_yaml(self, data: Dict, **kwargs) -> Dict:
        """Convert JSON to YAML"""
        import yaml
        
        yaml_str = yaml.dump(data, **kwargs)
        return {"yaml": yaml_str}
    
    def _yaml_to_json(self, data: Dict, **kwargs) -> Dict:
        """Convert YAML to JSON"""
        if isinstance(data, dict) and "yaml" in data:
            import yaml
            
            return yaml.safe_load(data["yaml"])
        return data
    
    def _csv_to_json(self, data: Dict, **kwargs) -> List[Dict]:
        """Convert CSV to JSON"""
        if isinstance(data, dict) and "csv" in data:
            import csv
            import io
            
            result = []
            f = io.StringIO(data["csv"])
            reader = csv.DictReader(f)
            for row in reader:
                result.append(dict(row))
            return result
        return data
    
    def _json_to_csv(self, data: List[Dict], **kwargs) -> Dict:
        """Convert JSON to CSV"""
        import csv
        import io
        
        if not isinstance(data, list):
            data = [data]
            
        output = io.StringIO()
        if data:
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
            
        return {"csv": output.getvalue()}
    
    def _field_map(self, data: Dict, mapping: Dict, **kwargs) -> Dict:
        """Map fields from one structure to another"""
        result = {}
        
        for target_field, source_field in mapping.items():
            # Handle nested fields with dot notation
            if "." in source_field:
                parts = source_field.split(".")
                value = data
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        value = None
                        break
                result[target_field] = value
            else:
                result[target_field] = data.get(source_field)
                
        return result
    
    def _filter_fields(self, data: Dict, fields: List[str], include: bool = True, **kwargs) -> Dict:
        """Filter fields from data"""
        if include:
            # Include only specified fields
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if k in fields}
            elif isinstance(data, list):
                return [{k: v for k, v in item.items() if k in fields} for item in data]
        else:
            # Exclude specified fields
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if k not in fields}
            elif isinstance(data, list):
                return [{k: v for k, v in item.items() if k not in fields} for item in data]
                
        return data
    
    def _add_fields(self, data: Dict, fields: Dict, **kwargs) -> Dict:
        """Add fields to data"""
        if isinstance(data, dict):
            result = data.copy()
            result.update(fields)
            return result
        elif isinstance(data, list):
            return [item.copy().update(fields) for item in data]
            
        return data
    
    def _rename_fields(self, data: Dict, renames: Dict, **kwargs) -> Dict:
        """Rename fields in data"""
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                new_key = renames.get(key, key)
                result[new_key] = value
            return result
        elif isinstance(data, list):
            return [self._rename_fields(item, renames) for item in data]
            
        return data
    
    def _flatten_nested(self, data: Dict, delimiter: str = ".", **kwargs) -> Dict:
        """Flatten nested dictionaries"""
        def _flatten(d, parent_key=""):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{delimiter}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(_flatten(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)
            
        if isinstance(data, dict):
            return _flatten(data)
        elif isinstance(data, list):
            return [_flatten(item) for item in data]
            
        return data
    
    def _nest_fields(self, data: Dict, nesting: Dict, **kwargs) -> Dict:
        """Nest fields into sub-dictionaries"""
        result = data.copy()
        
        for nest_key, fields in nesting.items():
            # Create the nested dict
            nested = {}
            for field in fields:
                if field in result:
                    nested[field] = result[field]
                    del result[field]
                    
            # Add the nested dict
            result[nest_key] = nested
            
        return result


class AuthenticationManager:
    """
    Authentication manager for handling various auth methods.
    
    Supports API keys, basic auth, OAuth, JWT, etc.
    """
    
    def __init__(self):
        """Initialize the authentication manager"""
        # Registry of auth handlers
        self.auth_handlers = {
            AuthType.NONE: self._handle_no_auth,
            AuthType.API_KEY: self._handle_api_key,
            AuthType.BASIC: self._handle_basic_auth,
            AuthType.BEARER: self._handle_bearer_token,
            AuthType.OAUTH1: self._handle_oauth1,
            AuthType.OAUTH2: self._handle_oauth2,
            AuthType.API_KEY_QUERY: self._handle_api_key_query,
            AuthType.CERTIFICATE: self._handle_certificate,
            AuthType.DIGEST: self._handle_digest_auth
        }
        
        # Cache for OAuth tokens
        self.token_cache = {}
    
    async def authenticate(
        self,
        auth_config: AuthConfig,
        request_headers: Dict[str, str] = None,
        request_params: Dict[str, Any] = None
    ) -> Dict[str, str]:
        """
        Apply authentication to request headers/params.
        
        Args:
            auth_config: Authentication configuration
            request_headers: Existing request headers to augment
            request_params: Existing request parameters to augment
            
        Returns:
            Updated headers with authentication
        """
        # Initialize headers and params if not provided
        headers = request_headers.copy() if request_headers else {}
        params = request_params.copy() if request_params else {}
        
        # Check if token refresh is needed for OAuth2
        if auth_config.auth_type == AuthType.OAUTH2:
            await self._check_refresh_token(auth_config)
        
        # Apply custom auth if provided
        if auth_config.custom_auth_handler:
            return await auth_config.custom_auth_handler(auth_config, headers, params)
        
        # Use appropriate auth handler
        if auth_config.auth_type in self.auth_handlers:
            return await self.auth_handlers[auth_config.auth_type](auth_config, headers, params)
        
        # Default to no auth
        return headers
    
    async def _check_refresh_token(self, auth_config: AuthConfig) -> None:
        """Check if OAuth2 token needs refreshing and refresh if needed"""
        if (auth_config.token_expiry and 
            auth_config.token_expiry < datetime.datetime.now() and
            auth_config.refresh_token and 
            auth_config.token_url):
            
            # Refresh token
            await self._refresh_oauth2_token(auth_config)
    
    async def _refresh_oauth2_token(self, auth_config: AuthConfig) -> None:
        """Refresh an OAuth2 token"""
        import aiohttp
        
        # Prepare request data
        data = {
            "grant_type": "refresh_token",
            "refresh_token": auth_config.refresh_token
        }
        
        # Add client credentials if available
        if "client_id" in auth_config.credentials:
            data["client_id"] = auth_config.credentials["client_id"]
            
        if "client_secret" in auth_config.credentials:
            data["client_secret"] = auth_config.credentials["client_secret"]
        
        # Send request
        async with aiohttp.ClientSession() as session:
            async with session.post(auth_config.token_url, data=data) as response:
                if response.status == 200:
                    token_data = await response.json()
                    
                    # Update credentials with new tokens
                    auth_config.credentials["access_token"] = token_data["access_token"]
                    
                    if "refresh_token" in token_data:
                        auth_config.refresh_token = token_data["refresh_token"]
                        
                    if "expires_in" in token_data:
                        auth_config.token_expiry = datetime.datetime.now() + datetime.timedelta(seconds=token_data["expires_in"])
                else:
                    # Failed to refresh token
                    raise Exception(f"Failed to refresh OAuth2 token: {response.status}")
    
    # Auth handlers
    
    async def _handle_no_auth(self, auth_config: AuthConfig, headers: Dict, params: Dict) -> Dict:
        """Handle no authentication"""
        return headers
    
    async def _handle_api_key(self, auth_config: AuthConfig, headers: Dict, params: Dict) -> Dict:
        """Handle API key authentication"""
        if "api_key" in auth_config.credentials:
            # Use specified header or default to X-API-Key
            header_name = auth_config.header_name or "X-API-Key"
            headers[header_name] = auth_config.credentials["api_key"]
        return headers
    
    async def _handle_basic_auth(self, auth_config: AuthConfig, headers: Dict, params: Dict) -> Dict:
        """Handle basic authentication"""
        import base64
        
        if "username" in auth_config.credentials and "password" in auth_config.credentials:
            auth_str = f"{auth_config.credentials['username']}:{auth_config.credentials['password']}"
            encoded = base64.b64encode(auth_str.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"
        return headers
    
    async def _handle_bearer_token(self, auth_config: AuthConfig, headers: Dict, params: Dict) -> Dict:
        """Handle bearer token authentication"""
        if "access_token" in auth_config.credentials:
            headers["Authorization"] = f"Bearer {auth_config.credentials['access_token']}"
        return headers
    
    async def _handle_oauth1(self, auth_config: AuthConfig, headers: Dict, params: Dict) -> Dict:
        """Handle OAuth1 authentication"""
        # Note: OAuth1 is complex and would typically use a library like oauthlib
        # This is a simplified placeholder
        return headers
    
    async def _handle_oauth2(self, auth_config: AuthConfig, headers: Dict, params: Dict) -> Dict:
        """Handle OAuth2 authentication"""
        if "access_token" in auth_config.credentials:
            headers["Authorization"] = f"Bearer {auth_config.credentials['access_token']}"
        return headers
    
    async def _handle_api_key_query(self, auth_config: AuthConfig, headers: Dict, params: Dict) -> Dict:
        """Handle API key in query parameters"""
        if "api_key" in auth_config.credentials:
            # Add API key to query parameters
            param_name = auth_config.header_name or "api_key"
            params[param_name] = auth_config.credentials["api_key"]
        return headers
    
    async def _handle_certificate(self, auth_config: AuthConfig, headers: Dict, params: Dict) -> Dict:
        """Handle certificate authentication"""
        # Certificate auth is handled at the connection level, not in headers
        return headers
    
    async def _handle_digest_auth(self, auth_config: AuthConfig, headers: Dict, params: Dict) -> Dict:
        """Handle digest authentication"""
        # Note: Digest auth is complex and requires multiple exchanges
        # This is a simplified placeholder
        return headers


class ApiGatewaySystem:
    """
    Central system for connecting agents with external APIs and services.
    
    This system provides a unified interface for agents to interact with
    various external systems, handling authentication, data transformation,
    rate limiting, caching, error handling, and audit logging.
    """
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        log_level: LogLevel = LogLevel.INFO,
        cache_dir: Optional[str] = None,
        default_timeout_ms: int = 30000,
        default_retry_count: int = 3,
        mask_sensitive_data: bool = True
    ):
        """Initialize the API Gateway System"""
        # Initialize components
        self.auth_manager = AuthenticationManager()
        self.transformer = DataTransformer()
        self.logger = AuditLogger(
            log_file=log_file,
            log_level=log_level,
            mask_sensitive_data=mask_sensitive_data
        )
        
        # Default configurations
        self.default_timeout_ms = default_timeout_ms
        self.default_retry_count = default_retry_count
        
        # Registered APIs
        self.apis: Dict[str, ApiConfig] = {}
        
        # Component instances per API
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.cache_managers: Dict[str, CacheManager] = {}
        self.circuit_breakers: Dict[str, Dict[str, CircuitBreaker]] = {}
