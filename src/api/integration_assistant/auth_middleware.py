"""
Authentication Middleware for Integration Assistant API

Provides API key validation for protecting endpoints.
"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from typing import Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from generate_api_key import APIKeyManager


# API Key header configuration
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Initialize API Key Manager
api_key_manager = APIKeyManager()


async def get_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Validate API key from request header.
    
    Args:
        api_key: API key from X-API-Key header
        
    Returns:
        Validated API key
        
    Raises:
        HTTPException: If API key is invalid or missing
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key",
            headers={"WWW-Authenticate": "API-Key"},
        )
    
    # Validate the key
    key_data = api_key_manager.validate_key(api_key)
    if not key_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API Key",
            headers={"WWW-Authenticate": "API-Key"},
        )
    
    return api_key


async def require_permission(
    permission: str,
    api_key: str = Security(get_api_key)
) -> dict:
    """
    Require specific permission for an endpoint.
    
    Args:
        permission: Required permission
        api_key: Validated API key
        
    Returns:
        Key data with permissions
        
    Raises:
        HTTPException: If permission is not granted
    """
    key_data = api_key_manager.validate_key(api_key)
    
    if permission not in key_data.get("permissions", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission '{permission}' required"
        )
    
    return key_data


# Dependency functions for specific permissions
async def require_read(api_key: str = Security(get_api_key)) -> dict:
    """Require read permission."""
    return await require_permission("read", api_key)


async def require_write(api_key: str = Security(get_api_key)) -> dict:
    """Require write permission."""
    return await require_permission("write", api_key)


async def require_connect(api_key: str = Security(get_api_key)) -> dict:
    """Require connect permission."""
    return await require_permission("connect", api_key)
