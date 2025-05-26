"""
Integration Assistant API Package

Provides FastAPI backend for agent registration, real-time updates,
and integration management.
"""

from .app import app
from .models import (
    AgentRegistration,
    ConnectionStatus,
    IntegrationCodeRequest,
    IntegrationCodeResponse
)
from .websocket_manager import ConnectionManager

__all__ = [
    'app',
    'AgentRegistration',
    'ConnectionStatus',
    'IntegrationCodeRequest',
    'IntegrationCodeResponse',
    'ConnectionManager'
]
