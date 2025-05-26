"""
LLM Adapter Interface

This module defines the base class and registry for LLM adapters.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union


class AuthenticationError(Exception):
    """Raised when authentication with the LLM provider fails."""
    pass


class RateLimitError(Exception):
    """Raised when the API rate limit is exceeded."""
    pass


class ModelNotFoundError(Exception):
    """Raised when a requested model is not found or not available."""
    pass


logger = logging.getLogger(__name__)

class RequestError(Exception):
    """Exception raised when there's an error with the request."""
    pass


class ResponseError(Exception):
    """Exception raised when there's an error with the response."""
    pass


class BaseLLMAdapter(ABC):
    """
    Base abstract class for LLM adapters.
    
    All LLM adapters must implement this interface to ensure
    consistent behavior across different providers.
    """
    
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> str:
        """
        Generate a completion for the given prompt.
        
        Args:
            prompt: The prompt to generate a completion for
            model: The model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences to end generation
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text as a string
        """
        pass
    
    @abstractmethod
    async def chat_complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a chat completion for the given messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: The model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences to end generation
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Dictionary with generated response
        """
        pass
    
    async def close(self) -> None:
        """Close any resources the adapter has open."""
        pass


class LLMAdapterRegistry:
    """
    Registry for LLM adapters.
    
    This class acts as a central registry for all LLM adapters,
    allowing them to be looked up by provider name.
    """
    
    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._adapters = {}
    
    def register_adapter(self, adapter: BaseLLMAdapter, provider: str) -> None:
        """
        Register an adapter with the registry.
        
        Args:
            adapter: The adapter to register
            provider: The provider name to register the adapter under
        """
        self._adapters = {**self._adapters, provider: adapter}
        logger.debug(f"Registered LLM adapter for provider: {provider}")
    
    def get_adapter(self, provider: str) -> Optional[BaseLLMAdapter]:
        """
        Get an adapter by provider name.
        
        Args:
            provider: The provider name to look up
            
        Returns:
            The adapter, or None if not found
        """
        return self._adapters.get(provider)
    
    def list_adapters(self) -> List[str]:
        """
        Get a list of registered provider names.
        
        Returns:
            List of provider names
        """
        return list(self._adapters.keys())
    
    async def close_all(self) -> None:
        """Close all registered adapters."""
        for provider, adapter in self._adapters.items():
            try:
                await adapter.close()
                logger.debug(f"Closed adapter for provider: {provider}")
            except Exception as e:
                logger.error(f"Error closing adapter for provider {provider}: {str(e)}")
