"""
Anthropic Claude Adapter

This module provides an adapter for Anthropic Claude models.
"""

import asyncio
import json
import logging
import time
import os
from typing import Dict, List, Any, Optional, Union

import aiohttp
from llm_adapter_interface import BaseLLMAdapter

logger = logging.getLogger(__name__)

class AnthropicAdapter(BaseLLMAdapter):
    """
    Adapter for Anthropic Claude language models.
    
    This adapter provides a unified interface to the Anthropic Claude API.
    """
    
    # Available Claude models - Updated with current models as of 2024
    CLAUDE_MODELS = {
        "claude-3-opus-20240229": {
            "context_length": 200000,
            "description": "Most powerful Claude model"
        },
        "claude-3-sonnet-20240229": {
            "context_length": 180000,
            "description": "Balanced performance and cost"
        },
        "claude-3-haiku-20240307": {
            "context_length": 180000,
            "description": "Fastest Claude model"
        },
        "claude-3-opus": {
            "context_length": 200000,
            "description": "Most powerful Claude model"
        },
        "claude-3-sonnet": {
            "context_length": 180000,
            "description": "Balanced performance and cost"
        },
        "claude-3-haiku": {
            "context_length": 180000,
            "description": "Fastest Claude model"
        },
        "claude-3": {
            "context_length": 180000,
            "description": "Base Claude 3 model"
        },
        "claude-2.1": {
            "context_length": 100000,
            "description": "Legacy Claude 2.1 model"
        },
        "claude-2.0": {
            "context_length": 100000,
            "description": "Legacy Claude 2.0 model"
        },
        "claude-2": {
            "context_length": 100000,
            "description": "Legacy Claude 2 model"
        },
        "claude-instant-1.2": {
            "context_length": 100000, 
            "description": "Fast, lower-cost legacy model"
        },
        "claude-instant": {
            "context_length": 100000, 
            "description": "Fast, lower-cost model"
        },
        # Adding newest model names per Anthropic API docs
        "claude-3-opus-20240229": {
            "context_length": 200000,
            "description": "Most powerful Claude model"
        },
        "claude-3-sonnet-20240229": {
            "context_length": 180000,
            "description": "Balanced performance and cost"
        },
        "claude-3-haiku-20240307": {
            "context_length": 180000,
            "description": "Fastest Claude model"
        }
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-3-haiku-20240307", 
        timeout: int = 120,
        max_retries: int = 3,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the Anthropic adapter.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model_name: Name of the model to use
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for API requests
            system_prompt: Default system prompt for chat completions
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No Anthropic API key provided. Set ANTHROPIC_API_KEY or pass api_key.")
            
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.system_prompt = system_prompt
        self._session = None
        
    async def _ensure_session(self) -> None:
        """Ensure an aiohttp session exists."""
        if self._session is None or self._session.closed:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "messages-2023-12-15",
                "content-type": "application/json"
            }
            self._session = aiohttp.ClientSession(headers=headers)
    
    async def close(self) -> None:
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.CLAUDE_MODELS.keys())
    
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
            model: The model to use (defaults to self.model_name)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences to end generation
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text as a string
        """
        model_name = model or self.model_name
        
        # Convert to chat format for Claude API
        system_prompt = kwargs.pop("system_prompt", self.system_prompt)
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = await self.chat_complete(
            messages=messages,
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            **kwargs
        )
        
        # Extract text from the response
        if "content" in response and len(response["content"]) > 0:
            return response["content"][0]["text"]
        
        return ""
    
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
            model: The model to use (defaults to self.model_name)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences to end generation
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Dictionary with generated response
        """
        if not self.api_key:
            raise ValueError("API key is required for Claude API requests")
            
        model_name = model or self.model_name
        
        # Ensure we have a session
        await self._ensure_session()
        
        # Log request for debugging
        logger.info(f"Making request to Anthropic API with model: {model_name}")
        
        # Convert OpenAI format to Anthropic format
        claude_messages = []
        system_content = None
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                system_content = content
            elif role == "user":
                claude_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                claude_messages.append({"role": "assistant", "content": content})
            # Ignore other roles
        
        # Prepare the request data
        request_data = {
            "model": model_name,
            "messages": claude_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        # Add system prompt if present
        if system_content:
            request_data["system"] = system_content
        
        # Add stop sequences if provided
        if stop:
            if isinstance(stop, str):
                request_data["stop_sequences"] = [stop]
            else:
                request_data["stop_sequences"] = stop
        
        # Add any additional parameters
        for k, v in kwargs.items():
            if k not in request_data:
                request_data[k] = v
        
        # Make the API request
        attempt = 0
        while attempt < self.max_retries:
            try:
                async with self._session.post(
                    "https://api.anthropic.com/v1/messages",
                    json=request_data,
                    timeout=self.timeout
                ) as response:
                    response_text = await response.text()
                    if response.status == 200:
                        # Parse and return the response
                        resp_json = json.loads(response_text)
                        
                        # Convert to our standardized response format
                        result = {
                            "id": resp_json.get("id", ""),
                            "model": resp_json.get("model", model_name),
                            "content": [
                                {
                                    "type": content.get("type", "text"),
                                    "text": content.get("text", "")
                                }
                                for content in resp_json.get("content", [])
                            ],
                            "usage": resp_json.get("usage", {}),
                            "stop_reason": resp_json.get("stop_reason", "stop"),
                            "stop_sequence": resp_json.get("stop_sequence", None)
                        }
                        
                        return result
                    
                    # Handle rate limiting
                    if response.status == 429:
                        attempt += 1
                        if attempt < self.max_retries:
                            backoff_time = min(2 ** attempt, 60)
                            logger.warning(f"Rate limited by Anthropic API. Retrying in {backoff_time} seconds.")
                            await asyncio.sleep(backoff_time)
                            continue
                    
                    # Special handling for 404 (model not found)
                    if response.status == 404:
                        logger.warning(f"Model not found: {model_name}. Trying alternatives.")
                        
                        # Define fallback sequence based on current model
                        fallback_models = []
                        
                        # Remove date suffix if present
                        if "-20240" in model_name:
                            base_model = model_name.split("-20240")[0]
                            fallback_models.append(base_model)
                        
                        # Add standard fallbacks
                        if "claude-3" in model_name:
                            if "claude-3" != model_name:
                                fallback_models.append("claude-3")
                            fallback_models.extend(["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"])
                        elif "claude-2" in model_name and "claude-2" != model_name:
                            fallback_models.append("claude-2")
                            fallback_models.append("claude-2.1")
                        elif "claude-instant" in model_name and "claude-instant" != model_name:
                            fallback_models.append("claude-instant")
                        
                        # Try next fallback if available
                        if fallback_models:
                            next_model = fallback_models[0]
                            logger.info(f"Trying fallback model: {next_model}")
                            model_name = next_model
                            request_data["model"] = next_model
                            attempt += 1
                            continue
                    
                    # Other errors
                    logger.error(f"Anthropic API error ({response.status}): {response_text}")
                    raise Exception(f"Anthropic API error ({response.status}): {response_text}")
                    
            except asyncio.TimeoutError:
                attempt += 1
                if attempt < self.max_retries:
                    logger.warning(f"Timeout when calling Anthropic API. Retrying... ({attempt}/{self.max_retries})")
                    continue
                else:
                    logger.error("Max retries reached for Anthropic API timeout")
                    raise
            except Exception as e:
                logger.error(f"Error during Anthropic API call: {str(e)}")
                raise
        
        # If we get here, all retries failed
        error_msg = f"Failed to get response from Anthropic API after {self.max_retries} attempts"
        logger.error(f"Error: {error_msg}")
        raise Exception(error_msg)
