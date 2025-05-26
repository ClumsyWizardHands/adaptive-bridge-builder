"""
Mistral LLM Adapter

This module provides an adapter for Mistral AI models, supporting both:
1. Mistral AI API (cloud-based)
2. Local deployment using GGUF models
"""

import asyncio
import json
import logging
import os
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import aiohttp
from llm_adapter_interface import BaseLLMAdapter

logger = logging.getLogger(__name__)

class MistralAdapter(BaseLLMAdapter):
    """
    Adapter for Mistral AI language models.
    
    This adapter provides a unified interface for both cloud API and local models.
    """
    
    # Cloud model details for reference
    CLOUD_MODELS = {
        "mistral-tiny": {
            "context_length": 32000,
            "description": "Fast and cost-effective for basic tasks"
        },
        "mistral-small": {
            "context_length": 32000,
            "description": "Good balance of performance and cost"
        },
        "mistral-medium": {
            "context_length": 32000,
            "description": "High performance with reasonable cost" 
        },
        "mistral-large-latest": {
            "context_length": 32000,
            "description": "Most powerful Mistral model"
        }
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "mistral-medium",
        endpoint_url: Optional[str] = None,
        local_deployment: bool = False,
        backend: str = "auto",
        timeout: int = 60,
        max_retries: int = 3,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the Mistral adapter.
        
        Args:
            api_key: Mistral AI API key (only needed for cloud API)
            model_name: Name of the model to use
            endpoint_url: Custom endpoint URL or path to local model
            local_deployment: Whether to use local deployment
            backend: Backend for local deployment ('llama_cpp', 'ctransformers', or 'auto')
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for API requests
            system_prompt: Default system prompt for chat completions
        """
        self.api_key = api_key
        self.model_name = model_name
        self.endpoint_url = endpoint_url or "https://api.mistral.ai"
        self.timeout = timeout
        self.max_retries = max_retries
        self.system_prompt = system_prompt
        self.local_deployment = local_deployment
        self.backend = backend
        self._session = None
        self._local_model = None
        
        # Initialize the appropriate backend
        if local_deployment:
            self._init_local_model()
    
    def _init_local_model(self) -> None:
        """Initialize local model backend."""
        if self.backend == "auto":
            # Try to use llama.cpp first, then ctransformers
            try:
                self._init_llamacpp()
            except ImportError:
                try:
                    self._init_ctransformers()
                except ImportError:
                    raise ImportError(
                        "No local model backend available. "
                        "Please install either llama-cpp-python or ctransformers."
                    )
        elif self.backend == "llama_cpp":
            self._init_llamacpp()
        elif self.backend == "ctransformers":
            self._init_ctransformers()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _init_llamacpp(self) -> None:
        """Initialize llama.cpp backend."""
        try:
            from llama_cpp import Llama
            
            # Check if the endpoint_url is a valid file path
            model_path = self.endpoint_url
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load the model
            logger.info(f"Loading local model using llama.cpp: {model_path}")
            self._local_model = Llama(
                model_path=model_path,
                n_ctx=4096,  # Context window size
                n_threads=os.cpu_count() or 4,  # Use all available cores
                n_batch=512,  # Batch size for prompt processing
                verbose=False
            )
            self.backend = "llama_cpp"
            logger.info("Successfully loaded local model with llama.cpp backend")
        except ImportError:
            raise ImportError("Failed to import llama_cpp. Please install with: pip install llama-cpp-python")
    
    def _init_ctransformers(self) -> None:
        """Initialize ctransformers backend."""
        try:
            from ctransformers import AutoModelForCausalLM
            
            # Check if the endpoint_url is a valid file path
            model_path = self.endpoint_url
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load the model
            logger.info(f"Loading local model using ctransformers: {model_path}")
            self._local_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type="mistral",
                max_new_tokens=2048,
                context_length=4096,
                gpu_layers=0  # CPU only for compatibility (change for GPU acceleration)
            )
            self.backend = "ctransformers"
            logger.info("Successfully loaded local model with ctransformers backend")
        except ImportError:
            raise ImportError("Failed to import ctransformers. Please install with: pip install ctransformers")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        if not self.api_key:
            raise ValueError("API key is required for cloud API requests")
        
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def _ensure_session(self) -> None:
        """Ensure an aiohttp session exists."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self._get_headers())
    
    async def close(self) -> None:
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            
        # No explicit cleanup needed for local models
        self._local_model = None
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        if self.local_deployment:
            return [self.model_name]
        return list(self.CLOUD_MODELS.keys())
    
    async def _complete_local(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> str:
        """
        Generate a completion using the local model.
        
        Args:
            prompt: The prompt to generate a completion for
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences to end generation
            
        Returns:
            Generated text as a string
        """
        if not self._local_model:
            raise RuntimeError("Local model not initialized")
        
        # Format the prompt for chat-like interaction if needed
        chat_format = kwargs.get("chat_format", True)
        system_prompt = kwargs.get("system_prompt", self.system_prompt)
        formatted_prompt = prompt
        
        if chat_format:
            if self.backend == "llama_cpp":
                # Format for llama.cpp
                if system_prompt:
                    formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                # Generic formatting for other backends
                if system_prompt:
                    formatted_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
                else:
                    formatted_prompt = f"User: {prompt}\nAssistant:"
        
        # Create stop sequences list
        stop_sequences = []
        if stop:
            if isinstance(stop, str):
                stop_sequences.append(stop)
            else:
                stop_sequences.extend(stop)
        
        # Add default stop sequences for chat format
        if chat_format:
            stop_sequences.append("<|im_end|>")
        
        # Run the generation in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        
        if self.backend == "llama_cpp":
            result = await loop.run_in_executor(
                None,
                lambda: self._local_model(
                    formatted_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop_sequences if stop_sequences else None,
                    echo=False
                )
            )
            
            # Extract generated text
            if isinstance(result, dict):
                return result.get("choices", [{}])[0].get("text", "")
            return str(result)
            
        else:  # ctransformers
            result = await loop.run_in_executor(
                None,
                lambda: self._local_model(
                    formatted_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop_sequences if stop_sequences else None
                )
            )
            
            # Remove the prompt from the result if it's included
            if result.startswith(formatted_prompt):
                result = result[len(formatted_prompt):].strip()
                
            return result
    
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
        
        # Use local model if configured
        if self.local_deployment:
            return await self._complete_local(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                **kwargs
            )
        
        # Cloud API - convert to chat format for Mistral API
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
        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"]
        
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
        model_name = model or self.model_name
        
        # If using local model, convert to completion format
        if self.local_deployment:
            # Extract system prompt if present
            system_prompt = None
            user_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                elif msg["role"] == "user":
                    user_messages.append(msg["content"])
            
            # Combine all user messages for simplicity
            combined_prompt = "\n".join(user_messages)
            
            # Get completion
            completion = await self._complete_local(
                prompt=combined_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                system_prompt=system_prompt,
                chat_format=True,
                **kwargs
            )
            
            # Convert to chat format response
            return {
                "id": f"local-chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": completion
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": -1,  # Not tracked for local model
                    "completion_tokens": -1,
                    "total_tokens": -1
                }
            }
        
        # Cloud API
        # Ensure we have a session
        await self._ensure_session()
        
        # Prepare the request data
        request_data = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        # Add stop sequences if provided
        if stop:
            if isinstance(stop, str):
                request_data["stop"] = [stop]
            else:
                request_data["stop"] = stop
        
        # Add any additional parameters
        request_data.update({k: v for k, v in kwargs.items() if k not in request_data})
        
        # Make the API request
        attempt = 0
        while attempt < self.max_retries:
            try:
                async with self._session.post(
                    f"{self.endpoint_url}/v1/chat/completions",
                    json=request_data,
                    timeout=self.timeout
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    
                    error_text = await response.text()
                    
                    # Handle rate limiting
                    if response.status == 429:
                        attempt += 1
                        if attempt < self.max_retries:
                            backoff_time = min(2 ** attempt, 60)
                            logger.warning(f"Rate limited by Mistral API. Retrying in {backoff_time} seconds.")
                            await asyncio.sleep(backoff_time)
                            continue
                    
                    # Other errors
                    logger.error(f"Mistral API error ({response.status}): {error_text}")
                    raise Exception(f"Mistral API error ({response.status}): {error_text}")
                    
            except asyncio.TimeoutError:
                attempt += 1
                if attempt < self.max_retries:
                    logger.warning(f"Timeout when calling Mistral API. Retrying... ({attempt}/{self.max_retries})")
                    continue
                else:
                    logger.error("Max retries reached for Mistral API timeout")
                    raise
            except Exception as e:
                logger.error(f"Error during Mistral API call: {str(e)}")
                raise
        
        # If we get here, all retries failed
        raise Exception(f"Failed to get response from Mistral API after {self.max_retries} attempts")
