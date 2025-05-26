"""
OpenAI GPT Adapter

This module provides an adapter for OpenAI's GPT models, implementing the BaseLLMAdapter
interface. It handles authentication, request formatting, and response parsing
specific to OpenAI's API.
"""

import time
import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Callable, Union

# Import needed for token counting
try:
    import tiktoken
except ImportError:
    tiktoken = None

from llm_adapter_interface import (
    BaseLLMAdapter,
    AuthenticationError,
    RequestError,
    ResponseError,
    RateLimitError,
    ModelNotFoundError
)
from llm_key_manager import LLMKeyManager


class OpenAIGPTAdapter(BaseLLMAdapter):
    """
    Adapter for OpenAI's GPT models.
    
    This adapter implements the BaseLLMAdapter interface for OpenAI's GPT models,
    handling the specific details of the OpenAI API.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o",
        organization: Optional[str] = None,
        key_manager: Optional[LLMKeyManager] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the OpenAI GPT adapter.
        
        Args:
            api_key: Optional API key for OpenAI (if None, will use key_manager)
            model_name: Default model to use (default: gpt-4o)
            organization: Optional OpenAI organization ID
            key_manager: Optional key manager for retrieving API keys
            logger: Optional logger for tracking events
        """
        # Initialize the base class
        super().__init__(
            provider_name="openai",
            api_key=api_key,
            model_name=model_name,
            key_manager=key_manager,
            logger=logger
        )
        
        # OpenAI-specific attributes
        self.organization = organization
        self.client = None
        
        # Initialize the OpenAI client
        self._init_client()
        
        # Initialize token counters if tiktoken is available
        self.tokenizers = {}
        if tiktoken:
            try:
                # Load the tokenizer for the default model
                self._get_tokenizer(model_name)
            except Exception as e:
                self.logger.warning(f"Failed to load tokenizer for {model_name}: {e}")
    
    def _init_client(self) -> None:
        """Initialize the OpenAI client."""
        try:
            import openai
            
            # Configure the client
            client_params = {
                "api_key": self.api_key
            }
            
            if self.organization:
                client_params["organization"] = self.organization
                
            self.client = openai.OpenAI(**client_params)
            
            # Test the connection with a model list request
            try:
                self.client.models.list()
                self.logger.info("Successfully connected to OpenAI API")
            except Exception as e:
                raise AuthenticationError(f"Failed to connect to OpenAI API: {e}")
                
        except ImportError:
            self.logger.error("OpenAI library not installed. Please install it with 'pip install openai'")
            raise ImportError("OpenAI library not installed. Please install it with 'pip install openai'")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise AuthenticationError(f"Failed to initialize OpenAI client: {e}")
    
    @property
    def provider_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a dictionary of available OpenAI models and their capabilities.
        
        This property provides a hardcoded list of commonly used GPT models with
        their capabilities, context sizes, and other relevant information. For a
        dynamic list, use the list_available_models() method, but note that it 
        requires an API call.
        
        Returns:
            Dict mapping model names to their capabilities
        """
        return {
            "gpt-4o": {
                "name": "GPT-4o",
                "description": "OpenAI's most advanced multimodal model that can process both text and vision inputs",
                "context_size": 128000,
                "capabilities": ["text_generation", "image_processing", "function_calling"],
                "pricing": {
                    "input": 0.01,  # $ per 1000 tokens (approximate)
                    "output": 0.03   # $ per 1000 tokens (approximate)
                }
            },
            "gpt-4-turbo": {
                "name": "GPT-4-Turbo",
                "description": "OpenAI's most capable and cost effective model in the GPT-4 family",
                "context_size": 128000,
                "capabilities": ["text_generation", "function_calling"],
                "pricing": {
                    "input": 0.01,  # $ per 1000 tokens (approximate)
                    "output": 0.03   # $ per 1000 tokens (approximate)
                }
            },
            "gpt-4": {
                "name": "GPT-4",
                "description": "OpenAI's flagship model with strong reasoning capabilities",
                "context_size": 8192,
                "capabilities": ["text_generation", "function_calling"],
                "pricing": {
                    "input": 0.03,  # $ per 1000 tokens
                    "output": 0.06   # $ per 1000 tokens
                }
            },
            "gpt-3.5-turbo": {
                "name": "GPT-3.5-Turbo",
                "description": "OpenAI's cost-effective model with good general capabilities",
                "context_size": 16385,
                "capabilities": ["text_generation", "function_calling"],
                "pricing": {
                    "input": 0.0005,  # $ per 1000 tokens
                    "output": 0.0015   # $ per 1000 tokens
                }
            }
        }
    
    async def list_available_models(self) -> List[str]:
        """
        Get a list of available models from the OpenAI API.
        
        Returns:
            List of model names
            
        Raises:
            AuthenticationError: If authentication fails
            RequestError: If the request fails
        """
        try:
            # Request the list of models from the API
            models = self.client.models.list()
            
            # Extract model names and filter for GPT models
            model_names = [
                model.id for model in models.data
                if model.id.startswith("gpt-") or "gpt" in model.id.lower()
            ]
            
            return model_names
            
        except Exception as e:
            self.logger.error(f"Failed to list available models: {e}")
            raise RequestError(f"Failed to list available models: {e}")
    
    async def send_request(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a request to the OpenAI API and get the response.
        
        Args:
            prompt: The prompt to send to the model
            model: Model to use (defaults to self.model_name)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Optional list of strings that stop generation
            **kwargs: Additional OpenAI-specific parameters
                - system_message: System message for the conversation
                - top_p: Nucleus sampling parameter
                - presence_penalty: Penalty for token presence
                - frequency_penalty: Penalty for token frequency
                - functions: List of function definitions for function calling
                - function_call: Controls function calling behavior
                
        Returns:
            Dict containing the raw response from the API
            
        Raises:
            AuthenticationError: If authentication fails
            RequestError: If the request fails
            ResponseError: If processing the response fails
            RateLimitError: If the rate limit is reached
        """
        # Use the provided model or fall back to the default
        model_name = model or self.model_name
        
        if not model_name:
            raise ValueError("No model specified and no default model set")
        
        # Set up the request parameters
        system_message = kwargs.get("system_message", "You are a helpful assistant.")
        
        # Format the prompt as messages for the Chat API
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        # Extract OpenAI-specific parameters
        openai_params = {
            "temperature": temperature
        }
        
        if max_tokens:
            openai_params["max_tokens"] = max_tokens
            
        if stop_sequences:
            openai_params["stop"] = stop_sequences
            
        # Add optional parameters if provided
        for param in ["top_p", "presence_penalty", "frequency_penalty"]:
            if param in kwargs:
                openai_params[param] = kwargs[param]
                
        # Handle function calling if specified
        if "functions" in kwargs:
            openai_params["functions"] = kwargs["functions"]
            
            if "function_call" in kwargs:
                openai_params["function_call"] = kwargs["function_call"]
        
        try:
            # Send the request
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                **openai_params
            )
            
            end_time = time.time()
            request_time = end_time - start_time
            
            # Track the request
            tokens = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            self._track_request(model_name, tokens)
            
            # Convert to a serializable dictionary
            response_dict = self._convert_response_to_dict(response)
            
            # Add timing information
            response_dict["request_time"] = request_time
            
            return response_dict
            
        except Exception as e:
            error_message = str(e)
            
            if "rate limit" in error_message.lower():
                raise RateLimitError(f"OpenAI rate limit exceeded: {e}")
            elif "auth" in error_message.lower() or "api key" in error_message.lower():
                raise AuthenticationError(f"OpenAI authentication failed: {e}")
            else:
                raise RequestError(f"OpenAI request failed: {e}")
    
    def _convert_response_to_dict(self, response) -> Dict[str, Any]:
        """
        Convert the OpenAI response object to a serializable dictionary.
        
        Args:
            response: OpenAI response object
            
        Returns:
            Dict containing the response data
        """
        if hasattr(response, "model_dump"):
            # Use model_dump for newer OpenAI client versions
            return response.model_dump()
        elif hasattr(response, "to_dict"):
            # Use to_dict for older versions
            return response.to_dict()
        else:
            # Manual conversion if neither method is available
            response_dict = {
                "id": response.id,
                "object": response.object,
                "created": response.created,
                "model": response.model,
                "choices": [
                    {
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content
                        },
                        "finish_reason": choice.finish_reason
                    }
                    for choice in response.choices
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            return response_dict
    
    def process_response(self, response: Dict[str, Any]) -> str:
        """
        Process the raw OpenAI response into a usable format.
        
        Args:
            response: Raw response from send_request
            
        Returns:
            Processed response text
            
        Raises:
            ResponseError: If processing the response fails
        """
        try:
            choices = response.get("choices", [])
            
            if not choices:
                raise ResponseError("No choices found in the response")
                
            # Get the first choice (we only support single completions for now)
            choice = choices[0]
            
            # Check if it's a function call
            message = choice.get("message", {})
            
            if "function_call" in message:
                function_call = message["function_call"]
                return json.dumps({
                    "function_call": {
                        "name": function_call.get("name", ""),
                        "arguments": function_call.get("arguments", "{}")
                    }
                })
            
            # Otherwise return the content
            return message.get("content", "")
            
        except Exception as e:
            raise ResponseError(f"Failed to process OpenAI response: {e}")
    
    async def stream_request(
        self,
        prompt: str,
        callback: Callable[[str], None],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a streaming request to the OpenAI API.
        
        Args:
            prompt: The prompt to send to the model
            callback: Function to call with each text chunk
            model: Model to use (defaults to self.model_name)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Optional list of strings that stop generation
            **kwargs: Additional OpenAI-specific parameters
                
        Returns:
            Dict containing the complete response information
            
        Raises:
            AuthenticationError: If authentication fails
            RequestError: If the request fails
            ResponseError: If processing the response fails
            RateLimitError: If the rate limit is reached
        """
        # Use the provided model or fall back to the default
        model_name = model or self.model_name
        
        if not model_name:
            raise ValueError("No model specified and no default model set")
        
        # Set up the request parameters
        system_message = kwargs.get("system_message", "You are a helpful assistant.")
        
        # Format the prompt as messages for the Chat API
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        # Extract OpenAI-specific parameters
        openai_params = {
            "temperature": temperature,
            "stream": True
        }
        
        if max_tokens:
            openai_params["max_tokens"] = max_tokens
            
        if stop_sequences:
            openai_params["stop"] = stop_sequences
            
        # Add optional parameters if provided
        for param in ["top_p", "presence_penalty", "frequency_penalty"]:
            if param in kwargs:
                openai_params[param] = kwargs[param]
                
        # Handle function calling if specified
        if "functions" in kwargs:
            openai_params["functions"] = kwargs["functions"]
            
            if "function_call" in kwargs:
                openai_params["function_call"] = kwargs["function_call"]
        
        try:
            # Send the streaming request
            start_time = time.time()
            full_response = ""
            
            # Create the stream
            stream = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                **openai_params
            )
            
            # Assemble the full response information
            response_info = {
                "model": model_name,
                "created": int(time.time()),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": ""
                        },
                        "finish_reason": None
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            
            # Process the stream
            for chunk in stream:
                # Extract the content from the chunk
                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    
                    if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                        content = choice.delta.content
                        
                        if content:
                            # Append to the full response
                            full_response += content
                            
                            # Call the callback with the chunk
                            callback(content)
                            
                            # Update the response info
                            response_info["choices"][0]["message"]["content"] = full_response
                    
                    # Update finish reason if available
                    if hasattr(choice, "finish_reason") and choice.finish_reason:
                        response_info["choices"][0]["finish_reason"] = choice.finish_reason
            
            end_time = time.time()
            request_time = end_time - start_time
            
            # Estimate token usage since streaming doesn't provide it
            input_tokens = self.get_token_count(prompt) + self.get_token_count(system_message)
            output_tokens = self.get_token_count(full_response)
            
            tokens = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
            
            # Update the response info with token usage
            response_info["usage"] = tokens
            
            # Track the request
            self._track_request(model_name, tokens)
            
            # Add timing information
            response_info["request_time"] = request_time
            
            return response_info
            
        except Exception as e:
            error_message = str(e)
            
            if "rate limit" in error_message.lower():
                raise RateLimitError(f"OpenAI rate limit exceeded: {e}")
            elif "auth" in error_message.lower() or "api key" in error_message.lower():
                raise AuthenticationError(f"OpenAI authentication failed: {e}")
            else:
                raise RequestError(f"OpenAI request failed: {e}")
    
    def _get_tokenizer(self, model_name: str) -> None:
        """
        Get a tokenizer for the specified model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Tokenizer for the model
        """
        if not tiktoken:
            raise ImportError("tiktoken not installed, cannot count tokens")
            
        if model_name in self.tokenizers:
            return self.tokenizers[model_name]
            
        try:
            # Get the encoding for the model
            if model_name.startswith("gpt-4"):
                encoding_name = "cl100k_base"
            elif model_name.startswith("gpt-3.5-turbo"):
                encoding_name = "cl100k_base"
            else:
                # Try to find the right encoding for the model
                encoding_name = tiktoken.encoding_for_model(model_name)
                
            encoding = tiktoken.get_encoding(encoding_name)
            self.tokenizers = {**self.tokenizers, model_name: encoding}
            return encoding
            
        except Exception as e:
            self.logger.warning(f"Failed to load tokenizer for {model_name}: {e}")
            # Fall back to a default encoding if one is available
            if not self.tokenizers:
                # Try to create a default tokenizer
                try:
                    default_encoding = tiktoken.get_encoding("cl100k_base")
                    self.tokenizers = {**self.tokenizers, "default": default_encoding}
                    return default_encoding
                except Exception as e2:
                    self.logger.error(f"Failed to load default tokenizer: {e2}")
                    raise ImportError("Could not initialize any tokenizers") from e2
            
            # Use the first available tokenizer
            return next(iter(self.tokenizers.values()))
    
    def get_token_count(self, text: str) -> int:
        """
        Get the number of tokens in the text for OpenAI models.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
            
        if not tiktoken:
            # If tiktoken isn't available, use a rough estimate
            # This is a very rough approximation (1 token ≈ 4 chars)
            return len(text) // 4
            
        try:
            # Try to get the tokenizer for the current model
            model_name = self.model_name or "gpt-3.5-turbo"
            tokenizer = self._get_tokenizer(model_name)
            
            # Count tokens
            tokens = tokenizer.encode(text)
            return len(tokens)
            
        except Exception as e:
            self.logger.warning(f"Error counting tokens: {e}")
            # Fall back to rough estimate if token counting fails
            return len(text) // 4
