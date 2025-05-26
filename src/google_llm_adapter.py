import google.generativeai as genai
"""
Google Gemini Adapter

This module provides an adapter for Google's Gemini models, implementing the BaseLLMAdapter
interface. It handles authentication, request formatting, and response parsing
specific to Google's Gemini API.
"""

import time
import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Callable, Union

from llm_adapter_interface import (
    BaseLLMAdapter,
    AuthenticationError,
    RequestError,
    ResponseError,
    RateLimitError,
    ModelNotFoundError
)
from llm_key_manager import LLMKeyManager


class GoogleGeminiAdapter(BaseLLMAdapter):
    """
    Adapter for Google's Gemini models.
    
    This adapter implements the BaseLLMAdapter interface for Google's Gemini models,
    handling the specific details of the Google AI API.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-pro",
        project_id: Optional[str] = None,
        key_manager: Optional[LLMKeyManager] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Google Gemini adapter.
        
        Args:
            api_key: Optional API key for Google (if None, will use key_manager)
            model_name: Default model to use (default: gemini-pro)
            project_id: Optional Google Cloud project ID
            key_manager: Optional key manager for retrieving API keys
            logger: Optional logger for tracking events
        """
        # Initialize the base class
        super().__init__(
            provider_name="google",
            api_key=api_key,
            model_name=model_name,
            key_manager=key_manager,
            logger=logger
        )
        
        # Google-specific attributes
        self.project_id = project_id
        self.client = None
        
        # Initialize the Google client
        self._init_client()
        
        # Initialize token counter for Google models
        self.tokenizers = {}
    
    def _init_client(self) -> None:
        """Initialize the Google Gemini client."""
        try:
            # Import Google Generative AI library
            try:
                import google.generativeai as genai
            except ImportError:
                self.logger.error("Google Generative AI library not installed. Please install it with 'pip install google-generativeai'")
                raise ImportError("Google Generative AI library not installed. Please install it with 'pip install google-generativeai'")
            
            # Configure the API key
            genai.configure(api_key=self.api_key)
            
            # Store the genai module for later use
            self.genai = genai
            
            # Test the connection with a models list request
            try:
                models = genai.list_models()
                self.logger.info(f"Successfully connected to Google AI API. Found {len(models)} models.")
            except Exception as e:
                raise AuthenticationError(f"Failed to connect to Google AI API: {e}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Google client: {e}")
            raise AuthenticationError(f"Failed to initialize Google client: {e}")
    
    @property
    def provider_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a dictionary of available Google Gemini models and their capabilities.
        
        Returns:
            Dict mapping model names to their capabilities
        """
        return {
            "gemini-pro": {
                "name": "Gemini Pro",
                "description": "Google's most capable text model for complex tasks",
                "context_size": 32768,
                "capabilities": ["text_generation", "function_calling"],
                "pricing": {
                    "input": 0.00025,  # $ per 1000 tokens (approximate)
                    "output": 0.0005   # $ per 1000 tokens (approximate)
                }
            },
            "gemini-pro-vision": {
                "name": "Gemini Pro Vision",
                "description": "Multimodal model that can process both text and images",
                "context_size": 16384,
                "capabilities": ["text_generation", "image_processing"],
                "pricing": {
                    "input": 0.0025,  # $ per 1000 tokens (approximate)
                    "output": 0.0005  # $ per 1000 tokens (approximate)
                }
            },
            "gemini-1.5-pro": {
                "name": "Gemini 1.5 Pro",
                "description": "Advanced model with longer context and improved capabilities",
                "context_size": 1000000,  # 1M token context window
                "capabilities": ["text_generation", "image_processing", "audio_processing"],
                "pricing": {
                    "input": 0.0005,  # $ per 1000 tokens (approximate)
                    "output": 0.0015  # $ per 1000 tokens (approximate)
                }
            },
            "gemini-1.5-flash": {
                "name": "Gemini 1.5 Flash",
                "description": "Faster, more efficient model for high-throughput applications",
                "context_size": 1000000,  # 1M token context window
                "capabilities": ["text_generation", "image_processing"],
                "pricing": {
                    "input": 0.00025,  # $ per 1000 tokens (approximate)
                    "output": 0.0005  # $ per 1000 tokens (approximate)
                }
            }
        }
    
    async def list_available_models(self) -> List[str]:
        """
        Get a list of available Gemini models from the Google AI API.
        
        Returns:
            List of model names
            
        Raises:
            AuthenticationError: If authentication fails
            RequestError: If the request fails
        """
        try:
            # Request the list of models from the API
            models = self.genai.list_models()
            
            # Extract model names and filter for Gemini models
            model_names = [
                model.name.split('/')[-1] for model in models
                if "gemini" in model.name.lower()
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
        Send a request to the Google Gemini API and get the response.
        
        Args:
            prompt: The prompt to send to the model
            model: Model to use (defaults to self.model_name)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Optional list of strings that stop generation
            **kwargs: Additional Google-specific parameters
                - system_message: System message for the conversation
                - top_p: Nucleus sampling parameter
                - top_k: Number of highest probability tokens to consider
                
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
        
        # Create a safety settings object if provided
        safety_settings = kwargs.get("safety_settings", None)
        
        # Set up generation config
        generation_config = {
            "temperature": temperature
        }
        
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens
            
        if stop_sequences:
            generation_config["stop_sequences"] = stop_sequences
            
        # Add optional parameters if provided
        for param in ["top_p", "top_k"]:
            if param in kwargs:
                generation_config[param] = kwargs[param]
        
        try:
            # Get the model
            gemini_model = self.genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Start timing
            start_time = time.time()
            
            # Send the request
            if system_message:
                # Use chat for system message
                chat = gemini_model.start_chat(system_instruction=system_message)
                response = chat.send_message(prompt)
            else:
                # Use generate_content for simple prompts
                response = gemini_model.generate_content(prompt)
            
            end_time = time.time()
            request_time = end_time - start_time
            
            # Convert to a serializable dictionary
            response_dict = self._convert_response_to_dict(response)
            
            # Add timing information
            response_dict["request_time"] = request_time
            
            # Estimate token usage since Gemini doesn't provide it directly
            prompt_tokens = self.get_token_count(prompt) + self.get_token_count(system_message)
            completion_tokens = self.get_token_count(response_dict.get("text", ""))
            tokens = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
            
            # Track the request
            self._track_request(model_name, tokens)
            
            return response_dict
            
        except Exception as e:
            error_message = str(e)
            
            if "quota" in error_message.lower() or "rate limit" in error_message.lower():
                raise RateLimitError(f"Google rate limit exceeded: {e}")
            elif "auth" in error_message.lower() or "api key" in error_message.lower():
                raise AuthenticationError(f"Google authentication failed: {e}")
            else:
                raise RequestError(f"Google request failed: {e}")
    
    def _convert_response_to_dict(self, response) -> Dict[str, Any]:
        """
        Convert the Google Gemini response object to a serializable dictionary.
        
        Args:
            response: Google Gemini response object
            
        Returns:
            Dict containing the response data
        """
        try:
            # Handle chat response vs generate_content response
            if hasattr(response, "parts"):
                # This is likely a generate_content response
                text = "".join([part.text for part in response.parts if hasattr(part, "text")])
                
                response_dict = {
                    "text": text,
                    "finish_reason": getattr(response, "finish_reason", None),
                    "model": getattr(response, "model", self.model_name),
                    "safety_ratings": [
                        {"category": sr.category, "probability": sr.probability}
                        for sr in getattr(response, "safety_ratings", [])
                    ]
                }
            else:
                # This could be a chat response
                text = response.text if hasattr(response, "text") else str(response)
                
                response_dict = {
                    "text": text,
                    "role": getattr(response, "role", "model"),
                    "model": self.model_name
                }
                
            return response_dict
            
        except Exception as e:
            self.logger.error(f"Error converting response to dict: {e}")
            # Fallback conversion
            return {
                "text": str(response),
                "model": self.model_name
            }
    
    def process_response(self, response: Dict[str, Any]) -> str:
        """
        Process the raw Google Gemini response into a usable format.
        
        Args:
            response: Raw response from send_request
            
        Returns:
            Processed response text
            
        Raises:
            ResponseError: If processing the response fails
        """
        try:
            # Extract the text from the response
            text = response.get("text", "")
            
            if not text and "choices" in response:
                choices = response["choices"]
                if choices and len(choices) > 0:
                    text = choices[0].get("text", "")
                    
            return text
            
        except Exception as e:
            raise ResponseError(f"Failed to process Google Gemini response: {e}")
    
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
        Send a streaming request to the Google Gemini API.
        
        Args:
            prompt: The prompt to send to the model
            callback: Function to call with each text chunk
            model: Model to use (defaults to self.model_name)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Optional list of strings that stop generation
            **kwargs: Additional Google-specific parameters
                
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
        
        # Create a safety settings object if provided
        safety_settings = kwargs.get("safety_settings", None)
        
        # Set up generation config
        generation_config = {
            "temperature": temperature
        }
        
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens
            
        if stop_sequences:
            generation_config["stop_sequences"] = stop_sequences
            
        # Add optional parameters if provided
        for param in ["top_p", "top_k"]:
            if param in kwargs:
                generation_config[param] = kwargs[param]
        
        try:
            # Get the model
            gemini_model = self.genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Start timing
            start_time = time.time()
            full_response = ""
            
            # Initiate streaming request
            if system_message:
                # Use chat for system message
                chat = gemini_model.start_chat(system_instruction=system_message)
                stream = chat.send_message(prompt, stream=True)
            else:
                # Use generate_content for simple prompts
                stream = gemini_model.generate_content(prompt, stream=True)
            
            # Process the stream
            for chunk in stream:
                # Extract content from the chunk
                if hasattr(chunk, "parts") and chunk.parts:
                    for part in chunk.parts:
                        if hasattr(part, "text") and part.text:
                            content = part.text
                            # Append to the full response
                            full_response += content
                            # Call the callback with the chunk
                            callback(content)
                elif hasattr(chunk, "text") and chunk.text:
                    content = chunk.text
                    # Append to the full response
                    full_response += content
                    # Call the callback with the chunk
                    callback(content)
            
            end_time = time.time()
            request_time = end_time - start_time
            
            # Construct the response info
            response_info = {
                "text": full_response,
                "model": model_name,
                "request_time": request_time,
            }
            
            # Estimate token usage
            prompt_tokens = self.get_token_count(prompt) + self.get_token_count(system_message)
            completion_tokens = self.get_token_count(full_response)
            tokens = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
            
            # Add token info to response
            response_info["usage"] = tokens
            
            # Track the request
            self._track_request(model_name, tokens)
            
            return response_info
            
        except Exception as e:
            error_message = str(e)
            
            if "quota" in error_message.lower() or "rate limit" in error_message.lower():
                raise RateLimitError(f"Google rate limit exceeded: {e}")
            elif "auth" in error_message.lower() or "api key" in error_message.lower():
                raise AuthenticationError(f"Google authentication failed: {e}")
            else:
                raise RequestError(f"Google request failed: {e}")
    
    def get_token_count(self, text: str) -> int:
        """
        Get the number of tokens in the text for Google Gemini models.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens (estimated)
        """
        if not text:
            return 0
            
        try:
            # If the Gemini API provides a count_tokens method, use it
            if hasattr(self.genai, "count_tokens"):
                model = self.genai.GenerativeModel(self.model_name)
                result = model.count_tokens(text)
                return result.total_tokens
        except Exception as e:
            self.logger.warning(f"Error using Gemini token counter: {e}")
        
        # Fallback to a rough estimate (1 token ≈ 4 chars)
        return len(text) // 4