"""
LLM Key Manager

This module provides a secure way to manage API keys for different LLM providers.
It supports loading keys from environment variables, configuration files, or secure storage.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

class LLMKeyManager:
    """
    Manages API keys for LLM providers.
    
    This class provides a central place to manage API keys for different LLM providers,
    with support for loading keys from various sources and basic security measures.
    """
    
    # Environment variable prefix for API keys
    ENV_PREFIX = ""  # No prefix by default (e.g., ANTHROPIC_API_KEY)
    
    # Default key configuration file name
    DEFAULT_CONFIG_FILE = "llm_keys.json"
    
    def __init__(
        self,
        config_file: Optional[str] = None,
        env_prefix: Optional[str] = None,
        auto_load: bool = True
    ):
        """
        Initialize the key manager.
        
        Args:
            config_file: Path to a JSON configuration file with API keys
            env_prefix: Prefix for environment variables with API keys
            auto_load: Whether to automatically load keys from environment and config file
        """
        self.keys = {}
        self.config_file = config_file or self.DEFAULT_CONFIG_FILE
        self.env_prefix = env_prefix or self.ENV_PREFIX
        
        # Auto-load keys if requested
        if auto_load:
            # Try environment variables first
            self.load_from_environment()
            
            # Then try config file, which can override environment variables
            self.load_from_file()
    
    def get_key(self, provider: str) -> Optional[str]:
        """
        Get the API key for a provider.
        
        Args:
            provider: The provider name (e.g., 'openai', 'anthropic')
            
        Returns:
            The API key, or None if not found
        """
        # First try to get from instance cache
        if provider in self.keys:
            return self.keys[provider]
        
        # Then try environment variables
        env_var_name = f"{self.env_prefix}{provider.upper()}_API_KEY"
        if env_var_name in os.environ:
            key = os.environ[env_var_name]
            self.keys = {**self.keys, provider: key}
            return key
        
        # First try standardized environment variable naming (most common)
        simplified_env_var = f"{provider.upper()}_API_KEY"
        if simplified_env_var in os.environ:
            key = os.environ[simplified_env_var]
            self.keys = {**self.keys, provider: key}
            return key
            
        # Not found
        return None
    
    def set_key(self, provider: str, key: str, save_to_file: bool = False) -> None:
        """
        Set the API key for a provider.
        
        Args:
            provider: The provider name (e.g., 'openai', 'anthropic')
            key: The API key
            save_to_file: Whether to save to the configuration file
        """
        if not key:
            logger.warning(f"Attempted to set empty API key for provider '{provider}'")
            return
        
        # Store in instance cache
        self.keys = {**self.keys, provider: key}
        
        # Save to file if requested
        if save_to_file:
            self.save_to_file()
    
    def load_from_environment(self) -> None:
        """
        Load API keys from environment variables.
        
        Looks for environment variables matching the pattern:
        {ENV_PREFIX}{PROVIDER}_API_KEY
        
        For example, with ENV_PREFIX="LLM_", it would look for:
        LLM_OPENAI_API_KEY, LLM_ANTHROPIC_API_KEY, etc.
        
        It also tries without the prefix for common cases like:
        OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
        """
        # Try to find API keys in environment variables
        prefix = self.env_prefix
        prefix_upper = prefix.upper()
        
        # Find all environment variables with _API_KEY suffix
        for name, value in os.environ.items():
            # Skip empty values
            if not value:
                continue
                
            # Try with prefix
            if prefix and name.startswith(prefix_upper) and name.endswith("_API_KEY"):
                provider = name[len(prefix_upper):].replace("_API_KEY", "").lower()
                self.keys = {**self.keys, provider: value}
                logger.debug(f"Loaded API key for provider '{provider}' from environment variable {name}")
            
            # Try without prefix (more common)
            elif name.endswith("_API_KEY") and not prefix:
                provider = name.replace("_API_KEY", "").lower()
                # Don't override if we already have this key
                if provider not in self.keys:
                    self.keys = {**self.keys, provider: value}
                    logger.debug(f"Loaded API key for provider '{provider}' from environment variable {name}")
    
    def load_from_file(self) -> bool:
        """
        Load API keys from a configuration file.
        
        Returns:
            True if keys were loaded successfully, False otherwise
        """
        # Check if the file exists
        try:
            config_path = Path(self.config_file)
            if not config_path.exists():
                logger.debug(f"LLM key configuration file not found: {self.config_file}")
                return False
            
            # Read and parse the file
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Extract API keys
            if "api_keys" in config and isinstance(config["api_keys"], dict):
                for provider, key in config["api_keys"].items():
                    if key:  # Skip empty keys
                        self.keys = {**self.keys, provider.lower(): key}
                        logger.debug(f"Loaded API key for provider '{provider}' from configuration file")
                return True
            else:
                logger.warning(f"Invalid LLM key configuration format in {self.config_file}")
                return False
                
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error loading LLM key configuration: {str(e)}")
            return False
    
    def save_to_file(self) -> bool:
        """
        Save API keys to a configuration file.
        
        Returns:
            True if keys were saved successfully, False otherwise
        """
        try:
            # Create parent directories if needed
            config_path = Path(self.config_file)
            os.makedirs(config_path.parent, exist_ok=True)
            
            # Create or update the configuration
            config = {"api_keys": self.keys}
            
            # Write to file
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            
            logger.debug(f"Saved API keys to configuration file: {self.config_file}")
            return True
            
        except IOError as e:
            logger.warning(f"Error saving LLM key configuration: {str(e)}")
            return False
    
    def remove_key(self, provider: str, save_to_file: bool = False) -> bool:
        """
        Remove an API key for a provider.
        
        Args:
            provider: The provider name (e.g., 'openai', 'anthropic')
            save_to_file: Whether to save the updated configuration to file
            
        Returns:
            True if the key was removed, False if it wasn't found
        """
        provider = provider.lower()
        if provider in self.keys:
            self.keys = {k: v for k, v in self.keys.items() if k != provider}
            logger.debug(f"Removed API key for provider '{provider}'")
            
            # Save to file if requested
            if save_to_file:
                self.save_to_file()
                
            return True
        
        return False
    
    def get_available_providers(self) -> list:
        """
        Get a list of providers with available API keys.
        
        Returns:
            List of provider names with available keys
        """
        return list(self.keys.keys())
