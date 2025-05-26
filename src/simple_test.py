"""
Simple LLM Test for Anthropic and Mistral integration.

This script tests basic functionality with your API key.
"""

import os
import asyncio
import logging

from llm_adapter_interface import LLMAdapterRegistry
from anthropic_llm_adapter import AnthropicClaudeAdapter
from mistral_llm_adapter import MistralAdapter
from llm_key_manager import LLMKeyManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simple_test")

async def test_anthropic(api_key):
    """Test Anthropic Claude API."""
    logger.info("Testing Anthropic Claude API")
    
    # Use the full model names as specified in Anthropic docs
    models_to_try = ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"]
    
    for model in models_to_try:
        try:
            logger.info(f"Trying with model: {model}")
            adapter = AnthropicClaudeAdapter(api_key=api_key, model_name=model)
            response = await adapter.complete(
                prompt="Hello! How are you today?",
                max_tokens=100
            )
            
            logger.info(f"Anthropic response from {model}: {response}")
            await adapter.close()
            logger.info(f"✅ Successfully connected to Anthropic API with model: {model}")
            return True
        except Exception as e:
            logger.error(f"Error with Anthropic API using {model}: {str(e)}")
            # Continue to next model
    
    # If we reached here, all models failed
    logger.error("All Anthropic models failed to connect")
    return False

async def test_mistral_mock():
    """Test Mistral adapter with a mock."""
    logger.info("Testing Mistral adapter with mock local model")
    
    try:
        # Create a dummy model path
        dummy_path = "models/dummy-mistral-model.gguf"
        
        # Override the _init_local_model method temporarily to avoid
        # actually trying to load any backends
        original_init = MistralAdapter._init_local_model
        
        def mock_init_local_model(self):
            logger.info("Mock initialization of local model")
            self.backend = "mock"
            self._local_model = {}
            logger.info("Successfully initialized mock Mistral model")
        
        # Apply our monkey patch
        MistralAdapter._init_local_model = mock_init_local_model
        
        # Now create the adapter
        adapter = MistralAdapter(
            endpoint_url=dummy_path,
            local_deployment=True,
            model_name="mock-mistral"
        )
        
        logger.info("Successfully created Mistral adapter instance")
        
        # Restore the original method
        MistralAdapter._init_local_model = original_init
        
        return True
    except Exception as e:
        logger.error(f"Error initializing Mistral adapter: {str(e)}")
        return False

async def main():
    """Main test function."""
    print("=" * 60)
    print("SIMPLE LLM INTEGRATION TEST")
    print("=" * 60)
    
    # Check for API key
    key_manager = LLMKeyManager()
    api_key = key_manager.get_key("anthropic")
    
    if not api_key:
        # Try from environment variable
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        
    if not api_key:
        api_key = input("Enter your Anthropic API key: ")
        
    if not api_key:
        logger.error("No API key provided. Exiting.")
        return
    
    # Create registry
    registry = LLMAdapterRegistry()
    
    # Test Anthropic
    anthropic_success = await test_anthropic(api_key)
    
    # Test mock Mistral
    mistral_success = await test_mistral_mock()
    
    # Report results
    print("\nTest Results:")
    print(f"Anthropic Claude API: {'✅ PASSED' if anthropic_success else '❌ FAILED'}")
    print(f"Mistral Setup: {'✅ PASSED' if mistral_success else '❌ FAILED'}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    asyncio.run(main())
