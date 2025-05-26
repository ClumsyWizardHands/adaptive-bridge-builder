from ctransformers import AutoModelForCausalLM
"""
Anthropic and Mistral Integration Test

This script sets up and tests the multi-model LLM architecture using:
1. Anthropic Claude (using your API key)
2. Mistral models (either local or via API if you have a key)

To run this test:
1. Set your Anthropic API key: 
   - On Windows: set ANTHROPIC_API_KEY=your_key_here
   - On Linux/Mac: export ANTHROPIC_API_KEY=your_key_here
2. Optionally set MISTRAL_API_KEY if you have one
3. Run: python src/test_anthropic_mistral.py
"""

import os
import asyncio
import logging
import json
from pathlib import Path
from typing import Any, Coroutine, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("anthropic_mistral_test")

# Import our LLM infrastructure
from llm_adapter_interface import LLMAdapterRegistry, BaseLLMAdapter
from anthropic_llm_adapter import AnthropicClaudeAdapter
from mistral_llm_adapter import MistralAdapter
from llm_selector import LLMSelector
from llm_key_manager import LLMKeyManager


async def setup_llm_infrastructure() -> Coroutine[Any, Any, None]:
    """Set up LLM infrastructure with Anthropic and Mistral."""
    logger.info("Setting up LLM infrastructure...")
    
    # Initialize the registry and key manager
    registry = LLMAdapterRegistry()
    key_manager = LLMKeyManager()
    
    # Get API keys
    anthropic_key = key_manager.get_key("anthropic") or os.environ.get("ANTHROPIC_API_KEY")
    mistral_key = key_manager.get_key("mistral") or os.environ.get("MISTRAL_API_KEY")
    
    # Check if Anthropic key is available
    if not anthropic_key:
        logger.warning("❌ Anthropic API key not found. Please set ANTHROPIC_API_KEY environment variable.")
        logger.warning("Example: set ANTHROPIC_API_KEY=your_key_here (Windows) or export ANTHROPIC_API_KEY=your_key_here (Linux/Mac)")
    else:
        try:
            # Register Anthropic adapter
            logger.info("Initializing Anthropic Claude adapter...")
            anthropic_adapter = AnthropicClaudeAdapter(
                api_key=anthropic_key,
                model_name="claude-3-sonnet-20240229"  # Or "claude-3-opus-20240229" for more powerful model
            )
            registry.register_adapter(anthropic_adapter, "anthropic")
            logger.info("✅ Registered Anthropic Claude adapter")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Anthropic adapter: {str(e)}")
    
    # Check if Mistral API key is available
    if mistral_key:
        try:
            # Register Mistral cloud adapter
            logger.info("Initializing Mistral cloud adapter...")
            mistral_adapter = MistralAdapter(
                api_key=mistral_key,
                model_name="mistral-medium"
            )
            registry.register_adapter(mistral_adapter, "mistral")
            logger.info("✅ Registered Mistral cloud adapter")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Mistral cloud adapter: {str(e)}")
    
    # Try to find local Mistral model
    logger.info("Looking for local Mistral model...")
    potential_model_paths = [
        "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "models/mistral-7b-instruct.gguf",
        "models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
        "models/mixtral-8x7b-instruct.gguf",
        os.environ.get("LOCAL_MODEL_PATH", "")
    ]
    
    local_model_found = False
    for path in potential_model_paths:
        if path and os.path.exists(path):
            try:
                logger.info(f"Found local model: {path}")
                local_adapter = MistralAdapter(
                    model_name="local-mistral",
                    endpoint_url=path,
                    local_deployment=True,
                    backend="auto"  # Will try llama-cpp first, then ctransformers
                )
                registry.register_adapter(local_adapter, "mistral-local")
                logger.info(f"✅ Registered local Mistral adapter using {path}")
                local_model_found = True
                break
            except Exception as e:
                logger.error(f"❌ Failed to initialize local model: {str(e)}")
    
    if not local_model_found:
        logger.warning("ℹ️ No local model found. To use a local model:")
        logger.warning("1. Download a GGUF Mistral or Mixtral model")
        logger.warning("2. Place it in a 'models' directory or set LOCAL_MODEL_PATH")
        logger.warning("3. Install llama-cpp-python or ctransformers")
    
    # Check if we have any adapters
    providers = registry.list_adapters()
    if not providers:
        logger.error("❌ No LLM adapters registered. Please provide at least one API key or local model.")
        return None
    
    logger.info(f"Registered adapters: {', '.join(providers)}")
    return registry


async def test_model_comparison(registry) -> Coroutine[Any, Any, None]:
    """Run a comparison test between models."""
    logger.info("\n=== Model Comparison Test ===")
    
    if not registry or not registry.list_adapters():
        logger.error("No adapters available for testing")
        return
    
    # Define test prompts of varying complexity
    test_prompts = [
        {
            "name": "Simple question",
            "prompt": "What is the capital of France?"
        },
        {
            "name": "Creative task",
            "prompt": "Write a haiku about artificial intelligence."
        },
        {
            "name": "Analytical task",
            "prompt": "Explain briefly how transformer neural networks work."
        },
        {
            "name": "Ethical reasoning",
            "prompt": "What are the ethical considerations of using facial recognition in public spaces?"
        }
    ]
    
    # Get all available adapters
    available_adapters = registry.list_adapters()
    
    # For each prompt, test all available adapters
    for prompt_info in test_prompts:
        logger.info(f"\n--- Testing: {prompt_info['name']} ---")
        prompt = prompt_info["prompt"]
        
        for provider in available_adapters:
            adapter = registry.get_adapter(provider)
            if not adapter:
                continue
            
            # Get default model name
            model_name = getattr(adapter, "model_name", "default")
            
            logger.info(f"Testing {provider}/{model_name}...")
            
            try:
                # Set a timeout for the completion
                start_time = asyncio.get_event_loop().time()
                
                # Get completion
                response = await adapter.complete(
                    prompt=prompt,
                    max_tokens=200,
                    temperature=0.7
                )
                
                # Calculate response time
                end_time = asyncio.get_event_loop().time()
                response_time = end_time - start_time
                
                # Truncate response for display
                truncated_response = response[:150] + ("..." if len(response) > 150 else "")
                
                logger.info(f"Response time: {response_time:.2f}s")
                logger.info(f"Response: {truncated_response}")
            except Exception as e:
                logger.error(f"Error testing {provider}: {str(e)}")


async def test_model_selection(registry) -> Coroutine[Any, Any, None]:
    """Test the LLM selector with different selection criteria."""
    logger.info("\n=== Model Selection Test ===")
    
    if not registry or not registry.list_adapters():
        logger.error("No adapters available for testing")
        return
    
    # Create the selector
    selector = LLMSelector(registry)
    
    # Define different selection scenarios
    scenarios = [
        {
            "name": "Simple question (optimized for cost)",
            "criteria": {
                "task_complexity": LLMSelector.COMPLEXITY_SIMPLE,
                "cost_preference": LLMSelector.COST_LOWEST
            },
            "prompt": "What is the largest planet in our solar system?"
        },
        {
            "name": "Complex ethical reasoning",
            "criteria": {
                "task_complexity": LLMSelector.COMPLEXITY_COMPLEX,
                "cost_preference": LLMSelector.COST_PERFORMANCE
            },
            "prompt": "Discuss the ethical implications of autonomous weapons systems."
        },
        {
            "name": "Privacy-sensitive task",
            "criteria": {
                "task_complexity": LLMSelector.COMPLEXITY_MODERATE,
                "privacy_requirement": LLMSelector.PRIVACY_MAXIMUM
            },
            "prompt": "Analyze this sensitive data about our upcoming product launch."
        }
    ]
    
    # Test each scenario
    for scenario in scenarios:
        logger.info(f"\n--- Scenario: {scenario['name']} ---")
        
        # Select the model
        model_info = selector.select_model(**scenario["criteria"])
        
        if not model_info["adapter"]:
            logger.warning(f"No suitable model found for {scenario['name']}")
            continue
            
        # Show selection result
        provider = model_info["provider"]
        model = model_info["model"]
        score = model_info["score"]
        
        logger.info(f"Selected model: {provider}/{model} (score: {score:.2f})")
        
        # Show selection reasons
        if "reasons" in model_info:
            logger.info(f"Selection reasons: {', '.join(model_info['reasons'][:3])}")
        
        # Test the selected model
        logger.info(f"Testing with prompt: {scenario['prompt']}")
        
        try:
            # Set a timeout for the completion
            start_time = asyncio.get_event_loop().time()
            
            # Get completion
            response = await model_info["adapter"].complete(
                prompt=scenario["prompt"],
                model=model_info["model"],
                max_tokens=200,
                temperature=0.7
            )
            
            # Calculate response time
            end_time = asyncio.get_event_loop().time()
            response_time = end_time - start_time
            
            # Truncate response for display
            truncated_response = response[:150] + ("..." if len(response) > 150 else "")
            
            logger.info(f"Response time: {response_time:.2f}s")
            logger.info(f"Response: {truncated_response}")
        except Exception as e:
            logger.error(f"Error testing selected model: {str(e)}")


async def main() -> Coroutine[Any, Any, None]:
    """Main function to run the tests."""
    logger.info("Starting Anthropic and Mistral integration test...")
    
    # Set up infrastructure
    registry = await setup_llm_infrastructure()
    
    if not registry or not registry.list_adapters():
        logger.error("Failed to set up LLM infrastructure.")
        return
    
    # Run tests
    await test_model_comparison(registry)
    await test_model_selection(registry)
    
    # Clean up
    await registry.close_all()
    
    logger.info("\n✅ All tests completed")


if __name__ == "__main__":
    asyncio.run(main())