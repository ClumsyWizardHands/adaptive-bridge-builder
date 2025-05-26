"""
LLM Interaction Example

This script demonstrates how to use the LLM adapters to interact with
different language models through a unified interface. It shows how to
retrieve an adapter from the registry, send requests, and process responses.
"""

import asyncio
import logging
import argparse
from typing import Any, Coroutine, Optional

from llm_key_manager import LLMKeyManager
from llm_adapter_interface import LLMAdapterRegistry, BaseLLMAdapter
from openai_llm_adapter import OpenAIGPTAdapter
from anthropic_llm_adapter import AnthropicClaudeAdapter


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("llm_interaction")


async def setup_registry() -> LLMAdapterRegistry:
    """
    Set up the LLM adapter registry with available adapters.
    
    Returns:
        Configured LLMAdapterRegistry
    """
    # Create the key manager
    key_manager = LLMKeyManager()
    
    # Create the registry
    registry = LLMAdapterRegistry()
    
    # Try to register OpenAI adapter
    try:
        openai_adapter = OpenAIGPTAdapter(
            key_manager=key_manager,
            model_name="gpt-3.5-turbo"  # Use a less expensive model for examples
        )
        registry.register_adapter(openai_adapter)
        logger.info("Registered OpenAI adapter")
    except Exception as e:
        logger.warning(f"Could not register OpenAI adapter: {e}")
    
    # Try to register Anthropic adapter
    try:
        anthropic_adapter = AnthropicClaudeAdapter(
            key_manager=key_manager,
            model_name="claude-3-haiku-20240307"  # Use a less expensive model for examples
        )
        registry.register_adapter(anthropic_adapter)
        logger.info("Registered Anthropic adapter")
    except Exception as e:
        logger.warning(f"Could not register Anthropic adapter: {e}")
    
    return registry


async def send_request_to_provider(
    registry: LLMAdapterRegistry,
    provider: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> str:
    """
    Send a request to a specific LLM provider.
    
    Args:
        registry: The LLM adapter registry
        provider: Name of the provider (e.g., "openai", "anthropic")
        prompt: Prompt to send to the LLM
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        The response text
        
    Raises:
        KeyError: If the provider is not found
        Various LLMAdapterError subclasses for other errors
    """
    try:
        # Get the adapter for the specified provider
        adapter = registry.get_adapter(provider)
        
        # Log some information about the adapter
        logger.info(f"Using {provider} adapter with model {adapter.model_name}")
        
        # Send the request
        response = await adapter.send_request(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Process the response
        result = adapter.process_response(response)
        
        # Log token usage if available
        if "usage" in response:
            usage = response["usage"]
            logger.info(
                f"Token usage: {usage.get('prompt_tokens', 0)} prompt, "
                f"{usage.get('completion_tokens', 0)} completion, "
                f"{usage.get('total_tokens', 0)} total"
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error while using {provider} adapter: {e}")
        raise


async def demonstrate_streaming(
    registry: LLMAdapterRegistry,
    provider: str,
    prompt: str
) -> None:
    """
    Demonstrate streaming responses from an LLM.
    
    Args:
        registry: The LLM adapter registry
        provider: Name of the provider (e.g., "openai", "anthropic")
        prompt: Prompt to send to the LLM
    """
    try:
        # Get the adapter for the specified provider
        adapter = registry.get_adapter(provider)
        
        logger.info(f"Streaming response from {provider} (model: {adapter.model_name})...")
        
        # Define a callback for handling streaming chunks
        def chunk_callback(chunk: str) -> None:
            print(chunk, end="", flush=True)
        
        # Send streaming request
        await adapter.stream_request(
            prompt=prompt,
            callback=chunk_callback
        )
        
        print("\n")  # Add a newline after streaming is complete
        
    except Exception as e:
        logger.error(f"Error while streaming from {provider} adapter: {e}")
        raise


async def compare_providers(
    registry: LLMAdapterRegistry,
    prompt: str,
    providers: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Compare responses from multiple providers for the same prompt.
    
    Args:
        registry: The LLM adapter registry
        prompt: Prompt to send to the LLMs
        providers: List of provider names to compare (defaults to all registered)
        
    Returns:
        Dict mapping provider names to their responses
    """
    if not providers:
        providers = registry.list_adapters()
    
    results = {}
    
    for provider in providers:
        try:
            response = await send_request_to_provider(registry, provider, prompt)
            results[provider] = response
        except Exception as e:
            results[provider] = f"ERROR: {e}"
    
    return results


async def main() -> Coroutine[Any, Any, None]:
    """Run the LLM interaction example."""
    parser = argparse.ArgumentParser(description="Demonstrate LLM interactions")
    parser.add_argument(
        "--provider", 
        choices=["openai", "anthropic", "auto"],
        default="auto",
        help="LLM provider to use (default: auto)"
    )
    parser.add_argument(
        "--prompt", 
        default="Explain the concept of language model adapters in 3 sentences.",
        help="Prompt to send to the LLM"
    )
    parser.add_argument(
        "--stream", 
        action="store_true",
        help="Use streaming mode for the response"
    )
    parser.add_argument(
        "--compare", 
        action="store_true",
        help="Compare responses from all available providers"
    )
    args = parser.parse_args()
    
    # Set up the registry with available adapters
    registry = await setup_registry()
    available_providers = registry.list_adapters()
    
    if not available_providers:
        logger.error("No LLM adapters could be registered. Check API keys and dependencies.")
        return
    
    print(f"Available providers: {', '.join(available_providers)}")
    
    # Determine the provider to use
    provider = args.provider
    if provider == "auto" or provider not in available_providers:
        provider = available_providers[0]
        logger.info(f"Using {provider} as the selected provider")
    
    # Determine what to do based on arguments
    if args.compare:
        print("\n=== Comparing LLM Providers ===")
        results = await compare_providers(registry, args.prompt)
        
        for provider_name, response in results.items():
            print(f"\n--- {provider_name.upper()} ---")
            print(response)
            
    elif args.stream:
        print("\n=== Streaming Response ===")
        await demonstrate_streaming(registry, provider, args.prompt)
        
    else:
        print("\n=== Standard Request ===")
        response = await send_request_to_provider(registry, provider, args.prompt)
        print(f"\nResponse from {provider.upper()}:")
        print(response)


if __name__ == "__main__":
    asyncio.run(main())
