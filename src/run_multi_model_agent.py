"""
Multi-Model Agent with Mistral and Claude

This script sets up and runs an agent that can use both:
1. Local Mistral LLM for privacy-sensitive operations
2. Anthropic Claude for more complex reasoning tasks

The script uses the adapter architecture and model selection
to intelligently route tasks to the appropriate model.
"""

import os
import asyncio
import logging
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import ctransformers if available
try:
    from ctransformers import AutoModelForCausalLM
except ImportError:
    AutoModelForCausalLM = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("multi_model_agent")

# Import LLM infrastructure
from llm_adapter_interface import LLMAdapterRegistry, BaseLLMAdapter
from anthropic_llm_adapter import AnthropicClaudeAdapter

# Import Mistral adapter only if we need it, to avoid errors when it's not installed
try:
    from mistral_llm_adapter import MistralAdapter
    MISTRAL_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ Mistral adapter not available. Local models will not be usable.")
    MISTRAL_AVAILABLE = False

from llm_selector import LLMSelector
from llm_key_manager import LLMKeyManager


class MultiModelAgent:
    """
    Agent that can intelligently route tasks between local Mistral and cloud Claude models.
    """
    
    def __init__(self, registry: LLMAdapterRegistry, selector: LLMSelector):
        """
        Initialize the agent with the model registry and selector.
        
        Args:
            registry: LLM adapter registry with registered models
            selector: LLM selector for intelligent model routing
        """
        self.registry = registry
        self.selector = selector
        self.conversation_history = []
        
        # Check which providers are available
        self.available_providers = registry.list_adapters()
        logger.info(f"Available LLM providers: {', '.join(self.available_providers)}")
    
    async def process_message(self, message: str, metadata: Optional[Dict] = None) -> str:
        """
        Process a user message and return a response.
        
        Args:
            message: User message to process
            metadata: Optional metadata about the message (privacy level, complexity, etc.)
            
        Returns:
            Response from the appropriate LLM
        """
        # Set default metadata
        metadata = metadata or {}
        
        # Extract task properties (or use defaults)
        task_complexity = metadata.get("complexity", LLMSelector.COMPLEXITY_MODERATE)
        privacy_requirement = metadata.get("privacy", LLMSelector.PRIVACY_STANDARD)
        latency_requirement = metadata.get("latency", LLMSelector.LATENCY_MEDIUM)
        cost_preference = metadata.get("cost", LLMSelector.COST_BALANCED)
        
        # Force local model for maximum privacy if specified
        if privacy_requirement == LLMSelector.PRIVACY_MAXIMUM:
            logger.info("Privacy-sensitive task detected, using local model")
            adapter = self.registry.get_adapter("mistral-local")
            if adapter:
                model_name = getattr(adapter, "model_name", "local-mistral")
                return await self._get_response(adapter, message, model_name)
            else:
                # Fall back to Claude if local model not available
                logger.warning("⚠️ Local model not available for privacy-sensitive task")
                logger.warning("⚠️ Using Claude instead - this will send data to Anthropic's cloud API")
                adapter = self.registry.get_adapter("anthropic")
                if adapter:
                    model_name = getattr(adapter, "model_name", "claude-3-haiku-20240307")
                    return await self._get_response(adapter, message, model_name)
                else:
                    return "No suitable model found. Please check your configuration."
        
        # Select the most appropriate model
        logger.info(f"Selecting model for task (complexity={task_complexity}, privacy={privacy_requirement})")
        model_info = self.selector.select_model(
            task_complexity=task_complexity,
            privacy_requirement=privacy_requirement,
            latency_requirement=latency_requirement,
            cost_preference=cost_preference
        )
        
        if not model_info["adapter"]:
            # Try to use Claude as a fallback
            adapter = self.registry.get_adapter("anthropic")
            if adapter:
                logger.info("Using Claude as fallback since no suitable model was found")
                model_name = getattr(adapter, "model_name", "claude-3-haiku-20240307")
                return await self._get_response(adapter, message, model_name)
            else:
                return "No suitable model found for this task. Please check your configuration."
        
        # Log selection info
        provider = model_info["provider"]
        model = model_info["model"]
        logger.info(f"Selected model: {provider}/{model}")
        if "reasons" in model_info:
            logger.info(f"Selection reasons: {', '.join(model_info['reasons'][:3])}")
        
        # Get response from selected model
        return await self._get_response(model_info["adapter"], message, model)
    
    async def _get_response(self, adapter: BaseLLMAdapter, message: str, model: Optional[str] = None) -> str:
        """Get response from a specific adapter."""
        try:
            # Update conversation history
            self.conversation_history = [*self.conversation_history, {"role": "user", "content": message}]
            
            # For simple completion
            response = await adapter.complete(
                prompt=message,
                model=model,
                max_tokens=1000,
                temperature=0.7
            )
            
            # Update conversation history
            self.conversation_history = [*self.conversation_history, {"role": "assistant", "content": response}]
            
            return response
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            return f"Error: {str(e)}"


async def setup_llm_infrastructure(anthropic_key: str, local_model_path: Optional[str] = None):
    """
    Set up LLM infrastructure with Anthropic and Mistral.
    
    Args:
        anthropic_key: Anthropic API key
        local_model_path: Path to local Mistral model (optional)
        
    Returns:
        Tuple of (registry, selector)
    """
    logger.info("Setting up LLM infrastructure...")
    
    # Initialize the registry and key manager
    registry = LLMAdapterRegistry()
    
    # Register Anthropic adapter if key provided
    if anthropic_key:
        try:
            logger.info("Initializing Anthropic Claude adapter...")
            anthropic_adapter = AnthropicClaudeAdapter(
                api_key=anthropic_key,
                model_name="claude-3-haiku-20240307"  # Fast and cost-effective
            )
            registry.register_adapter(anthropic_adapter, "anthropic")
            logger.info("✅ Registered Anthropic Claude adapter")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Anthropic adapter: {str(e)}")
    else:
        logger.warning("⚠️ No Anthropic API key provided, Claude will not be available")
    
    # Try to find and register local Mistral model if the adapter is available
    local_model_found = False
    
    # Skip local model setup if SKIP_LOCAL_MODELS is set
    if os.environ.get("SKIP_LOCAL_MODELS", "").lower() in ("true", "1", "yes"):
        logger.info("Skipping local model setup (SKIP_LOCAL_MODELS=true)")
    elif MISTRAL_AVAILABLE:
        logger.info("Looking for local Mistral model...")
        
        # Use provided path or search in common locations
        model_paths = []
        if local_model_path:
            model_paths.append(local_model_path)
        
        model_paths.extend([
            "models/dummy-mistral-model.gguf",  # The path from the file list
            "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            "models/mistral-7b-instruct.gguf",
            "models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
            "models/mixtral-8x7b-instruct.gguf"
        ])
        
        for path in model_paths:
            if os.path.exists(path):
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
    else:
        logger.warning("⚠️ Mistral adapter not available - please run setup.bat/setup.sh to install requirements")
    
    if not local_model_found:
        logger.warning("⚠️ No local model found. Only Claude will be available.")
        
        # If we have Claude, we can still proceed
        if registry.get_adapter("anthropic"):
            logger.info("✅ Claude is available. Proceeding with cloud-only configuration.")
        else:
            # No adapters available
            logger.error("❌ No LLM adapters available. Please provide a valid API key or local model.")
            return None, None
    
    # Create and return the selector
    selector = LLMSelector(registry)
    
    return registry, selector


async def interactive_loop(agent: MultiModelAgent):
    """Run an interactive loop with the agent."""
    logger.info("\n=== Multi-Model Agent Interactive Mode ===")
    logger.info("Type 'exit' or 'quit' to end the session")
    logger.info("Use /privacy, /balanced, or /cloud to change privacy settings")
    logger.info("Use /simple, /moderate, or /complex to change complexity settings")
    
    # Default settings
    current_settings = {
        "privacy": LLMSelector.PRIVACY_STANDARD,
        "complexity": LLMSelector.COMPLEXITY_MODERATE,
        "cost": LLMSelector.COST_BALANCED,
        "latency": LLMSelector.LATENCY_MEDIUM
    }
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                logger.info("Exiting...")
                break
            
            # Check for commands
            if user_input.startswith("/"):
                command = user_input.lower()
                
                # Privacy settings
                if command == "/privacy":
                    current_settings["privacy"] = LLMSelector.PRIVACY_MAXIMUM
                    print("✅ Privacy set to MAXIMUM (will use local model if available, otherwise Claude)")
                    continue
                elif command == "/balanced":
                    current_settings["privacy"] = LLMSelector.PRIVACY_HIGH
                    print("✅ Privacy set to HIGH (prefers local but may use cloud)")
                    continue
                elif command == "/cloud":
                    current_settings["privacy"] = LLMSelector.PRIVACY_STANDARD
                    print("✅ Privacy set to STANDARD (will select best model)")
                    continue
                
                # Complexity settings
                elif command == "/simple":
                    current_settings["complexity"] = LLMSelector.COMPLEXITY_SIMPLE
                    print("✅ Complexity set to SIMPLE")
                    continue
                elif command == "/moderate":
                    current_settings["complexity"] = LLMSelector.COMPLEXITY_MODERATE
                    print("✅ Complexity set to MODERATE")
                    continue
                elif command == "/complex":
                    current_settings["complexity"] = LLMSelector.COMPLEXITY_COMPLEX
                    print("✅ Complexity set to COMPLEX")
                    continue
                
                # Help command
                elif command == "/help":
                    print("\n=== Available Commands ===")
                    print("/privacy - Use maximum privacy (local model if available)")
                    print("/balanced - Use balanced privacy (prefers local)")
                    print("/cloud - Use standard privacy (selects best model)")
                    print("/simple - Set task complexity to simple")
                    print("/moderate - Set task complexity to moderate")
                    print("/complex - Set task complexity to complex")
                    print("/help - Show this help message")
                    print("/exit or /quit - Exit the program")
                    continue
            
            # Process the message
            print("\nProcessing...")
            response = await agent.process_message(user_input, current_settings)
            
            print(f"\nAssistant: {response}")
            
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user, exiting...")
            break
        except Exception as e:
            logger.error(f"Error in interactive loop: {str(e)}")


async def main():
    """Main function to run the Multi-Model Agent."""
    parser = argparse.ArgumentParser(description="Run a Multi-Model Agent using Mistral and Claude")
    parser.add_argument("--claude-key", help="Anthropic Claude API key")
    parser.add_argument("--model-path", help="Path to local Mistral model file (.gguf)")
    args = parser.parse_args()
    
    # Get Claude API key (from args, env var, or prompt)
    claude_key = args.claude_key or os.environ.get("ANTHROPIC_API_KEY")
    if not claude_key:
        claude_key = input("Enter your Claude API key (or press Enter to skip): ").strip()
    
    # Get model path (from args, env var, or search)
    model_path = args.model_path or os.environ.get("LOCAL_MODEL_PATH")
    
    # Set up LLM infrastructure
    registry, selector = await setup_llm_infrastructure(claude_key, model_path)
    
    if not registry or not selector:
        logger.error("Failed to set up LLM infrastructure.")
        return
    
    # Create agent
    agent = MultiModelAgent(registry, selector)
    
    # Run interactive loop
    await interactive_loop(agent)
    
    # Clean up
    await registry.close_all()
    
    logger.info("Agent session ended.")


if __name__ == "__main__":
    asyncio.run(main())
