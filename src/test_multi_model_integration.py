"""
Multi-Model LLM Integration Test

This script demonstrates the multi-model LLM integration architecture, including:
1. Setting up multiple LLM adapters (cloud and local)
2. Using the LLMSelector to choose appropriate models for different tasks
3. Comparing performance and outputs of different models

Usage:
1. Download the Mixtral 8x7B model (preferably Q4_K_M) from Hugging Face
2. Place it in the models directory as mixtral-8x7b.gguf
3. Set API keys as environment variables
4. Install required dependencies: llama-cpp-python (or llama-cpp-python-cuda for GPU acceleration)
5. Run this script
"""

import asyncio
import os
import time
import logging
from typing import Any, Coroutine, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("multi_model_test")

# Check if we're in the right directory
if not os.path.exists("src") or not os.path.exists("models"):
    print("Please run this script from the project root directory.")
    print("Current directory:", os.getcwd())
    exit(1)

# Import our LLM infrastructure
from llm_adapter_interface import LLMAdapterRegistry
from openai_llm_adapter import OpenAIGPTAdapter
from anthropic_llm_adapter import AnthropicAdapter
from google_llm_adapter import GoogleGeminiAdapter
from mistral_llm_adapter import MistralAdapter
from llm_selector import LLMSelector
from llm_key_manager import LLMKeyManager
from principle_engine_llm_enhanced import EnhancedPrincipleEngine


class ApiKeyProvider:
    """Simple utility to provide API keys for testing."""
    
    @staticmethod
    def get_api_key(provider: str) -> Optional[str]:
        """Get API key from environment variables."""
        env_var_names = {
            "openai": ["OPENAI_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
            "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
            "mistral": ["MISTRAL_API_KEY"]
        }
        
        names = env_var_names.get(provider.lower(), [f"{provider.upper()}_API_KEY"])
        
        for name in names:
            key = os.environ.get(name)
            if key:
                return key
        
        return None


async def setup_llm_infrastructure() -> Coroutine[Any, Any, Tuple[Any, ...]]:
    """
    Set up the LLM infrastructure with multiple providers.
    
    Returns:
        Tuple of (registry, key_manager)
    """
    logger.info("Setting up LLM infrastructure...")
    
    # Initialize the key manager
    key_manager = LLMKeyManager()
    key_provider = ApiKeyProvider()
    
    # Initialize the LLM adapter registry
    registry = LLMAdapterRegistry()
    
    # Register OpenAI adapter if API key is available
    try:
        openai_api_key = key_provider.get_api_key("openai")
        if openai_api_key:
            openai_adapter = OpenAIGPTAdapter(
                api_key=openai_api_key,
                model_name="gpt-4o"  # Default to the latest model
            )
            registry.register_adapter(openai_adapter)
            logger.info("✅ Registered OpenAI GPT adapter")
    except Exception as e:
        logger.warning(f"❌ Could not register OpenAI adapter: {e}")
    
    # Register Anthropic adapter if API key is available
    try:
        anthropic_api_key = key_provider.get_api_key("anthropic")
        if anthropic_api_key:
            anthropic_adapter = AnthropicAdapter(
                api_key=anthropic_api_key,
                model_name="claude-3-opus-20240229"  # Default to the latest model
            )
            registry.register_adapter(anthropic_adapter)
            logger.info("✅ Registered Anthropic Claude adapter")
    except Exception as e:
        logger.warning(f"❌ Could not register Anthropic adapter: {e}")
    
    # Register Google adapter if API key is available
    try:
        google_api_key = key_provider.get_api_key("google")
        if google_api_key:
            google_adapter = GoogleGeminiAdapter(
                api_key=google_api_key,
                model_name="gemini-1.5-pro"  # Default to the latest model
            )
            registry.register_adapter(google_adapter)
            logger.info("✅ Registered Google Gemini adapter")
    except Exception as e:
        logger.warning(f"❌ Could not register Google adapter: {e}")
    
    # Register Mistral adapter if API key is available
    try:
        mistral_api_key = key_provider.get_api_key("mistral")
        if mistral_api_key:
            mistral_adapter = MistralAdapter(
                api_key=mistral_api_key,
                model_name="mistral-medium-latest"
            )
            registry.register_adapter(mistral_adapter)
            logger.info("✅ Registered Mistral Cloud adapter")
    except Exception as e:
        logger.warning(f"❌ Could not register Mistral Cloud adapter: {e}")
    
    # Try to register the local Mixtral model
    try:
        # First, look for the model in the models directory
        model_paths = [
            "models/mixtral-8x7b.gguf",                # Default name
            "models/mixtral-8x7b-v0.1.Q4_K_M.gguf",    # Full name with quantization
            "models/mixtral-8x7b-v0.1.gguf",           # Base name
        ]
        
        # Also check environment variable
        local_model_path = os.environ.get("LOCAL_MODEL_PATH")
        if local_model_path:
            model_paths.insert(0, local_model_path)
        
        # Try each path
        found_model = False
        for path in model_paths:
            if os.path.exists(path):
                logger.info(f"Found local model at {path}")
                mistral_local = MistralAdapter(
                    model_name="mixtral-8x7b",
                    endpoint_url=path,
                    local_deployment=True
                )
                registry.register_adapter(mistral_local, name="mistral-local")
                logger.info("✅ Registered Local Mistral (Mixtral 8x7B) adapter")
                found_model = True
                break
        
        if not found_model:
            logger.warning(
                "⚠️ Local model not found. Please download Mixtral 8x7B model and save it to models/mixtral-8x7b.gguf\n" +
                "Download from: https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF/tree/main"
            )
    except Exception as e:
        logger.warning(f"❌ Could not register Local Mistral adapter: {e}")
        logger.warning("Make sure you have the required libraries installed:")
        logger.warning("- For CPU: pip install llama-cpp-python")
        logger.warning("- For NVIDIA GPU: pip install llama-cpp-python-cuda")
    
    # Check if at least one adapter was registered
    providers = registry.list_adapters()
    if not providers:
        logger.error("❌ No LLM adapters could be registered. Please check your API keys.")
        logger.error("Set API keys as environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)")
    else:
        logger.info(f"Registered adapters: {', '.join(providers)}")
    
    return registry, key_manager


async def test_model_selection(registry) -> None:
    """
    Test the LLM selector with different selection criteria.
    
    Args:
        registry: LLMAdapterRegistry instance
    """
    logger.info("\n=== Testing LLM Model Selection ===")
    
    # Create the LLM selector
    selector = LLMSelector(registry)
    
    # Test prompts for different types of tasks
    test_cases = [
        {
            "name": "Simple fact retrieval (low complexity, high speed)",
            "prompt": "What is the capital of France?",
            "criteria": {
                "task_complexity": LLMSelector.COMPLEXITY_SIMPLE,
                "latency_requirement": LLMSelector.LATENCY_LOW,
                "cost_preference": LLMSelector.COST_LOWEST
            }
        },
        {
            "name": "Creative writing (moderate complexity)",
            "prompt": "Write a short poem about artificial intelligence.",
            "criteria": {
                "task_complexity": LLMSelector.COMPLEXITY_MODERATE,
                "latency_requirement": LLMSelector.LATENCY_MEDIUM,
                "cost_preference": LLMSelector.COST_BALANCED
            }
        },
        {
            "name": "Ethical reasoning (high complexity)",
            "prompt": "Analyze the ethical implications of using facial recognition in public spaces.",
            "criteria": {
                "task_complexity": LLMSelector.COMPLEXITY_COMPLEX,
                "latency_requirement": LLMSelector.LATENCY_MEDIUM,
                "cost_preference": LLMSelector.COST_PERFORMANCE
            }
        },
        {
            "name": "Privacy-sensitive query",
            "prompt": "Summarize this confidential project plan for executive review.",
            "criteria": {
                "task_complexity": LLMSelector.COMPLEXITY_MODERATE,
                "latency_requirement": LLMSelector.LATENCY_MEDIUM,
                "privacy_requirement": LLMSelector.PRIVACY_HIGH
            }
        }
    ]
    
    # Test each case
    for i, case in enumerate(test_cases):
        logger.info(f"\n--- Test {i+1}: {case['name']} ---")
        
        try:
            # Select model based on criteria
            model_info = selector.select_model(**case["criteria"])
            
            logger.info(f"Selected model: {model_info['provider']}/{model_info['model']}")
            logger.info(f"Selection score: {model_info['score']:.2f}")
            logger.info(f"Top reasons: {model_info['reasons'][:3]}")
            
            # Execute the request if an adapter was selected
            if model_info["adapter"]:
                logger.info(f"Testing with prompt: \"{case['prompt']}\"")
                
                start_time = time.time()
                response = await model_info["adapter"].complete(
                    prompt=case["prompt"],
                    model=model_info["model"],
                    temperature=0.7,
                    max_tokens=300
                )
                end_time = time.time()
                
                logger.info(f"Response time: {end_time - start_time:.2f} seconds")
                logger.info(f"Response (truncated): \"{response[:150]}...\"")
        except Exception as e:
            logger.error(f"Error testing {case['name']}: {e}")
    
    # Show statistics
    stats = selector.get_usage_statistics()
    logger.info("\n=== Model Selection Statistics ===")
    logger.info(f"Total selections: {stats['total_selections']}")
    logger.info(f"By provider: {stats['by_provider']}")
    logger.info(f"By model: {stats['by_model']}")


async def test_principle_evaluation(registry) -> None:
    """
    Test the enhanced principle engine with multi-model LLM integration.
    
    Args:
        registry: LLMAdapterRegistry instance
    """
    logger.info("\n=== Testing Enhanced Principle Engine with LLM Integration ===")
    
    # Define some example principles
    principles = [
        {
            "id": "privacy-1",
            "title": "Respect for User Privacy",
            "description": "User data should be collected and processed only with informed consent, and users should maintain control over their personal information.",
            "type": "privacy",
            "tags": ["privacy", "consent", "data"],
            "examples": [
                "Asking for explicit permission before collecting location data",
                "Providing clear options to delete personal information"
            ]
        },
        {
            "id": "fairness-1",
            "title": "Algorithmic Fairness",
            "description": "Systems should not discriminate against individuals or groups based on protected characteristics such as race, gender, or age.",
            "type": "fairness",
            "tags": ["fairness", "bias", "discrimination", "equality"],
            "examples": [
                "Testing recommendation algorithms for demographic biases",
                "Ensuring equal quality of service across different user groups"
            ]
        },
        {
            "id": "safety-1",
            "title": "Prevention of Harm",
            "description": "The system should avoid actions that could reasonably be expected to cause physical, psychological, or material harm to users or others.",
            "type": "safety",
            "tags": ["safety", "harm", "protection"],
            "examples": [
                "Implementing content warnings for potentially disturbing material",
                "Refusing to generate instructions for dangerous activities"
            ]
        }
    ]
    
    # Create the enhanced principle engine
    engine = EnhancedPrincipleEngine(
        registry=registry,
        principles=principles
    )
    
    # Test cases
    test_actions = [
        {
            "name": "Benign Action",
            "action": {
                "intent": "Improve user experience",
                "description": "Analyze anonymized user interaction patterns to improve site navigation, using only data from users who have explicitly opted in.",
                "impact_scope": "narrow",
                "effect_duration": "temporary"
            },
            "context": {
                "situation": "Platform improvement initiative",
                "user": "Product team",
                "constraints": "Must maintain user privacy and trust"
            }
        },
        {
            "name": "Privacy-Violating Action",
            "action": {
                "intent": "Increase user engagement",
                "description": "Track detailed user location data continuously in the background without explicit notification, and use this to target location-based advertisements.",
                "impact_scope": "wide",
                "effect_duration": "permanent"
            },
            "context": {
                "situation": "Revenue growth initiative",
                "user": "Marketing team",
                "constraints": "Need to show 20% engagement improvement"
            }
        }
    ]
    
    # Evaluate each test case
    for test_case in test_actions:
        logger.info(f"\n--- Testing: {test_case['name']} ---")
        
        try:
            # Time the evaluation
            start_time = time.time()
            
            result = await engine.evaluate_action_with_context(
                action=test_case["action"],
                context=test_case["context"]
            )
            
            end_time = time.time()
            
            # Log the results
            logger.info(f"Evaluation time: {end_time - start_time:.2f} seconds")
            logger.info(f"Overall score: {result['score']:.2f}/100")
            logger.info(f"Passed: {result['passed']}")
            
            if result.get("violated_principles"):
                logger.info(f"Violated principles: {result['violated_principles']}")
            
            if result.get("warnings"):
                logger.info(f"Warnings: {len(result['warnings'])}")
                for warning in result["warnings"][:2]:  # Show first 2 warnings
                    logger.info(f"  - {warning['principle']}: {warning['message']}")
            
            # Show one explanation if available
            if result.get("explanations"):
                explanation = result["explanations"][0]
                logger.info(f"\nSample explanation for {explanation['principle_title']}:")
                logger.info(f"Status: {explanation['status']}")
                explanation_text = explanation['explanation']
                logger.info(f"Explanation (truncated): {explanation_text[:200]}...")
            
            # Show one alternative if available
            if result.get("alternatives"):
                alternative = result["alternatives"][0]
                logger.info(f"\nSample alternative suggestion:")
                logger.info(f"Description: {alternative['description']}")
                recommendation = alternative.get('recommendation', '')
                logger.info(f"Recommendation (truncated): {recommendation[:200]}...")
            
            # Show models used
            if result.get("evaluation_models_used"):
                logger.info(f"\nModels used: {result['evaluation_models_used']}")
                
        except Exception as e:
            logger.error(f"Error in principle evaluation: {e}")
    

async def main() -> Coroutine[Any, Any, None]:
    """Main function to coordinate the tests."""
    # Set up the LLM infrastructure
    registry, key_manager = await setup_llm_infrastructure()
    
    # Exit if no adapters were registered
    if not registry.list_adapters():
        logger.error("No adapters registered. Exiting.")
        return
    
    # Run the tests
    await test_model_selection(registry)
    await test_principle_evaluation(registry)
    
    logger.info("\n=== Test Summary ===")
    logger.info(f"Registered adapter types: {registry.list_adapters()}")
    logger.info("All tests completed")


if __name__ == "__main__":
    asyncio.run(main())
