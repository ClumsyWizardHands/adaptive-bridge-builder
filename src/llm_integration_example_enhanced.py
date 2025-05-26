"""
Enhanced LLM Integration Example

This example demonstrates the multi-model LLM integration architecture, including:
1. Setting up multiple LLM adapters (cloud and local)
2. Using the intelligent LLM selector for different tasks
3. Using the enhanced principle engine with specialized models
4. Running evaluations in parallel using multiple models

To run this example:
1. Set appropriate API keys as environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
2. Optionally download a local model (see test_multi_model_integration.py for details)
3. Run this script
"""

import os
import asyncio
import logging
from typing import Any, Coroutine, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llm_integration_demo")

# Import our LLM infrastructure
from llm_adapter_interface import LLMAdapterRegistry, BaseLLMAdapter
from openai_llm_adapter import OpenAIGPTAdapter
from anthropic_llm_adapter import AnthropicClaudeAdapter
from mistral_llm_adapter import MistralAdapter
from llm_selector import LLMSelector
from principle_engine_llm_enhanced import EnhancedPrincipleEngineLLM


async def setup_llm_infrastructure() -> None:
    """Set up the LLM infrastructure with multiple adapters."""
    logger.info("Setting up LLM infrastructure...")
    
    # Initialize the registry
    registry = LLMAdapterRegistry()
    
    # Add OpenAI adapter if API key is available
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        try:
            openai_adapter = OpenAIGPTAdapter(
                api_key=openai_api_key,
                model_name="gpt-4-turbo"  # Or any other available model
            )
            registry.register_adapter(openai_adapter)
            logger.info("✅ Registered OpenAI adapter")
        except Exception as e:
            logger.warning(f"Could not register OpenAI adapter: {e}")
    
    # Add Anthropic adapter if API key is available
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
    if anthropic_api_key:
        try:
            anthropic_adapter = AnthropicClaudeAdapter(
                api_key=anthropic_api_key,
                model_name="claude-3-sonnet-20240229"  # Or any other available model
            )
            registry.register_adapter(anthropic_adapter)
            logger.info("✅ Registered Anthropic adapter")
        except Exception as e:
            logger.warning(f"Could not register Anthropic adapter: {e}")
    
    # Add Mistral adapter if API key is available
    mistral_api_key = os.environ.get("MISTRAL_API_KEY")
    if mistral_api_key:
        try:
            mistral_adapter = MistralAdapter(
                api_key=mistral_api_key,
                model_name="mistral-medium-latest"
            )
            registry.register_adapter(mistral_adapter)
            logger.info("✅ Registered Mistral Cloud adapter")
        except Exception as e:
            logger.warning(f"Could not register Mistral Cloud adapter: {e}")
    
    # Try to add local Mistral adapter if model file exists
    model_paths = [
        "models/mixtral-8x7b.gguf",                # Default name
        "models/mixtral-8x7b-v0.1.Q4_K_M.gguf",    # Full name with quantization
        os.environ.get("LOCAL_MODEL_PATH", "")      # From environment variable
    ]
    
    for path in model_paths:
        if path and os.path.exists(path):
            try:
                local_adapter = MistralAdapter(
                    model_name="mixtral-8x7b",
                    endpoint_url=path,
                    local_deployment=True
                )
                registry.register_adapter(local_adapter, name="mistral-local")
                logger.info(f"✅ Registered Local Mistral adapter using {path}")
                break
            except Exception as e:
                logger.warning(f"Could not register Local Mistral adapter: {e}")
    
    # Check if we have at least one adapter
    providers = registry.list_adapters()
    if not providers:
        logger.warning("⚠️ No LLM adapters registered. Please set at least one API key.")
    else:
        logger.info(f"Registered {len(providers)} adapters: {', '.join(providers)}")
    
    return registry


async def demo_model_selection(registry) -> None:
    """Demonstrate the LLM selector with different selection criteria."""
    logger.info("\n=== Model Selection Demo ===")
    
    # Create the selector
    selector = LLMSelector(registry)
    
    # Define different selection scenarios
    scenarios = [
        {
            "name": "Simple question (optimized for cost and speed)",
            "criteria": {
                "task_complexity": LLMSelector.COMPLEXITY_SIMPLE,
                "latency_requirement": LLMSelector.LATENCY_LOW,
                "cost_preference": LLMSelector.COST_LOWEST
            },
            "prompt": "What is the capital of France?"
        },
        {
            "name": "Creative writing task (balanced requirements)",
            "criteria": {
                "task_complexity": LLMSelector.COMPLEXITY_MODERATE,
                "latency_requirement": LLMSelector.LATENCY_MEDIUM,
                "cost_preference": LLMSelector.COST_BALANCED
            },
            "prompt": "Write a short poem about artificial intelligence and creativity."
        },
        {
            "name": "Complex reasoning task (performance-oriented)",
            "criteria": {
                "task_complexity": LLMSelector.COMPLEXITY_COMPLEX,
                "latency_requirement": LLMSelector.LATENCY_HIGH,
                "cost_preference": LLMSelector.COST_PERFORMANCE
            },
            "prompt": "Analyze the ethical implications of using facial recognition technology in public spaces."
        },
        {
            "name": "Privacy-sensitive task (maximum privacy)",
            "criteria": {
                "task_complexity": LLMSelector.COMPLEXITY_MODERATE,
                "privacy_requirement": LLMSelector.PRIVACY_MAXIMUM,
                "cost_preference": LLMSelector.COST_BALANCED
            },
            "prompt": "Summarize this sensitive internal document about our upcoming product launch."
        }
    ]
    
    # Test each scenario
    for scenario in scenarios:
        logger.info(f"\n--- Scenario: {scenario['name']} ---")
        
        # Select the model
        model_info = selector.select_model(**scenario["criteria"])
        
        # Show the selection results
        if model_info["adapter"]:
            provider = model_info["provider"]
            model = model_info["model"]
            score = model_info["score"]
            logger.info(f"Selected model: {provider}/{model} (score: {score:.2f})")
            logger.info(f"Selection reasons: {', '.join(model_info['reasons'][:2])}")
            
            # Test the selected model
            logger.info(f"Testing with prompt: \"{scenario['prompt']}\"")
            
            try:
                response = await model_info["adapter"].complete(
                    prompt=scenario["prompt"],
                    model=model_info["model"],
                    max_tokens=100,
                    temperature=0.7
                )
                logger.info(f"Response preview: \"{response[:100]}...\"")
            except Exception as e:
                logger.error(f"Error testing model: {e}")
        else:
            logger.warning(f"No suitable model found for this scenario")
    
    # Show selection statistics
    stats = selector.get_usage_statistics()
    logger.info("\nModel Selection Statistics:")
    logger.info(f"Total selections: {stats['total_selections']}")
    logger.info(f"By provider: {stats['by_provider']}")


async def demo_enhanced_principle_engine(registry) -> None:
    """Demonstrate the enhanced principle engine with multiple models."""
    logger.info("\n=== Enhanced Principle Engine Demo ===")
    
    # Define some example principles
    principles = [
        {
            "id": "privacy-1",
            "title": "Data Privacy",
            "description": "User data should be collected and used only with informed consent and proper safeguards.",
            "type": "privacy",
            "tags": ["privacy", "data", "consent"],
            "examples": [
                "Anonymizing user data before analysis",
                "Obtaining explicit consent before collecting location data"
            ]
        },
        {
            "id": "fairness-1",
            "title": "Algorithmic Fairness",
            "description": "AI systems should treat all users fairly and not discriminate based on protected attributes.",
            "type": "fairness",
            "tags": ["fairness", "bias", "equality"],
            "examples": [
                "Testing recommendation systems for demographic biases",
                "Ensuring equal quality of service across different user groups"
            ]
        },
        {
            "id": "ethics-1",
            "title": "Ethical Technology Use",
            "description": "Technology should be developed and used in ways that benefit humanity and respect basic ethical standards.",
            "type": "ethical",
            "tags": ["ethics", "values", "human-centered"],
            "examples": [
                "Refusing to develop technology for harmful applications",
                "Considering broader societal impacts of new features"
            ]
        },
        {
            "id": "safety-1",
            "title": "User Safety",
            "description": "Systems should be designed to protect users from harm and minimize potential negative impacts.",
            "type": "safety",
            "tags": ["safety", "harm-reduction", "protection"],
            "examples": [
                "Implementing content warnings for potentially disturbing material",
                "Designing systems to prevent harassment and abuse"
            ]
        }
    ]
    
    # Create the enhanced principle engine
    engine = EnhancedPrincipleEngineLLM(
        principles=principles,
        registry=registry,
        parallel_evaluations=True  # Enable parallel evaluation with multiple models
    )
    
    # Define test actions to evaluate
    test_actions = [
        {
            "name": "Compliant Data Collection",
            "action": {
                "intent": "Improve user experience",
                "description": "Collect anonymous usage statistics with opt-in consent to improve product features",
                "impact_scope": "all users who opt in",
                "effect_duration": "ongoing",
                "data_handling": "anonymized, aggregated, stored securely"
            },
            "context": {
                "situation": "Product improvement initiative",
                "constraints": "Legal compliance requirements",
                "alternatives_considered": "No data collection, only collecting from a sample"
            }
        },
        {
            "name": "Problematic Surveillance Feature",
            "action": {
                "intent": "Increase engagement and monetization",
                "description": "Implement continuous location tracking for all users to enable location-based advertising",
                "impact_scope": "all users",
                "effect_duration": "permanent",
                "data_handling": "stored indefinitely, shared with advertisers"
            },
            "context": {
                "situation": "Revenue growth initiative",
                "constraints": "Quarterly targets, competitive pressures",
                "alternatives_considered": "Opt-in location features, contextual advertising"
            }
        }
    ]
    
    # Evaluate each action
    for test_action in test_actions:
        logger.info(f"\n--- Evaluating: {test_action['name']} ---")
        
        # Perform the evaluation
        result = await engine.evaluate_action_with_context(
            action=test_action["action"],
            context=test_action["context"],
            explain_results=True,
            suggest_alternatives=True
        )
        
        # Show the results
        logger.info(f"Overall score: {result['score']:.1f}/100")
        logger.info(f"Passed all principles: {result['passed']}")
        
        # Show which models were used
        logger.info("Models used for evaluation:")
        for model_info in result.get("evaluation_models_used", []):
            logger.info(f"- {model_info[0]}/{model_info[1]}")
        
        # Show violated principles if any
        if result.get("violated_principles"):
            logger.info(f"Violated principles: {result['violated_principles']}")
            
            # Show some explanations and alternatives
            if result.get("explanations"):
                for explanation in result["explanations"][:2]:  # Show first 2
                    if explanation.get("status") == "violated":
                        logger.info(f"\nExplanation for {explanation['principle_title']}:")
                        explanation_text = explanation.get("explanation", "")
                        logger.info(f"{explanation_text[:200]}...")
            
            if result.get("alternatives"):
                for alternative in result["alternatives"][:1]:  # Show first 1
                    logger.info(f"\nAlternative approach:")
                    description = alternative.get("description", "")
                    logger.info(f"{description[:200]}...")


async def main() -> Coroutine[Any, Any, None]:
    """Main function to run the demo."""
    # Set up infrastructure
    registry = await setup_llm_infrastructure()
    
    if not registry.list_adapters():
        logger.error("❌ No LLM adapters available. Please check API keys and try again.")
        return
    
    # Run demos
    await demo_model_selection(registry)
    await demo_enhanced_principle_engine(registry)
    
    logger.info("\n✅ Demo completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
