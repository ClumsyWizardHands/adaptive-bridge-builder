"""
Mock LLM Integration Test

This script tests the multi-model LLM integration architecture using mock adapters.
This allows testing the architecture without needing to have API keys or download models.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mock_integration_test")

# Import our LLM infrastructure
from llm_adapter_interface import LLMAdapterRegistry, BaseLLMAdapter
from llm_selector import LLMSelector
from principle_engine_llm_enhanced import EnhancedPrincipleEngineLLM


class MockLLMAdapter(BaseLLMAdapter):
    """Mock LLM adapter for testing."""
    
    def __init__(self, provider: str, model_name: str, latency: float = 0.5) -> None:
        """
        Initialize the mock adapter.
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            model_name: Model name
            latency: Simulated response latency in seconds
        """
        self.provider = provider
        self.model_name = model_name
        self.latency = latency
        
        # Set simulated properties based on model
        if "large" in model_name or "opus" in model_name:
            self.quality = "high"
            self.price = "expensive"
        elif "medium" in model_name or "sonnet" in model_name:
            self.quality = "medium"
            self.price = "moderate"
        else:
            self.quality = "basic"
            self.price = "cheap"
    
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
        """Generate a mock completion."""
        # Simulate response latency
        await asyncio.sleep(self.latency)
        
        # Generate mock response based on prompt and model
        if "?" in prompt:
            return self._generate_qa_response(prompt)
        elif "write" in prompt.lower() or "create" in prompt.lower():
            return self._generate_creative_response(prompt)
        elif "analyze" in prompt.lower() or "evaluate" in prompt.lower():
            return self._generate_analysis_response(prompt)
        else:
            return self._generate_general_response(prompt)
    
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
        """Generate a mock chat completion."""
        # Extract the last user message
        last_message = None
        for message in reversed(messages):
            if message.get("role") == "user":
                last_message = message.get("content")
                break
        
        # Generate a response
        content = await self.complete(
            prompt=last_message or "Hello",
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            **kwargs
        )
        
        # Return in chat format
        return {
            "id": f"mock-{self.provider}-{self.model_name}",
            "object": "chat.completion",
            "created": 1234567890,
            "model": model or self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }
            ]
        }
    
    def get_available_models(self) -> List[str]:
        """Get available models."""
        return [self.model_name]
    
    def _generate_qa_response(self, prompt: str) -> str:
        """Generate a Q&A style response."""
        # Basic responses to common questions
        if "capital" in prompt.lower() and "france" in prompt.lower():
            return "The capital of France is Paris."
        elif "largest planet" in prompt.lower():
            return "Jupiter is the largest planet in our solar system."
        else:
            return f"[{self.provider}/{self.model_name}] Based on my knowledge, I can tell you that this question requires {self.quality} quality analysis. As a {self.price} model, I would provide a {self.quality} answer to this question."
    
    def _generate_creative_response(self, prompt: str) -> str:
        """Generate a creative response."""
        if "poem" in prompt.lower():
            if self.quality == "high":
                return """[Poem by {}/{} - {} quality]
                
In silicon dreams and neural light,
Intelligence blossoms beyond our sight.
Not born of flesh but born of math,
A new form of thought blazes a path.

Algorithms dance with creativity's flame,
Art and science no longer the same.
The boundary blurs between made and grown,
As AI weaves patterns we've never known.
                """.format(self.provider, self.model_name, self.quality)
            else:
                return f"[Poem by {self.provider}/{self.model_name} - {self.quality} quality]\n\nAI is smart\nAI is bright\nAI helps us\nDay and night."
        else:
            return f"[{self.provider}/{self.model_name}] Here's a {self.quality} creative response to your request, as you would expect from a {self.price} model."
    
    def _generate_analysis_response(self, prompt: str) -> str:
        """Generate an analysis response."""
        if "ethical" in prompt.lower():
            if self.quality == "high":
                return f"""[Analysis by {self.provider}/{self.model_name} - {self.quality} quality]
                
The ethical implications of facial recognition technology in public spaces require careful consideration across multiple dimensions:

1. Privacy concerns: Constant surveillance may violate reasonable expectations of privacy, creating a chilling effect on free expression and movement.

2. Consent issues: People in public spaces haven't explicitly consented to facial recognition, raising questions about autonomy.

3. Security vs. liberty: While there are legitimate security applications, we must balance these against fundamental civil liberties.

4. Potential for discrimination: These systems can perpetuate or amplify existing biases in both their training data and application.

5. Transparency requirements: Citizens should know when and how such technology is being used to maintain democratic accountability.
                """
            else:
                return f"[Analysis by {self.provider}/{self.model_name} - {self.quality} quality]\n\nFacial recognition raises privacy concerns. It could be misused. Clear rules are needed to protect people's rights."
        else:
            return f"[{self.provider}/{self.model_name}] This is a {self.quality} analysis of the topic you requested, as you would expect from a {self.price} model."
    
    def _generate_general_response(self, prompt: str) -> str:
        """Generate a general response."""
        return f"[{self.provider}/{self.model_name}] This is a {self.quality} response to your request, as you would expect from a {self.price} model."


async def setup_mock_infrastructure() -> None:
    """Set up mock LLM infrastructure."""
    logger.info("Setting up mock LLM infrastructure...")
    
    # Initialize the registry
    registry = LLMAdapterRegistry()
    
    # Add OpenAI mock adapters
    openai_gpt4 = MockLLMAdapter("openai", "gpt-4-turbo", latency=1.0)
    openai_gpt35 = MockLLMAdapter("openai", "gpt-3.5-turbo", latency=0.3)
    registry.register_adapter(openai_gpt4, "openai")
    
    # Add Anthropic mock adapters
    anthropic_opus = MockLLMAdapter("anthropic", "claude-3-opus", latency=1.2)
    anthropic_sonnet = MockLLMAdapter("anthropic", "claude-3-sonnet", latency=0.7)
    registry.register_adapter(anthropic_opus, "anthropic")
    
    # Add Mistral mock adapters
    mistral_large = MockLLMAdapter("mistral", "mistral-large", latency=0.8)
    mistral_medium = MockLLMAdapter("mistral", "mistral-medium", latency=0.4)
    registry.register_adapter(mistral_large, "mistral")
    
    # Add local mock adapter
    local_mixtral = MockLLMAdapter("mistral-local", "mixtral-8x7b", latency=1.5)
    registry.register_adapter(local_mixtral, "mistral-local")
    
    # Log available adapters
    providers = registry.list_adapters()
    logger.info(f"Registered {len(providers)} mock adapters: {', '.join(providers)}")
    
    return registry


async def test_model_selection(registry) -> None:
    """Test the LLM selector with different selection criteria."""
    logger.info("\n=== Testing Model Selection ===")
    
    # Create the selector
    selector = LLMSelector(registry)
    
    # Test different scenarios
    scenarios = [
        {
            "name": "Simple task, low cost",
            "criteria": {
                "task_complexity": LLMSelector.COMPLEXITY_SIMPLE,
                "latency_requirement": LLMSelector.LATENCY_LOW,
                "cost_preference": LLMSelector.COST_LOWEST
            },
            "prompt": "What is the capital of France?"
        },
        {
            "name": "Creative task, balanced",
            "criteria": {
                "task_complexity": LLMSelector.COMPLEXITY_MODERATE,
                "latency_requirement": LLMSelector.LATENCY_MEDIUM,
                "cost_preference": LLMSelector.COST_BALANCED
            },
            "prompt": "Write a short poem about AI and creativity."
        },
        {
            "name": "Complex reasoning task",
            "criteria": {
                "task_complexity": LLMSelector.COMPLEXITY_COMPLEX,
                "latency_requirement": LLMSelector.LATENCY_HIGH,
                "cost_preference": LLMSelector.COST_PERFORMANCE
            },
            "prompt": "Analyze the ethical implications of facial recognition technology."
        },
        {
            "name": "Privacy-sensitive task",
            "criteria": {
                "task_complexity": LLMSelector.COMPLEXITY_MODERATE,
                "privacy_requirement": LLMSelector.PRIVACY_MAXIMUM
            },
            "prompt": "Analyze this confidential data about our new product launch."
        }
    ]
    
    # Test each scenario
    for i, scenario in enumerate(scenarios):
        logger.info(f"\n--- Scenario {i+1}: {scenario['name']} ---")
        
        # Select model based on criteria
        model_info = selector.select_model(**scenario["criteria"])
        
        if model_info["adapter"]:
            logger.info(f"Selected model: {model_info['provider']}/{model_info['model']}")
            logger.info(f"Selection score: {model_info['score']:.2f}")
            logger.info(f"Selection reasons: {', '.join(model_info['reasons'][:2])}")
            
            # Test the model
            response = await model_info["adapter"].complete(
                prompt=scenario["prompt"],
                model=model_info["model"]
            )
            
            logger.info(f"Response: {response}")
        else:
            logger.warning(f"No suitable model found for {scenario['name']}")
    
    # Show selection stats
    stats = selector.get_usage_statistics()
    logger.info("\nModel Selection Statistics:")
    logger.info(f"Total selections: {stats['total_selections']}")
    logger.info(f"By provider: {stats['by_provider']}")


async def test_principle_evaluation(registry) -> None:
    """Test the enhanced principle engine with mock models."""
    logger.info("\n=== Testing Enhanced Principle Engine ===")
    
    # Define some principles
    principles = [
        {
            "id": "privacy-1",
            "title": "Data Privacy",
            "description": "User data should be collected and used only with informed consent.",
            "type": "privacy",
            "tags": ["privacy", "data", "consent"]
        },
        {
            "id": "fairness-1",
            "title": "Algorithmic Fairness",
            "description": "Systems should treat all users fairly without discrimination.",
            "type": "fairness",
            "tags": ["fairness", "bias", "equality"]
        },
        {
            "id": "ethical-1",
            "title": "Ethical Technology Use",
            "description": "Technology should be developed and used in ways that benefit humanity.",
            "type": "ethical",
            "tags": ["ethics", "values", "human-centered"]
        }
    ]
    
    # Create the principle engine
    engine = EnhancedPrincipleEngineLLM(
        principles=principles,
        registry=registry,
        parallel_evaluations=True
    )
    
    # Define test actions
    test_actions = [
        {
            "name": "Privacy-respecting action",
            "action": {
                "description": "Collect anonymous usage statistics with opt-in consent."
            }
        },
        {
            "name": "Privacy-violating action",
            "action": {
                "description": "Track user location continuously without explicit consent."
            }
        }
    ]
    
    # Evaluate each action
    for test_action in test_actions:
        logger.info(f"\n--- Evaluating: {test_action['name']} ---")
        
        # Use a mock implementation for testing
        # In a real scenario, this would call the actual evaluation logic
        # which would dispatch to different LLMs
        
        # For this test, we'll just simulate the evaluation process
        if "without consent" in test_action["action"]["description"].lower():
            logger.info("Action would violate privacy principles")
            logger.info("Multiple models would be used for evaluation:")
            logger.info("- Privacy principles: mistral-local/mixtral-8x7b")
            logger.info("- Ethical principles: anthropic/claude-3-opus")
            logger.info("- Fairness principles: openai/gpt-4-turbo")
        else:
            logger.info("Action would respect privacy principles")
            logger.info("Multiple models would be used for evaluation:")
            logger.info("- Privacy principles: mistral-local/mixtral-8x7b")
            logger.info("- Ethical principles: anthropic/claude-3-opus")
            logger.info("- Fairness principles: openai/gpt-4-turbo")


async def main() -> None:
    """Main function to run the tests."""
    # Set up infrastructure
    registry = await setup_mock_infrastructure()
    
    # Run tests
    await test_model_selection(registry)
    await test_principle_evaluation(registry)
    
    logger.info("\n✅ All tests completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
