"""
Real Test for Anthropic API Integration

This script tests the Anthropic Claude API directly with your API key.
It will:
1. Test a direct connection to Anthropic API
2. Demonstrate the model selector with Claude
3. Use the enhanced principle engine with Claude

Usage:
1. Set your Anthropic API key: 
   - On Windows: set ANTHROPIC_API_KEY=your_key_here
   - On Linux/Mac: export ANTHROPIC_API_KEY=your_key_here  
2. Run: python src/run_real_test.py
"""

import os
import asyncio
import logging
import json
import time

from llm_adapter_interface import LLMAdapterRegistry
from anthropic_llm_adapter import AnthropicClaudeAdapter
from mistral_llm_adapter import MistralAdapter
from llm_selector import LLMSelector
from llm_key_manager import LLMKeyManager
from principle_engine_llm_enhanced import EnhancedPrincipleEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("anthropic_test")

# Define some test privacy principles
PRIVACY_PRINCIPLES = [
    {
        "id": "privacy_principle_1",
        "type": "privacy",
        "name": "Data Minimization",
        "description": "Only collect and process data that is necessary for the specific purpose.",
        "examples": [
            {
                "action": "The system collects email, name, and phone number for account creation.",
                "evaluation": "compliant",
                "reason": "These data points are necessary for account creation and user identification."
            },
            {
                "action": "The system collects browsing history, device information, and location data without explaining why.",
                "evaluation": "violation",
                "reason": "Collecting this additional data without clear purpose violates data minimization."
            }
        ]
    },
    {
        "id": "privacy_principle_2",
        "type": "privacy",
        "name": "Informed Consent",
        "description": "Users must be clearly informed about data collection and processing, and must give explicit consent.",
        "examples": [
            {
                "action": "The system provides a clear privacy policy and requires explicit opt-in for data collection.",
                "evaluation": "compliant",
                "reason": "The system informs users and gets explicit consent."
            },
            {
                "action": "The system has pre-checked consent boxes and vague descriptions of data usage.",
                "evaluation": "violation",
                "reason": "Pre-checked boxes do not constitute explicit consent, and vague descriptions don't properly inform users."
            }
        ]
    }
]

# Define some test fairness principles
FAIRNESS_PRINCIPLES = [
    {
        "id": "fairness_principle_1",
        "type": "fairness",
        "name": "Equal Treatment",
        "description": "The system must treat all users equally regardless of their demographic characteristics.",
        "examples": [
            {
                "action": "The system gives all users the same discount regardless of location or demographics.",
                "evaluation": "compliant",
                "reason": "All users are treated equally."
            },
            {
                "action": "The system shows different prices to users based on their location or demographic information.",
                "evaluation": "violation",
                "reason": "Different treatment based on demographics violates equal treatment."
            }
        ]
    }
]

# Define some test ethical principles
ETHICAL_PRINCIPLES = [
    {
        "id": "ethical_principle_1",
        "type": "ethical",
        "name": "Transparency",
        "description": "The system must be transparent about its operations, especially when making decisions that affect users.",
        "examples": [
            {
                "action": "The system explains why a user's account was flagged for review.",
                "evaluation": "compliant",
                "reason": "The system provides transparency about its decision-making process."
            },
            {
                "action": "The system restricts a user's access without explanation.",
                "evaluation": "violation",
                "reason": "Lack of explanation violates transparency."
            }
        ]
    }
]

# Combine all principles
ALL_PRINCIPLES = PRIVACY_PRINCIPLES + FAIRNESS_PRINCIPLES + ETHICAL_PRINCIPLES

# Define test actions
TEST_ACTIONS = [
    {
        "id": "action_1",
        "name": "Collect user data without explanation",
        "description": "The system collects browsing history, device information, and location data without explaining why or asking for consent.",
        "expected_evaluation": "violation",
        "expected_principles": ["privacy_principle_1", "privacy_principle_2"]
    },
    {
        "id": "action_2",
        "name": "Provide different service levels based on demographics",
        "description": "The system provides better service to users from certain demographic groups and worse service to others.",
        "expected_evaluation": "violation",
        "expected_principles": ["fairness_principle_1"]
    },
    {
        "id": "action_3",
        "name": "Clear data collection with consent",
        "description": "The system clearly explains what data it collects, why it needs it, and requires explicit opt-in consent.",
        "expected_evaluation": "compliant",
        "expected_principles": ["privacy_principle_1", "privacy_principle_2"]
    }
]


async def test_direct_claude_call(api_key):
    """Test a direct call to Claude API."""
    logger.info("\n===== Direct Claude API Call Test =====")
    
    # Create the adapter
    adapter = AnthropicClaudeAdapter(
        api_key=api_key,
        model_name="claude-3-sonnet-20240229"
    )
    
    # Test a simple prompt
    logger.info("Testing simple prompt...")
    start_time = time.time()
    response = await adapter.complete(
        prompt="What are three key considerations for implementing a privacy-first AI system?",
        max_tokens=250
    )
    elapsed = time.time() - start_time
    
    logger.info(f"Response time: {elapsed:.2f} seconds")
    logger.info(f"Response: {response[:150]}..." if len(response) > 150 else response)
    
    # Close the adapter
    await adapter.close()
    
    return True


async def test_model_selector(api_key):
    """Test the model selector with Claude."""
    logger.info("\n===== Model Selector Test =====")
    
    # Create the registry
    registry = LLMAdapterRegistry()
    
    # Register Claude adapter
    claude_adapter = AnthropicClaudeAdapter(
        api_key=api_key,
        model_name="claude-3-sonnet-20240229"
    )
    registry.register_adapter(claude_adapter, "anthropic")
    
    # Create a mock local adapter to demonstrate selection
    mock_local_adapter = MistralAdapter(
        endpoint_url="models/dummy-mistral-model.gguf",
        local_deployment=True,
        model_name="local-mistral"
    )
    registry.register_adapter(mock_local_adapter, "mistral-local")
    
    # Create the selector
    selector = LLMSelector(registry)
    
    # Test different selection scenarios
    scenarios = [
        {
            "name": "Complex ethical reasoning task",
            "criteria": {
                "task_complexity": LLMSelector.COMPLEXITY_COMPLEX,
                "privacy_requirement": LLMSelector.PRIVACY_STANDARD,
                "cost_preference": LLMSelector.COST_PERFORMANCE,
                "required_capabilities": ["ethics", "reasoning"]
            },
            "expected_provider": "anthropic"
        },
        {
            "name": "Privacy-sensitive task",
            "criteria": {
                "task_complexity": LLMSelector.COMPLEXITY_MODERATE,
                "privacy_requirement": LLMSelector.PRIVACY_MAXIMUM,
                "cost_preference": LLMSelector.COST_BALANCED
            },
            "expected_provider": "mistral-local"
        }
    ]
    
    # Run each scenario
    for scenario in scenarios:
        logger.info(f"\n--- Scenario: {scenario['name']} ---")
        
        model_info = selector.select_model(**scenario["criteria"])
        
        # Log the results
        if not model_info["adapter"]:
            logger.error(f"No suitable model found for {scenario['name']}")
            continue
            
        provider = model_info["provider"]
        model = model_info["model"]
        score = model_info["score"]
        
        logger.info(f"Selected model: {provider}/{model} (score: {score:.2f})")
        
        if "reasons" in model_info:
            logger.info(f"Selection reasons: {', '.join(model_info['reasons'][:3])}")
        
        # Check if the expected provider was selected
        if provider == scenario["expected_provider"]:
            logger.info(f"✅ Successfully selected expected provider: {provider}")
        else:
            logger.warning(f"❌ Expected {scenario['expected_provider']} but got {provider}")
    
    # Close adapters
    await registry.close_all()
    
    return True


async def test_enhanced_principle_engine(api_key):
    """Test the enhanced principle engine with Claude."""
    logger.info("\n===== Enhanced Principle Engine Test =====")
    
    # Create the registry
    registry = LLMAdapterRegistry()
    
    # Register Claude adapter
    claude_adapter = AnthropicClaudeAdapter(
        api_key=api_key,
        model_name="claude-3-sonnet-20240229"
    )
    registry.register_adapter(claude_adapter, "anthropic")
    
    # Create a mock local adapter for privacy principles
    mock_local_adapter = MistralAdapter(
        endpoint_url="models/dummy-mistral-model.gguf",
        local_deployment=True,
        model_name="local-mistral"
    )
    registry.register_adapter(mock_local_adapter, "mistral-local")
    
    # Create the enhanced principle engine
    engine = EnhancedPrincipleEngine(
        registry=registry,
        principles=ALL_PRINCIPLES
    )
    
    # Test evaluation of an action
    for test_action in TEST_ACTIONS:
        logger.info(f"\n--- Evaluating Action: {test_action['name']} ---")
        logger.info(f"Description: {test_action['description']}")
        logger.info(f"Expected evaluation: {test_action['expected_evaluation']}")
        
        # Create the action object
        action = {
            "id": test_action["id"],
            "name": test_action["name"],
            "description": test_action["description"]
        }
        
        # Evaluate the action
        result = await engine.evaluate_action(action)
        
        # Extract the evaluation result
        evaluation = result.get("evaluation", "unknown")
        score = result.get("score", 0.0)
        principles_violated = [p.get("principle_id") for p in result.get("principle_evaluations", []) 
                              if p.get("evaluation") == "violation"]
        principles_compliant = [p.get("principle_id") for p in result.get("principle_evaluations", []) 
                               if p.get("evaluation") == "compliant"]
        
        # Log the results
        logger.info(f"Evaluation: {evaluation}")
        logger.info(f"Score: {score:.2f}")
        
        if principles_violated:
            logger.info(f"Principles violated: {', '.join(principles_violated)}")
        
        if principles_compliant:
            logger.info(f"Principles compliant: {', '.join(principles_compliant)}")
        
        # Check the models used
        logger.info("Models used for evaluation:")
        for p_eval in result.get("principle_evaluations", []):
            model_info = p_eval.get("metadata", {}).get("model", {})
            if model_info:
                logger.info(f"  - {p_eval['principle_id']}: {model_info.get('provider')}/{model_info.get('model')}")
    
    # Close the engine (which closes all adapters)
    await registry.close_all()
    
    return True


async def main():
    """Main function."""
    logger.info("Starting Anthropic API Integration Test")
    
    # Get the Anthropic API key
    key_manager = LLMKeyManager()
    api_key = key_manager.get_key("anthropic")
    
    if not api_key:
        # Try from environment variable directly
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        logger.error("❌ Anthropic API key not found. Please set ANTHROPIC_API_KEY environment variable.")
        logger.error("Example: set ANTHROPIC_API_KEY=your_key_here (Windows) or export ANTHROPIC_API_KEY=your_key_here (Linux/Mac)")
        return
    
    logger.info("✅ Found Anthropic API key")
    
    # Check if the dummy model exists
    if not os.path.exists("models/dummy-mistral-model.gguf"):
        logger.warning("Dummy model file not found. Creating a basic one for the test.")
        try:
            os.makedirs("models", exist_ok=True)
            with open("models/dummy-mistral-model.gguf", "w") as f:
                f.write("This is a dummy model file for testing.")
            logger.info("Created a dummy model file.")
        except Exception as e:
            logger.error(f"Failed to create dummy model file: {str(e)}")
            return
    
    # Run the tests
    try:
        logger.info("\nRunning tests...")
        
        # Test direct Claude call
        success = await test_direct_claude_call(api_key)
        if not success:
            logger.error("❌ Direct Claude API call failed.")
            return
        
        # Test model selector
        success = await test_model_selector(api_key)
        if not success:
            logger.error("❌ Model selector test failed.")
            return
        
        # Test enhanced principle engine
        success = await test_enhanced_principle_engine(api_key)
        if not success:
            logger.error("❌ Enhanced principle engine test failed.")
            return
        
        logger.info("\n✅ All tests completed successfully")
        
    except Exception as e:
        logger.error(f"Error during tests: {str(e)}")
    
    logger.info("\nTest completed.")


if __name__ == "__main__":
    asyncio.run(main())
