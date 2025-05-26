import emoji
"""
Example usage of the LangChain integration module.

This example demonstrates:
1. Creating chains and agents
2. Using EMPIRE components as tools
3. Managing conversation memory
4. Handling async/sync operations
5. Error handling and validation
"""

import asyncio
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import EMPIRE components
from .langchain_integration import (
    LangChainIntegration,
    create_principled_chain,
    create_empire_agent
)
from .principle_engine import PrincipleEngine
from .emoji_translation_engine import EmojiTranslationEngine
from .fairness_evaluator import FairnessEvaluator
from .conflict_resolver import ConflictResolver
from .anthropic_llm_adapter import AnthropicLLMAdapter


# Example 1: Basic Chain Creation
def example_basic_chain() -> None:
    """Demonstrate basic chain creation and usage."""
    print("\n=== Example 1: Basic Chain ===")
    
    # Initialize LLM adapter
    llm_adapter = AnthropicLLMAdapter(
        api_key="your-api-key-here"  # Replace with actual key
    )
    
    # Create integration
    integration = LangChainIntegration(llm_adapter=llm_adapter)
    
    # Create a simple chain
    chain = integration.create_chain(
        prompt_template="Write a helpful response about: {input}"
    )
    
    # Run the chain
    result = integration.run_chain(
        chain, 
        "What are the benefits of using LangChain?"
    )
    
    print(f"Chain Result: {result}")
    
    # Check conversation history
    history = integration.get_conversation_history()
    print(f"Conversation History: {history}")


# Example 2: Principled Chain with Validation
def example_principled_chain() -> None:
    """Demonstrate chain with principle validation."""
    print("\n=== Example 2: Principled Chain ===")
    
    # Initialize components
    llm_adapter = AnthropicLLMAdapter(api_key="your-api-key-here")
    principle_engine = PrincipleEngine()
    
    # Add some principles
    principle_engine.add_principle(
        name="helpfulness",
        description="Always provide helpful and constructive responses",
        weight=1.0
    )
    principle_engine.add_principle(
        name="respect",
        description="Treat all users with respect and dignity",
        weight=0.9
    )
    
    # Create principled chain
    chain = create_principled_chain(
        llm_adapter=llm_adapter,
        principle_engine=principle_engine,
        prompt_template="As a helpful assistant, respond to: {input}"
    )
    
    # Run with validation
    result = chain({"input": "How can I improve my communication skills?"})
    
    print(f"Response: {result['output']}")
    print(f"Validated: {result['validated']}")
    print(f"Principles Alignment: {result['principles_alignment']}")


# Example 3: Creating an Agent with EMPIRE Tools
async def example_agent_with_tools() -> None:
    """Demonstrate agent creation with EMPIRE framework tools."""
    print("\n=== Example 3: Agent with EMPIRE Tools ===")
    
    # Initialize components
    llm_adapter = AnthropicLLMAdapter(api_key="your-api-key-here")
    emoji_engine = EmojiTranslationEngine()
    fairness_evaluator = FairnessEvaluator()
    conflict_resolver = ConflictResolver()
    
    # Create integration
    integration = LangChainIntegration(
        llm_adapter=llm_adapter,
        memory_type="buffer"
    )
    
    # Register EMPIRE components as tools
    integration.register_tool(
        func=lambda text: emoji_engine.translate_to_emoji(text, mode="emotional"),
        name="emoji_translate",
        description="Translate text to emoji representation focusing on emotional content"
    )
    
    integration.register_tool(
        func=lambda action, context="": fairness_evaluator.evaluate(
            {"action": action, "context": context}
        ),
        name="evaluate_fairness",
        description="Evaluate the fairness of an action or decision"
    )
    
    integration.register_tool(
        func=lambda positions: conflict_resolver.resolve(
            positions=[{"position": p} for p in positions.split(",")]
        ),
        name="resolve_conflict",
        description="Resolve conflicts between multiple positions (comma-separated)"
    )
    
    # Create agent
    agent = integration.create_agent(
        system_message="You are an EMPIRE framework assistant with access to specialized tools for emoji translation, fairness evaluation, and conflict resolution."
    )
    
    # Run agent with a complex query
    query = "I'm feeling stressed about a disagreement with my coworker. Can you help me understand the situation better?"
    
    result = await integration.arun_agent(agent, query)
    print(f"Agent Response: {result}")
    
    # Show tool usage
    events = integration.get_callback_events()
    tool_uses = [e for e in events if e["type"] == "agent_action"]
    print(f"\nTools Used: {[t['tool'] for t in tool_uses]}")


# Example 4: Conversation with Memory Management
async def example_conversation_memory() -> None:
    """Demonstrate conversation memory management."""
    print("\n=== Example 4: Conversation Memory ===")
    
    # Initialize
    llm_adapter = AnthropicLLMAdapter(api_key="your-api-key-here")
    
    # Create integration with summary memory
    integration = LangChainIntegration(
        llm_adapter=llm_adapter,
        memory_type="summary",
        memory_kwargs={"max_token_limit": 200}
    )
    
    # Create conversation chain
    chain = integration.create_conversation_chain(
        system_message="You are a helpful assistant tracking our conversation context."
    )
    
    # Have a conversation
    messages = [
        "Hi! My name is Alex and I'm interested in learning about AI ethics.",
        "What are the main principles of AI ethics?",
        "How do these principles apply to real-world applications?",
        "Can you remind me what we've discussed so far?"
    ]
    
    for msg in messages:
        print(f"\nUser: {msg}")
        response = await integration.arun_chain(chain, msg)
        print(f"Assistant: {response}")
    
    # Export session
    integration.export_session("conversation_session.json")
    print("\nSession exported successfully!")


# Example 5: Error Handling and Recovery
async def example_error_handling() -> None:
    """Demonstrate error handling and recovery."""
    print("\n=== Example 5: Error Handling ===")
    
    # Create a mock adapter that fails
    class FailingAdapter(AnthropicLLMAdapter):
        async def complete(self, prompt, **kwargs) -> None:
            if "fail" in prompt.lower():
                raise Exception("Simulated failure!")
            return await super().complete(prompt, **kwargs)
    
    llm_adapter = FailingAdapter(api_key="your-api-key-here")
    integration = LangChainIntegration(llm_adapter=llm_adapter)
    
    # Create chain
    chain = integration.create_chain()
    
    # Try with failing input
    try:
        result = await integration.arun_chain(chain, "This will fail")
    except Exception as e:
        print(f"Caught error: {e}")
        
        # Check error events
        events = integration.get_callback_events()
        errors = [e for e in events if e["type"] == "llm_error"]
        print(f"Error events logged: {len(errors)}")
    
    # Try with successful input
    result = await integration.arun_chain(chain, "This will succeed")
    print(f"Success result: {result}")


# Example 6: Complex Agent Workflow
def example_complex_workflow() -> None:
    """Demonstrate a complex multi-step agent workflow."""
    print("\n=== Example 6: Complex Workflow ===")
    
    # Initialize components
    llm_adapter = AnthropicLLMAdapter(api_key="your-api-key-here")
    
    # Define workflow tools
    empire_tools = [
        {
            "func": lambda x: f"Analyzed: {x}",
            "name": "analyze_input",
            "description": "Analyze user input for key themes"
        },
        {
            "func": lambda x: f"Generated plan for: {x}",
            "name": "generate_plan",
            "description": "Generate an action plan based on analysis"
        },
        {
            "func": lambda x: f"Validated: {x}",
            "name": "validate_plan",
            "description": "Validate the generated plan against principles"
        }
    ]
    
    # Create agent
    agent = create_empire_agent(
        llm_adapter=llm_adapter,
        empire_tools=empire_tools,
        system_message="You are a strategic planning assistant. Analyze requests, generate plans, and validate them."
    )
    
    # Run complex query
    query = "I need to improve team communication in my remote workplace"
    result = agent.run(query)
    
    print(f"Final Result: {result}")


# Example 7: Async/Sync Interoperability
async def example_async_sync() -> None:
    """Demonstrate async/sync interoperability."""
    print("\n=== Example 7: Async/Sync Interoperability ===")
    
    llm_adapter = AnthropicLLMAdapter(api_key="your-api-key-here")
    integration = LangChainIntegration(llm_adapter=llm_adapter)
    
    # Create chain
    chain = integration.create_chain(
        prompt_template="Summarize this text: {input}"
    )
    
    text = "LangChain is a framework for developing applications powered by language models."
    
    # Sync execution
    print("Sync execution:")
    sync_result = integration.run_chain(chain, text)
    print(f"Result: {sync_result}")
    
    # Async execution
    print("\nAsync execution:")
    async_result = await integration.arun_chain(chain, text)
    print(f"Result: {async_result}")
    
    # Compare timing
    import time
    
    # Time sync execution
    start = time.time()
    for _ in range(3):
        integration.run_chain(chain, text)
    sync_time = time.time() - start
    
    # Time async execution
    start = time.time()
    await asyncio.gather(*[
        integration.arun_chain(chain, text) for _ in range(3)
    ])
    async_time = time.time() - start
    
    print(f"\nSync time (3 calls): {sync_time:.2f}s")
    print(f"Async time (3 calls): {async_time:.2f}s")


# Main execution
async def main() -> None:
    """Run all examples."""
    # Note: Replace "your-api-key-here" with actual API key
    
    try:
        # Run basic examples
        example_basic_chain()
        example_principled_chain()
        
        # Run async examples
        await example_agent_with_tools()
        await example_conversation_memory()
        await example_error_handling()
        
        # Run complex workflow
        example_complex_workflow()
        
        # Run async/sync comparison
        await example_async_sync()
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())