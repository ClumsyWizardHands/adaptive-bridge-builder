# LangChain Integration Guide for EMPIRE Framework

## Overview

The LangChain integration module provides seamless integration between the EMPIRE framework and LangChain, enabling you to leverage the power of both systems. This integration supports:

- **Tool Creation**: Convert any EMPIRE component into a LangChain tool
- **Agent Creation**: Build intelligent agents with access to EMPIRE capabilities
- **Memory Management**: Persistent conversation memory with multiple strategies
- **Async/Sync Support**: Full support for both synchronous and asynchronous operations
- **Principle Validation**: Automatic validation of LLM outputs against EMPIRE principles
- **Error Handling**: Comprehensive error handling and logging

## Installation

First, ensure you have LangChain installed:

```bash
pip install langchain langchain-core
```

## Quick Start

### Basic Usage

```python
from src.langchain_integration import LangChainIntegration
from src.anthropic_llm_adapter import AnthropicLLMAdapter

# Initialize LLM adapter
llm_adapter = AnthropicLLMAdapter(api_key="your-api-key")

# Create integration
integration = LangChainIntegration(llm_adapter=llm_adapter)

# Create a simple chain
chain = integration.create_chain(
    prompt_template="Write a helpful response about: {input}"
)

# Run the chain
result = integration.run_chain(chain, "What is EMPIRE framework?")
print(result)
```

### Creating an Agent with Tools

```python
# Register EMPIRE components as tools
integration.register_tool(
    func=emoji_engine.translate_to_emoji,
    name="emoji_translate",
    description="Translate text to emoji representation"
)

integration.register_tool(
    func=fairness_evaluator.evaluate,
    name="evaluate_fairness",
    description="Evaluate the fairness of an action"
)

# Create agent
agent = integration.create_agent(
    system_message="You are an EMPIRE assistant with specialized tools."
)

# Run agent
result = integration.run_agent(agent, "How do I handle a conflict fairly?")
```

## Core Components

### 1. LangChainIntegration Class

The main integration class that provides all functionality:

```python
integration = LangChainIntegration(
    llm_adapter=llm_adapter,           # EMPIRE LLM adapter
    session_manager=session_manager,    # Optional session manager
    principle_engine=principle_engine,  # Optional for validation
    memory_type="buffer",              # "buffer" or "summary"
    memory_kwargs={}                   # Additional memory config
)
```

### 2. EMPIRELangChainWrapper

Wraps EMPIRE LLM adapters to make them compatible with LangChain:

```python
from src.langchain_integration import EMPIRELangChainWrapper

# Any EMPIRE LLM adapter can be wrapped
langchain_llm = EMPIRELangChainWrapper(
    llm_adapter=your_empire_adapter,
    max_tokens=1024,
    temperature=0.7
)
```

### 3. PrincipledOutputParser

Validates LLM outputs against EMPIRE principles:

```python
from src.langchain_integration import PrincipledOutputParser

parser = PrincipledOutputParser(principle_engine=principle_engine)
chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)
```

### 4. EMPIREToolWrapper

Converts EMPIRE components into LangChain tools:

```python
from src.langchain_integration import EMPIREToolWrapper

tool = EMPIREToolWrapper.create_tool(
    func=your_empire_function,
    name="tool_name",
    description="What this tool does",
    return_direct=False
)
```

## Memory Management

The integration supports two types of memory:

### Buffer Memory
Stores complete conversation history:

```python
integration = LangChainIntegration(
    llm_adapter=llm_adapter,
    memory_type="buffer"
)
```

### Summary Memory
Stores summarized conversation history:

```python
integration = LangChainIntegration(
    llm_adapter=llm_adapter,
    memory_type="summary",
    memory_kwargs={"max_token_limit": 200}
)
```

### Memory Operations

```python
# Get conversation history
history = integration.get_conversation_history()

# Clear memory
integration.clear_memory()

# Export session
integration.export_session("session.json")

# Import session
integration.import_session("session.json")
```

## Async/Sync Operations

The integration fully supports both synchronous and asynchronous operations:

### Synchronous

```python
# Sync chain execution
result = integration.run_chain(chain, "input text")

# Sync agent execution
result = integration.run_agent(agent, "input text")
```

### Asynchronous

```python
# Async chain execution
result = await integration.arun_chain(chain, "input text")

# Async agent execution
result = await integration.arun_agent(agent, "input text")
```

## Creating Chains

### Basic Chain

```python
chain = integration.create_chain(
    prompt_template="Template with {input}"
)
```

### Conversation Chain

```python
chain = integration.create_conversation_chain(
    system_message="You are a helpful assistant"
)
```

### Principled Chain

```python
from src.langchain_integration import create_principled_chain

chain = create_principled_chain(
    llm_adapter=llm_adapter,
    principle_engine=principle_engine,
    prompt_template="Your template here"
)
```

## Creating Agents

### Basic Agent

```python
agent = integration.create_agent(
    tools=tools,                    # Optional, uses registered tools
    agent_type="react",            # Agent type
    system_message="System prompt",
    max_iterations=6,
    early_stopping_method="generate"
)
```

### EMPIRE Agent

```python
from src.langchain_integration import create_empire_agent

agent = create_empire_agent(
    llm_adapter=llm_adapter,
    empire_tools=[
        {
            "func": your_function,
            "name": "tool_name",
            "description": "Tool description",
            "return_direct": False
        }
    ],
    system_message="Custom system message"
)
```

## Error Handling

The integration includes comprehensive error handling:

```python
try:
    result = await integration.arun_chain(chain, "input")
except Exception as e:
    # Check logged events
    events = integration.get_callback_events()
    errors = [e for e in events if e["type"] == "llm_error"]
    logger.error(f"Errors: {errors}")
```

## Callback System

Monitor all LangChain operations through the callback system:

```python
# Get all events
events = integration.get_callback_events()

# Filter by event type
llm_starts = [e for e in events if e["type"] == "llm_start"]
agent_actions = [e for e in events if e["type"] == "agent_action"]
```

Event types include:
- `llm_start`: LLM operation started
- `llm_end`: LLM operation completed
- `llm_error`: LLM operation failed
- `chain_start`: Chain execution started
- `chain_end`: Chain execution completed
- `agent_action`: Agent performed an action
- `agent_finish`: Agent completed execution

## Best Practices

### 1. Tool Design

When creating tools from EMPIRE components:

```python
# Good: Clear, single-purpose tool
integration.register_tool(
    func=lambda text: emoji_engine.translate_to_emoji(text, mode="emotional"),
    name="translate_emotional_emoji",
    description="Convert text to emojis that capture emotional content"
)

# Avoid: Overly complex tools
# Instead, break into multiple focused tools
```

### 2. Memory Management

Choose the appropriate memory type:

- Use **buffer memory** for:
  - Short conversations
  - When full context is important
  - Debugging and development

- Use **summary memory** for:
  - Long conversations
  - Token-limited scenarios
  - Production deployments

### 3. Async Operations

Use async operations for better performance:

```python
# Good: Parallel execution
results = await asyncio.gather(
    integration.arun_chain(chain, "input1"),
    integration.arun_chain(chain, "input2"),
    integration.arun_chain(chain, "input3")
)

# Less efficient: Sequential execution
result1 = integration.run_chain(chain, "input1")
result2 = integration.run_chain(chain, "input2")
result3 = integration.run_chain(chain, "input3")
```

### 4. Principle Validation

Always validate important outputs:

```python
# Create chain with automatic validation
chain = integration.create_chain(
    prompt_template="Generate a response for: {input}",
    output_parser=PrincipledOutputParser(principle_engine)
)

# Check validation results
result = chain({"input": "user query"})
if result["validated"] and result["principles_alignment"]["score"] > 0.8:
    # Use the response
    pass
```

### 5. Session Management

Persist important conversations:

```python
# After important conversation
integration.export_session(f"sessions/conversation_{timestamp}.json")

# Resume conversation later
integration.import_session("sessions/conversation_12345.json")
```

## Advanced Examples

### Multi-Tool Agent with Validation

```python
# Initialize components
llm_adapter = AnthropicLLMAdapter(api_key=api_key)
principle_engine = PrincipleEngine()
integration = LangChainIntegration(
    llm_adapter=llm_adapter,
    principle_engine=principle_engine
)

# Register multiple tools
tools = [
    ("analyze_sentiment", sentiment_analyzer.analyze, "Analyze emotional sentiment"),
    ("resolve_conflict", conflict_resolver.resolve, "Find fair resolution"),
    ("evaluate_fairness", fairness_evaluator.evaluate, "Check fairness"),
    ("translate_emoji", emoji_engine.translate, "Convert to emoji")
]

for name, func, desc in tools:
    integration.register_tool(func=func, name=name, description=desc)

# Create specialized agent
agent = integration.create_agent(
    system_message="""You are an EMPIRE framework assistant specializing in 
    fair conflict resolution and emotional intelligence. Use your tools to 
    help users navigate complex situations ethically."""
)

# Use with validation
async def handle_query(query: str):
    result = await integration.arun_agent(agent, query)
    
    # Validate result
    if integration.principle_engine:
        alignment = integration.principle_engine.evaluate_action(
            action=result,
            context={"query": query, "type": "agent_response"}
        )
        
        if alignment["score"] < 0.7:
            # Re-run with additional constraints
            result = await integration.arun_agent(
                agent, 
                f"{query}\n\nPlease ensure your response strongly aligns with fairness principles."
            )
    
    return result
```

### Custom Chain with Multiple Parsers

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Define response structure
response_schemas = [
    ResponseSchema(name="analysis", description="Detailed analysis"),
    ResponseSchema(name="recommendation", description="Recommended action"),
    ResponseSchema(name="confidence", description="Confidence level (0-1)")
]

# Create structured parser
structured_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Combine with principle validation
class CombinedParser(BaseOutputParser):
    def __init__(self, structured_parser, principle_engine):
        self.structured_parser = structured_parser
        self.principle_parser = PrincipledOutputParser(principle_engine)
    
    def parse(self, text: str):
        # Parse structure
        structured = self.structured_parser.parse(text)
        
        # Validate principles
        principled = self.principle_parser.parse(text)
        
        # Combine results
        return {
            **structured,
            "principles_validation": principled["principles_alignment"]
        }

# Use in chain
chain = integration.create_chain(
    prompt_template=prompt_with_format_instructions,
    output_parser=CombinedParser(structured_parser, principle_engine)
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Ensure LangChain is installed
   pip install langchain langchain-core
   ```

2. **Async/Sync Mismatch**
   ```python
   # Use the correct method for your context
   # In async function: await integration.arun_chain()
   # In sync function: integration.run_chain()
   ```

3. **Tool Registration Failures**
   ```python
   # Ensure functions are properly wrapped
   # Async functions are automatically converted
   ```

4. **Memory Overflow**
   ```python
   # Use summary memory for long conversations
   # Set appropriate token limits
   ```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Check callback events
events = integration.get_callback_events()
for event in events:
    print(f"{event['type']}: {event}")
```

## Integration with Other EMPIRE Components

The LangChain integration works seamlessly with all EMPIRE components:

- **Principle Engine**: Automatic validation of outputs
- **Session Manager**: Persistent conversation context
- **Agent Registry**: Dynamic tool discovery
- **Orchestrator Engine**: Complex workflow management
- **Communication Adapters**: Multi-channel support

## Future Enhancements

Planned improvements include:

1. **Advanced Memory Strategies**: Custom memory implementations
2. **Multi-Agent Coordination**: LangChain agents working together
3. **Streaming Support**: Real-time response streaming
4. **Enhanced Validation**: More sophisticated principle checking
5. **Performance Optimization**: Caching and batching improvements

## Conclusion

The LangChain integration brings the best of both worlds together, combining EMPIRE's principled, ethical framework with LangChain's powerful agent and chain capabilities. This integration enables you to build sophisticated, ethically-aligned AI applications with ease.

For more examples and advanced usage, see the `src/langchain_integration_example.py` file.
