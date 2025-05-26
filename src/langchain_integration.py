"""
LangChain Integration Module for EMPIRE Framework

This module provides comprehensive integration with LangChain, supporting both
Tool and Agent interfaces, with async/sync operations, conversation memory,
and robust error handling.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Type
from datetime import datetime, timezone
from functools import wraps

# LangChain imports
try:
    from langchain.agents import Tool, AgentExecutor, create_react_agent
    from langchain.agents import AgentOutputParser
    from langchain.chains import LLMChain, ConversationChain
    from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
    from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain.schema.agent import AgentAction, AgentFinish
    from langchain.schema.output_parser import BaseOutputParser
    from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
    from langchain.llms.base import LLM
    from langchain.chat_models.base import BaseChatModel
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.llms import BaseLLM
except ImportError as e:
    raise ImportError(
        "LangChain is not installed. Please install it with: pip install langchain langchain-core"
    ) from e

from .llm_adapter_interface import BaseLLMAdapter
from .principle_engine import PrincipleEngine
from .session_manager import SessionManager

logger = logging.getLogger(__name__)


class EMPIRECallbackHandler(BaseCallbackHandler):
    """Custom callback handler for EMPIRE framework integration."""
    
    def __init__(self, session_id: Optional[str] = None) -> None:
        """Initialize the callback handler."""
        super().__init__()
        self.session_id = session_id
        self.events = []
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Log when LLM starts."""
        event = {
            "type": "llm_start",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "prompts": prompts
        }
        self.events = [*self.events, event]
        logger.debug(f"LLM started with {len(prompts)} prompts")
    
    def on_llm_end(self, response: Any, **kwargs) -> None:
        """Log when LLM ends."""
        event = {
            "type": "llm_end",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "response": str(response)
        }
        self.events = [*self.events, event]
        logger.debug("LLM completed")
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """Log LLM errors."""
        event = {
            "type": "llm_error",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "error": str(error)
        }
        self.events = [*self.events, event]
        logger.error(f"LLM error: {error}")
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Log when chain starts."""
        event = {
            "type": "chain_start",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "inputs": inputs
        }
        self.events = [*self.events, event]
        logger.debug(f"Chain started with inputs: {list(inputs.keys())}")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Log when chain ends."""
        event = {
            "type": "chain_end",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "outputs": outputs
        }
        self.events = [*self.events, event]
        logger.debug("Chain completed")
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Log agent actions."""
        event = {
            "type": "agent_action",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "tool": action.tool,
            "tool_input": action.tool_input,
            "log": action.log
        }
        self.events = [*self.events, event]
        logger.debug(f"Agent action: {action.tool} with input: {action.tool_input}")
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Log when agent finishes."""
        event = {
            "type": "agent_finish",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "output": finish.return_values
        }
        self.events = [*self.events, event]
        logger.debug("Agent finished")


class EMPIRELangChainWrapper(BaseLLM):
    """
    Wrapper to make EMPIRE LLM adapters compatible with LangChain.
    
    This wrapper allows any EMPIRE LLM adapter to be used as a LangChain LLM.
    """
    
    llm_adapter: BaseLLMAdapter
    model_name: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    
    def __init__(self, llm_adapter: BaseLLMAdapter, **kwargs) -> None:
        """Initialize the wrapper with an EMPIRE LLM adapter."""
        super().__init__(**kwargs)
        self.llm_adapter = llm_adapter
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return f"empire_{self.llm_adapter.__class__.__name__}"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Call the LLM adapter synchronously."""
        # Run async method in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.llm_adapter.complete(
                    prompt=prompt,
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop=stop,
                    **kwargs
                )
            )
        finally:
            loop.close()
    
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Call the LLM adapter asynchronously."""
        return await self.llm_adapter.complete(
            prompt=prompt,
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=stop,
            **kwargs
        )


class PrincipledOutputParser(BaseOutputParser):
    """Output parser that validates responses against EMPIRE principles."""
    
    def __init__(self, principle_engine: Optional[PrincipleEngine] = None) -> None:
        """Initialize with optional principle engine."""
        self.principle_engine = principle_engine
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse and validate the output."""
        result = {"output": text, "validated": False, "principles_alignment": None}
        
        if self.principle_engine:
            # Validate against principles
            alignment = self.principle_engine.evaluate_action(
                action=text,
                context={"type": "llm_response"}
            )
            result["validated"] = True
            result["principles_alignment"] = alignment
        
        return result
    
    def get_format_instructions(self) -> str:
        """Return format instructions."""
        return "Provide a response that aligns with the system's ethical principles."


class EMPIREToolWrapper:
    """Wrapper to convert EMPIRE components into LangChain tools."""
    
    @staticmethod
    def create_tool(
        func: Callable,
        name: str,
        description: str,
        return_direct: bool = False,
        args_schema: Optional[Type] = None
    ) -> Tool:
        """
        Create a LangChain tool from a function.
        
        Args:
            func: The function to wrap
            name: Name of the tool
            description: Description of what the tool does
            return_direct: Whether to return the result directly
            args_schema: Optional schema for arguments
            
        Returns:
            LangChain Tool object
        """
        # Handle async functions
        if asyncio.iscoroutinefunction(func):
            def sync_wrapper(*args, **kwargs) -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(func(*args, **kwargs))
                finally:
                    loop.close()
            
            wrapped_func = sync_wrapper
        else:
            wrapped_func = func
        
        return Tool(
            name=name,
            func=wrapped_func,
            description=description,
            return_direct=return_direct,
            args_schema=args_schema
        )


class LangChainIntegration:
    """
    Main integration class for LangChain with EMPIRE framework.
    
    This class provides comprehensive integration including:
    - Tool creation from EMPIRE components
    - Agent creation with memory management
    - Async/sync operation support
    - Error handling and logging
    """
    
    def __init__(
        self,
        llm_adapter: BaseLLMAdapter,
        session_manager: Optional[SessionManager] = None,
        principle_engine: Optional[PrincipleEngine] = None,
        memory_type: str = "buffer",
        memory_kwargs: Optional[Dict] = None
    ):
        """
        Initialize the LangChain integration.
        
        Args:
            llm_adapter: EMPIRE LLM adapter to use
            session_manager: Optional session manager for context
            principle_engine: Optional principle engine for validation
            memory_type: Type of memory to use ("buffer" or "summary")
            memory_kwargs: Additional arguments for memory initialization
        """
        self.llm_adapter = llm_adapter
        self.session_manager = session_manager or SessionManager()
        self.principle_engine = principle_engine
        self.memory_type = memory_type
        self.memory_kwargs = memory_kwargs or {}
        
        # Create LangChain LLM wrapper
        self.llm = EMPIRELangChainWrapper(llm_adapter=llm_adapter)
        
        # Initialize memory
        self.memory = self._create_memory()
        
        # Tool registry
        self.tools: List[Tool] = []
        
        # Callback handler
        self.callback_handler = EMPIRECallbackHandler()
        
        logger.info("LangChain integration initialized")
    
    def _create_memory(self) -> Union[ConversationBufferMemory, ConversationSummaryMemory]:
        """Create the appropriate memory type."""
        if self.memory_type == "summary":
            return ConversationSummaryMemory(
                llm=self.llm,
                **self.memory_kwargs
            )
        else:
            return ConversationBufferMemory(
                **self.memory_kwargs
            )
    
    def register_tool(
        self,
        func: Callable,
        name: str,
        description: str,
        return_direct: bool = False,
        args_schema: Optional[Type] = None
    ) -> Tool:
        """
        Register a function as a LangChain tool.
        
        Args:
            func: Function to register
            name: Tool name
            description: Tool description
            return_direct: Whether to return result directly
            args_schema: Optional argument schema
            
        Returns:
            Created Tool object
        """
        tool = EMPIREToolWrapper.create_tool(
            func=func,
            name=name,
            description=description,
            return_direct=return_direct,
            args_schema=args_schema
        )
        self.tools = [*self.tools, tool]
        logger.info(f"Registered tool: {name}")
        return tool
    
    def create_chain(
        self,
        prompt_template: Optional[str] = None,
        output_parser: Optional[BaseOutputParser] = None
    ) -> LLMChain:
        """
        Create a basic LLM chain.
        
        Args:
            prompt_template: Optional custom prompt template
            output_parser: Optional output parser
            
        Returns:
            LLMChain object
        """
        if prompt_template:
            prompt = PromptTemplate.from_template(prompt_template)
        else:
            prompt = PromptTemplate.from_template("{input}")
        
        if not output_parser and self.principle_engine:
            output_parser = PrincipledOutputParser(self.principle_engine)
        
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            memory=self.memory,
            output_parser=output_parser,
            callbacks=[self.callback_handler]
        )
        
        return chain
    
    def create_conversation_chain(
        self,
        system_message: Optional[str] = None
    ) -> ConversationChain:
        """
        Create a conversation chain with memory.
        
        Args:
            system_message: Optional system message
            
        Returns:
            ConversationChain object
        """
        chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            callbacks=[self.callback_handler]
        )
        
        if system_message:
            chain.prompt.template = f"System: {system_message}\n\n" + chain.prompt.template
        
        return chain
    
    def create_agent(
        self,
        tools: Optional[List[Tool]] = None,
        agent_type: str = "react",
        system_message: Optional[str] = None,
        max_iterations: int = 6,
        early_stopping_method: str = "generate"
    ) -> AgentExecutor:
        """
        Create an agent with tools.
        
        Args:
            tools: List of tools (uses registered tools if not provided)
            agent_type: Type of agent to create
            system_message: Optional system message
            max_iterations: Maximum iterations for agent
            early_stopping_method: Method for early stopping
            
        Returns:
            AgentExecutor object
        """
        tools = tools or self.tools
        
        if not tools:
            raise ValueError("No tools available for agent")
        
        # Create prompt template with system message
        if system_message:
            template = f"""System: {system_message}

You have access to the following tools:

{{tools}}

Use the following format:

Thought: you should always think about what to do
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {{input}}
{{agent_scratchpad}}"""
        else:
            template = """You have access to the following tools:

{{tools}}

Use the following format:

Thought: you should always think about what to do
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {{input}}
{{agent_scratchpad}}"""
        
        prompt = PromptTemplate.from_template(template)
        
        # Create agent
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        # Create executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            callbacks=[self.callback_handler],
            max_iterations=max_iterations,
            early_stopping_method=early_stopping_method,
            handle_parsing_errors=True
        )
        
        return agent_executor
    
    async def arun_chain(self, chain: Union[LLMChain, ConversationChain], input_text: str) -> str:
        """
        Run a chain asynchronously.
        
        Args:
            chain: Chain to run
            input_text: Input text
            
        Returns:
            Output string
        """
        try:
            result = await chain.arun(input_text)
            return result
        except Exception as e:
            logger.error(f"Error running chain: {e}")
            raise
    
    def run_chain(self, chain: Union[LLMChain, ConversationChain], input_text: str) -> str:
        """
        Run a chain synchronously.
        
        Args:
            chain: Chain to run
            input_text: Input text
            
        Returns:
            Output string
        """
        try:
            result = chain.run(input_text)
            return result
        except Exception as e:
            logger.error(f"Error running chain: {e}")
            raise
    
    async def arun_agent(self, agent: AgentExecutor, input_text: str) -> str:
        """
        Run an agent asynchronously.
        
        Args:
            agent: Agent to run
            input_text: Input text
            
        Returns:
            Output string
        """
        try:
            result = await agent.arun(input_text)
            return result
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            raise
    
    def run_agent(self, agent: AgentExecutor, input_text: str) -> str:
        """
        Run an agent synchronously.
        
        Args:
            agent: Agent to run
            input_text: Input text
            
        Returns:
            Output string
        """
        try:
            result = agent.run(input_text)
            return result
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            raise
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of message dictionaries
        """
        if hasattr(self.memory, 'chat_memory'):
            messages = []
            for message in self.memory.chat_memory.messages:
                if isinstance(message, HumanMessage):
                    messages.append({"role": "human", "content": message.content})
                elif isinstance(message, AIMessage):
                    messages.append({"role": "ai", "content": message.content})
                elif isinstance(message, SystemMessage):
                    messages.append({"role": "system", "content": message.content})
            return messages
        return []
    
    def clear_memory(self) -> None:
        """Clear the conversation memory."""
        self.memory = self._create_memory()
        logger.info("Conversation memory cleared")
    
    def get_callback_events(self) -> List[Dict[str, Any]]:
        """
        Get all callback events.
        
        Returns:
            List of event dictionaries
        """
        return self.callback_handler.events
    
    def export_session(self, filepath: str) -> None:
        """
        Export the current session data.
        
        Args:
            filepath: Path to save the session data
        """
        session_data = {
            "conversation_history": self.get_conversation_history(),
            "callback_events": self.get_callback_events(),
            "memory_type": self.memory_type,
            "tools": [{"name": tool.name, "description": tool.description} for tool in self.tools]
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(f"Session exported to {filepath}")
    
    def import_session(self, filepath: str) -> None:
        """
        Import session data.
        
        Args:
            filepath: Path to load the session data from
        """
        with open(filepath, 'r') as f:
            session_data = json.load(f)
        
        # Restore conversation history
        if "conversation_history" in session_data:
            self.clear_memory()
            for message in session_data["conversation_history"]:
                if message["role"] == "human":
                    self.memory.chat_memory.add_user_message(message["content"])
                elif message["role"] == "ai":
                    self.memory.chat_memory.add_ai_message(message["content"])
        
        logger.info(f"Session imported from {filepath}")


# Utility functions for common patterns

def create_principled_chain(
    llm_adapter: BaseLLMAdapter,
    principle_engine: PrincipleEngine,
    prompt_template: str
) -> LLMChain:
    """
    Create a chain that validates outputs against principles.
    
    Args:
        llm_adapter: LLM adapter to use
        principle_engine: Principle engine for validation
        prompt_template: Prompt template string
        
    Returns:
        Configured LLMChain
    """
    integration = LangChainIntegration(
        llm_adapter=llm_adapter,
        principle_engine=principle_engine
    )
    
    return integration.create_chain(
        prompt_template=prompt_template,
        output_parser=PrincipledOutputParser(principle_engine)
    )


def create_empire_agent(
    llm_adapter: BaseLLMAdapter,
    empire_tools: List[Dict[str, Any]],
    system_message: Optional[str] = None
) -> AgentExecutor:
    """
    Create an agent with EMPIRE framework tools.
    
    Args:
        llm_adapter: LLM adapter to use
        empire_tools: List of tool definitions
        system_message: Optional system message
        
    Returns:
        Configured AgentExecutor
    """
    integration = LangChainIntegration(llm_adapter=llm_adapter)
    
    # Register all tools
    for tool_def in empire_tools:
        integration.register_tool(
            func=tool_def["func"],
            name=tool_def["name"],
            description=tool_def["description"],
            return_direct=tool_def.get("return_direct", False)
        )
    
    # Create and return agent
    return integration.create_agent(
        system_message=system_message or "You are an EMPIRE framework agent with access to specialized tools."
    )
