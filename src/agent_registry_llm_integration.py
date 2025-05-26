"""
Agent Registry LLM Integration

This module demonstrates how to integrate LLM adapters into the AgentRegistry.
It extends the registry to support registering and retrieving LLM adapters,
making them available for other components like the OrchestratorEngine.
"""

import logging
from typing import Dict, List, Optional, Any, Union

from agent_registry import AgentRegistry, CapabilityInfo, CapabilityLevel, TaskType
from orchestrator_engine import AgentRole
from llm_adapter_interface import BaseLLMAdapter, LLMAdapterRegistry
from llm_key_manager import LLMKeyManager
from openai_llm_adapter import OpenAIGPTAdapter
from anthropic_llm_adapter import AnthropicAdapter


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AgentRegistryLLM")


class LLMAgentRegistry(AgentRegistry):
    """
    Extended AgentRegistry with support for LLM adapters.
    
    This class adds methods to register and retrieve LLM adapters,
    integrating them with the existing agent registry.
    """
    
    def __init__(self, *args, **kwargs) -> None:
        """Initialize the LLM-enabled agent registry."""
        super().__init__(*args, **kwargs)
        
        # Add LLM adapter registry
        self.llm_registry = LLMAdapterRegistry()
        
        # Map of adapter IDs to agent IDs
        self.llm_adapter_map: Dict[str, str] = {}
    
    def register_llm_adapter(
        self,
        adapter: BaseLLMAdapter,
        roles: Optional[List[AgentRole]] = None
    ) -> str:
        """
        Register an LLM adapter with the registry.
        
        This creates an agent entry for the LLM adapter, mapping its capabilities
        to the AgentRegistry system and registering it with the LLM adapter registry.
        
        Args:
            adapter: The LLM adapter to register
            roles: Optional list of roles the LLM agent can fulfill
            
        Returns:
            Agent ID for the registered LLM adapter
        """
        if not roles:
            # Default roles for LLMs
            roles = [
                AgentRole.GENERATOR,  # LLMs are good at generating content
                AgentRole.TRANSFORMER,  # LLMs are good at transforming content
                AgentRole.ANALYZER  # LLMs can analyze content
            ]
        
        # Create a unique agent ID for this adapter
        agent_id = f"llm-{adapter.provider_name}-{adapter.model_name}"
        
        # Map capabilities from the adapter to AgentRegistry capabilities
        capabilities = {}
        
        # Add text generation capability
        capabilities["text_generation"] = CapabilityInfo(
            name="text_generation",
            description=f"Generate text using {adapter.provider_name}'s {adapter.model_name}",
            level=CapabilityLevel.EXPERT,
            task_types=[TaskType.GENERATION, TaskType.COMMUNICATION]
        )
        
        # Add text transformation capability
        capabilities["text_transformation"] = CapabilityInfo(
            name="text_transformation",
            description=f"Transform text using {adapter.provider_name}'s {adapter.model_name}",
            level=CapabilityLevel.EXPERT,
            task_types=[TaskType.TRANSFORMATION]
        )
        
        # Add analysis capability
        capabilities["content_analysis"] = CapabilityInfo(
            name="content_analysis",
            description=f"Analyze content using {adapter.provider_name}'s {adapter.model_name}",
            level=CapabilityLevel.PROFICIENT,
            task_types=[TaskType.ANALYSIS, TaskType.RESEARCH]
        )
        
        # Add model-specific capabilities
        model_info = adapter.get_model_info(adapter.model_name)
        for capability in model_info.get("capabilities", []):
            if capability == "image_understanding" or capability == "image_processing":
                capabilities["image_understanding"] = CapabilityInfo(
                    name="image_understanding",
                    description=f"Process and understand images using {adapter.provider_name}'s {adapter.model_name}",
                    level=CapabilityLevel.PROFICIENT,
                    task_types=[TaskType.ANALYSIS, TaskType.EXTRACTION]
                )
            
            if capability == "function_calling":
                capabilities["function_calling"] = CapabilityInfo(
                    name="function_calling",
                    description=f"Execute function calls using {adapter.provider_name}'s {adapter.model_name}",
                    level=CapabilityLevel.PROFICIENT,
                    task_types=[TaskType.EXECUTION]
                )
        
        # Register the LLM as an agent
        super().register_agent(
            agent_id=agent_id,
            roles=roles,
            declared_capabilities=capabilities
        )
        
        # Register with the LLM adapter registry
        self.llm_registry.register_adapter(adapter)
        
        # Map adapter ID to agent ID
        self.llm_adapter_map = {**self.llm_adapter_map, adapter.provider_name: agent_id}
        
        logger.info(f"Registered LLM adapter {adapter.provider_name} as agent {agent_id}")
        return agent_id
    
    def get_llm_adapter(self, agent_id: str) -> Optional[BaseLLMAdapter]:
        """
        Get the LLM adapter for a registered agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            The LLM adapter or None if not found
        """
        # Check if this is an LLM agent
        if not agent_id.startswith("llm-"):
            return None
        
        # Extract provider name from agent ID
        parts = agent_id.split("-")
        if len(parts) < 2:
            return None
            
        provider_name = parts[1]
        
        try:
            # Get the adapter from the LLM registry
            return self.llm_registry.get_adapter(provider_name)
        except KeyError:
            return None
    
    def get_adapter_by_provider(self, provider_name: str) -> Optional[BaseLLMAdapter]:
        """
        Get an LLM adapter by provider name.
        
        Args:
            provider_name: Name of the provider (e.g., "openai", "anthropic")
            
        Returns:
            The LLM adapter or None if not found
        """
        try:
            return self.llm_registry.get_adapter(provider_name)
        except KeyError:
            return None
    
    def list_llm_adapters(self) -> List[str]:
        """
        List all registered LLM adapter provider names.
        
        Returns:
            List of provider names
        """
        return self.llm_registry.list_adapters()


def setup_llm_registry() -> LLMAgentRegistry:
    """
    Set up an LLMAgentRegistry with available LLM adapters.
    
    Returns:
        Configured LLMAgentRegistry
    """
    # Create the key manager
    key_manager = LLMKeyManager()
    
    # Create the registry
    registry = LLMAgentRegistry()
    
    # Try to register OpenAI adapter
    try:
        openai_adapter = OpenAIGPTAdapter(
            key_manager=key_manager,
            model_name="gpt-3.5-turbo"  # Use a less expensive model for examples
        )
        registry.register_llm_adapter(openai_adapter)
        logger.info("Registered OpenAI adapter")
    except Exception as e:
        logger.warning(f"Could not register OpenAI adapter: {e}")
    
    # Try to register Anthropic adapter
    try:
        anthropic_adapter = AnthropicAdapter(
            key_manager=key_manager,
            model_name="claude-3-haiku-20240307"  # Use a less expensive model for examples
        )
        registry.register_llm_adapter(anthropic_adapter)
        logger.info("Registered Anthropic adapter")
    except Exception as e:
        logger.warning(f"Could not register Anthropic adapter: {e}")
    
    return registry


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def run_example() -> None:
        # Set up the registry
        registry = setup_llm_registry()
        
        # List available LLM adapters
        llm_adapters = registry.list_llm_adapters()
        print(f"Registered LLM adapters: {llm_adapters}")
        
        # Check capability-based matching
        agents_with_generation = registry.find_agents_with_capability(
            capability_name="text_generation",
            task_type=TaskType.GENERATION
        )
        print(f"Agents with text generation capability: {agents_with_generation}")
        
        # Test an actual LLM request
        if llm_adapters:
            provider_name = llm_adapters[0]
            adapter = registry.get_adapter_by_provider(provider_name)
            
            if adapter:
                print(f"\nTesting request to {provider_name} adapter...")
                
                try:
                    response = await adapter.send_request(
                        prompt="What are the key benefits of using an adapter pattern for LLM interactions?"
                    )
                    result = adapter.process_response(response)
                    
                    print(f"Response from {provider_name}:")
                    print(result)
                except Exception as e:
                    print(f"Error making request: {e}")
    
    asyncio.run(run_example())
