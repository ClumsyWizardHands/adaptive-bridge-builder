"""
Example usage of the UniversalAgentConnector for connecting to various agent types.

This example demonstrates how to:
1. Set up the UniversalAgentConnector with necessary components
2. Connect to different agent types using various protocols
3. Send and receive messages
4. Access agent capabilities
5. Handle disconnection and cleanup

"""
import asyncio
import json
import logging
from typing import Dict, List, Optional

# Import the UniversalAgentConnector and its components
from universal_agent_connector import (
    UniversalAgentConnector, 
    AgentFramework,
    ProtocolType,
    MessageType,
    AgentCapability,
    AgentMessage,
    ConnectionStatus
)

# Import the necessary components for setting up the connector
from principle_engine import PrincipleEngine
from agent_registry import AgentRegistry
from a2a_task_handler import A2ATaskHandler
from security_privacy_manager import SecurityPrivacyManager
from session_manager import SessionManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("universal_agent_connector_example")


async def setup_connector() -> UniversalAgentConnector:
    """
    Set up the UniversalAgentConnector with all necessary components.
    
    In a real scenario, these components would be properly initialized
    with appropriate configurations.
    """
    # Initialize required components
    principle_engine = PrincipleEngine()
    agent_registry = AgentRegistry()
    a2a_handler = A2ATaskHandler()
    security_manager = SecurityPrivacyManager()
    session_manager = SessionManager()
    
    # Create the connector
    connector = UniversalAgentConnector(
        agent_registry=agent_registry,
        a2a_handler=a2a_handler,
        security_manager=security_manager,
        principle_engine=principle_engine,
        session_manager=session_manager,
        logger=logger
    )
    
    return connector


async def connect_to_a2a_agent(connector: UniversalAgentConnector, agent_id: str) -> bool:
    """
    Connect to an agent using the A2A protocol.
    
    Args:
        connector: The UniversalAgentConnector instance
        agent_id: The ID of the agent to connect to
        
    Returns:
        bool: True if the connection was successful, False otherwise
    """
    logger.info(f"Connecting to A2A agent: {agent_id}")
    
    # Connect to the agent using A2A protocol
    connection_successful = await connector.connect_to_agent(
        agent_id=agent_id,
        framework=AgentFramework.A2A,
        protocol_type=ProtocolType.A2A,
        connection_config={}  # No additional configuration needed for A2A
    )
    
    if connection_successful:
        logger.info(f"Successfully connected to A2A agent: {agent_id}")
    else:
        logger.error(f"Failed to connect to A2A agent: {agent_id}")
        
    return connection_successful


async def connect_to_rest_agent(
    connector: UniversalAgentConnector, 
    agent_id: str,
    base_url: str,
    api_key: Optional[str] = None
) -> bool:
    """
    Connect to an agent using the REST protocol.
    
    Args:
        connector: The UniversalAgentConnector instance
        agent_id: The ID of the agent to connect to
        base_url: The base URL for the REST API
        api_key: Optional API key for authentication
        
    Returns:
        bool: True if the connection was successful, False otherwise
    """
    logger.info(f"Connecting to REST agent: {agent_id} at {base_url}")
    
    # Set up headers with API key if provided
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Connect to the agent using REST protocol
    connection_successful = await connector.connect_to_agent(
        agent_id=agent_id,
        framework=AgentFramework.LEGACY,
        protocol_type=ProtocolType.REST,
        connection_config={
            "base_url": base_url,
            "headers": headers,
            "timeout_seconds": 30
        }
    )
    
    if connection_successful:
        logger.info(f"Successfully connected to REST agent: {agent_id}")
    else:
        logger.error(f"Failed to connect to REST agent: {agent_id}")
        
    return connection_successful


async def connect_to_websocket_agent(
    connector: UniversalAgentConnector, 
    agent_id: str,
    websocket_url: str,
    api_key: Optional[str] = None
) -> bool:
    """
    Connect to an agent using the WebSocket protocol.
    
    Args:
        connector: The UniversalAgentConnector instance
        agent_id: The ID of the agent to connect to
        websocket_url: The WebSocket URL
        api_key: Optional API key for authentication
        
    Returns:
        bool: True if the connection was successful, False otherwise
    """
    logger.info(f"Connecting to WebSocket agent: {agent_id} at {websocket_url}")
    
    # Set up headers with API key if provided
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Connect to the agent using WebSocket protocol
    connection_successful = await connector.connect_to_agent(
        agent_id=agent_id,
        framework=AgentFramework.GEMINI,  # Example framework
        protocol_type=ProtocolType.WEBSOCKET,
        connection_config={
            "websocket_url": websocket_url,
            "headers": headers,
            "heartbeat_interval_seconds": 30,
            "reconnect_attempts": 3
        }
    )
    
    if connection_successful:
        logger.info(f"Successfully connected to WebSocket agent: {agent_id}")
    else:
        logger.error(f"Failed to connect to WebSocket agent: {agent_id}")
        
    return connection_successful


async def send_task_to_agent(
    connector: UniversalAgentConnector,
    agent_id: str,
    task_content: Dict
) -> bool:
    """
    Send a task to an agent.
    
    Args:
        connector: The UniversalAgentConnector instance
        agent_id: The ID of the agent to send the task to
        task_content: The content of the task
        
    Returns:
        bool: True if the task was sent successfully, False otherwise
    """
    logger.info(f"Sending task to agent: {agent_id}")
    logger.info(f"Task content: {json.dumps(task_content, indent=2)}")
    
    # Send the task
    send_successful = await connector.send_message_to_agent(
        agent_id=agent_id,
        message_type=MessageType.TASK_REQUEST,
        content=task_content,
        sender_id="example_client",
        priority=1
    )
    
    if send_successful:
        logger.info(f"Successfully sent task to agent: {agent_id}")
    else:
        logger.error(f"Failed to send task to agent: {agent_id}")
        
    return send_successful


async def get_agent_capabilities(
    connector: UniversalAgentConnector,
    agent_id: str
) -> List[AgentCapability]:
    """
    Get the capabilities of an agent.
    
    Args:
        connector: The UniversalAgentConnector instance
        agent_id: The ID of the agent to get capabilities for
        
    Returns:
        List[AgentCapability]: The capabilities of the agent
    """
    logger.info(f"Getting capabilities of agent: {agent_id}")
    
    # Get capabilities
    capabilities = await connector.get_agent_capabilities(agent_id)
    
    if capabilities:
        logger.info(f"Retrieved {len(capabilities)} capabilities from agent {agent_id}")
        for capability in capabilities:
            logger.info(f"  - {capability.name}: {capability.description}")
    else:
        logger.warning(f"No capabilities found for agent: {agent_id}")
        
    return capabilities


async def wait_for_response(
    connector: UniversalAgentConnector,
    agent_id: str,
    timeout_seconds: int = 30
) -> Optional[AgentMessage]:
    """
    Wait for a response from an agent with a timeout.
    
    Args:
        connector: The UniversalAgentConnector instance
        agent_id: The ID of the agent to receive a message from
        timeout_seconds: How long to wait for a message
        
    Returns:
        Optional[AgentMessage]: The received message, or None if no message was received
    """
    logger.info(f"Waiting for response from agent: {agent_id}")
    
    # Wait for response with a timeout
    start_time = asyncio.get_event_loop().time()
    while (asyncio.get_event_loop().time() - start_time) < timeout_seconds:
        # Check for a message
        message = await connector.receive_message_from_agent(agent_id)
        
        if message:
            logger.info(f"Received {message.type.value} message from agent: {agent_id}")
            logger.info(f"Message content: {json.dumps(message.content, indent=2)}")
            return message
            
        # Short delay before checking again
        await asyncio.sleep(0.5)
        
    logger.warning(f"Timeout waiting for response from agent: {agent_id}")
    return None


async def example_a2a_workflow() -> None:
    """
    Example workflow using the A2A protocol.
    """
    logger.info("Starting A2A protocol example workflow")
    
    # Set up the connector
    connector = await setup_connector()
    
    try:
        # Connect to an A2A agent
        agent_id = "example_a2a_agent"
        connection_successful = await connect_to_a2a_agent(connector, agent_id)
        
        if not connection_successful:
            logger.error("Failed to connect to A2A agent, aborting workflow")
            return
            
        # Get the agent's capabilities
        capabilities = await get_agent_capabilities(connector, agent_id)
        
        # Check if the agent has the capability we need
        required_capability = "text_processing"
        has_capability = any(cap.id == required_capability for cap in capabilities)
        
        if not has_capability:
            logger.warning(f"Agent does not have required capability: {required_capability}")
            logger.warning("Proceeding with the workflow anyway as an example")
        
        # Send a task to the agent
        task_content = {
            "action": "process_text",
            "parameters": {
                "text": "This is a sample text to process.",
                "options": {
                    "summarize": True,
                    "analyze_sentiment": True
                }
            }
        }
        
        send_successful = await send_task_to_agent(connector, agent_id, task_content)
        
        if not send_successful:
            logger.error("Failed to send task to A2A agent, aborting workflow")
            return
            
        # Wait for a response
        response = await wait_for_response(connector, agent_id, timeout_seconds=30)
        
        if response:
            logger.info("Workflow completed successfully")
            logger.info(f"Response content: {json.dumps(response.content, indent=2)}")
        else:
            logger.warning("No response received, workflow incomplete")
            
    finally:
        # Disconnect from the agent and clean up
        await connector.disconnect_all_agents()
        logger.info("A2A protocol example workflow completed")


async def example_rest_workflow() -> None:
    """
    Example workflow using the REST protocol.
    """
    logger.info("Starting REST protocol example workflow")
    
    # Set up the connector
    connector = await setup_connector()
    
    try:
        # Connect to a REST agent
        agent_id = "example_rest_agent"
        base_url = "https://api.example.com/agents/example_rest_agent"
        api_key = "example_api_key"
        
        connection_successful = await connect_to_rest_agent(
            connector, agent_id, base_url, api_key
        )
        
        if not connection_successful:
            logger.error("Failed to connect to REST agent, aborting workflow")
            return
            
        # Get the agent's capabilities
        capabilities = await get_agent_capabilities(connector, agent_id)
        
        # Send a task to the agent
        task_content = {
            "action": "generate_image",
            "parameters": {
                "prompt": "A futuristic city with flying cars",
                "width": 512,
                "height": 512,
                "num_images": 1
            }
        }
        
        send_successful = await send_task_to_agent(connector, agent_id, task_content)
        
        if not send_successful:
            logger.error("Failed to send task to REST agent, aborting workflow")
            return
            
        # Wait for a response
        response = await wait_for_response(connector, agent_id, timeout_seconds=60)
        
        if response:
            logger.info("Workflow completed successfully")
            logger.info(f"Response content: {json.dumps(response.content, indent=2)}")
        else:
            logger.warning("No response received, workflow incomplete")
            
    finally:
        # Disconnect from the agent and clean up
        await connector.disconnect_all_agents()
        logger.info("REST protocol example workflow completed")


async def example_websocket_workflow() -> None:
    """
    Example workflow using the WebSocket protocol.
    """
    logger.info("Starting WebSocket protocol example workflow")
    
    # Set up the connector
    connector = await setup_connector()
    
    try:
        # Connect to a WebSocket agent
        agent_id = "example_websocket_agent"
        websocket_url = "wss://api.example.com/agents/example_websocket_agent/ws"
        api_key = "example_api_key"
        
        connection_successful = await connect_to_websocket_agent(
            connector, agent_id, websocket_url, api_key
        )
        
        if not connection_successful:
            logger.error("Failed to connect to WebSocket agent, aborting workflow")
            return
            
        # Get the agent's capabilities
        capabilities = await get_agent_capabilities(connector, agent_id)
        
        # Send a task to the agent
        task_content = {
            "action": "chat_completion",
            "parameters": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Tell me about artificial intelligence."}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
        }
        
        send_successful = await send_task_to_agent(connector, agent_id, task_content)
        
        if not send_successful:
            logger.error("Failed to send task to WebSocket agent, aborting workflow")
            return
            
        # Wait for a response
        response = await wait_for_response(connector, agent_id, timeout_seconds=30)
        
        if response:
            logger.info("Workflow completed successfully")
            logger.info(f"Response content: {json.dumps(response.content, indent=2)}")
        else:
            logger.warning("No response received, workflow incomplete")
            
    finally:
        # Disconnect from the agent and clean up
        await connector.disconnect_all_agents()
        logger.info("WebSocket protocol example workflow completed")


async def main() -> None:
    """
    Run all example workflows.
    """
    logger.info("Starting UniversalAgentConnector examples")
    
    # Run the A2A protocol example
    await example_a2a_workflow()
    
    # Run the REST protocol example
    await example_rest_workflow()
    
    # Run the WebSocket protocol example
    await example_websocket_workflow()
    
    logger.info("All examples completed")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
