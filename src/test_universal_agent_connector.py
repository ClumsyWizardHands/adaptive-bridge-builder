"""
Unit tests for the UniversalAgentConnector class.

This module contains unit tests that verify the correct operation
of the UniversalAgentConnector, including:
1. Connection to agents using different protocols
2. Sending and receiving messages
3. Retrieving agent capabilities
4. Error handling and recovery
5. Security and authentication
"""
import asyncio
import datetime
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from universal_agent_connector import (
    UniversalAgentConnector,
    AgentFramework,
    ProtocolType,
    MessageType,
    AgentCapability,
    AgentMessage,
    ConnectionStatus,
    A2AProtocolAdapter,
    RestApiAdapter,
    WebSocketAdapter
)


class TestUniversalAgentConnector(unittest.TestCase):
    """
    Test case for the UniversalAgentConnector class.
    """
    
    def setUp(self):
        """
        Set up for the tests.
        
        Creates mock dependencies and initializes the connector.
        """
        # Create mock dependencies
        self.agent_registry = MagicMock()
        self.a2a_handler = MagicMock()
        self.security_manager = MagicMock()
        self.principle_engine = MagicMock()
        self.session_manager = MagicMock()
        self.logger = MagicMock()
        
        # Default to allowing all security checks
        self.security_manager.can_connect_to_agent.return_value = True
        self.security_manager.can_send_message.return_value = True
        self.security_manager.can_receive_message.return_value = True
        self.security_manager.can_use_capability.return_value = True
        
        # Initialize the connector
        self.connector = UniversalAgentConnector(
            agent_registry=self.agent_registry,
            a2a_handler=self.a2a_handler,
            security_manager=self.security_manager,
            principle_engine=self.principle_engine,
            session_manager=self.session_manager,
            logger=self.logger
        )
        
        # Setup event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """
        Clean up after the tests.
        
        Closes the event loop.
        """
        self.loop.close()
    
    def test_initialize_built_in_adapters(self):
        """
        Test that built-in adapters are properly initialized.
        """
        # Should have adapters for A2A, REST, and WebSocket
        self.assertIn(AgentFramework.A2A, self.connector._framework_adapters)
        self.assertIn(AgentFramework.LEGACY, self.connector._framework_adapters)
        self.assertIn(AgentFramework.GEMINI, self.connector._framework_adapters)
        self.assertIn(AgentFramework.CUSTOM, self.connector._framework_adapters)
        
        # Check A2A adapter configuration
        a2a_adapter = self.connector._framework_adapters[AgentFramework.A2A]
        self.assertEqual(a2a_adapter["adapter_class"], A2AProtocolAdapter)
        self.assertEqual(a2a_adapter["protocol_type"], ProtocolType.A2A)
        
        # Check REST adapter configuration
        rest_adapter = self.connector._framework_adapters[AgentFramework.LEGACY]
        self.assertEqual(rest_adapter["adapter_class"], RestApiAdapter)
        self.assertEqual(rest_adapter["protocol_type"], ProtocolType.REST)
        
        # Check WebSocket adapter configuration
        ws_adapter = self.connector._framework_adapters[AgentFramework.GEMINI]
        self.assertEqual(ws_adapter["adapter_class"], WebSocketAdapter)
        self.assertEqual(ws_adapter["protocol_type"], ProtocolType.WEBSOCKET)
    
    async def test_connect_to_agent_a2a(self):
        """
        Test connecting to an agent using the A2A protocol.
        """
        # Mock A2A adapter
        mock_adapter = AsyncMock()
        mock_adapter.connect.return_value = True
        mock_adapter.get_status.return_value = ConnectionStatus.CONNECTED
        
        # Mock adapter class initialization
        with patch('universal_agent_connector.A2AProtocolAdapter', return_value=mock_adapter):
            # Connect to the agent
            agent_id = "test_agent"
            result = await self.connector.connect_to_agent(
                agent_id=agent_id,
                framework=AgentFramework.A2A,
                protocol_type=ProtocolType.A2A,
                connection_config={}
            )
            
            # Verify the result
            self.assertTrue(result)
            self.assertEqual(self.connector._active_connections[agent_id], mock_adapter)
            
            # Verify that the adapter was properly initialized
            self.security_manager.can_connect_to_agent.assert_called_once_with(agent_id)
            mock_adapter.connect.assert_called_once()
    
    async def test_connect_to_agent_rest(self):
        """
        Test connecting to an agent using the REST protocol.
        """
        # Mock REST adapter
        mock_adapter = AsyncMock()
        mock_adapter.connect.return_value = True
        mock_adapter.get_status.return_value = ConnectionStatus.CONNECTED
        
        # Mock adapter class initialization
        with patch('universal_agent_connector.RestApiAdapter', return_value=mock_adapter):
            # Connect to the agent
            agent_id = "test_rest_agent"
            base_url = "https://api.example.com/agents/test_rest_agent"
            result = await self.connector.connect_to_agent(
                agent_id=agent_id,
                framework=AgentFramework.LEGACY,
                protocol_type=ProtocolType.REST,
                connection_config={
                    "base_url": base_url,
                    "headers": {"Authorization": "Bearer api_key"},
                    "timeout_seconds": 30
                }
            )
            
            # Verify the result
            self.assertTrue(result)
            self.assertEqual(self.connector._active_connections[agent_id], mock_adapter)
            
            # Verify that the adapter was properly initialized
            self.security_manager.can_connect_to_agent.assert_called_once_with(agent_id)
            mock_adapter.connect.assert_called_once()
    
    async def test_connect_to_agent_websocket(self):
        """
        Test connecting to an agent using the WebSocket protocol.
        """
        # Mock WebSocket adapter
        mock_adapter = AsyncMock()
        mock_adapter.connect.return_value = True
        mock_adapter.get_status.return_value = ConnectionStatus.CONNECTED
        
        # Mock adapter class initialization
        with patch('universal_agent_connector.WebSocketAdapter', return_value=mock_adapter):
            # Connect to the agent
            agent_id = "test_ws_agent"
            websocket_url = "wss://api.example.com/agents/test_ws_agent/ws"
            result = await self.connector.connect_to_agent(
                agent_id=agent_id,
                framework=AgentFramework.GEMINI,
                protocol_type=ProtocolType.WEBSOCKET,
                connection_config={
                    "websocket_url": websocket_url,
                    "headers": {"Authorization": "Bearer api_key"},
                    "heartbeat_interval_seconds": 30,
                    "reconnect_attempts": 3
                }
            )
            
            # Verify the result
            self.assertTrue(result)
            self.assertEqual(self.connector._active_connections[agent_id], mock_adapter)
            
            # Verify that the adapter was properly initialized
            self.security_manager.can_connect_to_agent.assert_called_once_with(agent_id)
            mock_adapter.connect.assert_called_once()
    
    async def test_disconnect_agent(self):
        """
        Test disconnecting from an agent.
        """
        # Mock adapter
        mock_adapter = AsyncMock()
        mock_adapter.disconnect.return_value = True
        
        # Add the adapter to active connections
        agent_id = "test_agent"
        self.connector._active_connections[agent_id] = mock_adapter
        
        # Disconnect from the agent
        result = await self.connector._disconnect_agent(agent_id)
        
        # Verify the result
        self.assertTrue(result)
        self.assertNotIn(agent_id, self.connector._active_connections)
        mock_adapter.disconnect.assert_called_once()
        
        # If session manager is available, end the session
        if self.session_manager:
            self.session_manager.end_session.assert_called_once_with(
                agent_id=agent_id,
                session_type="agent_connection"
            )
    
    async def test_disconnect_all_agents(self):
        """
        Test disconnecting from all agents.
        """
        # Mock adapters
        mock_adapter1 = AsyncMock()
        mock_adapter1.disconnect.return_value = True
        mock_adapter2 = AsyncMock()
        mock_adapter2.disconnect.return_value = True
        
        # Add the adapters to active connections
        self.connector._active_connections = {
            "agent1": mock_adapter1,
            "agent2": mock_adapter2
        }
        
        # Disconnect from all agents
        result = await self.connector.disconnect_all_agents()
        
        # Verify the result
        self.assertTrue(result)
        self.assertEqual(len(self.connector._active_connections), 0)
        mock_adapter1.disconnect.assert_called_once()
        mock_adapter2.disconnect.assert_called_once()
    
    async def test_send_message_to_agent(self):
        """
        Test sending a message to an agent.
        """
        # Mock adapter
        mock_adapter = AsyncMock()
        mock_adapter.get_status.return_value = ConnectionStatus.CONNECTED
        mock_adapter.send_message.return_value = True
        
        # Add the adapter to active connections
        agent_id = "test_agent"
        self.connector._active_connections[agent_id] = mock_adapter
        
        # Send a message to the agent
        message_type = MessageType.TASK_REQUEST
        content = {"action": "test_action", "parameters": {"param1": "value1"}}
        sender_id = "test_sender"
        result = await self.connector.send_message_to_agent(
            agent_id=agent_id,
            message_type=message_type,
            content=content,
            sender_id=sender_id
        )
        
        # Verify the result
        self.assertTrue(result)
        mock_adapter.send_message.assert_called_once()
        
        # Verify that security check was performed
        self.security_manager.can_send_message.assert_called_once_with(
            sender_id, agent_id, message_type, content
        )
        
        # Verify that the message was properly created
        message = mock_adapter.send_message.call_args[0][0]
        self.assertEqual(message.type, message_type)
        self.assertEqual(message.content, content)
        self.assertEqual(message.sender_id, sender_id)
        self.assertEqual(message.recipient_id, agent_id)
    
    async def test_receive_message_from_agent(self):
        """
        Test receiving a message from an agent.
        """
        # Create a message
        message = AgentMessage(
            id="test_message_id",
            type=MessageType.TASK_RESPONSE,
            content={"result": "test_result"},
            sender_id="test_agent",
            recipient_id="test_recipient",
            timestamp=datetime.datetime.now()
        )
        
        # Mock adapter
        mock_adapter = AsyncMock()
        mock_adapter.get_status.return_value = ConnectionStatus.CONNECTED
        mock_adapter.receive_message.return_value = message
        
        # Add the adapter to active connections
        agent_id = "test_agent"
        self.connector._active_connections[agent_id] = mock_adapter
        
        # Receive a message from the agent
        result = await self.connector.receive_message_from_agent(agent_id)
        
        # Verify the result
        self.assertEqual(result, message)
        mock_adapter.receive_message.assert_called_once()
        
        # Verify that security check was performed
        self.security_manager.can_receive_message.assert_called_once_with(
            message.sender_id, message.recipient_id, message.type, message.content
        )
    
    async def test_get_agent_capabilities(self):
        """
        Test getting the capabilities of an agent.
        """
        # Create capabilities
        capabilities = [
            AgentCapability(
                id="test_capability_1",
                name="Test Capability 1",
                description="Test description 1"
            ),
            AgentCapability(
                id="test_capability_2",
                name="Test Capability 2",
                description="Test description 2"
            )
        ]
        
        # Mock adapter
        mock_adapter = AsyncMock()
        mock_adapter.get_status.return_value = ConnectionStatus.CONNECTED
        mock_adapter.get_capabilities.return_value = capabilities
        
        # Add the adapter to active connections
        agent_id = "test_agent"
        self.connector._active_connections[agent_id] = mock_adapter
        
        # Get capabilities from the agent
        result = await self.connector.get_agent_capabilities(agent_id)
        
        # Verify the result
        self.assertEqual(result, capabilities)
        mock_adapter.get_capabilities.assert_called_once()
        
        # Verify that security checks were performed
        self.security_manager.can_use_capability.assert_called()


if __name__ == "__main__":
    unittest.main()
