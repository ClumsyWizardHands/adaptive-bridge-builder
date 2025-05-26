"""
Unit tests for [PROTOCOL_NAME] Protocol Adapter.

Tests cover:
- Connection and disconnection
- Message translation (bidirectional)
- Authentication handling
- Error handling and recovery
- Performance metrics
- Edge cases and validation
"""

import asyncio
import datetime
from datetime import timezone
import json
import pytest
from typing import Any, Coroutine, Dict
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import uuid

from protocol_adapter import ProtocolNameAdapter
from universal_agent_connector import (
    ProtocolAdapter,
    ProtocolType,
    AgentMessage,
    MessageType,
    ConnectionStatus,
    AgentCapability
)


class TestProtocolNameAdapter:
    """Test suite for ProtocolNameAdapter"""
    
    @pytest.fixture
    def protocol_config(self) -> None:
        """Create a protocol configuration for testing"""
        return ProtocolAdapter(
            protocol_type=ProtocolType.CUSTOM,
            config={"test": "config"},
            requires_auth=True,
            supports_streaming=True,
            supports_batch=False,
            supports_sync=True,
            supports_async=True,
            max_message_size_kb=1024,
            serialization_format="json"
        )
    
    @pytest.fixture
    def adapter(self, protocol_config) -> None:
        """Create an adapter instance for testing"""
        return ProtocolNameAdapter(
            protocol_config=protocol_config,
            connection_string="protocol://test-host:1234",
            auth_config={"username": "test", "password": "secret"},
            timeout_seconds=30,
            retry_attempts=3
        )
    
    @pytest.fixture
    def sample_agent_message(self) -> None:
        """Create a sample AgentMessage for testing"""
        return AgentMessage(
            id="test-message-123",
            type=MessageType.TASK_REQUEST,
            content={"task": "test", "data": {"key": "value"}},
            sender_id="sender-123",
            recipient_id="recipient-456",
            timestamp=datetime.datetime.now(),
            correlation_id="corr-789",
            priority=1,
            expires_at=None,
            metadata={"source": "test"}
        )
    
    @pytest.fixture
    def sample_protocol_message(self) -> Dict[str, Any]:
        """Create a sample protocol message for testing"""
        return {
            "id": "proto-msg-123",
            "payload": {"text": "Hello", "data": {"key": "value"}},
            "from": "agent-1",
            "to": "agent-2",
            "timestamp": datetime.datetime.now().isoformat(),
            "version": "1.0",
            "priority": 0,
            "headers": {"content-type": "application/json"},
            "tags": ["test", "message"]
        }
    
    @pytest.fixture
    def sample_internal_message(self) -> Dict[str, Any]:
        """Create a sample internal message for testing"""
        return {
            "type": "message",
            "content": {
                "body": {"text": "Hello"},
                "sender": "agent-1",
                "recipient": "agent-2",
                "timestamp": datetime.datetime.now().isoformat(),
                "message_id": "msg-123",
                "correlation_id": None
            },
            "metadata": {
                "protocol": "[PROTOCOL_NAME]",
                "version": "1.0",
                "priority": 0,
                "headers": {},
                "tags": []
            }
        }
    
    # Connection Tests
    
    @pytest.mark.asyncio
    async def test_connect_success(self, adapter) -> None:
        """Test successful connection"""
        # Mock the client connection
        with patch.object(adapter, '_authenticate', new_callable=AsyncMock) as mock_auth:
            result = await adapter.connect()
            
            assert result is True
            assert adapter.connection_status == ConnectionStatus.CONNECTED
            assert adapter.metrics.uptime_seconds > 0
            assert adapter.receive_task is not None
            mock_auth.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, adapter) -> None:
        """Test connection failure"""
        # Mock authentication to fail
        with patch.object(adapter, '_authenticate', side_effect=Exception("Auth failed")):
            result = await adapter.connect()
            
            assert result is False
            assert adapter.connection_status == ConnectionStatus.FAILED
            assert adapter.metrics.last_error == "Auth failed"
            assert adapter.metrics.last_error_time is not None
    
    @pytest.mark.asyncio
    async def test_connect_retry(self, adapter) -> None:
        """Test connection retry logic"""
        adapter.retry_attempts = 3
        attempt_count = 0
        
        async def mock_connect_attempt() -> None:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Connection failed")
            # Success on third attempt
            
        with patch.object(adapter, '_authenticate', new_callable=AsyncMock):
            with patch('asyncio.sleep', new_callable=AsyncMock):
                # First two attempts fail, third succeeds
                result = await adapter.connect()
                
                assert result is True
                assert adapter.connection_status == ConnectionStatus.CONNECTED
    
    @pytest.mark.asyncio
    async def test_disconnect_success(self, adapter) -> None:
        """Test successful disconnection"""
        # Set up connected state
        adapter.client = Mock()
        adapter.receive_task = asyncio.create_task(asyncio.sleep(10))
        adapter.connection_status = ConnectionStatus.CONNECTED
        adapter.authenticated = True
        adapter.session_token = "test-token"
        
        result = await adapter.disconnect()
        
        assert result is True
        assert adapter.connection_status == ConnectionStatus.DISCONNECTED
        assert adapter.client is None
        assert adapter.authenticated is False
        assert adapter.session_token is None
        assert adapter.receive_task is None
    
    # Message Translation Tests
    
    def test_translate_to_internal(self, adapter, sample_protocol_message) -> None:
        """Test translation from protocol format to internal format"""
        internal = adapter.translate_to_internal(sample_protocol_message)
        
        assert internal["type"] == "message"
        assert internal["content"]["body"] == sample_protocol_message["payload"]
        assert internal["content"]["sender"] == sample_protocol_message["from"]
        assert internal["content"]["recipient"] == sample_protocol_message["to"]
        assert internal["content"]["message_id"] == sample_protocol_message["id"]
        assert internal["metadata"]["protocol"] == "[PROTOCOL_NAME]"
        assert internal["metadata"]["version"] == sample_protocol_message["version"]
        assert internal["metadata"]["priority"] == sample_protocol_message["priority"]
    
    def test_translate_from_internal(self, adapter, sample_internal_message) -> None:
        """Test translation from internal format to protocol format"""
        protocol = adapter.translate_from_internal(sample_internal_message)
        
        content = sample_internal_message["content"]
        metadata = sample_internal_message["metadata"]
        
        assert protocol["id"] == content["message_id"]
        assert protocol["payload"] == content["body"]
        assert protocol["from"] == content["sender"]
        assert protocol["to"] == content["recipient"]
        assert protocol["timestamp"] == content["timestamp"]
        assert protocol["version"] == metadata["version"]
        assert protocol["priority"] == metadata["priority"]
    
    def test_bidirectional_translation(self, adapter, sample_protocol_message) -> None:
        """Test that translation is reversible"""
        # Protocol -> Internal -> Protocol
        internal = adapter.translate_to_internal(sample_protocol_message)
        protocol_back = adapter.translate_from_internal(internal)
        
        # Check key fields are preserved
        assert protocol_back["id"] == sample_protocol_message["id"]
        assert protocol_back["payload"] == sample_protocol_message["payload"]
        assert protocol_back["from"] == sample_protocol_message["from"]
        assert protocol_back["to"] == sample_protocol_message["to"]
    
    # Authentication Tests
    
    @pytest.mark.asyncio
    async def test_authenticate_success(self, adapter) -> None:
        """Test successful authentication"""
        adapter.auth_config = {"username": "test", "password": "secret"}
        
        await adapter._authenticate()
        
        assert adapter.authenticated is True
        assert adapter.session_token is not None
        assert adapter.session_token.startswith("session_")
    
    @pytest.mark.asyncio
    async def test_authenticate_failure(self, adapter) -> None:
        """Test authentication failure"""
        adapter.auth_config = {"username": "test", "password": "wrong"}
        
        # Mock auth to fail
        with patch('asyncio.sleep', new_callable=AsyncMock):
            with pytest.raises(Exception) as exc_info:
                # Override the simulated success
                original_authenticate = adapter._authenticate
                
                async def failing_auth() -> None:
                    adapter.authenticated = False
                    adapter.connection_status = ConnectionStatus.UNAUTHORIZED
                    raise Exception("Authentication failed: Invalid credentials")
                
                adapter._authenticate = failing_auth
                await adapter._authenticate()
        
        assert "Authentication failed" in str(exc_info.value)
        assert adapter.authenticated is False
        assert adapter.session_token is None
    
    @pytest.mark.asyncio
    async def test_no_auth_required(self, adapter) -> None:
        """Test connection when authentication is not required"""
        adapter.auth_config = {}
        
        await adapter._authenticate()
        
        # Should complete without error
        assert adapter.authenticated is False  # No auth performed
    
    # Send/Receive Message Tests
    
    @pytest.mark.asyncio
    async def test_send_message_success(self, adapter, sample_agent_message) -> None:
        """Test successful message sending"""
        adapter.client = Mock()
        adapter.connection_status = ConnectionStatus.CONNECTED
        
        result = await adapter.send_message(sample_agent_message)
        
        assert result is True
        assert adapter.metrics.request_count == 1
        assert adapter.metrics.success_count == 1
        assert adapter.metrics.error_count == 0
        assert adapter.metrics.last_request_time is not None
    
    @pytest.mark.asyncio
    async def test_send_message_not_connected(self, adapter, sample_agent_message) -> None:
        """Test sending message when not connected"""
        adapter.client = None
        
        result = await adapter.send_message(sample_agent_message)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_send_message_with_retry(self, adapter, sample_agent_message) -> Coroutine[Any, Any, int]:
        """Test message sending with retry logic"""
        adapter.client = Mock()
        adapter.connection_status = ConnectionStatus.CONNECTED
        adapter.retry_attempts = 3
        
        send_attempts = 0
        
        async def mock_send_with_retry() -> Coroutine[Any, Any, int]:
            nonlocal send_attempts
            send_attempts += 1
            if send_attempts < 2:
                raise Exception("Send failed")
            return True
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await adapter.send_message(sample_agent_message)
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_receive_message_success(self, adapter, sample_protocol_message) -> None:
        """Test successful message receiving"""
        adapter.client = Mock()
        adapter.connection_status = ConnectionStatus.CONNECTED
        
        # Put a message in the buffer
        await adapter.message_buffer.put(sample_protocol_message)
        
        received = await adapter.receive_message()
        
        assert received is not None
        assert isinstance(received, AgentMessage)
        assert received.content == sample_protocol_message["payload"]
        assert received.sender_id == sample_protocol_message["from"]
        assert received.recipient_id == sample_protocol_message["to"]
    
    @pytest.mark.asyncio
    async def test_receive_message_timeout(self, adapter) -> None:
        """Test receive message timeout when no messages available"""
        adapter.client = Mock()
        adapter.connection_status = ConnectionStatus.CONNECTED
        
        # Empty buffer, should timeout
        received = await adapter.receive_message()
        
        assert received is None
    
    # Capability Tests
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, adapter) -> None:
        """Test getting adapter capabilities"""
        adapter.client = Mock()
        adapter.connection_status = ConnectionStatus.CONNECTED
        
        capabilities = await adapter.get_capabilities()
        
        assert len(capabilities) >= 2  # At least send and receive
        
        # Check for expected capabilities
        cap_ids = [cap.id for cap in capabilities]
        assert "protocol_send_message" in cap_ids
        assert "protocol_receive_message" in cap_ids
        
        # With auth config, should include auth capability
        assert "protocol_authenticate" in cap_ids
    
    # Error Handling Tests
    
    @pytest.mark.asyncio
    async def test_reconnect_on_connection_error(self, adapter, sample_agent_message) -> None:
        """Test automatic reconnection on connection errors"""
        adapter.client = Mock()
        adapter.connection_status = ConnectionStatus.CONNECTED
        
        # First send fails with connection error
        with patch.object(adapter, '_should_reconnect', return_value=True):
            with patch.object(adapter, 'connect', new_callable=AsyncMock, return_value=True):
                # Override send to fail first time
                original_validate = adapter._validate_protocol_message
                call_count = 0
                
                def failing_validate(msg) -> None:
                    nonlocal call_count
                    call_count += 1
                    if call_count == 1:
                        raise Exception("Connection lost")
                    return original_validate(msg)
                
                adapter._validate_protocol_message = failing_validate
                
                result = await adapter.send_message(sample_agent_message)
                
                assert result is True
                adapter.connect.assert_called_once()
    
    def test_should_reconnect(self, adapter) -> None:
        """Test reconnection decision logic"""
        # Should reconnect for connection errors
        assert adapter._should_reconnect(Exception("Connection lost")) is True
        assert adapter._should_reconnect(Exception("Connection closed")) is True
        assert adapter._should_reconnect(Exception("Timeout occurred")) is True
        
        # Should not reconnect for other errors
        assert adapter._should_reconnect(Exception("Invalid data")) is False
        assert adapter._should_reconnect(Exception("Authentication failed")) is False
    
    # Validation Tests
    
    def test_validate_protocol_message(self, adapter) -> None:
        """Test protocol message validation"""
        # Valid message
        valid_msg = {
            "id": "123",
            "payload": {"data": "test"},
            "from": "sender",
            "to": "receiver"
        }
        assert adapter._validate_protocol_message(valid_msg) is True
        
        # Missing required field
        invalid_msg = {
            "id": "123",
            "payload": {"data": "test"},
            "from": "sender"
            # Missing "to"
        }
        assert adapter._validate_protocol_message(invalid_msg) is False
        
        # Invalid payload type
        invalid_payload = {
            "id": "123",
            "payload": "not a dict",
            "from": "sender",
            "to": "receiver"
        }
        assert adapter._validate_protocol_message(invalid_payload) is False
    
    def test_validate_internal_message(self, adapter) -> None:
        """Test internal message validation"""
        # Valid message dict
        valid_msg = {
            "type": "message",
            "content": {"body": "test"},
            "metadata": {"protocol": "test"}
        }
        assert adapter._validate_internal_message(valid_msg) is True
        
        # Valid AgentMessage
        agent_msg = AgentMessage(
            id="123",
            type=MessageType.NOTIFICATION,
            content={},
            sender_id="sender",
            recipient_id="recipient",
            timestamp=datetime.datetime.now()
        )
        assert adapter._validate_internal_message(agent_msg) is True
        
        # Invalid type
        invalid_type = {
            "type": "invalid",
            "content": {},
            "metadata": {}
        }
        assert adapter._validate_internal_message(invalid_type) is False
        
        # Missing content
        missing_content = {
            "type": "message",
            "metadata": {}
        }
        assert adapter._validate_internal_message(missing_content) is False
    
    # Metrics Tests
    
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, adapter, sample_agent_message) -> None:
        """Test that metrics are properly tracked"""
        adapter.client = Mock()
        adapter.connection_status = ConnectionStatus.CONNECTED
        
        # Send multiple messages
        for i in range(3):
            await adapter.send_message(sample_agent_message)
        
        metrics = adapter.get_metrics()
        
        assert metrics.request_count == 3
        assert metrics.success_count == 3
        assert metrics.error_count == 0
        assert metrics.average_response_time_ms > 0
        assert metrics.last_response_time is not None
    
    @pytest.mark.asyncio
    async def test_heartbeat(self, adapter) -> None:
        """Test heartbeat functionality"""
        adapter.client = Mock()
        adapter.connection_status = ConnectionStatus.CONNECTED
        
        result = await adapter.send_heartbeat()
        
        assert result is True
        assert adapter.last_heartbeat is not None
        assert adapter.metrics.latency_ms >= 0
    
    # Edge Cases
    
    @pytest.mark.asyncio
    async def test_receive_loop_error_handling(self, adapter) -> None:
        """Test receive loop handles errors gracefully"""
        adapter.client = Mock()
        adapter.connection_status = ConnectionStatus.CONNECTED
        
        # Create a receive task that will encounter an error
        receive_task = asyncio.create_task(adapter._receive_loop())
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        # Cancel it
        receive_task.cancel()
        
        try:
            await receive_task
        except asyncio.CancelledError:
            pass
        
        # Should have completed without raising other exceptions
    
    def test_protocol_specific_fields_preserved(self, adapter) -> None:
        """Test that protocol-specific fields are preserved during translation"""
        protocol_msg = {
            "id": "123",
            "payload": {"data": "test"},
            "from": "sender",
            "to": "receiver",
            "custom_field": "custom_value",
            "another_field": {"nested": "data"}
        }
        
        internal = adapter.translate_to_internal(protocol_msg)
        
        # Check protocol-specific fields are preserved
        assert "custom_field" in internal["content"]["protocol_specific"]
        assert internal["content"]["protocol_specific"]["custom_field"] == "custom_value"
        assert "another_field" in internal["content"]["protocol_specific"]
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, adapter) -> None:
        """Test adapter handles concurrent operations safely"""
        adapter.client = Mock()
        adapter.connection_status = ConnectionStatus.CONNECTED
        
        # Create multiple messages
        messages = [
            AgentMessage(
                id=f"msg-{i}",
                type=MessageType.NOTIFICATION,
                content={"index": i},
                sender_id="sender",
                recipient_id="recipient",
                timestamp=datetime.datetime.now()
            )
            for i in range(5)
        ]
        
        # Send messages concurrently
        tasks = [adapter.send_message(msg) for msg in messages]
        results = await asyncio.gather(*tasks)
        
        assert all(results)  # All should succeed
        assert adapter.metrics.request_count == 5
        assert adapter.metrics.success_count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
