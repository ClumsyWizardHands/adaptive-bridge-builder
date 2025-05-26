"""
Example Client for Integration Assistant API

Demonstrates how to use the Integration Assistant API endpoints
and WebSocket connections.
"""

import asyncio
import json
import requests
import websockets
from datetime import datetime, timezone
from typing import Any, Coroutine


# API base URL
BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"


async def test_websocket_connection() -> None:
    """Test WebSocket connection and real-time updates"""
    print("\n=== Testing WebSocket Connection ===")
    
    async with websockets.connect(WS_URL) as websocket:
        # Receive connection confirmation
        response = await websocket.recv()
        data = json.loads(response)
        print(f"Connected! Client ID: {data['data']['client_id']}")
        
        # Send ping
        await websocket.send(json.dumps({
            "type": "ping"
        }))
        
        # Receive pong
        response = await websocket.recv()
        data = json.loads(response)
        print(f"Received: {data['data']['type']}")
        
        # Associate with an agent
        await websocket.send(json.dumps({
            "type": "associate_agent",
            "agent_id": "agent-test123"
        }))
        
        # Receive confirmation
        response = await websocket.recv()
        data = json.loads(response)
        print(f"Associated with agent: {data['data']['agent_id']}")
        
        await websocket.close()


def test_agent_registration() -> None:
    """Test agent registration"""
    print("\n=== Testing Agent Registration ===")
    
    registration_data = {
        "name": "Example LangChain Agent",
        "framework": "langchain",
        "version": "1.0.0",
        "description": "A test agent using LangChain framework",
        "capabilities": [
            {
                "name": "text_processing",
                "description": "Process and analyze text",
                "parameters": {"max_length": 1000},
                "required": True
            }
        ],
        "endpoint_url": "http://localhost:8001/agent",
        "metadata": {
            "author": "Test User",
            "created": datetime.utcnow().isoformat()
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/api/agents/register",
        json=registration_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Agent registered successfully!")
        print(f"Agent ID: {result['agent_id']}")
        print(f"Connection Token: {result['connection_token']}")
        return result['agent_id']
    else:
        print(f"Registration failed: {response.text}")
        return None


def test_code_generation(agent_id: str) -> None:
    """Test integration code generation"""
    print("\n=== Testing Code Generation ===")
    
    # Generate LangChain code
    request_data = {
        "agent_id": agent_id,
        "framework": "langchain",
        "language": "python",
        "include_examples": True
    }
    
    response = requests.post(
        f"{BASE_URL}/api/integration/code",
        json=request_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Generated code for {result['framework']}:")
        print("-" * 50)
        print(result['code'][:500] + "...")  # Print first 500 chars
        print("-" * 50)
        print(f"Dependencies: {', '.join(result['dependencies'])}")
        print(f"Documentation: {result['documentation_url']}")
    else:
        print(f"Code generation failed: {response.text}")


def test_framework_detection() -> None:
    """Test AI framework detection"""
    print("\n=== Testing Framework Detection ===")
    
    code_snippet = """
    from langchain.tools import Tool
    from langchain.agents import AgentExecutor
    
    tool = Tool(
        name="Calculator",
        func=lambda x: eval(x),
        description="Useful for math"
    )
    """
    
    response = requests.post(
        f"{BASE_URL}/api/detect-framework",
        json={"code_snippet": code_snippet}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Detected Framework: {result['detected_framework']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Suggested Adapter: {result['suggested_adapter']}")
    else:
        print(f"Detection failed: {response.text}")


def test_connection_test(agent_id: str) -> None:
    """Test agent connection testing"""
    print("\n=== Testing Connection Test ===")
    
    request_data = {
        "agent_id": agent_id,
        "test_payload": {"test": True},
        "timeout": 10
    }
    
    response = requests.post(
        f"{BASE_URL}/api/agents/{agent_id}/test",
        json=request_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Connection Test Result:")
        print(f"  Success: {result['success']}")
        print(f"  Response Time: {result['response_time_ms']:.2f}ms")
        if result.get('error_message'):
            print(f"  Error: {result['error_message']}")
    else:
        print(f"Connection test failed: {response.text}")


def test_list_agents() -> None:
    """Test listing agents"""
    print("\n=== Testing List Agents ===")
    
    response = requests.get(f"{BASE_URL}/api/agents")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Found {result['count']} agents:")
        for agent in result['agents']:
            print(f"  - {agent['name']} ({agent['id']}) - {agent['framework']}")
    else:
        print(f"List failed: {response.text}")


async def main() -> Coroutine[Any, Any, None]:
    """Run all tests"""
    print("Integration Assistant API Example Client")
    print("=" * 50)
    
    # Check if server is running
    try:
        health = requests.get(f"{BASE_URL}/health")
        if health.status_code == 200:
            print("✓ Server is healthy")
        else:
            print("✗ Server health check failed")
            return
    except:
        print("✗ Cannot connect to server. Please run the server first.")
        print("  Run: python run_integration_assistant.bat (Windows)")
        print("  Run: ./run_integration_assistant.sh (Linux/Mac)")
        return
    
    # Run tests
    test_framework_detection()
    
    agent_id = test_agent_registration()
    if agent_id:
        test_code_generation(agent_id)
        test_connection_test(agent_id)
    
    test_list_agents()
    
    # Test WebSocket
    await test_websocket_connection()
    
    print("\n✓ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
