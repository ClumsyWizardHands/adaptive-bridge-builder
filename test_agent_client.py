#!/usr/bin/env python3
"""
Test client for the Adaptive Bridge Builder HTTP API
"""

import requests
import json

# Base URL for the agent server
BASE_URL = "http://localhost:8080"

def test_home():
    """Test the home endpoint"""
    response = requests.get(f"{BASE_URL}/")
    print("Home endpoint:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_agent_card():
    """Test getting the agent card"""
    response = requests.get(f"{BASE_URL}/agent-card")
    print("Agent Card:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_echo():
    """Test the echo method"""
    message = {
        "jsonrpc": "2.0",
        "method": "echo",
        "params": {
            "message": "Hello, Adaptive Bridge Builder!",
            "timestamp": "2025-05-28T12:00:00Z"
        },
        "id": "test-echo-1"
    }
    
    response = requests.post(f"{BASE_URL}/process", json=message)
    print("Echo response:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_route():
    """Test message routing"""
    message = {
        "jsonrpc": "2.0",
        "method": "route",
        "params": {
            "destination": "agent-123",
            "message": "Test routing message",
            "conversation_id": "conv-456"
        },
        "id": "test-route-1"
    }
    
    response = requests.post(f"{BASE_URL}/process", json=message)
    print("Route response:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_health():
    """Test the health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health check:")
    print(json.dumps(response.json(), indent=2))
    print()

if __name__ == "__main__":
    print("Testing Adaptive Bridge Builder Agent API")
    print("=" * 50)
    
    try:
        test_home()
        test_agent_card()
        test_echo()
        test_route()
        test_health()
        
        print("All tests completed successfully!")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure it's running on http://localhost:8080")
    except Exception as e:
        print(f"Error: {e}")
