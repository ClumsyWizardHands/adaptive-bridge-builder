#!/usr/bin/env python3
"""
Test script for Adaptive Bridge Builder HTTP endpoints

This script tests the HTTP endpoints of the Adaptive Bridge Builder agent,
verifying that they are properly set up and functioning.
"""

import sys
import json
import requests
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()

# Server base URL
BASE_URL = "http://localhost:8080"

def print_header(text):
    """Print a formatted header."""
    print(f"\n=== {text} ===")

def test_home_endpoint():
    """Test the home endpoint."""
    print_header("Testing Home Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"{Fore.GREEN}Home endpoint responded with status 200{Style.RESET_ALL}")
            print(f"Agent name: {data.get('name', 'Not found')}")
            print(f"Version: {data.get('version', 'Not found')}")
            print(f"Status: {data.get('status', 'Not found')}")
            print(f"Endpoints: {len(data.get('endpoints', []))} found")
            return True
        else:
            print(f"{Fore.RED}Home endpoint returned status {response.status_code}{Style.RESET_ALL}")
            return False
    except Exception as e:
        print(f"{Fore.RED}Error connecting to server: {e}{Style.RESET_ALL}")
        return False

def test_agent_card_endpoint():
    """Test the agent card endpoint."""
    print_header("Testing Agent Card Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/agent-card")
        if response.status_code == 200:
            data = response.json()
            print(f"{Fore.GREEN}Agent card endpoint responded with status 200{Style.RESET_ALL}")
            print(f"Agent ID: {data.get('agent_id', 'Not found')}")
            print(f"Name: {data.get('name', 'Not found')}")
            
            # Check for capabilities
            capabilities = data.get('capabilities', [])
            if capabilities:
                print(f"Capabilities: {len(capabilities)} found")
                for i, cap in enumerate(capabilities[:3]):  # Show first 3 capabilities
                    print(f"  - {cap.get('name', 'Unknown')}: {cap.get('description', 'No description')}")
                if len(capabilities) > 3:
                    print(f"  - ... and {len(capabilities) - 3} more")
            
            # Check for communication endpoints
            communication = data.get('communication', {})
            endpoints = communication.get('endpoints', [])
            if endpoints:
                print(f"Communication endpoints: {len(endpoints)} found")
                for i, endpoint in enumerate(endpoints):
                    print(f"  - Type: {endpoint.get('type', 'Unknown')}, URL: {endpoint.get('url', 'No URL')}")
            
            return True
        else:
            print(f"{Fore.RED}Agent card endpoint returned status {response.status_code}{Style.RESET_ALL}")
            return False
    except Exception as e:
        print(f"{Fore.RED}Error connecting to server: {e}{Style.RESET_ALL}")
        return False

def test_health_endpoint():
    """Test the health endpoint."""
    print_header("Testing Health Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"{Fore.GREEN}Health endpoint responded with status 200{Style.RESET_ALL}")
            print(f"Status: {data.get('status', 'Not found')}")
            print(f"Agent: {data.get('agent', 'Not found')}")
            return True
        else:
            print(f"{Fore.RED}Health endpoint returned status {response.status_code}{Style.RESET_ALL}")
            return False
    except Exception as e:
        print(f"{Fore.RED}Error connecting to server: {e}{Style.RESET_ALL}")
        return False

def test_process_endpoint():
    """Test the process (JSON-RPC) endpoint."""
    print_header("Testing Process Endpoint")
    try:
        # First try a GET request to check if the endpoint exists
        response = requests.get(f"{BASE_URL}/process")
        if response.status_code == 200:
            print(f"{Fore.GREEN}Process endpoint GET responded with status 200{Style.RESET_ALL}")
            print(f"Response: {response.json().get('message', 'No message')}")
            
            # Now try a POST request with a JSON-RPC payload
            jsonrpc_payload = {
                "jsonrpc": "2.0",
                "id": "test-1",
                "method": "echo",
                "params": {
                    "message": "Hello from the test script"
                }
            }
            
            response = requests.post(
                f"{BASE_URL}/process", 
                json=jsonrpc_payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"{Fore.GREEN}Process endpoint POST responded with status 200{Style.RESET_ALL}")
                print(f"JSON-RPC version: {data.get('jsonrpc', 'Not found')}")
                print(f"ID: {data.get('id', 'Not found')}")
                
                # Check if we got a result or error
                if 'result' in data:
                    print(f"Result: {data['result'].get('message', 'No message')} (success)")
                    return True
                elif 'error' in data:
                    print(f"{Fore.YELLOW}Error: {data['error'].get('message', 'Unknown error')}{Style.RESET_ALL}")
                    return False
                else:
                    print(f"{Fore.YELLOW}Response doesn't contain result or error!{Style.RESET_ALL}")
                    return False
            else:
                print(f"{Fore.RED}Process endpoint POST returned status {response.status_code}{Style.RESET_ALL}")
                return False
        else:
            print(f"{Fore.RED}Process endpoint GET returned status {response.status_code}{Style.RESET_ALL}")
            return False
    except Exception as e:
        print(f"{Fore.RED}Error connecting to server: {e}{Style.RESET_ALL}")
        return False

def main():
    """Main function to run all tests."""
    print("Testing Adaptive Bridge Builder HTTP endpoints")
    print(f"Server URL: {BASE_URL}")
    
    # Run all tests
    home_result = test_home_endpoint()
    agent_card_result = test_agent_card_endpoint()
    health_result = test_health_endpoint()
    process_result = test_process_endpoint()
    
    # Print results summary
    print("\n=== Test Results ===")
    print(f"Home Endpoint: {'[PASS]' if home_result else '[FAIL]'}")
    print(f"Agent Card Endpoint: {'[PASS]' if agent_card_result else '[FAIL]'}")
    print(f"Health Endpoint: {'[PASS]' if health_result else '[FAIL]'}")
    print(f"Process Endpoint: {'[PASS]' if process_result else '[FAIL]'}")
    
    # Determine if all tests passed
    if all([home_result, agent_card_result, health_result, process_result]):
        print(f"\n{Fore.GREEN}All tests passed!{Style.RESET_ALL}")
        return 0
    else:
        print(f"\n{Fore.RED}Some tests failed!{Style.RESET_ALL}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
