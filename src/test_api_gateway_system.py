"""
Unit tests for the ApiGatewaySystem.

This module contains tests that verify the functionality of the ApiGatewaySystem,
including authentication, rate limiting, caching, error handling, and data transformation.
"""
import asyncio
import datetime
import json
import logging
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from api_gateway_system import (
    ApiGatewaySystem,
    ApiConfig,
    EndpointConfig,
    AuthConfig,
    AuthType,
    HttpMethod,
    DataFormat,
    RateLimitConfig,
    CacheConfig,
    ErrorHandlingConfig,
    LogLevel,
    RateLimiter,
    CacheManager,
    CircuitBreaker,
    DataTransformer
)


# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_gateway_test")


class TestApiGatewaySystem(unittest.TestCase):
    """
    Test case for the ApiGatewaySystem class.
    """
    
    def setUp(self):
        """Set up for the tests."""
        # Create the API Gateway System
        self.gateway = ApiGatewaySystem(
            log_file=None,  # No file logging for tests
            log_level=LogLevel.INFO,
            default_timeout_ms=1000,
            default_retry_count=1,
            mask_sensitive_data=True
        )
        
        # Set up a test loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Register a test API
        self.register_test_api()
    
    def tearDown(self):
        """Clean up after the tests."""
        # Close the event loop
        self.loop.close()
    
    def register_test_api(self):
        """Register a test API for use in tests."""
        # Authentication configuration
        auth_config = AuthConfig(
            auth_type=AuthType.API_KEY,
            credentials={"api_key": "test_api_key"},
            header_name="X-API-Key"
        )
        
        # Create API configuration
        test_api = ApiConfig(
            name="test_api",
            base_url="https://api.test.example.com",
            auth=auth_config,
            headers={"Accept": "application/json"},
            global_rate_limit=RateLimitConfig(
                requests_per_minute=10,
                burst_size=3
            ),
            global_cache_config=CacheConfig(
                enabled=True,
                ttl_seconds=60
            ),
            global_error_config=ErrorHandlingConfig(
                retry_count=2,
                circuit_breaker_enabled=True,
                circuit_breaker_threshold=3
            ),
            description="Test API"
        )
        
        # Add test endpoints
        test_api.endpoints["get_data"] = EndpointConfig(
            name="get_data",
            url="/data/{id}",
            method=HttpMethod.GET,
            auth=auth_config,
            description="Get data by ID"
        )
        
        test_api.endpoints["create_data"] = EndpointConfig(
            name="create_data",
            url="/data",
            method=HttpMethod.POST,
            auth=auth_config,
            description="Create new data",
            cache_config=CacheConfig(enabled=False)  # No caching for POST
        )
        
        test_api.endpoints["error_endpoint"] = EndpointConfig(
            name="error_endpoint",
            url="/error",
            method=HttpMethod.GET,
            auth=auth_config,
            description="Always returns an error"
        )
        
        # Register the API
        self.gateway.register_api(test_api)
    
    def test_api_registration(self):
        """Test API registration and listing."""
        # List registered APIs
        apis = self.gateway.list_apis()
        
        # Verify there's at least one API (the test API)
        self.assertTrue(len(apis) > 0)
        
        # Find the test API
        test_api = next((api for api in apis if api["name"] == "test_api"), None)
        self.assertIsNotNone(test_api)
        self.assertEqual(test_api["description"], "Test API")
        
        # Get detailed API info
        api_details = self.gateway.get_api_details("test_api")
        self.assertIsNotNone(api_details)
        self.assertEqual(api_details["name"], "test_api")
        self.assertEqual(api_details["base_url"], "https://api.test.example.com")
        
        # Check endpoints
        self.assertIn("get_data", api_details["endpoints"])
        self.assertIn("create_data", api_details["endpoints"])
    
    def test_unregistration(self):
        """Test API unregistration."""
        # Unregister the test API
        result = self.gateway.unregister_api("test_api")
        self.assertTrue(result)
        
        # Verify it's gone
        apis = self.gateway.list_apis()
        test_api = next((api for api in apis if api["name"] == "test_api"), None)
        self.assertIsNone(test_api)
        
        # Try to unregister a non-existent API
        result = self.gateway.unregister_api("nonexistent_api")
        self.assertFalse(result)
    
    @patch("aiohttp.ClientSession")
    async def test_call_endpoint_success(self, mock_session):
        """Test successful API endpoint call."""
        # Set up mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"id": "123", "name": "Test Data"}
        mock_response.headers = {"Content-Type": "application/json"}
        
        # Set up session mock
        session_instance = MagicMock()
        session_instance.__aenter__.return_value = mock_response
        
        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__.return_value = mock_session
        mock_session.return_value = mock_session_cm
        
        mock_session.get.return_value = session_instance
        
        # Call the endpoint
        response = await self.gateway.call_endpoint(
            api_name="test_api",
            endpoint_name="get_data",
            client_id="test_client",
            parameters={"id": "123"}
        )
        
        # Verify the response
        self.assertEqual(response["status"], 200)
        self.assertEqual(response["data"]["id"], "123")
        self.assertEqual(response["data"]["name"], "Test Data")
    
    @patch("aiohttp.ClientSession")
    async def test_call_endpoint_error(self, mock_session):
        """Test API endpoint call with error."""
        # Set up mock response
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.text.return_value = "Not found"
        mock_response.headers = {"Content-Type": "text/plain"}
        
        # Set up session mock
        session_instance = MagicMock()
        session_instance.__aenter__.return_value = mock_response
        
        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__.return_value = mock_session
        mock_session.return_value = mock_session_cm
        
        mock_session.get.return_value = session_instance
        
        # Call the endpoint
        response = await self.gateway.call_endpoint(
            api_name="test_api",
            endpoint_name="get_data",
            client_id="test_client",
            parameters={"id": "123"}
        )
        
        # Verify the response
        self.assertEqual(response["status"], 404)
        self.assertEqual(response["error"], "Not found")
    
    async def test_rate_limiter(self):
        """Test the rate limiter functionality."""
        # Create a rate limiter with a low limit for testing
        rate_limit_config = RateLimitConfig(
            requests_per_minute=5,
            burst_size=2,
            per_client=True,
            per_endpoint=True
        )
        rate_limiter = RateLimiter(rate_limit_config)
        
        # First request should go through
        allowed, wait_time = await rate_limiter.check_rate_limit("test_client", "test_endpoint")
        self.assertTrue(allowed)
        self.assertIsNone(wait_time)
        
        # Second request should go through (using burst)
        allowed, wait_time = await rate_limiter.check_rate_limit("test_client", "test_endpoint")
        self.assertTrue(allowed)
        self.assertIsNone(wait_time)
        
        # Third request should be limited
        allowed, wait_time = await rate_limiter.check_rate_limit("test_client", "test_endpoint")
        self.assertFalse(allowed)
        self.assertIsNotNone(wait_time)
        self.assertTrue(wait_time > 0)
        
        # Different client should not be affected
        allowed, wait_time = await rate_limiter.check_rate_limit("other_client", "test_endpoint")
        self.assertTrue(allowed)
        
        # Different endpoint should not be affected
        allowed, wait_time = await rate_limiter.check_rate_limit("test_client", "other_endpoint")
        self.assertTrue(allowed)
        
        # Reset the rate limiter
        rate_limiter.reset("test_client", "test_endpoint")
        
        # Now should be allowed again
        allowed, wait_time = await rate_limiter.check_rate_limit("test_client", "test_endpoint")
        self.assertTrue(allowed)
    
    async def test_cache_manager(self):
        """Test the cache manager functionality."""
        # Create a cache manager
        cache_config = CacheConfig(
            enabled=True,
            ttl_seconds=60,
            max_size_mb=1,
            cache_keys=["id", "filter"]
        )
        cache_manager = CacheManager(cache_config)
        
        # Test cache miss
        hit, data, headers = cache_manager.get(
            endpoint="test_endpoint",
            params={"id": "123", "filter": "active"},
            headers={}
        )
        self.assertFalse(hit)
        self.assertIsNone(data)
        self.assertEqual(headers["X-Cache"], "MISS")
        
        # Set cache
        test_data = {"id": "123", "name": "Test"}
        cache_manager.set(
            endpoint="test_endpoint",
            params={"id": "123", "filter": "active"},
            headers={},
            response=test_data
        )
        
        # Test cache hit
        hit, data, headers = cache_manager.get(
            endpoint="test_endpoint",
            params={"id": "123", "filter": "active"},
            headers={}
        )
        self.assertTrue(hit)
        self.assertEqual(data, test_data)
        self.assertEqual(headers["X-Cache"], "HIT")
        
        # Test cache key sensitivity
        hit, data, headers = cache_manager.get(
            endpoint="test_endpoint",
            params={"id": "123", "filter": "inactive"},  # Different filter
            headers={}
        )
        self.assertFalse(hit)
        
        # Test invalidation
        cache_manager.invalidate(endpoint="test_endpoint")
        
        # Should be a miss again
        hit, data, headers = cache_manager.get(
            endpoint="test_endpoint",
            params={"id": "123", "filter": "active"},
            headers={}
        )
        self.assertFalse(hit)
    
    async def test_circuit_breaker(self):
        """Test the circuit breaker functionality."""
        # Create a circuit breaker
        error_config = ErrorHandlingConfig(
            circuit_breaker_enabled=True,
            circuit_breaker_threshold=2,
            circuit_breaker_reset_seconds=1
        )
        circuit_breaker = CircuitBreaker(error_config)
        
        # Define a test function that fails
        async def fail_func():
            raise Exception("Test failure")
        
        # Define a successful function
        async def success_func():
            return "Success"
        
        # First attempt - should raise the exception
        with self.assertRaises(Exception):
            await circuit_breaker.execute(fail_func)
        
        # Second attempt - should also raise and open the circuit
        with self.assertRaises(Exception):
            await circuit_breaker.execute(fail_func)
        
        # Third attempt - circuit is open, should raise circuit open exception
        with self.assertRaises(Exception) as context:
            await circuit_breaker.execute(fail_func)
        self.assertIn("Circuit breaker open", str(context.exception))
        
        # Wait for circuit reset time
        await asyncio.sleep(1.1)
        
        # Circuit should be half-open now, let's try a successful call
        result = await circuit_breaker.execute(success_func)
        self.assertEqual(result, "Success")
        
        # Another successful call should close the circuit
        result = await circuit_breaker.execute(success_func)
        self.assertEqual(result, "Success")
        
        # Verify the circuit is closed
        state = circuit_breaker.get_state()
        self.assertEqual(state["state"], circuit_breaker.CLOSED)
    
    def test_data_transformer(self):
        """Test the data transformer functionality."""
        transformer = DataTransformer()
        
        # Test JSON conversion
        json_data = {"name": "John", "age": 30}
        
        # JSON to string
        json_str = transformer.transform(
            json_data,
            DataFormat.JSON,
            DataFormat.JSON
        )
        self.assertIsInstance(json_str, str)
        self.assertEqual(json.loads(json_str), json_data)
        
        # Field mapping
        mapped_data = transformer.transform(
            json_data,
            DataFormat.JSON,
            DataFormat.JSON,
            transformation_steps=[
                {
                    "type": "field_map",
                    "config": {
                        "mapping": {
                            "fullName": "name",
                            "userAge": "age"
                        }
                    }
                }
            ]
        )
        self.assertEqual(json.loads(mapped_data)["fullName"], "John")
        self.assertEqual(json.loads(mapped_data)["userAge"], 30)
        
        # Adding fields
        added_fields_data = transformer.transform(
            json_data,
            DataFormat.JSON,
            DataFormat.JSON,
            transformation_steps=[
                {
                    "type": "add_fields",
                    "config": {
                        "fields": {
                            "status": "active",
                            "created": "2025-01-01"
                        }
                    }
                }
            ]
        )
        added_data = json.loads(added_fields_data)
        self.assertEqual(added_data["status"], "active")
        self.assertEqual(added_data["created"], "2025-01-01")
        
        # Register a custom transformer
        def custom_transform(data, prefix="", **kwargs):
            if isinstance(data, dict):
                return {f"{prefix}{k}": v for k, v in data.items()}
            return data
        
        transformer.register_transformer("custom_prefix", custom_transform)
        
        # Use the custom transformer
        custom_data = transformer.transform(
            json_data,
            DataFormat.JSON,
            DataFormat.JSON,
            transformation_steps=[
                {
                    "type": "custom_prefix",
                    "config": {
                        "prefix": "user_"
                    }
                }
            ]
        )
        custom_result = json.loads(custom_data)
        self.assertEqual(custom_result["user_name"], "John")
        self.assertEqual(custom_result["user_age"], 30)
    
    @patch("aiohttp.ClientSession")
    async def test_authentication(self, mock_session):
        """Test different authentication methods."""
        # Set up mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"success": True}
        
        # Set up session mock
        session_instance = MagicMock()
        session_instance.__aenter__.return_value = mock_response
        
        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__.return_value = mock_session
        mock_session.return_value = mock_session_cm
        
        mock_session.get.return_value = session_instance
        mock_session.post.return_value = session_instance
        
        # Test API key authentication
        await self.gateway.auth_manager.authenticate(
            AuthConfig(
                auth_type=AuthType.API_KEY,
                credentials={"api_key": "test_key"},
                header_name="X-API-Key"
            )
        )
        
        # Test Bearer token authentication
        await self.gateway.auth_manager.authenticate(
            AuthConfig(
                auth_type=AuthType.BEARER,
                credentials={"access_token": "test_token"}
            )
        )
        
        # Test Basic authentication
        await self.gateway.auth_manager.authenticate(
            AuthConfig(
                auth_type=AuthType.BASIC,
                credentials={"username": "test_user", "password": "test_pass"}
            )
        )


if __name__ == "__main__":
    unittest.main()
