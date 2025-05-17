"""
Examples of using ApiGatewaySystem to connect with various external APIs.

This file demonstrates how to:
1. Configure and initialize the API Gateway
2. Register various types of APIs (REST, OAuth, etc.)
3. Handle different authentication methods securely
4. Transform data between formats
5. Implement rate limiting and error handling
6. Use caching to improve performance
7. Track all external system interactions with audit logs

The examples include connecting to:
- GitHub API (REST with OAuth2)
- Twitter/X API (OAuth1.0a)
- Salesforce API (OAuth2 with token refresh)
- Custom Business System (API Key auth)
- Weather API (simple REST with API Key)
"""
import asyncio
import datetime
import json
import logging
import os
from typing import Dict, List, Optional, Any

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
    LogLevel
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_gateway_example")


async def initialize_api_gateway() -> ApiGatewaySystem:
    """Initialize and configure the API Gateway System"""
    gateway = ApiGatewaySystem(
        log_file="api_gateway_audit.log",
        log_level=LogLevel.INFO,
        cache_dir="./cache",
        default_timeout_ms=30000,
        default_retry_count=2,
        mask_sensitive_data=True
    )
    
    logger.info("API Gateway System initialized")
    return gateway


async def register_github_api(gateway: ApiGatewaySystem) -> None:
    """Register GitHub API with OAuth2 authentication"""
    # Authentication configuration for GitHub
    auth_config = AuthConfig(
        auth_type=AuthType.OAUTH2,
        credentials={
            "client_id": os.environ.get("GITHUB_CLIENT_ID", "your_client_id"),
            "client_secret": os.environ.get("GITHUB_CLIENT_SECRET", "your_client_secret"),
            "access_token": os.environ.get("GITHUB_ACCESS_TOKEN", "your_access_token")
        },
        token_url="https://github.com/login/oauth/access_token",
        scope="repo user"
    )
    
    # Rate limiting configuration (GitHub API has strict rate limits)
    rate_limit_config = RateLimitConfig(
        requests_per_minute=30,  # GitHub allows 5000 requests per hour for authenticated requests
        burst_size=5,
        per_endpoint=True,
        per_client=True
    )
    
    # Caching configuration
    cache_config = CacheConfig(
        enabled=True,
        ttl_seconds=300,  # 5 minutes cache for GitHub responses
        max_size_mb=50,
        cache_keys=["page", "per_page", "sort"],
        ignore_params=["access_token"]
    )
    
    # Error handling configuration
    error_config = ErrorHandlingConfig(
        retry_count=3,
        retry_delay_ms=1000,
        timeout_ms=10000,
        circuit_breaker_enabled=True,
        circuit_breaker_threshold=5,
        fallback_response={"message": "GitHub API currently unavailable"}
    )
    
    # Create API configuration
    github_api = ApiConfig(
        name="github",
        base_url="https://api.github.com",
        auth=auth_config,
        headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "ApiGateway"
        },
        global_rate_limit=rate_limit_config,
        global_cache_config=cache_config,
        global_error_config=error_config,
        description="GitHub REST API v3",
        tags=["development", "source-control"],
        version="v3"
    )
    
    # Add repository endpoints
    github_api.endpoints["list_repos"] = EndpointConfig(
        name="list_repos",
        url="/user/repos",
        method=HttpMethod.GET,
        auth=auth_config,
        description="List repositories for the authenticated user"
    )
    
    github_api.endpoints["get_repo"] = EndpointConfig(
        name="get_repo",
        url="/repos/{owner}/{repo}",
        method=HttpMethod.GET,
        auth=auth_config,
        description="Get a repository"
    )
    
    github_api.endpoints["list_issues"] = EndpointConfig(
        name="list_issues",
        url="/repos/{owner}/{repo}/issues",
        method=HttpMethod.GET,
        auth=auth_config,
        description="List issues for a repository",
        # Special cache config for issues (shorter TTL)
        cache_config=CacheConfig(
            enabled=True,
            ttl_seconds=60,  # Issues change more frequently
            cache_keys=["state", "page"]
        )
    )
    
    github_api.endpoints["create_issue"] = EndpointConfig(
        name="create_issue",
        url="/repos/{owner}/{repo}/issues",
        method=HttpMethod.POST,
        auth=auth_config,
        description="Create an issue",
        # No caching for POST requests
        cache_config=CacheConfig(enabled=False)
    )
    
    # Register the API with the gateway
    gateway.register_api(github_api)
    logger.info("GitHub API registered")


async def register_twitter_api(gateway: ApiGatewaySystem) -> None:
    """Register Twitter/X API with OAuth1.0a authentication"""
    # Authentication configuration for Twitter
    auth_config = AuthConfig(
        auth_type=AuthType.OAUTH1,
        credentials={
            "consumer_key": os.environ.get("TWITTER_CONSUMER_KEY", "your_consumer_key"),
            "consumer_secret": os.environ.get("TWITTER_CONSUMER_SECRET", "your_consumer_secret"),
            "access_token": os.environ.get("TWITTER_ACCESS_TOKEN", "your_access_token"),
            "access_token_secret": os.environ.get("TWITTER_ACCESS_TOKEN_SECRET", "your_access_token_secret")
        }
    )
    
    # Rate limiting configuration (Twitter API has strict rate limits)
    rate_limit_config = RateLimitConfig(
        requests_per_minute=15,  # Twitter often limits to 15 requests per 15 minutes
        burst_size=3,
        per_endpoint=True
    )
    
    # Error handling configuration
    error_config = ErrorHandlingConfig(
        retry_count=3,
        retry_delay_ms=5000,  # Longer delay for Twitter API
        circuit_breaker_enabled=True
    )
    
    # Create API configuration
    twitter_api = ApiConfig(
        name="twitter",
        base_url="https://api.twitter.com/2",  # Using Twitter API v2
        auth=auth_config,
        global_rate_limit=rate_limit_config,
        global_error_config=error_config,
        description="Twitter/X API v2",
        tags=["social-media"]
    )
    
    # Add Tweet endpoints
    twitter_api.endpoints["get_tweet"] = EndpointConfig(
        name="get_tweet",
        url="/tweets/{id}",
        method=HttpMethod.GET,
        auth=auth_config,
        description="Get a tweet by ID",
        cache_config=CacheConfig(
            enabled=True,
            ttl_seconds=3600  # Tweets rarely change after posting
        )
    )
    
    twitter_api.endpoints["create_tweet"] = EndpointConfig(
        name="create_tweet",
        url="/tweets",
        method=HttpMethod.POST,
        auth=auth_config,
        description="Create a new tweet",
        cache_config=CacheConfig(enabled=False)  # No caching for POST requests
    )
    
    twitter_api.endpoints["search_tweets"] = EndpointConfig(
        name="search_tweets",
        url="/tweets/search/recent",
        method=HttpMethod.GET,
        auth=auth_config,
        description="Search for tweets",
        query_params={"tweet.fields": "created_at,author_id,public_metrics"},
        cache_config=CacheConfig(
            enabled=True,
            ttl_seconds=60,  # Search results change frequently
            cache_keys=["query", "max_results"]
        )
    )
    
    # Register the API with the gateway
    gateway.register_api(twitter_api)
    logger.info("Twitter API registered")


async def register_salesforce_api(gateway: ApiGatewaySystem) -> None:
    """Register Salesforce API with OAuth2 authentication and token refresh"""
    # Authentication configuration for Salesforce
    auth_config = AuthConfig(
        auth_type=AuthType.OAUTH2,
        credentials={
            "client_id": os.environ.get("SALESFORCE_CLIENT_ID", "your_client_id"),
            "client_secret": os.environ.get("SALESFORCE_CLIENT_SECRET", "your_client_secret"),
            "access_token": os.environ.get("SALESFORCE_ACCESS_TOKEN", "your_access_token")
        },
        token_url="https://login.salesforce.com/services/oauth2/token",
        refresh_token=os.environ.get("SALESFORCE_REFRESH_TOKEN", "your_refresh_token"),
        # Token typically expires in 2 hours for Salesforce
        token_expiry=datetime.datetime.now() + datetime.timedelta(hours=2)
    )
    
    # Create API configuration
    salesforce_api = ApiConfig(
        name="salesforce",
        base_url="https://yourinstance.salesforce.com/services/data/v56.0",
        auth=auth_config,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json"
        },
        global_rate_limit=RateLimitConfig(
            requests_per_minute=100,
            burst_size=10
        ),
        global_cache_config=CacheConfig(
            enabled=True,
            ttl_seconds=300,  # 5 minutes cache
            respect_cache_control=True
        ),
        description="Salesforce REST API",
        tags=["crm", "business"],
        version="v56.0",
        contact_info={"email": "admin@example.com"}
    )
    
    # Add Salesforce object endpoints
    salesforce_api.endpoints["query"] = EndpointConfig(
        name="query",
        url="/query",
        method=HttpMethod.GET,
        auth=auth_config,
        description="Execute a SOQL query",
        query_params={"q": "{soql_query}"}
    )
    
    salesforce_api.endpoints["get_account"] = EndpointConfig(
        name="get_account",
        url="/sobjects/Account/{id}",
        method=HttpMethod.GET,
        auth=auth_config,
        description="Get account details"
    )
    
    salesforce_api.endpoints["create_account"] = EndpointConfig(
        name="create_account",
        url="/sobjects/Account",
        method=HttpMethod.POST,
        auth=auth_config,
        description="Create a new account",
        cache_config=CacheConfig(enabled=False)  # No caching for POST
    )
    
    salesforce_api.endpoints["update_account"] = EndpointConfig(
        name="update_account",
        url="/sobjects/Account/{id}",
        method=HttpMethod.PATCH,
        auth=auth_config,
        description="Update an account",
        cache_config=CacheConfig(enabled=False)  # No caching for PATCH
    )
    
    salesforce_api.endpoints["create_contact"] = EndpointConfig(
        name="create_contact",
        url="/sobjects/Contact",
        method=HttpMethod.POST,
        auth=auth_config,
        description="Create a new contact",
        cache_config=CacheConfig(enabled=False)  # No caching for POST
    )
    
    # Register the API with the gateway
    gateway.register_api(salesforce_api)
    logger.info("Salesforce API registered")


async def register_business_system_api(gateway: ApiGatewaySystem) -> None:
    """Register a custom internal business system API"""
    # Authentication using API key
    auth_config = AuthConfig(
        auth_type=AuthType.API_KEY,
        credentials={"api_key": os.environ.get("BUSINESS_SYSTEM_API_KEY", "your_api_key")},
        header_name="X-API-Key"
    )
    
    # Create API configuration - assuming internal system with no rate limits
    business_api = ApiConfig(
        name="internal_erp",
        base_url="https://erp.internal.example.com/api",
        auth=auth_config,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json"
        },
        global_cache_config=CacheConfig(
            enabled=True,
            ttl_seconds=60,  # Short cache time for business data
            max_size_mb=200  # Larger cache for business data
        ),
        description="Internal ERP System API",
        tags=["internal", "erp", "business"]
    )
    
    # Add business system endpoints
    business_api.endpoints["get_customers"] = EndpointConfig(
        name="get_customers",
        url="/customers",
        method=HttpMethod.GET,
        auth=auth_config,
        description="List all customers"
    )
    
    business_api.endpoints["get_inventory"] = EndpointConfig(
        name="get_inventory",
        url="/inventory",
        method=HttpMethod.GET,
        auth=auth_config,
        description="Get current inventory levels",
        transform_response=lambda data: {
            "inventory_items": data["items"],
            "total_value": sum(item["price"] * item["quantity"] for item in data["items"]),
            "low_stock_items": [item for item in data["items"] if item["quantity"] < item["reorder_level"]]
        }
    )
    
    business_api.endpoints["create_order"] = EndpointConfig(
        name="create_order",
        url="/orders",
        method=HttpMethod.POST,
        auth=auth_config,
        description="Create a new customer order",
        transform_request=lambda data: {
            "order": {
                "customer_id": data["customer_id"],
                "order_date": datetime.datetime.now().isoformat(),
                "items": [{
                    "product_id": item["id"],
                    "quantity": item["quantity"],
                    "unit_price": item["price"]
                } for item in data["items"]]
            }
        }
    )
    
    business_api.endpoints["get_financial_report"] = EndpointConfig(
        name="get_financial_report",
        url="/reports/financial",
        method=HttpMethod.GET,
        auth=auth_config,
        description="Get financial reports",
        # This endpoint returns CSV data instead of JSON
        response_format=DataFormat.CSV,
        # Convert CSV to JSON for easier consumption
        transform_response=lambda data: {"csv_data": data}
    )
    
    # Register the API with the gateway
    gateway.register_api(business_api)
    logger.info("Business System API registered")


async def register_weather_api(gateway: ApiGatewaySystem) -> None:
    """Register a public Weather API (Example)"""
    # Authentication using API key in query parameter
    auth_config = AuthConfig(
        auth_type=AuthType.API_KEY_QUERY,
        credentials={"api_key": os.environ.get("WEATHER_API_KEY", "your_api_key")},
        header_name="key"  # Parameter name for the API key
    )
    
    # Create API configuration with caching and rate limiting
    weather_api = ApiConfig(
        name="weather",
        base_url="https://api.weatherapi.com/v1",
        auth=auth_config,
        global_rate_limit=RateLimitConfig(
            requests_per_minute=60,
            burst_size=5
        ),
        global_cache_config=CacheConfig(
            enabled=True,
            ttl_seconds=1800,  # 30 minutes cache for weather data
            cache_keys=["q", "days"]
        ),
        description="Weather API for forecast and current conditions",
        tags=["weather", "public-api"]
    )
    
    # Add weather endpoints
    weather_api.endpoints["current"] = EndpointConfig(
        name="current",
        url="/current.json",
        method=HttpMethod.GET,
        auth=auth_config,
        description="Get current weather",
        transform_response=lambda data: {
            "location": f"{data['location']['name']}, {data['location']['country']}",
            "temperature_c": data["current"]["temp_c"],
            "temperature_f": data["current"]["temp_f"],
            "condition": data["current"]["condition"]["text"],
            "wind_kph": data["current"]["wind_kph"],
            "humidity": data["current"]["humidity"],
            "last_updated": data["current"]["last_updated"]
        }
    )
    
    weather_api.endpoints["forecast"] = EndpointConfig(
        name="forecast",
        url="/forecast.json",
        method=HttpMethod.GET,
        auth=auth_config,
        description="Get weather forecast",
        query_params={"days": "3"}  # Default to 3-day forecast
    )
    
    # Register the API with the gateway
    gateway.register_api(weather_api)
    logger.info("Weather API registered")


async def example_github_api_usage(gateway: ApiGatewaySystem) -> None:
    """Example of using the GitHub API through the gateway"""
    logger.info("GitHub API Example:")
    
    # List repositories for authenticated user
    logger.info("Listing repositories...")
    repos_response = await gateway.call_endpoint(
        api_name="github",
        endpoint_name="list_repos",
        client_id="example_client",
        parameters={"visibility": "public", "per_page": 5}
    )
    
    if repos_response["status"] == 200:
        repos = repos_response["data"]
        logger.info(f"Found {len(repos)} repositories")
        for repo in repos[:3]:  # Show first 3 repos
            logger.info(f"  - {repo['full_name']}: {repo['description']}")
    else:
        logger.error(f"Failed to list repositories: {repos_response['status']}")
    
    # Get details for a specific repository
    repo_owner = "octocat"
    repo_name = "hello-world"
    logger.info(f"Getting details for {repo_owner}/{repo_name}...")
    
    repo_response = await gateway.call_endpoint(
        api_name="github",
        endpoint_name="get_repo",
        client_id="example_client",
        parameters={"owner": repo_owner, "repo": repo_name}
    )
    
    if repo_response["status"] == 200:
        repo = repo_response["data"]
        logger.info(f"Repository: {repo['full_name']}")
        logger.info(f"Description: {repo['description']}")
        logger.info(f"Stars: {repo['stargazers_count']}")
        logger.info(f"Forks: {repo['forks_count']}")
    else:
        logger.error(f"Failed to get repository: {repo_response['status']}")


async def example_weather_api_usage(gateway: ApiGatewaySystem) -> None:
    """Example of using the Weather API through the gateway"""
    logger.info("\nWeather API Example:")
    
    # Get current weather for New York
    logger.info("Getting current weather for New York...")
    weather_response = await gateway.call_endpoint(
        api_name="weather",
        endpoint_name="current",
        client_id="example_client",
        parameters={"q": "New York"}
    )
    
    if weather_response["status"] == 200:
        weather = weather_response["data"]
        logger.info(f"Location: {weather['location']}")
        logger.info(f"Temperature: {weather['temperature_c']}°C / {weather['temperature_f']}°F")
        logger.info(f"Condition: {weather['condition']}")
        logger.info(f"Wind: {weather['wind_kph']} kph")
        logger.info(f"Humidity: {weather['humidity']}%")
    else:
        logger.error(f"Failed to get weather: {weather_response['status']}")
    
    # Get forecast for Tokyo (demonstrating rate limiting and caching)
    logger.info("\nGetting weather forecast for Tokyo...")
    forecast_response = await gateway.call_endpoint(
        api_name="weather",
        endpoint_name="forecast",
        client_id="example_client",
        parameters={"q": "Tokyo", "days": 3}
    )
    
    if forecast_response["status"] == 200:
        forecast = forecast_response["data"]
        logger.info(f"Forecast for: {forecast['location']['name']}, {forecast['location']['country']}")
        for day in forecast["forecast"]["forecastday"]:
            date = day["date"]
            max_temp = day["day"]["maxtemp_c"]
            min_temp = day["day"]["mintemp_c"]
            condition = day["day"]["condition"]["text"]
            logger.info(f"  - {date}: {min_temp}°C to {max_temp}°C, {condition}")
    else:
        logger.error(f"Failed to get forecast: {forecast_response['status']}")
    
    # Demonstrate caching by making the same request again
    logger.info("\nMaking the same request again to demonstrate caching...")
    start_time = datetime.datetime.now()
    cached_response = await gateway.call_endpoint(
        api_name="weather",
        endpoint_name="forecast",
        client_id="example_client",
        parameters={"q": "Tokyo", "days": 3}
    )
    elapsed = (datetime.datetime.now() - start_time).total_seconds() * 1000
    
    if "X-Cache" in cached_response["headers"] and cached_response["headers"]["X-Cache"] == "HIT":
        logger.info(f"Cache hit! Response time: {elapsed:.2f}ms")
    else:
        logger.info(f"Cache miss. Response time: {elapsed:.2f}ms")


async def example_business_system_usage(gateway: ApiGatewaySystem) -> None:
    """Example of using the internal business system API through the gateway"""
    logger.info("\nBusiness System API Example:")
    
    # Get inventory with data transformation
    logger.info("Getting inventory with data transformation...")
    inventory_response = await gateway.call_endpoint(
        api_name="internal_erp",
        endpoint_name="get_inventory",
        client_id="example_client"
    )
    
    if inventory_response["status"] == 200:
        inventory = inventory_response["data"]
        logger.info(f"Total inventory value: ${inventory['total_value']:,.2f}")
        logger.info(f"Total items: {len(inventory['inventory_items'])}")
        logger.info(f"Low stock items: {len(inventory['low_stock_items'])}")
        
        # Show some low stock items if there are any
        if inventory['low_stock_items']:
            logger.info("Low stock items:")
            for item in inventory['low_stock_items'][:3]:  # Show first 3
                logger.info(f"  - {item['name']}: {item['quantity']} in stock (reorder at {item['reorder_level']})")
    else:
        logger.error(f"Failed to get inventory: {inventory_response['status']}")
    
    # Create an order with data transformation
    order_data = {
        "customer_id": "CUST-12345",
        "items": [
            {"id": "PROD-001", "quantity": 5, "price": 19.99},
            {"id": "PROD-002", "quantity": 3, "price": 29.99}
        ]
    }
    
    logger.info("\nCreating a new order...")
    order_response = await gateway.call_endpoint(
        api_name="internal_erp",
        endpoint_name="create_order",
        client_id="example_client",
        body=order_data
    )
    
    if order_response["status"] == 201:  # Created
        logger.info(f"Order created successfully: {order_response['data']['order_id']}")
    else:
        logger.error(f"Failed to create order: {order_response['status']}")
    
    # Get financial report (CSV data transformed to JSON)
    logger.info("\nGetting financial report (CSV to JSON transformation)...")
    report_response = await gateway.call_endpoint(
        api_name="internal_erp",
        endpoint_name="get_financial_report",
        client_id="example_client",
        parameters={"period": "monthly", "month": "05", "year": "2025"}
    )
    
    if report_response["status"] == 200:
        # Show that we have the CSV data as a string in the JSON response
        csv_data = report_response["data"]["csv_data"]
        csv_lines = csv_data.strip().split("\n")
        logger.info(f"Received financial report with {len(csv_lines) - 1} data rows")
        logger.info(f"Headers: {csv_lines[0]}")
        if len(csv_lines) > 1:
            logger.info(f"First row: {csv_lines[1]}")
    else:
        logger.error(f"Failed to get financial report: {report_response['status']}")


async def example_data_transformation(gateway: ApiGatewaySystem) -> None:
    """Example of using the data transformation features"""
    logger.info("\nData Transformation Examples:")
    
    # JSON to XML transformation
    json_data = {
        "person": {
            "name": "John Doe",
            "age": 30,
            "address": {
                "street": "123 Main St",
                "city": "Springfield",
                "state": "IL"
            },
            "phones": ["555-1234", "555-5678"]
        }
    }
    
    logger.info("Transforming JSON to XML...")
    xml_result = gateway.transformer.transform(
        data=json_data,
        source_format=DataFormat.JSON,
        target_format=DataFormat.XML
    )
    
    logger.info(f"XML Result:\n{xml_result[:300]}...")  # Show first part of XML
    
    # Field mapping and filtering transformation
    logger.info("\nApplying field mapping and filtering...")
    transformed_data = gateway.transformer.transform(
        data=json_data,
        source_format=DataFormat.JSON,
        target_format=DataFormat.JSON,
        transformation_steps=[
            {
                "type": "field_map",
                "config": {
                    "mapping": {
                        "fullName": "person.name",
                        "userAge": "person.age",
                        "city": "person.address.city",
                        "primaryPhone": "person.phones.0"
                    }
                }
            },
            {
                "type": "add_fields",
                "config": {
                    "fields": {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "source": "example_transformation"
                    }
                }
            }
        ]
    )
    
    logger.info(f"Transformed data: {transformed_data}")
    
    # Flattening nested structures
    logger.info("\nFlattening nested structures...")
    flattened_data = gateway.transformer.transform(
        data=json_data,
        source_format=DataFormat.JSON,
        target_format=DataFormat.JSON,
        transformation_steps=[
            {
                "type": "flatten_nested",
                "config": {
                    "delimiter": "_"
                }
            }
        ]
    )
    
    logger.info(f"Flattened data: {flattened_data}")


async def main() -> None:
    """Main function to run the examples"""
    logger.info("API Gateway System Examples")
    logger.info("==========================")
    
    # Initialize the API Gateway
    gateway = await initialize_api_gateway()
    
    # Register example APIs
    await register_github_api(gateway)
    await register_twitter_api(gateway)
    await register_salesforce_api(gateway)
    await register_business_system_api(gateway)
    await register_weather_api(gateway)
    
    # Run examples
    await example_github_api_usage(gateway)
    await example_weather_api_usage(gateway)
    await example_business_system_usage(gateway)
    await example_data_transformation(gateway)
    
    # Show stats
    apis = gateway.list_apis()
    logger.info("\nRegistered APIs:")
    for api in apis:
        logger.info(f"  - {api['name']} ({api['description']})")


if __name__ == "__main__":
    asyncio.run(main())
