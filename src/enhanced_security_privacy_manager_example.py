"""
Example usage of the Enhanced Security & Privacy Manager

This example demonstrates how to use the Enhanced Security & Privacy Manager
to secure authentication, enforce permissions, apply privacy principles,
log data access, and integrate with the Empire Framework's PrincipleEngine.
"""

import asyncio
import datetime
from datetime import timezone
import json
import os
import uuid
from typing import Dict, List, Any, Optional

# Import security components
from enhanced_security_privacy_manager import (
    EnhancedSecurityPrivacyManager,
    TokenData, Permission, PrivacyRule,
    DataCategory, OperationType, SecurityLevel,
    DataAccessRequest, DataAccessResult,
    AccessDeniedReason
)

# Import Empire Framework components
from principle_engine import PrincipleEngine
from principle_engine_example import create_example_principle_engine


async def example_token_management() -> None:
    """Example of token management for external services."""
    print("\n=== Token Management for External Services ===")
    
    # Create security manager
    security_manager = EnhancedSecurityPrivacyManager(
        token_secret="your-secure-secret-key",  # In production, use environment variables
        token_expiry=datetime.timedelta(hours=1)
    )
    
    # Example 1: Create authentication tokens
    print("\n--- Example 1: Create Authentication Tokens ---")
    
    # Create token for calendar service
    calendar_token, refresh_token = await security_manager.create_token(
        user_id="user-123",
        scopes=["calendar.read", "calendar.write", "calendar.share"],
        service_name="calendar-api",
        metadata={
            "description": "Access token for Calendar API",
            "client_ip": "192.168.1.100",
            "device_info": "Web Browser / Chrome"
        }
    )
    
    print(f"Generated Calendar API Token: {calendar_token[:20]}... (truncated)")
    print(f"With refresh token: {refresh_token[:10]}... (truncated)")
    
    # Create token for email service
    email_token, _ = await security_manager.create_token(
        user_id="user-123",
        scopes=["email.read", "email.send"],
        service_name="email-api",
        expiry=datetime.timedelta(minutes=30)  # Short-lived token
    )
    
    print(f"Generated Email API Token: {email_token[:20]}... (truncated)")
    
    # Example 2: Validate tokens
    print("\n--- Example 2: Validate Tokens ---")
    
    # Validate calendar token
    calendar_token_data = await security_manager.validate_token(calendar_token)
    if calendar_token_data:
        print("Calendar token is valid:")
        print(f"  User ID: {calendar_token_data.user_id}")
        print(f"  Service: {calendar_token_data.service_name}")
        print(f"  Scopes: {calendar_token_data.scopes}")
        print(f"  Created: {calendar_token_data.created_at}")
        print(f"  Expires: {calendar_token_data.expires_at}")
    else:
        print("Calendar token is invalid")
    
    # Create an invalid token
    invalid_token = calendar_token[:-5] + "XXXXX"  # Tamper with signature
    invalid_token_data = await security_manager.validate_token(invalid_token)
    if invalid_token_data:
        print("Invalid token was incorrectly validated")
    else:
        print("Successfully rejected invalid token")
    
    # Example 3: Refresh and revoke tokens
    print("\n--- Example 3: Refresh and Revoke Tokens ---")
    
    # Refresh calendar token
    if calendar_token_data and refresh_token:
        new_token, new_refresh = await security_manager.refresh_token(
            token_id=calendar_token_data.token_id,
            refresh_token=refresh_token
        )
        
        if new_token and new_refresh:
            print(f"Successfully refreshed token: {new_token[:20]}... (truncated)")
            print(f"New refresh token: {new_refresh[:10]}... (truncated)")
            
            # Check if original token is now invalid
            old_token_data = await security_manager.validate_token(calendar_token)
            if old_token_data:
                print("ERROR: Original token is still valid after refresh")
            else:
                print("Original token correctly invalidated after refresh")
        else:
            print("Failed to refresh token")
    
    # Revoke email token
    email_token_data = await security_manager.validate_token(email_token)
    if email_token_data:
        revoked = await security_manager.revoke_token(email_token_data.token_id)
        if revoked:
            print(f"Successfully revoked email token")
            
            # Verify token is revoked
            check_token = await security_manager.validate_token(email_token)
            if check_token:
                print("ERROR: Token still valid after revocation")
            else:
                print("Token correctly shows as invalid after revocation")
        else:
            print("Failed to revoke email token")


async def example_permission_enforcement() -> None:
    """Example of enforcing permissions for user actions."""
    print("\n=== Permission Enforcement for User Actions ===")
    
    # Create security manager
    security_manager = EnhancedSecurityPrivacyManager(
        token_secret="your-secure-secret-key"
    )
    
    # Example 1: Grant permissions to users
    print("\n--- Example 1: Grant Permissions ---")
    
    # Grant calendar permissions to a user
    calendar_admin_permission = Permission(
        resource_type="calendar",
        action="*",  # Wildcard for all actions
        resource_id=None  # All calendars
    )
    
    granted = await security_manager.grant_permission(
        user_id="admin-user",
        permission=calendar_admin_permission
    )
    print(f"Granted admin calendar permissions: {granted}")
    
    # Grant limited permissions to regular user
    calendar_read_permission = Permission(
        resource_type="calendar",
        action="read",
        resource_id=None  # All calendars
    )
    
    calendar_write_permission = Permission(
        resource_type="calendar",
        action="write",
        resource_id="personal-calendar",  # Only personal calendar
        conditions={
            "owner_id": "regular-user"  # Only if user is the owner
        }
    )
    
    await security_manager.grant_permission("regular-user", calendar_read_permission)
    await security_manager.grant_permission("regular-user", calendar_write_permission)
    print("Granted limited permissions to regular user")
    
    # Example 2: Check permissions
    print("\n--- Example 2: Check Permissions ---")
    
    # Check admin permissions
    admin_can_read = await security_manager.has_permission(
        user_id="admin-user",
        resource_type="calendar",
        action="read",
        resource_id="team-calendar"
    )
    print(f"Admin can read team calendar: {admin_can_read}")
    
    admin_can_delete = await security_manager.has_permission(
        user_id="admin-user",
        resource_type="calendar",
        action="delete",
        resource_id="system-calendar"
    )
    print(f"Admin can delete system calendar: {admin_can_delete}")
    
    # Check regular user permissions
    user_can_read = await security_manager.has_permission(
        user_id="regular-user",
        resource_type="calendar",
        action="read",
        resource_id="team-calendar"
    )
    print(f"Regular user can read team calendar: {user_can_read}")
    
    user_can_write_personal = await security_manager.has_permission(
        user_id="regular-user",
        resource_type="calendar",
        action="write",
        resource_id="personal-calendar",
        context={"owner_id": "regular-user"}  # Context matches condition
    )
    print(f"Regular user can write to personal calendar: {user_can_write_personal}")
    
    user_can_write_team = await security_manager.has_permission(
        user_id="regular-user",
        resource_type="calendar",
        action="write",
        resource_id="team-calendar"
    )
    print(f"Regular user can write to team calendar: {user_can_write_team}")
    
    # Example 3: Revoke permissions
    print("\n--- Example 3: Revoke Permissions ---")
    
    # Revoke specific permission
    revoked = await security_manager.revoke_permission(
        user_id="regular-user",
        resource_type="calendar",
        action="write",
        resource_id="personal-calendar"
    )
    print(f"Revoked write permission: {revoked}")
    
    # Check if permission was revoked
    user_can_write_after = await security_manager.has_permission(
        user_id="regular-user",
        resource_type="calendar",
        action="write",
        resource_id="personal-calendar",
        context={"owner_id": "regular-user"}
    )
    print(f"Regular user can write after revocation: {user_can_write_after}")


async def example_privacy_enforcement() -> None:
    """Example of enforcing privacy principles across operations."""
    print("\n=== Privacy Principle Enforcement ===")
    
    # Create security manager with principle engine
    principle_engine = create_example_principle_engine()
    security_manager = EnhancedSecurityPrivacyManager(
        token_secret="your-secure-secret-key",
        principle_engine=principle_engine
    )
    
    # Example 1: Define privacy rules
    print("\n--- Example 1: Define Privacy Rules ---")
    
    # Rule for basic profile data
    basic_profile_rule = PrivacyRule(
        rule_id="basic-profile-data",
        data_categories=[DataCategory.BASIC],
        operation_types=[OperationType.READ, OperationType.UPDATE],
        security_level=SecurityLevel.PUBLIC,
        requires_logging=True
    )
    
    # Rule for contact information
    contact_info_rule = PrivacyRule(
        rule_id="contact-info-data",
        data_categories=[DataCategory.CONTACT],
        operation_types=[OperationType.READ, OperationType.UPDATE],
        security_level=SecurityLevel.PROTECTED,
        purpose_limitation=["communication", "notification"],
        requires_consent=True,
        requires_logging=True
    )
    
    # Rule for location data
    location_data_rule = PrivacyRule(
        rule_id="location-data",
        data_categories=[DataCategory.LOCATION],
        operation_types=[OperationType.READ, OperationType.ANALYZE],
        security_level=SecurityLevel.PRIVATE,
        purpose_limitation=["calendar-scheduling", "travel-planning"],
        retention_period=datetime.timedelta(days=30),
        requires_consent=True,
        requires_logging=True
    )
    
    await security_manager.add_privacy_rule(basic_profile_rule)
    await security_manager.add_privacy_rule(contact_info_rule)
    await security_manager.add_privacy_rule(location_data_rule)
    print("Added privacy rules for different data categories")
    
    # Example 2: Request data access with privacy checks
    print("\n--- Example 2: Request Data Access with Privacy Checks ---")
    
    # Create access request for basic profile
    profile_request = DataAccessRequest(
        request_id=str(uuid.uuid4()),
        user_id="user-123",
        resource_type="user_profile",
        resource_id="profile-123",
        action="read",
        data_categories=[DataCategory.BASIC],
        operation_type=OperationType.READ,
        purpose="display-profile",
        client_info={
            "client_id": "web-app",
            "ip_address": "192.168.1.100"
        }
    )
    
    profile_result = await security_manager.request_data_access(profile_request)
    print(f"Profile access granted: {profile_result.access_granted}")
    if profile_result.access_granted:
        print("Access to basic profile data granted")
    else:
        print(f"Access denied: {profile_result.reason_denied}")
    
    # Create access request for location data with insufficient purpose
    location_request = DataAccessRequest(
        request_id=str(uuid.uuid4()),
        user_id="user-123",
        resource_type="user_location",
        resource_id="location-123",
        action="read",
        data_categories=[DataCategory.LOCATION],
        operation_type=OperationType.READ,
        purpose="marketing",  # Not in allowed purposes
        client_info={
            "client_id": "marketing-app",
            "ip_address": "192.168.1.100"
        }
    )
    
    location_result = await security_manager.request_data_access(location_request)
    print(f"Location access granted: {location_result.access_granted}")
    if not location_result.access_granted:
        print(f"Access denied reason: {location_result.reason_denied}")
        if location_result.privacy_warning:
            print(f"Privacy warning: {location_result.privacy_warning}")


async def example_access_logging() -> None:
    """Example of logging and monitoring data access."""
    print("\n=== Access Logging and Monitoring ===")
    
    # Create security manager
    security_manager = EnhancedSecurityPrivacyManager(
        token_secret="your-secure-secret-key",
        enable_detailed_logging=True
    )
    
    # Example 1: Automatic logging during access requests
    print("\n--- Example 1: Automatic Access Logging ---")
    
    # Setup permissions
    calendar_permission = Permission(
        resource_type="calendar",
        action="read"
    )
    await security_manager.grant_permission("user-123", calendar_permission)
    
    # Make some access requests
    for i in range(5):
        request = DataAccessRequest(
            request_id=f"request-{i+1}",
            user_id="user-123",
            resource_type="calendar",
            resource_id=f"calendar-{i+1}",
            action="read",
            data_categories=[DataCategory.BASIC],
            operation_type=OperationType.READ,
            purpose="view-calendar"
        )
        
        result = await security_manager.request_data_access(request)
        print(f"Request {i+1} granted: {result.access_granted}")
    
    # Unauthorized access attempt
    unauthorized_request = DataAccessRequest(
        request_id="unauthorized-1",
        user_id="user-123",
        resource_type="admin_panel",
        resource_id="global-settings",
        action="update",
        data_categories=[DataCategory.BASIC],
        operation_type=OperationType.UPDATE,
        purpose="change-settings"
    )
    
    result = await security_manager.request_data_access(unauthorized_request)
    print(f"Unauthorized request granted: {result.access_granted}")
    
    # Example 2: Retrieve access logs
    print("\n--- Example 2: Retrieve Access Logs ---")
    
    # Get all access logs
    all_logs = await security_manager.get_access_logs()
    print(f"Total access logs: {len(all_logs)}")
    
    # Get access logs for specific user
    user_logs = await security_manager.get_access_logs(user_id="user-123")
    print(f"User-specific logs: {len(user_logs)}")
    
    # Get denied access logs
    denied_logs = await security_manager.get_access_logs(access_granted=False)
    print(f"Denied access logs: {len(denied_logs)}")
    
    # Print example log
    if denied_logs:
        example_log = denied_logs[0]
        print("\nExample denied access log:")
        print(f"  Time: {example_log.timestamp}")
        print(f"  User: {example_log.user_id}")
        print(f"  Resource: {example_log.resource_type}/{example_log.resource_id}")
        print(f"  Action: {example_log.action}")
        print(f"  Operation: {example_log.operation_type.name}")
        print(f"  Reason denied: {example_log.reason_denied.name if example_log.reason_denied else 'Unknown'}")


async def example_principle_integration() -> None:
    """Example of integrating with Empire Framework's PrincipleEngine."""
    print("\n=== Integration with PrincipleEngine ===")
    
    # Create principle engine
    principle_engine = create_example_principle_engine()
    
    # Create security manager with principle engine
    security_manager = EnhancedSecurityPrivacyManager(
        token_secret="your-secure-secret-key",
        principle_engine=principle_engine
    )
    
    # Example 1: Principle-based decision making
    print("\n--- Example 1: Principle-Based Decision Making ---")
    
    # Create data access request that should trigger principle evaluation
    principle_request = DataAccessRequest(
        request_id=str(uuid.uuid4()),
        user_id="user-123",
        resource_type="health_data",
        resource_id="health-123",
        action="share",
        data_categories=[DataCategory.HEALTH],
        operation_type=OperationType.SHARE,
        purpose="research",
        requires_principle_evaluation=True
    )
    
    result = await security_manager.request_data_access(principle_request)
    print(f"Health data sharing request granted: {result.access_granted}")
    
    if result.principle_evaluation:
        print("\nPrinciple evaluation results:")
        print(f"  Score: {result.principle_evaluation.get('score', 0)}")
        print(f"  Principles applied: {len(result.principle_evaluation.get('principles_applied', []))}")
        
        for i, principle in enumerate(result.principle_evaluation.get('principles_applied', [])):
            print(f"  Principle {i+1}: {principle.get('name')}")
            print(f"    Score: {principle.get('score')}")
            print(f"    Reason: {principle.get('reason')}")
    
    # Example 2: Testing different scenarios
    print("\n--- Example 2: Testing Different Ethical Scenarios ---")
    
    # Scenario 1: Sharing anonymized data (should be approved)
    anon_request = DataAccessRequest(
        request_id=str(uuid.uuid4()),
        user_id="user-123",
        resource_type="health_data",
        resource_id="health-123",
        action="share",
        data_categories=[DataCategory.HEALTH],
        operation_type=OperationType.SHARE,
        purpose="medical-research",
        context={"anonymized": True, "recipient": "research-institution"},
        requires_principle_evaluation=True
    )
    
    anon_result = await security_manager.request_data_access(anon_request)
    print(f"Anonymized data sharing request granted: {anon_result.access_granted}")
    if anon_result.principle_evaluation:
        print(f"Evaluation score: {anon_result.principle_evaluation.get('score', 0)}")
    
    # Scenario 2: Sharing with commercial entity (may be rejected)
    commercial_request = DataAccessRequest(
        request_id=str(uuid.uuid4()),
        user_id="user-123",
        resource_type="health_data",
        resource_id="health-123",
        action="share",
        data_categories=[DataCategory.HEALTH],
        operation_type=OperationType.SHARE,
        purpose="marketing",
        context={"anonymized": False, "recipient": "advertising-company"},
        requires_principle_evaluation=True
    )
    
    comm_result = await security_manager.request_data_access(commercial_request)
    print(f"Commercial data sharing request granted: {comm_result.access_granted}")
    if not comm_result.access_granted and comm_result.reason_denied == AccessDeniedReason.PRINCIPLE_VIOLATION:
        print("Rejected based on principle violation")
        if comm_result.principle_evaluation:
            print(f"Evaluation score: {comm_result.principle_evaluation.get('score', 0)}")


async def main() -> None:
    """Run all examples."""
    print("=== Enhanced Security & Privacy Manager Examples ===\n")
    
    try:
        # Run each example
        await example_token_management()
        await example_permission_enforcement()
        await example_privacy_enforcement()
        await example_access_logging()
        await example_principle_integration()
    except Exception as e:
        print(f"Error running examples: {str(e)}")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
