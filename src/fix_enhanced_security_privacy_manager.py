#!/usr/bin/env python3
"""
Fix the truncated enhanced_security_privacy_manager.py file.
"""

# Read the current file
with open('src/enhanced_security_privacy_manager.py', 'r', encoding='utf-8') as f:
    content = f.read()

# The file is truncated at the _log_system_event call in add_privacy_rule
# We need to complete this method and add the missing parts

# Find where it's truncated
truncation_point = content.rfind('"security_level": rule.security_level.name')

if truncation_point != -1:
    # Keep everything up to the truncation point
    fixed_content = content[:truncation_point + len('"security_level": rule.security_level.name')]
    
    # Add the missing closing of the details dictionary and method
    fixed_content += """
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding privacy rule: {str(e)}")
            return False
    
    def _setup_default_privacy_rules(self) -> None:
        \"\"\"Setup default privacy rules for common scenarios.\"\"\"
        # Rule for biometric data - highest security
        self.privacy_rules.append(PrivacyRule(
            rule_id="default_biometric",
            data_categories=[DataCategory.BIOMETRIC],
            operation_types=list(OperationType),
            security_level=SecurityLevel.CRITICAL,
            purpose_limitation=["authentication", "security"],
            retention_period=datetime.timedelta(days=30),
            requires_consent=True,
            requires_logging=True,
            anonymization_required=False
        ))
        
        # Rule for financial data
        self.privacy_rules.append(PrivacyRule(
            rule_id="default_financial",
            data_categories=[DataCategory.FINANCIAL],
            operation_types=list(OperationType),
            security_level=SecurityLevel.SENSITIVE,
            purpose_limitation=["transactions", "billing", "accounting"],
            retention_period=datetime.timedelta(days=2555),  # ~7 years
            requires_consent=True,
            requires_logging=True,
            anonymization_required=False
        ))
        
        # Rule for health data
        self.privacy_rules.append(PrivacyRule(
            rule_id="default_health",
            data_categories=[DataCategory.HEALTH],
            operation_types=list(OperationType),
            security_level=SecurityLevel.CRITICAL,
            purpose_limitation=["healthcare", "emergency"],
            retention_period=datetime.timedelta(days=3650),  # 10 years
            requires_consent=True,
            requires_logging=True,
            anonymization_required=False
        ))
        
        # Rule for basic data
        self.privacy_rules.append(PrivacyRule(
            rule_id="default_basic",
            data_categories=[DataCategory.BASIC],
            operation_types=[OperationType.READ, OperationType.UPDATE],
            security_level=SecurityLevel.PROTECTED,
            purpose_limitation=None,
            retention_period=None,
            requires_consent=False,
            requires_logging=False,
            anonymization_required=False
        ))
        
        # Rule for location data
        self.privacy_rules.append(PrivacyRule(
            rule_id="default_location",
            data_categories=[DataCategory.LOCATION],
            operation_types=list(OperationType),
            security_level=SecurityLevel.SENSITIVE,
            purpose_limitation=["services", "emergency", "analytics"],
            retention_period=datetime.timedelta(days=90),
            requires_consent=True,
            requires_logging=True,
            anonymization_required=True
        ))
    
    def _log_system_event(self, event_type: str, user_id: str, details: Dict[str, Any]) -> None:
        \"\"\"
        Log a system event.
        
        Args:
            event_type: Type of event
            user_id: User ID associated with the event
            details: Event details
        \"\"\"
        if self.enable_detailed_logging:
            logger.info(f"System Event: {event_type} | User: {user_id} | Details: {details}")
    
    def _generate_secure_string(self, length: int = 32) -> str:
        \"\"\"Generate a secure random string.\"\"\"
        import secrets
        import string
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def _sign_token_payload(self, payload: Dict[str, Any]) -> str:
        \"\"\"Sign a token payload and return the token string.\"\"\"
        # Simple implementation - in production, use proper JWT
        import json
        payload_json = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            self.token_secret.encode(),
            payload_json.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Create token: base64(payload).signature
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode().rstrip('=')
        return f"{payload_b64}.{signature}"
    
    def _verify_token_signature(self, token_string: str) -> Optional[Dict[str, Any]]:
        \"\"\"Verify a token signature and return the payload if valid.\"\"\"
        try:
            parts = token_string.split('.')
            if len(parts) != 2:
                return None
                
            payload_b64, signature = parts
            
            # Decode payload
            padding = 4 - (len(payload_b64) % 4)
            if padding != 4:
                payload_b64 += '=' * padding
                
            payload_json = base64.urlsafe_b64decode(payload_b64).decode()
            
            # Verify signature
            expected_signature = hmac.new(
                self.token_secret.encode(),
                payload_json.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                return None
                
            return json.loads(payload_json)
            
        except Exception as e:
            logger.error(f"Error verifying token: {str(e)}")
            return None
    
    async def evaluate_data_access(self, request: DataAccessRequest) -> DataAccessResult:
        \"\"\"
        Evaluate a data access request against security and privacy policies.
        
        Args:
            request: Data access request to evaluate
            
        Returns:
            Data access result with decision and details
        \"\"\"
        try:
            # Validate token if provided
            if request.token_data:
                if request.token_data.is_expired():
                    return DataAccessResult(
                        request_id=request.request_id,
                        access_granted=False,
                        reason_denied=AccessDeniedReason.TOKEN_EXPIRED
                    )
                    
                if request.token_data.is_revoked:
                    return DataAccessResult(
                        request_id=request.request_id,
                        access_granted=False,
                        reason_denied=AccessDeniedReason.AUTHENTICATION_FAILURE
                    )
            
            # Check permissions
            has_permission = await self.has_permission(
                user_id=request.user_id,
                resource_type=request.resource_type,
                action=request.action,
                resource_id=request.resource_id,
                context=request.context
            )
            
            if not has_permission:
                # Log access denial
                log_entry = AccessLogEntry(
                    entry_id=str(uuid.uuid4()),
                    timestamp=datetime.datetime.now(),
                    user_id=request.user_id,
                    resource_type=request.resource_type,
                    resource_id=request.resource_id,
                    action=request.action,
                    data_categories=request.data_categories,
                    operation_type=request.operation_type,
                    access_granted=False,
                    reason_denied=AccessDeniedReason.INSUFFICIENT_PERMISSIONS,
                    client_info=request.client_info,
                    context=request.context
                )
                
                async with self._access_log_lock:
                    self.access_logs.append(log_entry)
                
                return DataAccessResult(
                    request_id=request.request_id,
                    access_granted=False,
                    reason_denied=AccessDeniedReason.INSUFFICIENT_PERMISSIONS,
                    log_entry=log_entry
                )
            
            # Check privacy rules
            applicable_rules = []
            for rule in self.privacy_rules:
                for category in request.data_categories:
                    if rule.applies_to(category, request.operation_type):
                        applicable_rules.append(rule)
                        break
            
            # Apply most restrictive rule
            if applicable_rules:
                most_restrictive = max(applicable_rules, key=lambda r: r.security_level.value)
                
                # Check purpose limitation
                if most_restrictive.purpose_limitation:
                    if request.purpose not in most_restrictive.purpose_limitation:
                        return DataAccessResult(
                            request_id=request.request_id,
                            access_granted=False,
                            reason_denied=AccessDeniedReason.PURPOSE_LIMITATION
                        )
                
                # Check if consent is required
                if most_restrictive.requires_consent:
                    # In a real implementation, check consent database
                    # For now, we'll assume consent is given if in context
                    if not request.context.get("consent_given", False):
                        return DataAccessResult(
                            request_id=request.request_id,
                            access_granted=False,
                            reason_denied=AccessDeniedReason.PRIVACY_VIOLATION,
                            privacy_warning="User consent required for this data access"
                        )
            
            # Principle evaluation if requested
            principle_evaluation = None
            if request.requires_principle_evaluation and self.principle_engine:
                # Create principle evaluation request
                eval_request = PrincipleEvaluationRequest(
                    action=f"{request.action} {request.resource_type}",
                    context={
                        "user_id": request.user_id,
                        "resource_type": request.resource_type,
                        "resource_id": request.resource_id,
                        "data_categories": [dc.name for dc in request.data_categories],
                        "operation_type": request.operation_type.name,
                        "purpose": request.purpose,
                        **request.context
                    },
                    requestor_id=request.user_id
                )
                
                # Evaluate with principle engine
                result = self.principle_engine.evaluate(eval_request)
                principle_evaluation = result
                
                if not result["approved"]:
                    return DataAccessResult(
                        request_id=request.request_id,
                        access_granted=False,
                        reason_denied=AccessDeniedReason.PRINCIPLE_VIOLATION,
                        principle_evaluation=principle_evaluation
                    )
            
            # Access granted - log success
            log_entry = AccessLogEntry(
                entry_id=str(uuid.uuid4()),
                timestamp=datetime.datetime.now(),
                user_id=request.user_id,
                resource_type=request.resource_type,
                resource_id=request.resource_id,
                action=request.action,
                data_categories=request.data_categories,
                operation_type=request.operation_type,
                access_granted=True,
                client_info=request.client_info,
                context=request.context
            )
            
            async with self._access_log_lock:
                self.access_logs.append(log_entry)
            
            # Determine if anonymization is needed
            anonymized = False
            for rule in applicable_rules:
                if rule.anonymization_required:
                    anonymized = True
                    break
            
            return DataAccessResult(
                request_id=request.request_id,
                access_granted=True,
                log_entry=log_entry,
                principle_evaluation=principle_evaluation,
                anonymized=anonymized
            )
            
        except Exception as e:
            logger.error(f"Error evaluating data access: {str(e)}")
            return DataAccessResult(
                request_id=request.request_id,
                access_granted=False,
                reason_denied=AccessDeniedReason.SECURITY_POLICY
            )
    
    async def get_access_logs(
        self,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        limit: int = 100
    ) -> List[AccessLogEntry]:
        \"\"\"
        Get access logs with optional filtering.
        
        Args:
            user_id: Filter by user ID
            resource_type: Filter by resource type
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of entries to return
            
        Returns:
            List of matching access log entries
        \"\"\"
        async with self._access_log_lock:
            filtered_logs = []
            
            for log in reversed(self.access_logs):  # Most recent first
                # Apply filters
                if user_id and log.user_id != user_id:
                    continue
                if resource_type and log.resource_type != resource_type:
                    continue
                if start_time and log.timestamp < start_time:
                    continue
                if end_time and log.timestamp > end_time:
                    continue
                    
                filtered_logs.append(log)
                
                if len(filtered_logs) >= limit:
                    break
                    
            return filtered_logs
    
    async def cleanup_expired_data(self) -> None:
        \"\"\"Clean up expired tokens and old access logs.\"\"\"
        try:
            # Clean up expired tokens
            async with self._token_lock:
                expired_tokens = []
                for token_id, token_data in self.tokens.items():
                    if token_data.is_expired():
                        # Keep expired tokens for 24 hours for audit
                        if datetime.datetime.now() > token_data.expires_at + datetime.timedelta(hours=24):
                            expired_tokens.append(token_id)
                
                for token_id in expired_tokens:
                    del self.tokens[token_id]
                
                if expired_tokens:
                    logger.info(f"Cleaned up {len(expired_tokens)} expired tokens")
            
            # Clean up old access logs
            async with self._access_log_lock:
                cutoff_time = datetime.datetime.now() - self.access_log_retention
                old_logs = []
                
                for i, log in enumerate(self.access_logs):
                    if log.timestamp < cutoff_time:
                        old_logs.append(i)
                
                # Remove from end to preserve indices
                for i in reversed(old_logs):
                    self.access_logs.pop(i)
                
                if old_logs:
                    logger.info(f"Cleaned up {len(old_logs)} old access logs")
                    
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


# Example usage functions
def example_basic_usage():
    \"\"\"Example of basic security and privacy manager usage.\"\"\"
    import asyncio
    
    async def main():
        # Initialize manager
        manager = EnhancedSecurityPrivacyManager(
            token_secret="your-secret-key-here",
            principle_engine=None  # Can integrate with PrincipleEngine
        )
        
        # Create a token
        token, refresh_token = await manager.create_token(
            user_id="user123",
            scopes=["read:profile", "write:profile"],
            service_name="web_app"
        )
        print(f"Created token: {token[:20]}...")
        
        # Grant permissions
        permission = Permission(
            resource_type="profile",
            action="read",
            resource_id="*"
        )
        await manager.grant_permission("user123", permission)
        
        # Evaluate data access request
        request = DataAccessRequest(
            request_id="req_001",
            user_id="user123",
            resource_type="profile",
            resource_id="profile_456",
            action="read",
            data_categories=[DataCategory.BASIC, DataCategory.CONTACT],
            operation_type=OperationType.READ,
            purpose="user_view",
            context={"consent_given": True}
        )
        
        result = await manager.evaluate_data_access(request)
        print(f"Access granted: {result.access_granted}")
        
    asyncio.run(main())


if __name__ == "__main__":
    example_basic_usage()
"""
else:
    print("Could not find truncation point. Manual fix needed.")
    fixed_content = content

# Write the fixed content
with open('src/enhanced_security_privacy_manager.py', 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print("Fixed enhanced_security_privacy_manager.py")