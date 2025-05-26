#!/usr/bin/env python3
"""
Tests for the ApiGatewaySystem Calendar Extension

This module provides comprehensive tests for the CalendarServiceAdapter's functionality
including principle-guided operations, error handling, and data retrieval.
"""

import unittest
import asyncio
import json
from datetime import datetime, timedelta, date
from unittest.mock import MagicMock, patch

from api_gateway_system import ApiGatewaySystem, LogLevel
from api_gateway_system_calendar import (
    CalendarServiceAdapter, CalendarConfig, CalendarOperation,
    EventPrivacy, EventPriority, EventStatus, AttendeeStatus,
    TimeSlot, CalendarEvent, CalendarInfo, AvailabilityInfo
)
from principle_engine import PrincipleEngine
from principle_engine_example import PrincipleEvaluationRequest


class MockPrincipleEngine:
    """Mock PrincipleEngine for testing."""
    
    async def evaluate_content(self, request: PrincipleEvaluationRequest) -> dict:
        """Mock evaluation that checks for after-hours meetings."""
        content = request.content
        context = request.context
        
        # Default to approving most operations
        score = 0.9
        violated_principles = []
        
        # Special case: Check for after-hours meetings
        if "CREATE_EVENT" in content:
            # Check if this is a late night or early morning meeting
            if "event_start_hour" in context:
                hour = context["event_start_hour"]
                if hour < 7 or hour > 19:  # Before 7am or after 7pm
                    score = 0.6  # Below the threshold
                    violated_principles.append("work_life_balance")
                    
            # Check if this is a weekend meeting
            if context.get("is_weekend", False):
                score = 0.65  # Below the threshold
                violated_principles.append("work_life_balance")
                
        # Return evaluation result
        return {
            "score": score,
            "violated_principles": violated_principles,
            "reasoning": "Automated test evaluation"
        }


class TestCalendarServiceAdapter(unittest.TestCase):
    """Test cases for the CalendarServiceAdapter."""
    
    def setUp(self) -> None:
        """Set up the test environment."""
        # Create API Gateway
        self.api_gateway = ApiGatewaySystem(log_level=LogLevel.INFO)
        
        # Create Calendar Configuration
        self.calendar_config = CalendarConfig(
            calendar_api_url="https://test-calendar-api.example.com",
            api_key="test-api-key",
            default_calendar_id="primary",
            timezone="timezone.utc"
        )
        
        # Create mock principle engine
        self.principle_engine = MockPrincipleEngine()
        
        # Create Calendar Service Adapter
        self.calendar_adapter = CalendarServiceAdapter(
            api_gateway=self.api_gateway,
            calendar_config=self.calendar_config,
            principle_engine=self.principle_engine,
            agent_id="test-agent"
        )
    
    def test_initialization(self) -> None:
        """Test that the adapter initializes correctly."""
        self.assertEqual(self.calendar_adapter.agent_id, "test-agent")
        self.assertEqual(self.calendar_adapter.calendar_config.calendar_api_url, 
                        "https://test-calendar-api.example.com")
        self.assertEqual(self.calendar_adapter.calendar_config.api_key, 
                        "test-api-key")
        self.assertIsNotNone(self.calendar_adapter.principle_engine)
    
    async def test_list_calendars(self) -> None:
        """Test listing calendars."""
        result = await self.calendar_adapter.execute_operation(
            CalendarOperation.LIST_CALENDARS,
            {}
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 3)  # Should have 3 sample calendars
        
        # Check that we get calendar objects
        calendars = result["calendars"]
        self.assertEqual(len(calendars), 3)
        
        # Check for expected fields
        self.assertIn("calendar_id", calendars[0])
        self.assertIn("name", calendars[0])
        self.assertIn("description", calendars[0])
        self.assertIn("is_primary", calendars[0])
        
        # Test cache hit on second call
        result = await self.calendar_adapter.execute_operation(
            CalendarOperation.LIST_CALENDARS,
            {}
        )
        self.assertTrue(result["from_cache"])
    
    async def test_get_events(self) -> None:
        """Test getting events from calendar."""
        # Get events for the next 7 days
        today = datetime.now().date()
        next_week = today + timedelta(days=7)
        
        result = await self.calendar_adapter.execute_operation(
            CalendarOperation.GET_EVENTS,
            {
                "calendar_id": "primary",
                "start_date": today.isoformat(),
                "end_date": next_week.isoformat()
            }
        )
        
        self.assertTrue(result["success"])
        self.assertGreater(result["count"], 0)  # Should have some events
        
        # Check that we get event objects
        events = result["events"]
        self.assertGreater(len(events), 0)
        
        # Check for expected fields
        self.assertIn("event_id", events[0])
        self.assertIn("title", events[0])
        self.assertIn("start_time", events[0])
        self.assertIn("end_time", events[0])
        self.assertIn("attendees", events[0])
    
    async def test_create_event(self) -> None:
        """Test creating a calendar event."""
        # Create an event for tomorrow
        tomorrow = (datetime.now() + timedelta(days=1)).date()
        start_time = datetime.combine(
            tomorrow, 
            datetime.min.time().replace(hour=14, minute=0)
        )
        end_time = datetime.combine(
            tomorrow, 
            datetime.min.time().replace(hour=15, minute=0)
        )
        
        result = await self.calendar_adapter.execute_operation(
            CalendarOperation.CREATE_EVENT,
            {
                "calendar_id": "primary",
                "title": "Test Meeting",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "description": "Test meeting description",
                "location": "Test Room",
                "attendees": [
                    {
                        "email": "attendee1@example.com",
                        "name": "Attendee 1",
                        "is_optional": False
                    },
                    {
                        "email": "attendee2@example.com",
                        "name": "Attendee 2",
                        "is_optional": True
                    }
                ],
                "privacy": EventPrivacy.INTERNAL.value,
                "priority": EventPriority.HIGH.value
            }
        )
        
        self.assertTrue(result["success"])
        self.assertIn("event_id", result)
        self.assertIn("event", result)
        
        # Check that the event has the properties we set
        event = result["event"]
        self.assertEqual(event["title"], "Test Meeting")
        self.assertEqual(event["description"], "Test meeting description")
        self.assertEqual(event["location"], "Test Room")
        self.assertEqual(len(event["attendees"]), 2)
        self.assertEqual(event["privacy"], EventPrivacy.INTERNAL.value)
        self.assertEqual(event["priority"], EventPriority.HIGH.value)
    
    async def test_principle_evaluation_working_hours(self) -> None:
        """Test principle evaluation for working hours enforcement."""
        # Create an event during work hours (should pass)
        tomorrow = (datetime.now() + timedelta(days=1)).date()
        start_time = datetime.combine(
            tomorrow, 
            datetime.min.time().replace(hour=14, minute=0)
        )
        end_time = datetime.combine(
            tomorrow, 
            datetime.min.time().replace(hour=15, minute=0)
        )
        
        result = await self.calendar_adapter.execute_operation(
            CalendarOperation.CREATE_EVENT,
            {
                "calendar_id": "primary",
                "title": "Work Hours Meeting",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "description": "Meeting within working hours",
            }
        )
        
        self.assertTrue(result["success"])
        self.assertIn("event_id", result)
        
        # Create an event outside work hours (should fail principle evaluation)
        late_start_time = datetime.combine(
            tomorrow, 
            datetime.min.time().replace(hour=22, minute=0)
        )
        late_end_time = datetime.combine(
            tomorrow, 
            datetime.min.time().replace(hour=23, minute=0)
        )
        
        result = await self.calendar_adapter.execute_operation(
            CalendarOperation.CREATE_EVENT,
            {
                "calendar_id": "primary",
                "title": "Late Night Meeting",
                "start_time": late_start_time.isoformat(),
                "end_time": late_end_time.isoformat(),
                "description": "Meeting outside working hours",
            }
        )
        
        self.assertFalse(result["success"])
        self.assertIn("principle_evaluation", result)
        self.assertLess(result["principle_evaluation"]["score"], 0.7)
        self.assertIn("work_life_balance", result["principle_evaluation"]["violated_principles"])
    
    async def test_availability_calculation(self) -> None:
        """Test availability checking and calculation."""
        # Check availability for multiple users tomorrow
        tomorrow = (datetime.now() + timedelta(days=1)).date()
        
        result = await self.calendar_adapter.execute_operation(
            CalendarOperation.CHECK_AVAILABILITY,
            {
                "emails": [
                    "user1@example.com",
                    "user2@example.com"
                ],
                "date": tomorrow.isoformat()
            }
        )
        
        self.assertTrue(result["success"])
        self.assertEqual(len(result["availability"]), 2)
        
        # Check that we get availability objects with free slots
        for user_avail in result["availability"]:
            self.assertIn("email", user_avail)
            self.assertIn("busy_slots", user_avail)
            self.assertIn("free_slots", user_avail)
            self.assertGreater(len(user_avail["free_slots"]), 0)
    
    async def test_find_meeting_time(self) -> None:
        """Test finding optimal meeting times."""
        # Find a meeting time for multiple attendees
        today = datetime.now().date()
        next_week = today + timedelta(days=7)
        
        result = await self.calendar_adapter.execute_operation(
            CalendarOperation.FIND_MEETING_TIME,
            {
                "attendees": [
                    "user1@example.com",
                    "user2@example.com"
                ],
                "duration_minutes": 60,
                "date_range_start": today.isoformat(),
                "date_range_end": next_week.isoformat()
            }
        )
        
        self.assertTrue(result["success"])
        self.assertGreater(len(result["suggested_times"]), 0)
        
        # Check that suggestions include expected fields
        suggestion = result["suggested_times"][0]
        self.assertIn("start_time", suggestion)
        self.assertIn("end_time", suggestion)
        self.assertIn("score", suggestion)
        self.assertIn("available_attendees", suggestion)
        self.assertIn("total_attendees", suggestion)
    
    async def test_error_handling(self) -> None:
        """Test error handling for invalid parameters."""
        # Test missing required parameters
        result = await self.calendar_adapter.execute_operation(
            CalendarOperation.CREATE_EVENT,
            {
                "calendar_id": "primary",
                # Missing title, start_time, and end_time
            }
        )
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("Missing required parameters", result["error"])
        
        # Test invalid status value
        tomorrow = (datetime.now() + timedelta(days=1)).date()
        start_time = datetime.combine(
            tomorrow, 
            datetime.min.time().replace(hour=14, minute=0)
        )
        end_time = datetime.combine(
            tomorrow, 
            datetime.min.time().replace(hour=15, minute=0)
        )
        
        result = await self.calendar_adapter.execute_operation(
            CalendarOperation.RESPOND_TO_INVITATION,
            {
                "event_id": "test-event",
                "response": "INVALID_STATUS",  # Invalid status
                "attendee_email": "user@example.com"
            }
        )
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertIn("Invalid response status", result["error"])
    
    async def test_enhance_principle_context(self) -> None:
        """Test the context enhancement for principle evaluation."""
        # Create a request with minimal context
        request = PrincipleEvaluationRequest(
            content="Calendar operation: CREATE_EVENT",
            context={
                "operation": CalendarOperation.CREATE_EVENT.value,
                "parameters": {},
                "agent_id": "test-agent",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
        # Prepare parameters for a team meeting
        tomorrow = (datetime.now() + timedelta(days=1)).date()
        start_time = datetime.combine(
            tomorrow, 
            datetime.min.time().replace(hour=14, minute=0)
        )
        end_time = datetime.combine(
            tomorrow, 
            datetime.min.time().replace(hour=15, minute=0)
        )
        
        params = {
            "title": "Team Meeting",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "description": "Weekly team sync",
            "attendees": [
                {"email": "user1@example.com"},
                {"email": "user2@example.com"}
            ],
            "location": "Conference Room"
        }
        
        # Call the enhance context method
        self.calendar_adapter._enhance_principle_context(
            CalendarOperation.CREATE_EVENT,
            params,
            request
        )
        
        # Check that context was enhanced with expected fields
        self.assertIn("current_time", request.context)
        self.assertIn("current_day_of_week", request.context)
        self.assertIn("current_hour", request.context)
        self.assertIn("event_title", request.context)
        self.assertIn("event_description", request.context)
        self.assertIn("event_attendees", request.context)
        self.assertIn("event_attendee_count", request.context)
        self.assertIn("event_location", request.context)
        self.assertIn("event_start_time", request.context)
        self.assertIn("event_end_time", request.context)
        self.assertIn("event_start_hour", request.context)
        self.assertIn("event_day_of_week", request.context)
        self.assertIn("event_duration_minutes", request.context)
        
        # Check specific values
        self.assertEqual(request.context["event_title"], "Team Meeting")
        self.assertEqual(request.context["event_description"], "Weekly team sync")
        self.assertEqual(request.context["event_attendee_count"], 2)
        self.assertEqual(request.context["event_location"], "Conference Room")
        self.assertEqual(request.context["event_start_hour"], 14)
        self.assertEqual(request.context["event_duration_minutes"], 60)
        self.assertFalse(request.context["is_weekend"])
        self.assertFalse(request.context.get("is_long_meeting", False))


if __name__ == "__main__":
    # Run tests using asyncio to support async test methods
    import asyncio
    
    def run_async_test(test_case) -> None:
        """Run async test methods."""
        test_method = getattr(test_case, test_case._testMethodName)
        if asyncio.iscoroutinefunction(test_method):
            return asyncio.run(test_method())
        return test_method()
    
    unittest.TestCase.run = lambda self, *args, **kwargs: run_async_test(self)
    unittest.main()
