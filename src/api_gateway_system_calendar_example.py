"""
Example usage of the ApiGatewaySystem Calendar Extension

This example demonstrates how to use the ApiGatewaySystem Calendar extension
to manage events and scheduling through the A2A Protocol and Empire Framework.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta, date
from typing import Dict, Any, List

# Import ApiGatewaySystem components
from api_gateway_system import (
    ApiGatewaySystem, LogLevel
)

# Import Calendar Extension components
from api_gateway_system_calendar import (
    CalendarServiceAdapter, CalendarConfig, CalendarOperation,
    CalendarTaskTypes, EventPrivacy, EventPriority, EventStatus,
    AttendeeStatus, RecurrenceFrequency
)

# Import Empire Framework components
from principle_engine import PrincipleEngine
from principle_engine_example import create_example_principle_engine
from empire_framework.a2a.component_task_handler import Task, TaskStatus
from empire_framework.a2a.streaming_adapter import StreamingAdapter, StreamEventType


async def example_basic_calendar_operations() -> None:
    """Example of basic calendar operations."""
    print("\n=== Basic Calendar Operations ===")
    
    # Create API Gateway System
    api_gateway = ApiGatewaySystem(
        log_level=LogLevel.INFO
    )
    
    # Create Calendar Configuration
    calendar_config = CalendarConfig(
        calendar_api_url="https://calendar-api.example.com",
        api_key="sample-api-key",
        default_calendar_id="primary",
        timezone="America/New_York"
    )
    
    # Create Calendar Service Adapter
    calendar_adapter = CalendarServiceAdapter(
        api_gateway=api_gateway,
        calendar_config=calendar_config,
        agent_id="agent-123"
    )
    
    # Example 1: List Calendars
    print("\n--- Example 1: List Calendars ---")
    try:
        result = await calendar_adapter.execute_operation(
            CalendarOperation.LIST_CALENDARS,
            {}
        )
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Found {result['count']} calendars:")
            for calendar in result['calendars']:
                print(f"  - {calendar['name']} ({calendar['calendar_id']})")
                print(f"    Description: {calendar['description']}")
                print(f"    Owner: {calendar['owner']}")
                print(f"    Primary: {calendar['is_primary']}")
                print("    ---")
    except Exception as e:
        print(f"Error listing calendars: {str(e)}")
    
    # Example 2: Get Events
    print("\n--- Example 2: Get Events from Calendar ---")
    try:
        # Get events for the next 7 days
        today = datetime.now().date()
        next_week = today + timedelta(days=7)
        
        result = await calendar_adapter.execute_operation(
            CalendarOperation.GET_EVENTS,
            {
                "calendar_id": "primary",
                "start_date": today.isoformat(),
                "end_date": next_week.isoformat(),
                "max_results": 10
            }
        )
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Found {result['count']} events from {result['start_date']} to {result['end_date']}:")
            for event in result['events']:
                print(f"  - {event['title']}")
                print(f"    Time: {event['start_time']} to {event['end_time']}")
                print(f"    Location: {event['location']}")
                print(f"    Attendees: {len(event['attendees'])}")
                print("    ---")
    except Exception as e:
        print(f"Error getting events: {str(e)}")
    
    # Example 3: Create Event
    print("\n--- Example 3: Create a Calendar Event ---")
    try:
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
        
        result = await calendar_adapter.execute_operation(
            CalendarOperation.CREATE_EVENT,
            {
                "calendar_id": "primary",
                "title": "Project Planning Meeting",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "description": "Discuss Q3 project goals and timeline",
                "location": "Conference Room A",
                "attendees": [
                    {
                        "email": "team1@example.com",
                        "name": "Team Member 1",
                        "is_optional": False
                    },
                    {
                        "email": "team2@example.com",
                        "name": "Team Member 2",
                        "is_optional": True
                    }
                ],
                "privacy": EventPrivacy.INTERNAL.value,
                "priority": EventPriority.HIGH.value
            }
        )
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Created event with ID: {result['event_id']}")
            print(f"Title: {result['event']['title']}")
            print(f"Time: {result['event']['start_time']} to {result['event']['end_time']}")
            print(f"Attendees: {len(result['event']['attendees'])}")
    except Exception as e:
        print(f"Error creating event: {str(e)}")


async def example_scheduling_algorithms() -> None:
    """Example of scheduling algorithms for calendar operations."""
    print("\n=== Calendar Scheduling Algorithms ===")
    
    # Create API Gateway System
    api_gateway = ApiGatewaySystem(
        log_level=LogLevel.INFO
    )
    
    # Create Calendar Configuration
    calendar_config = CalendarConfig(
        calendar_api_url="https://calendar-api.example.com",
        api_key="sample-api-key",
        default_calendar_id="primary",
        timezone="America/New_York"
    )
    
    # Create Calendar Service Adapter
    calendar_adapter = CalendarServiceAdapter(
        api_gateway=api_gateway,
        calendar_config=calendar_config,
        agent_id="agent-123"
    )
    
    # Example 1: Check Availability
    print("\n--- Example 1: Check User Availability ---")
    try:
        # Check availability for multiple users tomorrow
        tomorrow = (datetime.now() + timedelta(days=1)).date()
        
        result = await calendar_adapter.execute_operation(
            CalendarOperation.CHECK_AVAILABILITY,
            {
                "emails": [
                    "user1@example.com",
                    "user2@example.com",
                    "user3@example.com"
                ],
                "date": tomorrow.isoformat(),
                "start_time": datetime.combine(
                    tomorrow,
                    datetime.min.time().replace(hour=9, minute=0)
                ).isoformat(),
                "end_time": datetime.combine(
                    tomorrow,
                    datetime.min.time().replace(hour=17, minute=0)
                ).isoformat()
            }
        )
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Checked availability for {len(result['availability'])} users on {result['date']}:")
            for user_avail in result['availability']:
                print(f"  - {user_avail['email']}")
                print(f"    Busy periods: {len(user_avail['busy_slots'])}")
                print(f"    Free periods: {len(user_avail['free_slots'])}")
                print("    Sample free slots:")
                for i, slot in enumerate(user_avail['free_slots'][:2]):
                    print(f"      {i+1}. {slot['start_time']} to {slot['end_time']}")
                print("    Preferred meeting times:")
                for i, pref in enumerate(user_avail['preferred_meeting_times'][:2]):
                    print(f"      {i+1}. {pref['start']} to {pref['end']}")
                print("    ---")
    except Exception as e:
        print(f"Error checking availability: {str(e)}")
    
    # Example 2: Find Optimal Meeting Time
    print("\n--- Example 2: Find Optimal Meeting Time ---")
    try:
        # Find a meeting time for multiple attendees
        today = datetime.now().date()
        next_week = today + timedelta(days=7)
        
        result = await calendar_adapter.execute_operation(
            CalendarOperation.FIND_MEETING_TIME,
            {
                "attendees": [
                    "manager@example.com",
                    "team1@example.com",
                    "team2@example.com",
                    "client@example.com"
                ],
                "duration_minutes": 60,
                "date_range_start": today.isoformat(),
                "date_range_end": next_week.isoformat(),
                "priority": "availability"  # availability, preferences, time_of_day
            }
        )
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Found {len(result['suggested_times'])} potential meeting times:")
            for i, suggestion in enumerate(result['suggested_times']):
                print(f"  {i+1}. {suggestion['start_time']} to {suggestion['end_time']}")
                print(f"     Score: {suggestion['score']}")
                print(f"     Available Attendees: {suggestion['available_attendees']}/{suggestion['total_attendees']}")
                if 'conflicts' in suggestion and suggestion['conflicts']:
                    print(f"     Conflicts: {len(suggestion['conflicts'])}")
                    for conflict in suggestion['conflicts']:
                        print(f"       - {conflict['email']}: {conflict['reason']}")
                print(f"     Location Suggestions: {suggestion['location_suggestions'][0] if suggestion['location_suggestions'] else 'N/A'}")
                print("     ---")
    except Exception as e:
        print(f"Error finding meeting time: {str(e)}")


async def example_with_principle_evaluation() -> None:
    """Example of calendar operations with principle evaluation."""
    print("\n=== Calendar Operations with Principle Evaluation ===")
    
    # Create API Gateway System
    api_gateway = ApiGatewaySystem(
        log_level=LogLevel.INFO
    )
    
    # Create Calendar Configuration
    calendar_config = CalendarConfig(
        calendar_api_url="https://calendar-api.example.com",
        api_key="sample-api-key",
        default_calendar_id="primary",
        timezone="America/New_York"
    )
    
    # Create Principle Engine
    principle_engine = create_example_principle_engine()
    
    # Create Calendar Service Adapter with Principle Engine
    calendar_adapter = CalendarServiceAdapter(
        api_gateway=api_gateway,
        calendar_config=calendar_config,
        principle_engine=principle_engine,
        agent_id="agent-123"
    )
    
    # Example: Create events with principle evaluation
    print("\n--- Create Events with Principle Evaluation ---")
    
    # Example 1: Standard meeting (should pass principle evaluation)
    try:
        tomorrow = (datetime.now() + timedelta(days=1)).date()
        result = await calendar_adapter.execute_operation(
            CalendarOperation.CREATE_EVENT,
            {
                "calendar_id": "primary",
                "title": "Team Sync Meeting",
                "start_time": datetime.combine(
                    tomorrow, 
                    datetime.min.time().replace(hour=10, minute=0)
                ).isoformat(),
                "end_time": datetime.combine(
                    tomorrow, 
                    datetime.min.time().replace(hour=11, minute=0)
                ).isoformat(),
                "description": "Weekly team sync to discuss progress",
                "attendees": [
                    {"email": "team1@example.com", "name": "Team Member 1"},
                    {"email": "team2@example.com", "name": "Team Member 2"}
                ],
                "privacy": EventPrivacy.INTERNAL.value
            }
        )
        print(f"Standard meeting creation - Success: {result['success']}")
        if result['success']:
            print(f"Created event with ID: {result['event_id']}")
    except Exception as e:
        print(f"Error creating event: {str(e)}")
        
    # Example 2: After-hours meeting (might not pass principle evaluation)
    try:
        tomorrow = (datetime.now() + timedelta(days=1)).date()
        result = await calendar_adapter.execute_operation(
            CalendarOperation.CREATE_EVENT,
            {
                "calendar_id": "primary",
                "title": "Late Night Deployment",
                "start_time": datetime.combine(
                    tomorrow, 
                    datetime.min.time().replace(hour=23, minute=0)
                ).isoformat(),
                "end_time": datetime.combine(
                    tomorrow + timedelta(days=1), 
                    datetime.min.time().replace(hour=2, minute=0)
                ).isoformat(),
                "description": "Deploy new system during off-hours",
                "attendees": [
                    {"email": "team1@example.com", "name": "Team Member 1"},
                    {"email": "team2@example.com", "name": "Team Member 2"},
                    {"email": "team3@example.com", "name": "Team Member 3"}
                ],
                "priority": EventPriority.HIGH.value
            }
        )
        print(f"Late night meeting creation - Success: {result['success']}")
        
        if not result['success'] and "principle_evaluation" in result:
            print("Event was rejected by principle evaluation:")
            print(f"  Score: {result['principle_evaluation'].get('score', 0)}")
            print(f"  Violated Principles: {result['principle_evaluation'].get('violated_principles', [])}")
            
            # Suggest alternative time
            print("\nSuggesting alternative time during business hours...")
            result = await calendar_adapter.execute_operation(
                CalendarOperation.CREATE_EVENT,
                {
                    "calendar_id": "primary",
                    "title": "System Deployment Planning",
                    "start_time": datetime.combine(
                        tomorrow, 
                        datetime.min.time().replace(hour=15, minute=0)
                    ).isoformat(),
                    "end_time": datetime.combine(
                        tomorrow, 
                        datetime.min.time().replace(hour=16, minute=0)
                    ).isoformat(),
                    "description": "Plan for the overnight system deployment",
                    "attendees": [
                        {"email": "team1@example.com", "name": "Team Member 1"},
                        {"email": "team2@example.com", "name": "Team Member 2"},
                        {"email": "team3@example.com", "name": "Team Member 3"}
                    ],
                    "priority": EventPriority.HIGH.value
                }
            )
            print(f"Alternative meeting creation - Success: {result['success']}")
            if result['success']:
                print(f"Created alternative event with ID: {result['event_id']}")
    except Exception as e:
        print(f"Error with principle evaluation: {str(e)}")


async def example_a2a_tasks() -> None:
    """Example of using A2A tasks for calendar operations."""
    print("\n=== Calendar Operations with A2A Tasks ===")
    
    # Create API Gateway System
    api_gateway = ApiGatewaySystem(
        log_level=LogLevel.INFO
    )
    
    # Create Calendar Configuration
    calendar_config = CalendarConfig(
        calendar_api_url="https://calendar-api.example.com",
        api_key="sample-api-key",
        default_calendar_id="primary",
        timezone="America/New_York"
    )
    
    # Create Calendar Service Adapter
    calendar_adapter = CalendarServiceAdapter(
        api_gateway=api_gateway,
        calendar_config=calendar_config,
        agent_id="agent-123"
    )
    
    # Example 1: Create a task to schedule a meeting
    print("\n--- Example 1: Schedule Meeting Task ---")
    try:
        # Create a task to schedule a meeting with multiple attendees
        today = datetime.now().date()
        
        task_id = uuid.uuid4().hex
        task = Task(
            task_id=task_id,
            task_type=CalendarTaskTypes.SCHEDULE_MEETING,
            component_ids=[],
            task_data={
                "title": "Project Kickoff Meeting",
                "description": "Discuss project scope, roles, and timeline",
                "duration_minutes": 90,
                "preferred_date_range": {
                    "start": today.isoformat(),
                    "end": (today + timedelta(days=5)).isoformat()
                },
                "preferred_time_ranges": [
                    {
                        "start": "09:00",
                        "end": "12:00"
                    },
                    {
                        "start": "14:00",
                        "end": "16:00"
                    }
                ],
                "attendees": [
                    {
                        "email": "manager@example.com",
                        "name": "Project Manager",
                        "required": True
                    },
                    {
                        "email": "developer1@example.com",
                        "name": "Lead Developer",
                        "required": True
                    },
                    {
                        "email": "designer@example.com",
                        "name": "UI Designer",
                        "required": True
                    },
                    {
                        "email": "stakeholder@example.com",
                        "name": "Stakeholder",
                        "required": False
                    }
                ],
                "location_preferences": ["Conference Room A", "Conference Room B", "Virtual Meeting"],
                "priority": "high"
            },
            priority="high",
            created_by="agent-123"
        )
        
        # Process the task
        print(f"Processing task: {task.task_id}")
        print(f"Task Type: {task.task_type}")
        print(f"Attendees: {len(task.task_data['attendees'])}")
        print(f"Duration: {task.task_data['duration_minutes']} minutes")
        
        # Simulate task processing
        await asyncio.sleep(2)
        
        # Return simulated result
        result = {
            "task_id": task.task_id,
            "status": "completed",
            "scheduled_meeting": {
                "event_id": f"event-{uuid.uuid4().hex[:8]}",
                "title": task.task_data["title"],
                "start_time": datetime.combine(
                    today + timedelta(days=2),
                    datetime.min.time().replace(hour=10, minute=0)
                ).isoformat(),
                "end_time": datetime.combine(
                    today + timedelta(days=2),
                    datetime.min.time().replace(hour=11, minute=30)
                ).isoformat(),
                "location": "Conference Room A",
                "attendees_confirmed": 3,
                "attendees_pending": 1
            },
            "alternative_times": [
                {
                    "start_time": datetime.combine(
                        today + timedelta(days=3),
                        datetime.min.time().replace(hour=14, minute=0)
                    ).isoformat(),
                    "end_time": datetime.combine(
                        today + timedelta(days=3),
                        datetime.min.time().replace(hour=15, minute=30)
                    ).isoformat(),
                    "location": "Conference Room B",
                    "score": 0.85
                }
            ]
        }
        
        print("\nTask completed successfully")
        print(f"Scheduled meeting: {result['scheduled_meeting']['title']}")
        print(f"Time: {result['scheduled_meeting']['start_time']} to {result['scheduled_meeting']['end_time']}")
        print(f"Location: {result['scheduled_meeting']['location']}")
        print(f"Attendees confirmed: {result['scheduled_meeting']['attendees_confirmed']}")
        
        if result['alternative_times']:
            print(f"\nAlternative times available: {len(result['alternative_times'])}")
            for i, alt in enumerate(result['alternative_times']):
                print(f"  {i+1}. {alt['start_time']} to {alt['end_time']} at {alt['location']} (Score: {alt['score']})")
    
    except Exception as e:
        print(f"Error with task: {str(e)}")


async def example_streaming_updates() -> None:
    """Example of streaming calendar updates using SSE."""
    print("\n=== Streaming Calendar Updates via SSE ===")
    
    # Create API Gateway System
    api_gateway = ApiGatewaySystem(
        log_level=LogLevel.INFO
    )
    
    # Create Calendar Configuration
    calendar_config = CalendarConfig(
        calendar_api_url="https://calendar-api.example.com",
        api_key="sample-api-key",
        default_calendar_id="primary",
        timezone="America/New_York"
    )
    
    # Create Calendar Service Adapter
    calendar_adapter = CalendarServiceAdapter(
        api_gateway=api_gateway,
        calendar_config=calendar_config,
        agent_id="agent-123"
    )
    
    # Create Streaming Adapter
    streaming_adapter = StreamingAdapter()
    
    # Set up streaming channel for calendar events
    channel_id = f"calendar-events-{uuid.uuid4().hex[:8]}"
    
    # Event handler for streaming events
    async def calendar_event_handler(event_type: StreamEventType, data: Dict[str, Any]) -> None:
        print(f"\nStreaming Event: {event_type.value}")
        print(f"Data: {json.dumps(data, indent=2)}")
    
    # Register event handler
    streaming_adapter.register_event_handler(calendar_event_handler)
    
    # Function to simulate calendar events
    async def simulate_calendar_events() -> None:
        # Simulate new event creation
        new_event_data = {
            "event_id": f"event-{uuid.uuid4().hex[:8]}",
            "title": "Quarterly Planning Session",
            "start_time": (datetime.now() + timedelta(days=3)).isoformat(),
            "end_time": (datetime.now() + timedelta(days=3, hours=2)).isoformat(),
            "calendar_id": "primary",
            "created_by": "manager@example.com",
            "attendees": ["agent-123@example.com", "team1@example.com", "team2@example.com"]
        }
        
        # Stream the new event
        await streaming_adapter.stream_event(
            channel_id=channel_id,
            event_type=StreamEventType.DATA,
            data={
                "type": "event.created",
                "event": new_event_data
            }
        )
        
        # Wait a moment
        await asyncio.sleep(2)
        
        # Simulate event update
        await streaming_adapter.stream_event(
            channel_id=channel_id,
            event_type=StreamEventType.DATA,
            data={
                "type": "event.updated",
                "event_id": new_event_data["event_id"],
                "changes": {
                    "location": "Conference Room C",
                    "description": "Quarterly planning and budget review",
                    "attendees_added": ["finance@example.com"]
                }
            }
        )
        
        # Wait a moment
        await asyncio.sleep(2)
        
        # Simulate attendee response
        await streaming_adapter.stream_event(
            channel_id=channel_id,
            event_type=StreamEventType.DATA,
            data={
                "type": "event.attendee_response",
                "event_id": new_event_data["event_id"],
                "attendee": {
                    "email": "team1@example.com",
                    "name": "Team Member 1",
                    "response": "accepted",
                    "comment": "Looking forward to it!"
                }
            }
        )
    
    # Start streaming and simulate events
    print("\n--- Starting Calendar Streaming ---")
    try:
        # Connect to streaming channel
        await streaming_adapter.connect_channel(channel_id)
        print(f"Connected to streaming channel: {channel_id}")
        
        # Simulate calendar events
        await simulate_calendar_events()
        
        # Disconnect from channel
        await streaming_adapter.disconnect_channel(channel_id)
        print(f"Disconnected from streaming channel: {channel_id}")
    except Exception as e:
        print(f"Error with streaming: {str(e)}")


async def main() -> None:
    """Run all examples."""
    print("=== ApiGatewaySystem Calendar Extension Examples ===\n")
    
    try:
        # Run each example
        await example_basic_calendar_operations()
        await example_scheduling_algorithms()
        await example_with_principle_evaluation()
        await example_a2a_tasks()
        await example_streaming_updates()
    except Exception as e:
        print(f"Error running examples: {str(e)}")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
