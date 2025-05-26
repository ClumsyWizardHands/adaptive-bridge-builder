"""
ApiGatewaySystem Calendar Extension

This module extends the ApiGatewaySystem to handle calendar operations,
enabling agents to view, create, and manage calendar events using A2A Protocol and
Empire Framework components.
"""

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dateutil import parser as date_parser
import heapq

# Import ApiGatewaySystem components
from api_gateway_system import (
    ApiGatewaySystem, ApiConfig, EndpointConfig, AuthConfig, AuthType,
    DataFormat, HttpMethod, CacheConfig, RateLimitConfig, ErrorHandlingConfig,
    AuditLogEntry, LogLevel
)

# Import Empire Framework components
from principle_engine import PrincipleEngine
from empire_framework.a2a.component_task_handler import Task, ComponentTaskTypes, TaskStatus
from empire_framework.a2a.streaming_adapter import StreamingAdapter, StreamEventType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ApiGatewaySystemCalendar")


@dataclass
class PrincipleEvaluationRequest:
    """Request for principle evaluation."""
    action: str
    context: dict
    agent_id: str = "api_gateway"


class CalendarOperation(Enum):
    """Calendar operations supported by the system."""
    LIST_CALENDARS = "list_calendars"        # List available calendars
    GET_EVENTS = "get_events"                # Get events from calendar
    CREATE_EVENT = "create_event"            # Create a new event
    UPDATE_EVENT = "update_event"            # Update an existing event
    DELETE_EVENT = "delete_event"            # Delete an event
    CHECK_AVAILABILITY = "check_availability" # Check user availability
    FIND_MEETING_TIME = "find_meeting_time"  # Find optimal meeting time
    SUGGEST_LOCATION = "suggest_location"    # Suggest meeting location
    GET_ATTENDEES = "get_attendees"          # Get potential attendees
    SEND_INVITATION = "send_invitation"      # Send calendar invitation
    RESPOND_TO_INVITATION = "respond_to_invitation" # Respond to invitation


class CalendarTaskTypes:
    """A2A Task types for calendar operations."""
    SCHEDULE_MEETING = "calendar.scheduleMeeting"   # Schedule a meeting
    BATCH_EVENTS = "calendar.batchEvents"           # Process multiple events
    OPTIMIZE_SCHEDULE = "calendar.optimizeSchedule" # Optimize existing schedule
    ANALYZE_AVAILABILITY = "calendar.analyzeAvailability"  # Analyze availability patterns
    RESOLVE_CONFLICTS = "calendar.resolveConflicts" # Resolve scheduling conflicts
    SUGGEST_ALTERNATIVES = "calendar.suggestAlternatives" # Suggest alternative times
    GENERATE_AGENDA = "calendar.generateAgenda"     # Generate meeting agenda


class EventPriority(Enum):
    """Priority levels for calendar events."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class EventPrivacy(Enum):
    """Privacy levels for calendar events."""
    PUBLIC = "public"       # Visible to all users
    INTERNAL = "internal"   # Visible only within organization
    PRIVATE = "private"     # Visible only to invited attendees
    CONFIDENTIAL = "confidential"  # Details hidden, only shows as busy time


class EventStatus(Enum):
    """Status of calendar events."""
    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    CANCELLED = "cancelled"


class AttendeeStatus(Enum):
    """Attendee response status."""
    NEEDS_ACTION = "needs_action"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    TENTATIVE = "tentative"
    DELEGATED = "delegated"


class RecurrenceFrequency(Enum):
    """Frequency for recurring events."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


@dataclass
class CalendarConfig:
    """Configuration for calendar service."""
    calendar_api_url: str
    api_key: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    default_calendar_id: str = "primary"
    use_ssl: bool = True
    max_results: int = 100
    cache_ttl_minutes: int = 5
    timezone: str = "timezone.utc"
    auth_type: str = "api_key"  # api_key, oauth, none


@dataclass
class Attendee:
    """Calendar event attendee."""
    email: str
    name: Optional[str] = None
    status: AttendeeStatus = AttendeeStatus.NEEDS_ACTION
    is_organizer: bool = False
    is_optional: bool = False
    comment: Optional[str] = None


@dataclass
class TimeSlot:
    """Time slot for calendar events."""
    start_time: datetime
    end_time: datetime
    
    def duration_minutes(self) -> int:
        """Get duration of time slot in minutes."""
        delta = self.end_time - self.start_time
        return int(delta.total_seconds() / 60)
    
    def overlaps(self, other: 'TimeSlot') -> bool:
        """Check if this time slot overlaps with another."""
        return (
            (self.start_time < other.end_time) and
            (self.end_time > other.start_time)
        )
    
    def contains(self, time: datetime) -> bool:
        """Check if this time slot contains a specific time."""
        return self.start_time <= time < self.end_time
    
    def as_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat()
        }


@dataclass
class CalendarEvent:
    """Calendar event representation."""
    event_id: str
    title: str
    start_time: datetime
    end_time: datetime
    calendar_id: str
    description: Optional[str] = None
    location: Optional[str] = None
    attendees: List[Attendee] = field(default_factory=list)
    status: EventStatus = EventStatus.CONFIRMED
    privacy: EventPrivacy = EventPrivacy.PUBLIC
    priority: EventPriority = EventPriority.NORMAL
    is_recurring: bool = False
    recurrence_frequency: Optional[RecurrenceFrequency] = None
    recurrence_end_date: Optional[date] = None
    recurrence_count: Optional[int] = None
    recurrence_days: Optional[List[int]] = None  # 0=Monday, 6=Sunday
    color: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    url: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "event_id": self.event_id,
            "title": self.title,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "calendar_id": self.calendar_id,
            "description": self.description,
            "location": self.location,
            "attendees": [
                {
                    "email": a.email,
                    "name": a.name,
                    "status": a.status.value if a.status else None,
                    "is_organizer": a.is_organizer,
                    "is_optional": a.is_optional
                }
                for a in self.attendees
            ],
            "status": self.status.value if self.status else None,
            "privacy": self.privacy.value if self.privacy else None,
            "priority": self.priority.value if self.priority else None,
            "is_recurring": self.is_recurring,
            "recurrence_frequency": self.recurrence_frequency.value if self.recurrence_frequency else None,
            "recurrence_end_date": self.recurrence_end_date.isoformat() if self.recurrence_end_date else None,
            "recurrence_count": self.recurrence_count,
            "recurrence_days": self.recurrence_days,
            "color": self.color,
            "tags": self.tags,
            "url": self.url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalendarEvent':
        """Create event from dictionary data."""
        # Process attendees
        attendees = []
        for a_data in data.get("attendees", []):
            status_value = a_data.get("status")
            status = AttendeeStatus(status_value) if status_value else AttendeeStatus.NEEDS_ACTION
            
            attendee = Attendee(
                email=a_data.get("email"),
                name=a_data.get("name"),
                status=status,
                is_organizer=a_data.get("is_organizer", False),
                is_optional=a_data.get("is_optional", False),
                comment=a_data.get("comment")
            )
            attendees.append(attendee)
        
        # Process other enum fields
        status_value = data.get("status")
        status = EventStatus(status_value) if status_value else EventStatus.CONFIRMED
        
        privacy_value = data.get("privacy")
        privacy = EventPrivacy(privacy_value) if privacy_value else EventPrivacy.PUBLIC
        
        priority_value = data.get("priority")
        priority = EventPriority(priority_value) if priority_value else EventPriority.NORMAL
        
        recurrence_freq_value = data.get("recurrence_frequency")
        recurrence_frequency = RecurrenceFrequency(recurrence_freq_value) if recurrence_freq_value else None
        
        # Process date/time fields
        start_time = date_parser.parse(data.get("start_time"))
        end_time = date_parser.parse(data.get("end_time"))
        
        recurrence_end_date = None
        if data.get("recurrence_end_date"):
            recurrence_end_date_str = data.get("recurrence_end_date")
            recurrence_end_date = date_parser.parse(recurrence_end_date_str).date()
            
        created_at = None
        if data.get("created_at"):
            created_at = date_parser.parse(data.get("created_at"))
            
        updated_at = None
        if data.get("updated_at"):
            updated_at = date_parser.parse(data.get("updated_at"))
            
        return cls(
            event_id=data.get("event_id"),
            title=data.get("title"),
            start_time=start_time,
            end_time=end_time,
            calendar_id=data.get("calendar_id"),
            description=data.get("description"),
            location=data.get("location"),
            attendees=attendees,
            status=status,
            privacy=privacy,
            priority=priority,
            is_recurring=data.get("is_recurring", False),
            recurrence_frequency=recurrence_frequency,
            recurrence_end_date=recurrence_end_date,
            recurrence_count=data.get("recurrence_count"),
            recurrence_days=data.get("recurrence_days"),
            color=data.get("color"),
            tags=data.get("tags", []),
            url=data.get("url"),
            created_at=created_at,
            updated_at=updated_at
        )


@dataclass
class CalendarInfo:
    """Information about a calendar."""
    calendar_id: str
    name: str
    description: Optional[str] = None
    owner: Optional[str] = None
    is_primary: bool = False
    color: Optional[str] = None
    timezone: Optional[str] = None
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "calendar_id": self.calendar_id,
            "name": self.name,
            "description": self.description,
            "owner": self.owner,
            "is_primary": self.is_primary,
            "color": self.color,
            "timezone": self.timezone
        }


@dataclass
class AvailabilityInfo:
    """User availability information."""
    email: str
    date: date
    busy_slots: List[TimeSlot] = field(default_factory=list)
    working_hours: Optional[Tuple[datetime, datetime]] = None
    preferred_meeting_times: List[Tuple[datetime, datetime]] = field(default_factory=list)
    
    def get_free_slots(self, 
                       min_duration_minutes: int = 30,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> List[TimeSlot]:
        """
        Calculate free time slots based on busy slots and working hours.
        
        Args:
            min_duration_minutes: Minimum length of free slot in minutes
            start_time: Start of period to consider (defaults to start of working hours)
            end_time: End of period to consider (defaults to end of working hours)
            
        Returns:
            List of free time slots
        """
        # Set default time range to working hours if available
        if not start_time and not end_time and self.working_hours:
            start_time = self.working_hours[0]
            end_time = self.working_hours[1]
        elif not start_time:
            # Default to 9 AM on the specified date
            start_time = datetime.combine(self.date, datetime.min.time().replace(hour=9))
        elif not end_time:
            # Default to 5 PM on the specified date
            end_time = datetime.combine(self.date, datetime.min.time().replace(hour=17))
            
        # Sort busy slots by start time
        sorted_busy = sorted(self.busy_slots, key=lambda x: x.start_time)
        
        # Combine overlapping busy slots
        combined_busy = []
        for slot in sorted_busy:
            # Skip slots outside the time range
            if slot.end_time <= start_time or slot.start_time >= end_time:
                continue
                
            # Adjust slots that extend beyond the time range
            slot_start = max(slot.start_time, start_time)
            slot_end = min(slot.end_time, end_time)
            
            if not combined_busy:
                combined_busy.append(TimeSlot(slot_start, slot_end))
            else:
                last_slot = combined_busy[-1]
                
                # If overlaps with the last slot, merge them
                if slot_start <= last_slot.end_time:
                    last_slot.end_time = max(last_slot.end_time, slot_end)
                else:
                    combined_busy.append(TimeSlot(slot_start, slot_end))
        
        # Create free slots from gaps between busy slots
        free_slots = []
        current_time = start_time
        
        for busy_slot in combined_busy:
            # If there's a gap before this busy slot, add a free slot
            if current_time < busy_slot.start_time:
                duration = (busy_slot.start_time - current_time).total_seconds() / 60
                if duration >= min_duration_minutes:
                    free_slots.append(TimeSlot(current_time, busy_slot.start_time))
            
            # Move current time to end of busy slot
            current_time = busy_slot.end_time
            
        # Add final free slot if there's time after the last busy slot
        if current_time < end_time:
            duration = (end_time - current_time).total_seconds() / 60
            if duration >= min_duration_minutes:
                free_slots.append(TimeSlot(current_time, end_time))
                
        return free_slots
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "email": self.email,
            "date": self.date.isoformat(),
            "busy_slots": [slot.as_dict() for slot in self.busy_slots],
            "working_hours": {
                "start": self.working_hours[0].isoformat() if self.working_hours else None,
                "end": self.working_hours[1].isoformat() if self.working_hours else None
            } if self.working_hours else None,
            "preferred_meeting_times": [
                {"start": start.isoformat(), "end": end.isoformat()}
                for start, end in self.preferred_meeting_times
            ]
        }


class CalendarServiceAdapter:
    """
    Adapter for connecting calendar services to the ApiGatewaySystem.
    
    This adapter enables interaction with calendar systems through a standardized API,
    supporting event creation, scheduling, availability checking, and optimal time finding.
    """
    
    def __init__(
        self,
        api_gateway: ApiGatewaySystem,
        calendar_config: CalendarConfig,
        principle_engine: Optional[PrincipleEngine] = None,
        agent_id: str = "calendar-agent",
    ):
        """
        Initialize the calendar service adapter.
        
        Args:
            api_gateway: ApiGatewaySystem to extend
            calendar_config: Calendar configuration
            principle_engine: Optional principle engine for decision-making
            agent_id: ID of the agent using the adapter
        """
        self.api_gateway = api_gateway
        self.calendar_config = calendar_config
        self.principle_engine = principle_engine
        self.agent_id = agent_id
        
        # Create authentication configuration based on auth_type
        if calendar_config.auth_type == "api_key":
            self.auth_config = AuthConfig(
                auth_type=AuthType.API_KEY,
                credentials={"api_key": calendar_config.api_key}
            )
        elif calendar_config.auth_type == "oauth":
            self.auth_config = AuthConfig(
                auth_type=AuthType.OAUTH2,
                credentials={
                    "client_id": calendar_config.client_id,
                    "client_secret": calendar_config.client_secret,
                    "access_token": calendar_config.access_token,
                    "refresh_token": calendar_config.refresh_token
                }
            )
        else:
            self.auth_config = AuthConfig(auth_type=AuthType.NONE)
            
        # Create API configuration
        self._register_calendar_api()
        
        # Cache for calendar data
        self.event_cache: Dict[str, List[CalendarEvent]] = {}
        self.calendar_cache: Dict[str, CalendarInfo] = {}
        self.last_cache_update = datetime.min
        
        # Track operations for auditing
        self.operation_history: List[Dict[str, Any]] = []
        
        logger.info(f"CalendarServiceAdapter initialized")
    
    def _register_calendar_api(self) -> None:
        """Register the calendar API with the ApiGatewaySystem."""
        # Create endpoints for calendar operations
        endpoints = {
            "list_calendars": EndpointConfig(
                name="list_calendars",
                url="/calendars",
                method=HttpMethod.GET,
                auth=self.auth_config,
                description="List available calendars"
            ),
            "get_events": EndpointConfig(
                name="get_events",
                url="/calendars/{calendar_id}/events",
                method=HttpMethod.GET,
                auth=self.auth_config,
                description="Get events from calendar"
            ),
            "create_event": EndpointConfig(
                name="create_event",
                url="/calendars/{calendar_id}/events",
                method=HttpMethod.POST,
                auth=self.auth_config,
                description="Create a new event"
            ),
            "update_event": EndpointConfig(
                name="update_event",
                url="/calendars/{calendar_id}/events/{event_id}",
                method=HttpMethod.PUT,
                auth=self.auth_config,
                description="Update an existing event"
            ),
            "delete_event": EndpointConfig(
                name="delete_event",
                url="/calendars/{calendar_id}/events/{event_id}",
                method=HttpMethod.DELETE,
                auth=self.auth_config,
                description="Delete an event"
            ),
            "check_availability": EndpointConfig(
                name="check_availability",
                url="/availability",
                method=HttpMethod.GET,
                auth=self.auth_config,
                description="Check user availability"
            ),
            "find_meeting_time": EndpointConfig(
                name="find_meeting_time",
                url="/find-time",
                method=HttpMethod.POST,
                auth=self.auth_config,
                description="Find optimal meeting time"
            )
        }
        
        # Create API configuration
        calendar_api_config = ApiConfig(
            name="calendar_service",
            base_url=self.calendar_config.calendar_api_url,
            auth=self.auth_config,
            endpoints=endpoints,
            description="Calendar service API",
            tags=["calendar", "scheduling"],
            version="1.0"
        )
        
        # Register the API with the gateway
        self.api_gateway.register_api(calendar_api_config)
        
        logger.info(f"Calendar API registered with API Gateway")
    
    async def execute_operation(
        self,
        operation: CalendarOperation,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a calendar operation.
        
        Args:
            operation: Calendar operation to execute
            params: Parameters for the operation
            
        Returns:
            Result of the operation
        """
        try:
            # Track operation start time for history
            start_time = time.time()
            
            # Apply principle evaluation if available
            approved = True
            evaluation_result = None
            if self.principle_engine:
                approved, evaluation_result = await self._evaluate_with_principles(operation, params)
                
                if not approved:
                    logger.warning(f"Operation {operation.value} rejected by principle evaluation")
                    return {
                        "success": False,
                        "error": "Operation rejected by principle evaluation",
                        "principle_evaluation": evaluation_result
                    }
            
            # Execute operation
            result = await self._dispatch_operation(operation, params)
            
            # Track in operation history
            elapsed_time = time.time() - start_time
            self.operation_history.append({
                "operation": operation.value,
                "params": params,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": result.get("success", False),
                "duration_ms": int(elapsed_time * 1000),
                "principle_evaluation": evaluation_result
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing calendar operation {operation.value}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _dispatch_operation(
        self,
        operation: CalendarOperation,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Dispatch a calendar operation to the appropriate handler.
        
        Args:
            operation: Calendar operation to execute
            params: Parameters for the operation
            
        Returns:
            Result of the operation
        """
        # Dispatch to the appropriate handler
        if operation == CalendarOperation.LIST_CALENDARS:
            return await self._list_calendars(params)
        elif operation == CalendarOperation.GET_EVENTS:
            return await self._get_events(params)
        elif operation == CalendarOperation.CREATE_EVENT:
            return await self._create_event(params)
        elif operation == CalendarOperation.UPDATE_EVENT:
            return await self._update_event(params)
        elif operation == CalendarOperation.DELETE_EVENT:
            return await self._delete_event(params)
        elif operation == CalendarOperation.CHECK_AVAILABILITY:
            return await self._check_availability(params)
        elif operation == CalendarOperation.FIND_MEETING_TIME:
            return await self._find_meeting_time(params)
        elif operation == CalendarOperation.SUGGEST_LOCATION:
            return await self._suggest_location(params)
        elif operation == CalendarOperation.GET_ATTENDEES:
            return await self._get_attendees(params)
        elif operation == CalendarOperation.SEND_INVITATION:
            return await self._send_invitation(params)
        elif operation == CalendarOperation.RESPOND_TO_INVITATION:
            return await self._respond_to_invitation(params)
        else:
            raise ValueError(f"Unsupported calendar operation: {operation}")
    
    async def _evaluate_with_principles(
        self,
        operation: CalendarOperation,
        params: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate an operation against principles.
        
        Args:
            operation: Calendar operation to evaluate
            params: Parameters for the operation
            
        Returns:
            Tuple of (approved, evaluation_result)
        """
        if not self.principle_engine:
            return True, None
        
        try:
            # Create base evaluation request
            request = PrincipleEvaluationRequest(
                content=f"Calendar operation: {operation.value}",
                context={
                    "operation": operation.value,
                    "parameters": params,
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "operation_type": "calendar",
                    "provider": self.calendar_config.calendar_api_url
                }
            )
            
            # Enhance context based on operation type
            self._enhance_principle_context(operation, params, request)
            
            # Evaluate against principles
            result = await self.principle_engine.evaluate_content(request)
            
            # Check if operation is approved (with a configurable threshold)
            threshold = 0.7  # Could be moved to a configuration setting
            score = result.get("score", 0)
            approved = score >= threshold
            
            # Log principle evaluation results
            if not approved:
                self.logger.warning(
                    f"Calendar operation {operation.value} rejected by principle evaluation. "
                    f"Score: {score:.2f} (threshold: {threshold:.2f})"
                )
                for principle in result.get("violated_principles", []):
                    self.logger.warning(f"Violated principle: {principle}")
            
            return approved, result
            
        except Exception as e:
            # Log error but default to permissive behavior to prevent complete failure
            self.logger.error(f"Error during principle evaluation: {e}")
            # Return True with error information in the result
            return True, {
                "error": f"Principle evaluation error: {str(e)}",
                "score": 1.0,  # Default to passing when evaluation fails
                "fallback": True,
                "evaluation_status": "error"
            }
    
    def _enhance_principle_context(
        self,
        operation: CalendarOperation,
        params: Dict[str, Any],
        request: PrincipleEvaluationRequest
    ) -> None:
        """
        Enhance principle evaluation context with operation-specific details.
        
        Args:
            operation: Calendar operation
            params: Operation parameters
            request: Principle evaluation request to enhance
        """
        # Add common time-related context for all operations
        current_time = datetime.now(timezone.utc)
        request.context["current_time"] = current_time.isoformat()
        request.context["current_day_of_week"] = current_time.strftime("%A")
        request.context["current_hour"] = current_time.hour
        
        # Add operation-specific context
        if operation == CalendarOperation.CREATE_EVENT:
            # Add detailed event context
            start_time = params.get("start_time", "")
            end_time = params.get("end_time", "")
            
            # Parse times if available
            try:
                if start_time:
                    start_dt = date_parser.parse(start_time)
                    request.context["event_start_hour"] = start_dt.hour
                    request.context["event_day_of_week"] = start_dt.strftime("%A")
                    request.context["is_weekend"] = start_dt.weekday() >= 5  # 5=Saturday, 6=Sunday
                
                if start_time and end_time:
                    start_dt = date_parser.parse(start_time)
                    end_dt = date_parser.parse(end_time)
                    duration = (end_dt - start_dt).total_seconds() / 60
                    request.context["event_duration_minutes"] = duration
                    request.context["is_long_meeting"] = duration > 90
            except Exception as e:
                self.logger.warning(f"Error parsing event times for principle context: {e}")
            
            # Add event details
            request.context.update({
                "event_title": params.get("title", ""),
                "event_description": params.get("description", ""),
                "event_attendees": params.get("attendees", []),
                "event_attendee_count": len(params.get("attendees", [])),
                "event_location": params.get("location", ""),
                "event_privacy": params.get("privacy", EventPrivacy.PUBLIC.value),
                "event_priority": params.get("priority", EventPriority.NORMAL.value),
                "event_start_time": start_time,
                "event_end_time": end_time,
                "is_recurring": params.get("is_recurring", False)
            })
        
        elif operation == CalendarOperation.UPDATE_EVENT:
            # Add context about what's being updated
            updates = params.get("updates", {})
            request.context.update({
                "event_id": params.get("event_id", ""),
                "is_time_update": "start_time" in updates or "end_time" in updates,
                "is_attendee_update": "attendees" in updates,
                "is_content_update": "title" in updates or "description" in updates,
                "update_fields": list(updates.keys())
            })
        
        elif operation == CalendarOperation.SEND_INVITATION:
            # Add invitation context
            request.context.update({
                "invitation_event_id": params.get("event_id", ""),
                "invitation_recipients": params.get("recipients", []),
                "invitation_recipient_count": len(params.get("recipients", [])),
                "invitation_message": params.get("message", ""),
                "is_mass_invitation": len(params.get("recipients", [])) > 10
            })
            
        elif operation == CalendarOperation.CHECK_AVAILABILITY:
            # Add availability check context
            request.context.update({
                "emails": params.get("emails", []),
                "user_count": len(params.get("emails", [])),
                "date": params.get("date", ""),
                "time_range_specified": bool(params.get("start_time")) and bool(params.get("end_time"))
            })
    
    # Calendar operation implementations
    
    async def _list_calendars(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        List available calendars.
        
        Args:
            params: Parameters for the operation
                
        Returns:
            Result of the operation with calendars
        """
        # TODO: In a real implementation, this would call the calendar API
        # For this example, we'll create some sample calendars
        
        # Check if we can use cached data
        cache_age = (datetime.now() - self.last_cache_update).total_seconds() / 60
        if self.calendar_cache and cache_age < self.calendar_config.cache_ttl_minutes:
            return {
                "success": True,
                "calendars": [cal.as_dict() for cal in self.calendar_cache.values()],
                "count": len(self.calendar_cache),
                "from_cache": True
            }
        
        # Sample calendars
        calendars = [
            CalendarInfo(
                calendar_id="primary",
                name="My Calendar",
                description="Primary calendar",
                owner=f"{self.agent_id}@example.com",
                is_primary=True,
                color="#4285F4",
                timezone=self.calendar_config.timezone
            ),
            CalendarInfo(
                calendar_id="work",
                name="Work Calendar",
                description="Work-related events",
                owner=f"{self.agent_id}@example.com",
                is_primary=False,
                color="#0B8043",
                timezone=self.calendar_config.timezone
            ),
            CalendarInfo(
                calendar_id="personal",
                name="Personal Calendar",
                description="Personal events",
                owner=f"{self.agent_id}@example.com",
                is_primary=False,
                color="#D50000",
                timezone=self.calendar_config.timezone
            )
        ]
        
        # Update cache
        self.calendar_cache = {cal.calendar_id: cal for cal in calendars}
        self.last_cache_update = datetime.now()
        
        return {
            "success": True,
            "calendars": [cal.as_dict() for cal in calendars],
            "count": len(calendars),
            "from_cache": False
        }
    
    async def _get_events(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get events from a calendar.
        
        Args:
            params: Parameters for the operation
                - calendar_id: ID of the calendar to get events from
                - start_date: Start date for the events to retrieve
                - end_date: End date for the events to retrieve
                - max_results: Maximum number of events to retrieve
                - include_recurring: Whether to include recurring events
                
        Returns:
            Result of the operation with events
        """
        try:
            # Extract parameters
            calendar_id = params.get("calendar_id", self.calendar_config.default_calendar_id)
            start_date_str = params.get("start_date")
            end_date_str = params.get("end_date")
            max_results = params.get("max_results", self.calendar_config.max_results)
            include_recurring = params.get("include_recurring", True)
            
            # Parse dates
            start_date = None
            end_date = None
            
            if start_date_str:
                start_date = date_parser.parse(start_date_str).date()
            else:
                start_date = datetime.now().date()
                
            if end_date_str:
                end_date = date_parser.parse(end_date_str).date()
            else:
                end_date = start_date + timedelta(days=7)  # Default to one week ahead
            
            # Check if we can use cached data
            cache_key = f"{calendar_id}_{start_date}_{end_date}_{include_recurring}"
            cache_age = (datetime.now() - self.last_cache_update).total_seconds() / 60
            
            if cache_key in self.event_cache and cache_age < self.calendar_config.cache_ttl_minutes:
                return {
                    "success": True,
                    "events": [event.as_dict() for event in self.event_cache[cache_key]],
                    "count": len(self.event_cache[cache_key]),
                    "from_cache": True
                }
                
            # TODO: In a real implementation, this would call the calendar API
            # For this example, we'll create some sample events
            events = []
            
            # Create events for each day in the range
            current_date = start_date
            while current_date <= end_date:
                # Create events for this day
                day_events = self._generate_sample_events(
                    calendar_id=calendar_id,
                    day=current_date,
                    count=2  # 2 events per day
                )
                events.extend(day_events)
                
                # Move to next day
                current_date += timedelta(days=1)
                
            # Update cache
            self.event_cache = {**self.event_cache, cache_key: events}
            self.last_cache_update = datetime.now()
            
            return {
                "success": True,
                "events": [event.as_dict() for event in events],
                "count": len(events),
                "from_cache": False
            }
            
        except Exception as e:
            logger.error(f"Error getting events: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_sample_events(self, calendar_id: str, day: date, count: int = 2) -> List[CalendarEvent]:
        """
        Generate sample events for testing.
        
        Args:
            calendar_id: Calendar ID to generate events for
            day: Day to generate events for
            count: Number of events to generate
            
        Returns:
            List of sample CalendarEvent objects
        """
        events = []
        
        # Create events spread throughout the day
        if count > 0:
            # Morning meeting
            morning_start = datetime.combine(day, datetime.min.time().replace(hour=9, minute=30))
            morning_end = morning_start + timedelta(minutes=60)
            
            morning_event = CalendarEvent(
                event_id=f"{calendar_id}-{day.isoformat()}-morning",
                title="Morning Status Meeting",
                description="Daily team status update",
                start_time=morning_start,
                end_time=morning_end,
                calendar_id=calendar_id,
                location="Conference Room A",
                attendees=[
                    Attendee(
                        email=f"{self.agent_id}@example.com",
                        name=self.agent_id,
                        is_organizer=True,
                        status=AttendeeStatus.ACCEPTED
                    ),
                    Attendee(
                        email="team-lead@example.com",
                        name="Team Lead",
                        status=AttendeeStatus.ACCEPTED
                    ),
                    Attendee(
                        email="manager@example.com",
                        name="Manager",
                        status=AttendeeStatus.TENTATIVE
                    )
                ],
                status=EventStatus.CONFIRMED,
                priority=EventPriority.HIGH,
                created_at=datetime.now() - timedelta(days=7),
                updated_at=datetime.now() - timedelta(days=2)
            )
            events.append(morning_event)
        
        if count > 1:
            # Afternoon meeting
            afternoon_start = datetime.combine(day, datetime.min.time().replace(hour=14, minute=0))
            afternoon_end = afternoon_start + timedelta(minutes=90)
            
            afternoon_event = CalendarEvent(
                event_id=f"{calendar_id}-{day.isoformat()}-afternoon",
                title="Project Planning",
                description="Quarterly project planning session",
                start_time=afternoon_start,
                end_time=afternoon_end,
                calendar_id=calendar_id,
                location="Main Conference Room",
                attendees=[
                    Attendee(
                        email=f"{self.agent_id}@example.com",
                        name=self.agent_id,
                        status=AttendeeStatus.ACCEPTED
                    ),
                    Attendee(
                        email="product@example.com",
                        name="Product Manager",
                        is_organizer=True,
                        status=AttendeeStatus.ACCEPTED
                    ),
                    Attendee(
                        email="designer@example.com",
                        name="Designer",
                        status=AttendeeStatus.ACCEPTED
                    ),
                    Attendee(
                        email="engineer@example.com",
                        name="Engineer",
                        status=AttendeeStatus.NEEDS_ACTION
                    )
                ],
                status=EventStatus.CONFIRMED,
                priority=EventPriority.NORMAL,
                created_at=datetime.now() - timedelta(days=14),
                updated_at=datetime.now() - timedelta(days=1)
            )
            events.append(afternoon_event)
            
        if count > 2:
            # Evening reminder
            evening_start = datetime.combine(day, datetime.min.time().replace(hour=18, minute=0))
            evening_end = evening_start + timedelta(minutes=30)
            
            evening_event = CalendarEvent(
                event_id=f"{calendar_id}-{day.isoformat()}-evening",
                title="Submit Daily Report",
                description="Reminder to submit daily progress report",
                start_time=evening_start,
                end_time=evening_end,
                calendar_id=calendar_id,
                attendees=[
                    Attendee(
                        email=f"{self.agent_id}@example.com",
                        name=self.agent_id,
                        is_organizer=True,
                        status=AttendeeStatus.ACCEPTED
                    )
                ],
                status=EventStatus.CONFIRMED,
                priority=EventPriority.LOW,
                created_at=datetime.now() - timedelta(days=30),
                updated_at=datetime.now() - timedelta(days=30),
                is_recurring=True,
                recurrence_frequency=RecurrenceFrequency.DAILY,
                recurrence_end_date=(day + timedelta(days=30)).date()
            )
            events.append(evening_event)
            
        return events
    
    async def _create_event(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new calendar event.
        
        Args:
            params: Parameters for the operation
                - calendar_id: ID of the calendar to create the event in
                - title: Event title
                - start_time: Event start time (ISO format)
                - end_time: Event end time (ISO format)
                - description: Optional event description
                - location: Optional event location
                - attendees: Optional list of attendees
                - privacy: Optional privacy level
                - priority: Optional priority level
                - is_recurring: Optional flag for recurring events
                - ... other event properties
                
        Returns:
            Result of the operation with created event
        """
        try:
            # Extract required parameters
            calendar_id = params.get("calendar_id", self.calendar_config.default_calendar_id)
            title = params.get("title")
            start_time_str = params.get("start_time")
            end_time_str = params.get("end_time")
            
            # Validate required parameters
            if not all([title, start_time_str, end_time_str]):
                return {
                    "success": False,
                    "error": "Missing required parameters: title, start_time, end_time"
                }
                
            # Parse dates
            start_time = date_parser.parse(start_time_str)
            end_time = date_parser.parse(end_time_str)
            
            # Generate event ID
            event_id = f"{calendar_id}-{uuid.uuid4().hex[:12]}"
            
            # Process attendees
            attendees = []
            for attendee_data in params.get("attendees", []):
                email = attendee_data.get("email")
                name = attendee_data.get("name")
                is_optional = attendee_data.get("is_optional", False)
                is_organizer = attendee_data.get("is_organizer", False)
                status_value = attendee_data.get("status", AttendeeStatus.NEEDS_ACTION.value)
                
                try:
                    status = AttendeeStatus(status_value) if status_value else AttendeeStatus.NEEDS_ACTION
                except ValueError:
                    status = AttendeeStatus.NEEDS_ACTION
                
                if email:
                    attendees.append(Attendee(
                        email=email,
                        name=name,
                        status=status,
                        is_organizer=is_organizer,
                        is_optional=is_optional
                    ))
            
            # Process other enum fields
            status_value = params.get("status")
            try:
                status = EventStatus(status_value) if status_value else EventStatus.CONFIRMED
            except ValueError:
                status = EventStatus.CONFIRMED
                
            privacy_value = params.get("privacy")
            try:
                privacy = EventPrivacy(privacy_value) if privacy_value else EventPrivacy.PUBLIC
            except ValueError:
                privacy = EventPrivacy.PUBLIC
                
            priority_value = params.get("priority")
            try:
                priority = EventPriority(priority_value) if priority_value else EventPriority.NORMAL
            except ValueError:
                priority = EventPriority.NORMAL
            
            # Create event
            event = CalendarEvent(
                event_id=event_id,
                title=title,
                start_time=start_time,
                end_time=end_time,
                calendar_id=calendar_id,
                description=params.get("description"),
                location=params.get("location"),
                attendees=attendees,
                status=status,
                privacy=privacy,
                priority=priority,
                is_recurring=params.get("is_recurring", False),
                tags=params.get("tags", []),
                color=params.get("color"),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # In a real implementation, this would call the calendar API
            # For this example, we'll just return the created event
            
            return {
                "success": True,
                "event_id": event.event_id,
                "event": event.as_dict()
            }
            
        except Exception as e:
            logger.error(f"Error creating event: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _update_event(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing calendar event.
        
        Args:
            params: Parameters for the operation
                - calendar_id: ID of the calendar containing the event
                - event_id: ID of the event to update
                - updates: Dictionary of fields to update
                
        Returns:
            Result of the operation with updated event
        """
        try:
            # Extract parameters
            calendar_id = params.get("calendar_id", self.calendar_config.default_calendar_id)
            event_id = params.get("event_id")
            
            # Validate required parameters
            if not event_id:
                return {
                    "success": False,
                    "error": "Missing required parameter: event_id"
                }
                
            # In a real implementation, this would call the calendar API
            # For this example, we'll simulate updating an event
            
            # Simulate finding the event (would fetch from API in real implementation)
            # For demo purposes, create a placeholder event
            event = CalendarEvent(
                event_id=event_id,
                title="Original Event Title",
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(hours=1),
                calendar_id=calendar_id,
                created_at=datetime.now() - timedelta(days=1),
                updated_at=datetime.now()
            )
            
            # Apply updates
            updates = params.get("updates", {})
            for key, value in updates.items():
                if hasattr(event, key):
                    # Special handling for date fields
                    if key in ["start_time", "end_time"] and isinstance(value, str):
                        value = date_parser.parse(value)
                    elif key == "recurrence_end_date" and isinstance(value, str):
                        value = date_parser.parse(value).date()
                    # Special handling for enum fields
                    elif key == "status" and isinstance(value, str):
                        try:
                            value = EventStatus(value)
                        except ValueError:
                            continue
                    elif key == "privacy" and isinstance(value, str):
                        try:
                            value = EventPrivacy(value)
                        except ValueError:
                            continue
                    elif key == "priority" and isinstance(value, str):
                        try:
                            value = EventPriority(value)
                        except ValueError:
                            continue
                    elif key == "recurrence_frequency" and isinstance(value, str):
                        try:
                            value = RecurrenceFrequency(value)
                        except ValueError:
                            continue
                            
                    setattr(event, key, value)
            
            # Update the updated_at timestamp
            event.updated_at = datetime.now()
            
            return {
                "success": True,
                "event_id": event.event_id,
                "event": event.as_dict()
            }
            
        except Exception as e:
            logger.error(f"Error updating event: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _delete_event(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delete a calendar event.
        
        Args:
            params: Parameters for the operation
                - calendar_id: ID of the calendar containing the event
                - event_id: ID of the event to delete
                
        Returns:
            Result of the operation
        """
        try:
            # Extract parameters
            calendar_id = params.get("calendar_id", self.calendar_config.default_calendar_id)
            event_id = params.get("event_id")
            
            # Validate required parameters
            if not event_id:
                return {
                    "success": False,
                    "error": "Missing required parameter: event_id"
                }
                
            # In a real implementation, this would call the calendar API
            # For this example, we'll just return success
            
            return {
                "success": True,
                "event_id": event_id,
                "message": f"Event {event_id} deleted successfully from calendar {calendar_id}"
            }
            
        except Exception as e:
            logger.error(f"Error deleting event: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _check_availability(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check user availability.
        
        Args:
            params: Parameters for the operation
                - emails: List of email addresses to check
                - date: Date to check availability for
                - start_time: Optional start of time range
                - end_time: Optional end of time range
                
        Returns:
            Result of the operation with availability information
        """
        try:
            # Extract parameters
            emails = params.get("emails", [])
            date_str = params.get("date")
            start_time_str = params.get("start_time")
            end_time_str = params.get("end_time")
            
            # Validate required parameters
            if not emails or not date_str:
                return {
                    "success": False,
                    "error": "Missing required parameters: emails, date"
                }
                
            # Parse date
            check_date = date_parser.parse(date_str).date()
            
            # Parse times if provided
            start_time = None
            end_time = None
            
            if start_time_str:
                start_time = date_parser.parse(start_time_str)
            else:
                start_time = datetime.combine(check_date, datetime.min.time().replace(hour=9))
                
            if end_time_str:
                end_time = date_parser.parse(end_time_str)
            else:
                end_time = datetime.combine(check_date, datetime.min.time().replace(hour=17))
            
            # In a real implementation, this would call the calendar API
            # For this example, we'll generate sample availability data
            availability_results = []
            
            for email in emails:
                # Generate busy slots (random for this example)
                busy_slots = []
                
                # Morning meeting (9:30-10:30)
                if email.startswith("user1") or email.startswith("manager"):
                    morning_start = datetime.combine(check_date, datetime.min.time().replace(hour=9, minute=30))
                    morning_end = morning_start + timedelta(hours=1)
                    busy_slots.append(TimeSlot(morning_start, morning_end))
                
                # Lunch (12:00-13:00)
                lunch_start = datetime.combine(check_date, datetime.min.time().replace(hour=12))
                lunch_end = lunch_start + timedelta(hours=1)
                busy_slots.append(TimeSlot(lunch_start, lunch_end))
                
                # Afternoon meeting (14:00-15:30)
                if email.startswith("user2") or email.startswith("team"):
                    afternoon_start = datetime.combine(check_date, datetime.min.time().replace(hour=14))
                    afternoon_end = afternoon_start + timedelta(hours=1, minutes=30)
                    busy_slots.append(TimeSlot(afternoon_start, afternoon_end))
                
                # Create availability info
                availability = AvailabilityInfo(
                    email=email,
                    date=check_date,
                    busy_slots=busy_slots,
                    working_hours=(start_time, end_time)
                )
                
                # Calculate free slots
                free_slots = availability.get_free_slots(
                    min_duration_minutes=30,
                    start_time=start_time,
                    end_time=end_time
                )
                
                # Add some preferred meeting times
                preferred_times = []
                if len(free_slots) > 0:
                    # First free slot
                    preferred_times.append((free_slots[0].start_time, free_slots[0].end_time))
                
                if len(free_slots) > 1:
                    # Last free slot
                    preferred_times.append((free_slots[-1].start_time, free_slots[-1].end_time))
                
                # Add to results
                availability_result = availability.as_dict()
                availability_result["free_slots"] = [slot.as_dict() for slot in free_slots]
                availability_results.append(availability_result)
            
            return {
                "success": True,
                "date": check_date.isoformat(),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "availability": availability_results
            }
            
        except Exception as e:
            logger.error(f"Error checking availability: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _find_meeting_time(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find optimal meeting time.
        
        Args:
            params: Parameters for the operation
                - attendees: List of attendee email addresses
                - duration_minutes: Meeting duration in minutes
                - date_range_start: Start of date range to search
                - date_range_end: End of date range to search
                - priority: Priority type for selecting time (availability, preferences, time_of_day)
                
        Returns:
            Result of the operation with suggested meeting times
        """
        try:
            # Extract parameters
            attendees = params.get("attendees", [])
            duration_minutes = params.get("duration_minutes", 60)
            date_range_start_str = params.get("date_range_start")
            date_range_end_str = params.get("date_range_end")
            priority = params.get("priority", "availability")
            
            # Validate required parameters
            if not attendees:
                return {
                    "success": False,
                    "error": "Missing required parameter: attendees"
                }
                
            # Parse dates
            if date_range_start_str:
                date_range_start = date_parser.parse(date_range_start_str).date()
            else:
                date_range_start = datetime.now().date()
                
            if date_range_end_str:
                date_range_end = date_parser.parse(date_range_end_str).date()
            else:
                date_range_end = date_range_start + timedelta(days=7)
            
            # In a real implementation, this would call the calendar API and apply complex
            # scheduling algorithms. For this example, we'll generate sample suggestions.
            suggested_times = []
            
            # Generate suggestions for the next few days
            for day_offset in range((date_range_end - date_range_start).days + 1):
                current_date = date_range_start + timedelta(days=day_offset)
                
                # Morning suggestion (10:00 AM)
                morning_start = datetime.combine(current_date, datetime.min.time().replace(hour=10))
                morning_end = morning_start + timedelta(minutes=duration_minutes)
                
                # Calculate score (random for this example)
                morning_score = 0.85 - (day_offset * 0.05)  # Higher score for closer dates
                
                # Calculate available attendees (random for this example)
                available_attendees = len(attendees)
                if day_offset > 1:
                    # Some conflicts on later days
                    available_attendees = max(1, len(attendees) - (day_offset // 2))
                
                # Add conflicts for some attendees
                conflicts = []
                if available_attendees < len(attendees):
                    for i in range(len(attendees) - available_attendees):
                        if i < len(attendees):
                            conflicts.append({
                                "email": attendees[i],
                                "reason": "Busy during this time"
                            })
                
                # Morning suggestion
                suggested_times.append({
                    "start_time": morning_start.isoformat(),
                    "end_time": morning_end.isoformat(),
                    "score": round(morning_score, 2),
                    "available_attendees": available_attendees,
                    "total_attendees": len(attendees),
                    "conflicts": conflicts,
                    "location_suggestions": ["Conference Room A", "Virtual Meeting"]
                })
                
                # Afternoon suggestion (2:00 PM)
                afternoon_start = datetime.combine(current_date, datetime.min.time().replace(hour=14))
                afternoon_end = afternoon_start + timedelta(minutes=duration_minutes)
                
                # Different score for afternoon
                afternoon_score = 0.75 - (day_offset * 0.05)
                
                # Different availability pattern
                afternoon_available = len(attendees)
                if current_date.weekday() in [2, 4]:  # Wednesday and Friday
                    afternoon_available = max(1, len(attendees) - 2)
                
                # Add conflicts for some attendees
                afternoon_conflicts = []
                if afternoon_available < len(attendees):
                    for i in range(len(attendees) - afternoon_available):
                        if i < len(attendees):
                            afternoon_conflicts.append({
                                "email": attendees[i],
                                "reason": "Scheduled meeting conflict"
                            })
                
                # Afternoon suggestion
                suggested_times.append({
                    "start_time": afternoon_start.isoformat(),
                    "end_time": afternoon_end.isoformat(),
                    "score": round(afternoon_score, 2),
                    "available_attendees": afternoon_available,
                    "total_attendees": len(attendees),
                    "conflicts": afternoon_conflicts,
                    "location_suggestions": ["Conference Room B", "Virtual Meeting"]
                })
                
                # Only generate a few suggestions to avoid too much data
                if len(suggested_times) >= 6:
                    break
            
            # Sort by score (highest first)
            suggested_times.sort(key=lambda x: x["score"], reverse=True)
            
            return {
                "success": True,
                "date_range_start": date_range_start.isoformat(),
                "date_range_end": date_range_end.isoformat(),
                "duration_minutes": duration_minutes,
                "attendees": attendees,
                "priority": priority,
                "suggested_times": suggested_times
            }
            
        except Exception as e:
            logger.error(f"Error finding meeting time: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _suggest_location(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest meeting location.
        
        Args:
            params: Parameters for the operation
                - attendees: List of attendee email addresses
                - start_time: Meeting start time
                - end_time: Meeting end time
                - meeting_type: Type of meeting (in_person, virtual, hybrid)
                
        Returns:
            Result of the operation with suggested locations
        """
        try:
            # Extract parameters
            attendees = params.get("attendees", [])
            start_time_str = params.get("start_time")
            meeting_type = params.get("meeting_type", "in_person")
            
            # In a real implementation, this would apply location selection algorithms
            # For this example, we'll return sample suggestions
            
            suggested_locations = []
            
            if meeting_type == "virtual" or meeting_type == "hybrid":
                suggested_locations.append({
                    "name": "Virtual Meeting Room",
                    "type": "virtual",
                    "url": "https://meet.example.com/abcdef",
                    "score": 0.95,
                    "features": ["screen_sharing", "recording", "whiteboard"]
                })
            
            if meeting_type == "in_person" or meeting_type == "hybrid":
                suggested_locations.extend([
                    {
                        "name": "Conference Room A",
                        "type": "room",
                        "building": "Main Office",
                        "capacity": 8,
                        "score": 0.85,
                        "features": ["projector", "whiteboard", "video_conference"]
                    },
                    {
                        "name": "Conference Room B",
                        "type": "room",
                        "building": "Main Office",
                        "capacity": 12,
                        "score": 0.80,
                        "features": ["large_display", "whiteboard", "video_conference"]
                    },
                    {
                        "name": "Meeting Room 101",
                        "type": "room",
                        "building": "East Building",
                        "capacity": 6,
                        "score": 0.75,
                        "features": ["whiteboard"]
                    }
                ])
            
            return {
                "success": True,
                "meeting_type": meeting_type,
                "attendee_count": len(attendees),
                "suggested_locations": suggested_locations
            }
            
        except Exception as e:
            logger.error(f"Error suggesting location: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_attendees(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get potential attendees for a meeting.
        
        Args:
            params: Parameters for the operation
                - topic: Meeting topic or keywords
                - roles: Required roles for attendees
                - team: Team name for filtering
                - max_results: Maximum number of attendees to suggest
                
        Returns:
            Result of the operation with suggested attendees
        """
        try:
            # Extract parameters
            topic = params.get("topic", "")
            roles = params.get("roles", [])
            team = params.get("team")
            max_results = params.get("max_results", 10)
            
            # In a real implementation, this would query organizational data
            # For this example, we'll return sample suggestions
            
            # Generate sample attendees based on topic/roles/team
            suggested_attendees = []
            
            # Some example mappings for demonstration
            topic_experts = {
                "project": ["project.manager@example.com", "senior.pm@example.com"],
                "design": ["ui.designer@example.com", "ux.lead@example.com"],
                "development": ["lead.dev@example.com", "senior.engineer@example.com"],
                "marketing": ["marketing.lead@example.com", "content.manager@example.com"],
                "sales": ["sales.director@example.com", "account.manager@example.com"],
                "finance": ["finance.manager@example.com", "accountant@example.com"]
            }
            
            role_people = {
                "manager": ["team.manager@example.com", "senior.manager@example.com"],
                "developer": ["developer1@example.com", "developer2@example.com"],
                "designer": ["designer@example.com", "ui.specialist@example.com"],
                "stakeholder": ["stakeholder@example.com", "executive@example.com"],
                "analyst": ["data.analyst@example.com", "business.analyst@example.com"]
            }
            
            team_members = {
                "engineering": ["engineer1@example.com", "engineer2@example.com", "tech.lead@example.com"],
                "design": ["designer1@example.com", "designer2@example.com"],
                "product": ["product.manager@example.com", "product.owner@example.com"],
                "marketing": ["marketer1@example.com", "marketer2@example.com"],
                "sales": ["sales1@example.com", "sales2@example.com"]
            }
            
            # Add topic-based attendees
            for key, experts in topic_experts.items():
                if key in topic.lower():
                    for expert in experts:
                        suggested_attendees.append({
                            "email": expert,
                            "name": expert.split("@")[0].replace(".", " ").title(),
                            "role": key.title() + " Expert",
                            "reason": f"Expert in {key}",
                            "required": True,
                            "score": 0.9
                        })
            
            # Add role-based attendees
            for role in roles:
                if role in role_people:
                    for person in role_people[role]:
                        if not any(a["email"] == person for a in suggested_attendees):
                            suggested_attendees.append({
                                "email": person,
                                "name": person.split("@")[0].replace(".", " ").title(),
                                "role": role.title(),
                                "reason": f"Assigned {role} role",
                                "required": True,
                                "score": 0.85
                            })
            
            # Add team-based attendees
            if team and team in team_members:
                for member in team_members[team]:
                    if not any(a["email"] == member for a in suggested_attendees):
                        suggested_attendees.append({
                            "email": member,
                            "name": member.split("@")[0].replace(".", " ").title(),
                            "role": f"{team.title()} Team Member",
                            "reason": f"Member of {team} team",
                            "required": False,
                            "score": 0.8
                        })
            
            # Limit results
            suggested_attendees = suggested_attendees[:max_results]
            
            return {
                "success": True,
                "topic": topic,
                "roles": roles,
                "team": team,
                "suggested_attendees": suggested_attendees,
                "count": len(suggested_attendees)
            }
            
        except Exception as e:
            logger.error(f"Error getting attendees: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _send_invitation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send calendar invitation.
        
        Args:
            params: Parameters for the operation
                - event_id: ID of the event to send invitation for
                - recipients: List of recipient email addresses
                - message: Optional message to include with invitation
                
        Returns:
            Result of the operation
        """
        try:
            # Extract parameters
            event_id = params.get("event_id")
            recipients = params.get("recipients", [])
            message = params.get("message", "")
            
            # Validate required parameters
            if not event_id or not recipients:
                return {
                    "success": False,
                    "error": "Missing required parameters: event_id, recipients"
                }
            
            # In a real implementation, this would send invitations via email/API
            # For this example, we'll just return success
            
            return {
                "success": True,
                "event_id": event_id,
                "recipients": recipients,
                "sent_count": len(recipients),
                "message": "Invitations sent successfully"
            }
            
        except Exception as e:
            logger.error(f"Error sending invitation: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _respond_to_invitation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Respond to calendar invitation.
        
        Args:
            params: Parameters for the operation
                - event_id: ID of the event
                - response: Response status (accepted, declined, tentative)
                - attendee_email: Email address of the attendee responding
                - comment: Optional comment with response
                
        Returns:
            Result of the operation
        """
        try:
            # Extract parameters
            event_id = params.get("event_id")
            response_status = params.get("response")
            attendee_email = params.get("attendee_email")
            comment = params.get("comment", "")
            
            # Validate required parameters
            if not all([event_id, response_status, attendee_email]):
                return {
                    "success": False,
                    "error": "Missing required parameters: event_id, response, attendee_email"
                }
                
            # Validate response status
            try:
                status = AttendeeStatus(response_status)
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid response status: {response_status}"
                }
            
            # In a real implementation, this would update the invitation in the calendar API
            # For this example, we'll just return success
            
            return {
                "success": True,
                "event_id": event_id,
                "attendee_email": attendee_email,
                "response": response_status,
                "message": f"Response updated successfully to {response_status}"
            }
            
        except Exception as e:
            logger.error(f"Error responding to invitation: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
