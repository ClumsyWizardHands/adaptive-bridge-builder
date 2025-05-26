#!/usr/bin/env python3
"""
Test module for ProjectOrchestrator

This module provides basic tests for the ProjectOrchestrator extension to verify
its functionality in managing complex, multi-stage projects.
"""

import unittest
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional

from project_orchestrator import (
    ProjectOrchestrator, MilestoneStatus, ResourceType,
    Resource, Milestone, Project, ScheduleEvent, ProjectIssue, StatusUpdate
)
from orchestrator_engine import (
    OrchestratorEngine, TaskType, AgentRole, AgentAvailability,
    DependencyType, TaskDecompositionStrategy
)
from collaborative_task_handler import TaskStatus, TaskPriority
from principle_engine import PrincipleEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TestProjectOrchestrator")


class TestProjectOrchestrator(unittest.TestCase):
    """Test class for ProjectOrchestrator functionality."""
    
    def setUp(self) -> None:
        """Set up test environment before each test."""
        # Create orchestrator engine
        self.orchestrator_engine = OrchestratorEngine(
            agent_id="test-orchestrator",
            storage_dir="test_data/orchestration"
        )
        
        # Register test agents
        self._register_test_agents()
        
        # Create project orchestrator
        self.project_orchestrator = ProjectOrchestrator(
            agent_id="test-orchestrator",
            orchestrator_engine=self.orchestrator_engine,
            storage_dir="test_data/projects"
        )
    
    def _register_test_agents(self) -> None:
        """Register test agents with different capabilities."""
        # Strategy agent
        self.orchestrator_engine.register_agent(
            agent_id="test-strategy-agent",
            roles=[AgentRole.SPECIALIST],
            capabilities=["business_strategy", "market_analysis"],
            specialization={
                TaskType.ANALYSIS: 0.9,
                TaskType.DECISION: 0.8,
            },
            max_load=2
        )
        
        # Research agent
        self.orchestrator_engine.register_agent(
            agent_id="test-research-agent",
            roles=[AgentRole.RESEARCHER],
            capabilities=["market_research", "data_collection"],
            specialization={
                TaskType.RESEARCH: 0.9,
                TaskType.EXTRACTION: 0.8,
            },
            max_load=2
        )
        
        # Analysis agent
        self.orchestrator_engine.register_agent(
            agent_id="test-analysis-agent",
            roles=[AgentRole.ANALYZER],
            capabilities=["data_analysis", "statistical_modeling"],
            specialization={
                TaskType.ANALYSIS: 0.9,
                TaskType.TRANSFORMATION: 0.7,
            },
            max_load=2
        )
    
    def test_create_project(self) -> None:
        """Test creating a project."""
        # Create project
        now = datetime.now(timezone.utc)
        start_date = now.strftime("%Y-%m-%d")
        end_date = (now + timedelta(days=30)).strftime("%Y-%m-%d")
        
        project = self.project_orchestrator.create_project(
            name="Test Project",
            description="A test project for unit testing",
            start_date=start_date,
            end_date=end_date,
            stakeholders=[
                {
                    "id": "test@example.com",
                    "name": "Test User",
                    "role": "tester"
                }
            ],
            tags=["test", "unit_testing"]
        )
        
        # Verify project was created
        self.assertIsNotNone(project)
        self.assertEqual(project.name, "Test Project")
        self.assertEqual(project.status, "planning")
        self.assertEqual(len(project.stakeholders), 1)
        self.assertEqual(len(project.tags), 2)
    
    def test_add_milestone(self) -> None:
        """Test adding a milestone to a project."""
        # Create project
        now = datetime.now(timezone.utc)
        project = self.project_orchestrator.create_project(
            name="Milestone Test Project",
            description="Testing milestone functionality",
            start_date=now.strftime("%Y-%m-%d"),
            end_date=(now + timedelta(days=30)).strftime("%Y-%m-%d")
        )
        
        # Add milestone
        milestone = self.project_orchestrator.add_project_milestone(
            project_id=project.project_id,
            name="Test Milestone",
            description="A test milestone",
            target_date=(now + timedelta(days=15)).strftime("%Y-%m-%d"),
            completion_criteria={
                "required_deliverables": ["test_report"]
            }
        )
        
        # Verify milestone was added
        self.assertIsNotNone(milestone)
        self.assertEqual(milestone.name, "Test Milestone")
        self.assertEqual(milestone.status, MilestoneStatus.NOT_STARTED)
        self.assertIn(milestone.milestone_id, project.milestones)
    
    def test_create_task(self) -> None:
        """Test creating a task in a milestone."""
        # Create project with milestone
        now = datetime.now(timezone.utc)
        project = self.project_orchestrator.create_project(
            name="Task Test Project",
            description="Testing task functionality",
            start_date=now.strftime("%Y-%m-%d"),
            end_date=(now + timedelta(days=30)).strftime("%Y-%m-%d")
        )
        
        milestone = self.project_orchestrator.add_project_milestone(
            project_id=project.project_id,
            name="Test Milestone",
            description="A test milestone",
            target_date=(now + timedelta(days=15)).strftime("%Y-%m-%d")
        )
        
        # Create task
        task = self.project_orchestrator.create_project_task(
            project_id=project.project_id,
            milestone_id=milestone.milestone_id,
            title="Test Task",
            description="A test task for the milestone",
            task_type=TaskType.RESEARCH,
            required_capabilities=["market_research"],
            priority=TaskPriority.HIGH,
            estimated_duration=60  # minutes
        )
        
        # Verify task was created
        self.assertIsNotNone(task)
        self.assertEqual(task.title, "Test Task")
        self.assertEqual(task.status, TaskStatus.CREATED)
        self.assertIn(task.task_id, milestone.task_ids)
        self.assertEqual(task.metadata.get("milestone_id"), milestone.milestone_id)
        self.assertEqual(task.metadata.get("estimated_duration"), 60)
    
    def test_resource_management(self) -> None:
        """Test resource management functionality."""
        # Create project
        project = self.project_orchestrator.create_project(
            name="Resource Test Project",
            description="Testing resource management",
            start_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            end_date=(datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")
        )
        
        # Register a resource
        resource = self.project_orchestrator.register_resource(
            resource_type=ResourceType.COMPUTE,
            name="Test Compute Resource",
            capacity=100.0,
            project_id=project.project_id,
            cost_per_unit=1.0,
            tags=["test", "compute"]
        )
        
        # Verify resource was registered
        self.assertIsNotNone(resource)
        self.assertEqual(resource.name, "Test Compute Resource")
        self.assertEqual(resource.capacity, 100.0)
        self.assertEqual(resource.allocated, 0.0)
        self.assertIn(resource.resource_id, project.resources)
        
        # Create milestone and task
        milestone = self.project_orchestrator.add_project_milestone(
            project_id=project.project_id,
            name="Resource Test Milestone",
            description="Testing resource allocation",
            target_date=(datetime.now(timezone.utc) + timedelta(days=15)).strftime("%Y-%m-%d")
        )
        
        task = self.project_orchestrator.create_project_task(
            project_id=project.project_id,
            milestone_id=milestone.milestone_id,
            title="Resource Test Task",
            description="A task to test resource allocation",
            task_type=TaskType.ANALYSIS,
            required_capabilities=["data_analysis"],
            priority=TaskPriority.MEDIUM
        )
        
        # Allocate resource to task
        allocation_result = self.project_orchestrator.allocate_resources(
            project_id=project.project_id,
            task_id=task.task_id,
            resource_id=resource.resource_id,
            amount=25.0
        )
        
        # Verify allocation
        self.assertTrue(allocation_result)
        updated_resource = project.resources[resource.resource_id]
        self.assertEqual(updated_resource.allocated, 25.0)
        self.assertEqual(updated_resource.available_capacity(), 75.0)
        
        # Release resource
        release_result = self.project_orchestrator.release_resources(
            project_id=project.project_id,
            task_id=task.task_id,
            resource_id=resource.resource_id
        )
        
        # Verify release
        self.assertTrue(release_result)
        updated_resource = project.resources[resource.resource_id]
        self.assertEqual(updated_resource.allocated, 0.0)
        self.assertEqual(updated_resource.available_capacity(), 100.0)
    
    def test_track_milestone_progress(self) -> None:
        """Test tracking milestone progress."""
        # Create project with milestone and tasks
        now = datetime.now(timezone.utc)
        project = self.project_orchestrator.create_project(
            name="Progress Test Project",
            description="Testing progress tracking",
            start_date=now.strftime("%Y-%m-%d"),
            end_date=(now + timedelta(days=30)).strftime("%Y-%m-%d")
        )
        
        milestone = self.project_orchestrator.add_project_milestone(
            project_id=project.project_id,
            name="Progress Test Milestone",
            description="Testing progress tracking",
            target_date=(now + timedelta(days=15)).strftime("%Y-%m-%d")
        )
        
        # Create two tasks
        task1 = self.project_orchestrator.create_project_task(
            project_id=project.project_id,
            milestone_id=milestone.milestone_id,
            title="Progress Test Task 1",
            description="First task for progress testing",
            task_type=TaskType.RESEARCH,
            required_capabilities=["market_research"],
            priority=TaskPriority.MEDIUM
        )
        
        task2 = self.project_orchestrator.create_project_task(
            project_id=project.project_id,
            milestone_id=milestone.milestone_id,
            title="Progress Test Task 2",
            description="Second task for progress testing",
            task_type=TaskType.ANALYSIS,
            required_capabilities=["data_analysis"],
            priority=TaskPriority.MEDIUM
        )
        
        # Update task progress
        self.orchestrator_engine.task_coordinator.update_task_progress(
            task_id=task1.task_id,
            progress=1.0,  # 100% complete
            agent_id="test-research-agent"
        )
        
        self.orchestrator_engine.task_coordinator.update_task_progress(
            task_id=task2.task_id,
            progress=0.5,  # 50% complete
            agent_id="test-analysis-agent"
        )
        
        # Complete first task
        self.orchestrator_engine.task_coordinator.complete_task(
            task_id=task1.task_id,
            agent_id="test-research-agent"
        )
        
        # Track milestone progress
        milestone_progress = self.project_orchestrator.track_milestone_progress(
            project_id=project.project_id,
            milestone_id=milestone.milestone_id
        )
        
        # Verify progress
        self.assertIsNotNone(milestone_progress)
        self.assertEqual(milestone_progress, 0.75)  # (1.0 + 0.5) / 2 = 0.75
        self.assertEqual(project.milestones[milestone.milestone_id].progress, 0.75)
        self.assertEqual(project.milestones[milestone.milestone_id].status, MilestoneStatus.IN_PROGRESS)
        
        # Verify project progress is updated
        self.assertGreater(project.completion_percentage, 0.0)
    
    def test_status_update(self) -> None:
        """Test creating status updates."""
        # Create project
        project = self.project_orchestrator.create_project(
            name="Status Update Test Project",
            description="Testing status updates",
            start_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            end_date=(datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d"),
            stakeholders=[
                {
                    "id": "stakeholder@example.com",
                    "name": "Test Stakeholder",
                    "role": "sponsor"
                }
            ]
        )
        
        # Create status update
        status_update = self.project_orchestrator.create_status_update(
            project_id=project.project_id,
            title="Weekly Status Update",
            summary="Progress update for the test project",
            update_type="regular",
            project_health="on_track",
            accomplishments=[
                {"task": "Research", "status": "completed"}
            ],
            current_focus=[
                {"task": "Analysis", "status": "in_progress"}
            ],
            next_steps=[
                {"task": "Implementation Planning", "target": "next week"}
            ],
            stakeholders=["stakeholder@example.com"]
        )
        
        # Verify status update
        self.assertIsNotNone(status_update)
        self.assertEqual(status_update.title, "Weekly Status Update")
        self.assertEqual(status_update.project_health, "on_track")
        self.assertIn(status_update.update_id, project.status_updates)
        self.assertEqual(len(status_update.accomplishments), 1)
        self.assertEqual(len(status_update.current_focus), 1)
        self.assertEqual(len(status_update.next_steps), 1)


if __name__ == "__main__":
    unittest.main()