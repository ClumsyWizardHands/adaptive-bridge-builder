"""
Test Suite for OrchestrationAnalytics

This module provides test cases for the OrchestrationAnalytics system, ensuring that
it correctly tracks metrics, identifies bottlenecks, generates recommendations,
measures principle alignment, and creates visualizations.
"""

import unittest
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from orchestration_analytics import (
    OrchestrationAnalytics, MetricType, VisualizationType,
    AnalysisPeriod, BottleneckType, RecommendationCategory
)
from orchestrator_engine import OrchestratorEngine, TaskType, AgentRole, AgentProfile
from project_orchestrator import ProjectOrchestrator, Project, Resource, ResourceType
from principle_engine import PrincipleEngine, Principle


class TestOrchestrationAnalytics(unittest.TestCase):
    """Test cases for the OrchestrationAnalytics system."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create mock components
        self.orchestrator_engine = MagicMock(spec=OrchestratorEngine)
        self.orchestrator_engine.agent_profiles = {}
        
        self.project_orchestrator = MagicMock(spec=ProjectOrchestrator)
        self.project_orchestrator.projects = {}
        
        self.principle_engine = MagicMock(spec=PrincipleEngine)
        self.principle_engine.principles = {}
        self.principle_engine.get_principle = MagicMock(return_value=None)
        self.principle_engine.get_all_principles = MagicMock(return_value=[])
        
        # Initialize the analytics system
        self.analytics = OrchestrationAnalytics(
            agent_id="test-analytics",
            orchestrator_engine=self.orchestrator_engine,
            project_orchestrator=self.project_orchestrator,
            principle_engine=self.principle_engine
        )
    
    def test_register_metric(self):
        """Test that metrics can be registered correctly."""
        # Register a metric
        metric_id = self.analytics.register_metric(
            name="Test Metric",
            description="A metric for testing",
            type=MetricType.PERFORMANCE,
            unit="ops/sec",
            aggregation_method="avg",
            ideal_trend="increase",
            warning_threshold=10.0,
            critical_threshold=5.0
        )
        
        # Verify metric was registered
        self.assertIn(metric_id, self.analytics.metric_definitions)
        self.assertEqual(self.analytics.metric_definitions[metric_id].name, "Test Metric")
        self.assertEqual(self.analytics.metric_definitions[metric_id].unit, "ops/sec")
        self.assertEqual(self.analytics.metric_definitions[metric_id].type, MetricType.PERFORMANCE)
    
    def test_record_and_retrieve_metric(self):
        """Test that metrics can be recorded and retrieved correctly."""
        # Register a metric
        metric_id = self.analytics.register_metric(
            name="Test Metric",
            description="A metric for testing",
            type=MetricType.PERFORMANCE,
            unit="ops/sec",
            aggregation_method="avg",
            ideal_trend="increase"
        )
        
        # Record values
        now = datetime.utcnow()
        self.analytics.record_metric(metric_id, 10.0, now.isoformat())
        self.analytics.record_metric(metric_id, 20.0, (now + timedelta(minutes=5)).isoformat())
        self.analytics.record_metric(metric_id, 30.0, (now + timedelta(minutes=10)).isoformat())
        
        # Get aggregated value
        value = self.analytics.get_metric_value(
            metric_id=metric_id,
            aggregation_period=AnalysisPeriod.HOURLY,
            start_time=now.isoformat(),
            end_time=(now + timedelta(minutes=15)).isoformat()
        )
        
        # Verify the average was calculated correctly
        self.assertEqual(value, 20.0)  # (10 + 20 + 30) / 3 = 20.0
    
    def test_analyze_agent_capacity_bottlenecks(self):
        """Test that agent capacity bottlenecks are correctly identified."""
        # Set up an agent profile with high utilization
        self.orchestrator_engine.agent_profiles = {
            "agent-001": AgentProfile(
                agent_id="agent-001",
                roles=[AgentRole.EXECUTOR],
                capabilities=["processing"],
                specialization={TaskType.EXECUTION: 0.9},
                current_load=9,
                max_load=10,
                task_history=["task-1", "task-2"]
            )
        }
        
        # Analyze bottlenecks
        bottlenecks = self.analytics.analyze_bottlenecks()
        
        # Verify a bottleneck was identified
        self.assertEqual(len(bottlenecks), 1)
        self.assertEqual(bottlenecks[0].bottleneck_type, BottleneckType.AGENT_CAPACITY)
        self.assertEqual(bottlenecks[0].affected_items["agents"], ["agent-001"])
        self.assertGreaterEqual(bottlenecks[0].severity, 0.9)  # Should be at least 90%
    
    def test_analyze_resource_contention_bottlenecks(self):
        """Test that resource contention bottlenecks are correctly identified."""
        # Set up a project with a highly utilized resource
        resource = MagicMock(spec=Resource)
        resource.resource_id = "resource-001"
        resource.name = "Test Resource"
        resource.utilization_percentage.return_value = 95.0  # 95% utilized
        
        project = MagicMock(spec=Project)
        project.milestones = {}
        project.resources = {"resource-001": resource}
        
        self.project_orchestrator.projects = {"project-001": project}
        
        # Analyze bottlenecks
        bottlenecks = self.analytics.analyze_bottlenecks()
        
        # Verify a bottleneck was identified
        self.assertTrue(any(b.bottleneck_type == BottleneckType.RESOURCE_CONTENTION for b in bottlenecks))
        resource_bottleneck = next(b for b in bottlenecks if b.bottleneck_type == BottleneckType.RESOURCE_CONTENTION)
        self.assertEqual(resource_bottleneck.affected_items["resources"], ["resource-001"])
        self.assertGreaterEqual(resource_bottleneck.severity, 0.9)  # Should be at least 90%
    
    def test_measure_principle_alignment(self):
        """Test measurement of principle alignment."""
        # Set up a principle
        principle = MagicMock(spec=Principle)
        principle.principle_id = "principle-001"
        principle.name = "Efficiency"
        
        # Add metric related to principle
        metric_id = self.analytics.register_metric(
            name="Test Principle Metric",
            description="A metric for testing principle alignment",
            type=MetricType.ALIGNMENT,
            unit="score",
            aggregation_method="avg",
            ideal_trend="increase",
            related_principle="principle-001"
        )
        
        # Record some values
        self.analytics.record_metric(metric_id, 0.7)  # 70% alignment
        
        # Setup principle engine mock
        self.principle_engine.get_principle.return_value = principle
        self.principle_engine.get_all_principles.return_value = [principle]
        self.principle_engine.principles = {"principle-001": principle}
        
        # Measure alignment
        alignment = self.analytics.measure_principle_alignment(["principle-001"])
        
        # Verify measurement
        self.assertIn("principle-001", alignment)
        self.assertGreaterEqual(alignment["principle-001"].alignment_score, 0.5)
    
    def test_visualization_creation(self):
        """Test creation of visualizations."""
        # Create a visualization
        visualization = self.analytics.create_visualization(
            visualization_type=VisualizationType.TIMELINE,
            title="Test Timeline",
            description="A test timeline visualization",
            time_range=(
                (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                datetime.utcnow().isoformat()
            ),
            data_sources={"tasks": True, "agents": ["agent-001"]},
            filters={"task_types": [TaskType.EXECUTION.name]},
            parameters={"show_dependencies": True}
        )
        
        # Verify visualization was created
        self.assertIsNotNone(visualization)
        self.assertEqual(visualization.visualization_type, VisualizationType.TIMELINE)
        self.assertEqual(visualization.render_format, "json")
    
    def test_recommendation_generation(self):
        """Test generation of optimization recommendations."""
        # Set up a bottleneck
        bottleneck = self.analytics._create_test_bottleneck()
        
        # Generate recommendations
        recommendations = self.analytics.generate_recommendations([bottleneck])
        
        # Verify recommendations were generated
        self.assertGreaterEqual(len(recommendations), 1)
        self.assertIn(bottleneck.bottleneck_id, recommendations[0].related_bottlenecks)
    
    def tearDown(self):
        """Clean up after tests."""
        pass


if __name__ == "__main__":
    unittest.main()
