"""
Communication Examples

This module demonstrates how to use the various communication handlers
to enable agents to exchange different content types, share files,
collaborate on tasks, and adapt communications to each other's capabilities.
"""

import json
import os
import base64
from typing import Dict, List, Any, Optional
from datetime import datetime

from content_handler import ContentHandler, ContentFormat
from file_exchange_handler import FileExchangeHandler, TransferStatus, TransferType
from collaborative_task_handler import (
    TaskCoordinator, Task, TaskStatus, TaskPriority, 
)
from communication_adapter import (
    CommunicationAdapter, AgentProfile, AgentCapability,
    CommunicationProtocol, MessageTransformation
)

# Create output directory for examples
os.makedirs("example_output", exist_ok=True)

# Configure agent IDs for examples
AGENT_IDS = {
    "coordinator": "coordinator-agent-001",
    "data_processor": "data-processor-agent-001",
    "file_handler": "file-handler-agent-001",
    "analyst": "analyst-agent-001",
    "basic_agent": "basic-agent-001",
    "advanced_agent": "advanced-agent-001",
}


def example_content_handling():
    """Example demonstrating content handling with different data types."""
    print("\n===== CONTENT HANDLING EXAMPLE =====\n")
    
    # Initialize content handler
    content_handler = ContentHandler()
    
    # Example 1: JSON data processing
    print("Example 1: JSON data processing")
    json_data = {
        "name": "Data Analysis Project",
        "status": "in_progress",
        "tasks": [
            {"id": "task-001", "description": "Collect data", "status": "completed"},
            {"id": "task-002", "description": "Process data", "status": "in_progress"},
            {"id": "task-003", "description": "Visualize results", "status": "pending"}
        ],
        "metadata": {
            "priority": "high",
            "deadline": "2025-06-01"
        }
    }
    
    # Detect format
    detected_format = content_handler.detect_format(json_data)
    print(f"Detected format: {detected_format.value}")
    
    # Validate JSON
    is_valid, error = content_handler.validate_content(json_data, ContentFormat.JSON)
    print(f"JSON validation: {'Valid' if is_valid else 'Invalid'}")
    if error:
        print(f"Validation error: {error}")
        
    # Convert JSON to different formats
    markdown_content, success = content_handler.convert_content(
        json_data, ContentFormat.JSON, ContentFormat.MARKDOWN
    )
    print("\nJSON converted to Markdown:")
    print(markdown_content[:300] + "..." if len(markdown_content) > 300 else markdown_content)
    
    # Example 2: XML handling
    print("\nExample 2: XML handling")
    xml_content = """
    <project>
        <name>Data Analysis Project</name>
        <status>in_progress</status>
        <tasks>
            <task id="task-001">
                <description>Collect data</description>
                <status>completed</status>
            </task>
            <task id="task-002">
                <description>Process data</description>
                <status>in_progress</status>
            </task>
        </tasks>
    </project>
    """
    
    # Detect format
    detected_format = content_handler.detect_format(xml_content)
    print(f"Detected format: {detected_format.value}")
    
    # Convert XML to JSON
    json_from_xml, success = content_handler.convert_content(
        xml_content, ContentFormat.XML, ContentFormat.JSON
    )
    print("\nXML converted to JSON:")
    print(json.dumps(json_from_xml, indent=2)[:300] + "..." if len(json.dumps(json_from_xml, indent=2)) > 300 else json.dumps(json_from_xml, indent=2))
    
    # Example 3: Simplifying content for limited agents
    print("\nExample 3: Simplifying complex content")
    complex_json = {
        "analysisResults": {
            "dataPoints": [
                {"x": 1, "y": 5, "label": "Point A", "confidence": 0.95},
                {"x": 2, "y": 7, "label": "Point B", "confidence": 0.87},
                {"x": 3, "y": 2, "label": "Point C", "confidence": 0.92}
            ],
            "trendLine": {
                "slope": -1.5,
                "intercept": 6.5,
                "rSquared": 0.87
            },
            "clusters": [
                {
                    "centroid": {"x": 1.5, "y": 6},
                    "members": ["Point A", "Point B"],
                    "cohesion": 0.8
                },
                {
                    "centroid": {"x": 3, "y": 2},
                    "members": ["Point C"],
                    "cohesion": 1.0
                }
            ],
            "metadata": {
                "generatedAt": "2025-05-16T12:34:56Z",
                "algorithm": "advanced-clustering-v2",
                "parameters": {
                    "distanceMetric": "euclidean",
                    "minSamples": 1,
                    "eps": 2.0
                }
            }
        }
    }
    
    # Simplify to different complexity levels
    simplified_level1 = content_handler.simplify_content(
        complex_json, ContentFormat.JSON, target_complexity=1
    )
    simplified_level3 = content_handler.simplify_content(
        complex_json, ContentFormat.JSON, target_complexity=3
    )
    
    print("\nSimplified content (Level 1 - Most simplified):")
    print(json.dumps(simplified_level1, indent=2))
    
    print("\nSimplified content (Level 3 - Moderately simplified):")
    print(json.dumps(simplified_level3, indent=2)[:300] + "..." if len(json.dumps(simplified_level3, indent=2)) > 300 else json.dumps(simplified_level3, indent=2))
    
    # Example 4: Extracting metadata
    print("\nExample 4: Extracting metadata from content")
    markdown_with_frontmatter = """---
title: Data Analysis Report
author: Analyst Agent
date: 2025-05-16
tags: [data, analysis, report]
---

# Data Analysis Report

This report presents the findings from our recent data analysis project.

## Key Findings

1. The data shows a significant trend in...
2. We identified three main clusters of...

## Recommendations

Based on our analysis, we recommend the following actions:
"""
    
    metadata = content_handler.extract_metadata(
        markdown_with_frontmatter, ContentFormat.MARKDOWN
    )
    
    print("Extracted metadata:")
    print(json.dumps(metadata, indent=2))
    
    # Example 5: Enhancing content
    print("\nExample 5: Enhancing content with additional formatting")
    basic_md = """
# Project Status Update

## Completed Tasks
- Data collection
- Initial processing

## In Progress
- Advanced analytics
- Report generation

## Pending
- Client presentation
- Follow-up actions
"""
    
    enhanced_md = content_handler.enhance_content(
        basic_md, ContentFormat.MARKDOWN, enhancement_level=3
    )
    
    print("Enhanced markdown content:")
    print(enhanced_md[:500] + "..." if len(enhanced_md) > 500 else enhanced_md)


def example_file_exchange():
    """Example demonstrating file exchange between agents."""
    print("\n===== FILE EXCHANGE EXAMPLE =====\n")
    
    # Create directories for file exchange example
    os.makedirs("example_output/agent1", exist_ok=True)
    os.makedirs("example_output/agent2", exist_ok=True)
    
    # Create a test file for exchange
    test_file_path = "example_output/agent1/data_report.json"
    with open(test_file_path, 'w') as f:
        json.dump({
            "title": "Monthly Data Report",
            "date": "2025-05",
            "metrics": {
                "users": 12500,
                "transactions": 54321,
                "revenue": 98765.43
            },
            "trends": {
                "user_growth": "+5.2%",
                "transaction_volume": "+7.8%",
                "average_value": "+3.1%"
            }
        }, f, indent=2)
    
    # Initialize file exchange handlers for two agents
    sender_handler = FileExchangeHandler(
        agent_id=AGENT_IDS["file_handler"],
        storage_dir="example_output/agent1/file_exchange"
    )
    
    receiver_handler = FileExchangeHandler(
        agent_id=AGENT_IDS["data_processor"],
        storage_dir="example_output/agent2/file_exchange"
    )
    
    print("Initialized file exchange handlers")
    
    # Start a file upload
    transfer_id = sender_handler.upload_file(
        file_path=test_file_path,
        recipient_id=AGENT_IDS["data_processor"],
        metadata={"description": "Monthly data report", "priority": "high"}
    )
    
    print(f"Started file upload with transfer ID: {transfer_id}")
    
    # Get the first chunk
    chunk_info = sender_handler.send_file_chunk(transfer_id, 0)
    
    print(f"Prepared first chunk: {chunk_info['chunk_index']} of {chunk_info['total_chunks']}")
    print(f"First chunk size: {len(chunk_info['chunk_data'])} bytes")
    
    # Receive the chunk on the other side
    transfer_status = receiver_handler.receive_file_chunk(
        transfer_id=transfer_id,
        chunk_index=chunk_info['chunk_index'],
        chunk_data=chunk_info['chunk_data'],
        sender_id=AGENT_IDS["file_handler"],
        total_chunks=chunk_info['total_chunks'],
        file_info=chunk_info.get('file_info')
    )
    
    print(f"Received first chunk, transfer status: {transfer_status['status']}")
    
    # If multiple chunks, continue with the rest
    current_chunk = 1
    while current_chunk < chunk_info['total_chunks']:
        next_chunk = sender_handler.send_file_chunk(transfer_id, current_chunk)
        
        transfer_status = receiver_handler.receive_file_chunk(
            transfer_id=transfer_id,
            chunk_index=next_chunk['chunk_index'],
            chunk_data=next_chunk['chunk_data'],
            sender_id=AGENT_IDS["file_handler"],
            total_chunks=next_chunk['total_chunks'],
            file_info=None  # Only needed for first chunk
        )
        
        print(f"Received chunk {current_chunk}, transfer status: {transfer_status['status']}")
        current_chunk += 1
    
    # Download the completed file
    if transfer_status['status'] == TransferStatus.COMPLETED.value:
        output_path = receiver_handler.download_file(
            transfer_id=transfer_id,
            output_path="example_output/agent2/received_report.json"
        )
        
        print(f"File downloaded to: {output_path}")
        
        # Verify the downloaded file
        with open(output_path, 'r') as f:
            received_data = json.load(f)
            
        print(f"Received file contains {len(json.dumps(received_data))} bytes of data")
        print(f"Sample content: {json.dumps(received_data)[:100]}...")
    
    # Example of file manifest for multiple files
    print("\nCreating file manifest for multiple files:")
    
    # Create a second test file
    test_file2_path = "example_output/agent1/metrics_summary.txt"
    with open(test_file2_path, 'w') as f:
        f.write("""MONTHLY METRICS SUMMARY
======================
User Growth: +5.2%
Transaction Volume: +7.8%
Average Value: +3.1%

Key insights:
- User retention has improved significantly
- Mobile transactions now represent 68% of total volume
- Premium tier subscriptions increased by 12%
""")
    
    # Create a manifest for both files
    manifest = sender_handler.create_file_manifest(
        files=[test_file_path, test_file2_path],
        recipient_id=AGENT_IDS["data_processor"]
    )
    
    print(f"Created manifest with ID: {manifest['manifest_id']}")
    print(f"Manifest includes {manifest['file_count']} files")
    print(f"Total size: {manifest['total_size']} bytes")
    
    # Process the manifest on the receiver side
    processed_manifest = receiver_handler.process_file_manifest(
        manifest=manifest,
        accept=True,
        output_dir="example_output/agent2/received_files"
    )
    
    print(f"Processed manifest with status: {processed_manifest['status']}")
    print(f"Files will be stored in: {processed_manifest.get('output_dir')}")


def example_collaborative_tasks():
    """Example demonstrating collaborative task handling between agents."""
    print("\n===== COLLABORATIVE TASKS EXAMPLE =====\n")
    
    # Initialize a task coordinator
    coordinator = TaskCoordinator(
        agent_id=AGENT_IDS["coordinator"],
        storage_dir="example_output/tasks"
    )
    
    # Register agent capabilities
    coordinator.update_agent_capabilities(
        AGENT_IDS["data_processor"],
        ["data_processing", "statistical_analysis", "data_validation"]
    )
    
    coordinator.update_agent_capabilities(
        AGENT_IDS["analyst"],
        ["data_visualization", "report_generation", "trend_analysis"]
    )
    
    print("Registered agent capabilities")
    
    # Create a complex task with subtasks
    main_task = coordinator.create_task(
        title="Quarterly Data Analysis Project",
        description="Analyze and report on Q2 2025 business metrics",
        required_capabilities=["data_processing", "report_generation"],
        priority=TaskPriority.HIGH,
        metadata={"client": "ACME Corp", "deadline": "2025-06-30"}
    )
    
    print(f"Created main task: {main_task.title} (ID: {main_task.task_id})")
    
    # Create subtasks
    data_processing_task = coordinator.add_subtask(
        parent_task_id=main_task.task_id,
        title="Process Raw Data",
        description="Clean and normalize the raw data files",
        required_capabilities=["data_processing", "data_validation"]
    )
    
    analysis_task = coordinator.add_subtask(
        parent_task_id=main_task.task_id,
        title="Perform Statistical Analysis",
        description="Run statistical analysis on processed data",
        required_capabilities=["statistical_analysis"]
    )
    
    visualization_task = coordinator.add_subtask(
        parent_task_id=main_task.task_id,
        title="Create Visualizations",
        description="Create charts and visualizations of key metrics",
        required_capabilities=["data_visualization"]
    )
    
    report_task = coordinator.add_subtask(
        parent_task_id=main_task.task_id,
        title="Generate Final Report",
        description="Compile analysis and visualizations into a comprehensive report",
        required_capabilities=["report_generation"]
    )
    
    print(f"Added {len(main_task.subtasks)} subtasks to the main task")
    
    # Add dependencies between tasks
    analysis_task.dependencies.append(data_processing_task.task_id)
    visualization_task.dependencies.append(analysis_task.task_id)
    report_task.dependencies.append(visualization_task.task_id)
    
    print("Established task dependencies")
    
    # Suggest agents for each task
    for subtask in main_task.subtasks.values():
        suggested_agents = coordinator.suggest_agents_for_task(subtask.task_id)
        print(f"Suggested agents for '{subtask.title}': {', '.join(suggested_agents)}")
    
    # Assign tasks to agents
    coordinator.assign_task(data_processing_task.task_id, [AGENT_IDS["data_processor"]])
    coordinator.assign_task(analysis_task.task_id, [AGENT_IDS["data_processor"]])
    coordinator.assign_task(visualization_task.task_id, [AGENT_IDS["analyst"]])
    coordinator.assign_task(report_task.task_id, [AGENT_IDS["analyst"]])
    
    print("Assigned tasks to appropriate agents")
    
    # Start the first task
    coordinator.start_task(data_processing_task.task_id)
    print(f"Started task: {data_processing_task.title}")
    
    # Update progress on the first task
    coordinator.update_task_progress(
        data_processing_task.task_id, 
        0.5, 
        AGENT_IDS["data_processor"]
    )
    print(f"Updated progress on {data_processing_task.title} to 50%")
    
    # Add a message to the task
    coordinator.add_task_message(
        data_processing_task.task_id,
        AGENT_IDS["data_processor"],
        "Found 15 duplicate records that have been removed. Data quality looks good overall."
    )
    print("Added status message to the task")
    
    # Complete the first task
    coordinator.add_task_result(
        data_processing_task.task_id,
        AGENT_IDS["data_processor"],
        {
            "records_processed": 5280,
            "duplicates_removed": 15,
            "missing_values_imputed": 87,
            "outliers_normalized": 23,
            "output_file": "processed_data_q2_2025.csv"
        }
    )
    
    coordinator.complete_task(data_processing_task.task_id, AGENT_IDS["data_processor"])
    print(f"Completed task: {data_processing_task.title}")
    
    # Check if the next task can start
    next_task_status = coordinator.start_task(analysis_task.task_id)
    print(f"Starting next task ({analysis_task.title}): {'Success' if next_task_status else 'Failed'}")
    
    # Export task graph
    task_graph = coordinator.export_task_graph()
    print(f"\nTask dependency graph contains {len(task_graph['nodes'])} nodes and {len(task_graph['edges'])} edges")
    
    # Save task graph to file for visualization
    with open("example_output/task_graph.json", "w") as f:
        json.dump(task_graph, f, indent=2)
        
    print("Task graph saved to example_output/task_graph.json")


def example_communication_adaptation():
    """Example demonstrating communication adaptation based on agent capabilities."""
    print("\n===== COMMUNICATION ADAPTATION EXAMPLE =====\n")
    
    # Initialize content handler and communication adapter
    content_handler = ContentHandler()
    adapter = CommunicationAdapter(
        agent_id=AGENT_IDS["advanced_agent"],
        content_handler=content_handler
    )
    
    # Define agent profiles with different capabilities
    basic_agent_profile = AgentProfile(
        agent_id=AGENT_IDS["basic_agent"],
        capabilities=[],  # No special capabilities
        preferred_format=ContentFormat.TEXT,
        max_message_size=10000
    )
    
    advanced_agent_profile = AgentProfile(
        agent_id=AGENT_IDS["advanced_agent"],
        capabilities=[
            AgentCapability.STRUCTURED_DATA,
            AgentCapability.MARKDOWN,
            AgentCapability.HTML,
            AgentCapability.CODE_SNIPPETS,
            AgentCapability.BINARY_DATA
        ],
        preferred_format=ContentFormat.JSON,
        protocol=CommunicationProtocol.DIRECT
    )
    
    analyst_agent_profile = AgentProfile(
        agent_id=AGENT_IDS["analyst"],
        capabilities=[
            AgentCapability.STRUCTURED_DATA,
            AgentCapability.MARKDOWN,
            AgentCapability.CODE_SNIPPETS
        ],
        preferred_format=ContentFormat.MARKDOWN,
        protocol=CommunicationProtocol.REST_API,
        metadata={
            "api_headers": {
                "Authorization": "Bearer sample_token",
                "Content-Type": "application/json"
            }
        }
    )
    
    # Register the profiles
    adapter.register_agent_profile(basic_agent_profile)
    adapter.register_agent_profile(advanced_agent_profile)
    adapter.register_agent_profile(analyst_agent_profile)
    
    print("Registered agent profiles")
    
    # Check compatibility between agents
    compatibility = adapter.check_compatibility(
        AGENT_IDS["advanced_agent"], AGENT_IDS["basic_agent"]
    )
    
    print(f"Compatibility check between advanced and basic agent:")
    print(f"  Compatible: {compatibility['compatible']}")
    print(f"  Recommended format: {compatibility['recommended_format']}")
    print(f"  Adaptation needed: {compatibility['adaptation_needed']}")
    
    # Example 1: Adapt structured data for basic agent
    complex_data = {
        "analysis_results": {
            "metrics": {
                "accuracy": 0.928,
                "precision": 0.873,
                "recall": 0.942,
                "f1_score": 0.906
            },
            "predictions": [
                {"id": "sample1", "predicted": "class_a", "probability": 0.89},
                {"id": "sample2", "predicted": "class_b", "probability": 0.76},
                {"id": "sample3", "predicted": "class_a", "probability": 0.95}
            ],
            "model_info": {
                "name": "GradientBoostingClassifier",
                "parameters": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 3
                },
                "training_time": "12m 34s"
            }
        }
    }
    
    # Adapt for basic agent (will simplify and convert to text)
    basic_adapted = adapter.adapt_message(
        complex_data,
        AGENT_IDS["basic_agent"],
        transformations=[MessageTransformation.SIMPLIFY]
    )
    
    print("\nAdapted complex data for basic agent:")
    print(f"  Format: {basic_adapted.get('format', 'text')}")
    if isinstance(basic_adapted['content'], dict):
        print(f"  Content (simplified): {json.dumps(basic_adapted['content'], indent=2)}")
    else:
        print(f"  Content (simplified): {basic_adapted['content'][:300]}...")
    
    # Example 2: Adapt markdown for REST API agent
    markdown_content = """
# Analysis Summary

## Model Performance
- **Accuracy**: 92.8%
- **Precision**: 87.3%
- **Recall**: 94.2%
- **F1 Score**: 90.6%

## Key Insights
1. Class A predictions show higher confidence
2. Model performed best on structured data inputs
3. Feature importance analysis reveals that features X, Y, and Z contributed most significantly

## Recommendations
For improved performance, consider:
- Increasing training data diversity
- Tuning the learning rate parameter
- Adding more domain-specific features
"""
    
    # Adapt for analyst agent (using REST API protocol)
    analyst_adapted = adapter.adapt_message(
        markdown_content,
        AGENT_IDS["analyst"]
    )
    
    print("\nAdapted markdown content for analyst agent (REST API):")
    print(f"  Protocol formatting applied: {analyst_adapted.keys()}")
    print(f"  Headers included: {'headers' in analyst_adapted}")
    if 'data' in analyst_adapted:
        print(f"  Content included as data field: {type(analyst_adapted['data'])}")
    
    # Example 3: Adapt large content with chunking
    large_text = "This is a very large message that exceeds the basic agent's size limit. " * 200
    
    # Set a small message size limit for demonstration
    basic_agent_profile.max_message_size = 500
    adapter.register_agent_profile(basic_agent_profile)
    
    # Adapt with chunking transformation
    chunked_message = adapter.adapt_message(
        large_text,
        AGENT_IDS["basic_agent"],
        transformations=[MessageTransformation.CHUNK]
    )
    
    print("\nAdapted large content with chunking:")
    print(f"  Chunking applied: {'chunked' in chunked_message}")
    if 'chunked' in chunked_message and chunked_message['chunked']:
        print(f"  Chunk index: {chunked_message['chunk_index']}")
        print(f"  Total chunks: {chunked_message['total_chunks']}")
        print(f"  Chunk size: {len(chunked_message['content'])}")
    
    # Generate compatibility matrix for all agents
    matrix = adapter.generate_compatibility_matrix([
        AGENT_IDS["basic_agent"],
        AGENT_IDS["advanced_agent"],
        AGENT_IDS["analyst"]
    ])
    
    print("\nCompatibility matrix generated for all agents")
    print(f"  Matrix dimensions: {len(matrix.keys())} x {len(next(iter(matrix.values()))) if matrix else 0}")
    
    # Save compatibility matrix to file
    with open("example_output/compatibility_matrix.json", "w") as f:
        json.dump(matrix, f, indent=2)
        
    print("Compatibility matrix saved to example_output/compatibility_matrix.json")


def integrated_example():
    """
    Integrated example combining content handling, file exchange,
    collaborative tasks, and communication adaptation.
    """
    print("\n===== INTEGRATED EXAMPLE =====\n")
    
    # Initialize all handlers
    content_handler = ContentHandler()
    file_handler_agent = FileExchangeHandler(
        agent_id=AGENT_IDS["file_handler"],
        storage_dir="example_output/integrated/file_exchange"
    )
    task_coordinator = TaskCoordinator(
        agent_id=AGENT_IDS["coordinator"],
        storage_dir="example_output/integrated/tasks"
    )
    comm_adapter = CommunicationAdapter(
        agent_id=AGENT_IDS["coordinator"],
        content_handler=content_handler
    )
    
    # Create output directory
    os.makedirs("example_output/integrated", exist_ok=True)
    
    print("Initialized all handlers")
    
    # Define agent profiles with specific capabilities
    data_processor_profile = AgentProfile(
        agent_id=AGENT_IDS["data_processor"],
        capabilities=[
            AgentCapability.STRUCTURED_DATA,
            AgentCapability.CODE_SNIPPETS
        ],
        preferred_format=ContentFormat.JSON
    )
    
    analyst_profile = AgentProfile(
        agent_id=AGENT_IDS["analyst"],
        capabilities=[
            AgentCapability.STRUCTURED_DATA,
            AgentCapability.MARKDOWN,
            AgentCapability.HTML
        ],
        preferred_format=ContentFormat.MARKDOWN
    )
    
    # Register profiles and capabilities
    comm_adapter.register_agent_profile(data_processor_profile)
    comm_adapter.register_agent_profile(analyst_profile)
    
    task_coordinator.update_agent_capabilities(
        AGENT_IDS["data_processor"],
        ["data_processing", "statistical_analysis", "data_validation"]
    )
    
    task_coordinator.update_agent_capabilities(
        AGENT_IDS["analyst"],
        ["data_visualization", "report_generation", "trend_analysis"]
    )
    
    print("Registered agent profiles and capabilities")
    
    # 1. Create a main data analysis task
    main_task = task_coordinator.create_task(
        title="Customer Behavior Analysis",
        description="Analyze customer behavior patterns from transaction data",
        required_capabilities=["data_processing", "trend_analysis"],
        priority=TaskPriority.HIGH
    )
    
    # 2. Create subtasks
    data_prep_task = task_coordinator.add_subtask(
        parent_task_id=main_task.task_id,
        title="Prepare Transaction Data",
        description="Clean and preprocess transaction dataset",
        required_capabilities=["data_processing"]
    )
    
    analysis_task = task_coordinator.add_subtask(
        parent_task_id=main_task.task_id,
        title="Identify Behavior Patterns",
        description="Run analytics to identify customer behavior patterns",
        required_capabilities=["statistical_analysis", "trend_analysis"]
    )
    
    # Set up dependencies
    analysis_task.dependencies.append(data_prep_task.task_id)
    
    print(f"Created main task with {len(main_task.subtasks)} subtasks")
    
    # 3. Assign tasks to appropriate agents
    task_coordinator.assign_task(data_prep_task.task_id, [AGENT_IDS["data_processor"]])
    task_coordinator.assign_task(analysis_task.task_id, [AGENT_IDS["analyst"]])
    
    # Start the data prep task
    task_coordinator.start_task(data_prep_task.task_id)
    
    print("Assigned and started tasks")
    
    # 4. Create a sample data file for processing
    transactions_file = "example_output/integrated/transactions.json"
    with open(transactions_file, 'w') as f:
        json.dump({
            "transactions": [
                {"id": "t001", "customer_id": "c123", "amount": 59.99, "date": "2025-05-01", "category": "electronics"},
                {"id": "t002", "customer_id": "c456", "amount": 25.50, "date": "2025-05-01", "category": "groceries"},
                {"id": "t003", "customer_id": "c123", "amount": 12.99, "date": "2025-05-02", "category": "entertainment"},
                {"id": "t004", "customer_id": "c789", "amount": 199.00, "date": "2025-05-03", "category": "electronics"},
                {"id": "t005", "customer_id": "c456", "amount": 45.75, "date": "2025-05-03", "category": "clothing"}
            ],
            "metadata": {
                "record_count": 5,
                "date_range": "2025-05-01 to 2025-05-03",
                "source": "POS System"
            }
        }, f, indent=2)
    
    # 5. Share the file with the data processor agent
    transfer_id = file_handler_agent.upload_file(
        file_path=transactions_file,
        recipient_id=AGENT_IDS["data_processor"],
        metadata={"task_id": data_prep_task.task_id}
    )
    
    # Simulate file transfer completion (simplified for the example)
    print(f"File transfer initiated with ID: {transfer_id}")
    
    # 6. Data processor processes the data and updates task
    # In a real scenario, this would be done by the data processor agent
    task_coordinator.update_task_progress(
        data_prep_task.task_id, 
        0.75,
        AGENT_IDS["data_processor"]
    )
    
    print("Data processor processed the transaction data")
    
    # 7. Add processing results with appropriate content format for data processor
    processing_results = {
        "processed_records": 5,
        "anomalies_detected": 0,
        "customer_segments": [
            {"segment": "electronics_buyers", "customers": ["c123", "c789"]},
            {"segment": "frequent_shoppers", "customers": ["c123", "c456"]}
        ],
        "summary_stats": {
            "total_spent": 343.23,
            "avg_transaction": 68.65,
            "categories": ["electronics", "groceries", "entertainment", "clothing"]
        }
    }
    
    # Add results to the task
    task_coordinator.add_task_result(
        data_prep_task.task_id,
        AGENT_IDS["data_processor"],
        processing_results
    )
    
    # Complete the data preparation task
    task_coordinator.complete_task(data_prep_task.task_id, AGENT_IDS["data_processor"])
    
    print("Data preparation task completed with results")
    
    # 8. Now the analysis task can start
    task_coordinator.start_task(analysis_task.task_id)
    
    # 9. Adapt the processing results for the analyst agent who prefers markdown
    adapted_results = comm_adapter.adapt_message(
        processing_results, 
        AGENT_IDS["analyst"]
    )
    
    print("\nAdapted results for analyst agent:")
    print(f"  Format: {adapted_results.get('format', 'unknown')}")
    print(f"  Protocol: {AGENT_IDS['analyst']} prefers {analyst_profile.protocol.value}")
    
    # 10. Analyst performs analysis (simulated)
    analysis_results = """
# Customer Behavior Analysis Results

## Customer Segments
- **Electronics Buyers**: 2 customers (c123, c789)
- **Frequent Shoppers**: 2 customers (c123, c456)

## Key Behavioral Patterns
1. Customer c123 appears in multiple segments and shops across categories
2. Electronics has the highest average transaction value ($129.50)
3. 40% of transactions occur on a single day (2025-05-01)

## Recommendations
- Target customer c123 with cross-category promotions
- Implement a loyalty program for frequent shoppers
- Create special promotions for electronics buyers
"""
    
    # Update analysis task progress
    task_coordinator.update_task_progress(
        analysis_task.task_id,
        1.0,  # 100% complete
        AGENT_IDS["analyst"]
    )
    
    # Add analysis results to the task
    task_coordinator.add_task_result(
        analysis_task.task_id,
        AGENT_IDS["analyst"],
        analysis_results
    )
    
    # Complete the analysis task
    task_coordinator.complete_task(analysis_task.task_id, AGENT_IDS["analyst"])
    
    print("Analysis task completed with results")
    
    # 11. Create an output file with the analysis results
    analysis_file = "example_output/integrated/analysis_report.md"
    with open(analysis_file, 'w') as f:
        f.write(analysis_results)
    
    # 12. Get the overall task status
    main_task_update = main_task.get_status_update()
    
    print("\nMain task status:")
    print(f"  Status: {main_task_update['status']}")
    print(f"  Progress: {main_task_update['progress']:.0%}")
    print(f"  Subtasks completed: {sum(1 for subtask in main_task.subtasks.values() if subtask.status == TaskStatus.COMPLETED)}/{len(main_task.subtasks)}")
    
    print("\nCollaborative workflow completed successfully!")


if __name__ == "__main__":
    print("Running communication handler examples...")
    
    # Run individual examples
    example_content_handling()
    example_file_exchange()
    example_collaborative_tasks()
    example_communication_adaptation()
    
    # Run the integrated example
    integrated_example()
    
    print("\nAll examples completed!")
