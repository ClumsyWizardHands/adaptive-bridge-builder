#!/usr/bin/env python3
"""
Media Content Processor Example

This module demonstrates how the MediaContentProcessor can be used to generate,
process, analyze, and adapt various media types including images, charts, and
structured documents. It includes examples of media generation, information
extraction, accessibility enhancements, and device adaptation.
"""

import os
import io
import json
import logging
import base64
import tempfile
from typing import Dict, Any, List, Optional, Union

from media_content_processor import (
    MediaContentProcessor, MediaType, ImageFormat, ChartType, DocumentFormat,
    DeviceCategory, AccessibilityFeature, DeviceProfile,
    ImageContent, ChartContent, DocumentContent, TableContent
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MediaContentProcessorExample")


def setup_media_processor() -> MediaContentProcessor:
    """Set up a media content processor instance with temporary storage."""
    # Create a temporary directory for media storage
    temp_dir = tempfile.mkdtemp(prefix="media_content_")
    logger.info(f"Created temporary media storage directory: {temp_dir}")
    
    # Initialize processor
    processor = MediaContentProcessor(
        agent_id="adaptive_bridge_builder",
        media_storage_path=temp_dir
    )
    
    return processor


def demonstrate_image_processing(processor: MediaContentProcessor):
    """Demonstrate image processing capabilities."""
    print("\n=== IMAGE PROCESSING DEMONSTRATION ===")
    
    # Skip if PIL is not available
    if not processor._check_dependencies():
        print("Skipping image processing demo - PIL (Pillow) is required")
        return
    
    try:
        # Create a simple test image
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a new image with white background
        width, height = 800, 600
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        
        # Draw some shapes
        draw.rectangle([50, 50, 750, 550], outline="blue", width=5)
        draw.ellipse([200, 150, 600, 450], fill="lightblue", outline="blue", width=5)
        
        # Add text
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except IOError:
            # Fallback to default font
            font = ImageFont.load_default()
            
        draw.text((400, 300), "Hello, Agent!", fill="darkblue", font=font, anchor="mm")
        
        # Save to a buffer
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Process the image
        print("Creating an image content entry...")
        image_content = processor.create_image(
            image_data=buffer.getvalue(),
            format=ImageFormat.PNG,
            alt_text="A test image with a blue rectangle, light blue circle, and 'Hello, Agent!' text",
            metadata={
                "description": "Test image for demonstration",
                "creator": "MediaContentProcessorExample",
                "purpose": "Demonstration"
            }
        )
        
        print(f"Created image content with ID: {image_content.content_id}")
        print(f"Image dimensions: {image_content.width}x{image_content.height}")
        print(f"Format: {image_content.format.value}")
        print(f"Thumbnail created: {image_content.thumbnail_data is not None}")
        
        # Add accessibility features
        processor._ensure_accessibility(
            image_content,
            [AccessibilityFeature.ALT_TEXT, AccessibilityFeature.HIGH_CONTRAST]
        )
        
        print("Accessibility features:")
        for feature, value in image_content.accessibility.items():
            print(f"- {feature.value}: {value}")
            
        # Save the image content
        processor._save_content(image_content)
        print(f"Image saved with ID: {image_content.content_id}")
        
        # Demonstrate loading the image
        loaded_image = processor._load_content(image_content.content_id)
        print(f"Successfully loaded image: {loaded_image is not None}")
        
        # Demonstrate adaptation for different devices
        print("\nAdapting image for different devices:")
        devices = [
            DeviceCategory.DESKTOP,
            DeviceCategory.MOBILE,
            DeviceCategory.LOW_BANDWIDTH
        ]
        
        for device_category in devices:
            device_profile = DeviceProfile.get_default_profiles()[device_category]
            adapted_image = adapt_image_for_device(image_content, device_profile)
            
            print(f"- {device_category.value}: "
                  f"Adapted to {adapted_image.width}x{adapted_image.height}, "
                  f"size reduction: {adapted_image.metadata.get('size_reduction')}%")
        
    except Exception as e:
        print(f"Error during image processing demo: {str(e)}")


def adapt_image_for_device(
    image: ImageContent,
    device_profile: DeviceProfile
) -> ImageContent:
    """
    Adapt an image for a specific device profile.
    
    Args:
        image: The image to adapt
        device_profile: The target device profile
        
    Returns:
        Adapted image content
    """
    # This is a simplified demonstration - in a real implementation,
    # this would use more sophisticated adaptation techniques
    
    from PIL import Image
    import io
    import random  # For demonstration purposes only
    
    # Create a copy of the image metadata
    metadata = dict(image.metadata)
    
    # Parse the original image
    if isinstance(image.data, str):
        # Base64 string
        img_data = base64.b64decode(image.data)
        img = Image.open(io.BytesIO(img_data))
    else:
        # Raw bytes
        img = Image.open(io.BytesIO(image.data))
    
    original_width, original_height = img.size
    
    # Determine target size based on device
    if device_profile.category == DeviceCategory.MOBILE:
        # Mobile device - smaller size
        max_width = min(device_profile.screen_width or 375, 375)
        max_height = min(device_profile.screen_height or 667, 667)
    elif device_profile.category == DeviceCategory.LOW_BANDWIDTH:
        # Low bandwidth - even smaller and lower quality
        max_width = min(device_profile.screen_width or 375, 375)
        max_height = min(device_profile.screen_height or 667, 667)
        # Would also use higher compression in real implementation
    else:
        # Desktop or other - use original size, but cap at device dimensions
        max_width = device_profile.screen_width or original_width
        max_height = device_profile.screen_height or original_height
    
    # Calculate new dimensions while preserving aspect ratio
    if original_width > max_width or original_height > max_height:
        # Scale down
        width_ratio = max_width / original_width
        height_ratio = max_height / original_height
        ratio = min(width_ratio, height_ratio)
        
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        # Resize the image
        img = img.resize((new_width, new_height), Image.LANCZOS)
    else:
        new_width, new_height = original_width, original_height
    
    # Convert to bytes
    buffer = io.BytesIO()
    
    # Lower quality for low bandwidth
    quality = 95
    if device_profile.category == DeviceCategory.LOW_BANDWIDTH:
        quality = 70
        
    # Save with appropriate format and quality
    img.save(buffer, format=image.format.value.upper(), quality=quality)
    img_bytes = buffer.getvalue()
    
    # Calculate size reduction
    original_size = len(image.data) if isinstance(image.data, bytes) else len(base64.b64decode(image.data))
    new_size = len(img_bytes)
    size_reduction = round((1 - (new_size / original_size)) * 100, 1)
    
    # Update metadata
    metadata["original_width"] = original_width
    metadata["original_height"] = original_height
    metadata["size_reduction"] = size_reduction
    metadata["adapted_for"] = device_profile.category.value
    
    # Create new image content
    adapted_image = ImageContent(
        media_type=MediaType.IMAGE,
        content_id=f"{image.content_id}_adapted_{device_profile.category.value}",
        format=image.format,
        width=new_width,
        height=new_height,
        color_mode=image.color_mode,
        data=img_bytes,
        metadata=metadata,
        accessibility=image.accessibility.copy()
    )
    
    return adapted_image


def demonstrate_chart_generation(processor: MediaContentProcessor):
    """Demonstrate chart generation capabilities."""
    print("\n=== CHART GENERATION DEMONSTRATION ===")
    
    if not processor._check_dependencies():
        print("Skipping chart demo - matplotlib is required")
        return
    
    try:
        # Define data for different chart types
        chart_data = {
            ChartType.BAR: {
                "labels": ["Q1", "Q2", "Q3", "Q4"],
                "values": [15, 30, 25, 40],
                "title": "Quarterly Sales",
                "y_label": "Sales (in millions)"
            },
            ChartType.PIE: {
                "labels": ["Product A", "Product B", "Product C", "Product D"],
                "values": [35, 25, 20, 20],
                "title": "Product Market Share"
            },
            ChartType.LINE: {
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "y": [5, 7, 6, 9, 12, 14, 15, 18, 21, 19, 23, 25],
                "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                "title": "Monthly Growth",
                "x_label": "Month",
                "y_label": "Growth (%)"
            }
        }
        
        # Generate and show different chart types
        for chart_type, data in chart_data.items():
            print(f"\nGenerating {chart_type.value} chart...")
            
            # Create chart content
            chart_content = create_chart(
                processor,
                chart_type,
                data,
                f"Example {chart_type.value} chart"
            )
            
            print(f"Created chart with ID: {chart_content.content_id}")
            print(f"Chart type: {chart_content.chart_type.value}")
            print(f"Has rendered image: {chart_content.rendered_image is not None}")
            
            # Check accessibility features
            if AccessibilityFeature.ALT_TEXT in chart_content.accessibility:
                print(f"Alt text: {chart_content.accessibility[AccessibilityFeature.ALT_TEXT]}")
                
            # Demonstrate decision-making about when to use this chart type
            print(f"When to use {chart_type.value} charts:")
            print(get_chart_usage_guidance(chart_type))
            
        # Demonstrate adaptation for accessibility
        print("\nAdapting charts for accessibility:")
        chart_content = create_chart(
            processor,
            ChartType.BAR,
            chart_data[ChartType.BAR],
            "Accessible bar chart example"
        )
        
        # Add accessibility features
        accessible_chart = adapt_chart_for_accessibility(
            processor,
            chart_content,
            [AccessibilityFeature.COLOR_BLIND, AccessibilityFeature.HIGH_CONTRAST]
        )
        
        print("Accessibility features:")
        for feature, value in accessible_chart.accessibility.items():
            print(f"- {feature.value}: {value}")
            
    except Exception as e:
        print(f"Error during chart demo: {str(e)}")


def create_chart(
    processor: MediaContentProcessor,
    chart_type: ChartType,
    data: Dict[str, Any],
    title: str
) -> ChartContent:
    """
    Create a chart content entry.
    
    Args:
        processor: The media content processor
        chart_type: The type of chart to create
        data: The data for the chart
        title: The title of the chart
        
    Returns:
        Created chart content
    """
    import matplotlib.pyplot as plt
    import io
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    if chart_type == ChartType.BAR:
        plt.bar(data["labels"], data["values"])
        plt.xlabel("Categories")
        plt.ylabel(data.get("y_label", "Values"))
        
    elif chart_type == ChartType.PIE:
        plt.pie(data["values"], labels=data["labels"], autopct="%1.1f%%")
        
    elif chart_type == ChartType.LINE:
        plt.plot(data["x"], data["y"], marker='o')
        plt.xticks(data["x"], data["labels"])
        plt.xlabel(data.get("x_label", "X Axis"))
        plt.ylabel(data.get("y_label", "Y Axis"))
        
    # Add title and grid
    plt.title(data.get("title", title))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    plt.close()
    
    # Create chart content
    content_id = processor._generate_content_id(f"chart_{chart_type.value}")
    
    chart_content = ChartContent(
        media_type=MediaType.CHART,
        content_id=content_id,
        chart_type=chart_type,
        data=data,
        rendered_image=buffer.getvalue(),
        config={
            "title": title,
            "figure_size": (10, 6),
            "dpi": 100
        }
    )
    
    # Add alt text
    alt_text = processor._generate_chart_alt_text(chart_content)
    chart_content.accessibility[AccessibilityFeature.ALT_TEXT] = alt_text
    
    # Save the content
    processor._save_content(chart_content)
    
    return chart_content


def adapt_chart_for_accessibility(
    processor: MediaContentProcessor,
    chart: ChartContent,
    features: List[AccessibilityFeature]
) -> ChartContent:
    """
    Adapt a chart for accessibility.
    
    Args:
        processor: The media content processor
        chart: The chart to adapt
        features: The accessibility features to add
        
    Returns:
        Adapted chart content
    """
    # Return the original chart if matplotlib is not available
    if not processor._check_dependencies():
        return chart
    
    import matplotlib.pyplot as plt
    import io
    
    # Create a copy of the chart data and config
    data = dict(chart.data)
    config = dict(chart.config)
    
    # Update config for accessibility
    if AccessibilityFeature.COLOR_BLIND in features:
        # Use a colorblind-friendly palette
        config["colormap"] = "viridis"  # Colorblind-friendly colormap
        
    if AccessibilityFeature.HIGH_CONTRAST in features:
        # Increase contrast
        config["contrast"] = "high"
        config["linewidth"] = 3
        
    # Recreate the chart with accessibility features
    plt.figure(figsize=config.get("figure_size", (10, 6)))
    
    if chart.chart_type == ChartType.BAR:
        if AccessibilityFeature.COLOR_BLIND in features:
            # Use accessible colors and patterns
            bars = plt.bar(data["labels"], data["values"])
            
            # Add patterns to bars for additional distinction
            patterns = ['/', '\\', 'x', '.', 'o', '*']
            for i, bar in enumerate(bars):
                bar.set_hatch(patterns[i % len(patterns)])
                
        else:
            plt.bar(data["labels"], data["values"])
            
        plt.xlabel("Categories")
        plt.ylabel(data.get("y_label", "Values"))
        
    elif chart.chart_type == ChartType.PIE:
        plt.pie(
            data["values"], 
            labels=data["labels"], 
            autopct="%1.1f%%", 
            textprops={'fontsize': 14}
        )
        
    elif chart.chart_type == ChartType.LINE:
        plt.plot(
            data["x"], 
            data["y"], 
            marker='o', 
            linewidth=config.get("linewidth", 2),
            markersize=10
        )
        plt.xticks(data["x"], data["labels"])
        plt.xlabel(data.get("x_label", "X Axis"))
        plt.ylabel(data.get("y_label", "Y Axis"))
        
    # Add title and grid
    plt.title(data.get("title", config.get("title", "")), fontsize=16)
    plt.grid(True, linestyle='-', alpha=0.8)
    
    # Increase font sizes for better readability
    plt.rcParams.update({'font.size': 14})
    
    # Add direct text labels on data points for clarity
    if chart.chart_type == ChartType.BAR or chart.chart_type == ChartType.LINE:
        for i, v in enumerate(data["values"] if "values" in data else data["y"]):
            plt.text(
                i if chart.chart_type == ChartType.BAR else data["x"][i],
                v,
                str(v),
                ha='center',
                va='bottom',
                fontweight='bold'
            )
    
    # Save to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=config.get("dpi", 100))
    buffer.seek(0)
    plt.close()
    
    # Create adapted chart content
    content_id = f"{chart.content_id}_accessible"
    
    # Copy accessibility features and add new ones
    accessibility = dict(chart.accessibility)
    for feature in features:
        accessibility[feature] = True
        
    adapted_chart = ChartContent(
        media_type=MediaType.CHART,
        content_id=content_id,
        chart_type=chart.chart_type,
        data=data,
        rendered_image=buffer.getvalue(),
        config=config,
        accessibility=accessibility
    )
    
    # Generate detailed alt text for screen readers
    alt_text = processor._generate_chart_alt_text(adapted_chart)
    adapted_chart.accessibility[AccessibilityFeature.ALT_TEXT] = alt_text + " (Accessible version)"
    
    # Save the content
    processor._save_content(adapted_chart)
    
    return adapted_chart


def get_chart_usage_guidance(chart_type: ChartType) -> str:
    """Get guidance on when to use a particular chart type."""
    guidance = {
        ChartType.BAR: (
            "- Use for comparing discrete categories\n"
            "- Ideal for showing relative sizes or proportions\n"
            "- Good for displaying survey results, rankings, or categorical comparisons\n"
            "- Choose when numerical differences between categories are important"
        ),
        ChartType.PIE: (
            "- Use when showing parts of a whole or composition\n"
            "- Best for 6 or fewer categories\n"
            "- Ideal for percentage or proportional data\n"
            "- Most effective when one or two segments need to be highlighted"
        ),
        ChartType.LINE: (
            "- Use for showing trends over time or continuous data\n"
            "- Ideal for temporal data with many data points\n"
            "- Good for showing rate of change or patterns\n"
            "- Choose when the relationship between consecutive data points is important"
        ),
        ChartType.SCATTER: (
            "- Use for showing correlation between two variables\n"
            "- Ideal for identifying patterns, clusters, or outliers\n"
            "- Best when you need to show individual data points\n"
            "- Choose when distribution of data points matters"
        ),
        ChartType.HEATMAP: (
            "- Use for showing patterns in a matrix or grid\n"
            "- Ideal for displaying complex, multi-variable data\n"
            "- Good for showing intensity levels or frequency\n"
            "- Choose when data has two categorical axes"
        )
    }
    
    return guidance.get(chart_type, "No specific guidance available for this chart type.")


def demonstrate_document_generation(processor: MediaContentProcessor):
    """Demonstrate document generation capabilities."""
    print("\n=== DOCUMENT GENERATION DEMONSTRATION ===")
    
    try:
        # Create a markdown document
        print("Creating a markdown document...")
        markdown_content = """# Sample Report

## Executive Summary
This is a sample report generated by the MediaContentProcessor.

## Key Findings
- Finding 1: Lorem ipsum dolor sit amet
- Finding 2: Consectetur adipiscing elit
- Finding 3: Sed do eiusmod tempor incididunt

## Data Analysis
The following table summarizes our results:

| Metric | Value | Change |
|--------|-------|--------|
| Revenue | $1.2M | +15% |
| Costs | $0.8M | -5% |
| Profit | $0.4M | +30% |

## Conclusion
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
"""

        markdown_doc = DocumentContent(
            media_type=MediaType.DOCUMENT,
            content_id=processor._generate_content_id("doc_md"),
            format=DocumentFormat.MARKDOWN,
            title="Sample Markdown Report",
            content=markdown_content,
            metadata={
                "author": "Adaptive Bridge Builder",
                "date": "2025-05-16",
                "tags": ["report", "example", "markdown"]
            }
        )
        
        # Add accessibility features
        markdown_doc.accessibility[AccessibilityFeature.SCREEN_READER] = True
        markdown_doc.accessibility[AccessibilityFeature.TEXT_SCALING] = True
        
        processor._save_content(markdown_doc)
        print(f"Created markdown document with ID: {markdown_doc.content_id}")
        
        # Create an HTML document
        print("\nCreating an HTML document...")
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            grid-gap: 20px;
        }
        .card {
            background: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 20px;
        }
        .card h2 {
            margin-top: 0;
            color: #0066cc;
        }
        .metric {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        .change {
            font-size: 1em;
            padding: 3px 6px;
            border-radius: 3px;
            margin-left: 10px;
        }
        .positive {
            background-color: #e6f7e6;
            color: #2e8b57;
        }
        .negative {
            background-color: #ffeded;
            color: #d9534f;
        }
    </style>
</head>
<body>
    <h1>Interactive Dashboard</h1>
    <div class="dashboard">
        <div class="card">
            <h2>Revenue</h2>
            <div class="metric">$1.2M <span class="change positive">+15%</span></div>
            <p>Year-over-year growth exceeds market average.</p>
        </div>
        <div class="card">
            <h2>Costs</h2>
            <div class="metric">$0.8M <span class="change positive">-5%</span></div>
            <p>Cost reduction initiatives showing positive results.</p>
        </div>
        <div class="card">
            <h2>Profit</h2>
            <div class="metric">$0.4M <span class="change positive">+30%</span></div>
            <p>Significant improvement in profitability.</p>
        </div>
    </div>
</body>
</html>"""

        html_doc = DocumentContent(
            media_type=MediaType.DOCUMENT,
            content_id=processor._generate_content_id("doc_html"),
            format=DocumentFormat.HTML,
            title="Interactive Dashboard",
            content=html_content,
            metadata={
                "author": "Adaptive Bridge Builder",
                "date": "2025-05-16",
                "tags": ["dashboard", "interactive", "html"],
                "interactive": True
            }
        )
        
        # Add accessibility features for HTML
        html_doc.accessibility[AccessibilityFeature.SCREEN_READER] = True
        html_doc.accessibility[AccessibilityFeature.TEXT_SCALING] = True
        html_doc.accessibility[AccessibilityFeature.KEYBOARD_NAV] = True
        html_doc.accessibility[AccessibilityFeature.HIGH_CONTRAST] = True
        
        processor._save_content(html_doc)
        print(f"Created HTML document with ID: {html_doc.content_id}")
        
        # Demonstrate PDF generation
        if processor._check_dependencies():
            try:
                print("\nGenerating a PDF document...")
                pdf_doc = create_pdf_document(processor)
                print(f"Created PDF document with ID: {pdf_doc.content_id}")
            except Exception as e:
                print(f"PDF generation failed: {str(e)}")
        
        # Demonstrate when to use different document formats
        print("\nWhen to use different document formats:")
        formats = [
            DocumentFormat.MARKDOWN,
            DocumentFormat.HTML,
            DocumentFormat.PDF,
            DocumentFormat.PLAIN_TEXT
        ]
        
        for format in formats:
            print(f"\n{format.value.upper()}:")
            print(get_document_format_guidance(format))
                
    except Exception as e:
        print(f"Error during document demo: {str(e)}")


def create_pdf_document(processor: MediaContentProcessor) -> DocumentContent:
    """
    Create a PDF document.
    
    Args:
        processor: The media content processor
        
    Returns:
        Created document content
    """
    if not processor._check_dependencies():
        raise ImportError("reportlab is required for PDF generation")
    
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    import io
    
    # Create a buffer to write the PDF to
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        name='TitleStyle',
        parent=styles['Title'],
        fontSize=20,
        leading=24,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        name='HeadingStyle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.darkblue
    )
    
    # Create elements for the PDF
    elements = []
    
    # Add title
    elements.append(Paragraph("Quarterly Financial Report", title_style))
    elements.append(Spacer(1, 20))
    
    # Add introduction
    elements.append(Paragraph("Executive Summary", heading_style))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(
        "This report provides a summary of the company's financial performance for Q2 2025. "
        "Overall, the company has exceeded expectations with a 30% increase in profits "
        "compared to the same period last year.",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 15))
    
    # Add financial data
    elements.append(Paragraph("Financial Performance", heading_style))
    elements.append(Spacer(1, 10))
    
    # Create a table
    data = [
        ["Metric", "Q2 2024", "Q2 2025", "Change"],
        ["Revenue", "$1.0M", "$1.2M", "+20%"],
        ["Costs", "$0.85M", "$0.8M", "-5.9%"],
        ["Profit", "$0.15M", "$0.4M", "+166.7%"],
        ["Margin", "15%", "33.3%", "+18.3 pts"]
    ]
    
    # Create the table style
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.darkblue),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT')
    ])
    
    # Add a special style for the positive changes
    for i in range(2, len(data)):
        if "+" in data[i][3]:  # Positive change
            table_style.add('TEXTCOLOR', (3, i), (3, i), colors.green)
        elif "-" in data[i][3]:  # Negative change
            table_style.add('TEXTCOLOR', (3, i), (3, i), colors.red)
    
    # Create the table
    table = Table(data)
    table.setStyle(table_style)
    elements.append(table)
    elements.append(Spacer(1, 15))
    
    # Add conclusion
    elements.append(Paragraph("Conclusion", heading_style))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(
        "The financial data indicates strong growth in revenue and profit margins. "
        "Cost reduction initiatives have been effective, contributing to the "
        "significant improvement in overall profitability.",
        styles["Normal"]
    ))
    
    # Build the PDF
    doc.build(elements)
    
    # Get the PDF data
    pdf_data = buffer.getvalue()
    buffer.close()
    
    # Create document content
    content_id = processor._generate_content_id("doc_pdf")
    
    pdf_doc = DocumentContent(
        media_type=MediaType.DOCUMENT,
        content_id=content_id,
        format=DocumentFormat.PDF,
        title="Quarterly Financial Report",
        content=pdf_data,
        page_count=1,
        metadata={
            "author": "Adaptive Bridge Builder",
            "date": "2025-05-16",
            "tags": ["report", "financial", "pdf"],
            "generated": True
        }
    )
    
    # Add accessibility features for PDF
    pdf_doc.accessibility[AccessibilityFeature.SCREEN_READER] = True
    
    # Save the content
    processor._save_content(pdf_doc)
    
    return pdf_doc


def get_document_format_guidance(format: DocumentFormat) -> str:
    """Get guidance on when to use a particular document format."""
    guidance = {
        DocumentFormat.MARKDOWN: (
            "- Use for content that needs to be easily readable in plain text\n"
            "- Ideal for documentation, READMEs, and content that will be viewed in code repositories\n"
            "- Good for collaborative editing and version control\n"
            "- Choose when you need a simple, portable format that converts well to other formats"
        ),
        DocumentFormat.HTML: (
            "- Use for web-based content and interactive presentations\n"
            "- Ideal when you need precise control over styling and layout\n"
            "- Good for content that includes interactive elements\n"
            "- Choose when the content will be viewed in a browser\n"
            "- Best for dashboards, interactive reports, or content with complex layouts"
        ),
        DocumentFormat.PDF: (
            "- Use for formal documents that need to maintain exact formatting\n"
            "- Ideal for reports, contracts, and official documents\n"
            "- Good for documents that will be printed or shared externally\n"
            "- Choose when you need a fixed layout that looks the same on all devices\n"
            "- Best for final, non-editable versions of documents"
        ),
        DocumentFormat.PLAIN_TEXT: (
            "- Use for maximum compatibility and simplicity\n"
            "- Ideal for config files, logs, or simple notes\n"
            "- Good for content that needs to be processed by simple text parsers\n"
            "- Choose when formatting is not important and pure content is the focus\n"
            "- Best for system-readable content or when working with limited tooling"
        )
    }
    
    return guidance.get(format, "No specific guidance available for this format.")


def demonstrate_table_content(processor: MediaContentProcessor):
    """Demonstrate structured table data capabilities."""
    print("\n=== TABLE DATA DEMONSTRATION ===")
    
    try:
        # Create a simple data table
        print("Creating a structured data table...")
        
        # Define headers and rows
        headers = ["Product", "Category", "Price", "Stock", "Rating"]
        rows = [
            ["Laptop Pro", "Electronics", 1299.99, 45, 4.8],
            ["Smart Watch", "Electronics", 249.99, 120, 4.5],
            ["Office Chair", "Furniture", 189.99, 30, 4.2],
            ["Desk Lamp", "Home", 39.99, 80, 4.3],
            ["Wireless Headphones", "Electronics", 159.99, 65, 4.7]
        ]
        
        # Create table content
        table_content = TableContent(
            media_type=MediaType.TABLE,
            content_id=processor._generate_content_id("table"),
            headers=headers,
            rows=rows,
            column_types=["text", "text", "number", "number", "number"],
            summary="Product inventory table with 5 products across Electronics, Furniture, and Home categories"
        )
        
        # Add accessibility features
        table_content.accessibility[AccessibilityFeature.SCREEN_READER] = True
        
        # Save the content
        processor._save_content(table_content)
        print(f"Created table content with ID: {table_content.content_id}")
        print(f"Table dimensions: {len(table_content.rows)} rows x {len(table_content.headers)} columns")
        print(f"Summary: {table_content.summary}")
        
        # Demonstrate table analysis
        print("\nAnalyzing table data...")
        analysis = analyze_table(table_content)
        print(analysis)
        
        # Demonstrate conversion to other formats
        print("\nConverting table to different formats:")
        
        formats = ["CSV", "HTML Table", "Markdown Table"]
        for format in formats:
            print(f"\n{format}:")
            print(convert_table_to_format(table_content, format))
            
    except Exception as e:
        print(f"Error during table demo: {str(e)}")


def analyze_table(table: TableContent) -> str:
    """
    Analyze table data to extract insights.
    
    Args:
        table: The table content to analyze
        
    Returns:
        Analysis text
    """
    # Simple demo analysis - a real implementation would use more sophisticated analytics
    
    # Extract categories for grouping
    categories = {}
    category_idx = table.headers.index("Category")
    price_idx = table.headers.index("Price")
    stock_idx = table.headers.index("Stock")
    rating_idx = table.headers.index("Rating")
    
    for row in table.rows:
        category = row[category_idx]
        if category not in categories:
            categories[category] = {
                "count": 0,
                "total_price": 0,
                "total_stock": 0,
                "ratings": []
            }
        
        categories[category]["count"] += 1
        categories[category]["total_price"] += row[price_idx]
        categories[category]["total_stock"] += row[stock_idx]
        categories[category]["ratings"].append(row[rating_idx])
    
    # Generate analysis text
    analysis = "Table Analysis:\n\n"
    analysis += f"Total products: {len(table.rows)}\n"
    analysis += f"Categories: {', '.join(categories.keys())}\n\n"
    
    # Category breakdown
    analysis += "Category Breakdown:\n"
    for category, data in categories.items():
        avg_price = data["total_price"] / data["count"]
        avg_rating = sum(data["ratings"]) / len(data["ratings"])
        analysis += f"- {category}: {data['count']} products, ${avg_price:.2f} avg price, "
        analysis += f"{data['total_stock']} total stock, {avg_rating:.1f} avg rating\n"
    
    # Find highest/lowest values
    all_prices = [row[price_idx] for row in table.rows]
    all_ratings = [row[rating_idx] for row in table.rows]
    
    most_expensive_idx = all_prices.index(max(all_prices))
    cheapest_idx = all_prices.index(min(all_prices))
    best_rated_idx = all_ratings.index(max(all_ratings))
    
    analysis += f"\nMost expensive product: {table.rows[most_expensive_idx][0]} (${all_prices[most_expensive_idx]:.2f})\n"
    analysis += f"Cheapest product: {table.rows[cheapest_idx][0]} (${all_prices[cheapest_idx]:.2f})\n"
    analysis += f"Highest rated product: {table.rows[best_rated_idx][0]} ({all_ratings[best_rated_idx]} stars)\n"
    
    # Inventory value calculation
    total_inventory_value = sum(row[price_idx] * row[stock_idx] for row in table.rows)
    analysis += f"\nTotal inventory value: ${total_inventory_value:.2f}"
    
    return analysis


def convert_table_to_format(table: TableContent, format: str) -> str:
    """
    Convert a table to a specific textual format.
    
    Args:
        table: The table content to convert
        format: The format to convert to (CSV, HTML, Markdown)
        
    Returns:
        Formatted table as a string
    """
    if format == "CSV":
        lines = [",".join([str(h) for h in table.headers])]
        for row in table.rows:
            lines.append(",".join([str(cell) for cell in row]))
        return "\n".join(lines)
        
    elif format == "HTML Table":
        html = ["<table border='1'>"]
        
        # Add header row
        html.append("<thead><tr>")
        for header in table.headers:
            html.append(f"<th>{header}</th>")
        html.append("</tr></thead>")
        
        # Add data rows
        html.append("<tbody>")
        for row in table.rows:
            html.append("<tr>")
            for cell in row:
                # Format numbers with appropriate precision
                if isinstance(cell, float):
                    html.append(f"<td>{cell:.2f}</td>")
                else:
                    html.append(f"<td>{cell}</td>")
            html.append("</tr>")
        html.append("</tbody>")
        
        html.append("</table>")
        return "\n".join(html)
        
    elif format == "Markdown Table":
        # Create header row
        md = ["| " + " | ".join(str(h) for h in table.headers) + " |"]
        
        # Add separator row
        md.append("| " + " | ".join("---" for _ in table.headers) + " |")
        
        # Add data rows
        for row in table.rows:
            # Format numbers with appropriate precision
            formatted_row = []
            for cell in row:
                if isinstance(cell, float):
                    formatted_row.append(f"{cell:.2f}")
                else:
                    formatted_row.append(str(cell))
            
            md.append("| " + " | ".join(formatted_row) + " |")
            
        return "\n".join(md)
        
    else:
        return "Unsupported format"


def demonstrate_media_analysis(processor: MediaContentProcessor):
    """Demonstrate media content analysis capabilities."""
    print("\n=== MEDIA ANALYSIS DEMONSTRATION ===")
    
    try:
        # Create an example document for analysis
        content = """
# Project Proposal: Smart City Integration

## Executive Summary
This proposal outlines a comprehensive plan to implement smart city technologies in Downtown Metro.
The initiative will focus on three key areas: traffic management, energy efficiency, and public safety.

## Key Components
1. **Traffic Management System**
   - Intelligent traffic lights with real-time adjustment
   - Digital traffic flow monitoring using IoT sensors
   - Mobile app integration for route optimization

2. **Energy Efficiency**
   - Smart grid implementation for the downtown area
   - Energy-efficient LED street lighting with adaptive controls
   - Solar-powered charging stations for electric vehicles

3. **Public Safety Network**
   - High-definition security camera network
   - Emergency response system integration
   - Public Wi-Fi with enhanced security features

## Budget Estimate
The proposed budget for this initiative is $4.5 million, distributed as follows:
- Traffic Management: $1.8M
- Energy Efficiency: $1.5M
- Public Safety: $1.2M

## Timeline
- Phase 1 (Traffic Management): Q3 2025 - Q1 2026
- Phase 2 (Energy Efficiency): Q4 2025 - Q2 2026
- Phase 3 (Public Safety): Q1 2026 - Q3 2026

## Contact
For more information, please contact the Smart City Planning Office at city.planning@metroville.gov
"""

        # Create a document content object
        doc = DocumentContent(
            media_type=MediaType.DOCUMENT,
            content_id=processor._generate_content_id("analysis_doc"),
            format=DocumentFormat.MARKDOWN,
            title="Smart City Integration Proposal",
            content=content
        )
        
        processor._save_content(doc)
        print(f"Created document for analysis with ID: {doc.content_id}")
        
        # Analyze the document
        print("\nAnalyzing document content...")
        analysis_result = analyze_text_content(doc.content)
        
        print(f"Analysis completed in {analysis_result.processing_time:.2f}s (confidence: {analysis_result.confidence:.2f})")
        print(f"Extracted {len(analysis_result.entities)} entities")
        
        # Show extracted entities
        print("\nExtracted entities:")
        for entity in analysis_result.entities[:5]:  # Show first 5 entities
            print(f"- {entity['type']}: {entity['text']}")
            
        # Show extracted structured data
        print("\nExtracted structured data:")
        for key, value in analysis_result.extracted_data.items():
            if isinstance(value, dict):
                print(f"- {key}:")
                for subkey, subvalue in value.items():
                    print(f"  - {subkey}: {subvalue}")
            else:
                print(f"- {key}: {value}")
                
        # Show text summary
        if analysis_result.extracted_text:
            print("\nSummary:")
            print(analysis_result.extracted_text)
            
    except Exception as e:
        print(f"Error during media analysis demo: {str(e)}")


def analyze_text_content(content: str) -> AnalysisResult:
    """
    Analyze text content to extract entities, structure, and summary.
    
    Args:
        content: The text content to analyze
        
    Returns:
        Analysis result
    """
    from media_content_processor import AnalysisResult, MediaType
    import time
    import re
    
    # Start timing
    start_time = time.time()
    
    # Extract entities with simple regex patterns
    # In a real implementation, this would use NLP and entity recognition
    entities = []
    
    # Find dates
    date_pattern = r'Q[1-4]\s+\d{4}'
    for match in re.finditer(date_pattern, content):
        entities.append({
            "type": "date",
            "text": match.group(0),
            "position": match.span()
        })
    
    # Find monetary values
    money_pattern = r'\$\d+(?:\.\d+)?[MBKmb]?'
    for match in re.finditer(money_pattern, content):
        entities.append({
            "type": "money",
            "text": match.group(0),
            "position": match.span()
        })
    
    # Find percentages
    percent_pattern = r'\d+(?:\.\d+)?%'
    for match in re.finditer(percent_pattern, content):
        entities.append({
            "type": "percentage",
            "text": match.group(0),
            "position": match.span()
        })
    
    # Extract sections with headings
    section_pattern = r'##\s+(.*?)\n(.*?)(?=##|\Z)'
    sections = {}
    for match in re.finditer(section_pattern, content, re.DOTALL):
        heading = match.group(1).strip()
        section_content = match.group(2).strip()
        sections[heading] = section_content
        
        # Add heading as entity
        entities.append({
            "type": "section_heading",
            "text": heading,
            "position": match.span(1)
        })
    
    # Extract title
    title_match = re.search(r'#\s+(.*?)\n', content)
    title = title_match.group(1).strip() if title_match else "Untitled Document"
    
    # Extract structured data
    extracted_data = {
        "title": title,
        "sections": list(sections.keys())
    }
    
    # Extract budget information
    budget_section = sections.get("Budget Estimate", "")
    budget_matches = re.findall(r'(\w+(?:\s+\w+)?): \$(\d+\.\d+)M', budget_section)
    if budget_matches:
        budget = {}
        for item, amount in budget_matches:
            budget[item] = float(amount)
        extracted_data["budget"] = budget
        
    # Extract timeline information
    timeline_section = sections.get("Timeline", "")
    timeline_matches = re.findall(r'Phase \d+.*?: (Q\d \d{4} - Q\d \d{4})', timeline_section)
    if timeline_matches:
        extracted_data["timeline"] = timeline_matches
        
    # Generate a simple summary
    # In a real implementation, this would use NLP for better summarization
    lines = content.split("\n")
    summary_lines = []
    for line in lines:
        if line.startswith("##"):  # It's a section heading
            summary_lines.append(line.strip("# "))
        elif "summary" in line.lower() and ":" in line:
            # Found a summary statement
            summary_lines.append(line.split(":", 1)[1].strip())
            
    summary = " ".join(summary_lines)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Create and return analysis result
    result = AnalysisResult(
        content_id="doc_analysis",
        media_type=MediaType.DOCUMENT,
        success=True,
        extracted_data=extracted_data,
        extracted_text=summary,
        entities=entities,
        confidence=0.85,  # Simulated confidence score
        processing_time=processing_time
    )
    
    return result


def demonstrate_media_selection(processor: MediaContentProcessor):
    """Demonstrate how the agent decides which media type to use in different scenarios."""
    print("\n=== MEDIA SELECTION DEMONSTRATION ===")
    
    print("This demonstration shows how an agent might decide when to use different types of media")
    print("based on the communication context, user needs, and device constraints.\n")
    
    # Define different scenarios
    scenarios = [
        {
            "name": "Data Visualization Need",
            "description": "User needs to understand quarterly sales trends",
            "user_profile": {
                "technical_level": "business user",
                "device": DeviceCategory.DESKTOP,
                "preferences": ["visual", "interactive"]
            }
        },
        {
            "name": "Technical Documentation",
            "description": "Explaining API authentication process",
            "user_profile": {
                "technical_level": "developer",
                "device": DeviceCategory.DESKTOP,
                "preferences": ["code", "details"]
            }
        },
        {
            "name": "Mobile User Experience",
            "description": "Providing directions to a location",
            "user_profile": {
                "technical_level": "general",
                "device": DeviceCategory.MOBILE,
                "bandwidth_level": 2,
                "preferences": ["concise", "visual"]
            }
        },
        {
            "name": "Accessibility Needs",
            "description": "Sharing financial report with visually impaired user",
            "user_profile": {
                "technical_level": "business user",
                "device": DeviceCategory.SCREEN_READER,
                "accessibility_needs": ["screen_reader", "text_descriptions"],
                "preferences": ["structured", "detailed"]
            }
        },
        {
            "name": "Low Bandwidth Situation",
            "description": "Providing product catalog in rural area with poor connectivity",
            "user_profile": {
                "technical_level": "general",
                "device": DeviceCategory.LOW_BANDWIDTH,
                "bandwidth_level": 1,
                "preferences": ["basic", "text"]
            }
        }
    ]
    
    # Show media selection for each scenario
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Context: {scenario['description']}")
        print(f"User: {scenario['user_profile']['technical_level']} on {scenario['user_profile']['device'].value} device")
        
        # Get media recommendations
        recommendations = select_media_for_scenario(scenario)
        
        print("\nRecommended media types:")
        for i, (media_type, reason) in enumerate(recommendations, 1):
            print(f"{i}. {media_type}: {reason}")


def select_media_for_scenario(scenario: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Select appropriate media types for a given scenario.
    
    Args:
        scenario: Description of the communication scenario
        
    Returns:
        List of (media_type, reason) tuples in priority order
    """
    results = []
    description = scenario["description"]
    user_profile = scenario["user_profile"]
    device = user_profile["device"]
    
    # Scenario 1: Data Visualization Need
    if "sales trends" in description or "data" in description:
        if device != DeviceCategory.SCREEN_READER:
            results.append((
                "Interactive Chart",
                "An interactive line or bar chart allows users to visualize trends and filter data as needed"
            ))
            results.append((
                "Static Chart Image with Alt Text",
                "A static chart with descriptive alt text provides the key insights with good accessibility"
            ))
        if device == DeviceCategory.SCREEN_READER:
            results.append((
                "Structured Table with Headers",
                "A well-structured table with proper headers is optimal for screen readers to navigate data"
            ))
            results.append((
                "Text Summary with Key Points",
                "A concise text summary highlighting the main trends and observations"
            ))
            
    # Scenario 2: Technical Documentation
    elif "API" in description or "technical" in description:
        results.append((
            "Code Blocks with Markdown",
            "Syntax-highlighted code examples in a markdown document provide clear implementation guidance"
        ))
        results.append((
            "Sequence Diagram",
            "Visual representation of the authentication flow helps clarify the process"
        ))
        results.append((
            "Interactive Documentation",
            "Interactive documentation allows developers to test the API directly"
        ))
        
    # Scenario 3: Mobile User Experience
    elif "directions" in description and device == DeviceCategory.MOBILE:
        results.append((
            "Interactive Map",
            "An interactive, mobile-optimized map provides the most intuitive navigation experience"
        ))
        results.append((
            "Static Map Image with Text Directions",
            "A lightweight static map with step-by-step text directions as backup"
        ))
        results.append((
            "Text-Only Directions",
            "Concise text directions for very low bandwidth situations"
        ))
        
    # Scenario 4: Accessibility Needs
    elif "visually impaired" in description or device == DeviceCategory.SCREEN_READER:
        results.append((
            "Structured Document with Proper Headings",
            "Well-organized document with semantic headings for easy screen reader navigation"
        ))
        results.append((
            "Data Tables with Proper Headers and Captions",
            "Accessible tables with proper row/column headers and summary captions"
        ))
        results.append((
            "Text Summary with Key Points",
            "Concise textual summary highlighting the most important information"
        ))
        
    # Scenario 5: Low Bandwidth
    elif "low bandwidth" in description or "poor connectivity" in description or device == DeviceCategory.LOW_BANDWIDTH:
        results.append((
            "Text-Based Content",
            "Plain text is the most bandwidth-efficient format for content delivery"
        ))
        results.append((
            "Compressed Images",
            "Highly compressed, low-resolution images when visual content is necessary"
        ))
        results.append((
            "Progressive Loading Document",
            "Document that loads basic content first, with optional enhanced content"
        ))
    
    # Add general recommendations if nothing specific matches
    if not results:
        if device == DeviceCategory.DESKTOP:
            results.append((
                "Rich Interactive Content",
                "Desktop users can benefit from feature-rich interactive content"
            ))
        elif device == DeviceCategory.MOBILE:
            results.append((
                "Mobile-Optimized Visual Content",
                "Touch-friendly, responsive visual content optimized for smaller screens"
            ))
        elif device == DeviceCategory.SCREEN_READER:
            results.append((
                "Accessible Text Content",
                "Well-structured text content with proper semantic markup"
            ))
        elif device == DeviceCategory.LOW_BANDWIDTH:
            results.append((
                "Lightweight Text Content",
                "Minimal, text-focused content to ensure delivery in low-bandwidth situations"
            ))
    
    return results


def main():
    """Run all demonstrations."""
    print("MEDIA CONTENT PROCESSOR EXAMPLE DEMONSTRATIONS")
    print("==============================================")
    
    # Set up processor
    processor = setup_media_processor()
    
    # Run demonstrations
    demonstrate_image_processing(processor)
    demonstrate_chart_generation(processor)
    demonstrate_document_generation(processor)
    demonstrate_table_content(processor)
    demonstrate_media_analysis(processor)
    demonstrate_media_selection(processor)
    
    print("\nAll demonstrations completed.")


if __name__ == "__main__":
    main()
