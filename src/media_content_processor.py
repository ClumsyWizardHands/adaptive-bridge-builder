from PIL import Image
import html
import markdown
import reportlab
#!/usr/bin/env python3
"""
Media Content Processor for Adaptive Bridge Builder

This module provides capabilities for generating, processing, and analyzing various 
media types including images, charts, and structured documents. It enables the agent
to work with rich media content while ensuring accessibility and adaptability across
different devices and bandwidth constraints.
"""

import base64
import io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Set, Tuple, Union

# Optional imports - will be dynamically loaded when needed
# Import placeholders for type hints
try:
    import numpy as np
    import PIL
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    
try:
    import matplotlib
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MediaContentProcessor")


class MediaType(str, Enum):
    """Types of media content supported by the processor."""
    IMAGE = "image"               # Photos, diagrams, etc.
    CHART = "chart"               # Data visualizations
    DOCUMENT = "document"         # PDFs, Word docs, etc.
    TABLE = "table"               # Structured tabular data
    VIDEO = "video"               # Video content
    AUDIO = "audio"               # Audio content
    INTERACTIVE = "interactive"   # Interactive media (e.g., widgets)


class ImageFormat(str, Enum):
    """Supported image formats."""
    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    SVG = "svg"
    WEBP = "webp"


class ChartType(str, Enum):
    """Types of charts that can be generated."""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"


class DocumentFormat(str, Enum):
    """Supported document formats."""
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    PLAIN_TEXT = "plain_text"


class DeviceCategory(str, Enum):
    """Categories of devices for media adaptation."""
    DESKTOP = "desktop"
    TABLET = "tablet"
    MOBILE = "mobile"
    SCREEN_READER = "screen_reader"
    E_READER = "e_reader"
    LOW_BANDWIDTH = "low_bandwidth"


class AccessibilityFeature(str, Enum):
    """Features to ensure content accessibility."""
    ALT_TEXT = "alt_text"              # Alternative text for images
    CAPTIONS = "captions"              # Captions for videos/audio
    HIGH_CONTRAST = "high_contrast"    # Enhanced contrast for visibility
    SCREEN_READER = "screen_reader"    # Screen reader compatibility
    TEXT_SCALING = "text_scaling"      # Adjustable text size
    KEYBOARD_NAV = "keyboard_nav"      # Keyboard navigation support
    COLOR_BLIND = "color_blind"        # Color-blind friendly palettes


@dataclass
class MediaContent:
    """Base class for all media content."""
    media_type: MediaType
    content_id: str
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    accessibility: Dict[AccessibilityFeature, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "media_type": self.media_type.value,
            "content_id": self.content_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "accessibility": {k.value: v for k, v in self.accessibility.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MediaContent':
        """Create from dictionary representation."""
        data_copy = dict(data)
        data_copy["media_type"] = MediaType(data_copy["media_type"])
        
        if "accessibility" in data_copy:
            data_copy["accessibility"] = {
                AccessibilityFeature(k): v 
                for k, v in data_copy["accessibility"].items()
            }
            
        return cls(**data_copy)


@dataclass
class ImageContent(MediaContent):
    """Image content data."""
    format: ImageFormat = ImageFormat.PNG
    width: int = 0
    height: int = 0
    color_mode: str = "RGB"  # e.g., "RGB", "RGBA", "grayscale"
    data: Union[str, bytes] = field(default="", repr=False)  # Base64 string or raw bytes
    thumbnail_data: Optional[Union[str, bytes]] = field(default=None, repr=False)
    
    def __post_init__(self) -> Any:
        """Ensure media_type is set to IMAGE."""
        self.media_type = MediaType.IMAGE
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        result.update({
            "format": self.format.value,
            "width": self.width,
            "height": self.height,
            "color_mode": self.color_mode
        })
        
        # Ensure data is in base64 string format for JSON serialization
        if isinstance(self.data, bytes):
            result["data"] = base64.b64encode(self.data).decode('utf-8')
        else:
            result["data"] = self.data
            
        if self.thumbnail_data:
            if isinstance(self.thumbnail_data, bytes):
                result["thumbnail_data"] = base64.b64encode(self.thumbnail_data).decode('utf-8')
            else:
                result["thumbnail_data"] = self.thumbnail_data
                
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImageContent':
        """Create from dictionary representation."""
        data_copy = dict(data)
        data_copy["format"] = ImageFormat(data_copy["format"])
        
        # Handle accessibility
        if "accessibility" in data_copy:
            data_copy["accessibility"] = {
                AccessibilityFeature(k): v 
                for k, v in data_copy["accessibility"].items()
            }
            
        return cls(**data_copy)


@dataclass
class ChartContent(MediaContent):
    """Chart or visualization content."""
    chart_type: ChartType = ChartType.BAR
    data: Dict[str, Any] = field(default_factory=dict)  # The data used to generate the chart
    rendered_image: Optional[Union[str, bytes]] = field(default=None, repr=False)
    config: Dict[str, Any] = field(default_factory=dict)  # Chart configuration
    interactive: bool = False
    
    def __post_init__(self) -> Any:
        """Ensure media_type is set to CHART."""
        self.media_type = MediaType.CHART
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        result.update({
            "chart_type": self.chart_type.value,
            "data": self.data,
            "config": self.config,
            "interactive": self.interactive
        })
        
        if self.rendered_image:
            if isinstance(self.rendered_image, bytes):
                result["rendered_image"] = base64.b64encode(self.rendered_image).decode('utf-8')
            else:
                result["rendered_image"] = self.rendered_image
                
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChartContent':
        """Create from dictionary representation."""
        data_copy = dict(data)
        data_copy["chart_type"] = ChartType(data_copy["chart_type"])
        
        # Handle accessibility
        if "accessibility" in data_copy:
            data_copy["accessibility"] = {
                AccessibilityFeature(k): v 
                for k, v in data_copy["accessibility"].items()
            }
            
        return cls(**data_copy)


@dataclass
class DocumentContent(MediaContent):
    """Document content (PDF, HTML, etc.)."""
    format: DocumentFormat = DocumentFormat.PLAIN_TEXT
    title: str = ""
    content: Union[str, bytes] = ""  # Text content or binary data for PDFs
    page_count: Optional[int] = None
    table_of_contents: Optional[List[Dict[str, Any]]] = None
    
    def __post_init__(self) -> Any:
        """Ensure media_type is set to DOCUMENT."""
        self.media_type = MediaType.DOCUMENT
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        result.update({
            "format": self.format.value,
            "title": self.title,
            "page_count": self.page_count,
            "table_of_contents": self.table_of_contents
        })
        
        # Handle binary content
        if isinstance(self.content, bytes):
            result["content"] = base64.b64encode(self.content).decode('utf-8')
            result["is_binary"] = True
        else:
            result["content"] = self.content
            result["is_binary"] = False
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentContent':
        """Create from dictionary representation."""
        data_copy = dict(data)
        data_copy["format"] = DocumentFormat(data_copy["format"])
        
        # Handle binary content
        if data_copy.get("is_binary", False) and isinstance(data_copy["content"], str):
            data_copy["content"] = base64.b64decode(data_copy["content"])
            
        # Handle accessibility
        if "accessibility" in data_copy:
            data_copy["accessibility"] = {
                AccessibilityFeature(k): v 
                for k, v in data_copy["accessibility"].items()
            }
            
        return cls(**data_copy)


@dataclass
class TableContent(MediaContent):
    """Structured tabular data."""
    headers: List[str] = field(default_factory=list)
    rows: List[List[Any]] = field(default_factory=list)
    column_types: Optional[List[str]] = None  # e.g., "text", "number", "date"
    summary: Optional[str] = None  # Text summary of table content
    
    def __post_init__(self) -> Any:
        """Ensure media_type is set to TABLE."""
        self.media_type = MediaType.TABLE
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        result.update({
            "headers": self.headers,
            "rows": self.rows,
            "column_types": self.column_types,
            "summary": self.summary
        })
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TableContent':
        """Create from dictionary representation."""
        # Handle accessibility
        data_copy = dict(data)
        if "accessibility" in data_copy:
            data_copy["accessibility"] = {
                AccessibilityFeature(k): v 
                for k, v in data_copy["accessibility"].items()
            }
            
        return cls(**data_copy)


class AnalysisResult:
    """Result of media content analysis."""
    
    def __init__(
        self,
        content_id: str,
        media_type: MediaType,
        success: bool,
        extracted_data: Optional[Dict[str, Any]] = None,
        extracted_text: Optional[str] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
        confidence: float = 0.0,
        processing_time: float = 0.0,
        error: Optional[str] = None
    ):
        """
        Initialize analysis result.
        
        Args:
            content_id: ID of the analyzed content
            media_type: Type of the analyzed media
            success: Whether analysis was successful
            extracted_data: Structured data extracted from content
            extracted_text: Text extracted from content
            entities: Entities detected in content
            confidence: Confidence score of analysis (0.0-1.0)
            processing_time: Time taken for analysis in seconds
            error: Error message if analysis failed
        """
        self.content_id = content_id
        self.media_type = media_type
        self.success = success
        self.extracted_data = extracted_data or {}
        self.extracted_text = extracted_text or ""
        self.entities = entities or []
        self.confidence = confidence
        self.processing_time = processing_time
        self.error = error
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "content_id": self.content_id,
            "media_type": self.media_type.value,
            "success": self.success,
            "extracted_data": self.extracted_data,
            "extracted_text": self.extracted_text,
            "entities": self.entities,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "error": self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """Create from dictionary representation."""
        data_copy = dict(data)
        data_copy["media_type"] = MediaType(data_copy["media_type"])
        return cls(**data_copy)
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the analysis result."""
        if not self.success:
            return f"Analysis failed: {self.error}"
            
        summary = f"Analysis of {self.media_type.value} (ID: {self.content_id}):\n"
        
        if self.extracted_text:
            summary += f"Extracted text ({len(self.extracted_text)} chars): {self.extracted_text[:100]}...\n"
            
        if self.entities:
            summary += f"Detected {len(self.entities)} entities:\n"
            for entity in self.entities[:3]:  # Show first 3
                summary += f"- {entity.get('type', 'Unknown')}: {entity.get('text', 'N/A')}\n"
            if len(self.entities) > 3:
                summary += f"- And {len(self.entities) - 3} more...\n"
                
        if self.extracted_data:
            summary += "Extracted data includes: "
            summary += ", ".join(self.extracted_data.keys()) + "\n"
            
        summary += f"Confidence: {self.confidence:.2f}, Processing time: {self.processing_time:.2f}s"
        return summary


class DeviceProfile:
    """Profile describing device capabilities for media adaptation."""
    
    def __init__(
        self,
        category: DeviceCategory,
        screen_width: Optional[int] = None,
        screen_height: Optional[int] = None,
        supports_images: bool = True,
        supports_interactive: bool = True,
        bandwidth_level: int = 3,  # 1-low, 2-medium, 3-high
        supports_audio: bool = True,
        supports_video: bool = True,
        accessibility_features: List[AccessibilityFeature] = None
    ):
        """
        Initialize device profile.
        
        Args:
            category: Device category
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            supports_images: Whether device supports images
            supports_interactive: Whether device supports interactive content
            bandwidth_level: Available bandwidth level (1-3)
            supports_audio: Whether device supports audio
            supports_video: Whether device supports video
            accessibility_features: List of required accessibility features
        """
        self.category = category
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.supports_images = supports_images
        self.supports_interactive = supports_interactive
        self.bandwidth_level = bandwidth_level
        self.supports_audio = supports_audio
        self.supports_video = supports_video
        self.accessibility_features = accessibility_features or []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "category": self.category.value,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "supports_images": self.supports_images,
            "supports_interactive": self.supports_interactive,
            "bandwidth_level": self.bandwidth_level,
            "supports_audio": self.supports_audio,
            "supports_video": self.supports_video,
            "accessibility_features": [f.value for f in self.accessibility_features]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeviceProfile':
        """Create from dictionary representation."""
        data_copy = dict(data)
        data_copy["category"] = DeviceCategory(data_copy["category"])
        
        if "accessibility_features" in data_copy:
            data_copy["accessibility_features"] = [
                AccessibilityFeature(f) for f in data_copy["accessibility_features"]
            ]
            
        return cls(**data_copy)
    
    @classmethod
    def get_default_profiles(cls) -> Dict[DeviceCategory, 'DeviceProfile']:
        """Get default device profiles for common device categories."""
        return {
            DeviceCategory.DESKTOP: cls(
                category=DeviceCategory.DESKTOP,
                screen_width=1920,
                screen_height=1080,
                supports_images=True,
                supports_interactive=True,
                bandwidth_level=3,
                supports_audio=True,
                supports_video=True
            ),
            DeviceCategory.TABLET: cls(
                category=DeviceCategory.TABLET,
                screen_width=1024,
                screen_height=768,
                supports_images=True,
                supports_interactive=True,
                bandwidth_level=2,
                supports_audio=True,
                supports_video=True
            ),
            DeviceCategory.MOBILE: cls(
                category=DeviceCategory.MOBILE,
                screen_width=375,
                screen_height=667,
                supports_images=True,
                supports_interactive=True,
                bandwidth_level=2,
                supports_audio=True,
                supports_video=True
            ),
            DeviceCategory.SCREEN_READER: cls(
                category=DeviceCategory.SCREEN_READER,
                supports_images=False,
                supports_interactive=False,
                bandwidth_level=2,
                supports_audio=True,
                supports_video=False,
                accessibility_features=[
                    AccessibilityFeature.ALT_TEXT,
                    AccessibilityFeature.SCREEN_READER
                ]
            ),
            DeviceCategory.LOW_BANDWIDTH: cls(
                category=DeviceCategory.LOW_BANDWIDTH,
                screen_width=800,
                screen_height=600,
                supports_images=True,
                supports_interactive=False,
                bandwidth_level=1,
                supports_audio=False,
                supports_video=False
            )
        }


class MediaContentProcessor:
    """
    Processes and analyzes various media types.
    
    Provides capabilities for generating, processing, and analyzing images,
    charts, documents, and other media content with considerations for
    accessibility and device adaptation.
    """
    
    def __init__(
        self,
        agent_id: str,
        media_storage_path: Optional[str] = None,
        default_device_profile: Optional[DeviceProfile] = None
    ):
        """
        Initialize media content processor.
        
        Args:
            agent_id: ID of the agent using this processor
            media_storage_path: Path for storing generated media files
            default_device_profile: Default device profile for adaptation
        """
        self.agent_id = agent_id
        self.media_storage_path = media_storage_path
        
        # Create storage directory if it doesn't exist
        if media_storage_path and not os.path.exists(media_storage_path):
            try:
                os.makedirs(media_storage_path)
                logger.info(f"Created media storage directory: {media_storage_path}")
            except Exception as e:
                logger.error(f"Failed to create media storage directory: {str(e)}")
                
        # Set default device profile (desktop if not specified)
        if default_device_profile:
            self.default_device_profile = default_device_profile
        else:
            profiles = DeviceProfile.get_default_profiles()
            self.default_device_profile = profiles[DeviceCategory.DESKTOP]
            
        # Store content by ID
        self.content_store: Dict[str, MediaContent] = {}
        
        # Dependency checking
        self._check_dependencies()
        
        logger.info(f"MediaContentProcessor initialized for agent {agent_id}")
        
    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        missing = []
        
        if not HAS_PIL:
            missing.append("PIL (Pillow)")
        if not HAS_MPL:
            missing.append("matplotlib")
        if not HAS_PANDAS:
            missing.append("pandas")
        if not HAS_REPORTLAB:
            missing.append("reportlab")
            
        if missing:
            logger.warning(f"Some media processing features will be limited. "
                           f"Missing dependencies: {', '.join(missing)}")
            
    def _generate_content_id(self, prefix: str = "media") -> str:
        """Generate a unique content ID."""
        import uuid
        return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
    def _save_content(self, content: MediaContent) -> None:
        """Save content to the content store."""
        self.content_store = {**self.content_store, content.content_id: content}
        
        # If storage path is configured, save to disk
        if self.media_storage_path:
            try:
                file_path = os.path.join(self.media_storage_path, f"{content.content_id}.json")
                with open(file_path, 'w') as f:
                    json.dump(content.to_dict(), f, indent=2)
                logger.debug(f"Saved content to {file_path}")
            except Exception as e:
                logger.error(f"Failed to save content to disk: {str(e)}")
                
    def _load_content(self, content_id: str) -> Optional[MediaContent]:
        """Load content from the content store or disk."""
        # First check in-memory store
        if content_id in self.content_store:
            return self.content_store[content_id]
            
        # If not found and storage path is configured, try loading from disk
        if self.media_storage_path:
            try:
                file_path = os.path.join(self.media_storage_path, f"{content_id}.json")
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Create appropriate content type based on media type
                    media_type = MediaType(data["media_type"])
                    
                    if media_type == MediaType.IMAGE:
                        content = ImageContent.from_dict(data)
                    elif media_type == MediaType.CHART:
                        content = ChartContent.from_dict(data)
                    elif media_type == MediaType.DOCUMENT:
                        content = DocumentContent.from_dict(data)
                    elif media_type == MediaType.TABLE:
                        content = TableContent.from_dict(data)
                    else:
                        content = MediaContent.from_dict(data)
                        
                    # Add to in-memory store and return
                    self.content_store = {**self.content_store, content_id: content}
                    return content
            except Exception as e:
                logger.error(f"Failed to load content {content_id} from disk: {str(e)}")
                
        return None
        
    def _ensure_accessibility(
        self,
        content: MediaContent,
        required_features: List[AccessibilityFeature]
    ) -> MediaContent:
        """
        Ensure content has required accessibility features.
        
        Args:
            content: The media content to check
            required_features: List of required accessibility features
            
        Returns:
            Updated content with accessibility features
        """
        for feature in required_features:
            # Skip if feature is already present
            if feature in content.accessibility:
                continue
                
            # Add missing features based on media type and feature
            if feature == AccessibilityFeature.ALT_TEXT:
                if content.media_type == MediaType.IMAGE:
                    content.accessibility[feature] = self._generate_alt_text(content)
                elif content.media_type == MediaType.CHART:
                    content.accessibility[feature] = self._generate_chart_alt_text(content)
                    
            elif feature == AccessibilityFeature.HIGH_CONTRAST:
                if content.media_type == MediaType.IMAGE:
                    # This would actually transform the image in a real implementation
                    content.accessibility[feature] = True
                elif content.media_type == MediaType.CHART:
                    # This would adjust chart colors in a real implementation
                    content.accessibility[feature] = True
                    
            elif feature == AccessibilityFeature.COLOR_BLIND:
                if content.media_type == MediaType.CHART:
                    # This would use colorblind-friendly palettes in a real implementation
                    content.accessibility[feature] = True
                    
        return content
        
    def _generate_alt_text(self, content: MediaContent) -> str:
        """
        Generate alternative text for a media content.
        
        Args:
            content: The media content to generate alt text for
            
        Returns:
            Generated alt text
        """
        # In a real implementation, this would use image recognition or other AI features
        # to generate meaningful descriptions
        
        if content.media_type == MediaType.IMAGE:
            if isinstance(content, ImageContent):
                return f"Image: {content.width}x{content.height} {content.format.value}"
            return "Image"
        elif content.media_type == MediaType.CHART:
            if isinstance(content, ChartContent):
                return f"Chart: {content.chart_type.value} visualization"
            return "Chart"
        elif content.media_type == MediaType.DOCUMENT:
            if isinstance(content, DocumentContent):
                return f"Document: {content.title} ({content.format.value})"
            return "Document"
        elif content.media_type == MediaType.TABLE:
            if isinstance(content, TableContent):
                return f"Table: {len(content.rows)} rows x {len(content.headers)} columns"
            return "Table"
        else:
            return f"{content.media_type.value.capitalize()}"
            
    def _generate_chart_alt_text(self, content: ChartContent) -> str:
        """
        Generate descriptive alternative text for a chart.
        
        Args:
            content: The chart content to generate alt text for
            
        Returns:
            Descriptive alt text
        """
        # This would use chart data to generate a description in a real implementation
        chart_type = content.chart_type.value
        data_desc = ""
        
        if content.data:
            if "labels" in content.data and "values" in content.data:
                # Simple label-value data
                labels = content.data["labels"]
                values = content.data["values"]
                
                if len(labels) <= 5:
                    # Describe all points for small datasets
                    points = [f"{labels[i]}: {values[i]}" for i in range(len(labels))]
                    data_desc = ", ".join(points)
                else:
                    # Summarize larger datasets
                    data_desc = (f"{len(labels)} data points, "
                                 f"ranging from {min(values)} to {max(values)}")
        
        return f"{chart_type.capitalize()} chart showing {data_desc or 'data visualization'}"
        
    def create_image(
        self,
        image_data: Union[str, bytes, BinaryIO],
        format: Optional[ImageFormat] = None,
        alt_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ImageContent:
        """
        Create an image content entry from data.
        
        Args:
            image_data: Image data as file path, bytes, or file-like object
            format: Image format (auto-detected if not specified)
            alt_text: Alternative text for the image
            metadata: Additional metadata for the image
            
        Returns:
            Created image content
        """
        if not HAS_PIL:
            raise ImportError("PIL (Pillow) is required for image processing")
            
        # Load the image
        if isinstance(image_data, str):
            if os.path.exists(image_data):
                # It's a file path
                img = Image.open(image_data)
            else:
                # It's a base64 string
                import base64
                img_data = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(img_data))
        elif isinstance(image_data, bytes):
            img = Image.open(io.BytesIO(image_data))
        else:
            # Assume it's a file-like object
            img = Image.open(image_data)
            
        # Get image properties
        width, height = img.size
        format = format or ImageFormat(img.format.lower())
        mode = img.mode
        
        # Create thumbnail
        thumbnail_data = None
        if max(width, height) > 300:
            img_thumb = img.copy()
            img_thumb.thumbnail((300, 300))
            thumb_buffer = io.BytesIO()
            img_thumb.save(thumb_buffer, format=img.format)
            thumbnail_data = thumb_buffer.getvalue()
            
        # Convert image to bytes if it's not already
        if isinstance(image_data, str) and os.path.exists(image_data):
            with open(image_data, 'rb') as f:
                img_bytes = f.read()
        elif isinstance(image_data, bytes):
            img_bytes = image_data
        else:
            buffer = io.BytesIO()
            img.save(buffer, format=img.format)
            img