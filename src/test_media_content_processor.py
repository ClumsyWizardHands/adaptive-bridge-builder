from PIL import Image
import markdown
#!/usr/bin/env python3
"""
Test Module for MediaContentProcessor

This module contains unit tests for the MediaContentProcessor class,
verifying its media generation, processing, analysis, and adaptation capabilities.
"""

import unittest
import os
import io
import tempfile
import shutil
import base64
from unittest.mock import Mock, patch, MagicMock
import json

from media_content_processor import (
    MediaContentProcessor, MediaType, ImageFormat, ChartType, DocumentFormat,
    DeviceCategory, AccessibilityFeature, DeviceProfile,
    ImageContent, ChartContent, DocumentContent, TableContent,
    AnalysisResult
)


class TestMediaContentProcessor(unittest.TestCase):
    """Test cases for the MediaContentProcessor class."""

    def setUp(self) -> None:
        """Set up test environment before each test."""
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        
        # Initialize the processor with the test directory
        self.processor = MediaContentProcessor(
            agent_id="test_agent",
            media_storage_path=self.test_dir
        )
        
        # Create test data
        self._create_test_data()

    def tearDown(self) -> None:
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def _create_test_data(self) -> None:
        """Create test data for testing."""
        # Create a simple test image
        self.test_image_data = b'\x89PNG\r\n\x1a\n' + os.urandom(100)  # Fake PNG header + random data
        
        # Create test document content
        self.test_doc_content = """# Test Document
        
## Section 1
This is test content.

## Section 2
More test content.
"""

        # Create test table data
        self.test_table_headers = ["Name", "Value", "Status"]
        self.test_table_rows = [
            ["Item 1", 100, "Active"],
            ["Item 2", 200, "Inactive"],
            ["Item 3", 300, "Active"]
        ]

    @patch('media_content_processor.Image')
    def test_create_image(self, mock_image) -> None:
        """Test creating an image content entry."""
        # Mock PIL Image behavior
        mock_img = MagicMock()
        mock_img.size = (800, 600)
        mock_img.format = "PNG"
        mock_img.mode = "RGB"
        
        mock_image.open.return_value = mock_img
        
        # Create a buffer with test image data
        buffer = io.BytesIO(self.test_image_data)
        
        # Create image content
        with patch('media_content_processor.HAS_PIL', True):
            image_content = self.processor.create_image(
                image_data=buffer,
                format=ImageFormat.PNG,
                alt_text="Test image",
                metadata={"test": "metadata"}
            )
            
            # Verify the image content
            self.assertEqual(image_content.media_type, MediaType.IMAGE)
            self.assertEqual(image_content.format, ImageFormat.PNG)
            self.assertEqual(image_content.width, 800)
            self.assertEqual(image_content.height, 600)
            self.assertEqual(image_content.color_mode, "RGB")
            self.assertEqual(image_content.metadata["test"], "metadata")
            self.assertIn(AccessibilityFeature.ALT_TEXT, image_content.accessibility)
            self.assertEqual(image_content.accessibility[AccessibilityFeature.ALT_TEXT], "Test image")

    def test_save_and_load_content(self) -> None:
        """Test saving and loading content."""
        # Create a simple media content
        content_id = "test_content_123"
        content = DocumentContent(
            media_type=MediaType.DOCUMENT,
            content_id=content_id,
            format=DocumentFormat.MARKDOWN,
            title="Test Document",
            content=self.test_doc_content
        )
        
        # Save content
        self.processor._save_content(content)
        
        # Check if file was created
        file_path = os.path.join(self.test_dir, f"{content_id}.json")
        self.assertTrue(os.path.exists(file_path))
        
        # Load content
        loaded_content = self.processor._load_content(content_id)
        
        # Verify loaded content
        self.assertIsNotNone(loaded_content)
        self.assertEqual(loaded_content.media_type, MediaType.DOCUMENT)
        self.assertEqual(loaded_content.content_id, content_id)
        self.assertEqual(loaded_content.format, DocumentFormat.MARKDOWN)
        self.assertEqual(loaded_content.title, "Test Document")
        self.assertEqual(loaded_content.content, self.test_doc_content)

    def test_ensure_accessibility(self) -> None:
        """Test ensuring accessibility features are added."""
        # Create a test image content without accessibility features
        image_content = ImageContent(
            media_type=MediaType.IMAGE,
            content_id="test_image_123",
            format=ImageFormat.PNG,
            width=800,
            height=600,
            color_mode="RGB",
            data=self.test_image_data
        )
        
        # Add accessibility features
        updated_content = self.processor._ensure_accessibility(
            image_content,
            [AccessibilityFeature.ALT_TEXT, AccessibilityFeature.HIGH_CONTRAST]
        )
        
        # Verify accessibility features
        self.assertIn(AccessibilityFeature.ALT_TEXT, updated_content.accessibility)
        self.assertIn(AccessibilityFeature.HIGH_CONTRAST, updated_content.accessibility)
        
        # Check if alt text was generated
        self.assertIsInstance(updated_content.accessibility[AccessibilityFeature.ALT_TEXT], str)
        self.assertTrue(len(updated_content.accessibility[AccessibilityFeature.ALT_TEXT]) > 0)

    def test_generate_content_id(self) -> None:
        """Test generating unique content IDs."""
        # Generate multiple IDs and check uniqueness
        ids = [self.processor._generate_content_id() for _ in range(10)]
        
        # Check that all IDs are unique
        self.assertEqual(len(ids), len(set(ids)))
        
        # Check ID format
        for id in ids:
            self.assertTrue(id.startswith("media_"))
            self.assertTrue(len(id) > 10)  # Should have timestamp and uuid

    def test_generate_alt_text(self) -> None:
        """Test generating alternative text for different media types."""
        # Test for image
        image_content = ImageContent(
            media_type=MediaType.IMAGE,
            content_id="test_image",
            format=ImageFormat.PNG,
            width=800,
            height=600,
            color_mode="RGB",
            data=self.test_image_data
        )
        
        alt_text = self.processor._generate_alt_text(image_content)
        self.assertIn("Image", alt_text)
        self.assertIn("800x600", alt_text)
        
        # Test for document
        doc_content = DocumentContent(
            media_type=MediaType.DOCUMENT,
            content_id="test_doc",
            format=DocumentFormat.MARKDOWN,
            title="Test Document",
            content=self.test_doc_content
        )
        
        alt_text = self.processor._generate_alt_text(doc_content)
        self.assertIn("Document", alt_text)
        self.assertIn("Test Document", alt_text)
        
        # Test for table
        table_content = TableContent(
            media_type=MediaType.TABLE,
            content_id="test_table",
            headers=self.test_table_headers,
            rows=self.test_table_rows
        )
        
        alt_text = self.processor._generate_alt_text(table_content)
        self.assertIn("Table", alt_text)
        self.assertIn("3 rows", alt_text)

    def test_device_profiles(self) -> None:
        """Test device profile functionality."""
        # Get default profiles
        profiles = DeviceProfile.get_default_profiles()
        
        # Check that profiles exist for all device categories
        for category in DeviceCategory:
            self.assertIn(category, profiles)
            
        # Check desktop profile properties
        desktop_profile = profiles[DeviceCategory.DESKTOP]
        self.assertEqual(desktop_profile.category, DeviceCategory.DESKTOP)
        self.assertTrue(desktop_profile.supports_images)
        self.assertTrue(desktop_profile.supports_interactive)
        self.assertEqual(desktop_profile.bandwidth_level, 3)
        
        # Check mobile profile properties
        mobile_profile = profiles[DeviceCategory.MOBILE]
        self.assertEqual(mobile_profile.category, DeviceCategory.MOBILE)
        self.assertTrue(mobile_profile.supports_images)
        self.assertTrue(mobile_profile.supports_interactive)
        self.assertEqual(mobile_profile.bandwidth_level, 2)
        
        # Check screen reader profile properties
        screen_reader_profile = profiles[DeviceCategory.SCREEN_READER]
        self.assertEqual(screen_reader_profile.category, DeviceCategory.SCREEN_READER)
        self.assertFalse(screen_reader_profile.supports_images)
        self.assertFalse(screen_reader_profile.supports_interactive)
        self.assertTrue(AccessibilityFeature.SCREEN_READER in screen_reader_profile.accessibility_features)
        
        # Test serialization and deserialization
        profile_dict = desktop_profile.to_dict()
        restored_profile = DeviceProfile.from_dict(profile_dict)
        
        self.assertEqual(restored_profile.category, desktop_profile.category)
        self.assertEqual(restored_profile.screen_width, desktop_profile.screen_width)
        self.assertEqual(restored_profile.supports_images, desktop_profile.supports_images)

    def test_analysis_result(self) -> None:
        """Test AnalysisResult functionality."""
        # Create an analysis result
        result = AnalysisResult(
            content_id="test_analysis",
            media_type=MediaType.DOCUMENT,
            success=True,
            extracted_data={"key1": "value1", "key2": "value2"},
            extracted_text="This is extracted text.",
            entities=[
                {"type": "entity1", "text": "text1"},
                {"type": "entity2", "text": "text2"}
            ],
            confidence=0.85,
            processing_time=1.5
        )
        
        # Check properties
        self.assertEqual(result.content_id, "test_analysis")
        self.assertEqual(result.media_type, MediaType.DOCUMENT)
        self.assertTrue(result.success)
        self.assertEqual(result.confidence, 0.85)
        self.assertEqual(result.processing_time, 1.5)
        self.assertEqual(len(result.entities), 2)
        self.assertEqual(result.extracted_text, "This is extracted text.")
        
        # Test serialization and deserialization
        result_dict = result.to_dict()
        restored_result = AnalysisResult.from_dict(result_dict)
        
        self.assertEqual(restored_result.content_id, result.content_id)
        self.assertEqual(restored_result.media_type, result.media_type)
        self.assertEqual(restored_result.success, result.success)
        self.assertEqual(restored_result.confidence, result.confidence)
        self.assertEqual(len(restored_result.entities), len(result.entities))
        
        # Test summary generation
        summary = result.get_summary()
        self.assertIsInstance(summary, str)
        self.assertIn("test_analysis", summary)
        self.assertIn("Confidence: 0.85", summary)

    def test_document_content_serialization(self) -> None:
        """Test DocumentContent serialization and deserialization."""
        # Create document content
        doc = DocumentContent(
            media_type=MediaType.DOCUMENT,
            content_id="test_doc_123",
            format=DocumentFormat.MARKDOWN,
            title="Test Document",
            content=self.test_doc_content,
            metadata={"author": "Test Author", "date": "2025-05-16"}
        )
        
        # Add accessibility feature
        doc.accessibility[AccessibilityFeature.SCREEN_READER] = True
        
        # Serialize
        doc_dict = doc.to_dict()
        
        # Check dictionary fields
        self.assertEqual(doc_dict["media_type"], "document")
        self.assertEqual(doc_dict["content_id"], "test_doc_123")
        self.assertEqual(doc_dict["format"], "markdown")
        self.assertEqual(doc_dict["title"], "Test Document")
        self.assertEqual(doc_dict["content"], self.test_doc_content)
        self.assertEqual(doc_dict["metadata"]["author"], "Test Author")
        self.assertTrue(doc_dict["accessibility"]["screen_reader"])
        
        # Deserialize
        restored_doc = DocumentContent.from_dict(doc_dict)
        
        # Check restored object
        self.assertEqual(restored_doc.media_type, MediaType.DOCUMENT)
        self.assertEqual(restored_doc.content_id, "test_doc_123")
        self.assertEqual(restored_doc.format, DocumentFormat.MARKDOWN)
        self.assertEqual(restored_doc.title, "Test Document")
        self.assertEqual(restored_doc.content, self.test_doc_content)
        self.assertEqual(restored_doc.metadata["author"], "Test Author")
        self.assertTrue(restored_doc.accessibility[AccessibilityFeature.SCREEN_READER])

    def test_table_content_serialization(self) -> None:
        """Test TableContent serialization and deserialization."""
        # Create table content
        table = TableContent(
            media_type=MediaType.TABLE,
            content_id="test_table_123",
            headers=self.test_table_headers,
            rows=self.test_table_rows,
            column_types=["text", "number", "text"],
            summary="Test table summary"
        )
        
        # Add accessibility feature
        table.accessibility[AccessibilityFeature.SCREEN_READER] = True
        
        # Serialize
        table_dict = table.to_dict()
        
        # Check dictionary fields
        self.assertEqual(table_dict["media_type"], "table")
        self.assertEqual(table_dict["content_id"], "test_table_123")
        self.assertEqual(table_dict["headers"], self.test_table_headers)
        self.assertEqual(table_dict["rows"], self.test_table_rows)
        self.assertEqual(table_dict["column_types"], ["text", "number", "text"])
        self.assertEqual(table_dict["summary"], "Test table summary")
        self.assertTrue(table_dict["accessibility"]["screen_reader"])
        
        # Deserialize
        restored_table = TableContent.from_dict(table_dict)
        
        # Check restored object
        self.assertEqual(restored_table.media_type, MediaType.TABLE)
        self.assertEqual(restored_table.content_id, "test_table_123")
        self.assertEqual(restored_table.headers, self.test_table_headers)
        self.assertEqual(restored_table.rows, self.test_table_rows)
        self.assertEqual(restored_table.column_types, ["text", "number", "text"])
        self.assertEqual(restored_table.summary, "Test table summary")
        self.assertTrue(restored_table.accessibility[AccessibilityFeature.SCREEN_READER])


if __name__ == '__main__':
    unittest.main()