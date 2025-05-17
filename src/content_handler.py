"""
Content Handler Module

This module provides functionality for handling, transforming, and validating
content across various formats and media types. It supports multiple data formats
and ensures consistent processing across the agent ecosystem.
"""

import json
import base64
import logging
import os
import re
import mimetypes
from typing import Dict, List, Any, Optional, Union, BinaryIO
from enum import Enum, auto

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ContentHandler")


class ContentType(Enum):
    """Types of content that can be handled."""
    TEXT = auto()
    JSON = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()
    BINARY = auto()
    MULTIPART = auto()
    UNKNOWN = auto()


class ContentFormat(Enum):
    """Specific formats for content."""
    PLAIN_TEXT = "text/plain"
    HTML = "text/html"
    MARKDOWN = "text/markdown"
    JSON = "application/json"
    XML = "application/xml"
    YAML = "application/yaml"
    CSV = "text/csv"
    JPEG = "image/jpeg"
    PNG = "image/png"
    GIF = "image/gif"
    SVG = "image/svg+xml"
    MP3 = "audio/mpeg"
    WAV = "audio/wav"
    MP4 = "video/mp4"
    WEBM = "video/webm"
    PDF = "application/pdf"
    ZIP = "application/zip"
    BINARY = "application/octet-stream"
    MULTIPART = "multipart/mixed"
    UNKNOWN = "application/unknown"


class ContentEncoding(Enum):
    """Encoding methods for content."""
    NONE = "none"
    BASE64 = "base64"
    UTF8 = "utf-8"
    ASCII = "ascii"
    ISO_8859_1 = "iso-8859-1"
    GZIP = "gzip"
    DEFLATE = "deflate"
    CUSTOM = "custom"


class ContentValidationError(Exception):
    """Exception raised for content validation errors."""
    pass


class ContentHandler:
    """
    Handles content processing, validation, transformation, and exchange
    between agents and systems in various formats.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the content handler.
        
        Args:
            config: Optional configuration settings
        """
        self.config = config or {}
        self.supported_types = {ct.name: ct for ct in ContentType}
        self.supported_formats = {cf.value: cf for cf in ContentFormat}
        self.supported_encodings = {ce.value: ce for ce in ContentEncoding}
        
        # Initialize format validators
        self.validators = {
            ContentFormat.JSON: self._validate_json,
            ContentFormat.XML: self._validate_xml,
            ContentFormat.YAML: self._validate_yaml,
            ContentFormat.CSV: self._validate_csv,
            ContentFormat.HTML: self._validate_html,
            ContentFormat.MARKDOWN: self._validate_markdown
        }
        
        # Initialize content transformers
        self.transformers = {
            (ContentFormat.JSON, ContentFormat.YAML): self._json_to_yaml,
            (ContentFormat.YAML, ContentFormat.JSON): self._yaml_to_json,
            (ContentFormat.JSON, ContentFormat.XML): self._json_to_xml,
            (ContentFormat.XML, ContentFormat.JSON): self._xml_to_json,
            (ContentFormat.HTML, ContentFormat.MARKDOWN): self._html_to_markdown,
            (ContentFormat.MARKDOWN, ContentFormat.HTML): self._markdown_to_html,
            (ContentFormat.CSV, ContentFormat.JSON): self._csv_to_json,
            (ContentFormat.JSON, ContentFormat.CSV): self._json_to_csv
        }
        
        # Custom MIME type detection
        mimetypes.init()
        mimetypes.add_type('text/markdown', '.md')
        mimetypes.add_type('text/markdown', '.markdown')
        mimetypes.add_type('application/yaml', '.yaml')
        mimetypes.add_type('application/yaml', '.yml')
        
        logger.info("ContentHandler initialized")
    
    def identify_content(self, 
                        content: Union[str, bytes, Dict, List],
                        hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Identify the type, format, and encoding of content.
        
        Args:
            content: The content to identify
            hint: Optional hint about the content (filename, MIME type, etc.)
            
        Returns:
            Dictionary with identified content metadata
        """
        result = {
            "type": ContentType.UNKNOWN,
            "format": ContentFormat.UNKNOWN,
            "encoding": ContentEncoding.NONE,
            "size": 0,
            "confidence": 0.0
        }
        
        # Determine size
        if isinstance(content, (str, bytes)):
            result["size"] = len(content)
        elif isinstance(content, (dict, list)):
            result["size"] = len(json.dumps(content))
        
        # Use hint if provided
        if hint:
            if hint.startswith('data:'):
                # Data URL format
                mime_match = re.match(r'data:([^;,]+)', hint)
                if mime_match:
                    mime_type = mime_match.group(1)
                    result["format"] = self.supported_formats.get(mime_type, ContentFormat.UNKNOWN)
                    result["confidence"] = 0.9
                    
                    if "base64" in hint:
                        result["encoding"] = ContentEncoding.BASE64
            elif os.path.exists(hint):
                # Treat hint as file path
                mime_type, _ = mimetypes.guess_type(hint)
                if mime_type:
                    result["format"] = self.supported_formats.get(mime_type, ContentFormat.UNKNOWN)
                    result["confidence"] = 0.8
            elif hint in self.supported_formats:
                # Direct MIME type
                result["format"] = self.supported_formats[hint]
                result["confidence"] = 0.9
        
        # Identify type and format based on content
        if isinstance(content, str):
            if result["format"] == ContentFormat.UNKNOWN:
                # Try to determine format from content
                if content.startswith('{') and content.endswith('}'):
                    try:
                        json.loads(content)
                        result["format"] = ContentFormat.JSON
                        result["confidence"] = 0.9
                    except:
                        pass
                elif content.startswith('<') and content.endswith('>'):
                    if '<html' in content.lower():
                        result["format"] = ContentFormat.HTML
                        result["confidence"] = 0.9
                    else:
                        result["format"] = ContentFormat.XML
                        result["confidence"] = 0.7
                elif content.startswith('#'):
                    result["format"] = ContentFormat.MARKDOWN
                    result["confidence"] = 0.6
                elif re.match(r'^[a-zA-Z0-9+/]+={0,2}$', content):
                    # Looks like base64
                    try:
                        decoded = base64.b64decode(content)
                        result["encoding"] = ContentEncoding.BASE64
                        # Try to identify the decoded content
                        if decoded.startswith(b'\xff\xd8'):
                            result["format"] = ContentFormat.JPEG
                            result["confidence"] = 0.9
                        elif decoded.startswith(b'\x89PNG'):
                            result["format"] = ContentFormat.PNG
                            result["confidence"] = 0.9
                    except:
                        # Not valid base64
                        pass
                else:
                    # Default to plain text if nothing else matches
                    result["format"] = ContentFormat.PLAIN_TEXT
                    result["confidence"] = 0.5
            
            # Determine content type based on format
            if result["format"] in [ContentFormat.PLAIN_TEXT, ContentFormat.HTML, 
                                  ContentFormat.MARKDOWN, ContentFormat.JSON, 
                                  ContentFormat.XML, ContentFormat.YAML, 
                                  ContentFormat.CSV]:
                result["type"] = ContentType.TEXT
            elif result["format"] in [ContentFormat.JPEG, ContentFormat.PNG, 
                                    ContentFormat.GIF, ContentFormat.SVG]:
                result["type"] = ContentType.IMAGE
            elif result["format"] in [ContentFormat.MP3, ContentFormat.WAV]:
                result["type"] = ContentType.AUDIO
            elif result["format"] in [ContentFormat.MP4, ContentFormat.WEBM]:
                result["type"] = ContentType.VIDEO
        
        elif isinstance(content, bytes):
            result["type"] = ContentType.BINARY
            # Try to identify binary format
            if content.startswith(b'\xff\xd8'):
                result["format"] = ContentFormat.JPEG
                result["type"] = ContentType.IMAGE
                result["confidence"] = 0.9
            elif content.startswith(b'\x89PNG'):
                result["format"] = ContentFormat.PNG
                result["type"] = ContentType.IMAGE
                result["confidence"] = 0.9
            elif content.startswith(b'GIF8'):
                result["format"] = ContentFormat.GIF
                result["type"] = ContentType.IMAGE
                result["confidence"] = 0.9
            elif content.startswith(b'ID3') or content.startswith(b'\xff\xfb'):
                result["format"] = ContentFormat.MP3
                result["type"] = ContentType.AUDIO
                result["confidence"] = 0.9
            else:
                result["format"] = ContentFormat.BINARY
                result["confidence"] = 0.5
        
        elif isinstance(content, dict):
            result["type"] = ContentType.JSON
            result["format"] = ContentFormat.JSON
            result["confidence"] = 1.0
        
        elif isinstance(content, list):
            result["type"] = ContentType.JSON
            result["format"] = ContentFormat.JSON
            result["confidence"] = 1.0
        
        return result
    
    def validate_content(self, 
                        content: Union[str, bytes, Dict, List],
                        expected_format: Union[str, ContentFormat],
                        strict: bool = False) -> Dict[str, Any]:
        """
        Validate content against expected format.
        
        Args:
            content: The content to validate
            expected_format: The expected format
            strict: Whether to perform strict validation
            
        Returns:
            Dictionary with validation results
        """
        if isinstance(expected_format, str):
            if expected_format in self.supported_formats:
                expected_format = self.supported_formats[expected_format]
            else:
                try:
                    expected_format = ContentFormat(expected_format)
                except:
                    expected_format = ContentFormat.UNKNOWN
        
        result = {
            "valid": False,
            "format": expected_format,
            "errors": []
        }
        
        # Skip validation for unknown format
        if expected_format == ContentFormat.UNKNOWN:
            result["valid"] = True
            result["errors"].append("Unknown format, skipping validation")
            return result
        
        # Get appropriate validator
        validator = self.validators.get(expected_format)
        if validator:
            validation_result = validator(content, strict)
            result.update(validation_result)
        else:
            # Basic validation for formats without specific validators
            try:
                # Identify content
                identified = self.identify_content(content)
                
                # Check if identified format matches expected format
                if identified["format"] == expected_format:
                    result["valid"] = True
                else:
                    result["valid"] = False
                    result["errors"].append(
                        f"Content identified as {identified['format'].value}, " +
                        f"but expected {expected_format.value}"
                    )
            except Exception as e:
                result["valid"] = False
                result["errors"].append(f"Validation error: {str(e)}")
        
        return result
    
    def transform_content(self,
                         content: Union[str, bytes, Dict, List],
                         source_format: Union[str, ContentFormat],
                         target_format: Union[str, ContentFormat],
                         options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Transform content from one format to another.
        
        Args:
            content: The content to transform
            source_format: The current format of the content
            target_format: The desired target format
            options: Optional transformation options
            
        Returns:
            Dictionary with transformed content and metadata
        """
        options = options or {}
        
        # Convert string format names to enum if needed
        if isinstance(source_format, str):
            if source_format in self.supported_formats:
                source_format = self.supported_formats[source_format]
            else:
                try:
                    source_format = ContentFormat(source_format)
                except:
                    source_format = ContentFormat.UNKNOWN
        
        if isinstance(target_format, str):
            if target_format in self.supported_formats:
                target_format = self.supported_formats[target_format]
            else:
                try:
                    target_format = ContentFormat(target_format)
                except:
                    target_format = ContentFormat.UNKNOWN
        
        result = {
            "success": False,
            "transformed_content": None,
            "source_format": source_format,
            "target_format": target_format,
            "errors": []
        }
        
        # If formats are the same, no transformation needed
        if source_format == target_format:
            result["success"] = True
            result["transformed_content"] = content
            return result
        
        # Get appropriate transformer
        transformer = self.transformers.get((source_format, target_format))
        if transformer:
            try:
                transformed = transformer(content, options)
                result["success"] = True
                result["transformed_content"] = transformed
            except Exception as e:
                result["errors"].append(f"Transformation error: {str(e)}")
        else:
            result["errors"].append(
                f"No transformer available from {source_format.value} to {target_format.value}"
            )
        
        return result
    
    def encode_content(self,
                      content: Union[str, bytes],
                      encoding: Union[str, ContentEncoding],
                      options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Encode content using the specified encoding.
        
        Args:
            content: The content to encode
            encoding: The encoding to use
            options: Optional encoding options
            
        Returns:
            Dictionary with encoded content and metadata
        """
        options = options or {}
        
        # Convert string encoding name to enum if needed
        if isinstance(encoding, str):
            if encoding in self.supported_encodings:
                encoding = self.supported_encodings[encoding]
            else:
                try:
                    encoding = ContentEncoding(encoding)
                except:
                    encoding = ContentEncoding.UNKNOWN
        
        result = {
            "success": False,
            "encoded_content": None,
            "encoding": encoding,
            "errors": []
        }
        
        try:
            if encoding == ContentEncoding.NONE:
                # No encoding, return as is
                result["success"] = True
                result["encoded_content"] = content
            
            elif encoding == ContentEncoding.BASE64:
                # Base64 encoding
                if isinstance(content, str):
                    encoded = base64.b64encode(content.encode('utf-8')).decode('ascii')
                else:
                    encoded = base64.b64encode(content).decode('ascii')
                
                result["success"] = True
                result["encoded_content"] = encoded
            
            elif encoding == ContentEncoding.UTF8:
                # UTF-8 encoding
                if isinstance(content, str):
                    encoded = content.encode('utf-8')
                else:
                    # Already bytes, assume UTF-8 compatible
                    encoded = content
                
                result["success"] = True
                result["encoded_content"] = encoded
            
            elif encoding == ContentEncoding.ASCII:
                # ASCII encoding
                if isinstance(content, str):
                    encoded = content.encode('ascii', errors=options.get('errors', 'strict'))
                else:
                    # Convert bytes to ASCII
                    encoded = content.decode('utf-8', errors='ignore').encode('ascii', errors=options.get('errors', 'strict'))
                
                result["success"] = True
                result["encoded_content"] = encoded
            
            elif encoding == ContentEncoding.ISO_8859_1:
                # ISO-8859-1 encoding
                if isinstance(content, str):
                    encoded = content.encode('iso-8859-1', errors=options.get('errors', 'strict'))
                else:
                    # Convert bytes to ISO-8859-1
                    encoded = content.decode('utf-8', errors='ignore').encode('iso-8859-1', errors=options.get('errors', 'strict'))
                
                result["success"] = True
                result["encoded_content"] = encoded
            
            else:
                result["errors"].append(f"Unsupported encoding: {encoding.value}")
                
        except Exception as e:
            result["errors"].append(f"Encoding error: {str(e)}")
        
        return result
    
    def decode_content(self,
                      content: Union[str, bytes],
                      encoding: Union[str, ContentEncoding],
                      options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Decode content from the specified encoding.
        
        Args:
            content: The content to decode
            encoding: The encoding to decode from
            options: Optional decoding options
            
        Returns:
            Dictionary with decoded content and metadata
        """
        options = options or {}
        
        # Convert string encoding name to enum if needed
        if isinstance(encoding, str):
            if encoding in self.supported_encodings:
                encoding = self.supported_encodings[encoding]
            else:
                try:
                    encoding = ContentEncoding(encoding)
                except:
                    encoding = ContentEncoding.UNKNOWN
        
        result = {
            "success": False,
            "decoded_content": None,
            "encoding": encoding,
            "errors": []
        }
        
        try:
            if encoding == ContentEncoding.NONE:
                # No encoding, return as is
                result["success"] = True
                result["decoded_content"] = content
            
            elif encoding == ContentEncoding.BASE64:
                # Base64 decoding
                if isinstance(content, str):
                    decoded = base64.b64decode(content)
                else:
                    # Already bytes, assume base64 encoded
                    decoded = base64.b64decode(content)
                
                result["success"] = True
                result["decoded_content"] = decoded
                
                # Try to convert to string if it looks like text
                try:
                    text_content = decoded.decode('utf-8')
                    if all(32 <= ord(c) <= 126 or ord(c) in (9, 10, 13) for c in text_content):
                        # Looks like text
                        result["decoded_content"] = text_content
                except:
                    # Keep as bytes
                    pass
            
            elif encoding == ContentEncoding.UTF8:
                # UTF-8 decoding
                if isinstance(content, bytes):
                    decoded = content.decode('utf-8', errors=options.get('errors', 'strict'))
                else:
                    # Already string, nothing to do
                    decoded = content
                
                result["success"] = True
                result["decoded_content"] = decoded
            
            elif encoding == ContentEncoding.ASCII:
                # ASCII decoding
                if isinstance(content, bytes):
                    decoded = content.decode('ascii', errors=options.get('errors', 'strict'))
                else:
                    # Already string, nothing to do
                    decoded = content
                
                result["success"] = True
                result["decoded_content"] = decoded
            
            elif encoding == ContentEncoding.ISO_8859_1:
                # ISO-8859-1 decoding
                if isinstance(content, bytes):
                    decoded = content.decode('iso-8859-1', errors=options.get('errors', 'strict'))
                else:
                    # Already string, nothing to do
                    decoded = content
                
                result["success"] = True
                result["decoded_content"] = decoded
            
            else:
                result["errors"].append(f"Unsupported encoding: {encoding.value}")
                
        except Exception as e:
            result["errors"].append(f"Decoding error: {str(e)}")
        
        return result
    
    def read_file(self, 
                 file_path: str, 
                 encoding: Optional[Union[str, ContentEncoding]] = None,
                 binary: bool = False) -> Dict[str, Any]:
        """
        Read content from a file.
        
        Args:
            file_path: Path to the file
            encoding: Optional encoding for text files
            binary: Whether to read as binary
            
        Returns:
            Dictionary with file content and metadata
        """
        result = {
            "success": False,
            "content": None,
            "file_path": file_path,
            "format": ContentFormat.UNKNOWN,
            "type": ContentType.UNKNOWN,
            "encoding": ContentEncoding.NONE,
            "size": 0,
            "errors": []
        }
        
        if not os.path.exists(file_path):
            result["errors"].append(f"File not found: {file_path}")
            return result
        
        try:
            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type:
                result["format"] = self.supported_formats.get(mime_type, ContentFormat.UNKNOWN)
                
                # Determine content type from MIME type
                if mime_type.startswith('text/') or mime_type in ['application/json', 'application/xml', 'application/yaml']:
                    result["type"] = ContentType.TEXT
                    binary = False
                elif mime_type.startswith('image/'):
                    result["type"] = ContentType.IMAGE
                    binary = True
                elif mime_type.startswith('audio/'):
                    result["type"] = ContentType.AUDIO
                    binary = True
                elif mime_type.startswith('video/'):
                    result["type"] = ContentType.VIDEO
                    binary = True
                else:
                    # Default to binary for unknown MIME types
                    result["type"] = ContentType.BINARY
                    binary = True
            
            # Handle encoding for text files
            if not binary and encoding:
                if isinstance(encoding, str):
                    if encoding in self.supported_encodings:
                        result["encoding"] = self.supported_encodings[encoding]
                    else:
                        try:
                            result["encoding"] = ContentEncoding(encoding)
                        except:
                            result["encoding"] = ContentEncoding.UTF8
                else:
                    result["encoding"] = encoding
            
            # Read the file
            if binary:
                with open(file_path, 'rb') as f:
                    result["content"] = f.read()
            else:
                # Default to UTF-8 for text files if no encoding specified
                if not encoding:
                    result["encoding"] = ContentEncoding.UTF8
                
                with open(file_path, 'r', encoding=result["encoding"].value) as f:
                    result["content"] = f.read()
            
            result["size"] = len(result["content"])
            result["success"] = True
            
        except Exception as e:
            result["errors"].append(f"Error reading file: {str(e)}")
        
        return result
    
    def write_file(self,
                  file_path: str,
                  content: Union[str, bytes],
                  encoding: Optional[Union[str, ContentEncoding]] = None,
                  binary: bool = False,
                  overwrite: bool = True) -> Dict[str, Any]:
        """
        Write content to a file.
        
        Args:
            file_path: Path to the file
            content: Content to write
            encoding: Optional encoding for text files
            binary: Whether to write as binary
            overwrite: Whether to overwrite existing file
            
        Returns:
            Dictionary with write result and metadata
        """
        result = {
            "success": False,
            "file_path": file_path,
            "size": len(content),
            "errors": []
        }
        
        if os.path.exists(file_path) and not overwrite:
            result["errors"].append(f"File already exists: {file_path}")
            return result
        
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                result["errors"].append(f"Failed to create directory: {str(e)}")
                return result
        
        try:
            # Handle encoding for text files
            if not binary and encoding:
                if isinstance(encoding, str):
                    if encoding in self.supported_encodings:
                        encoding = self.supported_encodings[encoding]
                    else:
                        try:
                            encoding = ContentEncoding(encoding)
                        except:
                            encoding = ContentEncoding.UTF8
            
            # Write the file
            if binary or isinstance(content, bytes):
                with open(file_path, 'wb') as f:
                    if isinstance(content, str):
                        # Convert string to bytes if binary write requested
                        if encoding and encoding != ContentEncoding.NONE:
                            content = content.encode(encoding.value)
                        else:
                            content = content.encode('utf-8')
                    f.write(content)
            else:
                # Default to UTF-8 for text files if no encoding specified
                if not encoding:
                    encoding = ContentEncoding.UTF8
                
                with open(file_path, 'w', encoding=encoding.value) as f:
                    f.write(content)
            
            result["success"] = True
            
        except Exception as e:
            result["errors"].append(f"Error writing file: {str(e)}")
        
        return result
    
    # Format validators
    def _validate_json(self, content: Union[str, Dict, List], strict: bool = False) -> Dict[str, Any]:
        """Validate JSON content."""
        result = {"valid": False, "errors": []}
        
        try:
            if isinstance(content, (dict, list)):
                # Already parsed
                result["valid"] = True
            else:
                # Parse JSON
                json.loads(content)
                result["valid"] = True
        except Exception as e:
            result["errors"].append(f"Invalid JSON: {str(e)}")
        
        return result
    
    def _validate_xml(self, content: str, strict: bool = False) -> Dict[str, Any]:
        """Validate XML content."""
        result = {"valid": False, "errors": []}
        
        try:
            # Use a simple regex for basic validation
            if re.match(r'^\s*<\?xml|^\s*<[a-zA-Z0-9_:]+(\s+[^>]+)?>.*</[a-zA-Z0-9_:]+>\s*$', content, re.DOTALL):
                result["valid"] = True
            else:
                result["errors"].append("Invalid XML structure")
        except Exception as e:
            result["errors"].append(f"XML validation error: {str(e)}")
        
        return result
    
    def _validate_yaml(self, content: str, strict: bool = False) -> Dict[str, Any]:
        """Validate YAML content."""
        result = {"valid": False, "errors": []}
        
        try:
            # Simple validation for YAML-like structure
            if any(content.startswith(prefix) for prefix in ['---', 'apiVersion:', 'name:', '#']):
                result["valid"] = True
            else:
                # Check for key-value pattern
                if re.search(r'^[a-zA-Z0-9_-]+:\s*\S+', content, re.MULTILINE):
                    result["valid"] = True
                else:
                    result["errors"].append("Content does not appear to be valid YAML")
        except Exception as e:
            result["errors"].append(f"YAML validation error: {str(e)}")
        
        return result
    
    def _validate_csv(self, content: str, strict: bool = False) -> Dict[str, Any]:
        """Validate CSV content."""
        result = {"valid": False, "errors": []}
        
        try:
            # Check for line breaks and commas
            if '\n' in content and ',' in content:
                # Check for consistent number of commas per line
                lines = content.strip().split('\n')
                comma_counts = [line.count(',') for line in lines]
                
                if len(set(comma_counts)) <= 1:
                    # All lines have the same number of commas
                    result["valid"] = True
                else:
                    result["errors"].append("Inconsistent number of columns in CSV")
            else:
                result["errors"].append("Content does not appear to be valid CSV")
        except Exception as e:
            result["errors"].append(f"CSV validation error: {str(e)}")
        
        return result
    
    def _validate_html(self, content: str, strict: bool = False) -> Dict[str, Any]:
        """Validate HTML content."""
        result = {"valid": False, "errors": []}
        
        try:
            # Simple validation for HTML structure
            if re.search(r'<html[\s>]', content, re.IGNORECASE) and re.search(r'</html>', content, re.IGNORECASE):
                result["valid"] = True
            elif re.search(r'<!DOCTYPE\s+html>', content, re.IGNORECASE):
                result["valid"] = True
            elif re.search(r'<body[\s>]', content, re.IGNORECASE) and re.search(r'</body>', content, re.IGNORECASE):
                result["valid"] = True
            elif re.search(r'<div[\s>]', content, re.IGNORECASE) or re.search(r'<p[\s>]', content, re.IGNORECASE):
                result["valid"] = True
            else:
                result["errors"].append("Content does not appear to be valid HTML")
        except Exception as e:
            result["errors"].append(f"HTML validation error: {str(e)}")
        
        return result
    
    def _validate_markdown(self, content: str, strict: bool = False) -> Dict[str, Any]:
        """Validate Markdown content."""
        result = {"valid": True, "errors": []}  # Markdown is very permissive
        
        # Check for some common Markdown elements
        has_headers = bool(re.search(r'^#{1,6}\s+\S+', content, re.MULTILINE))
        has_lists = bool(re.search(r'^\s*[-*+]\s+\S+', content, re.MULTILINE))
        
        # Only flag as invalid if strict validation is requested and no Markdown
