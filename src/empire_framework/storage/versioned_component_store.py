"""
Versioned Component Store for Empire Framework

This module provides a secure storage system for Empire Framework components with versioning,
integrity verification, and archiving capabilities.
"""

import os
import json
import hashlib
import shutil
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from pathlib import Path
from datetime import datetime, timezone
import tempfile
import threading
import zipfile
import glob
import time

# Set up logging
logger = logging.getLogger(__name__)


# Custom Exception Classes
class StorageError(Exception):
    """Base class for all storage-related errors."""
    pass


class ComponentIOError(StorageError):
    """Raised when there's an issue with reading/writing component data."""
    pass


class ComponentNotFoundInStoreError(StorageError):
    """Raised when a requested component is not found in the store."""
    def __init__(self, component_id: str):
        self.component_id = component_id
        message = f"Component with ID '{component_id}' not found in store"
        super().__init__(message)


class ComponentVersionNotFoundError(StorageError):
    """Raised when a specific version of a component is not found."""
    def __init__(self, component_id: str, version: str):
        self.component_id = component_id
        self.version = version
        message = f"Version '{version}' of component '{component_id}' not found"
        super().__init__(message)


class ComponentIntegrityError(StorageError):
    """Raised when a component's integrity check fails."""
    def __init__(self, component_id: str, version: str, expected_hash: str, actual_hash: str):
        self.component_id = component_id
        self.version = version
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        message = (f"Integrity check failed for component '{component_id}' version '{version}'. "
                  f"Expected hash: {expected_hash}, Actual hash: {actual_hash}")
        super().__init__(message)


class StorageSecurityError(StorageError):
    """Raised when there's a security concern with storage operations."""
    pass


class VersionedComponentStore:
    """
    Secure storage system for versioned Empire Framework components.
    
    This class provides a file-based storage system for components with versioning,
    data integrity verification, and archiving capabilities.
    """

    # Storage constants
    COMPONENT_FILENAME = "component.json"
    METADATA_FILENAME = "metadata.json"
    CHECKSUM_FILENAME = "checksum.sha256"
    ARCHIVE_DIRNAME = "_archived"
    ACTIVE_DIRNAME = "active"
    DEFAULT_STORAGE_ROOT = "data/components"
    
    # File lock to prevent concurrent writes to the same file
    _file_locks = {}
    _lock_manager = threading.Lock()

    def __init__(self, storage_root: str = None):
        """
        Initialize a new VersionedComponentStore.
        
        Args:
            storage_root: Path to the root directory for component storage.
                          If None, uses the default path (data/components).
        """
        self.storage_root = Path(storage_root or self.DEFAULT_STORAGE_ROOT)
        
        # Create directory structure if it doesn't exist
        self._ensure_store_structure()
        
        logger.info(f"VersionedComponentStore initialized with root: {self.storage_root}")

    def _ensure_store_structure(self) -> None:
        """Create the necessary directory structure for the component store."""
        # Main storage directory
        os.makedirs(self.storage_root, exist_ok=True)
        
        # Archive directory
        archive_path = self.storage_root / self.ARCHIVE_DIRNAME
        os.makedirs(archive_path, exist_ok=True)

    def _get_component_path(self, component_id: str, version: str = None, 
                           archived: bool = False) -> Path:
        """
        Get the path for a component.
        
        Args:
            component_id: The ID of the component
            version: Optional specific version
            archived: Whether to look in the archive
            
        Returns:
            Path to the component directory
        """
        base_path = self.storage_root
        if archived:
            base_path = base_path / self.ARCHIVE_DIRNAME
        
        if version:
            return base_path / component_id / version
        else:
            return base_path / component_id

    def _get_file_lock(self, file_path: Path) -> threading.Lock:
        """
        Get a lock for a specific file path to prevent concurrent writes.
        
        Args:
            file_path: The path to get a lock for
            
        Returns:
            Threading lock for the file
        """
        str_path = str(file_path)
        with self._lock_manager:
            if str_path not in self._file_locks:
                self._file_locks = {**self._file_locks, str_path: threading.Lock()}
            return self._file_locks[str_path]

    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """
        Calculate a SHA-256 checksum for component data.
        
        Args:
            data: The component data
            
        Returns:
            Hex digest of the SHA-256 hash
        """
        # Convert the data to a stable string representation for hashing
        # Sort keys to ensure consistent ordering regardless of dict insertion order
        serialized = json.dumps(data, sort_keys=True).encode('utf-8')
        return hashlib.sha256(serialized).hexdigest()

    def _verify_component_integrity(self, component_id: str, version: str, 
                                    data: Dict[str, Any]) -> bool:
        """
        Verify the integrity of component data using stored checksum.
        
        Args:
            component_id: The ID of the component
            version: The version to verify
            data: The component data to verify
            
        Returns:
            True if integrity is verified, raises ComponentIntegrityError otherwise
        """
        component_dir = self._get_component_path(component_id, version)
        checksum_path = component_dir / self.CHECKSUM_FILENAME
        
        try:
            with open(checksum_path, 'r') as f:
                stored_checksum = f.read().strip()
        except FileNotFoundError:
            logger.warning(f"Checksum file not found for {component_id} version {version}")
            return True  # No checksum to verify against (might be a legacy component)
        except Exception as e:
            raise ComponentIOError(f"Error reading checksum file: {str(e)}")
        
        calculated_checksum = self._calculate_checksum(data)
        
        if calculated_checksum != stored_checksum:
            raise ComponentIntegrityError(
                component_id, version, stored_checksum, calculated_checksum)
        
        return True

    def _write_component_safely(self, file_path: Path, data: Dict[str, Any]) -> None:
        """
        Write component data to disk with atomic safety.
        
        Uses a temporary file and atomic rename to ensure data integrity
        even in case of system crashes or power failures.
        
        Args:
            file_path: Path to write the component file
            data: Component data to write
        """
        # Create parent directory if it doesn't exist
        os.makedirs(file_path.parent, exist_ok=True)
        
        # Get lock for this file
        lock = self._get_file_lock(file_path)
        
        with lock:
            # Write to a temporary file first
            temp_fd, temp_path = tempfile.mkstemp(
                prefix=f"{file_path.stem}_", suffix=".tmp", dir=file_path.parent)
            try:
                with os.fdopen(temp_fd, 'w') as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                
                # Ensure file is fully written to disk
                os.fsync(temp_fd)
                
                # Rename temp file to target (atomic operation on most filesystems)
                os.replace(temp_path, file_path)
            except Exception as e:
                # Clean up the temporary file if anything goes wrong
                try:
                    os.unlink(temp_path)
                except:
                    pass
                raise ComponentIOError(f"Error writing component data: {str(e)}")

    def _is_valid_version_format(self, version: str) -> bool:
        """
        Check if a version string is in valid semantic versioning format.
        
        Args:
            version: The version string to check
            
        Returns:
            True if the version is in valid format
        """
        pattern = r'^\d+\.\d+\.\d+$'
        return bool(re.match(pattern, version))

    def _version_to_tuple(self, version: str) -> Tuple[int, int, int]:
        """
        Convert a version string to a tuple for comparison.
        
        Args:
            version: The version string
            
        Returns:
            Tuple of (major, minor, patch) as integers
        """
        try:
            return tuple(map(int, version.split('.')))
        except ValueError:
            # If the version can't be parsed, return a sentinel value
            return (-1, -1, -1)

    def _find_latest_version(self, component_id: str) -> Optional[str]:
        """
        Find the latest version of a component.
        
        Args:
            component_id: The ID of the component
            
        Returns:
            Latest version string or None if no versions found
        """
        versions = self.list_component_versions(component_id, include_archived=False)
        if not versions:
            return None
        
        # Sort versions using semantic versioning comparison
        return sorted(versions, key=self._version_to_tuple, reverse=True)[0]

    def save_component_version(self, component_data: Dict[str, Any], 
                              validate_integrity: bool = True) -> str:
        """
        Save a component version to the store.
        
        Args:
            component_data: The component data to save
            validate_integrity: Whether to validate the integrity after saving
            
        Returns:
            The saved version string
            
        Raises:
            ComponentIOError: If there's an issue writing the component
            ValueError: If the component data is missing required fields
        """
        # Check required fields
        if 'id' not in component_data or not component_data['id']:
            raise ValueError("Component must have an 'id' field")
        if 'version' not in component_data or not component_data['version']:
            raise ValueError("Component must have a 'version' field")
        
        component_id = component_data['id']
        version = component_data['version']
        
        # Validate version format
        if not self._is_valid_version_format(version):
            raise ValueError(f"Invalid version format: '{version}'. Expected format: X.Y.Z")
        
        # Get the component directory path
        component_dir = self._get_component_path(component_id, version)
        component_path = component_dir / self.COMPONENT_FILENAME
        
        # Check if this version already exists
        if component_path.exists():
            # We should never overwrite an existing version - versions are immutable
            raise ComponentIOError(
                f"Component {component_id} version {version} already exists. Versions are immutable.")
        
        # Create metadata
        metadata = {
            "created_timestamp": datetime.now().isoformat(),
            "component_id": component_id,
            "version": version,
            "component_type": component_data.get("component_type", "Unknown"),
            "is_archived": False
        }
        
        # Calculate checksum
        checksum = self._calculate_checksum(component_data)
        
        try:
            # Write component data
            self._write_component_safely(component_path, component_data)
            
            # Write metadata
            metadata_path = component_dir / self.METADATA_FILENAME
            self._write_component_safely(metadata_path, metadata)
            
            # Write checksum
            checksum_path = component_dir / self.CHECKSUM_FILENAME
            with open(checksum_path, 'w') as f:
                f.write(checksum)
            
            # Update latest version pointer
            self._update_latest_version_pointer(component_id, version)
            
            logger.info(f"Saved component {component_id} version {version}")
            
            # Validate integrity
            if validate_integrity:
                reloaded_data = self.load_component_version(component_id, version)
                self._verify_component_integrity(component_id, version, reloaded_data)
            
            return version
            
        except Exception as e:
            # Attempt to clean up if anything goes wrong
            try:
                if component_dir.exists():
                    shutil.rmtree(component_dir)
            except:
                pass
            
            if isinstance(e, StorageError):
                raise
            else:
                raise ComponentIOError(f"Failed to save component: {str(e)}")

    def _update_latest_version_pointer(self, component_id: str, version: str) -> None:
        """
        Update the pointer to the latest version of a component.
        
        Args:
            component_id: The ID of the component
            version: The version to set as latest
        """
        component_dir = self._get_component_path(component_id)
        latest_path = component_dir / "latest_version"
        
        try:
            # Write the version to the latest_version file
            with open(latest_path, 'w') as f:
                f.write(version)
        except Exception as e:
            logger.warning(f"Failed to update latest version pointer: {str(e)}")
            # Not raising an exception here since it's not critical

    def load_component_version(self, component_id: str, version: str, 
                              verify_integrity: bool = True) -> Dict[str, Any]:
        """
        Load a specific version of a component.
        
        Args:
            component_id: The ID of the component to load
            version: The version to load
            verify_integrity: Whether to verify data integrity
            
        Returns:
            The component data
            
        Raises:
            ComponentNotFoundInStoreError: If the component doesn't exist
            ComponentVersionNotFoundError: If the version doesn't exist
            ComponentIntegrityError: If integrity verification fails
            ComponentIOError: If there's an issue reading the component
        """
        # First check active components
        component_dir = self._get_component_path(component_id, version)
        component_path = component_dir / self.COMPONENT_FILENAME
        
        # If not found in active, check archived
        if not component_path.exists():
            archived_dir = self._get_component_path(component_id, version, archived=True)
            archived_path = archived_dir / self.COMPONENT_FILENAME
            
            if not archived_path.exists():
                # Check if the component exists at all
                if not self._get_component_path(component_id).exists() and \
                   not self._get_component_path(component_id, archived=True).exists():
                    raise ComponentNotFoundInStoreError(component_id)
                else:
                    raise ComponentVersionNotFoundError(component_id, version)
            
            component_path = archived_path
        
        try:
            # Read the component data
            with open(component_path, 'r') as f:
                component_data = json.load(f)
            
            # Verify integrity if requested
            if verify_integrity:
                self._verify_component_integrity(component_id, version, component_data)
            
            return component_data
            
        except json.JSONDecodeError:
            raise ComponentIOError(f"Invalid JSON in component file for {component_id} version {version}")
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            else:
                raise ComponentIOError(f"Error loading component: {str(e)}")

    def load_latest_component_version(self, component_id: str, 
                                     include_archived: bool = False,
                                     verify_integrity: bool = True) -> Dict[str, Any]:
        """
        Load the latest version of a component.
        
        Args:
            component_id: The ID of the component to load
            include_archived: Whether to include archived versions
            verify_integrity: Whether to verify data integrity
            
        Returns:
            The latest component data
            
        Raises:
            ComponentNotFoundInStoreError: If the component doesn't exist
            ComponentIOError: If there's an issue reading the component
        """
        # Try to get the latest version from the pointer file first (fast path)
        component_dir = self._get_component_path(component_id)
        latest_path = component_dir / "latest_version"
        
        if latest_path.exists() and not include_archived:
            try:
                with open(latest_path, 'r') as f:
                    latest_version = f.read().strip()
                
                # Verify the version actually exists
                version_dir = component_dir / latest_version
                if version_dir.exists():
                    return self.load_component_version(
                        component_id, latest_version, verify_integrity)
            except Exception:
                # If anything goes wrong with the fast path, fall back to scanning
                pass
        
        # Fall back to scanning all versions (slow path)
        versions = self.list_component_versions(component_id, include_archived)
        if not versions:
            raise ComponentNotFoundInStoreError(component_id)
        
        # Sort versions using semantic versioning comparison
        latest_version = sorted(versions, key=self._version_to_tuple, reverse=True)[0]
        
        return self.load_component_version(component_id, latest_version, verify_integrity)

    def list_component_versions(self, component_id: str, 
                               include_archived: bool = True) -> List[str]:
        """
        List all available versions of a component.
        
        Args:
            component_id: The ID of the component
            include_archived: Whether to include archived versions
            
        Returns:
            List of version strings sorted by semantic versioning
            
        Raises:
            ComponentNotFoundInStoreError: If the component doesn't exist
        """
        versions = set()
        
        # Check active components
        active_dir = self._get_component_path(component_id)
        
        if active_dir.exists():
            for item in active_dir.iterdir():
                if item.is_dir() and self._is_valid_version_format(item.name):
                    versions.add(item.name)
        
        # Check archived components if requested
        if include_archived:
            archived_dir = self._get_component_path(component_id, archived=True)
            
            if archived_dir.exists():
                for item in archived_dir.iterdir():
                    if item.is_dir() and self._is_valid_version_format(item.name):
                        versions.add(item.name)
        
        if not versions:
            # Check if the component exists at all
            if not active_dir.exists() and not (include_archived and archived_dir.exists()):
                raise ComponentNotFoundInStoreError(component_id)
            # Otherwise, just no versions available
        
        # Sort versions using semantic versioning comparison
        return sorted(list(versions), key=self._version_to_tuple)

    def list_components(self, component_type: str = None, 
                       include_archived: bool = False) -> List[str]:
        """
        List all component IDs in the store.
        
        Args:
            component_type: Optional type of components to list
            include_archived: Whether to include archived components
            
        Returns:
            List of component IDs
        """
        components = set()
        
        # Function to collect component IDs of a certain type
        def collect_components(base_dir: Path) -> None:
            if not base_dir.exists():
                return
                
            for item in base_dir.iterdir():
                if not item.is_dir():
                    continue
                    
                # Check if it matches the type filter
                if component_type:
                    # Need to check the metadata or component file
                    # First try metadata (faster)
                    metadata_path = item / self.METADATA_FILENAME
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            if metadata.get('component_type') == component_type:
                                components.add(item.name)
                        except:
                            pass
                    else:
                        # Try to find a version with component data
                        latest_version = self._find_latest_version(item.name)
                        if latest_version:
                            try:
                                component_data = self.load_component_version(
                                    item.name, latest_version, verify_integrity=False)
                                if component_data.get('component_type') == component_type:
                                    components.add(item.name)
                            except:
                                pass
                else:
                    # No type filter, add all directories
                    components.add(item.name)
        
        # Collect active components
        collect_components(self.storage_root)
        
        # Collect archived components if requested
        if include_archived:
            collect_components(self.storage_root / self.ARCHIVE_DIRNAME)
        
        return sorted(list(components))

    def delete_component_version(self, component_id: str, version: str, 
                                archive: bool = True) -> None:
        """
        Delete a specific version of a component.
        
        Args:
            component_id: The ID of the component
            version: The version to delete
            archive: If True, move to archive instead of deleting
            
        Raises:
            ComponentNotFoundInStoreError: If the component doesn't exist
            ComponentVersionNotFoundError: If the version doesn't exist
            ComponentIOError: If there's an issue with the deletion
        """
        # Check if the component and version exist
        component_dir = self._get_component_path(component_id, version)
        
        if not component_dir.exists():
            # Check if the component exists at all
            if not self._get_component_path(component_id).exists():
                raise ComponentNotFoundInStoreError(component_id)
            else:
                raise ComponentVersionNotFoundError(component_id, version)
        
        try:
            if archive:
                # Move to archive
                archive_dir = self._get_component_path(component_id, version, archived=True)
                
                # Create parent directories if they don't exist
                os.makedirs(archive_dir.parent, exist_ok=True)
                
                # If there's already a version in the archive, remove it first
                if archive_dir.exists():
                    shutil.rmtree(archive_dir)
                
                # Move the directory
                shutil.move(str(component_dir), str(archive_dir))
                
                # Update metadata to mark as archived
                metadata_path = archive_dir / self.METADATA_FILENAME
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        metadata['is_archived'] = True
                        metadata['archived_timestamp'] = datetime.now().isoformat()
                        
                        self._write_component_safely(metadata_path, metadata)
                    except:
                        # Not critical if this fails
                        logger.warning(f"Failed to update archived metadata for {component_id} version {version}")
                
                logger.info(f"Archived component {component_id} version {version}")
            else:
                # Actually delete
                shutil.rmtree(component_dir)
                logger.info(f"Deleted component {component_id} version {version}")
            
            # Update latest version pointer if needed
            latest_path = self._get_component_path(component_id) / "latest_version"
            if latest_path.exists():
                try:
                    with open(latest_path, 'r') as f:
                        current_latest = f.read().strip()
                    
                    if current_latest == version:
                        # Find the new latest version
                        versions = self.list_component_versions(component_id, include_archived=False)
                        if versions:
                            new_latest = sorted(versions, key=self._version_to_tuple, reverse=True)[0]
                            self._update_latest_version_pointer(component_id, new_latest)
                        else:
                            # No more versions left, remove the pointer
                            os.remove(latest_path)
                except:
                    # Not critical if this fails
                    logger.warning(f"Failed to update latest version pointer after deleting {component_id} version {version}")
            
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            else:
                raise ComponentIOError(f"Error deleting component version: {str(e)}")

    def delete_component(self, component_id: str, archive: bool = True) -> None:
        """
        Delete all versions of a component.
        
        Args:
            component_id: The ID of the component to delete
            archive: If True, move to archive instead of deleting
            
        Raises:
            ComponentNotFoundInStoreError: If the component doesn't exist
            ComponentIOError: If there's an issue with the deletion
        """
        # Check if the component exists
        component_dir = self._get_component_path(component_id)
        
        if not component_dir.exists():
            raise ComponentNotFoundInStoreError(component_id)
        
        try:
            # Get all versions
            versions = []
            for item in component_dir.iterdir():
                if item.is_dir() and self._is_valid_version_format(item.name):
                    versions.append(item.name)
            
            if archive:
                # Move each version to archive
                for version in versions:
                    self.delete_component_version(component_id, version, archive=True)
                
                # Remove the main component directory if empty
                remaining_items = list(component_dir.iterdir())
                if len(remaining_items) <= 1:  # Only latest_version file might remain
                    shutil.rmtree(component_dir)
                
                logger.info(f"Archived all versions of component {component_id}")
            else:
                # Just delete the entire directory
                shutil.rmtree(component_dir)
                logger.info(f"Deleted all versions of component {component_id}")
            
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            else:
                raise ComponentIOError(f"Error deleting component: {str(e)}")

    def restore_component_version(self, component_id: str, version: str) -> None:
        """
        Restore an archived component version.
        
        Args:
            component_id: The ID of the component
            version: The version to restore
            
        Raises:
            ComponentNotFoundInStoreError: If the component doesn't exist
            ComponentVersionNotFoundError: If the version doesn't exist in archive
            ComponentIOError: If there's an issue with the restoration
        """
        # Check if the archived version exists
        archived_dir = self._get_component_path(component_id, version, archived=True)
        
        if not archived_dir.exists():
            # Check if the component exists in the archive at all
            if not self._get_component_path(component_id, archived=True).exists():
                raise ComponentNotFoundInStoreError(component_id)
            else:
                raise ComponentVersionNotFoundError(component_id, version)
        
        # Check if the version already exists in active components
        active_dir = self._get_component_path(component_id, version)
        if active_dir.exists():
            raise ComponentIOError(
                f"Cannot restore component {component_id} version {version}: "
                f"already exists in active components")
        
        try:
            # Create parent directories if they don't exist
            os.makedirs(active_dir.parent, exist_ok=True)
            
            # Move the directory
            shutil.move(str(archived_dir), str(active_dir))
            
            # Update metadata to mark as active
            metadata_path = active_dir / self.METADATA_FILENAME
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    metadata['is_archived'] = False
                    metadata['restored_timestamp'] = datetime.now().isoformat()
                    
                    self._write_component_safely(metadata_path, metadata)
                except:
                    # Not critical if this fails
                    logger.warning(f"Failed to update restored metadata for {component_id} version {version}")
            
            # Update latest version pointer if needed
            latest_path = active_dir.parent / "latest_version"
            if not latest_path.exists() or self._version_to_tuple(version) > self._version_to_tuple(self._find_latest_version(component_id) or "0.0.0"):
                self._update_latest_version_pointer(component_id, version)
            
            logger.info(f"Restored component {component_id} version {version}")
            
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            else:
                raise ComponentIOError(f"Error restoring component version: {str(e)}")

    def verify_store_integrity(self, repair: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        Verify the integrity of all components in the store.
        
        Args:
            repair: Whether to attempt repairs of integrity issues
            
        Returns:
            Dictionary of integrity issues: {component_id: [list of issue details]}
        """
        integrity_issues = {}
        
        # Get all components
        components = self.list_components(include_archived=True)
        
        for component_id in components:
            component_issues = []
            
            # Check all versions
            try:
                versions = self.list_component_versions(component_id, include_archived=True)
                
                for version in versions:
                    try:
                        # Try to load and verify
                        self.load_component_version(component_id, version, verify_integrity=True)
                    except ComponentIntegrityError as e:
                        issue = {
                            "issue_type": "integrity_error",
                            "version": version,
                            "expected_hash": e.expected_hash,
                            "actual_hash": e.actual_hash
                        }
                        component_issues.append(issue)
                        
                        if repair:
                            # Attempt to repair by archiving corrupt version
                            try:
                                logger.warning(f"Archiving corrupt component {component_id} version {version}")
                                self.delete_component_version(component_id, version, archive=True)
                                issue["repair_action"] = "archived_corrupt_version"
                            except Exception as repair_e:
                                issue["repair_error"] = str(repair_e)
                    except Exception as e:
                        issue = {
                            "issue_type": "access_error",
                            "version": version,
                            "error": str(e)
                        }
                        component_issues.append(issue)
            except Exception as e:
                # Issue with the component itself
                issue = {
                    "issue_type": "component_error",
                    "error": str(e)
                }
                component_issues.append(issue)
            
            if component_issues:
                integrity_issues[component_id] = component_issues
        
        return integrity_issues

    def create_backup(self, backup_path: str, include_archived: bool = True) -> str:
        """
        Create a backup of the entire component store.
        
        Args:
            backup_path: Path where the backup should be created
            include_archived: Whether to include archived components
            
        Returns:
            Path to the created backup file
            
        Raises:
            ComponentIOError: If there's an issue creating the backup
        """
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"component_store_backup_{timestamp}.zip"
        full_backup_path = os.path.join(backup_path, backup_name)
        
        try:
            # Create backup directory if it doesn't exist
            os.makedirs(backup_path, exist_ok=True)
            
            # Create a zip file
            with zipfile.ZipFile(full_backup_path, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
                # Add active components
                for root, dirs, files in os.walk(self.storage_root):
                    # Skip the archive directory if not including archived
                    if not include_archived and self.ARCHIVE_DIRNAME in root:
                        continue
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Create relative path for the zip
                        arc_name = os.path.relpath(file_path, self.storage_root.parent)
                        backup_zip.write(file_path, arc_name)
            
            logger.info(f"Created backup at {full_backup_path}")
            return full_backup_path
            
        except Exception as e:
            raise ComponentIOError(f"Error creating backup: {str(e)}")
    
    def restore_from_backup(self, backup_path: str, restore_path: str = None) -> None:
        """
        Restore components from a backup.
        
        Args:
            backup_path: Path to the backup file
            restore_path: Optional path to restore to (defaults to current storage root)
            
        Raises:
            ComponentIOError: If there's an issue with the restoration
        """
        if not os.path.exists(backup_path):
            raise ComponentIOError(f"Backup file not found: {backup_path}")
        
        if restore_path is None:
            restore_path = str(self.storage_root.parent)
        
        try:
            # Extract the backup
            with zipfile.ZipFile(backup_path, 'r') as backup_zip:
                backup_zip.extractall(restore_path)
            
            logger.info(f"Restored backup from {backup_path} to {restore_path}")
            
        except Exception as e:
            raise ComponentIOError(f"Error restoring backup: {str(e)}")
    
    def get_component_metadata(self, component_id: str, version: str = None) -> Dict[str, Any]:
        """
        Get metadata for a component.
        
        Args:
            component_id: The ID of the component
            version: Optional specific version (uses latest if not specified)
            
        Returns:
            Component metadata
            
        Raises:
            ComponentNotFoundInStoreError: If the component doesn't exist
            ComponentVersionNotFoundError: If the version doesn't exist
            ComponentIOError: If there's an issue reading metadata
        """
        if version is None:
            version = self._find_latest_version(component_id)
            if version is None:
                raise ComponentNotFoundInStoreError(component_id)
        
        # Try active components first
        metadata_path = self._get_component_path(component_id, version) / self.METADATA_FILENAME
        
        # If not found, try archived
        if not metadata_path.exists():
            metadata_path = self._get_component_path(component_id, version, archived=True) / self.METADATA_FILENAME
            
            if not metadata_path.exists():
                raise ComponentVersionNotFoundError(component_id, version)
        
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ComponentIOError(f"Error reading metadata: {str(e)}")
    
    def cleanup_old_versions(self, component_id: str, keep_versions: int = 5) -> List[str]:
        """
        Clean up old versions of a component, keeping only the most recent ones.
        
        Args:
            component_id: The ID of the component
            keep_versions: Number of recent versions to keep
            
        Returns:
            List of deleted version strings
            
        Raises:
            ComponentNotFoundInStoreError: If the component doesn't exist
        """
        # Get all active versions
        versions = self.list_component_versions(component_id, include_archived=False)
        
        if not versions:
            raise ComponentNotFoundInStoreError(component_id)
        
        # Sort by version (newest first)
        sorted_versions = sorted(versions, key=self._version_to_tuple, reverse=True)
        
        # Determine which versions to delete
        versions_to_delete = sorted_versions[keep_versions:]
        deleted_versions = []
        
        for version in versions_to_delete:
            try:
                self.delete_component_version(component_id, version, archive=True)
                deleted_versions.append(version)
            except Exception as e:
                logger.warning(f"Failed to delete version {version}: {str(e)}")
        
        if deleted_versions:
            logger.info(f"Cleaned up {len(deleted_versions)} old versions of {component_id}")
        
        return deleted_versions
    
    def export_component(self, component_id: str, version: str, export_path: str) -> str:
        """
        Export a component version to a standalone file.
        
        Args:
            component_id: The ID of the component
            version: The version to export
            export_path: Directory to export to
            
        Returns:
            Path to the exported file
            
        Raises:
            ComponentNotFoundInStoreError: If the component doesn't exist
            ComponentVersionNotFoundError: If the version doesn't exist
            ComponentIOError: If there's an issue with the export
        """
        # Load the component
        component_data = self.load_component_version(component_id, version)
        metadata = self.get_component_metadata(component_id, version)
        
        # Create export package
        export_package = {
            "component": component_data,
            "metadata": metadata,
            "export_info": {
                "exported_at": datetime.now().isoformat(),
                "store_version": "1.0.0"
            }
        }
        
        # Create export filename
        safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', component_id)
        safe_version = version.replace('.', '_')
        export_filename = f"{safe_id}_v{safe_version}_export.json"
        export_file_path = os.path.join(export_path, export_filename)
        
        try:
            # Create export directory if needed
            os.makedirs(export_path, exist_ok=True)
            
            # Write export file
            with open(export_file_path, 'w') as f:
                json.dump(export_package, f, indent=2, sort_keys=True)
            
            logger.info(f"Exported {component_id} version {version} to {export_file_path}")
            return export_file_path
            
        except Exception as e:
            raise ComponentIOError(f"Error exporting component: {str(e)}")
    
    def import_component(self, import_path: str, overwrite: bool = False) -> Tuple[str, str]:
        """
        Import a component from an exported file.
        
        Args:
            import_path: Path to the export file
            overwrite: Whether to overwrite if the version already exists
            
        Returns:
            Tuple of (component_id, version)
            
        Raises:
            ComponentIOError: If there's an issue with the import
        """
        if not os.path.exists(import_path):
            raise ComponentIOError(f"Import file not found: {import_path}")
        
        try:
            # Read the export file
            with open(import_path, 'r') as f:
                export_package = json.load(f)
            
            # Validate export package
            if 'component' not in export_package or 'metadata' not in export_package:
                raise ComponentIOError("Invalid export file format")
            
            component_data = export_package['component']
            component_id = component_data.get('id')
            version = component_data.get('version')
            
            if not component_id or not version:
                raise ComponentIOError("Export file missing component ID or version")
            
            # Check if it already exists
            if not overwrite:
                try:
                    existing = self.load_component_version(component_id, version, verify_integrity=False)
                    raise ComponentIOError(
                        f"Component {component_id} version {version} already exists. "
                        f"Use overwrite=True to replace.")
                except ComponentNotFoundInStoreError:
                    pass
                except ComponentVersionNotFoundError:
                    pass
            
            # Save the component
            saved_version = self.save_component_version(component_data)
            
            logger.info(f"Imported {component_id} version {saved_version} from {import_path}")
            return component_id, saved_version
            
        except json.JSONDecodeError:
            raise ComponentIOError("Invalid JSON in import file")
        except Exception as e:
            if isinstance(e, ComponentIOError):
                raise
            else:
                raise ComponentIOError(f"Error importing component: {str(e)}")



    async def __aenter__(self):
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context and cleanup."""
        if hasattr(self, 'cleanup'):
            await self.cleanup()
        elif hasattr(self, 'close'):
            await self.close()
        return False
# Example usage
if __name__ == "__main__":
    # Example of using the VersionedComponentStore
    store = VersionedComponentStore()
    
    # Example component
    example_component = {
        "id": "test_component",
        "version": "1.0.0",
        "component_type": "Example",
        "data": {
            "description": "This is a test component",
            "value": 42
        }
    }
    
    # Save component
    try:
        version = store.save_component_version(example_component)
        print(f"Saved component version: {version}")
        
        # Load component
        loaded = store.load_latest_component_version("test_component")
        print(f"Loaded component: {loaded}")
        
        # List versions
        versions = store.list_component_versions("test_component")
        print(f"Available versions: {versions}")
        
    except StorageError as e:
        print(f"Storage error: {e}")
