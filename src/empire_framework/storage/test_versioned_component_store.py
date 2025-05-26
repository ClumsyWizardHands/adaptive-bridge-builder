"""
Unit tests for the VersionedComponentStore class.
"""

import os
import json
import tempfile
import shutil
import unittest
from pathlib import Path
from datetime import datetime, timedelta, timezone

from .versioned_component_store import (
    VersionedComponentStore,
    ComponentNotFoundInStoreError,
    ComponentVersionNotFoundError,
    ComponentIntegrityError,
    ComponentIOError
)

class TestVersionedComponentStore(unittest.TestCase):
    """Test case for the VersionedComponentStore class."""

    def setUp(self) -> None:
        """Set up a temporary directory for storage tests."""
        self.test_dir = tempfile.mkdtemp()
        self.store = VersionedComponentStore(storage_root=self.test_dir)
        
        # Create test components
        self.test_components = {
            "principle1": {
                "1.0.0": {
                    "id": "principle1",
                    "version": "1.0.0",
                    "component_type": "principle",
                    "name": "Test Principle 1",
                    "description": "A test principle",
                    "criteria": ["Be consistent", "Be clear"]
                },
                "1.1.0": {
                    "id": "principle1",
                    "version": "1.1.0",
                    "component_type": "principle",
                    "name": "Test Principle 1 - Revised",
                    "description": "A revised test principle",
                    "criteria": ["Be consistent", "Be clear", "Be concise"]
                }
            },
            "identity1": {
                "1.0.0": {
                    "id": "identity1",
                    "version": "1.0.0",
                    "component_type": "identity",
                    "name": "Test Identity 1",
                    "beliefs": ["I am helpful", "I am knowledgeable"]
                }
            }
        }
        
        # Add components to the store
        for component_id, versions in self.test_components.items():
            for version, component_data in versions.items():
                self.store.save_component_version(component_data)

    def tearDown(self) -> None:
        """Clean up the temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_save_and_load_component(self) -> None:
        """Test saving and loading a component."""
        component_data = {
            "id": "test_component",
            "version": "1.0.0",
            "component_type": "test",
            "name": "Test Component",
            "description": "A test component"
        }
        
        # Save the component
        version = self.store.save_component_version(component_data)
        self.assertEqual(version, "1.0.0")
        
        # Load the component
        loaded_data = self.store.load_component_version("test_component", "1.0.0")
        self.assertEqual(loaded_data, component_data)
        
        # Try to save the same version again (should fail)
        with self.assertRaises(ComponentIOError):
            self.store.save_component_version(component_data)
        
        # Save a new version
        component_data["version"] = "1.1.0"
        component_data["description"] = "An updated test component"
        version = self.store.save_component_version(component_data)
        self.assertEqual(version, "1.1.0")
        
        # Load the new version
        loaded_data = self.store.load_component_version("test_component", "1.1.0")
        self.assertEqual(loaded_data, component_data)
        
        # Load the latest version (should be 1.1.0)
        latest_data = self.store.load_latest_component_version("test_component")
        self.assertEqual(latest_data["version"], "1.1.0")

    def test_invalid_version_format(self) -> None:
        """Test handling of invalid version formats."""
        component_data = {
            "id": "invalid_version",
            "version": "1.0",  # Invalid format (missing patch)
            "component_type": "test"
        }
        
        with self.assertRaises(ValueError):
            self.store.save_component_version(component_data)
        
        component_data["version"] = "1.0.0-beta"  # Invalid format (has suffix)
        with self.assertRaises(ValueError):
            self.store.save_component_version(component_data)

    def test_missing_required_fields(self) -> None:
        """Test handling of missing required fields."""
        # Missing ID
        component_data = {
            "version": "1.0.0",
            "component_type": "test"
        }
        
        with self.assertRaises(ValueError):
            self.store.save_component_version(component_data)
        
        # Missing version
        component_data = {
            "id": "missing_version",
            "component_type": "test"
        }
        
        with self.assertRaises(ValueError):
            self.store.save_component_version(component_data)

    def test_list_component_versions(self) -> None:
        """Test listing component versions."""
        versions = self.store.list_component_versions("principle1")
        self.assertEqual(set(versions), {"1.0.0", "1.1.0"})
        
        # Test with a component that doesn't exist
        with self.assertRaises(ComponentNotFoundInStoreError):
            self.store.list_component_versions("nonexistent")

    def test_list_components(self) -> None:
        """Test listing all components."""
        components = self.store.list_components()
        self.assertEqual(set(components), {"principle1", "identity1"})
        
        # Test with a type filter
        principle_components = self.store.list_components(component_type="principle")
        self.assertEqual(set(principle_components), {"principle1"})
        
        identity_components = self.store.list_components(component_type="identity")
        self.assertEqual(set(identity_components), {"identity1"})
        
        # Test with a type that doesn't match any components
        nonexistent_components = self.store.list_components(component_type="nonexistent")
        self.assertEqual(nonexistent_components, [])

    def test_delete_component_version(self) -> None:
        """Test deleting a component version."""
        # First verify it exists
        versions = self.store.list_component_versions("principle1")
        self.assertIn("1.0.0", versions)
        
        # Archive it
        self.store.delete_component_version("principle1", "1.0.0", archive=True)
        
        # Should no longer be in the active versions
        versions = self.store.list_component_versions("principle1", include_archived=False)
        self.assertNotIn("1.0.0", versions)
        
        # But should be in the archived versions
        versions = self.store.list_component_versions("principle1", include_archived=True)
        self.assertIn("1.0.0", versions)
        
        # Delete a version that doesn't exist
        with self.assertRaises(ComponentVersionNotFoundError):
            self.store.delete_component_version("principle1", "2.0.0")
            
        # Delete a component that doesn't exist
        with self.assertRaises(ComponentNotFoundInStoreError):
            self.store.delete_component_version("nonexistent", "1.0.0")
            
        # Now truly delete without archiving
        self.store.delete_component_version("principle1", "1.1.0", archive=False)
        
        # Should not be found in either active or archived
        with self.assertRaises(ComponentVersionNotFoundError):
            self.store.load_component_version("principle1", "1.1.0")

    def test_delete_component(self) -> None:
        """Test deleting an entire component."""
        # First verify it exists
        components = self.store.list_components()
        self.assertIn("identity1", components)
        
        # Archive it
        self.store.delete_component("identity1", archive=True)
        
        # Should no longer be in the active components
        components = self.store.list_components(include_archived=False)
        self.assertNotIn("identity1", components)
        
        # But should be in the archived components
        components = self.store.list_components(include_archived=True)
        self.assertIn("identity1", components)
        
        # Delete a component that doesn't exist
        with self.assertRaises(ComponentNotFoundInStoreError):
            self.store.delete_component("nonexistent")
            
        # Now truly delete without archiving
        # First save a new version
        self.store.save_component_version({
            "id": "temp_component",
            "version": "1.0.0",
            "component_type": "temp"
        })
        
        self.store.delete_component("temp_component", archive=False)
        
        # Should be completely gone
        with self.assertRaises(ComponentNotFoundInStoreError):
            self.store.load_latest_component_version("temp_component")

    def test_restore_component_version(self) -> None:
        """Test restoring an archived component version."""
        # First archive a version
        self.store.delete_component_version("principle1", "1.0.0", archive=True)
        
        # Make sure it's archived
        versions = self.store.list_component_versions("principle1", include_archived=False)
        self.assertNotIn("1.0.0", versions)
        
        # Restore it
        self.store.restore_component_version("principle1", "1.0.0")
        
        # Should be back in the active versions
        versions = self.store.list_component_versions("principle1", include_archived=False)
        self.assertIn("1.0.0", versions)
        
        # Try to restore a version that isn't archived
        with self.assertRaises(ComponentVersionNotFoundError):
            self.store.restore_component_version("principle1", "3.0.0")
            
        # Try to restore a component that doesn't exist
        with self.assertRaises(ComponentNotFoundInStoreError):
            self.store.restore_component_version("nonexistent", "1.0.0")

    def test_integrity_verification(self) -> None:
        """Test component integrity verification."""
        # Create a component with integrity checking
        component_data = {
            "id": "integrity_test",
            "version": "1.0.0",
            "component_type": "test",
            "data": "original"
        }
        
        self.store.save_component_version(component_data, validate_integrity=True)
        
        # Load it with integrity checking (should succeed)
        loaded_data = self.store.load_component_version("integrity_test", "1.0.0", verify_integrity=True)
        self.assertEqual(loaded_data, component_data)
        
        # Tamper with the data file directly
        component_dir = Path(self.test_dir) / "integrity_test" / "1.0.0"
        component_path = component_dir / "component.json"
        
        with open(component_path, 'r') as f:
            data = json.load(f)
        
        # Modify the data
        data["data"] = "tampered"
        
        with open(component_path, 'w') as f:
            json.dump(data, f)
        
        # Try to load it with integrity checking (should fail)
        with self.assertRaises(ComponentIntegrityError):
            self.store.load_component_version("integrity_test", "1.0.0", verify_integrity=True)
        
        # Should still load with integrity checking disabled
        tampered_data = self.store.load_component_version("integrity_test", "1.0.0", verify_integrity=False)
        self.assertEqual(tampered_data["data"], "tampered")

    def test_verify_store_integrity(self) -> None:
        """Test store-wide integrity verification."""
        # Create a test component
        component_data = {
            "id": "integrity_verify_test",
            "version": "1.0.0",
            "component_type": "test",
            "data": "original"
        }
        
        self.store.save_component_version(component_data)
        
        # Tamper with the data file directly
        component_dir = Path(self.test_dir) / "integrity_verify_test" / "1.0.0"
        component_path = component_dir / "component.json"
        
        with open(component_path, 'r') as f:
            data = json.load(f)
        
        # Modify the data
        data["data"] = "tampered"
        
        with open(component_path, 'w') as f:
            json.dump(data, f)
        
        # Verify the store integrity
        integrity_issues = self.store.verify_store_integrity(repair=False)
        
        # Should report an issue with this component
        self.assertIn("integrity_verify_test", integrity_issues)
        issue = integrity_issues["integrity_verify_test"][0]
        self.assertEqual(issue["issue_type"], "integrity_error")
        self.assertEqual(issue["version"], "1.0.0")
        
        # Now verify with repair
        integrity_issues = self.store.verify_store_integrity(repair=True)
        
        # The issue should now be marked as repaired
        self.assertIn("integrity_verify_test", integrity_issues)
        issue = integrity_issues["integrity_verify_test"][0]
        self.assertEqual(issue["repair_action"], "archived_corrupt_version")
        
        # The version should now be archived
        versions = self.store.list_component_versions("integrity_verify_test", include_archived=False)
        self.assertNotIn("1.0.0", versions)
        
        versions = self.store.list_component_versions("integrity_verify_test", include_archived=True)
        self.assertIn("1.0.0", versions)

    def test_backup_and_restore(self) -> None:
        """Test creating a backup and restoring from it."""
        # Create a backup
        backup_file = os.path.join(self.test_dir, "backup.zip")
        self.store.create_backup(backup_file)
        
        # Verify the backup file exists
        self.assertTrue(os.path.exists(backup_file))
        
        # Delete a component
        self.store.delete_component("principle1", archive=False)
        
        # Verify it's gone
        with self.assertRaises(ComponentNotFoundInStoreError):
            self.store.load_latest_component_version("principle1")
        
        # Restore from backup
        new_store = VersionedComponentStore(storage_root=self.test_dir)
        new_store.restore_from_backup(backup_file)
        
        # Verify the component is restored
        principle_data = new_store.load_latest_component_version("principle1")
        self.assertEqual(principle_data["name"], "Test Principle 1 - Revised")
        
        # Verify the backup file includes a manifest
        import zipfile
        with zipfile.ZipFile(backup_file, 'r') as zf:
            self.assertIn("backup_manifest.json", zf.namelist())
            
            # Read the manifest
            with zf.open("backup_manifest.json") as f:
                manifest = json.load(f)
                
            # Should have metadata about the backup
            self.assertIn("timestamp", manifest)
            self.assertIn("component_count", manifest)
            self.assertIn("components", manifest)
            
            # Should list our components
            self.assertIn("principle1", manifest["components"])
            self.assertIn("identity1", manifest["components"])
