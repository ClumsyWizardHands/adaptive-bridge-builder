"""
Unit tests for the ComponentRegistry class.

This module contains tests for the ComponentRegistry functionality, including
CRUD operations, validation, versioning, filtering, and relationship queries.
"""

import unittest
import uuid
import datetime
from datetime import timezone
import json
import os
from typing import Dict, Any, List
from pathlib import Path
import tempfile

# Ensure src directory is in the path so we can import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from .component_registry import (
    ComponentRegistry,
    ComponentNotFoundError,
    ComponentAlreadyExistsError,
    InvalidVersionError,
    ComponentVersionConflictError,
    RegistryError
)

from ..validation.schema_validator import ValidationError
from ..validation.validator_example import (
    create_valid_end,
    create_valid_mean,
    create_valid_principle,
    create_valid_identity,
    create_valid_resentment,
    create_valid_emotion,
    create_invalid_end
)


class TestComponentRegistry(unittest.TestCase):
    """Test cases for the ComponentRegistry class."""

    def setUp(self) -> None:
        """Set up a fresh registry for each test."""
        self.registry = ComponentRegistry()
        
        # Create some test component data
        self.test_end = create_valid_end()
        self.test_mean = create_valid_mean()
        self.test_principle = create_valid_principle()
        self.test_identity = create_valid_identity()
        self.test_resentment = create_valid_resentment()
        self.test_emotion = create_valid_emotion()
        
        # Ensure unique IDs
        self.test_end['id'] = str(uuid.uuid4())
        self.test_mean['id'] = str(uuid.uuid4())
        self.test_principle['id'] = str(uuid.uuid4())
        self.test_identity['id'] = str(uuid.uuid4())
        self.test_resentment['id'] = str(uuid.uuid4())
        self.test_emotion['id'] = str(uuid.uuid4())
        
        # Set explicit versions
        self.test_end['version'] = '1.0.0'
        self.test_mean['version'] = '1.0.0'
        self.test_principle['version'] = '1.0.0'
        self.test_identity['version'] = '1.0.0'
        self.test_resentment['version'] = '1.0.0'
        self.test_emotion['version'] = '1.0.0'

    def test_add_component(self) -> None:
        """Test adding components to the registry."""
        # Add a component
        added = self.registry.add_component(self.test_end)
        
        # Verify it was added correctly
        self.assertEqual(added['id'], self.test_end['id'])
        self.assertEqual(added['version'], self.test_end['version'])
        
        # Try to add the same component again (should raise exception)
        with self.assertRaises(ComponentAlreadyExistsError):
            self.registry.add_component(self.test_end)
        
        # Add component without explicit ID (should generate one)
        test_component = create_valid_mean()
        del test_component['id']
        added = self.registry.add_component(test_component)
        self.assertTrue('id' in added)
        
        # Add component without explicit version (should default to 1.0.0)
        test_component = create_valid_principle()
        del test_component['version']
        added = self.registry.add_component(test_component)
        self.assertEqual(added['version'], '1.0.0')
        
        # Add invalid component with validation disabled
        invalid_component = create_invalid_end()
        added = self.registry.add_component(invalid_component, validate=False)
        self.assertEqual(added['id'], invalid_component['id'])
        
        # Add invalid component with validation enabled (should raise exception)
        with self.assertRaises(ValidationError):
            self.registry.add_component(create_invalid_end())

    def test_get_component(self) -> None:
        """Test retrieving components from the registry."""
        # Add a component
        self.registry.add_component(self.test_end)
        
        # Get the component
        component = self.registry.get_component(self.test_end['id'])
        self.assertEqual(component['id'], self.test_end['id'])
        self.assertEqual(component['component_type'], self.test_end['component_type'])
        
        # Get a component that doesn't exist (should raise exception)
        with self.assertRaises(ComponentNotFoundError):
            self.registry.get_component(str(uuid.uuid4()))
        
        # Get specific version
        component = self.registry.get_component(self.test_end['id'], version='1.0.0')
        self.assertEqual(component['version'], '1.0.0')
        
        # Get non-existent version (should raise exception)
        with self.assertRaises(ComponentNotFoundError):
            self.registry.get_component(self.test_end['id'], version='2.0.0')
        
        # Verify returned component is a copy (modifying it shouldn't affect the registry)
        component['description_text'] = 'Modified description'
        original = self.registry.get_component(self.test_end['id'])
        self.assertNotEqual(component['description_text'], original['description_text'])

    def test_update_component(self) -> None:
        """Test updating components in the registry."""
        # Add a component
        self.registry.add_component(self.test_end)
        
        # Update the component
        updated_data = {
            'description_text': 'Updated description',
            'status': 'ACTIVE'
        }
        updated = self.registry.update_component(self.test_end['id'], updated_data)
        
        # Verify it was updated correctly
        self.assertEqual(updated['description_text'], 'Updated description')
        self.assertEqual(updated['status'], 'ACTIVE')
        
        # Version should be auto-incremented
        self.assertEqual(updated['version'], '1.0.1')
        
        # Original version should still exist
        original = self.registry.get_component(self.test_end['id'], version='1.0.0')
        self.assertEqual(original['description_text'], self.test_end['description_text'])
        
        # Latest version should be the updated one
        latest = self.registry.get_component(self.test_end['id'])
        self.assertEqual(latest['version'], '1.0.1')
        
        # Update with explicit version
        updated_data = {
            'description_text': 'Version 2.0 description',
            'version': '2.0.0'
        }
        updated = self.registry.update_component(self.test_end['id'], updated_data)
        self.assertEqual(updated['version'], '2.0.0')
        
        # Update a component that doesn't exist (should raise exception)
        with self.assertRaises(ComponentNotFoundError):
            self.registry.update_component(str(uuid.uuid4()), updated_data)
        
        # Update with validation enabled (should validate)
        updated_data = {
            'description_text': 'Too short'  # Too short description (will fail validation)
        }
        with self.assertRaises(ValidationError):
            self.registry.update_component(self.test_end['id'], updated_data)
        
        # Update with validation disabled (should not validate)
        updated = self.registry.update_component(self.test_end['id'], updated_data, validate=False)
        self.assertEqual(updated['description_text'], 'Too short')
        
        # Try to update to an existing version (should raise exception)
        updated_data = {'version': '1.0.0'}
        with self.assertRaises(ComponentVersionConflictError):
            self.registry.update_component(self.test_end['id'], updated_data)
        
        # Update without auto-incrementing version
        updated_data = {'description_text': 'No version change'}
        updated = self.registry.update_component(
            self.test_end['id'], updated_data, auto_increment_version=False)
        self.assertEqual(updated['version'], '2.0.0')  # Still the latest version

    def test_delete_component(self) -> None:
        """Test deleting components from the registry."""
        # Add components
        self.registry.add_component(self.test_end)
        self.registry.add_component(self.test_mean)
        
        # Update to create multiple versions
        self.registry.update_component(self.test_end['id'], {'description_text': 'Updated'})
        
        # Delete a specific version (archive mode)
        self.registry.delete_component(self.test_end['id'], version='1.0.0', archive=True)
        
        # Should still be able to get it, but operational_status should be 'Archived'
        component = self.registry.get_component(self.test_end['id'], version='1.0.0')
        self.assertEqual(component['operational_status'], 'Archived')
        
        # Delete a specific version (non-archive mode)
        self.registry.delete_component(self.test_end['id'], version='1.0.0', archive=False)
        
        # Should no longer be able to get it
        with self.assertRaises(ComponentNotFoundError):
            self.registry.get_component(self.test_end['id'], version='1.0.0')
        
        # Should still be able to get the other version
        component = self.registry.get_component(self.test_end['id'], version='1.0.1')
        
        # Delete entire component (archive mode)
        self.registry.delete_component(self.test_mean['id'], archive=True)
        
        # Should still be able to get it, but operational_status should be 'Archived'
        component = self.registry.get_component(self.test_mean['id'])
        self.assertEqual(component['operational_status'], 'Archived')
        
        # Delete entire component (non-archive mode)
        self.registry.delete_component(self.test_end['id'], archive=False)
        
        # Should no longer be able to get it
        with self.assertRaises(ComponentNotFoundError):
            self.registry.get_component(self.test_end['id'])
        
        # Delete a component that doesn't exist (should raise exception)
        with self.assertRaises(ComponentNotFoundError):
            self.registry.delete_component(str(uuid.uuid4()))
        
        # Delete a specific version that doesn't exist (should raise exception)
        with self.assertRaises(ComponentNotFoundError):
            self.registry.delete_component(self.test_mean['id'], version='9.9.9')

    def test_versioning(self) -> None:
        """Test version management functionality."""
        # Add a component
        self.registry.add_component(self.test_end)
        
        # Create multiple versions through updates
        self.registry.update_component(
            self.test_end['id'], {'description_text': 'Version 1.0.1'})
        self.registry.update_component(
            self.test_end['id'], {'description_text': 'Version 1.0.2'})
        self.registry.update_component(
            self.test_end['id'], {'version': '2.0.0', 'description_text': 'Major version'})
        
        # Get all versions
        versions = self.registry.get_component_versions(self.test_end['id'])
        self.assertEqual(len(versions), 4)
        self.assertIn('1.0.0', versions)
        self.assertIn('1.0.1', versions)
        self.assertIn('1.0.2', versions)
        self.assertIn('2.0.0', versions)
        
        # Verify versions are sorted
        self.assertEqual(versions, ['1.0.0', '1.0.1', '1.0.2', '2.0.0'])
        
        # Get latest version
        latest = self.registry.get_latest_version(self.test_end['id'])
        self.assertEqual(latest, '2.0.0')
        
        # Invalid version format (should raise exception)
        with self.assertRaises(InvalidVersionError):
            self.registry.add_component({'id': str(uuid.uuid4()), 'version': 'invalid'})

    def test_filtering_and_querying(self) -> None:
        """Test filtering and querying functionality."""
        # Add various components
        self.registry.add_component(self.test_end)  # End type
        self.registry.add_component(self.test_mean)  # Mean type
        self.registry.add_component(self.test_principle)  # Principle type
        
        # Add tags to some components
        end_id = self.test_end['id']
        mean_id = self.test_mean['id']
        principle_id = self.test_principle['id']
        
        self.registry.update_component(end_id, {'tags': ['important', 'user-facing']})
        self.registry.update_component(mean_id, {'tags': ['important', 'technical']})
        self.registry.update_component(principle_id, {'tags': ['core', 'ethical']})
        
        # Add status to some components
        self.registry.update_component(end_id, {'operational_status': 'Active'})
        self.registry.update_component(mean_id, {'operational_status': 'Active'})
        self.registry.update_component(principle_id, {'operational_status': 'Experimental'})
        
        # Test get_components_by_type
        end_components = self.registry.get_components_by_type('End')
        self.assertEqual(len(end_components), 1)
        self.assertEqual(end_components[0]['id'], end_id)
        
        # Test get_components_by_tag
        important_components = self.registry.get_components_by_tag('important')
        self.assertEqual(len(important_components), 2)
        self.assertIn(end_id, [c['id'] for c in important_components])
        self.assertIn(mean_id, [c['id'] for c in important_components])
        
        ethical_components = self.registry.get_components_by_tag('ethical')
        self.assertEqual(len(ethical_components), 1)
        self.assertEqual(ethical_components[0]['id'], principle_id)
        
        # Test get_components_by_status
        active_components = self.registry.get_components_by_status('Active')
        self.assertEqual(len(active_components), 2)
        experimental_components = self.registry.get_components_by_status('Experimental')
        self.assertEqual(len(experimental_components), 1)
        
        # Archive a component
        self.registry.delete_component(end_id, archive=True)
        
        # Test include_archived parameter
        end_components = self.registry.get_components_by_type('End', include_archived=False)
        self.assertEqual(len(end_components), 0)
        end_components = self.registry.get_components_by_type('End', include_archived=True)
        self.assertEqual(len(end_components), 1)
        
        # Test query_components with filters
        results = self.registry.query_components(filters={'component_type': 'Mean'})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['id'], mean_id)
        
        results = self.registry.query_components(filters={'tags': ['important']})
        self.assertEqual(len(results), 1)  # Only the mean component (end is archived)
        
        # Test query_components with sorting
        all_components = self.registry.query_components(
            filters={}, sort_by='component_name', include_archived=True)
        names = [c['component_name'] for c in all_components]
        self.assertEqual(names, sorted(names))
        
        # Test query_components with pagination
        results = self.registry.query_components(limit=1, offset=0)
        self.assertEqual(len(results), 1)
        results = self.registry.query_components(limit=1, offset=1)
        self.assertEqual(len(results), 1)
        self.assertNotEqual(results[0]['id'], all_components[0]['id'])

    def test_relationship_queries(self) -> None:
        """Test relationship query functionality."""
        # Add components with relationships
        self.test_end['relationships'] = [
            {'target_component_id': self.test_mean['id'], 'relationship_type': 'SUPPORTS'}
        ]
        self.test_mean['relationships'] = [
            {'target_component_id': self.test_principle['id'], 'relationship_type': 'INFORMS'},
            {'target_component_id': self.test_identity['id'], 'relationship_type': 'SUPPORTS'}
        ]
        self.test_principle['relationships'] = [
            {'target_component_id': self.test_end['id'], 'relationship_type': 'INFORMS'}
        ]
        
        self.registry.add_component(self.test_end)
        self.registry.add_component(self.test_mean)
        self.registry.add_component(self.test_principle)
        self.registry.add_component(self.test_identity)
        
        # Test get_related_components
        related = self.registry.get_related_components(self.test_end['id'])
        self.assertEqual(len(related), 1)
        self.assertEqual(related[0]['id'], self.test_mean['id'])
        
        related = self.registry.get_related_components(self.test_mean['id'])
        self.assertEqual(len(related), 2)
        
        # Test filtering by relationship type
        related = self.registry.get_related_components(
            self.test_mean['id'], relationship_type='INFORMS')
        self.assertEqual(len(related), 1)
        self.assertEqual(related[0]['id'], self.test_principle['id'])
        
        # Test get_components_with_relationship_to
        components = self.registry.get_components_with_relationship_to(self.test_principle['id'])
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0]['id'], self.test_mean['id'])
        
        components = self.registry.get_components_with_relationship_to(self.test_end['id'])
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0]['id'], self.test_principle['id'])
        
        # Test filtering by relationship type
        components = self.registry.get_components_with_relationship_to(
            self.test_end['id'], relationship_type='SUPPORTS')
        self.assertEqual(len(components), 0)  # None have SUPPORTS relationship to end
        
        # Archive a component
        self.registry.delete_component(self.test_principle['id'], archive=True)
        
        # Test include_archived parameter
        related = self.registry.get_related_components(
            self.test_mean['id'], include_archived=False)
        self.assertEqual(len(related), 1)  # Only identity, not the archived principle
        
        related = self.registry.get_related_components(
            self.test_mean['id'], include_archived=True)
        self.assertEqual(len(related), 2)  # Both identity and principle

    def test_counting(self) -> None:
        """Test counting functionality."""
        # Add various components
        self.registry.add_component(self.test_end)
        self.registry.add_component(self.test_mean)
        self.registry.add_component(self.test_principle)
        self.registry.add_component(self.test_identity)
        self.registry.add_component(self.test_resentment)
        
        # Set statuses
        self.registry.update_component(
            self.test_end['id'], {'operational_status': 'Active'})
        self.registry.update_component(
            self.test_mean['id'], {'operational_status': 'Active'})
        self.registry.update_component(
            self.test_principle['id'], {'operational_status': 'Experimental'})
        self.registry.update_component(
            self.test_identity['id'], {'operational_status': 'ReviewNeeded'})
        self.registry.update_component(
            self.test_resentment['id'], {'operational_status': 'Active'})
        
        # Test total count
        count = self.registry.count_components()
        self.assertEqual(count, 5)
        
        # Test count by type
        counts = self.registry.count_components(by_type=True)
        self.assertEqual(counts['End'], 1)
        self.assertEqual(counts['Mean'], 1)
        self.assertEqual(counts['Principle'], 1)
        self.assertEqual(counts['Identity'], 1)
        self.assertEqual(counts['Resentment'], 1)
        
        # Test count by status
        counts = self.registry.count_components(by_status=True)
        self.assertEqual(counts['Active'], 3)
        self.assertEqual(counts['Experimental'], 1)
        self.assertEqual(counts['ReviewNeeded'], 1)
        
        # Archive a component
        self.registry.delete_component(self.test_end['id'], archive=True)
        
        # Test count by status after archiving
        counts = self.registry.count_components(by_status=True)
        self.assertEqual(counts['Active'], 2)
        self.assertEqual(counts['Archived'], 1)

    def test_validate_all(self) -> None:
        """Test validation of all components."""
        # Add valid components
        self.registry.add_component(self.test_end)
        self.registry.add_component(self.test_mean)
        
        # Add an invalid component with validation disabled
        invalid_component = create_invalid_end()
        self.registry.add_component(invalid_component, validate=False)
        
        # Test validate_all
        errors = self.registry.validate_all()
        self.assertEqual(len(errors), 1)
        self.assertIn(invalid_component['id'], errors)
        self.assertGreater(len(errors[invalid_component['id']]), 0)

    def test_import_export(self) -> None:
        """Test import/export functionality."""
        # Add some components
        self.registry.add_component(self.test_end)
        self.registry.add_component(self.test_mean)
        self.registry.add_component(self.test_principle)
        
        # Create a temporary file for export/import testing
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Export the registry
            self.registry.export_to_json(temp_path)
            
            # Create a new registry
            new_registry = ComponentRegistry()
            
            # Import the exported data
            count = new_registry.import_from_json(temp_path)
            
            # Check that all components were imported
            self.assertEqual(count, 3)
            
            # Verify the imported components match the original ones
            self.assertEqual(
                new_registry.get_component(self.test_end['id'])['component_name'],
                self.test_end['component_name']
            )
            self.assertEqual(
                new_registry.get_component(self.test_mean['id'])['component_name'],
                self.test_mean['component_name']
            )
            self.assertEqual(
                new_registry.get_component(self.test_principle['id'])['component_name'],
                self.test_principle['component_name']
            )
        finally:
            # Clean up
            os.remove(temp_path)


if __name__ == '__main__':
    unittest.main()
