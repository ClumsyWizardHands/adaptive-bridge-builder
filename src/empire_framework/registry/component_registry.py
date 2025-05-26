"""
Component Registry for Empire Framework

This module provides the ComponentRegistry class, which manages Empire Framework components,
handling CRUD operations, validation, versioning, and relationship queries.
"""

import copy
import json
import re
import uuid
import datetime
from datetime import timezone
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from pathlib import Path
import logging

# Import the validation functions
from empire_framework.validation.schema_validator import (
    validate_component, 
    check_component,
    ValidationError
)

# Set up logging
logger = logging.getLogger(__name__)


# Custom Exception Classes
class RegistryError(Exception):
    """Base class for all registry-related errors."""
    pass


class ComponentNotFoundError(RegistryError):
    """Raised when a requested component cannot be found."""
    def __init__(self, component_id: str, version: Optional[str] = None) -> None:
        self.component_id = component_id
        self.version = version
        if version:
            message = f"Component with ID '{component_id}' and version '{version}' not found"
        else:
            message = f"Component with ID '{component_id}' not found"
        super().__init__(message)


class ComponentAlreadyExistsError(RegistryError):
    """Raised when attempting to add a component that already exists."""
    def __init__(self, component_id: str) -> None:
        self.component_id = component_id
        message = f"Component with ID '{component_id}' already exists"
        super().__init__(message)


class InvalidVersionError(RegistryError):
    """Raised when a version string is invalid."""
    def __init__(self, version: str) -> None:
        self.version = version
        message = f"Invalid version format: '{version}'. Expected format: X.Y.Z"
        super().__init__(message)


class ComponentVersionConflictError(RegistryError):
    """Raised when there's a version conflict during update."""
    def __init__(self, component_id: str, attempted_version: str, current_version: str) -> None:
        self.component_id = component_id
        self.attempted_version = attempted_version
        self.current_version = current_version
        message = (f"Version conflict for component '{component_id}': "
                  f"Attempted to update version '{attempted_version}' but current version is '{current_version}'")
        super().__init__(message)


class ComponentRegistry:
    """
    Registry for Empire Framework components.
    
    This class provides functionality to manage Empire Framework components, including
    adding, retrieving, updating, and querying components with support for validation
    and versioning.
    """

    def __init__(self, storage_backend=None) -> None:
        """
        Initialize a new ComponentRegistry.
        
        Args:
            storage_backend: Optional storage backend for persistence.
                             If None, an in-memory store is used.
        """
        # Storage backend will be implemented in the future
        # For now, use a simple in-memory dictionary
        self._storage_backend = storage_backend
        
        # Dictionary to store components by ID and version
        # Structure: {component_id: {version: component_data}}
        self._components: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # Index of components by type for faster lookup
        # Structure: {component_type: set(component_ids)}
        self._type_index: Dict[str, Set[str]] = {}
        
        # Index of components by tag for faster lookup
        # Structure: {tag: set(component_ids)}
        self._tag_index: Dict[str, Set[str]] = {}
        
        # Index of components by operational status
        # Structure: {status: set(component_ids)}
        self._status_index: Dict[str, Set[str]] = {}
        
        # Index of relationships for faster relationship queries
        # Structure: {source_id: {relationship_type: set(target_ids)}}
        self._relationship_index: Dict[str, Dict[str, Set[str]]] = {}
        
        # Latest version index for quick access to the latest version of each component
        # Structure: {component_id: latest_version}
        self._latest_version_index: Dict[str, str] = {}
        
        logger.info("ComponentRegistry initialized")

    def add_component(self, component_data: Dict[str, Any], validate: bool = True) -> Dict[str, Any]:
        """
        Add a new component to the registry.
        
        Args:
            component_data: The component data to add
            validate: Whether to validate the component before adding
                      
        Returns:
            The added component data (which may include modified fields like ID and version)
            
        Raises:
            ValidationError: If the component fails validation
            ComponentAlreadyExistsError: If a component with the same ID already exists
            InvalidVersionError: If the version format is invalid
        """
        # Deep copy to avoid modifying the original data
        component = copy.deepcopy(component_data)
        
        # Generate a new UUID if one isn't provided
        if 'id' not in component or not component['id']:
            component['id'] = str(uuid.uuid4())
        component_id = component['id']
        
        # Check if component already exists
        if component_id in self._components:
            raise ComponentAlreadyExistsError(component_id)
        
        # Set initial version if not provided
        if 'version' not in component or not component['version']:
            component['version'] = "1.0.0"
        
        # Validate version format
        if not self._is_valid_version_format(component['version']):
            raise InvalidVersionError(component['version'])
        
        # Set creation and modification timestamps if not provided
        current_time = datetime.datetime.now().isoformat()
        if 'creation_date' not in component or not component['creation_date']:
            component['creation_date'] = current_time
        if 'last_modified_date' not in component or not component['last_modified_date']:
            component['last_modified_date'] = current_time
        
        # Validate component if required
        if validate:
            check_component(component)
        
        # Store the component
        self._components = {**self._components, component_id: {component['version']: component}}
        self._latest_version_index = {**self._latest_version_index, component_id: component['version']}
        
        # Update indexes
        self._update_indexes(component)
        
        logger.info(f"Added component {component_id} version {component['version']}")
        return component

    def get_component(self, component_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve a component by ID and optionally version.
        
        Args:
            component_id: The ID of the component to retrieve
            version: Optional version to retrieve. If None, retrieves the latest version.
            
        Returns:
            The component data (deep copy to prevent modification)
            
        Raises:
            ComponentNotFoundError: If the component doesn't exist
        """
        if component_id not in self._components:
            raise ComponentNotFoundError(component_id)
        
        # If version not specified, use the latest
        if version is None:
            version = self._latest_version_index[component_id]
        
        # Check if the specific version exists
        if version not in self._components[component_id]:
            raise ComponentNotFoundError(component_id, version)
        
        # Return a deep copy to prevent modification of stored data
        return copy.deepcopy(self._components[component_id][version])

    def update_component(self, component_id: str, updated_data: Dict[str, Any], 
                        validate: bool = True, 
                        auto_increment_version: bool = True) -> Dict[str, Any]:
        """
        Update an existing component.
        
        Args:
            component_id: The ID of the component to update
            updated_data: The updated component data
            validate: Whether to validate the updated component
            auto_increment_version: Whether to automatically increment the version
            
        Returns:
            The updated component data
            
        Raises:
            ComponentNotFoundError: If the component doesn't exist
            ValidationError: If the updated component fails validation
            ComponentVersionConflictError: If the version in updated_data conflicts
        """
        if component_id not in self._components:
            raise ComponentNotFoundError(component_id)
        
        # Get the current latest version
        current_version = self._latest_version_index[component_id]
        current_component = self._components[component_id][current_version]
        
        # Start with a copy of the current component
        updated_component = copy.deepcopy(current_component)
        
        # Apply the updates
        for key, value in updated_data.items():
            # Don't allow changing the ID
            if key == 'id':
                continue
            updated_component[key] = value
        
        # Handle versioning
        if 'version' in updated_data:
            # Use the provided version
            if updated_data['version'] != current_version:
                # Validate that this wouldn't create conflicts
                if updated_data['version'] in self._components[component_id]:
                    raise ComponentVersionConflictError(
                        component_id, updated_data['version'], current_version
                    )
        elif auto_increment_version:
            # Automatically increment the version
            updated_component['version'] = self._increment_version(current_version)
        
        # Update last_modified_date
        updated_component['last_modified_date'] = datetime.datetime.now().isoformat()
        
        # Validate updated component if required
        if validate:
            check_component(updated_component)
        
        # Remove old indexes before updating
        self._remove_from_indexes(current_component)
        
        # Store the updated component
        new_version = updated_component['version']
        self._components[component_id][new_version] = updated_component
        self._latest_version_index = {**self._latest_version_index, component_id: new_version}
        
        # Update indexes with the new component
        self._update_indexes(updated_component)
        
        logger.info(f"Updated component {component_id} to version {new_version}")
        return copy.deepcopy(updated_component)

    def delete_component(self, component_id: str, version: Optional[str] = None, 
                         archive: bool = True) -> None:
        """
        Delete a component (or a specific version of a component).
        
        Args:
            component_id: The ID of the component to delete
            version: Optional specific version to delete. If None, all versions are deleted.
            archive: If True, mark as archived instead of actually deleting
            
        Raises:
            ComponentNotFoundError: If the component doesn't exist
        """
        if component_id not in self._components:
            raise ComponentNotFoundError(component_id)
        
        if version is not None:
            # Delete a specific version
            if version not in self._components[component_id]:
                raise ComponentNotFoundError(component_id, version)
            
            if archive:
                # Mark as archived
                component = self._components[component_id][version]
                component['operational_status'] = 'Archived'
                component['last_modified_date'] = datetime.datetime.now().isoformat()
                # Update indexes
                self._update_indexes(component)
                logger.info(f"Archived component {component_id} version {version}")
            else:
                # Actually delete the version
                self._remove_from_indexes(self._components[component_id][version])
                del self._components[component_id][version]
                
                # Update latest version index if needed
                if version == self._latest_version_index[component_id]:
                    if self._components[component_id]:
                        # Find the new latest version
                        self._latest_version_index = {**self._latest_version_index, component_id: max(
                            self._components[component_id].keys(), 
                            key=self._version_to_tuple
                        )}
                    else:
                        # No versions left, remove from index
                        self._latest_version_index = {k: v for k, v in self._latest_version_index.items() if k != component_id}
                
                logger.info(f"Deleted component {component_id} version {version}")
        else:
            # Delete all versions of the component
            if archive:
                # Mark all versions as archived
                for ver, component in self._components[component_id].items():
                    component['operational_status'] = 'Archived'
                    component['last_modified_date'] = datetime.datetime.now().isoformat()
                    self._update_indexes(component)
                logger.info(f"Archived all versions of component {component_id}")
            else:
                # Actually delete all versions
                for ver, component in self._components[component_id].items():
                    self._remove_from_indexes(component)
                self._components = {k: v for k, v in self._components.items() if k != component_id}
                if component_id in self._latest_version_index:
                    self._latest_version_index = {k: v for k, v in self._latest_version_index.items() if k != component_id}
                logger.info(f"Deleted all versions of component {component_id}")

    def get_component_versions(self, component_id: str) -> List[str]:
        """
        Get all available versions of a component.
        
        Args:
            component_id: The ID of the component
            
        Returns:
            List of version strings sorted by semantic versioning
            
        Raises:
            ComponentNotFoundError: If the component doesn't exist
        """
        if component_id not in self._components:
            raise ComponentNotFoundError(component_id)
        
        # Sort versions using semantic versioning comparison
        return sorted(self._components[component_id].keys(), 
                      key=self._version_to_tuple)

    def get_latest_version(self, component_id: str) -> str:
        """
        Get the latest version of a component.
        
        Args:
            component_id: The ID of the component
            
        Returns:
            The latest version string
            
        Raises:
            ComponentNotFoundError: If the component doesn't exist
        """
        if component_id not in self._latest_version_index:
            raise ComponentNotFoundError(component_id)
        
        return self._latest_version_index[component_id]

    def get_components_by_type(self, component_type: str, 
                              include_archived: bool = False) -> List[Dict[str, Any]]:
        """
        Get all components of a specific type.
        
        Args:
            component_type: The type of components to retrieve
            include_archived: Whether to include archived components
            
        Returns:
            List of component data (latest versions)
        """
        if component_type not in self._type_index:
            return []
        
        components = []
        for component_id in self._type_index[component_type]:
            component = self.get_component(component_id)
            if include_archived or component.get('operational_status') != 'Archived':
                components.append(component)
        
        return components

    def get_components_by_tag(self, tag: str, 
                             include_archived: bool = False) -> List[Dict[str, Any]]:
        """
        Get all components with a specific tag.
        
        Args:
            tag: The tag to filter by
            include_archived: Whether to include archived components
            
        Returns:
            List of component data (latest versions)
        """
        if tag not in self._tag_index:
            return []
        
        components = []
        for component_id in self._tag_index[tag]:
            component = self.get_component(component_id)
            if include_archived or component.get('operational_status') != 'Archived':
                components.append(component)
        
        return components

    def get_components_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        Get all components with a specific operational status.
        
        Args:
            status: The operational status to filter by
            
        Returns:
            List of component data (latest versions)
        """
        if status not in self._status_index:
            return []
        
        components = []
        for component_id in self._status_index[status]:
            components.append(self.get_component(component_id))
        
        return components

    def query_components(self, filters: Dict[str, Any] = None,
                        sort_by: str = None,
                        descending: bool = False,
                        limit: int = None,
                        offset: int = 0) -> List[Dict[str, Any]]:
        """
        Query components using multiple filters.
        
        Args:
            filters: Dictionary of field names and values to filter by
            sort_by: Field to sort results by
            descending: Whether to sort in descending order
            limit: Maximum number of results to return
            offset: Number of results to skip
            
        Returns:
            List of component data matching the query
        """
        # Initialize with all component IDs
        result_ids = set(self._components.keys())
        
        # Apply filters if provided
        if filters:
            for field, value in filters.items():
                if field == 'component_type' and value in self._type_index:
                    result_ids &= self._type_index[value]
                elif field == 'tags' and isinstance(value, list):
                    # Intersect with all tags
                    tag_ids = set(result_ids)
                    for tag in value:
                        if tag in self._tag_index:
                            tag_ids &= self._tag_index[tag]
                    result_ids = tag_ids
                elif field == 'operational_status' and value in self._status_index:
                    result_ids &= self._status_index[value]
                else:
                    # Filter by field value directly
                    filtered_ids = set()
                    for component_id in result_ids:
                        component = self.get_component(component_id)
                        if field in component and component[field] == value:
                            filtered_ids.add(component_id)
                    result_ids = filtered_ids
        
        # Get the actual components
        results = [self.get_component(component_id) for component_id in result_ids]
        
        # Sort if requested
        if sort_by:
            results.sort(
                key=lambda c: c.get(sort_by, ''),
                reverse=descending
            )
        
        # Apply pagination
        if offset > 0:
            results = results[offset:]
        if limit is not None:
            results = results[:limit]
        
        return results

    def get_related_components(self, component_id: str, 
                              relationship_type: str = None,
                              include_archived: bool = False) -> List[Dict[str, Any]]:
        """
        Get components related to the specified component.
        
        Args:
            component_id: The ID of the component to find relations for
            relationship_type: Optional type of relationship to filter by
            include_archived: Whether to include archived components
            
        Returns:
            List of related component data
            
        Raises:
            ComponentNotFoundError: If the component doesn't exist
        """
        if component_id not in self._components:
            raise ComponentNotFoundError(component_id)
        
        if component_id not in self._relationship_index:
            return []
        
        related_ids = set()
        
        if relationship_type:
            # Get relationships of a specific type
            if relationship_type in self._relationship_index[component_id]:
                related_ids = self._relationship_index[component_id][relationship_type]
        else:
            # Get all relationships
            for rel_type, ids in self._relationship_index[component_id].items():
                related_ids.update(ids)
        
        # Retrieve the actual components
        related_components = []
        for related_id in related_ids:
            try:
                component = self.get_component(related_id)
                if include_archived or component.get('operational_status') != 'Archived':
                    related_components.append(component)
            except ComponentNotFoundError:
                # Skip if component doesn't exist
                continue
        
        return related_components

    def get_components_with_relationship_to(self, target_id: str, 
                                          relationship_type: str = None,
                                          include_archived: bool = False) -> List[Dict[str, Any]]:
        """
        Get components that have a relationship pointing to the target component.
        
        Args:
            target_id: The ID of the target component
            relationship_type: Optional type of relationship to filter by
            include_archived: Whether to include archived components
            
        Returns:
            List of component data that have relationships to the target
        """
        components = []
        
        # Search through the relationship index
        for component_id, relationships in self._relationship_index.items():
            for rel_type, target_ids in relationships.items():
                if relationship_type is None or rel_type == relationship_type:
                    if target_id in target_ids:
                        try:
                            component = self.get_component(component_id)
                            if include_archived or component.get('operational_status') != 'Archived':
                                components.append(component)
                        except ComponentNotFoundError:
                            # Skip if component doesn't exist
                            continue
                        break  # Found a matching relationship, no need to check more
        
        return components

    def count_components(self, by_type: bool = False, 
                        by_status: bool = False) -> Union[int, Dict[str, int]]:
        """
        Count components in the registry.
        
        Args:
            by_type: Whether to break down counts by component type
            by_status: Whether to break down counts by operational status
            
        Returns:
            Total count or dictionary of counts by type/status
        """
        if by_type:
            return {
                component_type: len(components)
                for component_type, components in self._type_index.items()
            }
        elif by_status:
            return {
                status: len(components)
                for status, components in self._status_index.items()
            }
        else:
            return len(self._components)

    def validate_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Validate all components in the registry.
        
        Returns:
            Dictionary mapping component IDs to lists of validation errors
        """
        validation_errors = {}
        
        for component_id, versions in self._components.items():
            for version, component in versions.items():
                is_valid, errors = validate_component(component)
                if not is_valid:
                    if component_id not in validation_errors:
                        validation_errors[component_id] = []
                    validation_errors[component_id].extend(errors)
        
        return validation_errors

    # Private helper methods
    
    def _update_indexes(self, component: Dict[str, Any]) -> None:
        """Update all indexes with a component."""
        component_id = component['id']
        
        # Update component type index
        component_type = component.get('component_type')
        if component_type:
            if component_type not in self._type_index:
                self._type_index = {**self._type_index, component_type: set()}
            self._type_index[component_type].add(component_id)
        
        # Update tag index
        tags = component.get('tags', [])
        if tags:
            for tag in tags:
                if tag not in self._tag_index:
                    self._tag_index = {**self._tag_index, tag: set()}
                self._tag_index[tag].add(component_id)
        
        # Update status index
        status = component.get('operational_status')
        if status:
            if status not in self._status_index:
                self._status_index = {**self._status_index, status: set()}
            self._status_index[status].add(component_id)
        
        # Update relationship index
        relationships = component.get('relationships', [])
        if relationships:
            if component_id not in self._relationship_index:
                self._relationship_index = {**self._relationship_index, component_id: {}}
            
            for relationship in relationships:
                rel_type = relationship.get('relationship_type')
                target_id = relationship.get('target_component_id')
                
                if rel_type and target_id:
                    if rel_type not in self._relationship_index[component_id]:
                        self._relationship_index[component_id][rel_type] = set()
                    self._relationship_index[component_id][rel_type].add(target_id)

    def _remove_from_indexes(self, component: Dict[str, Any]) -> None:
        """Remove a component from all indexes."""
        component_id = component['id']
        
        # Remove from component type index
        component_type = component.get('component_type')
        if component_type and component_type in self._type_index:
            self._type_index[component_type].discard(component_id)
            if not self._type_index[component_type]:
                self._type_index = {k: v for k, v in self._type_index.items() if k != component_type}
        
        # Remove from tag index
        tags = component.get('tags', [])
        for tag in tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(component_id)
                if not self._tag_index[tag]:
                    self._tag_index = {k: v for k, v in self._tag_index.items() if k != tag}
        
        # Remove from status index
        status = component.get('operational_status')
        if status and status in self._status_index:
            self._status_index[status].discard(component_id)
            if not self._status_index[status]:
                self._status_index = {k: v for k, v in self._status_index.items() if k != status}
        
        # Remove from relationship index
        if component_id in self._relationship_index:
            self._relationship_index = {k: v for k, v in self._relationship_index.items() if k != component_id}
        
        # Also remove references to this component in other components' relationship indexes
        for source_id, relationships in self._relationship_index.items():
            for rel_type, target_ids in relationships.items():
                target_ids.discard(component_id)

    def _is_valid_version_format(self, version: str) -> bool:
        """Check if a version string is in valid semantic versioning format."""
        # Simple regex for semantic versioning
        pattern = r'^\d+\.\d+\.\d+$'
        return bool(re.match(pattern, version))

    def _version_to_tuple(self, version: str) -> Tuple[int, int, int]:
        """Convert a version string to a tuple for comparison."""
        try:
            return tuple(map(int, version.split('.')))
        except ValueError:
            # If the version can't be parsed, return a sentinel value
            return (-1, -1, -1)

    def _increment_version(self, version: str) -> str:
        """Increment the version using semantic versioning (patch increment)."""
        try:
            # Parse the version
            major, minor, patch = map(int, version.split('.'))
            # Increment the patch number
            return f"{major}.{minor}.{patch + 1}"
        except ValueError:
            # If the version can't be parsed, start from 1.0.0
            return "1.0.0"

    def export_to_json(self, path: str) -> None:
        """
        Export the registry to a JSON file.
        
        Args:
            path: Path to save the JSON file
        """
        # Prepare data for export - flatten the version structure
        export_data = []
        for component_id, versions in self._components.items():
            for version, component in versions.items():
                export_data.append(component)
        
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(export_data)} components to {path}")

    def import_from_json(self, path: str, validate: bool = True) -> int:
        """
        Import components from a JSON file.
        
        Args:
            path: Path to the JSON file
            validate: Whether to validate components before importing
            
        Returns:
            Number of components imported
        """
        with open(path, 'r') as f:
            components = json.load(f)
        
        imported_count = 0
        for component in components:
            try:
                if component['id'] in self._components:
                    # Update if component already exists
                    self.update_component(component['id'], component, validate=validate)
                else:
                    # Add if component doesn't exist
                    self.add_component(component, validate=validate)
                imported_count += 1
            except (ValidationError, RegistryError) as e:
                logger.warning(f"Error importing component: {str(e)}")
                continue
        
        logger.info(f"Imported {imported_count} components from {path}")
        return imported_count
