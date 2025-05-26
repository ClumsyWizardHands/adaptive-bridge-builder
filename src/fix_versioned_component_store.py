#!/usr/bin/env python3
"""
Fix the truncated versioned_component_store.py file.
"""

import os

# Read the current file
with open('src/empire_framework/storage/versioned_component_store.py', 'r', encoding='utf-8') as f:
    content = f.read()

# The file is truncated at the create_backup method
# Find where it's truncated
truncation_point = content.rfind('def create_backup(self, backup_path: str, include_archived: bool = True) -> str:')

if truncation_point != -1:
    # Keep everything up to and including the method signature
    lines_before = content[:truncation_point].split('\n')
    indent = '    '  # Class method indentation
    
    # Add the complete create_backup method and any other missing methods
    completion = '''        """
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
'''
    
    # Write the fixed content
    fixed_content = content[:truncation_point] + completion
    
    with open('src/empire_framework/storage/versioned_component_store.py', 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("Fixed versioned_component_store.py")
else:
    print("Could not find truncation point. Manual fix needed.")
