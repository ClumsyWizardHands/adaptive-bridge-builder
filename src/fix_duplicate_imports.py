#!/usr/bin/env python3
"""
Fix duplicate import lines that were incorrectly added by the previous fix.
Removes the duplicate "from . import module" lines that were added before the actual imports.
"""

import os
from pathlib import Path
from typing import List, Tuple

class DuplicateImportFixer:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_path = project_root / "src"
        self.fixed_files = []
        
    def fix_file(self, filepath: Path) -> bool:
        """Fix duplicate imports in a single file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Look for pattern where we have duplicate imports
            fixed_lines = []
            i = 0
            changes_made = False
            
            while i < len(lines):
                # Check if current line is "from . import X" and next line is "from .X import"
                if (i + 1 < len(lines) and 
                    lines[i].strip().startswith("from . import ") and
                    lines[i+1].strip().startswith("from .")):
                    
                    # Extract module name from first line
                    first_line = lines[i].strip()
                    if "import " in first_line:
                        module_name = first_line.split("import ")[1].strip()
                        
                        # Check if next line starts with "from .module_name"
                        second_line = lines[i+1].strip()
                        if second_line.startswith(f"from .{module_name}"):
                            # Skip the first line (it's redundant)
                            fixed_lines.append(lines[i+1])
                            i += 2
                            changes_made = True
                            continue
                
                # Check for similar pattern with .. imports
                if (i + 1 < len(lines) and 
                    lines[i].strip().startswith("from ..") and
                    " import " in lines[i] and
                    lines[i+1].strip().startswith("from ..")):
                    
                    # Extract the parts
                    first_line = lines[i].strip()
                    second_line = lines[i+1].strip()
                    
                    # If they're importing from the same relative path, keep only the second
                    first_path = first_line.split(" import ")[0]
                    second_path = second_line.split(" import ")[0] if " import " in second_line else ""
                    
                    if first_path == second_path.rsplit(".", 1)[0]:
                        # Skip the first line
                        fixed_lines.append(lines[i+1])
                        i += 2
                        changes_made = True
                        continue
                
                fixed_lines.append(lines[i])
                i += 1
            
            if changes_made:
                # Write back the fixed content
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.writelines(fixed_lines)
                self.fixed_files.append(str(filepath.relative_to(self.project_root)))
                return True
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            
        return False
    
    def fix_all_files(self) -> None:
        """Fix all files that have the duplicate import issue."""
        # List of files that we know have the issue
        files_to_fix = [
            self.src_path / "api" / "integration_assistant" / "app.py",
            self.src_path / "api" / "integration_assistant" / "code_generator.py",
            self.src_path / "api" / "integration_assistant" / "websocket_manager.py",
            self.src_path / "api" / "integration_assistant" / "__init__.py",
            self.src_path / "empire_framework" / "a2a" / "component_task_handler.py",
            self.src_path / "empire_framework" / "a2a" / "streaming_adapter.py",
            self.src_path / "empire_framework" / "adk" / "example_usage.py",
            self.src_path / "empire_framework" / "registry" / "test_component_registry.py",
            self.src_path / "empire_framework" / "registry" / "__init__.py",
            self.src_path / "empire_framework" / "storage" / "test_versioned_component_store.py",
            self.src_path / "empire_framework" / "storage" / "__init__.py",
            self.src_path / "empire_framework" / "validation" / "test_schema_validator.py",
            self.src_path / "empire_framework" / "validation" / "validator_example.py",
            self.src_path / "empire_framework" / "validation" / "__init__.py",
        ]
        
        for filepath in files_to_fix:
            if filepath.exists():
                if self.fix_file(filepath):
                    print(f"Fixed: {filepath.relative_to(self.project_root)}")
    
    def generate_report(self) -> dict:
        """Generate a report of files fixed."""
        return {
            "files_fixed": len(self.fixed_files),
            "fixed_files": self.fixed_files
        }


def main():
    """Main function to fix duplicate imports."""
    project_root = Path(__file__).parent.parent
    fixer = DuplicateImportFixer(project_root)
    
    print("Fixing duplicate import issues...")
    fixer.fix_all_files()
    
    report = fixer.generate_report()
    print(f"\n=== Summary ===")
    print(f"Files fixed: {report['files_fixed']}")
    
    if report['files_fixed'] > 0:
        print("\nFixed files:")
        for file in report['fixed_files']:
            print(f"  - {file}")


if __name__ == "__main__":
    main()
