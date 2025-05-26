#!/usr/bin/env python3
"""
Fix malformed import statements that are missing what to import.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple


class MalformedImportFixer:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.src_root = self.project_root / "src"
        self.fixes_applied = []
    
    def fix_malformed_imports_in_file(self, file_path: Path):
        """Fix malformed imports in a single file."""
        if not file_path.exists():
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.splitlines()
            
            changed = False
            new_lines = []
            
            for line in lines:
                # Check for malformed imports like "from module import" with no target
                if re.match(r'^\s*from\s+[\w\.]+\s+import\s*$', line):
                    # This is a malformed import - remove it
                    changed = True
                    self.fixes_applied.append(f"Removed malformed import in {file_path.name}: {line.strip()}")
                    continue
                elif re.match(r'^\s*import\s*$', line):
                    # Bare import statement with nothing after it
                    changed = True
                    self.fixes_applied.append(f"Removed empty import in {file_path.name}: {line.strip()}")
                    continue
                
                new_lines.append(line)
            
            if changed:
                # Write back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(new_lines))
                    
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
    
    def fix_all_malformed_imports(self):
        """Fix malformed imports in all Python files."""
        print("Scanning for malformed imports...")
        
        # Get all Python files
        python_files = list(self.src_root.rglob("*.py"))
        
        for file_path in python_files:
            # Skip __pycache__
            if "__pycache__" in str(file_path):
                continue
                
            self.fix_malformed_imports_in_file(file_path)
        
        print(f"\nFixed {len(self.fixes_applied)} malformed imports:")
        for fix in self.fixes_applied[:20]:  # Show first 20
            print(f"  - {fix}")
        if len(self.fixes_applied) > 20:
            print(f"  ... and {len(self.fixes_applied) - 20} more")
        
        print("\nMalformed import fixes completed!")


def main():
    """Run the malformed import fixer."""
    fixer = MalformedImportFixer()
    fixer.fix_all_malformed_imports()


if __name__ == "__main__":
    main()
