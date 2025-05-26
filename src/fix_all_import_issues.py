#!/usr/bin/env python3
"""
Comprehensive import issue fixer for the Adaptive Bridge Builder project.
Fixes:
1. Creates missing __init__.py files
2. Fixes relative import paths
3. Updates requirements.txt with missing packages
4. Generates a report of changes made
"""

import os
import ast
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set
import re

class ImportIssueFixer:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_path = project_root / "src"
        self.changes_made = {
            "init_files_created": [],
            "relative_imports_fixed": [],
            "requirements_added": set()
        }
        
    def create_missing_init_files(self) -> None:
        """Create missing __init__.py files."""
        missing_init_dirs = [
            self.src_path / "tests",
            self.src_path / "empire_framework" / "adk"
        ]
        
        for dir_path in missing_init_dirs:
            if dir_path.exists() and dir_path.is_dir():
                init_file = dir_path / "__init__.py"
                if not init_file.exists():
                    init_file.write_text('"""Package initialization."""\n')
                    self.changes_made["init_files_created"].append(str(init_file.relative_to(self.project_root)))
                    print(f"Created: {init_file.relative_to(self.project_root)}")
    
    def fix_relative_imports(self) -> None:
        """Fix relative import issues in empire_framework and api modules."""
        files_to_fix = [
            # API integration assistant files
            (self.src_path / "api" / "integration_assistant" / "app.py", [
                ("from .models import", "from . import models\nfrom .models import"),
                ("from .websocket_manager import", "from . import websocket_manager\nfrom .websocket_manager import"),
                ("from .code_generator import", "from . import code_generator\nfrom .code_generator import"),
                ("from ..a2a.handlers import", "from src.api.a2a.handlers import")
            ]),
            (self.src_path / "api" / "integration_assistant" / "code_generator.py", [
                ("from .models import", "from . import models\nfrom .models import")
            ]),
            (self.src_path / "api" / "integration_assistant" / "websocket_manager.py", [
                ("from .models import", "from . import models\nfrom .models import")
            ]),
            (self.src_path / "api" / "integration_assistant" / "__init__.py", [
                ("from .app import", "from . import app\nfrom .app import"),
                ("from .models import", "from . import models\nfrom .models import"),
                ("from .websocket_manager import", "from . import websocket_manager\nfrom .websocket_manager import")
            ]),
            
            # Empire framework files
            (self.src_path / "empire_framework" / "a2a" / "component_task_handler.py", [
                ("from .message_structures import", "from . import message_structures\nfrom .message_structures import"),
                ("from ..registry.component_registry import", "from ..registry import component_registry as cr\nfrom ..registry.component_registry import"),
                ("from ..validation.schema_validator import", "from ..validation import schema_validator as sv\nfrom ..validation.schema_validator import")
            ]),
            (self.src_path / "empire_framework" / "a2a" / "streaming_adapter.py", [
                ("from .a2a_adapter import", "from . import a2a_adapter\nfrom .a2a_adapter import")
            ]),
            (self.src_path / "empire_framework" / "adk" / "example_usage.py", [
                ("from .empire_adk_adapter import", "from . import empire_adk_adapter\nfrom .empire_adk_adapter import")
            ]),
            (self.src_path / "empire_framework" / "registry" / "test_component_registry.py", [
                ("from ..registry.component_registry import", "from . import component_registry\nfrom .component_registry import"),
                ("from ..validation.schema_validator import", "from ..validation import schema_validator\nfrom ..validation.schema_validator import"),
                ("from ..validation.validator_example import", "from ..validation import validator_example\nfrom ..validation.validator_example import")
            ]),
            (self.src_path / "empire_framework" / "registry" / "__init__.py", [
                ("from .component_registry import", "from . import component_registry\nfrom .component_registry import")
            ]),
            (self.src_path / "empire_framework" / "storage" / "test_versioned_component_store.py", [
                ("from ..storage.versioned_component_store import", "from . import versioned_component_store\nfrom .versioned_component_store import")
            ]),
            (self.src_path / "empire_framework" / "storage" / "__init__.py", [
                ("from .versioned_component_store import", "from . import versioned_component_store\nfrom .versioned_component_store import")
            ]),
            (self.src_path / "empire_framework" / "validation" / "test_schema_validator.py", [
                ("from ..validation.schema_validator import", "from . import schema_validator\nfrom .schema_validator import"),
                ("from ..validation.validator_example import", "from . import validator_example\nfrom .validator_example import")
            ]),
            (self.src_path / "empire_framework" / "validation" / "validator_example.py", [
                ("from ..validation.schema_validator import", "from . import schema_validator\nfrom .schema_validator import")
            ]),
            (self.src_path / "empire_framework" / "validation" / "__init__.py", [
                ("from .schema_validator import", "from . import schema_validator\nfrom .schema_validator import")
            ])
        ]
        
        for file_path, replacements in files_to_fix:
            if file_path.exists():
                content = file_path.read_text(encoding='utf-8')
                original_content = content
                
                for old, new in replacements:
                    if old in content and new not in content:
                        content = content.replace(old, new)
                
                if content != original_content:
                    file_path.write_text(content, encoding='utf-8')
                    self.changes_made["relative_imports_fixed"].append(str(file_path.relative_to(self.project_root)))
                    print(f"Fixed imports in: {file_path.relative_to(self.project_root)}")
    
    def update_requirements(self) -> None:
        """Update requirements.txt with missing packages."""
        # Map of import names to package names
        package_mapping = {
            'markdown': 'markdown',
            'emoji': 'emoji',
            'dicttoxml': 'dicttoxml', 
            'xmltodict': 'xmltodict',
            'reportlab': 'reportlab',
            'ctransformers': 'ctransformers',
            'llama_cpp': 'llama-cpp-python',
            'google.generativeai': 'google-generativeai',
            'types-requests': 'types-requests',
            'types-pandas': 'types-pandas'
        }
        
        # Read current requirements
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            current_reqs = set(line.strip().split('==')[0].split('>=')[0].split('~=')[0] 
                              for line in req_file.read_text().splitlines() 
                              if line.strip() and not line.startswith('#'))
        else:
            current_reqs = set()
        
        # Packages to add based on unresolved imports
        packages_to_add = [
            'markdown>=3.4.0',
            'emoji>=2.8.0',
            'dicttoxml>=1.7.16',
            'xmltodict>=0.13.0',
            'reportlab>=4.0.0',
            'ctransformers>=0.2.27',
            'llama-cpp-python>=0.2.20',
            'google-generativeai>=0.3.0',
            'types-requests>=2.31.0',
            'types-pandas>=2.0.0'
        ]
        
        # Add only missing packages
        new_packages = []
        for package in packages_to_add:
            pkg_name = package.split('>=')[0].split('==')[0]
            if pkg_name not in current_reqs:
                new_packages.append(package)
                self.changes_made["requirements_added"].add(pkg_name)
        
        if new_packages:
            # Append to requirements.txt
            with open(req_file, 'a', encoding='utf-8') as f:
                f.write('\n# Added by fix_all_import_issues.py\n')
                for package in sorted(new_packages):
                    f.write(f'{package}\n')
            print(f"Added {len(new_packages)} packages to requirements.txt")
    
    def generate_report(self) -> Dict:
        """Generate a report of all changes made."""
        return {
            "summary": {
                "init_files_created": len(self.changes_made["init_files_created"]),
                "relative_imports_fixed": len(self.changes_made["relative_imports_fixed"]),
                "requirements_added": len(self.changes_made["requirements_added"])
            },
            "details": {
                "init_files_created": self.changes_made["init_files_created"],
                "relative_imports_fixed": self.changes_made["relative_imports_fixed"],
                "requirements_added": sorted(list(self.changes_made["requirements_added"]))
            }
        }


def main():
    """Main function to fix all import issues."""
    project_root = Path(__file__).parent.parent
    fixer = ImportIssueFixer(project_root)
    
    print("Fixing import issues...")
    print("\n1. Creating missing __init__.py files...")
    fixer.create_missing_init_files()
    
    print("\n2. Fixing relative imports...")
    fixer.fix_relative_imports()
    
    print("\n3. Updating requirements.txt...")
    fixer.update_requirements()
    
    # Generate and save report
    report = fixer.generate_report()
    report_path = project_root / "import_fixes_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n=== Import Fix Summary ===")
    for key, value in report["summary"].items():
        print(f"{key}: {value}")
    
    print(f"\nDetailed report saved to: {report_path}")
    print("\nNext steps:")
    print("1. Run: pip install -r requirements.txt")
    print("2. Run the import checker again to verify all issues are resolved")
    print("3. Run tests to ensure everything works correctly")


if __name__ == "__main__":
    main()
