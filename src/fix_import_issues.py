#!/usr/bin/env python3
"""
Fix import issues identified by the import audit.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple


class ImportFixer:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.src_root = self.project_root / "src"
        self.fixes_applied = []
        
    def fix_empire_framework_imports(self):
        """Fix absolute imports in empire_framework to use relative imports."""
        empire_root = self.src_root / "empire_framework"
        
        # Fix validation __init__.py
        self._fix_init_imports(
            empire_root / "validation" / "__init__.py",
            ["schema_validator", "ValidationError", "validate_component", "check_component"]
        )
        
        # Fix registry __init__.py
        self._fix_init_imports(
            empire_root / "registry" / "__init__.py",
            ["component_registry", "ComponentRegistry", "RegistryError"]
        )
        
        # Fix storage __init__.py
        self._fix_init_imports(
            empire_root / "storage" / "__init__.py",
            ["versioned_component_store", "VersionedComponentStore", "StorageError"]
        )
        
        # Fix absolute imports in test files
        test_files = [
            (empire_root / "registry" / "test_component_registry.py", "src.empire_framework.", ".."),
            (empire_root / "storage" / "test_versioned_component_store.py", "src.empire_framework.", ".."),
            (empire_root / "validation" / "test_schema_validator.py", "src.empire_framework.", ".."),
            (empire_root / "validation" / "validator_example.py", "src.empire_framework.", "..")
        ]
        
        for file_path, old_prefix, new_prefix in test_files:
            if file_path.exists():
                self._replace_imports_in_file(file_path, old_prefix, new_prefix)
                
        # Fix should_be_relative imports
        streaming_adapter = empire_root / "a2a" / "streaming_adapter.py"
        if streaming_adapter.exists():
            self._replace_import_line(
                streaming_adapter,
                "from empire_framework.a2a.a2a_adapter import",
                "from .a2a_adapter import"
            )
            
        adk_example = empire_root / "adk" / "example_usage.py"
        if adk_example.exists():
            self._replace_import_line(
                adk_example,
                "from empire_framework.adk.empire_adk_adapter import",
                "from .empire_adk_adapter import"
            )
    
    def fix_api_integration_assistant_imports(self):
        """Fix imports in api/integration_assistant modules."""
        api_root = self.src_root / "api" / "integration_assistant"
        
        # Fix app.py imports
        app_file = api_root / "app.py"
        if app_file.exists():
            replacements = [
                ("from models import", "from .models import"),
                ("from code_generator import", "from .code_generator import"),
                ("from websocket_manager import", "from .websocket_manager import"),
                ("import ai_framework_detector", "from ...ai_framework_detector import AIFrameworkDetector"),
                ("import universal_agent_connector", "from ...universal_agent_connector import UniversalAgentConnector"),
                ("import agent_registry", "from ...agent_registry import AgentRegistry"),
                ("from api.a2a.handlers import", "from ..a2a.handlers import")
            ]
            for old, new in replacements:
                self._replace_import_line(app_file, old, new)
        
        # Fix code_generator.py imports
        code_gen_file = api_root / "code_generator.py"
        if code_gen_file.exists():
            self._replace_import_line(code_gen_file, "from models import", "from .models import")
        
        # Fix websocket_manager.py imports
        ws_file = api_root / "websocket_manager.py"
        if ws_file.exists():
            self._replace_import_line(ws_file, "from models import", "from .models import")
        
        # Fix __init__.py imports
        init_file = api_root / "__init__.py"
        if init_file.exists():
            replacements = [
                ("from models import", "from .models import"),
                ("from app import", "from .app import"),
                ("from websocket_manager import", "from .websocket_manager import")
            ]
            for old, new in replacements:
                self._replace_import_line(init_file, old, new)
    
    def fix_empire_framework_a2a_imports(self):
        """Fix imports in empire_framework/a2a modules."""
        a2a_root = self.src_root / "empire_framework" / "a2a"
        
        # Fix component_task_handler.py
        handler_file = a2a_root / "component_task_handler.py"
        if handler_file.exists():
            replacements = [
                ("from registry.component_registry import", "from ..registry.component_registry import"),
                ("from message_structures import", "from .message_structures import"),
                ("from validation.schema_validator import", "from ..validation.schema_validator import")
            ]
            for old, new in replacements:
                self._replace_import_line(handler_file, old, new)
    
    def fix_syntax_error(self):
        """Fix the syntax error in update_deprecated_datetime.py."""
        file_path = self.src_root / "update_deprecated_datetime.py"
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for unterminated string literals around line 41
                lines = content.split('\n')
                if len(lines) >= 41:
                    line_40 = lines[40] if len(lines) > 40 else ""
                    line_41 = lines[41] if len(lines) > 41 else ""
                    
                    # Check for common string literal issues
                    if line_40.count('"') % 2 == 1 or line_40.count("'") % 2 == 1:
                        # Fix by adding closing quote
                        if line_40.count('"') % 2 == 1:
                            lines[40] = line_40 + '"'
                        else:
                            lines[40] = line_40 + "'"
                        
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write('\n'.join(lines))
                        
                        self.fixes_applied.append(f"Fixed syntax error in {file_path}")
            except Exception as e:
                print(f"Error fixing syntax in {file_path}: {e}")
    
    def _fix_init_imports(self, file_path: Path, modules: List[str]):
        """Fix imports in __init__.py files to use relative imports."""
        if not file_path.exists():
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace absolute imports with relative imports
            for module in modules:
                # Match various import patterns
                patterns = [
                    (f"from {module} import", f"from .{module} import"),
                    (f"import {module}", f"from . import {module}")
                ]
                
                for old, new in patterns:
                    if old in content:
                        content = content.replace(old, new)
                        self.fixes_applied.append(f"Fixed import in {file_path}: {old} -> {new}")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
    
    def _replace_imports_in_file(self, file_path: Path, old_prefix: str, new_prefix: str):
        """Replace import prefixes in a file."""
        if not file_path.exists():
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace the prefix in import statements
            pattern = re.compile(rf"from\s+{re.escape(old_prefix)}(\S+)\s+import")
            new_content = pattern.sub(rf"from {new_prefix}\1 import", content)
            
            if new_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                self.fixes_applied.append(f"Fixed imports in {file_path}: {old_prefix} -> {new_prefix}")
                
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
    
    def _replace_import_line(self, file_path: Path, old_line: str, new_line: str):
        """Replace a specific import line in a file."""
        if not file_path.exists():
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            modified = False
            for i, line in enumerate(lines):
                if line.strip().startswith(old_line):
                    # Preserve the rest of the line after the import statement
                    remainder = line.strip()[len(old_line):]
                    lines[i] = new_line + remainder + '\n'
                    modified = True
                    self.fixes_applied.append(f"Fixed import in {file_path}: {old_line} -> {new_line}")
            
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                    
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
    
    def run_all_fixes(self):
        """Run all import fixes."""
        print("Starting import fixes...")
        
        self.fix_syntax_error()
        self.fix_empire_framework_imports()
        self.fix_api_integration_assistant_imports()
        self.fix_empire_framework_a2a_imports()
        
        print(f"\nCompleted {len(self.fixes_applied)} fixes:")
        for fix in self.fixes_applied:
            print(f"  - {fix}")
        
        print("\nImport fixes completed!")


def main():
    """Run the import fixer."""
    fixer = ImportFixer()
    fixer.run_all_fixes()


if __name__ == "__main__":
    main()
