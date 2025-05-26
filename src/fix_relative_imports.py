#!/usr/bin/env python3
"""
Fix false positive relative import issues reported by the import checker.

This script validates that relative imports are correctly structured within their packages.
"""

import os
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class RelativeImportValidator:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.src_root = self.project_root / "src"
        self.validation_results = []
        
    def validate_relative_imports_in_file(self, file_path: Path) -> List[Dict]:
        """Validate relative imports in a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            results = []
            package_dir = file_path.parent
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    # Check if it's a relative import
                    if node.level > 0:
                        # Relative import
                        import_from = node.module or ""
                        import_level = node.level
                        
                        # Calculate the target directory
                        target_dir = package_dir
                        for _ in range(import_level - 1):
                            target_dir = target_dir.parent
                            
                        # Check if the import can be resolved
                        if import_from:
                            # Import from a specific module
                            parts = import_from.split('.')
                            target_path = target_dir / Path(*parts)
                            
                            # Check if it exists as a module or package
                            module_exists = (
                                (target_path.with_suffix('.py')).exists() or
                                (target_path / '__init__.py').exists()
                            )
                            
                            if module_exists:
                                results.append({
                                    'file': str(file_path.relative_to(self.project_root)),
                                    'line': node.lineno,
                                    'import': f"{'.' * import_level}{import_from}",
                                    'status': 'valid',
                                    'resolved_to': str(target_path.relative_to(self.project_root))
                                })
                            else:
                                results.append({
                                    'file': str(file_path.relative_to(self.project_root)),
                                    'line': node.lineno,
                                    'import': f"{'.' * import_level}{import_from}",
                                    'status': 'invalid',
                                    'error': f"Cannot resolve to {target_path.relative_to(self.project_root)}"
                                })
                        else:
                            # Import from parent package
                            if target_dir.exists():
                                results.append({
                                    'file': str(file_path.relative_to(self.project_root)),
                                    'line': node.lineno,
                                    'import': '.' * import_level,
                                    'status': 'valid',
                                    'resolved_to': str(target_dir.relative_to(self.project_root))
                                })
                                
            return results
            
        except Exception as e:
            return [{
                'file': str(file_path.relative_to(self.project_root)),
                'error': str(e),
                'status': 'error'
            }]
    
    def validate_all_files(self) -> Dict:
        """Validate relative imports in all Python files."""
        print("Validating relative imports...")
        
        all_results = []
        files_checked = 0
        valid_imports = 0
        invalid_imports = 0
        
        # Get all Python files
        python_files = list(self.src_root.rglob("*.py"))
        
        for file_path in python_files:
            # Skip __pycache__
            if "__pycache__" in str(file_path):
                continue
                
            results = self.validate_relative_imports_in_file(file_path)
            if results:
                files_checked += 1
                for result in results:
                    if result.get('status') == 'valid':
                        valid_imports += 1
                    elif result.get('status') == 'invalid':
                        invalid_imports += 1
                        all_results.append(result)
        
        # Check specific files that were flagged
        flagged_files = [
            "src/api/integration_assistant/app.py",
            "src/api/integration_assistant/code_generator.py",
            "src/api/integration_assistant/websocket_manager.py",
            "src/api/integration_assistant/__init__.py",
            "src/empire_framework/a2a/component_task_handler.py",
            "src/empire_framework/a2a/streaming_adapter.py",
            "src/empire_framework/adk/example_usage.py",
            "src/empire_framework/registry/test_component_registry.py",
            "src/empire_framework/registry/__init__.py",
            "src/empire_framework/storage/test_versioned_component_store.py",
            "src/empire_framework/storage/__init__.py",
            "src/empire_framework/validation/test_schema_validator.py",
            "src/empire_framework/validation/validator_example.py",
            "src/empire_framework/validation/__init__.py",
        ]
        
        print("\n=== Checking Specifically Flagged Files ===")
        for file_path_str in flagged_files:
            file_path = self.project_root / file_path_str
            if file_path.exists():
                results = self.validate_relative_imports_in_file(file_path)
                print(f"\n{file_path_str}:")
                if results:
                    for result in results:
                        if 'import' in result:
                            print(f"  Line {result['line']}: {result['import']} -> {result['status']}")
                            if result['status'] == 'valid':
                                print(f"    Resolves to: {result['resolved_to']}")
                            elif result['status'] == 'invalid':
                                print(f"    Error: {result['error']}")
                else:
                    print("  No relative imports found")
        
        return {
            'summary': {
                'files_checked': files_checked,
                'valid_imports': valid_imports,
                'invalid_imports': invalid_imports,
                'flagged_files_analysis': len(flagged_files)
            },
            'invalid_imports': all_results
        }
    
    def generate_import_mapping(self) -> Dict[str, str]:
        """Generate a mapping of relative imports to absolute imports."""
        mapping = {}
        
        # Map for api.integration_assistant package
        api_integration_path = self.src_root / "api" / "integration_assistant"
        if api_integration_path.exists():
            for module_path in api_integration_path.glob("*.py"):
                if module_path.name != "__init__.py":
                    module_name = module_path.stem
                    mapping[f"api.{module_name}"] = f"src.api.integration_assistant.{module_name}"
        
        # Map for empire_framework subpackages
        empire_path = self.src_root / "empire_framework"
        if empire_path.exists():
            for subpackage in ["a2a", "adk", "registry", "storage", "validation"]:
                subpackage_path = empire_path / subpackage
                if subpackage_path.exists():
                    for module_path in subpackage_path.glob("*.py"):
                        if module_path.name != "__init__.py":
                            module_name = module_path.stem
                            mapping[f"empire_framework.{module_name}"] = f"src.empire_framework.{subpackage}.{module_name}"
                            # Also map parent relative imports
                            mapping[f"{subpackage}.{module_name}"] = f"src.empire_framework.{subpackage}.{module_name}"
        
        return mapping


def main():
    """Run the relative import validator."""
    validator = RelativeImportValidator()
    
    # Validate all files
    results = validator.validate_all_files()
    
    print("\n=== Validation Summary ===")
    print(f"Files checked: {results['summary']['files_checked']}")
    print(f"Valid imports: {results['summary']['valid_imports']}")
    print(f"Invalid imports: {results['summary']['invalid_imports']}")
    
    if results['invalid_imports']:
        print("\n=== Invalid Imports Found ===")
        for imp in results['invalid_imports']:
            print(f"{imp['file']}:{imp['line']} - {imp['import']} - {imp['error']}")
    else:
        print("\nAll relative imports are valid!")
    
    # Generate import mapping
    print("\n=== Import Resolution Mapping ===")
    mapping = validator.generate_import_mapping()
    for incorrect, correct in sorted(mapping.items()):
        print(f"{incorrect} -> {correct}")
    
    print("\nRelative import validation completed!")


if __name__ == "__main__":
    main()
