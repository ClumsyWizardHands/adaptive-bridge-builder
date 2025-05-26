#!/usr/bin/env python3
"""
Analyze import patterns to identify potential improvements.
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json

class ImportPatternAnalyzer:
    def __init__(self, root_path: str = "src"):
        self.root_path = Path(root_path)
        self.findings: List[Dict[str, any]] = []
        
    def analyze_file(self, filepath: Path) -> None:
        """Analyze import patterns in a single file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content, filename=str(filepath))
            rel_path = str(filepath.relative_to(self.root_path.parent))
            
            # Check for various patterns
            self._check_import_organization(tree, rel_path)
            self._check_relative_vs_absolute(tree, rel_path)
            self._check_circular_risk(tree, rel_path)
            
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
            
    def _check_import_organization(self, tree: ast.AST, filepath: str) -> None:
        """Check if imports are well-organized."""
        imports = []
        first_non_import_line = None
        
        for i, node in enumerate(tree.body):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append((i, node))
            elif first_non_import_line is None and not isinstance(node, (ast.Expr, ast.Assign)):
                first_non_import_line = i
                
        # Check if imports are grouped
        if imports and first_non_import_line:
            last_import_idx = imports[-1][0]
            if last_import_idx > first_non_import_line:
                self.findings.append({
                    "type": "scattered_imports",
                    "file": filepath,
                    "description": "Imports are scattered throughout the file"
                })
                
    def _check_relative_vs_absolute(self, tree: ast.AST, filepath: str) -> None:
        """Check for opportunities to use relative imports."""
        file_parts = filepath.split('/')
        if len(file_parts) < 3:  # Not in a package
            return
            
        package = file_parts[1]  # e.g., 'empire_framework', 'api', etc.
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                # Check if importing from same package
                if node.module.startswith(f'src.{package}') or node.module.startswith(package):
                    if node.level == 0:  # Absolute import
                        self.findings.append({
                            "type": "could_use_relative",
                            "file": filepath,
                            "import": node.module,
                            "suggestion": "Consider using relative import within same package"
                        })
                        
    def _check_circular_risk(self, tree: ast.AST, filepath: str) -> None:
        """Check for imports that might create circular dependencies."""
        # Skip __init__.py files as they often import from submodules
        if filepath.endswith('__init__.py'):
            return
            
        file_module = filepath.replace('/', '.').replace('.py', '')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                # Check if importing from a parent module
                if file_module.startswith(node.module + '.'):
                    self.findings.append({
                        "type": "circular_risk",
                        "file": filepath,
                        "import": node.module,
                        "description": "Importing from parent module - potential circular dependency risk"
                    })
                    
    def analyze_all_files(self) -> None:
        """Analyze all Python files."""
        python_files = list(self.root_path.rglob("*.py"))
        print(f"Analyzing {len(python_files)} Python files for import patterns...")
        
        for filepath in python_files:
            self.analyze_file(filepath)
            
    def generate_recommendations(self) -> Dict[str, any]:
        """Generate recommendations based on findings."""
        recommendations = {
            "total_findings": len(self.findings),
            "by_type": {},
            "top_recommendations": []
        }
        
        # Count findings by type
        for finding in self.findings:
            finding_type = finding["type"]
            if finding_type not in recommendations["by_type"]:
                recommendations["by_type"][finding_type] = 0
            recommendations["by_type"][finding_type] += 1
            
        # Generate top recommendations
        if "could_use_relative" in recommendations["by_type"]:
            recommendations["top_recommendations"].append(
                "Consider using relative imports within packages for better maintainability"
            )
            
        if "scattered_imports" in recommendations["by_type"]:
            recommendations["top_recommendations"].append(
                "Group all imports at the top of files for better organization"
            )
            
        if "circular_risk" in recommendations["by_type"]:
            recommendations["top_recommendations"].append(
                "Review imports from parent modules to avoid circular dependencies"
            )
            
        return recommendations
        
    def save_report(self) -> None:
        """Save detailed report."""
        report = {
            "findings": self.findings[:50],  # First 50 findings
            "recommendations": self.generate_recommendations()
        }
        
        with open('import_pattern_analysis.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\nAnalysis complete!")
        print(f"Total findings: {len(self.findings)}")
        print("\nFindings by type:")
        for finding_type, count in report["recommendations"]["by_type"].items():
            print(f"  {finding_type}: {count}")
            
        if report["recommendations"]["top_recommendations"]:
            print("\nTop recommendations:")
            for i, rec in enumerate(report["recommendations"]["top_recommendations"], 1):
                print(f"  {i}. {rec}")


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
if __name__ == "__main__":
    analyzer = ImportPatternAnalyzer()
    analyzer.analyze_all_files()
    analyzer.save_report()
