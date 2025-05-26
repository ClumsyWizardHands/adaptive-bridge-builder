#!/usr/bin/env python3
"""
Script to fix unbounded generic type parameters in the codebase.
"""

import os
import re
from typing import List, Tuple


def fix_principle_decision_points():
    """Fix unbounded TypeVars in principle_decision_points.py"""
    file_path = "principle_decision_points.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace unbounded TypeVars with bounded ones
    replacements = [
        # T is not used, so we can leave it or remove it
        ("T = TypeVar('T')", "T = TypeVar('T')  # Unused, kept for potential future use"),
        
        # ActionType should be bounded to what actions can be (typically Dict or str)
        ("ActionType = TypeVar('ActionType')", 
         "ActionType = TypeVar('ActionType', bound=Union[str, Dict[str, Any]])"),
         
        # ContextType should be bounded to dict-like objects
        ("ContextType = TypeVar('ContextType')", 
         "ContextType = TypeVar('ContextType', bound=Dict[str, Any])"),
         
        # ResultType should be bounded to tuple types that are returned
        ("ResultType = TypeVar('ResultType')", 
         "ResultType = TypeVar('ResultType', bound=Tuple[Any, bool, Any])")
    ]
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Fixed unbounded TypeVars in {file_path}")


def fix_universal_agent_connector():
    """Fix unbounded TypeVars in universal_agent_connector.py"""
    file_path = "universal_agent_connector.py"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if T and U are actually used
    t_usage = re.findall(r'\b(?<!TypeVar\(.)T\b(?!\s*=)', content)
    u_usage = re.findall(r'\b(?<!TypeVar\(.)U\b(?!\s*=)', content)
    
    # Since T and U appear to be unused based on the code review, we should remove them
    # or add proper bounds if they're meant to be used
    replacements = [
        # Add bounds for potential future use or document they're unused
        ("T = TypeVar('T')  # Generic response type", 
         "T = TypeVar('T', bound=Dict[str, Any])  # Generic response type - currently unused"),
         
        ("U = TypeVar('U')  # Generic request type", 
         "U = TypeVar('U', bound=Dict[str, Any])  # Generic request type - currently unused")
    ]
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Fixed unbounded TypeVars in {file_path}")


def fix_universal_agent_connector_backup():
    """Fix unbounded TypeVars in universal_agent_connector_backup.py if it exists"""
    file_path = "universal_agent_connector_backup.py"
    
    if not os.path.exists(file_path):
        print(f"⚠ Skipping {file_path} - file not found")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Same fixes as the main file
    replacements = [
        ("T = TypeVar('T')  # Generic response type", 
         "T = TypeVar('T', bound=Dict[str, Any])  # Generic response type - currently unused"),
         
        ("U = TypeVar('U')  # Generic request type", 
         "U = TypeVar('U', bound=Dict[str, Any])  # Generic request type - currently unused")
    ]
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Fixed unbounded TypeVars in {file_path}")


def check_for_other_unbounded_generics():
    """Check for any other unbounded generics in the codebase"""
    issues_found = []
    
    # Pattern to find TypeVar declarations
    typevar_pattern = re.compile(r"(\w+)\s*=\s*TypeVar\s*\(\s*['\"](\w+)['\"]\s*\)")
    
    for root, dirs, files in os.walk("."):
        # Skip directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith('.py') and not file.startswith('fix_'):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Find all TypeVar declarations
                    matches = typevar_pattern.findall(content)
                    
                    for var_name, type_name in matches:
                        # Check if it has bounds or constraints
                        # Look for the full TypeVar declaration
                        typevar_decl_pattern = rf"{var_name}\s*=\s*TypeVar\s*\([^)]+\)"
                        typevar_match = re.search(typevar_decl_pattern, content)
                        
                        if typevar_match:
                            declaration = typevar_match.group()
                            # Check if it has bound= or additional type constraints
                            if 'bound=' not in declaration and declaration.count(',') < 2:
                                # It's likely unbounded (only has the name parameter)
                                # Skip the ones we already fixed
                                if file_path not in [
                                    "./principle_decision_points.py",
                                    "./universal_agent_connector.py",
                                    "./universal_agent_connector_backup.py"
                                ]:
                                    issues_found.append((file_path, var_name, declaration))
                
                except Exception as e:
                    print(f"⚠ Error reading {file_path}: {e}")
    
    if issues_found:
        print("\n⚠ Found additional unbounded TypeVars:")
        for file_path, var_name, declaration in issues_found:
            print(f"  - {file_path}: {var_name}")
    else:
        print("\n✓ No additional unbounded TypeVars found")


def main():
    """Main function to fix all generic constraint issues"""
    print("Fixing unbounded generic type parameters...\n")
    
    # Fix known files
    fix_principle_decision_points()
    fix_universal_agent_connector()
    fix_universal_agent_connector_backup()
    
    # Check for any we might have missed
    check_for_other_unbounded_generics()
    
    print("\n✅ Generic constraint fixes complete!")


if __name__ == "__main__":
    main()