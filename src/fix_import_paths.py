#!/usr/bin/env python3
"""
Utility script to fix common import path issues in Python test files.

This script adds missing sys.path modifications to test files to ensure
they can find their required modules regardless of how they're executed.

Usage:
    python fix_import_paths.py [--path PATH] [--dry-run]
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Define imports to add to the beginning of test files
IMPORT_ADDITION = """
import os
import sys

# Add the src directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

"""

def fix_file(file_path: Path, dry_run: bool = False) -> bool:
    """
    Fix import paths in a single test file.
    
    Args:
        file_path: Path to the file to fix
        dry_run: Whether to actually write changes
        
    Returns:
        Whether file was updated
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False
    
    # Check if the file already has sys.path modifications
    if re.search(r'sys\.path\.insert', content) or re.search(r'sys\.path\.append', content):
        # File already has path modifications
        return False
    
    # Find the best insertion point - after imports, before code
    import_block_end = 0
    
    # Look for the last import statement
    import_matches = list(re.finditer(r'^(?:import|from)\s+[\w\.]+', content, re.MULTILINE))
    if import_matches:
        last_import = import_matches[-1]
        # Find the end of the line of the last import
        line_end = content.find('\n', last_import.end())
        if line_end != -1:
            import_block_end = line_end + 1
    
    # If no imports found, insert at the beginning after any doc strings
    if import_block_end == 0:
        # Look for module docstring
        doc_match = re.search(r'^""".*?"""', content, re.DOTALL)
        if doc_match:
            import_block_end = doc_match.end() + 1
    
    # Create modified content
    modified_content = content[:import_block_end] + IMPORT_ADDITION + content[import_block_end:]
    
    if not dry_run:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            return True
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            return False
    return True

def scan_directory(directory: Path, patterns: List[str], dry_run: bool = False) -> Dict:
    """
    Scan a directory for files matching patterns and fix them.
    
    Args:
        directory: Path to directory to scan
        patterns: List of file patterns to match (e.g., 'test_*.py')
        dry_run: Whether to actually write changes
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'files_scanned': 0,
        'files_updated': 0,
        'errors': 0,
        'updated_files': []
    }
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.startswith(p.replace('*', '')) and file.endswith('.py') for p in patterns):
                file_path = Path(root) / file
                stats['files_scanned'] += 1
                
                try:
                    updated = fix_file(file_path, dry_run)
                    if updated:
                        stats['files_updated'] += 1
                        stats['updated_files'].append(str(file_path))
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    stats['errors'] += 1
    
    return stats

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Fix import paths in Python test files")
    parser.add_argument("--path", type=str, default="src",
                      help="Path to search for Python files (default: src)")
    parser.add_argument("--dry-run", action="store_true",
                      help="Don't actually write changes")
    args = parser.parse_args()
    
    directory = Path(args.path)
    if not directory.exists() or not directory.is_dir():
        print(f"Error: {args.path} is not a valid directory")
        return 1
    
    patterns = ["test_*.py", "Test*.py"]
    print(f"Scanning {directory} for test files...")
    stats = scan_directory(directory, patterns, args.dry_run)
    
    print("\nSummary:")
    print(f"Files scanned: {stats['files_scanned']}")
    print(f"Files updated: {stats['files_updated']}")
    print(f"Errors: {stats['errors']}")
    
    if stats['files_updated'] > 0:
        print("\nUpdated files:")
        for file in stats['updated_files']:
            print(f"  {file}")
            
    if args.dry_run:
        print("\nThis was a dry run. No files were actually modified.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
