#!/usr/bin/env python3
"""
Utility script to replace deprecated datetime.now(timezone.utc) with datetime.now(timezone.utc).

This script scans Python files in the specified directory and replaces
deprecated datetime.now(timezone.utc) calls with the recommended alternative.

Usage:
    python update_deprecated_datetime.py [--path PATH] [--dry-run]
"""

import argparse
import datetime
from datetime import timezone
import os
import re
import sys
from pathlib import Path

# Define the patterns to search and replace
PATTERNS = [
    # datetime.now(timezone.utc) pattern
    (
        r'datetime\.utcnow\(\)',
        r'datetime.now(timezone.utc)'
    ),
    # datetime.datetime.now(timezone.utc) pattern
    (
        r'datetime\.datetime\.utcnow\(\)',
        r'datetime.datetime.now(timezone.utc)'
    ),
]

def ensure_datetime_utc_import(content: str) -> str:
    """
    Ensure timezone.utc is imported properly.
    """
    # Check if the file imports datetime
    if not re.search(r'import\s+datetime', content):
        # Add import at the top
        return "import datetime\nfrom datetime import timezone\n" + content
    
    return content

def update_file(file_path: Path, dry_run: bool = False) -> tuple[bool, int]:
    """
    Update a single file by replacing deprecated datetime patterns.
    
    Args:
        file_path: Path to the file to update
        dry_run: Whether to actually write changes
        
    Returns:
        Tuple of (file_was_updated, number_of_replacements)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False, 0
    
    original_content = content
    replacements = 0
    
    # Apply each search/replace pattern
    for pattern, replacement in PATTERNS:
        new_content, count = re.subn(pattern, replacement, content)
        if count > 0:
            content = new_content
            replacements += count
    
    # If anything was replaced, ensure imports are present
    if replacements > 0:
        content = ensure_datetime_utc_import(content)
        
        if not dry_run:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, replacements
            except Exception as e:
                print(f"Error writing {file_path}: {e}")
                return False, replacements
        return True, replacements
    return False, 0

def scan_directory(directory: Path, dry_run: bool = False) -> dict:
    """
    Scan a directory for Python files and update them.
    
    Args:
        directory: Path to directory to scan
        dry_run: Whether to actually write changes
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'files_scanned': 0,
        'files_updated': 0,
        'replacements': 0,
        'errors': 0,
        'updated_files': []
    }
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                stats['files_scanned'] += 1
                
                try:
                    updated, count = update_file(file_path, dry_run)
                    if updated:
                        stats['files_updated'] += 1
                        stats['replacements'] += count
                        stats['updated_files'].append(str(file_path))
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    stats['errors'] += 1
    
    return stats

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Replace deprecated datetime.now(timezone.utc) with datetime.now(timezone.utc)")
    parser.add_argument("--path", type=str, default="src",
                      help="Path to search for Python files (default: src)")
    parser.add_argument("--dry-run", action="store_true",
                      help="Don't actually write changes")
    args = parser.parse_args()
    
    # Check Python version - timezone.utc constant was added in Python 3.11
    if not hasattr(datetime, 'timezone.utc'):
        print("Warning: timezone.utc is not available in your Python version.")
        print("This script is meant for Python 3.11+ or with python-dateutil installed.")
        print("Continuing with import from dateutil.tz if available...")
        
    directory = Path(args.path)
    if not directory.exists() or not directory.is_dir():
        print(f"Error: {args.path} is not a valid directory")
        return 1
    
    print(f"Scanning {directory}...")
    stats = scan_directory(directory, args.dry_run)
    
    print("\nSummary:")
    print(f"Files scanned: {stats['files_scanned']}")
    print(f"Files updated: {stats['files_updated']}")
    print(f"Total replacements: {stats['replacements']}")
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
