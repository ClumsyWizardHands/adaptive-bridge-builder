#!/usr/bin/env python3
"""
Fix timezone.utc compatibility issues across the codebase.
timezone.utc was introduced in Python 3.11, so we need to use timezone.utc for compatibility.
"""

import os
import re
from pathlib import Path


def fix_datetime_utc_in_file(filepath):
    """Fix timezone.utc references in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Check if datetime is imported
        if 'from datetime import' in content or 'import datetime' in content:
            # Add timezone import if not present
            if 'timezone' not in content:
                # Find the datetime import line
                import_pattern = r'(from datetime import[^;\n]+)'
                match = re.search(import_pattern, content)
                if match:
                    import_line = match.group(1)
                    if 'timezone' not in import_line:
                        new_import = import_line + ', timezone'
                        content = content.replace(import_line, new_import)
                else:
                    # Handle "import datetime" case
                    content = content.replace('import datetime', 'import datetime\nfrom datetime import timezone')
            
            # Replace timezone.utc with timezone.utc
            content = content.replace('timezone.utc', 'timezone.utc')
            
            # Also handle cases where timezone.utc might be used directly
            content = re.sub(r'\b(?<!datetime\.)timezone.utc\b', 'timezone.utc', content)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def find_python_files(directory):
    """Find all Python files in the directory."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip virtual environments and cache directories
        dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git', 'env']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files


def main():
    """Main function to fix timezone.utc issues."""
    src_dir = Path(__file__).parent
    
    print("🔧 Fixing timezone.utc compatibility issues...")
    print(f"Scanning directory: {src_dir}")
    
    python_files = find_python_files(src_dir)
    print(f"Found {len(python_files)} Python files")
    
    fixed_count = 0
    for filepath in python_files:
        if fix_datetime_utc_in_file(filepath):
            print(f"✅ Fixed: {os.path.relpath(filepath, src_dir)}")
            fixed_count += 1
    
    print(f"\n✨ Fixed {fixed_count} files with timezone.utc issues")


if __name__ == "__main__":
    main()
