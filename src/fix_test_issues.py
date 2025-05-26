import semver
"""
Script to fix common test issues in the Alex Familiar project
"""

import os
import re
import subprocess
import sys

def install_missing_dependencies():
    """Install missing Python dependencies."""
    print("Installing missing dependencies...")
    dependencies = ['semver', 'pytest', 'aiohttp', 'websockets']
    
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
            print(f"✅ Installed {dep}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {dep}")

def fix_import_paths():
    """Fix import paths that use 'src.' prefix."""
    print("\nFixing import paths...")
    
    test_files = [f for f in os.listdir('.') if f.startswith('test_') and f.endswith('.py')]
    
    for file in test_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace src. imports
            original_content = content
            content = re.sub(r'from src\.(\w+)', r'from \1', content)
            content = re.sub(r'import src\.(\w+)', r'import \1', content)
            
            if content != original_content:
                with open(file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ Fixed imports in {file}")
        except Exception as e:
            print(f"❌ Error fixing {file}: {e}")

def fix_datetime_deprecations():
    """Fix deprecated datetime.now(timezone.utc) usage."""
    print("\nFixing datetime deprecations...")
    
    all_files = [f for f in os.listdir('.') if f.endswith('.py')]
    
    for file in all_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Add import if needed
            if 'datetime.now(timezone.utc)' in content and 'from datetime import datetime' in content:
                if 'from datetime import timezone' not in content:
                    content = content.replace('from datetime import datetime', 
                                            'from datetime import datetime, timezone')
            
            # Replace utcnow() with now(timezone.utc)
            content = re.sub(r'datetime\.utcnow\(\)', r'datetime.now(timezone.utc)', content)
            
            if content != original_content:
                with open(file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ Fixed datetime in {file}")
        except Exception as e:
            print(f"❌ Error fixing {file}: {e}")

def create_missing_files():
    """Create missing required files."""
    print("\nCreating missing files...")
    
    # Create agent_card.json
    agent_card_content = {
        "id": "alex-familiar-agent",
        "name": "Alex Familiar",
        "version": "1.0.0",
        "description": "An adaptive AI agent with principle-based decision making",
        "capabilities": [
            "principle-based reasoning",
            "emotional intelligence",
            "multi-language support",
            "adaptive communication"
        ],
        "principles": {
            "core": [
                "Adaptability as a Form of Strength",
                "Reality Shapes Relationships",
                "Authenticity Creates Connection"
            ]
        }
    }
    
    try:
        import json
        with open('agent_card.json', 'w') as f:
            json.dump(agent_card_content, f, indent=2)
        print("✅ Created agent_card.json")
    except Exception as e:
        print(f"❌ Error creating agent_card.json: {e}")

def main():
    """Run all fixes."""
    print("🔧 Fixing Test Issues for Alex Familiar Project")
    print("=" * 50)
    
    install_missing_dependencies()
    fix_import_paths()
    fix_datetime_deprecations()
    create_missing_files()
    
    print("\n✅ Fixes completed!")
    print("\nNext steps:")
    print("1. Run 'python run_comprehensive_tests.py' again to see improvements")
    print("2. Manually fix any remaining issues in specific test files")
    print("3. Consider adding __init__.py files to make directories proper Python packages")

if __name__ == "__main__":
    main()