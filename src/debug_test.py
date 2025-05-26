"""
Debug Test for Anthropic and Mistral Integration

This diagnostic script will help identify the issues causing errors.
"""

import os
import sys
import traceback
import logging
import importlib

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("debug_test")

def check_module_exists(module_name):
    """Check if a module exists in the Python path."""
    try:
        __import__(module_name)
        logger.info(f"✅ Module '{module_name}' exists")
        return True
    except ImportError as e:
        logger.error(f"❌ Module '{module_name}' not found: {str(e)}")
        return False

def check_file_exists(file_path):
    """Check if a file exists."""
    if os.path.exists(file_path):
        logger.info(f"✅ File exists: {file_path}")
        return True
    else:
        logger.error(f"❌ File not found: {file_path}")
        return False

def test_imports():
    """Test importing all the necessary modules."""
    logger.info("Testing imports...")
    
    # Required standard modules
    required_modules = [
        "aiohttp",
        "asyncio",
        "json",
        "logging",
        "os",
        "pathlib",
        "time",
        "typing"
    ]
    
    # Check standard modules
    missing_standard = []
    for module in required_modules:
        if not check_module_exists(module):
            missing_standard.append(module)
    
    # Our custom modules
    our_modules = [
        "llm_adapter_interface",
        "anthropic_llm_adapter",
        "mistral_llm_adapter",
        "llm_selector",
        "llm_key_manager"
    ]
    
    # Check our custom modules
    missing_custom = []
    for module in our_modules:
        if not check_module_exists(module):
            missing_custom.append(module)
    
    # Check files in src directory
    src_files = [
        "src/llm_adapter_interface.py",
        "src/anthropic_llm_adapter.py",
        "src/mistral_llm_adapter.py",
        "src/llm_selector.py",
        "src/llm_key_manager.py",
        "src/run_real_test.py"
    ]
    
    missing_files = []
    for file in src_files:
        if not check_file_exists(file):
            missing_files.append(file)
    
    return missing_standard, missing_custom, missing_files

def test_module_content(module_name):
    """Test the content of a module by examining its attributes."""
    logger.info(f"\nExamining module: {module_name}")
    try:
        module = importlib.import_module(module_name)
        logger.info(f"✅ Successfully imported {module_name}")
        
        # Get all public attributes
        attributes = [attr for attr in dir(module) if not attr.startswith('_')]
        logger.info(f"Module contains {len(attributes)} public attributes")
        logger.info(f"Attributes: {', '.join(attributes[:10])}{'...' if len(attributes) > 10 else ''}")
        
        # Check for classes
        classes = [attr for attr in attributes if isinstance(getattr(module, attr), type)]
        logger.info(f"Classes: {', '.join(classes)}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error examining {module_name}: {str(e)}")
        traceback.print_exc()
        return False

def try_import_from_directory():
    """Try to import modules directly from current directory."""
    logger.info("\nTrying imports from current directory...")
    
    # Add current directory to path
    if '' not in sys.path:
        sys.path.insert(0, '')
    
    if 'src' not in sys.path:
        sys.path.insert(0, 'src')
    
    logger.info(f"Python path: {sys.path}")
    
    # Try imports again
    modules = [
        "llm_adapter_interface",
        "anthropic_llm_adapter",
        "llm_key_manager"
    ]
    
    for module in modules:
        try:
            importlib.import_module(module)
            logger.info(f"✅ Successfully imported {module} after path modification")
        except ImportError as e:
            logger.error(f"❌ Still can't import {module}: {str(e)}")

def test_create_dummy_classes():
    """Try to create simple instances to test if the code is syntactically valid."""
    logger.info("\nTesting class instantiation...")
    
    try:
        from llm_adapter_interface import LLMAdapterRegistry
        registry = LLMAdapterRegistry()
        logger.info("✅ Created LLMAdapterRegistry instance")
    except Exception as e:
        logger.error(f"❌ Error creating LLMAdapterRegistry: {str(e)}")
        traceback.print_exc()
    
    try:
        from llm_key_manager import LLMKeyManager
        key_manager = LLMKeyManager()
        logger.info("✅ Created LLMKeyManager instance")
    except Exception as e:
        logger.error(f"❌ Error creating LLMKeyManager: {str(e)}")
        traceback.print_exc()

def main():
    """Main function to run all tests."""
    print("=" * 60)
    print("DIAGNOSTIC TEST FOR ANTHROPIC-MISTRAL INTEGRATION")
    print("=" * 60)
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    
    # Check missing dependencies
    missing_standard, missing_custom, missing_files = test_imports()
    
    if missing_standard:
        logger.error(f"\n❌ Missing standard modules: {', '.join(missing_standard)}")
        logger.info("You may need to install them using pip:")
        logger.info(f"pip install {' '.join(missing_standard)}")
    
    if missing_custom:
        logger.error(f"\n❌ Missing custom modules: {', '.join(missing_custom)}")
        logger.info("These should be in your project directory and in the Python path.")
    
    if missing_files:
        logger.error(f"\n❌ Missing files: {', '.join(missing_files)}")
    
    # Try importing from directory
    try_import_from_directory()
    
    # Test module content
    for module in ["llm_adapter_interface", "anthropic_llm_adapter", "llm_key_manager"]:
        try:
            test_module_content(module)
        except Exception as e:
            logger.error(f"Error testing module content for {module}: {str(e)}")
    
    # Test creating instances
    test_create_dummy_classes()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()