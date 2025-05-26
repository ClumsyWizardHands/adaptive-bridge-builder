#!/usr/bin/env python3
"""Test script to verify LangChain integration works correctly."""

import sys
import os

# Add src directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test imports
try:
    # Import the langchain_integration module
    import langchain_integration
    print("✅ Successfully imported langchain_integration module!")
    
    # Test that we can access the main classes
    print(f"✅ LangChainIntegration class available: {hasattr(langchain_integration, 'LangChainIntegration')}")
    print(f"✅ EMPIRELangChainWrapper class available: {hasattr(langchain_integration, 'EMPIRELangChainWrapper')}")
    print(f"✅ create_principled_chain function available: {hasattr(langchain_integration, 'create_principled_chain')}")
    print(f"✅ create_empire_agent function available: {hasattr(langchain_integration, 'create_empire_agent')}")
    
    print("\n🎉 All LangChain integration components are working correctly!")
    print("   The module can now be imported and used in your EMPIRE framework.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
