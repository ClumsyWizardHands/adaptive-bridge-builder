#!/usr/bin/env python3
"""
Setup script to securely store the Anthropic API key.
This avoids having to enter it every time you use the Bridge to Claude connector.
"""

import os
import sys
import json
from pathlib import Path

def setup_api_key():
    """Set up the Anthropic API key for future use."""
    print("=" * 80)
    print("Anthropic Claude API Key Setup")
    print("=" * 80)
    print("\nThis script will securely store your Anthropic API key for use with the")
    print("Adaptive Bridge Builder to Claude connector.\n")
    
    # Check if key is already stored
    config_file = Path(".claude_config.json")
    
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            if "api_key" in config:
                print(f"API key is already stored in {config_file}.")
                replace = input("Do you want to replace it? (y/n): ").lower().strip()
                if replace != "y":
                    print("Keeping existing API key. Setup complete.")
                    return
        except Exception as e:
            print(f"Error reading existing config: {e}")
    
    # Get the API key from the user
    print("\nEnter your Anthropic API key (starts with 'sk-ant-'):")
    print("Note: Your key will be stored locally and not shared.")
    api_key = input("API Key: ").strip()
    
    if not api_key.startswith("sk-ant-"):
        print("Warning: The key you entered doesn't start with 'sk-ant-'.")
        print("This might not be a valid Anthropic API key.")
        proceed = input("Do you want to proceed anyway? (y/n): ").lower().strip()
        if proceed != "y":
            print("Setup cancelled.")
            return
    
    # Save the API key to a config file
    try:
        with open(config_file, "w") as f:
            json.dump({"api_key": api_key}, f)
        os.chmod(config_file, 0o600)  # Make file readable only by the user
        print(f"\nAPI key has been stored in {config_file}.")
        print("This file is readable only by your user account.")
    except Exception as e:
        print(f"Error storing API key: {e}")
        return
    
    print("\nSetup complete!")
    print(f"You can now run the Bridge to Claude connector with:")
    print("python src/bridge_to_claude.py")

if __name__ == "__main__":
    setup_api_key()
