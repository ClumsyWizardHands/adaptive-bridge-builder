#!/usr/bin/env python3
"""
Setup and Run Script for Adaptive Bridge Builder

This script automates the setup process and runs the Adaptive Bridge Builder
on your local console with minimal effort.
"""

import os
import sys
import subprocess
import platform
import shutil
import json
import re

def print_header(title):
    """Print a formatted header."""
    width = 80
    print(f"\n{'=' * 10} {title} {'=' * (width - 12 - len(title))}")

def check_python_version():
    """Check if Python version is 3.9 or higher."""
    print_header("CHECKING PYTHON VERSION")
    
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 9):
        print(f"Python 3.9 or higher is required, but you have {major}.{minor}")
        print("Please install a newer version of Python and try again.")
        sys.exit(1)
    
    print(f"Python version {major}.{minor} detected. ✓")
    return True

def setup_virtual_environment():
    """Set up a virtual environment."""
    print_header("SETTING UP VIRTUAL ENVIRONMENT")
    
    venv_dir = "venv"
    
    # Check if virtual environment already exists
    if os.path.exists(venv_dir):
        print(f"Virtual environment '{venv_dir}' already exists.")
        activate_venv()
        return True
    
    # Create virtual environment
    try:
        print(f"Creating virtual environment '{venv_dir}'...")
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        print(f"Virtual environment created successfully. ✓")
        activate_venv()
        return True
    except subprocess.CalledProcessError:
        print("Failed to create virtual environment.")
        sys.exit(1)

def activate_venv():
    """Activate the virtual environment."""
    venv_dir = "venv"
    
    if platform.system() == "Windows":
        activate_script = os.path.join(venv_dir, "Scripts", "activate")
    else:
        activate_script = os.path.join(venv_dir, "bin", "activate")
    
    print(f"To activate the virtual environment, run:")
    if platform.system() == "Windows":
        print(f"    {venv_dir}\\Scripts\\activate")
    else:
        print(f"    source {venv_dir}/bin/activate")
    
    # Modify PATH to include the virtual environment's bin directory
    if platform.system() == "Windows":
        bin_dir = os.path.join(os.path.abspath(venv_dir), "Scripts")
    else:
        bin_dir = os.path.join(os.path.abspath(venv_dir), "bin")
    
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
    
    # Update sys.path to include the virtual environment's packages
    sys.path.insert(0, bin_dir)
    
    # Update sys executable to use the one from the virtual environment
    if platform.system() == "Windows":
        sys.executable = os.path.join(bin_dir, "python.exe")
    else:
        sys.executable = os.path.join(bin_dir, "python")

def install_dependencies():
    """Install required dependencies."""
    print_header("INSTALLING DEPENDENCIES")
    
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print(f"Error: '{requirements_file}' not found.")
        sys.exit(1)
    
    try:
        print(f"Installing dependencies from '{requirements_file}'...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file], check=True)
        print("Dependencies installed successfully. ✓")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install dependencies.")
        sys.exit(1)

def verify_agent_files():
    """Verify that all required agent files exist."""
    print_header("VERIFYING AGENT FILES")
    
    required_files = [
        "src/adaptive_bridge_builder.py",
        "src/agent_card.json",
        "src/interactive_bridge.py",
        "src/interactive_agents.py",
        "src/demo_agent_system.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("Error: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        sys.exit(1)
    
    print("All required agent files are present. ✓")
    return True

def setup_empire_framework():
    """Set up Empire Framework directories and schemas."""
    print_header("SETTING UP EMPIRE FRAMEWORK")
    
    # Define required directories
    directories = [
        "resources/empire_framework_schemas",
        "resources/principles"
    ]
    
    # Create directories if they don't exist
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Creating directory '{directory}'...")
            os.makedirs(directory, exist_ok=True)
    
    # Check if schema files exist
    schema_file = "resources/empire_framework_schemas/principle_schema.json"
    if not os.path.exists(schema_file):
        print(f"Warning: Schema file '{schema_file}' is missing.")
        print("Please create it manually or download it from the repository.")
    else:
        print(f"Schema file '{schema_file}' is present. ✓")
    
    # Check if any principle files exist
    principle_files = os.listdir("resources/principles") if os.path.exists("resources/principles") else []
    principle_files = [f for f in principle_files if f.endswith(".json")]
    
    if not principle_files:
        print("No principle files found in 'resources/principles'.")
        print("You may need to create or download them.")
    else:
        print(f"Found {len(principle_files)} principle files. ✓")
    
    return True

def setup_pre_commit_hooks():
    """Set up pre-commit hooks for Empire Framework compatibility."""
    print_header("SETTING UP PRE-COMMIT HOOKS")
    
    pre_commit_config = ".pre-commit-config.yaml"
    if not os.path.exists(pre_commit_config):
        print(f"Warning: '{pre_commit_config}' not found.")
        print("Pre-commit hooks for Empire Framework will not be installed.")
        return False
    
    try:
        print("Installing pre-commit...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pre-commit"], check=True)
        
        print("Installing pre-commit hooks...")
        subprocess.run(["pre-commit", "install"], check=True)
        
        print("Pre-commit hooks installed successfully. ✓")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to set up pre-commit hooks: {e}")
        return False
    except FileNotFoundError:
        print("pre-commit command not found. Skipping hook installation.")
        return False

def validate_empire_schemas():
    """Validate Empire Framework schemas if requested."""
    print_header("VALIDATING EMPIRE FRAMEWORK SCHEMAS")
    
    try:
        import jsonschema
    except ImportError:
        print("jsonschema module not found. Skipping schema validation.")
        return False
    
    schema_dir = "resources/empire_framework_schemas"
    principle_dir = "resources/principles"
    
    if not os.path.exists(schema_dir) or not os.path.exists(principle_dir):
        print("Schema or principle directories not found. Skipping validation.")
        return False
    
    schema_files = [f for f in os.listdir(schema_dir) if f.endswith(".json")]
    principle_files = [f for f in os.listdir(principle_dir) if f.endswith(".json")]
    
    if not schema_files:
        print("No schema files found. Skipping validation.")
        return False
    
    if not principle_files:
        print("No principle files found. Skipping validation.")
        return False
    
    # Validate schema files themselves
    print("Validating schema files...")
    meta_schema = {"$schema": "http://json-schema.org/draft-07/schema#"}
    
    try:
        for schema_file in schema_files:
            file_path = os.path.join(schema_dir, schema_file)
            with open(file_path, 'r') as f:
                schema = json.load(f)
            
            jsonschema.validate(schema, meta_schema)
            print(f"  '{schema_file}' is valid. ✓")
        
        # Validate principle files against their schema
        principle_schema_path = os.path.join(schema_dir, "principle_schema.json")
        if os.path.exists(principle_schema_path):
            with open(principle_schema_path, 'r') as f:
                principle_schema = json.load(f)
            
            print("\nValidating principle files...")
            for principle_file in principle_files:
                file_path = os.path.join(principle_dir, principle_file)
                with open(file_path, 'r') as f:
                    principle = json.load(f)
                
                jsonschema.validate(principle, principle_schema)
                print(f"  '{principle_file}' is valid. ✓")
        
        print("\nAll schemas and principle files are valid. ✓")
        return True
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return False
    except jsonschema.exceptions.ValidationError as e:
        print(f"Schema validation error: {e}")
        return False

def validate_memory_bank():
    """Validate Memory Bank markdown files."""
    print_header("VALIDATING MEMORY BANK")
    
    memory_bank_dir = "memory-bank"
    if not os.path.exists(memory_bank_dir):
        print("Memory Bank directory not found. Skipping validation.")
        return False
    
    print("Checking Memory Bank files...")
    
    # Get all markdown files in the memory-bank directory and subdirectories
    markdown_files = []
    for root, dirs, files in os.walk(memory_bank_dir):
        for file in files:
            if file.endswith(".md"):
                markdown_files.append(os.path.join(root, file))
    
    if not markdown_files:
        print("No Markdown files found in Memory Bank. Skipping validation.")
        return False
    
    # Basic validation: check each file has at least one heading
    valid_files = 0
    invalid_files = []
    
    for file_path in markdown_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for headings (lines starting with # followed by space)
            headings = re.findall(r'^#+ .+$', content, re.MULTILINE)
            
            if headings:
                valid_files += 1
            else:
                invalid_files.append(os.path.relpath(file_path, memory_bank_dir))
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            invalid_files.append(os.path.relpath(file_path, memory_bank_dir))
    
    if invalid_files:
        print("The following files do not have proper headings:")
        for file in invalid_files:
            print(f"  - {file}")
        print(f"\n{valid_files} of {len(markdown_files)} files are valid.")
    else:
        print(f"All {valid_files} Memory Bank files are valid. ✓")
    
    return len(invalid_files) == 0

def run_agent():
    """Run the Adaptive Bridge Builder agent."""
    print_header("RUNNING ADAPTIVE BRIDGE BUILDER")
    
    print("Choose how you want to run the agent:")
    print("1. Interactive Bridge Terminal (Recommended for beginners)")
    print("2. Dual-Agent Interactive Terminal")
    print("3. Demonstration Script")
    print("4. Empire Framework Validation")
    
    choice = input("\nEnter your choice (1-4): ")
    
    if choice == "1":
        run_interactive_bridge()
    elif choice == "2":
        run_interactive_agents()
    elif choice == "3":
        run_demo_script()
    elif choice == "4":
        validate_empire_integration()
    else:
        print(f"Invalid choice: {choice}. Please enter 1, 2, 3, or 4.")
        run_agent()

def run_interactive_bridge():
    """Run the interactive bridge terminal."""
    print_header("STARTING INTERACTIVE BRIDGE TERMINAL")
    
    print("""
This terminal allows you to interact directly with the Bridge agent.
Available commands:
- card - View the Bridge agent's card
- send <message> - Send a message to the Bridge agent
- route <destination> <message> - Route a message through the Bridge
- translate <source> <target> <message> - Translate between protocols
- new - Start a new conversation
- exit - Exit the terminal
""")
    
    input("Press Enter to continue...")
    
    try:
        os.chdir("src")
        subprocess.run([sys.executable, "interactive_bridge.py"])
    except Exception as e:
        print(f"Error running interactive bridge: {e}")
        sys.exit(1)

def run_interactive_agents():
    """Run the dual-agent interactive terminal."""
    print_header("STARTING DUAL-AGENT INTERACTIVE TERMINAL")
    
    print("""
This terminal simulates communication between the Bridge agent and an External agent.
Available commands:
- bridge_to_external <message> - Send from Bridge to External agent
- external_to_bridge <message> - Send from External to Bridge agent
- get_bridge_card - Get Bridge agent's card
- get_external_card - Get External agent's card
- new_conversation - Start a new conversation
- exit - Exit the terminal
""")
    
    input("Press Enter to continue...")
    
    try:
        os.chdir("src")
        subprocess.run([sys.executable, "interactive_agents.py"])
    except Exception as e:
        print(f"Error running interactive agents: {e}")
        sys.exit(1)

def run_demo_script():
    """Run the demonstration script."""
    print_header("RUNNING DEMONSTRATION SCRIPT")
    
    print("""
This script demonstrates the core capabilities of the Adaptive Bridge Builder:
1. Agent Card retrieval
2. Echo message processing
3. Message routing capability
4. Protocol translation capability
""")
    
    input("Press Enter to continue...")
    
    try:
        os.chdir("src")
        subprocess.run([sys.executable, "demo_agent_system.py"])
    except Exception as e:
        print(f"Error running demo script: {e}")
        sys.exit(1)

def validate_empire_integration():
    """Validate Empire Framework integration."""
    print_header("VALIDATING EMPIRE FRAMEWORK INTEGRATION")
    
    print("""
This option validates your Empire Framework integration:
1. Schema validation
2. Principle file validation
3. Memory Bank structure validation
4. Directory structure verification
""")
    
    input("Press Enter to continue...")
    
    # Run validations
    setup_empire_framework()
    validate_empire_schemas()
    validate_memory_bank()
    
    print("\nValidation complete. See above for any issues found.")
    input("\nPress Enter to return to the main menu...")
    run_agent()

def main():
    """Main function to set up and run the agent."""
    print_header("ADAPTIVE BRIDGE BUILDER SETUP")
    print("This script will set up and run the Adaptive Bridge Builder on your local console.")
    
    # Check Python version
    check_python_version()
    
    # Set up virtual environment
    setup_virtual_environment()
    
    # Install dependencies
    install_dependencies()
    
    # Verify agent files
    verify_agent_files()
    
    # Set up Empire Framework
    setup_empire_framework()
    
    # Set up pre-commit hooks
    setup_pre_commit_hooks()
    
    # Run the agent
    run_agent()

if __name__ == "__main__":
    main()
