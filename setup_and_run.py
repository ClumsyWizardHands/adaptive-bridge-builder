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

def run_agent():
    """Run the Adaptive Bridge Builder agent."""
    print_header("RUNNING ADAPTIVE BRIDGE BUILDER")
    
    print("Choose how you want to run the agent:")
    print("1. Interactive Bridge Terminal (Recommended for beginners)")
    print("2. Dual-Agent Interactive Terminal")
    print("3. Demonstration Script")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == "1":
        run_interactive_bridge()
    elif choice == "2":
        run_interactive_agents()
    elif choice == "3":
        run_demo_script()
    else:
        print(f"Invalid choice: {choice}. Please enter 1, 2, or 3.")
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
    
    # Run the agent
    run_agent()

if __name__ == "__main__":
    main()
