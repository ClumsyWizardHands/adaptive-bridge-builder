#!/bin/bash

echo "Starting Integration Assistant API..."
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q fastapi uvicorn[standard] websockets pydantic

# Set Python path
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Run the FastAPI app
echo
echo "Starting FastAPI server on http://localhost:8000"
echo "API Documentation available at http://localhost:8000/docs"
echo "WebSocket endpoint: ws://localhost:8000/ws"
echo
echo "Press Ctrl+C to stop the server"
echo

python -m uvicorn src.api.integration_assistant.app:app --reload --host 0.0.0.0 --port 8000
