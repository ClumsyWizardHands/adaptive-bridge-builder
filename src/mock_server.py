#!/usr/bin/env python3
"""
Mock HTTP Server for Adaptive Bridge Builder Agent

A standalone HTTP server that doesn't require the AdaptiveBridgeBuilder class,
providing mock responses for testing purposes.
"""

import json
import logging
import os
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MockBridgeServer")

# Create Flask app
app = Flask(__name__)

# Default port
PORT = int(os.environ.get("BRIDGE_PORT", 8080))

# Load the agent card directly from file
try:
    with open("agent_card.json", 'r') as f:
        AGENT_CARD = json.load(f)
    logger.info("Agent card loaded successfully from file")
except Exception as e:
    logger.error(f"Error loading agent card: {e}")
    # Provide a minimal agent card as fallback
    AGENT_CARD = {
        "agent_id": "bridge-agent-001",
        "name": "Adaptive Bridge Builder (Mock Mode)",
        "description": "A2A Protocol bridge agent that connects different agent systems",
        "version": "0.1.0",
        "communication": {
            "endpoints": [
                {
                    "type": "http",
                    "url": f"http://localhost:{PORT}/process"
                }
            ]
        }
    }

@app.route("/", methods=["GET"])
def home():
    """Root endpoint with basic information."""
    return jsonify({
        "name": "Adaptive Bridge Builder Agent (Mock)",
        "version": "0.1.0",
        "status": "running",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "Home page with API information"},
            {"path": "/agent-card", "method": "GET", "description": "Get the agent card"},
            {"path": "/process", "method": "POST", "description": "Process JSON-RPC messages"},
            {"path": "/health", "method": "GET", "description": "Health check endpoint"}
        ]
    })

@app.route("/agent-card", methods=["GET"])
def get_agent_card():
    """Endpoint to get the agent card."""
    return jsonify(AGENT_CARD)

@app.route("/process", methods=["GET", "POST"])
def process_message():
    """Endpoint to process JSON-RPC messages."""
    # Handle GET requests (for browser testing)
    if request.method == "GET":
        return jsonify({
            "status": "ready",
            "message": "This is the JSON-RPC 2.0 endpoint for the Adaptive Bridge Builder agent. Send POST requests with JSON-RPC 2.0 formatted payloads."
        })
    
    # Process POST requests (actual API calls)
    try:
        # Get request data
        if not request.is_json:
            logger.warning("Non-JSON request received")
            return jsonify({"error": "Request must be JSON"}), 400

        message = request.get_json()
        logger.info(f"Processing message: {message.get('id', 'No ID')}")
        
        # Provide a mock response
        response = {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "result": {
                "status": "success",
                "message": "Message received (mock mode)",
                "data": message.get("params")
            }
        }
        logger.info("Returning mock response")
        
        # Return the response
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        
        # Create a proper JSON-RPC 2.0 error response
        error_response = {
            "jsonrpc": "2.0", 
            "id": message.get("id") if 'message' in locals() else None,
            "error": {
                "code": -32603,
                "message": "Internal error",
                "data": str(e)
            }
        }
        return jsonify(error_response), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "agent": "mock mode"})

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found", "message": str(e)}), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error", "message": str(e)}), 500

if __name__ == "__main__":
    # Run the server
    logger.info(f"Starting mock server on port {PORT} - http://localhost:{PORT}")
    app.run(
        host="0.0.0.0",  # Listen on all interfaces
        port=PORT,
        debug=False,
        use_reloader=False  # Don't use reloader in production
    )
