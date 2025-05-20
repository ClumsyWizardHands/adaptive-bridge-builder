#!/usr/bin/env python3
"""
HTTP Server for Adaptive Bridge Builder Agent

This script creates a Flask-based HTTP server that exposes the Adaptive Bridge Builder
agent through RESTful endpoints, making it accessible to other agents over HTTP.
"""

import json
import logging
import os
import traceback
from flask import Flask, request, jsonify, make_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("BridgeHTTPServer")

# Create Flask app
app = Flask(__name__)

# Global variables
agent = None
PORT = int(os.environ.get("BRIDGE_PORT", 8080))

def initialize_agent():
    """Initialize the agent safely."""
    global agent
    
    try:
        # First try to import and initialize the agent
        from adaptive_bridge_builder import AdaptiveBridgeBuilder
        agent = AdaptiveBridgeBuilder(agent_card_path="agent_card.json")
        logger.info("Adaptive Bridge Builder agent initialized successfully")
        return True
    except ImportError:
        logger.error("Error importing AdaptiveBridgeBuilder - running in mock mode")
        return False
    except Exception as e:
        logger.error(f"Error initializing agent: {e}")
        traceback.print_exc()
        return False

@app.route("/", methods=["GET"])
def home():
    """Root endpoint with basic information."""
    return jsonify({
        "name": "Adaptive Bridge Builder Agent",
        "version": "0.1.0",
        "status": "running" if agent else "mock mode",
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
    try:
        logger.info("Agent card requested")
        
        if agent:
            # Use the actual agent to get the card
            card = agent.get_agent_card()
        else:
            # Provide a mock card in case the agent couldn't be initialized
            card_path = "agent_card.json"
            try:
                with open(card_path, 'r') as f:
                    card = json.load(f)
                logger.info("Using agent card from file as fallback")
            except Exception as e:
                logger.error(f"Error reading agent card file: {e}")
                # Provide a minimal card as last resort
                card = {
                    "agent_id": "bridge-agent-001",
                    "name": "Adaptive Bridge Builder",
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
                logger.info("Using minimal agent card as fallback")
        
        return jsonify(card)
    except Exception as e:
        logger.error(f"Error getting agent card: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

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
        
        if agent:
            # Use the actual agent to process the message
            response = agent.process_message(message)
        else:
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
            logger.info("Providing mock response (agent not initialized)")
        
        # Return the response
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        traceback.print_exc()
        
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
    status = "healthy" if agent else "running in mock mode"
    return jsonify({"status": status, "agent": "available" if agent else "unavailable"})

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found", "message": str(e)}), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error", "message": str(e)}), 500

def run_server():
    """Run the HTTP server."""
    # Initialize the agent
    initialize_agent()
    
    # Update endpoint URL info
    logger.info(f"Agent endpoint URL set to: http://localhost:{PORT}/process")
    
    # Run the server
    logger.info(f"Starting server on port {PORT}")
    # More robust server configuration
    app.run(
        host="0.0.0.0",  # Listen on all interfaces
        port=PORT,
        debug=False,
        use_reloader=False  # Don't use reloader in production
    )

if __name__ == "__main__":
    run_server()
