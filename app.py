"""
Flask API Server for the Chatbot
Exposes REST endpoints consumed by the frontend.
"""

import os
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from chatbot import ChatbotEngine

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)  # Allow all origins (fine for a local portfolio project)

# Single shared engine instance — maintains per-server conversation history.
# For a multi-user production app you would store history per session.
engine = ChatbotEngine()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the chat frontend."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    POST /api/chat
    Body:  { "message": "<user text>" }
    Returns: { "response": str, "intent": str, "confidence": float }
    """
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field in request body."}), 400

    user_message: str = data["message"]

    if len(user_message) > 500:
        return jsonify({"error": "Message is too long (max 500 characters)."}), 400

    try:
        result = engine.get_response(user_message)
        return jsonify(result), 200
    except Exception as exc:  # pragma: no cover
        app.logger.error("Engine error: %s", exc)
        return jsonify({
            "error": "Something went wrong on the server. Please try again.",
            "response": "Oops! I had a little hiccup. Give me a moment and try again! 🔧",
        }), 500


@app.route("/api/clear", methods=["POST"])
def clear():
    """POST /api/clear — Wipe the conversation history."""
    engine.clear_history()
    return jsonify({"status": "ok", "message": "Conversation cleared."}), 200


@app.route("/api/history", methods=["GET"])
def history():
    """GET /api/history — Return current conversation history."""
    return jsonify({"history": engine.get_history()}), 200


@app.route("/api/health", methods=["GET"])
def health():
    """GET /api/health — Simple health-check endpoint."""
    return jsonify({"status": "healthy", "bot": "Malati"}), 200


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
