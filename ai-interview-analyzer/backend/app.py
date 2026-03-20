"""
AI Interview Analyzer - Main Flask Application
"""

import os
import json
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from api.analyze import analyze_interview
from utils.audio_processor import AudioProcessor
from utils.nlp_processor import NLPProcessor
from utils.scoring_engine import ScoringEngine

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "ogg"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

results_store = {}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "AI Interview Analyzer API is running"})


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Handle audio file upload."""
    if "audio" not in request.files and "text" not in request.form:
        return jsonify({"error": "No audio file or text provided"}), 400

    session_id = str(uuid.uuid4())
    result = {"session_id": session_id}

    # Handle audio upload
    if "audio" in request.files:
        file = request.files["audio"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(f"{session_id}_{file.filename}")
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            result["audio_path"] = filepath
            result["filename"] = file.filename
        else:
            return jsonify({"error": "Invalid file type. Use WAV, MP3, M4A, or OGG"}), 400

    # Handle direct text input
    if "text" in request.form:
        result["transcript"] = request.form["text"]

    if "question" in request.form:
        result["question"] = request.form["question"]

    results_store[session_id] = result
    return jsonify({"session_id": session_id, "status": "uploaded"})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Trigger analysis for a session."""
    data = request.get_json()
    if not data or "session_id" not in data:
        return jsonify({"error": "session_id required"}), 400

    session_id = data["session_id"]
    if session_id not in results_store:
        return jsonify({"error": "Session not found"}), 404

    session_data = results_store[session_id]

    try:
        analysis_result = analyze_interview(session_data)
        results_store[session_id]["analysis"] = analysis_result
        return jsonify({"status": "complete", "session_id": session_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/results/<session_id>", methods=["GET"])
def get_results(session_id):
    """Retrieve analysis results."""
    if session_id not in results_store:
        return jsonify({"error": "Session not found"}), 404

    session = results_store[session_id]
    if "analysis" not in session:
        return jsonify({"status": "pending"}), 202

    return jsonify(session["analysis"])


@app.route("/api/analyze-text", methods=["POST"])
def analyze_text_direct():
    """
    Direct text analysis endpoint (no audio required).
    Accepts JSON: { "text": "...", "question": "..." }
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "text field required"}), 400

    session_data = {
        "transcript": data["text"],
        "question": data.get("question", "Tell me about yourself."),
    }

    try:
        analysis_result = analyze_interview(session_data)
        return jsonify(analysis_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
