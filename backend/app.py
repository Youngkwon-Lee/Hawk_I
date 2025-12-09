"""
Flask Backend for HawkEye PD
Movement-based ROI Detection & Video Analysis API
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)

# Enable CORS for Next.js frontend
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000",
            "http://localhost:3001",
            os.getenv("FRONTEND_URL", "http://localhost:3000")
        ]
    },
    r"/uploads/*": {
        "origins": [
            "http://localhost:3000",
            "http://localhost:3001",
            os.getenv("FRONTEND_URL", "http://localhost:3000")
        ]
    },
    r"/files/*": {
        "origins": "*"
    }
})

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max file size
upload_dir = os.getenv('UPLOAD_FOLDER', './uploads')
if not os.path.isabs(upload_dir):
    upload_dir = os.path.abspath(upload_dir)
app.config['UPLOAD_FOLDER'] = upload_dir

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Import progress tracker
from services.progress_tracker import get_progress

# Import routes
from routes import analyze, chat, health, timeline, streaming, population_stats, history, vlm

# Register blueprints
app.register_blueprint(analyze.bp)
app.register_blueprint(chat.bp)
app.register_blueprint(health.bp)
app.register_blueprint(timeline.bp)
app.register_blueprint(streaming.bp)
app.register_blueprint(population_stats.bp)
app.register_blueprint(history.bp)
app.register_blueprint(vlm.bp)

@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        "service": "HawkEye PD Backend",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/health": "Health check",
            "/api/analyze": "Video analysis with ROI detection and task classification",
            "/api/extract-skeleton": "Extract skeleton keypoints using MediaPipe",
            "/api/predict-updrs": "Predict UPDRS score (0-3)",
            "/uploads/<filename>": "Static file serving for videos and results"
        }
    })


@app.route('/files/<path:filename>')
def serve_upload(filename):
    """Serve uploaded files and analysis results"""
    print(f"[FILE] Serving: {filename}")
    try:
        response = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    except Exception as e:
        print(f"[ERROR] Serving file failed: {e}")
        return jsonify({"error": str(e)}), 404

@app.route('/uploads/<path:filename>')
def serve_upload_legacy(filename):
    """Serve uploaded files (legacy route for backward compatibility)"""
    print(f"[FILE-LEGACY] Serving: {filename}")
    try:
        response = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    except Exception as e:
        print(f"[ERROR] Serving file failed: {e}")
        return jsonify({"error": str(e)}), 404


@app.route('/api/analysis/progress/<video_id>', methods=['GET'])
def get_analysis_progress(video_id):
    """Get real-time analysis progress for a video"""
    progress = get_progress(video_id)
    return jsonify(progress)


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        "success": False,
        "error": "File too large. Maximum size is 100MB."
    }), 413

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'production') == 'development'

    print(f"\n{'='*50}")
    print(f"[START] HawkEye PD Backend Starting...")
    print(f"{'='*50}")
    print(f"[PORT] Port: {port}")
    print(f"[DEBUG] Debug: {debug}")
    print(f"[FOLDER] Upload Folder: {app.config['UPLOAD_FOLDER']}")
    print(f"{'='*50}\n")

    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        use_reloader=False  # Disable auto-reload to prevent analysis interruption
    )
