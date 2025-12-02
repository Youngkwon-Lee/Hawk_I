"""
Streaming Analysis Route - WebSocket-based Real-time Analysis

Provides real-time video analysis via WebSocket connection.
Supports frame-by-frame processing with progressive score updates.

WebSocket Events:
- 'start_session': Initialize analysis session
- 'frame': Process single video frame (base64 encoded)
- 'end_session': End session and get final results

Response Events:
- 'session_started': Session initialized
- 'frame_result': Per-frame analysis result
- 'session_result': Final session result
- 'error': Error occurred
"""

from flask import Blueprint, request, jsonify
import base64
import numpy as np
import cv2
from typing import Dict, Optional
import traceback

# Import streaming analyzer
from services.streaming_analyzer import (
    StreamingAnalyzer,
    create_frame_message,
    create_session_message
)

bp = Blueprint('streaming', __name__, url_prefix='/api/streaming')

# Active sessions storage
_sessions: Dict[str, StreamingAnalyzer] = {}


def get_session(session_id: str) -> Optional[StreamingAnalyzer]:
    """Get active session by ID"""
    return _sessions.get(session_id)


def cleanup_session(session_id: str):
    """Clean up and remove session"""
    if session_id in _sessions:
        _sessions[session_id].close()
        del _sessions[session_id]


@bp.route('/start', methods=['POST'])
def start_session():
    """
    Start a new streaming analysis session

    Request JSON:
        {
            "task_type": "finger_tapping",  // or "gait", "hand_movement"
            "fps": 30.0
        }

    Response:
        {
            "success": true,
            "session_id": "abc123"
        }
    """
    try:
        data = request.get_json() or {}
        task_type = data.get('task_type', 'finger_tapping')
        fps = float(data.get('fps', 30.0))

        # Create analyzer
        analyzer = StreamingAnalyzer(task_type=task_type, fps=fps)
        session_id = analyzer.start_session()

        # Store session
        _sessions[session_id] = analyzer

        return jsonify({
            'success': True,
            'session_id': session_id,
            'task_type': task_type,
            'fps': fps
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/frame/<session_id>', methods=['POST'])
def process_frame(session_id: str):
    """
    Process a single video frame

    Request JSON:
        {
            "frame": "<base64 encoded image>"
        }

    Or multipart form:
        frame: image file

    Response:
        {
            "success": true,
            "data": { ... frame result ... }
        }
    """
    try:
        analyzer = get_session(session_id)
        if not analyzer:
            return jsonify({
                'success': False,
                'error': 'Session not found'
            }), 404

        # Get frame data
        frame = None

        if request.is_json:
            data = request.get_json()
            frame_b64 = data.get('frame', '')
            if frame_b64:
                # Decode base64
                frame_bytes = base64.b64decode(frame_b64)
                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        else:
            # Multipart form
            if 'frame' in request.files:
                file = request.files['frame']
                frame_bytes = file.read()
                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({
                'success': False,
                'error': 'No valid frame data'
            }), 400

        # Process frame
        result = analyzer.process_frame(frame)
        message = create_frame_message(result)

        return jsonify({
            'success': True,
            **message
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/end/<session_id>', methods=['POST'])
def end_session(session_id: str):
    """
    End streaming session and get final results

    Response:
        {
            "success": true,
            "data": { ... session result ... }
        }
    """
    try:
        analyzer = get_session(session_id)
        if not analyzer:
            return jsonify({
                'success': False,
                'error': 'Session not found'
            }), 404

        # Get final result
        result = analyzer.end_session()
        message = create_session_message(result)

        # Cleanup
        cleanup_session(session_id)

        return jsonify({
            'success': True,
            **message
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/status/<session_id>', methods=['GET'])
def get_session_status(session_id: str):
    """Get current session status"""
    analyzer = get_session(session_id)
    if not analyzer:
        return jsonify({
            'success': False,
            'error': 'Session not found'
        }), 404

    return jsonify({
        'success': True,
        'session_id': session_id,
        'task_type': analyzer.task_type,
        'frame_count': analyzer.frame_count,
        'processed_count': analyzer.processed_count,
        'current_score': analyzer.last_score,
        'current_confidence': analyzer.last_confidence,
        'events': analyzer.event_counts
    })


@bp.route('/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions"""
    sessions = []
    for sid, analyzer in _sessions.items():
        sessions.append({
            'session_id': sid,
            'task_type': analyzer.task_type,
            'frame_count': analyzer.frame_count
        })

    return jsonify({
        'success': True,
        'sessions': sessions,
        'count': len(sessions)
    })


# Cleanup route for testing
@bp.route('/cleanup', methods=['POST'])
def cleanup_all():
    """Cleanup all sessions (for testing)"""
    count = len(_sessions)
    for sid in list(_sessions.keys()):
        cleanup_session(sid)

    return jsonify({
        'success': True,
        'cleaned': count
    })
