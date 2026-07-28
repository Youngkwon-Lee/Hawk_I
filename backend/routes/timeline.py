from flask import Blueprint, jsonify

bp = Blueprint('timeline', __name__, url_prefix='/api')

@bp.route('/timeline/<patient_id>', methods=['GET'])
def get_timeline(patient_id):
    """Retire the legacy endpoint that returned generated clinical-looking data."""
    return jsonify({
        "success": False,
        "error": "The simulated medication timeline has been removed.",
        "replacement": "/api/history/timeline",
        "requires_authentication": True,
        "patient_id": patient_id,
    }), 410
