from flask import Blueprint, request, jsonify
from services.timeline_service import TimelineService

bp = Blueprint('timeline', __name__, url_prefix='/api')
timeline_service = TimelineService()

@bp.route('/timeline/<patient_id>', methods=['GET'])
def get_timeline(patient_id):
    """
    Get medication timeline data for a patient.
    
    Args:
        patient_id: Patient identifier from URL
        
    Returns:
        JSON response with timeline data and recommendations
    """
    try:
        timeline_data = timeline_service.get_patient_timeline(patient_id)
        
        return jsonify({
            "success": True,
            "data": timeline_data
        })
        
    except Exception as e:
        print(f"Timeline error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
