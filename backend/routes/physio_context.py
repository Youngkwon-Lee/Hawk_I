"""
physio_app context routes.
"""
from flask import Blueprint, g, jsonify, request

from services.physio_context import PhysioContextError, load_physio_subject_context
from services.supabase_auth import require_clinician


bp = Blueprint("physio_context", __name__, url_prefix="/api/physio")


@bp.route("/subjects", methods=["GET"])
@require_clinician
def get_subjects():
    """Return selectable physio_app subjects for Hawkeye analysis storage."""
    try:
        limit = int(request.args.get("limit", 80))
    except ValueError:
        limit = 80
    limit = max(1, min(limit, 200))

    try:
        return jsonify(load_physio_subject_context(
            access_token=g.authenticated_clinician.access_token,
            limit=limit,
        ))
    except PhysioContextError:
        return jsonify({
            "success": False,
            "enabled": True,
            "error": "failed to load physio_app subjects",
        }), 502
