"""
physio_app context routes.
"""
from flask import Blueprint, jsonify, request

from services.physio_context import PhysioContextError, load_physio_subject_context


bp = Blueprint("physio_context", __name__, url_prefix="/api/physio")


@bp.route("/subjects", methods=["GET"])
def get_subjects():
    """Return selectable physio_app subjects for Hawkeye analysis storage."""
    try:
        limit = int(request.args.get("limit", 80))
    except ValueError:
        limit = 80
    limit = max(1, min(limit, 200))

    try:
        return jsonify(load_physio_subject_context(limit=limit))
    except PhysioContextError as exc:
        return jsonify({
            "success": False,
            "enabled": True,
            "error": str(exc),
        }), 502
