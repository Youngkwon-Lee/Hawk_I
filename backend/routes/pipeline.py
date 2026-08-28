"""Privacy-safe research pipeline status endpoint."""

from flask import Blueprint, jsonify

from services.pipeline_status import get_pipeline_status


bp = Blueprint("pipeline", __name__, url_prefix="/api/pipeline")


@bp.route("/status", methods=["GET"])
def status():
    return jsonify(get_pipeline_status())

