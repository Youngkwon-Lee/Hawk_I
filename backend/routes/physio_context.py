"""
physio_app context routes.
"""
from flask import Blueprint, g, jsonify, request

from services.physio_context import (
    PhysioContextError,
    load_physio_self_context,
    load_physio_subject_context,
)
from services.supabase_auth import (
    SupabaseClinicianForbidden,
    require_authenticated_person,
    require_clinician,
)


bp = Blueprint("physio_context", __name__, url_prefix="/api/physio")


@bp.route("/self", methods=["GET"])
@require_authenticated_person
def get_self():
    """Return the authenticated person's only selectable timeline subject.

    This intentionally does not query or expose the clinic's subject directory.
    The subsequent timeline read uses the same caller JWT, so Supabase RLS is
    still the authorization boundary for every observation and medication row.
    """
    person = g.authenticated_person
    try:
        context = load_physio_self_context(person)
    except SupabaseClinicianForbidden:
        return jsonify({"success": False, "enabled": True, "error": "self access denied"}), 403
    except PhysioContextError:
        return jsonify({
            "success": False,
            "enabled": True,
            "error": "failed to load self physio_app context",
        }), 502

    return jsonify({
        "success": True,
        "enabled": True,
        "subject": context["subject"],
        "organization": context["organization"],
        "default_created_by_person_id": person.person_id,
        "default_performer_person_id": person.person_id,
        "contract_version": "hawkeye-self/v1",
        "persistence_owner": "self",
    })


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
