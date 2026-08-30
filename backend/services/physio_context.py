"""
Server-side physio_app context lookup for Hawkeye.

This module only runs in the Flask backend. It uses the same server Supabase
credentials as observation persistence and never exposes keys to the frontend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from services.supabase_auth import (
    AuthenticatedClinician,
    AuthenticatedPerson,
    SupabaseClinicianForbidden,
    authenticate_person,
    caller_headers,
)
from services.supabase_observations import SupabaseObservationConfig, get_supabase_observation_config


class PhysioContextError(RuntimeError):
    """Raised when the backend cannot load physio_app context."""


@dataclass(frozen=True)
class PhysioSubject:
    id: str
    display_name: str
    email: str | None
    user_type: str | None
    source_type: str | None
    role: str | None
    organization_id: str
    is_default: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "email": self.email,
            "user_type": self.user_type,
            "source_type": self.source_type,
            "role": self.role,
            "organization_id": self.organization_id,
            "is_default": self.is_default,
        }


def _get_rest(
    config: SupabaseObservationConfig,
    access_token: str,
    table: str,
    params: dict[str, str],
) -> list[dict[str, Any]]:
    try:
        response = requests.get(
            f"{config.url}/rest/v1/{table}",
            headers=caller_headers(config, access_token),
            params=params,
            timeout=config.timeout_seconds,
        )
    except requests.RequestException as exc:
        raise PhysioContextError(f"{table} lookup unavailable") from exc
    if response.status_code >= 400:
        raise PhysioContextError(f"{table} lookup failed with status {response.status_code}")
    try:
        data = response.json()
    except ValueError as exc:
        raise PhysioContextError(f"{table} lookup returned invalid data") from exc
    if not isinstance(data, list):
        raise PhysioContextError(f"{table} lookup returned an unexpected payload")
    return [item for item in data if isinstance(item, dict)]


def _in_filter(ids: list[str]) -> str:
    return f"in.({','.join(ids)})"


def authorize_physio_subject(
    clinician: AuthenticatedClinician,
    subject_person_id: str,
    activity_session_id: str | None = None,
    config: SupabaseObservationConfig | None = None,
) -> dict[str, Any]:
    """Return server-canonical context after caller-JWT/RLS authorization.

    Browser-supplied organization, author, performer, and display names are
    intentionally ignored. The selected subject and optional existing session
    must be visible to the authenticated clinician through physio_app RLS.
    """
    config = config or get_supabase_observation_config()
    if not config:
        raise PhysioContextError("Supabase context is not configured")
    if clinician.organization_id != config.organization_id:
        raise SupabaseClinicianForbidden("organization access denied")

    clients = _get_rest(
        config,
        clinician.access_token,
        "org_clients",
        {
            "select": "person_id,status",
            "organization_id": f"eq.{config.organization_id}",
            "person_id": f"eq.{subject_person_id}",
            "status": "eq.active",
            "limit": "1",
        },
    )
    if not clients:
        raise SupabaseClinicianForbidden("subject access denied")

    people = _get_rest(
        config,
        clinician.access_token,
        "persons",
        {
            "select": "id,display_name,email",
            "id": f"eq.{subject_person_id}",
            "is_active": "eq.true",
            "limit": "1",
        },
    )
    if not people:
        raise SupabaseClinicianForbidden("subject access denied")
    person = people[0]

    organizations = _get_rest(
        config,
        clinician.access_token,
        "organizations",
        {
            "select": "id,name,display_name",
            "id": f"eq.{config.organization_id}",
            "limit": "1",
        },
    )
    if not organizations:
        raise SupabaseClinicianForbidden("organization access denied")
    organization = organizations[0]

    if activity_session_id:
        sessions = _get_rest(
            config,
            clinician.access_token,
            config.activity_sessions_table,
            {
                "select": "id",
                "id": f"eq.{activity_session_id}",
                "organization_id": f"eq.{config.organization_id}",
                "subject_person_id": f"eq.{subject_person_id}",
                "limit": "1",
            },
        )
        if not sessions:
            raise SupabaseClinicianForbidden("activity session access denied")

    subject_display_name = person.get("display_name") or person.get("email")
    organization_display_name = organization.get("display_name") or organization.get("name")
    context = {
        "subject_person_id": subject_person_id,
        "organization_id": config.organization_id,
        "created_by_person_id": clinician.person_id,
        "performer_person_id": config.performer_person_id,
    }
    if subject_display_name:
        context["subject_display_name"] = str(subject_display_name)
    if organization_display_name:
        context["organization_display_name"] = str(organization_display_name)
    if activity_session_id:
        context["activity_session_id"] = activity_session_id
    return context


def authorize_parkicheck_session(
    access_token: str,
    activity_session_id: str,
    subject_person_id: str | None = None,
    organization_id: str | None = None,
    config: SupabaseObservationConfig | None = None,
) -> dict[str, Any]:
    """Authorize one ParkiCheck session with the caller JWT and database RLS.

    Unlike Hawk I's clinic workflow, ParkiCheck users can belong to personal
    organizations. The activity-session row is therefore the canonical source
    for subject and organization instead of the backend's configured clinic.
    """
    config = config or get_supabase_observation_config()
    if not config:
        raise PhysioContextError("Supabase context is not configured")

    caller = authenticate_person(access_token, config=config)
    params = {
        "select": "id,subject_person_id,organization_id,created_by",
        "id": f"eq.{activity_session_id}",
        "limit": "1",
    }
    if subject_person_id:
        params["subject_person_id"] = f"eq.{subject_person_id}"
    if organization_id:
        params["organization_id"] = f"eq.{organization_id}"

    sessions = _get_rest(
        config,
        caller.access_token,
        config.activity_sessions_table,
        params,
    )
    if not sessions:
        raise SupabaseClinicianForbidden("activity session access denied")

    session = sessions[0]
    canonical_subject_id = str(session.get("subject_person_id") or "").strip()
    canonical_organization_id = str(session.get("organization_id") or "").strip()
    if not canonical_subject_id or not canonical_organization_id:
        raise PhysioContextError("activity session context is incomplete")

    return {
        "subject_person_id": canonical_subject_id,
        "organization_id": canonical_organization_id,
        "created_by_person_id": caller.person_id,
        "performer_person_id": canonical_subject_id,
        "activity_session_id": activity_session_id,
    }


def load_physio_self_context(
    person: AuthenticatedPerson,
    config: SupabaseObservationConfig | None = None,
) -> dict[str, Any]:
    """Resolve the caller's own subject and organization through caller RLS."""
    config = config or get_supabase_observation_config()
    if not config:
        raise PhysioContextError("Supabase context is not configured")

    sessions = _get_rest(
        config,
        person.access_token,
        config.activity_sessions_table,
        {
            "select": "organization_id,subject_person_id,performed_at,created_at",
            "subject_person_id": f"eq.{person.person_id}",
            "order": "performed_at.desc.nullslast,created_at.desc",
            "limit": "1",
        },
    )
    organization_id = str(sessions[0].get("organization_id") or "").strip() if sessions else ""

    if not organization_id:
        memberships = _get_rest(
            config,
            person.access_token,
            "organization_members",
            {
                "select": "organization_id,person_id,role,status",
                "person_id": f"eq.{person.person_id}",
                "status": "eq.active",
                "deleted_at": "is.null",
                "limit": "2",
            },
        )
        organization_ids = list(dict.fromkeys(
            str(row.get("organization_id") or "").strip()
            for row in memberships
            if str(row.get("organization_id") or "").strip()
        ))
        if len(organization_ids) != 1:
            raise PhysioContextError("self organization context is unavailable or ambiguous")
        organization_id = organization_ids[0]

    people = _get_rest(
        config,
        person.access_token,
        "persons",
        {
            "select": "id,display_name,user_type,source_type",
            "id": f"eq.{person.person_id}",
            "is_active": "eq.true",
            "limit": "1",
        },
    )
    if not people:
        raise SupabaseClinicianForbidden("active self subject not found")

    organizations = _get_rest(
        config,
        person.access_token,
        "organizations",
        {
            "select": "id,name,display_name,slug,org_type,status",
            "id": f"eq.{organization_id}",
            "status": "eq.active",
            "limit": "1",
        },
    )
    if not organizations:
        raise SupabaseClinicianForbidden("active self organization not found")

    person_row = people[0]
    organization = organizations[0]
    subject_display_name = str(person_row.get("display_name") or "내 기록")
    organization_display_name = str(
        organization.get("display_name") or organization.get("name") or "개인 기록"
    )
    return {
        "subject_person_id": person.person_id,
        "organization_id": organization_id,
        "created_by_person_id": person.person_id,
        "performer_person_id": person.person_id,
        "subject_display_name": subject_display_name,
        "organization_display_name": organization_display_name,
        "subject": {
            "id": person.person_id,
            "display_name": subject_display_name,
            "email": None,
            "user_type": person_row.get("user_type"),
            "source_type": person_row.get("source_type") or "self",
            "role": "self",
            "organization_id": organization_id,
            "is_default": True,
        },
        "organization": organization,
    }


def authorize_physio_self(
    person: AuthenticatedPerson,
    subject_person_id: str,
    organization_id: str,
    config: SupabaseObservationConfig | None = None,
) -> dict[str, Any]:
    """Authorize one self analysis and return server-canonical context."""
    if subject_person_id != person.person_id:
        raise SupabaseClinicianForbidden("self subject access denied")

    context = load_physio_self_context(person, config=config)
    if context["organization_id"] != organization_id:
        raise SupabaseClinicianForbidden("self organization access denied")

    return {
        key: context[key]
        for key in (
            "subject_person_id",
            "organization_id",
            "created_by_person_id",
            "performer_person_id",
            "subject_display_name",
            "organization_display_name",
        )
    }


def load_physio_subject_context(
    access_token: str,
    limit: int = 80,
    config: SupabaseObservationConfig | None = None,
) -> dict[str, Any]:
    config = config or get_supabase_observation_config()
    if not config:
        return {
            "success": True,
            "enabled": False,
            "organization": None,
            "subjects": [],
            "default_subject_id": None,
            "reason": "missing Supabase URL/key or organization/operator context",
        }

    organizations = _get_rest(
        config,
        access_token,
        "organizations",
        {
            "select": "id,name,display_name,slug,org_type,status",
            "id": f"eq.{config.organization_id}",
            "limit": "1",
        },
    )
    organization = organizations[0] if organizations else {
        "id": config.organization_id,
        "display_name": "physio_app organization",
    }

    clients = _get_rest(
        config,
        access_token,
        "org_clients",
        {
            "select": "person_id,status,intake_date,created_at",
            "organization_id": f"eq.{config.organization_id}",
            "status": "eq.active",
            "limit": str(limit),
            "order": "intake_date.desc.nullslast,created_at.desc",
        },
    )

    person_ids: list[str] = []
    role_by_person: dict[str, str | None] = {}
    for client in clients:
        person_id = str(client.get("person_id") or "")
        if not person_id or person_id in role_by_person:
            continue
        person_ids.append(person_id)
        role_by_person[person_id] = "client"

    if not person_ids:
        return {
            "success": True,
            "enabled": True,
            "organization": organization,
            "subjects": [],
            "default_subject_id": None,
            "default_created_by_person_id": config.created_by,
            "default_performer_person_id": config.performer_person_id,
            "reason": "no active physio_app clients found for this organization",
        }

    people = _get_rest(
        config,
        access_token,
        "persons",
        {
            "select": "id,display_name,email,user_type,source_type",
            "id": _in_filter(person_ids),
            "limit": str(limit),
        },
    )
    people_by_id = {str(person.get("id")): person for person in people if person.get("id")}

    subjects: list[PhysioSubject] = []
    default_subject_id = (
        config.subject_person_id
        if config.subject_person_id and config.subject_person_id in role_by_person
        else person_ids[0]
    )

    for person_id in person_ids:
        person = people_by_id.get(person_id) or {"id": person_id}
        display_name = (
            person.get("display_name")
            or person.get("email")
            or f"physio_app person {person_id[:8]}"
        )
        subjects.append(
            PhysioSubject(
                id=person_id,
                display_name=str(display_name),
                email=person.get("email"),
                user_type=person.get("user_type"),
                source_type=person.get("source_type"),
                role=role_by_person.get(person_id),
                organization_id=config.organization_id,
                is_default=person_id == default_subject_id,
            )
        )

    return {
        "success": True,
        "enabled": True,
        "organization": organization,
        "subjects": [subject.as_dict() for subject in subjects],
        "default_subject_id": default_subject_id,
        "default_created_by_person_id": config.created_by,
        "default_performer_person_id": config.performer_person_id,
    }
