"""
Server-side physio_app context lookup for Hawkeye.

This module only runs in the Flask backend. It uses the same server Supabase
credentials as observation persistence and never exposes keys to the frontend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from services.supabase_observations import (
    SupabaseObservationConfig,
    get_supabase_observation_config,
)


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


def _headers(config: SupabaseObservationConfig) -> dict[str, str]:
    return {
        "apikey": config.key,
        "Authorization": f"Bearer {config.key}",
        "Accept": "application/json",
    }


def _get_rest(
    config: SupabaseObservationConfig,
    table: str,
    params: dict[str, str],
) -> list[dict[str, Any]]:
    response = requests.get(
        f"{config.url}/rest/v1/{table}",
        headers=_headers(config),
        params=params,
        timeout=config.timeout_seconds,
    )
    if response.status_code >= 400:
        raise PhysioContextError(f"{table} lookup failed: {response.text[:260]}")
    data = response.json()
    if not isinstance(data, list):
        raise PhysioContextError(f"{table} lookup returned an unexpected payload")
    return [item for item in data if isinstance(item, dict)]


def _in_filter(ids: list[str]) -> str:
    return f"in.({','.join(ids)})"


def load_physio_subject_context(limit: int = 80) -> dict[str, Any]:
    config = get_supabase_observation_config()
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
