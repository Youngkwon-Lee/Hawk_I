"""Supabase authentication and clinician authorization for protected APIs.

The server validates the caller's access token with Supabase Auth, then uses
that same token for Data API reads. The server-only key is used only as the
``apikey`` header; the caller JWT remains the ``Authorization`` bearer so
physio_app row-level security is applied to every query.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, TypeVar

from flask import g, jsonify, request
import requests

from services.supabase_observations import (
    SupabaseObservationConfig,
    get_supabase_observation_config,
)


CLINICIAN_ROLES = frozenset({"owner", "admin", "provider"})
F = TypeVar("F", bound=Callable[..., Any])


class SupabaseAuthError(RuntimeError):
    """Base class for protected API authentication errors."""


class SupabaseAuthUnavailable(SupabaseAuthError):
    """Raised when auth cannot be checked because configuration/upstream is unavailable."""


class SupabaseInvalidToken(SupabaseAuthError):
    """Raised when the request does not contain a valid Supabase access token."""


class SupabaseClinicianForbidden(SupabaseAuthError):
    """Raised when the authenticated user lacks an active clinician membership."""


@dataclass(frozen=True)
class AuthenticatedClinician:
    user_id: str
    person_id: str
    organization_id: str
    role: str
    access_token: str


@dataclass(frozen=True)
class AuthenticatedPerson:
    user_id: str
    person_id: str
    access_token: str


def extract_bearer_token(authorization_header: str | None) -> str:
    if not authorization_header:
        raise SupabaseInvalidToken("authentication required")

    scheme, separator, token = authorization_header.strip().partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not token.strip():
        raise SupabaseInvalidToken("invalid authorization header")
    token = token.strip()
    if len(token) > 8192:
        raise SupabaseInvalidToken("invalid access token")
    return token


def caller_headers(
    config: SupabaseObservationConfig,
    access_token: str,
) -> dict[str, str]:
    return {
        "apikey": config.key,
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }


def _read_rows(
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
        raise SupabaseAuthUnavailable("authorization service unavailable") from exc

    if response.status_code in {401, 403}:
        raise SupabaseInvalidToken("invalid or expired access token")
    if response.status_code >= 400:
        raise SupabaseAuthUnavailable("authorization lookup failed")

    try:
        payload = response.json()
    except ValueError as exc:
        raise SupabaseAuthUnavailable("authorization lookup returned invalid data") from exc
    if not isinstance(payload, list):
        raise SupabaseAuthUnavailable("authorization lookup returned invalid data")
    return [row for row in payload if isinstance(row, dict)]


def authenticate_person(
    access_token: str,
    config: SupabaseObservationConfig | None = None,
) -> AuthenticatedPerson:
    """Validate a Supabase user token and resolve its active person through RLS."""
    config = config or get_supabase_observation_config()
    if config is None:
        raise SupabaseAuthUnavailable("Supabase authentication is not configured")

    try:
        user_response = requests.get(
            f"{config.url}/auth/v1/user",
            headers=caller_headers(config, access_token),
            timeout=config.timeout_seconds,
        )
    except requests.RequestException as exc:
        raise SupabaseAuthUnavailable("authentication service unavailable") from exc

    if user_response.status_code in {401, 403}:
        raise SupabaseInvalidToken("invalid or expired access token")
    if user_response.status_code >= 400:
        raise SupabaseAuthUnavailable("authentication service unavailable")

    try:
        user = user_response.json()
    except ValueError as exc:
        raise SupabaseAuthUnavailable("authentication service returned invalid data") from exc
    user_id = str(user.get("id") or "") if isinstance(user, dict) else ""
    if not user_id:
        raise SupabaseInvalidToken("invalid user session")

    people = _read_rows(
        config,
        access_token,
        "persons",
        {
            "select": "id",
            "auth_user_id": f"eq.{user_id}",
            "is_active": "eq.true",
            "limit": "1",
        },
    )
    person_id = str(people[0].get("id") or "") if people else ""
    if not person_id:
        raise SupabaseClinicianForbidden("active physio_app person not found")

    return AuthenticatedPerson(
        user_id=user_id,
        person_id=person_id,
        access_token=access_token,
    )


def authenticate_clinician(
    access_token: str,
    config: SupabaseObservationConfig | None = None,
) -> AuthenticatedClinician:
    """Validate a Supabase user token and require an active clinician role."""
    config = config or get_supabase_observation_config()
    if config is None:
        raise SupabaseAuthUnavailable("Supabase authentication is not configured")

    person = authenticate_person(access_token, config=config)
    memberships = _read_rows(
        config,
        access_token,
        "organization_members",
        {
            "select": "organization_id,person_id,role,status",
            "organization_id": f"eq.{config.organization_id}",
            "person_id": f"eq.{person.person_id}",
            "status": "eq.active",
            "deleted_at": "is.null",
            "limit": "1",
        },
    )
    membership = memberships[0] if memberships else {}
    role = str(membership.get("role") or "").lower()
    if role not in CLINICIAN_ROLES:
        raise SupabaseClinicianForbidden("active clinician membership required")

    return AuthenticatedClinician(
        user_id=person.user_id,
        person_id=person.person_id,
        organization_id=config.organization_id,
        role=role,
        access_token=access_token,
    )


def require_clinician(view: F) -> F:
    """Protect a Flask route with Supabase authentication and clinician authz."""

    @wraps(view)
    def wrapped(*args: Any, **kwargs: Any):
        try:
            access_token = extract_bearer_token(request.headers.get("Authorization"))
            g.authenticated_clinician = authenticate_clinician(access_token)
        except SupabaseInvalidToken:
            return jsonify({"success": False, "error": "authentication required"}), 401
        except SupabaseClinicianForbidden:
            return jsonify({"success": False, "error": "clinician access required"}), 403
        except SupabaseAuthUnavailable:
            return jsonify({"success": False, "error": "authentication unavailable"}), 503
        return view(*args, **kwargs)

    return wrapped  # type: ignore[return-value]


def require_authenticated_person(view: F) -> F:
    """Protect a route with Supabase authentication, without a role elevation.

    Use this only for endpoints whose database reads remain constrained by the
    caller JWT and Supabase RLS (for example, a subject reading their own
    timeline). Routes that enumerate or read backend-local analysis files must
    continue to use :func:`require_clinician`.
    """

    @wraps(view)
    def wrapped(*args: Any, **kwargs: Any):
        try:
            access_token = extract_bearer_token(request.headers.get("Authorization"))
            g.authenticated_person = authenticate_person(access_token)
        except SupabaseInvalidToken:
            return jsonify({"success": False, "error": "authentication required"}), 401
        except SupabaseClinicianForbidden:
            return jsonify({"success": False, "error": "active person required"}), 403
        except SupabaseAuthUnavailable:
            return jsonify({"success": False, "error": "authentication unavailable"}), 503
        return view(*args, **kwargs)

    return wrapped  # type: ignore[return-value]
