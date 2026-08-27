#!/usr/bin/env python3
"""Approve or deactivate a ParkiCheck colleague tester in the Hawk I workspace.

This is an operator-only tool. It deliberately uses the Supabase service-role
key from the operator environment and never exposes that key to either web app.
The default mode is dry-run; a write requires an explicit ``--apply`` flag.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import requests

ALLOWED_APPROVER_ROLES = {"owner", "admin", "provider", "staff"}
ACTIVE_CLIENT_STATUS = "active"
INACTIVE_CLIENT_STATUS = "inactive"


class TesterOnboardingError(RuntimeError):
    """Raised when a tester cannot be safely approved or deactivated."""


def _require_uuid(value: str, label: str) -> str:
    try:
        return str(UUID(value))
    except (TypeError, ValueError) as exc:
        raise TesterOnboardingError(f"{label} must be a valid UUID") from exc


def mask_email(email: str) -> str:
    """Return a useful but non-identifying representation for operator logs."""
    local, separator, domain = email.strip().partition("@")
    if not separator or not local or not domain:
        return "<invalid-email>"
    return f"{local[:1]}{'*' * max(3, len(local) - 1)}@{domain}"


@dataclass(frozen=True)
class SupabaseAdminConfig:
    url: str
    service_role_key: str
    organization_id: str
    timeout_seconds: float = 15.0


class SupabaseAdminApi:
    """Small, redacted PostgREST client for this one administrative workflow."""

    def __init__(self, config: SupabaseAdminConfig, session: requests.Session | None = None):
        self.config = config
        self.session = session or requests.Session()
        self.headers = {
            "apikey": config.service_role_key,
            "Authorization": f"Bearer {config.service_role_key}",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        table: str,
        *,
        params: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        prefer: str | None = None,
    ) -> list[dict[str, Any]]:
        headers = dict(self.headers)
        if prefer:
            headers["Prefer"] = prefer
        try:
            response = self.session.request(
                method,
                f"{self.config.url.rstrip('/')}/rest/v1/{table}",
                headers=headers,
                params=params,
                json=json,
                timeout=self.config.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise TesterOnboardingError(f"Supabase {table} request unavailable") from exc
        if response.status_code >= 400:
            # Do not echo bodies: PostgREST errors can contain submitted values.
            raise TesterOnboardingError(
                f"Supabase {table} request failed with status {response.status_code}"
            )
        if response.status_code == 204 or not response.content:
            return []
        try:
            payload = response.json()
        except ValueError as exc:
            raise TesterOnboardingError(f"Supabase {table} returned invalid JSON") from exc
        if isinstance(payload, dict):
            return [payload]
        if not isinstance(payload, list):
            raise TesterOnboardingError(f"Supabase {table} returned an unexpected payload")
        return [item for item in payload if isinstance(item, dict)]

    def select(self, table: str, params: dict[str, str]) -> list[dict[str, Any]]:
        return self._request("GET", table, params=params)

    def insert(self, table: str, row: dict[str, Any]) -> list[dict[str, Any]]:
        return self._request("POST", table, json=row, prefer="return=representation")

    def update(
        self, table: str, filters: dict[str, str], values: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return self._request(
            "PATCH", table, params=filters, json=values, prefer="return=representation"
        )


def resolve_person_by_email(api: SupabaseAdminApi, email: str) -> dict[str, Any]:
    normalized = email.strip().lower()
    if not normalized or "@" not in normalized:
        raise TesterOnboardingError("a valid email is required")
    people = api.select(
        "persons",
        {
            "select": "id,display_name,user_type,auth_user_id,is_active",
            "email": f"eq.{normalized}",
            "limit": "2",
        },
    )
    if not people:
        raise TesterOnboardingError(
            "no account-backed person found; ask the colleague to sign up and open ParkiCheck once"
        )
    if len(people) > 1:
        raise TesterOnboardingError("multiple people matched; approve by person UUID instead")
    return people[0]


def load_person_by_id(api: SupabaseAdminApi, person_id: str) -> dict[str, Any]:
    canonical_id = _require_uuid(person_id, "person ID")
    people = api.select(
        "persons",
        {
            "select": "id,display_name,user_type,auth_user_id,is_active",
            "id": f"eq.{canonical_id}",
            "limit": "1",
        },
    )
    if not people:
        raise TesterOnboardingError("person ID was not found")
    return people[0]


def _validate_account_backed_client(person: dict[str, Any]) -> str:
    person_id = _require_uuid(str(person.get("id") or ""), "tester person ID")
    if person.get("is_active") is not True:
        raise TesterOnboardingError("tester person is not active")
    if not person.get("auth_user_id"):
        raise TesterOnboardingError("tester does not have a sign-in account")
    if person.get("user_type") != "client":
        raise TesterOnboardingError(
            "tester is not a client account; clinician access requires a separate staff approval"
        )
    return person_id


def _validate_approver(api: SupabaseAdminApi, organization_id: str, person: dict[str, Any]) -> str:
    person_id = _require_uuid(str(person.get("id") or ""), "approver person ID")
    memberships = api.select(
        "organization_members",
        {
            "select": "id,role,status",
            "organization_id": f"eq.{organization_id}",
            "person_id": f"eq.{person_id}",
            "status": "eq.active",
            "deleted_at": "is.null",
            "limit": "1",
        },
    )
    if not memberships or memberships[0].get("role") not in ALLOWED_APPROVER_ROLES:
        raise TesterOnboardingError("approver is not active Hawk I clinical staff")
    return person_id


def _load_client_link(
    api: SupabaseAdminApi, organization_id: str, tester_person_id: str
) -> dict[str, Any] | None:
    rows = api.select(
        "org_clients",
        {
            "select": "id,status,organization_id,person_id",
            "organization_id": f"eq.{organization_id}",
            "person_id": f"eq.{tester_person_id}",
            "limit": "1",
        },
    )
    return rows[0] if rows else None


def approve_tester(
    api: SupabaseAdminApi,
    tester: dict[str, Any],
    approver: dict[str, Any],
    *,
    apply: bool,
    reactivate: bool = False,
) -> dict[str, str]:
    organization_id = api.config.organization_id
    tester_person_id = _validate_account_backed_client(tester)
    approver_person_id = _validate_approver(api, organization_id, approver)
    existing = _load_client_link(api, organization_id, tester_person_id)

    if existing and existing.get("status") == ACTIVE_CLIENT_STATUS:
        return {"result": "already_active", "mode": "read_only"}
    if existing and not reactivate:
        raise TesterOnboardingError(
            "tester has a non-active clinical link; use --reactivate only after reviewing that status"
        )

    action = "reactivate" if existing else "create"
    if not apply:
        return {"result": f"would_{action}", "mode": "dry_run"}

    if existing:
        api.update(
            "org_clients",
            {"id": f"eq.{existing['id']}"},
            {"status": ACTIVE_CLIENT_STATUS},
        )
    else:
        api.insert(
            "org_clients",
            {
                "organization_id": organization_id,
                "person_id": tester_person_id,
                "status": ACTIVE_CLIENT_STATUS,
                "created_by": approver_person_id,
            },
        )

    verified = _load_client_link(api, organization_id, tester_person_id)
    if not verified or verified.get("status") != ACTIVE_CLIENT_STATUS:
        raise TesterOnboardingError("write completed but active-link readback failed")
    return {"result": "active", "mode": "applied"}


def deactivate_tester(
    api: SupabaseAdminApi,
    tester: dict[str, Any],
    approver: dict[str, Any],
    *,
    apply: bool,
) -> dict[str, str]:
    organization_id = api.config.organization_id
    tester_person_id = _validate_account_backed_client(tester)
    _validate_approver(api, organization_id, approver)
    existing = _load_client_link(api, organization_id, tester_person_id)
    if not existing:
        return {"result": "not_linked", "mode": "read_only"}
    if existing.get("status") != ACTIVE_CLIENT_STATUS:
        return {"result": "already_inactive", "mode": "read_only"}
    if not apply:
        return {"result": "would_deactivate", "mode": "dry_run"}

    api.update(
        "org_clients",
        {"id": f"eq.{existing['id']}"},
        {"status": INACTIVE_CLIENT_STATUS},
    )
    verified = _load_client_link(api, organization_id, tester_person_id)
    if not verified or verified.get("status") != INACTIVE_CLIENT_STATUS:
        raise TesterOnboardingError("write completed but inactive-link readback failed")
    return {"result": "inactive", "mode": "applied"}


def _env_first(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _load_config(args: argparse.Namespace) -> SupabaseAdminConfig:
    url = _env_first("HAWKEYE_SUPABASE_URL", "SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_URL")
    key = _env_first("HAWKEYE_SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_SERVICE_ROLE_KEY")
    organization_id = args.organization_id or _env_first(
        "HAWKEYE_SUPABASE_ORGANIZATION_ID", "HAWKEYE_ORGANIZATION_ID"
    )
    if not url:
        raise TesterOnboardingError("Supabase URL is not configured")
    if not key:
        raise TesterOnboardingError("Supabase service-role key is not configured")
    if not organization_id:
        raise TesterOnboardingError("Hawk I organization ID is not configured")
    return SupabaseAdminConfig(
        url=url,
        service_role_key=key,
        organization_id=_require_uuid(organization_id, "organization ID"),
    )


def _load_cli_person(
    api: SupabaseAdminApi, person_id: str | None, prompt_label: str
) -> tuple[dict[str, Any], str]:
    if person_id:
        return load_person_by_id(api, person_id), f"person:{person_id[:8]}"
    email = getpass.getpass(f"{prompt_label} email (input hidden): ")
    return resolve_person_by_email(api, email), mask_email(email)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely approve or deactivate one ParkiCheck colleague tester."
    )
    parser.add_argument("action", choices=("approve", "deactivate"), help="requested link state")
    parser.add_argument("--organization-id", help="Hawk I workspace UUID")
    parser.add_argument("--tester-person-id", help="use a UUID instead of prompting for email")
    parser.add_argument("--approver-person-id", help="use a UUID instead of prompting for email")
    parser.add_argument(
        "--reactivate",
        action="store_true",
        help="allow approve to reactivate a previously inactive link",
    )
    parser.add_argument(
        "--apply", action="store_true", help="perform the write; omission is a dry-run"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _load_config(args)
        api = SupabaseAdminApi(config)
        tester, tester_label = _load_cli_person(api, args.tester_person_id, "Tester")
        approver, approver_label = _load_cli_person(api, args.approver_person_id, "Approver")
        if args.action == "approve":
            result = approve_tester(
                api, tester, approver, apply=args.apply, reactivate=args.reactivate
            )
        else:
            result = deactivate_tester(api, tester, approver, apply=args.apply)
        print(
            f"tester={tester_label} approver={approver_label} "
            f"mode={result['mode']} result={result['result']}"
        )
        if not args.apply and result["mode"] == "dry_run":
            print("No database rows were changed. Re-run with --apply after review.")
        return 0
    except TesterOnboardingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
