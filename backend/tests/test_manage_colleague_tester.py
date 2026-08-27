from __future__ import annotations

from dataclasses import dataclass

import pytest

from scripts.manage_colleague_tester import (
    SupabaseAdminConfig,
    TesterOnboardingError,
    approve_tester,
    deactivate_tester,
    mask_email,
)

ORG_ID = "13d483d9-b785-41d4-8eb8-fc0b55d48c86"
TESTER_ID = "11111111-1111-4111-8111-111111111111"
APPROVER_ID = "22222222-2222-4222-8222-222222222222"
LINK_ID = "33333333-3333-4333-8333-333333333333"


@dataclass
class FakeApi:
    link_status: str | None = None
    approver_role: str = "provider"

    def __post_init__(self):
        self.config = SupabaseAdminConfig("https://example.invalid", "secret", ORG_ID)
        self.writes: list[tuple[str, str, dict]] = []

    def select(self, table, params):
        if table == "organization_members":
            return [{"id": "membership", "role": self.approver_role, "status": "active"}]
        if table == "org_clients":
            if self.link_status is None:
                return []
            return [
                {
                    "id": LINK_ID,
                    "status": self.link_status,
                    "organization_id": ORG_ID,
                    "person_id": TESTER_ID,
                }
            ]
        raise AssertionError(f"unexpected table: {table}")

    def insert(self, table, row):
        self.writes.append(("insert", table, row))
        self.link_status = row["status"]
        return [{"id": LINK_ID, **row}]

    def update(self, table, filters, values):
        self.writes.append(("update", table, {"filters": filters, "values": values}))
        self.link_status = values["status"]
        return [{"id": LINK_ID, **values}]


def person(person_id, user_type="client"):
    return {
        "id": person_id,
        "display_name": "Test person",
        "user_type": user_type,
        "auth_user_id": "44444444-4444-4444-8444-444444444444",
        "is_active": True,
    }


def test_mask_email_does_not_echo_local_part():
    assert mask_email("colleague@example.com") == "c********@example.com"


def test_approve_defaults_to_dry_run_without_write():
    api = FakeApi()
    result = approve_tester(api, person(TESTER_ID), person(APPROVER_ID), apply=False)
    assert result == {"result": "would_create", "mode": "dry_run"}
    assert api.writes == []


def test_approve_creates_and_reads_back_active_link():
    api = FakeApi()
    result = approve_tester(api, person(TESTER_ID), person(APPROVER_ID), apply=True)
    assert result == {"result": "active", "mode": "applied"}
    assert api.link_status == "active"
    assert api.writes[0][0:2] == ("insert", "org_clients")
    assert api.writes[0][2]["created_by"] == APPROVER_ID


def test_approve_is_idempotent_for_active_link():
    api = FakeApi(link_status="active")
    result = approve_tester(api, person(TESTER_ID), person(APPROVER_ID), apply=True)
    assert result == {"result": "already_active", "mode": "read_only"}
    assert api.writes == []


def test_reactivation_requires_explicit_flag():
    api = FakeApi(link_status="inactive")
    with pytest.raises(TesterOnboardingError, match="--reactivate"):
        approve_tester(api, person(TESTER_ID), person(APPROVER_ID), apply=True)
    assert api.writes == []


def test_reactivation_updates_only_after_review_flag():
    api = FakeApi(link_status="inactive")
    result = approve_tester(
        api,
        person(TESTER_ID),
        person(APPROVER_ID),
        apply=True,
        reactivate=True,
    )
    assert result == {"result": "active", "mode": "applied"}
    assert api.writes[0][0:2] == ("update", "org_clients")
    assert api.writes[0][2]["values"] == {"status": "active"}


def test_non_clinical_approver_is_rejected():
    api = FakeApi(approver_role="viewer")
    with pytest.raises(TesterOnboardingError, match="clinical staff"):
        approve_tester(api, person(TESTER_ID), person(APPROVER_ID), apply=False)


def test_professional_tester_is_not_silently_downgraded():
    api = FakeApi()
    with pytest.raises(TesterOnboardingError, match="separate staff approval"):
        approve_tester(
            api,
            person(TESTER_ID, user_type="professional"),
            person(APPROVER_ID),
            apply=False,
        )


def test_deactivate_is_dry_run_then_verified_write():
    api = FakeApi(link_status="active")
    dry_run = deactivate_tester(api, person(TESTER_ID), person(APPROVER_ID), apply=False)
    assert dry_run == {"result": "would_deactivate", "mode": "dry_run"}
    assert api.writes == []

    applied = deactivate_tester(api, person(TESTER_ID), person(APPROVER_ID), apply=True)
    assert applied == {"result": "inactive", "mode": "applied"}
    assert api.link_status == "inactive"
