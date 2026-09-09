"""API tests.

Run against the real 2,049-row dataset rather than a toy fixture, so filters, pagination
and export are exercised on the same messy values a reviewer will see.
"""

from __future__ import annotations

import csv
import io

import pytest

from app import config


# --------------------------------------------------------------------------------------
# Listing and filtering
# --------------------------------------------------------------------------------------


def test_list_returns_a_page_and_the_unpaginated_total(client) -> None:
    body = client.get("/leads", params={"limit": 10}).json()
    assert len(body["items"]) == 10
    assert body["total"] == 2049
    assert (body["limit"], body["offset"]) == (10, 0)


def test_status_filter_is_case_insensitive(client) -> None:
    """The export spells one status 34 ways; a caller should not have to guess which."""
    counts = {
        spelling: client.get("/leads", params={"status": spelling}).json()["total"]
        for spelling in ("Closed Won", "closed won", "CLOSED WON", "  closed won  ")
    }
    assert len(set(counts.values())) == 1
    assert next(iter(counts.values())) > 0


def test_unknown_status_is_rejected_with_the_allowed_values(client) -> None:
    response = client.get("/leads", params={"status": "Nurturing"})
    assert response.status_code == 422
    assert "Closed Won" in response.json()["detail"]


def test_owner_filter_matches_despite_stray_whitespace_in_the_source(client) -> None:
    """66 rows store the owner with a trailing space; both must land in one bucket."""
    total = client.get("/leads", params={"owner": "Marcus Wong"}).json()["total"]
    assert total > 0
    assert client.get("/leads", params={"owner": "marcus wong"}).json()["total"] == total


def test_country_filter_matches_despite_casing_in_the_source(client) -> None:
    total = client.get("/leads", params={"country": "United Kingdom"}).json()["total"]
    assert total > 0
    assert client.get("/leads", params={"country": "united kingdom"}).json()["total"] == total


def test_unknown_filter_values_return_an_empty_page_not_an_error(client) -> None:
    body = client.get("/leads", params={"country": "Atlantis"}).json()
    assert body["total"] == 0
    assert body["items"] == []


def test_q_searches_name_company_and_email(client) -> None:
    by_company = client.get("/leads", params={"q": "kim trading", "limit": 5}).json()
    assert by_company["total"] > 0
    assert all("kim trading" in item["company"].lower() for item in by_company["items"])

    by_email = client.get("/leads", params={"q": "@acme.sg", "limit": 5}).json()
    assert by_email["total"] > 0
    assert all("@acme.sg" in item["email"] for item in by_email["items"])

    by_name = client.get("/leads", params={"q": "asante", "limit": 5}).json()
    assert by_name["total"] > 0


def test_filters_combine_conjunctively(client) -> None:
    params = {"status": "Qualified", "country": "India"}
    body = client.get("/leads", params={**params, "limit": 200}).json()
    assert body["total"] <= client.get("/leads", params={"status": "Qualified"}).json()["total"]
    for item in body["items"]:
        assert item["status"] == "Qualified"
        assert item["country"] == "India"


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 500}, {"offset": -1}])
def test_pagination_bounds_are_enforced(client, params: dict) -> None:
    assert client.get("/leads", params=params).status_code == 422


def test_pagination_is_deterministic_and_non_overlapping(client) -> None:
    """Without a stable sort, paging silently repeats and drops rows."""
    first = client.get("/leads", params={"limit": 25, "offset": 0}).json()["items"]
    second = client.get("/leads", params={"limit": 25, "offset": 25}).json()["items"]
    first_ids = [item["id"] for item in first]
    assert not set(first_ids) & {item["id"] for item in second}
    assert first_ids == [item["id"] for item in client.get(
        "/leads", params={"limit": 25, "offset": 0}
    ).json()["items"]]


# --------------------------------------------------------------------------------------
# Detail
# --------------------------------------------------------------------------------------


def test_lead_detail_includes_the_untouched_source_row(client) -> None:
    """Provenance: a reviewer can see exactly what normalization changed."""
    body = client.get("/leads/100234811").json()
    assert body["display_name"] == "Yuki Aina"
    assert body["status"] == "New"
    assert body["raw_record"]["Lead Status"] == "New"
    assert body["raw_record"]["Contact Owner"] == "Marcus Wong "  # stray space preserved
    assert body["owner"] == "Marcus Wong"                          # normalized for filtering


def test_missing_lead_returns_404(client) -> None:
    response = client.get("/leads/does-not-exist")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_export_path_is_not_mistaken_for_a_lead_id(client) -> None:
    assert client.get("/leads/export").status_code == 200


# --------------------------------------------------------------------------------------
# PATCH
# --------------------------------------------------------------------------------------


def test_patch_updates_status_and_normalizes_casing(mutable_client) -> None:
    body = mutable_client.patch("/leads/100234811", json={"status": "closed won"}).json()
    assert body["status"] == "Closed Won"
    assert mutable_client.get("/leads/100234811").json()["status"] == "Closed Won"


def test_patch_updates_owner_and_notes(mutable_client) -> None:
    response = mutable_client.patch(
        "/leads/100234812", json={"owner": "  Grace Osei ", "notes": "Called, left message."}
    )
    assert response.status_code == 200
    assert response.json()["owner"] == "Grace Osei"
    assert response.json()["notes"] == "Called, left message."


@pytest.mark.parametrize(
    "payload",
    [
        {},                                   # nothing to do
        {"status": "Nurturing"},              # outside the taxonomy
        {"email": "new@example.com"},         # not an editable field
        {"status": "New", "id": "999"},       # attempts to move the record
        {"owner": "   "},                     # blank after trimming
    ],
)
def test_invalid_patch_bodies_are_rejected(mutable_client, payload: dict) -> None:
    assert mutable_client.patch("/leads/100234811", json=payload).status_code == 422


def test_patch_on_a_missing_lead_returns_404(mutable_client) -> None:
    assert mutable_client.patch("/leads/nope", json={"status": "New"}).status_code == 404


def test_patch_leaves_unmentioned_fields_alone(mutable_client) -> None:
    before = mutable_client.get("/leads/100234813").json()
    after = mutable_client.patch("/leads/100234813", json={"status": "Closed Lost"}).json()
    for field in ("owner", "notes", "email", "company", "source_channel"):
        assert after[field] == before[field]


def test_editing_notes_does_not_rewrite_a_confident_source(mutable_client) -> None:
    """A lead's original source is a first-touch fact.

    Lead 100234811 was met at a conference booth. Logging a later phone call must not
    reclassify where the lead came from.
    """
    before = mutable_client.get("/leads/100234811").json()
    assert before["source_channel"] == "Event"

    after = mutable_client.patch(
        "/leads/100234811", json={"notes": "Manual - added after inbound phone call."}
    ).json()
    assert after["notes"] == "Manual - added after inbound phone call."
    assert after["source_channel"] == "Event"
    assert after["source_detail"] == before["source_detail"]


def test_editing_notes_fills_in_a_source_that_was_unknown(mutable_client, seeded_conn) -> None:
    """The mirror case: when we never knew the source, a better note should fill it."""
    row = seeded_conn.execute(
        "SELECT id FROM leads WHERE source_needs_review = 1 LIMIT 1"
    ).fetchone()
    lead_id = row["id"]
    assert mutable_client.get(f"/leads/{lead_id}").json()["source_needs_review"] is True

    after = mutable_client.patch(
        f"/leads/{lead_id}",
        json={"notes": "Met her at the Web Summit 2026 booth, scanned our QR code."},
    ).json()
    assert after["source_channel"] == "Event"
    assert after["source_detail"] == "Web Summit 2026 — Booth QR Code"
    assert after["source_needs_review"] is False


# --------------------------------------------------------------------------------------
# CSV export
# --------------------------------------------------------------------------------------


def _parse_csv(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def test_export_returns_csv_with_the_documented_columns(client) -> None:
    response = client.get("/leads/export", params={"status": "New"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]

    rows = _parse_csv(response.text)
    assert list(rows[0].keys()) == list(config.EXPORT_COLUMNS)


def test_export_covers_the_whole_filtered_view_not_just_one_page(client) -> None:
    """Exporting a 300-lead filter must not hand back the default 50."""
    params = {"status": "New"}
    total = client.get("/leads", params=params).json()["total"]
    rows = _parse_csv(client.get("/leads/export", params=params).text)
    assert total > config.DEFAULT_PAGE_SIZE
    assert len(rows) == total


def test_export_respects_every_filter(client) -> None:
    params = {"country": "Singapore", "status": "Qualified"}
    total = client.get("/leads", params=params).json()["total"]
    rows = _parse_csv(client.get("/leads/export", params=params).text)
    assert len(rows) == total
    assert all(r["country"] == "Singapore" and r["status"] == "Qualified" for r in rows)


def test_export_rejects_an_invalid_filter_like_the_list_endpoint(client) -> None:
    assert client.get("/leads/export", params={"status": "Nurturing"}).status_code == 422


def test_export_of_an_empty_view_still_has_a_header(client) -> None:
    text = client.get("/leads/export", params={"country": "Atlantis"}).text
    assert text.strip() == ",".join(config.EXPORT_COLUMNS)


# --------------------------------------------------------------------------------------
# Source extraction endpoint
# --------------------------------------------------------------------------------------


def test_source_extract_endpoint(client) -> None:
    body = client.post(
        "/source/extract",
        json={"text": "He scanned our QR code at the Singapore FinTech Festival 2026 booth."},
    ).json()
    assert body["channel"] == "Event"
    assert body["detail"] == "Singapore FinTech Festival 2026 — Booth QR Code"
    assert body["needs_review"] is False


def test_source_extract_on_empty_text_admits_it_knows_nothing(client) -> None:
    body = client.post("/source/extract", json={"text": ""}).json()
    assert body["channel"] == "Other"
    assert body["detail"] is None
    assert body["needs_review"] is True


def test_source_extract_rejects_unknown_fields(client) -> None:
    assert client.post("/source/extract", json={"txt": "typo"}).status_code == 422


# --------------------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------------------


def test_dashboard_counts_reconcile_with_the_dataset(client) -> None:
    body = client.get("/dashboard").json()
    assert body["total_leads"] == 2049
    assert sum(body["by_status"].values()) == 2049
    assert sum(body["by_source_channel"].values()) == 2049
    assert set(body["by_status"]) <= set(config.LEAD_STATUSES)
    assert set(body["by_source_channel"]) <= set(config.SOURCE_CHANNELS)


def test_dashboard_reports_how_much_source_data_is_uncertain(client) -> None:
    """A channel breakdown that hides its own uncertainty invites false confidence."""
    body = client.get("/dashboard").json()
    assert body["needs_source_review"] > 0


def test_health(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["leads"] == 2049
