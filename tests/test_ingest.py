"""Ingest tests.

The expensive mistake here is attaching a submission to the wrong person: it corrupts a
record silently and nobody notices. Creating a near-duplicate is visible and reversible.
These tests pin that asymmetry, especially for the cases the real file does not contain —
a shared mailbox, a shared handset, and a match resting only on a guessed surname split.
"""

from __future__ import annotations

import pytest

from app import repository
from app.ingest import ingest_submission, submission_to_lead
from app.models import FormSubmission
from app.repository import LeadFilters


def submission(**overrides) -> FormSubmission:
    payload = {
        "name": "Test Person",
        "email": "test.person@example.com",
        "phone": "+65 8000 1111",
        "company": "Example Ltd",
        "country": "Singapore",
        "message": "Filled out the form on the pricing page.",
        "form_id": "form_contact_general",
        "form_name": "Contact Us",
        "page_url": "/pricing",
        "submitted_at": "2026-06-12T10:00:00Z",
    }
    payload.update(overrides)
    return FormSubmission(**payload)


@pytest.fixture
def conn(mutable_client):
    """A private copy of the seeded database, via the API fixture's connection."""
    from app.main import app, get_conn

    return app.dependency_overrides[get_conn]()


# --------------------------------------------------------------------------------------
# Real submissions
# --------------------------------------------------------------------------------------


def test_a_real_returning_lead_is_updated_not_duplicated(conn, submissions) -> None:
    """49 of the 90 real submissions are existing leads getting back in touch."""
    entry = next(s for s in submissions if s["email"] == "k.toure@liutrading.biz")
    before = repository.get_lead(conn, "100235478")
    outcome = ingest_submission(conn, FormSubmission(**entry))

    assert outcome.action == "updated"
    assert outcome.lead.id == before.id
    assert outcome.matched_by == "email"


def test_an_update_appends_the_message_without_losing_the_original_note(
    conn, submissions
) -> None:
    entry = next(s for s in submissions if s["email"] == "k.toure@liutrading.biz")
    before = repository.get_lead(conn, "100235478")
    outcome = ingest_submission(conn, FormSubmission(**entry))

    assert before.notes in outcome.lead.notes
    assert entry["message"] in outcome.lead.notes
    assert outcome.lead.notes != before.notes


def test_an_update_never_touches_status_or_owner(conn, submissions) -> None:
    """A web form does not get to move a lead through the pipeline or reassign it."""
    entry = next(s for s in submissions if s["email"] == "k.toure@liutrading.biz")
    before = repository.get_lead(conn, "100235478")
    outcome = ingest_submission(conn, FormSubmission(**entry))

    assert outcome.lead.status == before.status
    assert outcome.lead.owner == before.owner
    assert outcome.lead.lifecycle_stage == before.lifecycle_stage
    assert "status" not in outcome.changed_fields
    assert "owner" not in outcome.changed_fields


def test_a_newsletter_signup_does_not_rewrite_where_the_lead_came_from(
    conn, submissions
) -> None:
    """Karim Toure was scanned at a TechCrunch Disrupt booth.

    A later "please send more info" form carries no source signal at all, so the original
    first-touch attribution must survive it.
    """
    entry = next(s for s in submissions if s["email"] == "k.toure@liutrading.biz")
    before = repository.get_lead(conn, "100235478")
    assert before.source_channel == "Event"

    outcome = ingest_submission(conn, FormSubmission(**entry))
    assert outcome.lead.source_channel == "Event"
    assert outcome.lead.source_detail == before.source_detail


def test_a_real_new_person_is_created(conn, submissions) -> None:
    """41 of the 90 are genuinely new people, most at companies already in the file."""
    known = {lead.email for lead in repository.all_leads(conn)}
    entry = next(s for s in submissions if s["email"].lower() not in known)
    before = repository.count_leads(conn, LeadFilters())

    outcome = ingest_submission(conn, FormSubmission(**entry))
    assert outcome.action == "created"
    assert repository.count_leads(conn, LeadFilters()) == before + 1
    assert outcome.lead.status == "New"


def test_ingesting_the_whole_real_file_splits_cleanly(conn, submissions) -> None:
    """End to end over all 90: no false merges, no missed returning leads."""
    before = repository.count_leads(conn, LeadFilters())
    outcomes = [ingest_submission(conn, FormSubmission(**s)) for s in submissions]

    updated = [o for o in outcomes if o.action == "updated"]
    created = [o for o in outcomes if o.action == "created"]
    assert len(updated) == 49
    assert len(created) == 41
    assert repository.count_leads(conn, LeadFilters()) == before + 41
    assert all(o.matched_by == "email" for o in updated)


def test_replaying_a_submission_does_not_create_a_second_lead(conn, submissions) -> None:
    entry = next(s for s in submissions if s["email"] == "k.toure@liutrading.biz")
    ingest_submission(conn, FormSubmission(**entry))
    before = repository.count_leads(conn, LeadFilters())
    outcome = ingest_submission(conn, FormSubmission(**entry))
    assert outcome.action == "updated"
    assert repository.count_leads(conn, LeadFilters()) == before


# --------------------------------------------------------------------------------------
# No destructive overwrites
# --------------------------------------------------------------------------------------


def test_a_blank_incoming_field_never_clears_a_stored_one(conn) -> None:
    before = repository.get_lead(conn, "100234811")
    assert before.company and before.country

    outcome = ingest_submission(
        conn, submission(name=before.display_name, email=before.email, company="", country="")
    )
    assert outcome.action == "updated"
    assert outcome.lead.company == before.company
    assert outcome.lead.country == before.country


def test_a_differing_value_is_recorded_rather_than_overwritten(conn) -> None:
    """Someone entered the stored value with more context than a web form has."""
    before = repository.get_lead(conn, "100234811")
    outcome = ingest_submission(
        conn,
        submission(name=before.display_name, email=before.email, company="Totally Different Ltd"),
    )
    assert outcome.lead.company == before.company
    assert "Totally Different Ltd" in outcome.lead.notes


def test_a_missing_stored_field_is_filled_in(conn) -> None:
    repository.update_fields(conn, "100234811", {"company": "", "company_norm": ""})
    before = repository.get_lead(conn, "100234811")

    outcome = ingest_submission(
        conn, submission(name=before.display_name, email=before.email, company="Singh Logistics Ltd")
    )
    assert outcome.lead.company == "Singh Logistics Ltd"


def test_an_initial_only_name_is_upgraded_to_the_full_form(conn) -> None:
    """'J. Yoon' becoming 'Ji-woo Yoon' is strictly more information, so it is taken."""
    lead = repository.get_lead(conn, "100234956")
    assert lead.display_name == "J. Yoon"

    outcome = ingest_submission(conn, submission(name="Ji-woo Yoon", email=lead.email))
    assert outcome.action == "updated"
    assert outcome.lead.display_name == "Ji-woo Yoon"
    assert outcome.lead.given_name == "Ji-woo"


def test_a_different_full_name_does_not_replace_the_stored_one(conn) -> None:
    lead = repository.get_lead(conn, "100234811")
    outcome = ingest_submission(conn, submission(name="Yuki Aina", email=lead.email))
    assert outcome.lead.display_name == "Yuki Aina"


def test_an_unknown_source_is_filled_in_by_an_informative_message(conn) -> None:
    """The mirror of first-touch preservation: fill it when we never really knew it."""
    lead = next(
        lead for lead in repository.all_leads(conn) if lead.source_needs_review
    )
    outcome = ingest_submission(
        conn,
        submission(
            name=lead.display_name,
            email=lead.email,
            message="Met her at the Web Summit 2026 booth, scanned our QR code.",
        ),
    )
    assert outcome.action == "updated"
    assert outcome.lead.source_channel == "Event"
    assert outcome.lead.source_needs_review is False


# --------------------------------------------------------------------------------------
# Synthetic: identity conflicts the real file does not contain
# --------------------------------------------------------------------------------------


def test_a_shared_mailbox_with_a_conflicting_name_is_not_merged(conn) -> None:
    """Role addresses and shared inboxes are real.

    An exact email match is strong, but not strong enough to attach a submission to
    somebody with an unmistakably different name. The lead is created with a pointer to
    what it might be, so nothing is lost and nothing is corrupted.
    """
    existing = repository.get_lead(conn, "100234811")
    assert existing.display_name == "Yuki Aina"

    outcome = ingest_submission(
        conn, submission(name="Bernard Okonkwo", email=existing.email)
    )
    assert outcome.action == "created"
    assert existing.id in {ref[0].id for ref in outcome.possible_duplicates}
    assert repository.get_lead(conn, existing.id).display_name == "Yuki Aina"


def test_a_shared_phone_line_with_a_conflicting_name_is_not_merged(conn) -> None:
    existing = repository.get_lead(conn, "100234811")
    outcome = ingest_submission(
        conn,
        submission(
            name="Bernard Okonkwo",
            email="bernard.okonkwo@unrelated-domain.example",
            phone=existing.phone,
        ),
    )
    assert outcome.action == "created"


def test_a_match_resting_only_on_a_guessed_surname_split_is_not_auto_merged(conn) -> None:
    """Submissions carry one name field, so both name parts are a heuristic.

    With no email or phone agreement, that guess must not be enough to merge — no matter
    how well the name and company line up.
    """
    existing = repository.get_lead(conn, "100234811")
    outcome = ingest_submission(
        conn,
        submission(
            name=existing.display_name,
            email="yuki.aina@a-different-provider.example",
            phone="+65 9999 8888",
            company=existing.company,
            country=existing.country,
        ),
    )
    assert outcome.action == "created"
    assert any(ref[0].id == existing.id for ref in outcome.possible_duplicates)
    assert outcome.possible_duplicates[0][1].confidence == "medium"


def test_a_genuinely_unrelated_submission_reports_no_possible_duplicates(conn) -> None:
    outcome = ingest_submission(
        conn,
        submission(
            name="Zephyr Qubillington",
            email="zephyr@quillington-industries.example",
            phone="+1 202 555 0199",
            company="Quillington Industries",
            country="United States",
        ),
    )
    assert outcome.action == "created"
    assert outcome.possible_duplicates == []


def test_form_metadata_never_drives_the_source_channel(conn) -> None:
    """form_id, form_name and page_url contradict each other throughout the real file.

    The message text decides the channel; the metadata is recorded in the note as context.
    """
    outcome = ingest_submission(
        conn,
        submission(
            name="Zephyr Quillington",
            email="zephyr@quillington-industries.example",
            form_id="form_demo_request",
            form_name="Newsletter Signup",
            page_url="/blog",
            message="Met at the booth during Retail Asia Expo, said they'd follow up over email.",
        ),
    )
    assert outcome.lead.source_channel == "Event"
    assert "Newsletter Signup" in outcome.lead.notes


# --------------------------------------------------------------------------------------
# Normalization of the incoming payload
# --------------------------------------------------------------------------------------


def test_submission_is_normalized_the_same_way_as_a_stored_lead() -> None:
    lead = submission_to_lead(
        submission(name="  Xiu Ying  Ba ", email="  XY.BA@Acme.COM ", phone="+65 8000-1111")
    )
    assert lead.display_name == "Xiu Ying Ba"
    assert (lead.given_name, lead.family_name) == ("Xiu Ying", "Ba")
    assert lead.email == "xy.ba@acme.com"
    assert lead.phone_digits == "6580001111"


# --------------------------------------------------------------------------------------
# Through the API
# --------------------------------------------------------------------------------------


def test_api_returns_201_for_a_new_lead(mutable_client) -> None:
    response = mutable_client.post(
        "/leads/ingest",
        json={
            "name": "Zephyr Quillington",
            "email": "zephyr@quillington-industries.example",
            "phone": "+1 202 555 0199",
            "company": "Quillington Industries",
            "country": "United States",
            "message": "Filled out the form on the pricing page.",
        },
    )
    assert response.status_code == 201
    assert response.json()["action"] == "created"


def test_api_returns_200_for_an_updated_lead(mutable_client, submissions) -> None:
    entry = next(s for s in submissions if s["email"] == "k.toure@liutrading.biz")
    response = mutable_client.post("/leads/ingest", json=entry)
    assert response.status_code == 200
    body = response.json()
    assert body["action"] == "updated"
    assert body["matched_by"] == "email"
    assert body["reasons"]


@pytest.mark.parametrize(
    "payload",
    [
        {},                                                  # nothing at all
        {"name": "No Email Person"},                         # missing email
        {"name": "", "email": "a@b.com"},                    # blank name
        {"name": "Bad Email", "email": "not-an-email"},      # malformed email
    ],
)
def test_api_rejects_malformed_submissions(mutable_client, payload: dict) -> None:
    assert mutable_client.post("/leads/ingest", json=payload).status_code == 422


def test_api_reports_possible_duplicates_instead_of_merging(mutable_client) -> None:
    existing = mutable_client.get("/leads/100234811").json()
    response = mutable_client.post(
        "/leads/ingest",
        json={"name": "Bernard Okonkwo", "email": existing["email"], "message": "Hello"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["action"] == "created"
    assert body["possible_duplicates"]
    assert body["possible_duplicates"][0]["lead_id"] == "100234811"
    assert body["possible_duplicates"][0]["reasons"]
