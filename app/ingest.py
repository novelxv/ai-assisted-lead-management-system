"""Ingest a website form submission: match it to an existing lead, or create a new one.

This deliberately reuses the deduplication scorer rather than growing a second, subtly
different matcher. A record that dedupe would call the same person is the same person here
too, and when the two disagree there is one place to fix.

**The policy is conservative in both directions.** Attaching a submission to the wrong
person silently corrupts someone's record and is hard to notice; creating a near-duplicate
is visible, reversible, and already surfaced by `/leads/dedupe-candidates`. So an ambiguous
match creates a new lead carrying a pointer to what it might be, and never merges.

Nothing is ever overwritten with less information than it had. Notes are appended, blanks
are filled, and status/owner are never touched by an automated inbound path — those belong
to whoever owns the relationship.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from app import config, normalize as nz, repository
from app.dedupe.scoring import PairScore, score_pair
from app.llm import LLMClient
from app.models import FormSubmission, Lead
from app.source_extraction import SourceExtraction, extract_source, preserve_confident_source

# Name relations that positively support identity, as opposed to merely not contradicting it.
COMPATIBLE_NAME_RELATIONS = frozenset({"exact", "prefix", "initial"})


@dataclass
class IngestOutcome:
    action: Literal["created", "updated"]
    lead: Lead
    matched_by: str | None = None
    score: int | None = None
    confidence: str | None = None
    reasons: list[str] = field(default_factory=list)
    changed_fields: list[str] = field(default_factory=list)
    possible_duplicates: list[tuple[Lead, PairScore]] = field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def submission_to_lead(submission: FormSubmission, lead_id: str = "") -> Lead:
    """Normalize a submission into the same shape as a stored lead, so it can be scored.

    The form supplies one `name` field, so given/family come from the heuristic split. That
    is why a candidate resting on the name alone can never be auto-matched: see
    `normalize.split_full_name` and the band arithmetic in app/config.py.
    """
    given, family = nz.split_full_name(submission.name)
    email = nz.normalize_email(submission.email)
    local, domain = nz.split_email(email)
    digits = nz.normalize_phone(submission.phone)
    company = nz.collapse_ws(submission.company)
    created = nz.parse_date(submission.submitted_at[:10]) if submission.submitted_at else None

    return Lead(
        id=lead_id,
        display_name=submission.name,
        given_name=given,
        family_name=family,
        family_key=nz.name_key(family),
        company=company,
        company_norm=nz.normalize_company(company),
        job_title="",
        email=email,
        email_local=local,
        email_domain=domain,
        phone=nz.collapse_ws(submission.phone),
        phone_digits=digits,
        phone_last9=nz.phone_last9(digits),
        country=nz.normalize_country(submission.country),
        status=None,
        lifecycle_stage="",
        owner="",
        lead_score=None,
        created_at=nz.to_iso(created),
        last_modified_at=None,
        notes="",
        source_channel=None,
        source_detail=None,
        source_confidence=None,
        source_method=None,
        source_needs_review=False,
    )


def gather_candidates(conn: sqlite3.Connection, incoming: Lead) -> list[Lead]:
    """Fetch possible matches using the same blocking keys as the dedupe pass."""
    found: dict[str, Lead] = {}
    for lead in (
        *repository.find_by_email(conn, incoming.email),
        *repository.find_by_phone_last9(conn, incoming.phone_last9),
        *repository.find_by_domain_and_family(conn, incoming.email_domain, incoming.family_key),
        *repository.find_by_family_and_country(conn, incoming.family_key, incoming.country),
    ):
        found[lead.id] = lead
    return list(found.values())


def choose_match(
    incoming: Lead, candidates: list[Lead]
) -> tuple[Lead | None, str | None, PairScore | None, list[tuple[Lead, PairScore]]]:
    """Pick a lead to update, or return None with everything worth flagging.

    Returns (matched_lead, matched_by, score, possible_duplicates).
    """
    scored = sorted(
        ((candidate, score_pair(incoming, candidate)) for candidate in candidates),
        key=lambda item: -item[1].score,
    )
    if not scored:
        return None, None, None, []

    def shares_email(candidate: Lead) -> bool:
        return bool(incoming.email) and incoming.email == candidate.email

    def shares_phone(candidate: Lead) -> bool:
        return bool(incoming.phone_digits) and incoming.phone_digits == candidate.phone_digits

    plausible = [
        (candidate, score)
        for candidate, score in scored
        if score.confidence in {"high", "medium"}
    ]

    # A contact key is the strongest evidence a form can carry, so when the submitted email
    # or phone points at stored leads, only those leads are eligible. Auto-matching some
    # *other* record while quietly ignoring who owns the submitted address would be the
    # worst of both worlds.
    contact_matches = [
        (candidate, score) for candidate, score in scored if shares_email(candidate) or shares_phone(candidate)
    ]
    if contact_matches:
        # Tier 1: the same mailbox. Strong, but not unconditional — a name that explicitly
        # contradicts means a role address, a shared inbox or a typo, and attaching to the
        # wrong person is the more expensive mistake.
        for candidate, score in contact_matches:
            if shares_email(candidate) and not score.has_identity_conflict:
                return candidate, "email", score, []

        # Tier 2: the same handset, but only with positive name support. A shared line is
        # not an identity: reception desks, spouses and team numbers all exist.
        for candidate, score in contact_matches:
            if not shares_phone(candidate):
                continue
            given = nz.given_name_relation(incoming.given_name, candidate.given_name)
            family = nz.family_name_relation(incoming.family_name, candidate.family_name)
            if given in COMPATIBLE_NAME_RELATIONS and family != "conflict":
                return candidate, "phone", score, []

        # Tier 4a: refused despite a contact key. These are always reported, whatever they
        # scored — declining to merge must never mean losing the connection. A low score
        # here is precisely *why* we refused, so filtering on it would hide the evidence.
        seen = {id(candidate) for candidate, _ in contact_matches}
        extra = [item for item in plausible if id(item[0]) not in seen]
        return None, None, None, contact_matches + extra

    # Tier 3: no contact key at all, so only overwhelming combined evidence may match. A
    # candidate resting on the heuristic name split cannot reach `high` by construction
    # (see the band arithmetic in app/config.py), so this can never auto-merge on a guess.
    best_candidate, best_score = scored[0]
    if best_score.confidence == "high":
        return best_candidate, "score", best_score, []

    # Tier 4b: still plausible, so surfaced rather than merged.
    return None, None, None, plausible


# --------------------------------------------------------------------------------------
# Merge policy
# --------------------------------------------------------------------------------------


def _note_line(submission: FormSubmission, divergences: list[str]) -> str:
    """One appended line recording the touchpoint and anything that disagreed.

    Form metadata is recorded here rather than used to classify: in this dataset `form_id`,
    `form_name` and `page_url` routinely contradict each other (a "form_demo_request" named
    "Newsletter Signup" submitted from /blog), so none of them is trustworthy evidence of
    what the lead actually did.
    """
    when = (submission.submitted_at or _now())[:10]
    context = ["web form"]
    if submission.form_name:
        context.append(f'"{submission.form_name}"')
    if submission.page_url:
        context.append(submission.page_url)
    header = f"[{when} · {' · '.join(context)}]"
    parts = [header, submission.message or "(no message)"]
    if divergences:
        parts.append(f"(submitted {'; '.join(divergences)})")
    return " ".join(parts)


def merge_into(
    existing: Lead, incoming: Lead, submission: FormSubmission, source: SourceExtraction
) -> dict[str, Any]:
    """Build the column updates for a matched lead. Never destructive."""
    changes: dict[str, Any] = {}
    divergences: list[str] = []

    # Fill blanks only. A value already on the record was entered by someone with more
    # context than a web form has; a differing value is recorded in the note instead.
    for column, incoming_value in (
        ("company", incoming.company),
        ("country", incoming.country),
        ("phone", incoming.phone),
    ):
        current = getattr(existing, column)
        if incoming_value and not current:
            changes[column] = incoming_value
        elif incoming_value and current and incoming_value.lower() != current.lower():
            divergences.append(f"{column}: {incoming_value}")

    if changes.get("phone"):
        changes["phone_digits"] = incoming.phone_digits
        changes["phone_last9"] = incoming.phone_last9
    if changes.get("company"):
        changes["company_norm"] = incoming.company_norm

    # The match key may be the phone, in which case the submitted address can be new. Keep
    # the stored address and record the other one rather than choosing between them.
    if incoming.email and incoming.email != existing.email:
        divergences.append(f"email: {incoming.email}")

    # Names: fill when missing, and upgrade an initial to the full form. Never swap one
    # complete name for a different one.
    if incoming.display_name and not existing.display_name:
        changes.update(
            {
                "display_name": incoming.display_name,
                "given_name": incoming.given_name,
                "family_name": incoming.family_name,
                "family_key": incoming.family_key,
            }
        )
    elif (
        existing.given_name
        and incoming.given_name
        and nz.is_initial(nz.name_key(existing.given_name))
        and not nz.is_initial(nz.name_key(incoming.given_name))
        and nz.given_name_relation(existing.given_name, incoming.given_name) == "initial"
    ):
        changes.update(
            {"display_name": incoming.display_name, "given_name": incoming.given_name}
        )
    elif incoming.display_name and incoming.display_name != existing.display_name:
        divergences.append(f"name: {incoming.display_name}")

    # Notes are appended, never replaced: the message is the only record of what was said.
    appended = _note_line(submission, divergences)
    changes["notes"] = f"{existing.notes}\n{appended}".strip() if existing.notes else appended

    # Original source is a first-touch fact; only fill it when we never really knew it.
    stored = existing.stored_source()
    chosen = preserve_confident_source(stored, source)
    if chosen is not stored:
        changes.update(
            {
                "source_channel": chosen.channel,
                "source_detail": chosen.detail,
                "source_confidence": chosen.confidence,
                "source_method": chosen.method,
                "source_needs_review": int(chosen.needs_review),
            }
        )

    # Keep the earliest known first-touch date.
    if incoming.created_at and (not existing.created_at or incoming.created_at < existing.created_at):
        changes["created_at"] = incoming.created_at
    if incoming.created_at:
        changes["last_modified_at"] = incoming.created_at

    # status, owner and lifecycle_stage are intentionally absent: an inbound form does not
    # get to move a lead through the sales pipeline or reassign its owner.
    return changes




def _new_lead_params(
    conn: sqlite3.Connection,
    incoming: Lead,
    submission: FormSubmission,
    source: SourceExtraction,
) -> dict[str, Any]:
    now = _now()
    notes = _note_line(submission, [])
    return {
        "id": repository.next_lead_id(conn),
        "display_name": incoming.display_name,
        "given_name": incoming.given_name,
        "family_name": incoming.family_name,
        "family_key": incoming.family_key,
        "company": incoming.company,
        "company_norm": incoming.company_norm,
        "job_title": "",
        "email": incoming.email,
        "email_local": incoming.email_local,
        "email_domain": incoming.email_domain,
        "phone": incoming.phone,
        "phone_digits": incoming.phone_digits,
        "phone_last9": incoming.phone_last9,
        "country": incoming.country,
        # A brand-new inbound lead starts at the beginning of the pipeline.
        "status": "New",
        "lifecycle_stage": "Lead",
        "owner": "",
        "lead_score": None,
        "created_at": incoming.created_at or now[:10],
        "last_modified_at": incoming.created_at,
        "notes": notes,
        "source_channel": source.channel,
        "source_detail": source.detail,
        "source_confidence": source.confidence,
        "source_method": source.method,
        "source_needs_review": int(source.needs_review),
        "raw_record": json.dumps(submission.model_dump(), ensure_ascii=False),
        "created_ts": now,
        "updated_ts": now,
    }


def ingest_submission(
    conn: sqlite3.Connection, submission: FormSubmission, client: LLMClient | None = None
) -> IngestOutcome:
    """Match or create, then apply the update policy."""
    incoming = submission_to_lead(submission)
    source = extract_source(submission.message, client=client)
    candidates = gather_candidates(conn, incoming)
    matched, matched_by, score, duplicates = choose_match(incoming, candidates)

    if matched is not None:
        changes = merge_into(matched, incoming, submission, source)
        updated = repository.update_fields(conn, matched.id, changes)
        assert updated is not None
        return IngestOutcome(
            action="updated",
            lead=updated,
            matched_by=matched_by,
            score=score.score if score else None,
            confidence=score.confidence if score else None,
            reasons=score.reasons if score else [],
            changed_fields=sorted(changes),
        )

    params = _new_lead_params(conn, incoming, submission, source)
    repository.insert_lead(conn, params)
    created = repository.get_lead(conn, params["id"])
    assert created is not None
    return IngestOutcome(
        action="created", lead=created, possible_duplicates=duplicates
    )
