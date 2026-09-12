"""Source extraction tests.

The risky parts of this feature are not the obvious notes — they are precedence (a note
naming two surfaces), restraint (a note naming none), and the boundary with the LLM tier.
Those get the most coverage here.
"""

from __future__ import annotations

from typing import Any

import pytest

from app import config
from app.source_extraction import SourceExtraction, extract_source, preserve_confident_source
from tests.conftest import FakeLLM


# --------------------------------------------------------------------------------------
# The seven channels
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "note,channel",
    [
        ("Filled out the form on the contact page.", "Website"),
        ("Scanned the QR code at our Web Summit 2026 booth.", "Event"),
        ("Linkedin dm inbound asking about pricing.", "LinkedIn"),
        ("Found us through organic google search then landed on the pricing page.", "Organic Search"),
        ("Referred by Michael Zhang, warm intro.", "Referral"),
        ("Manual - added after inbound phone call.", "Manual/Sales"),
        ("Other - walked into our office without an appointment.", "Other"),
    ],
)
def test_each_channel_is_recognised(note: str, channel: str) -> None:
    result = extract_source(note)
    assert result.channel == channel
    assert result.method.startswith("rule:")
    assert result.confidence == config.SOURCE_CONFIDENCE_HIGH
    assert not result.needs_review


def test_trailing_sales_chatter_does_not_change_the_channel() -> None:
    """Real notes append a status sentence; it must not leak into classification."""
    result = extract_source(
        "Filled out the form on the pricing page. Very interested, wants pricing call."
    )
    assert result.channel == "Website"
    assert result.detail == "Pricing page — form submission"


# --------------------------------------------------------------------------------------
# Precedence: notes that name more than one surface
# --------------------------------------------------------------------------------------


def test_search_beats_the_page_the_visitor_landed_on() -> None:
    """The originating channel wins; the landing page is preserved as detail."""
    result = extract_source("Googled us and ended up on the book-a-demo page before booking a demo.")
    assert result.channel == "Organic Search"
    assert result.detail == "Google search → Book-a-demo page"


def test_paid_ad_is_not_reported_as_organic_search() -> None:
    """'clicking a google ad' contains 'google' but is the opposite of organic.

    Paid search has no home in the fixed taxonomy, so it maps to Other with the paid fact
    preserved rather than being silently mislabelled.
    """
    result = extract_source("Booked a demo via the book-a-demo page after clicking a google ad.")
    assert result.channel == "Other"
    assert result.detail == "Paid search (Google Ads) → Book-a-demo page"


@pytest.mark.parametrize(
    "note,expected",
    [
        # Google named -> Google may be named.
        ("Booked a demo via the book-a-demo page after clicking a google ad.",
         "Paid search (Google Ads) → Book-a-demo page"),
        ("Came in from an AdWords campaign.", "Paid search (Google Ads)"),
        # A different platform named -> echo that platform, never Google.
        ("Booked a demo after clicking a sponsored ad on Facebook.", "Paid social (Facebook)"),
        ("Clicked a sponsored LinkedIn post and booked a demo.", "Paid social (LinkedIn)"),
        # No platform named -> stay generic rather than pick one.
        ("Came via a PPC campaign then hit the pricing page.", "Paid search"),
        ("Paid search ad, landed on the pricing page.", "Paid search → Pricing page"),
        ("Retargeting ad brought them back to the homepage.", "Paid advertising → Homepage"),
    ],
)
def test_paid_detail_names_only_the_platform_the_note_names(note: str, expected: str) -> None:
    """A paid note says an ad was clicked; it does not always say whose ad.

    Defaulting to Google would invent an attribution — and attribution is precisely the
    field someone later reports on without re-reading the note.
    """
    result = extract_source(note)
    assert result.channel == "Other"
    assert result.detail == expected


@pytest.mark.parametrize(
    "note",
    [
        "Booked a demo after clicking a sponsored ad on Facebook.",
        "Clicked a sponsored LinkedIn post and booked a demo.",
        "Came via a PPC campaign then hit the pricing page.",
        "Retargeting ad brought them back to the homepage.",
    ],
)
def test_google_is_never_attributed_to_a_note_that_does_not_mention_it(note: str) -> None:
    assert "Google" not in (extract_source(note).detail or "")


@pytest.mark.parametrize(
    "note,expected",
    [
        ("Googled us and ended up on the pricing page before booking a demo.",
         "Google search → Pricing page"),
        ("Found us through organic google search then landed on the case study page.",
         "Google search → Case study page"),
        ("Found us through Bing search and landed on the pricing page.",
         "Bing search → Pricing page"),
        ("Found us through organic search and landed on the contact page.",
         "Organic search → Contact page"),
    ],
)
def test_organic_detail_names_only_the_engine_the_note_names(note: str, expected: str) -> None:
    """'Found us through Bing search' must not be reported as a Google search."""
    result = extract_source(note)
    assert result.channel == "Organic Search"
    assert result.detail == expected


def test_event_beats_the_form_it_arrived_through() -> None:
    """A real submission: an event conversation submitted later via the website."""
    result = extract_source(
        "Met at the booth during Retail Asia Expo, said they'd follow up over email."
    )
    assert result.channel == "Event"


def test_referral_beats_everything_else_in_the_sentence() -> None:
    result = extract_source("Referred by Elena Han, warm intro. Connected, sending proposal.")
    assert result.channel == "Referral"
    assert result.detail == "Referred by Elena Han"


# --------------------------------------------------------------------------------------
# Detail precision
# --------------------------------------------------------------------------------------


def test_qr_scan_is_reported_when_it_happened() -> None:
    result = extract_source("He scanned our QR code at the Singapore FinTech Festival 2026 booth.")
    assert result.detail == "Singapore FinTech Festival 2026 — Booth QR Code"


def test_qr_scan_is_not_claimed_when_the_note_denies_it() -> None:
    """'no QR scan logged' must never become 'Booth QR Code'.

    This is the smallest possible fabrication and exactly the kind that destroys trust in
    an extraction pipeline.
    """
    result = extract_source("Spoke with them at our Mobile World Congress booth, no QR scan logged.")
    assert result.channel == "Event"
    assert result.detail == "Mobile World Congress — Booth conversation (no QR scan)"
    assert "QR Code" not in result.detail


def test_booth_conversation_without_any_qr_mention_claims_neither() -> None:
    result = extract_source("Met at the booth during SaaStr Annual, said they'd follow up over email.")
    assert result.detail == "SaaStr Annual — Booth conversation"


def test_linkedin_detail_distinguishes_a_dm_from_a_comment() -> None:
    assert extract_source("Linkedin dm inbound asking about pricing.").detail == "Inbound LinkedIn DM"
    assert (
        extract_source("Connected on LinkedIn after commenting on our post.").detail
        == "Comment on our LinkedIn post"
    )


# --------------------------------------------------------------------------------------
# Restraint: notes with no source signal
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("note", ["", "   ", None])
def test_blank_notes_produce_no_invented_detail(note: str | None) -> None:
    result = extract_source(note)
    assert result.channel == "Other"
    assert result.detail is None
    assert result.needs_review
    assert result.method == "fallback"


def test_zero_signal_message_is_flagged_rather_than_guessed() -> None:
    """49 of the 90 real form submissions carry exactly this message."""
    result = extract_source("Following up after our earlier conversation, please send more info.")
    assert result.channel == "Other"
    assert result.detail is None
    assert result.confidence == config.SOURCE_CONFIDENCE_LOW
    assert result.needs_review


def test_unattributed_social_post_is_not_promoted_to_linkedin() -> None:
    """The note says a post was commented on but never says where.

    'Original Source' happens to say 'Social Media' for some of these rows, but the
    taxonomy has no generic social bucket and the text names no platform, so guessing
    LinkedIn would be invention.
    """
    result = extract_source("Saw our post about replacing hubspot and commented.")
    assert result.channel == "Other"
    assert result.needs_review


# --------------------------------------------------------------------------------------
# The LLM boundary
# --------------------------------------------------------------------------------------


def test_llm_is_not_consulted_when_a_rule_already_decides() -> None:
    """The LLM must stay on the ambiguous slice; calling it for clear notes would be waste."""
    spy = FakeLLM(response={"channel": "Website", "detail": "x", "confident": True})
    extract_source("Scanned the QR code at our Web Summit 2026 booth.", client=spy)
    extract_source("Referred by Michael Zhang, warm intro.", client=spy)
    extract_source("Filled out the form on the pricing page.", client=spy)
    assert spy.calls == []


def test_llm_is_consulted_for_the_ambiguous_slice() -> None:
    spy = FakeLLM(response={"channel": "LinkedIn", "detail": "Commented on our post", "confident": True})
    result = extract_source("Saw our post about replacing hubspot and commented.", client=spy)
    assert len(spy.calls) == 1
    assert result.channel == "LinkedIn"
    assert result.method == "llm"
    assert result.confidence == config.SOURCE_CONFIDENCE_MEDIUM
    assert not result.needs_review


def test_an_unconfident_llm_answer_stays_flagged_for_review() -> None:
    spy = FakeLLM(response={"channel": "Other", "detail": None, "confident": False})
    result = extract_source("Saw our post about replacing hubspot and commented.", client=spy)
    assert result.confidence == config.SOURCE_CONFIDENCE_LOW
    assert result.needs_review


def test_llm_response_outside_the_taxonomy_is_rejected() -> None:
    """The taxonomy is fixed by the spec; a model is not allowed to widen it."""
    spy = FakeLLM(response={"channel": "Social Media", "detail": "LinkedIn post", "confident": True})
    result = extract_source("Saw our post about replacing hubspot and commented.", client=spy)
    assert result.channel in config.SOURCE_CHANNELS
    assert result.channel == "Other"
    assert result.method == "fallback"


@pytest.mark.parametrize("confident", ["false", "true", 1, 0, None, "yes"])
def test_a_non_boolean_confidence_flag_is_rejected(confident: Any) -> None:
    """`bool("false")` is True.

    Coercing the field would turn a model that reported uncertainty into a confident
    answer — the precise opposite of what it said. A field we cannot read means the
    response is not trustworthy, so it is dropped in favour of the documented fallback.
    """
    spy = FakeLLM(response={"channel": "LinkedIn", "detail": "A post", "confident": confident})
    result = extract_source("Saw our post about replacing hubspot and commented.", client=spy)
    assert result.channel == "Other"
    assert result.method == "fallback"
    assert result.needs_review


@pytest.mark.parametrize(
    "response",
    [
        {"channel": "LinkedIn", "detail": {"nested": "object"}, "confident": True},
        {"detail": "no channel at all"},
        {},
    ],
)
def test_malformed_llm_responses_fall_back_safely(response: dict[str, Any]) -> None:
    spy = FakeLLM(response=response)
    result = extract_source("Saw our post about replacing hubspot and commented.", client=spy)
    assert result.channel == "Other"
    assert result.method == "fallback"


def test_extraction_is_fully_deterministic_without_a_client() -> None:
    """A reviewer with no API key still gets a complete, sane answer for every note."""
    note = "Saw our post about replacing hubspot and commented."
    first, second = extract_source(note), extract_source(note)
    assert first == second
    assert first.method == "fallback"


# --------------------------------------------------------------------------------------
# First-touch preservation
# --------------------------------------------------------------------------------------


def _extraction(channel: str, confidence: str, needs_review: bool = False) -> SourceExtraction:
    return SourceExtraction(
        channel=channel,
        detail=None,
        confidence=confidence,
        method="rule:test",
        evidence=None,
        needs_review=needs_review,
    )


def test_a_confident_source_is_never_overwritten() -> None:
    """Someone we met at a conference does not become a website lead by signing up later."""
    existing = _extraction("Event", config.SOURCE_CONFIDENCE_HIGH)
    incoming = _extraction("Website", config.SOURCE_CONFIDENCE_HIGH)
    assert preserve_confident_source(existing, incoming) is existing


def test_an_unknown_source_is_filled_in_by_a_confident_one() -> None:
    existing = _extraction("Other", config.SOURCE_CONFIDENCE_LOW, needs_review=True)
    incoming = _extraction("Event", config.SOURCE_CONFIDENCE_HIGH)
    assert preserve_confident_source(existing, incoming) is incoming


def test_an_unknown_source_is_not_replaced_by_another_unknown() -> None:
    existing = _extraction("Other", config.SOURCE_CONFIDENCE_LOW, needs_review=True)
    incoming = _extraction("Other", config.SOURCE_CONFIDENCE_LOW, needs_review=True)
    assert preserve_confident_source(existing, incoming) is existing


# --------------------------------------------------------------------------------------
# Against the real dataset
# --------------------------------------------------------------------------------------


def test_rules_resolve_the_bulk_of_real_notes_deterministically(seed_rows) -> None:
    """The justification for keeping the LLM narrow is measured, not asserted.

    If this ratio collapses, the rule table has drifted away from the data and the LLM tier
    would silently become the main path.
    """
    results = [extract_source(row["Notes"]) for row in seed_rows]
    by_rule = sum(1 for r in results if r.method.startswith("rule:"))
    assert by_rule / len(results) > 0.90


def test_every_real_note_yields_a_channel_inside_the_taxonomy(seed_rows, submissions) -> None:
    texts = [row["Notes"] for row in seed_rows] + [s["message"] for s in submissions]
    for text in texts:
        assert extract_source(text).channel in config.SOURCE_CHANNELS


def test_the_ambiguous_slice_matches_the_documented_call_volume(seed_rows, submissions) -> None:
    """Pins the numbers the README quotes for LLM call volume.

    The response cache keys on the whole note, so the same sentence with a different
    trailing sales remark is a separate call. Counting distinct *rows* instead of distinct
    *strings* would understate the cost, which is why this is asserted rather than assumed.
    """
    seed = [row["Notes"] for row in seed_rows if extract_source(row["Notes"]).method == "fallback"]
    subs = [s["message"] for s in submissions if extract_source(s["message"]).method == "fallback"]

    assert len(seed) == 91
    assert len({text.strip() for text in seed}) == 13
    assert len({text.strip() for text in seed + subs}) == 14


def test_no_real_note_gets_a_detail_without_evidence(seed_rows, submissions) -> None:
    """Detail must always be traceable to matched text, never produced out of nothing."""
    texts = [row["Notes"] for row in seed_rows] + [s["message"] for s in submissions]
    for text in texts:
        result = extract_source(text)
        if result.detail is not None:
            assert result.evidence is not None
