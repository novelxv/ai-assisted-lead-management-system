"""Extract a structured lead source from free-text notes.

Three tiers, in order:

1. **Rules** — an ordered pattern table. Handles every phrasing where the originating
   channel is stated outright. On the seed file this resolves 1,958 of 2,049 notes (95.6%).
2. **LLM** — only for text the rules flag as genuinely ambiguous, or that they cannot read
   at all. On the seed file that is one distinct note text.
3. **Fallback** — `Other`, no detail, `needs_review=True`. Never invents anything.

Why this split rather than an LLM over every note: the overwhelming majority of these notes
state their channel in plain words, and for those a rule is better in every dimension that
matters — free, instant, reproducible, and explainable to the sales team who will challenge
a classification. The LLM is reserved for the cases where a rule would have to guess.

**Precedence is the design.** Notes routinely name two surfaces: "Googled us and ended up on
the book-a-demo page" is Organic Search, not Website, and "clicking a google ad" is paid, not
organic. The ordering below encodes one principle — the ORIGINATING channel wins over the
surface the person eventually landed on — and the landing surface is preserved in `detail`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from app import config, llm as llm_module
from app.llm import LLMClient

MAX_DETAIL_LENGTH = 200


@dataclass(frozen=True)
class SourceExtraction:
    """A channel plus everything a reviewer needs to judge whether to trust it."""

    channel: str
    detail: str | None
    confidence: str
    method: str
    evidence: str | None
    needs_review: bool


# --------------------------------------------------------------------------------------
# Detail helpers
# --------------------------------------------------------------------------------------

# Page references. Ordered longest-context-first so "blog post on lead scoring" is not
# truncated to "blog post".
_PAGE_PATTERNS = (
    re.compile(r"\b(?:on|via|to)\s+(?:the\s+)?(blog post on [\w\- ]+?)(?=[.,]|\s+(?:before|after|then)\b|$)", re.I),
    re.compile(r"\b(?:on|via|to)\s+(?:the\s+)?([\w\-]+(?:\s+[\w\-]+){0,2})\s+page\b", re.I),
    re.compile(r"\b(?:on|via|to)\s+(?:the\s+)?(homepage)\b", re.I),
)

_EVENT_PATTERNS = (
    re.compile(r"\bat\s+(?:our|the)\s+(.+?)\s+booth\b", re.I),
    re.compile(r"\bbooth\s+(?:at|during)\s+(?:the\s+)?([^,.]+)", re.I),
    re.compile(r"\bat\s+(?:the\s+)?([^,.]+?)\s+(?:expo|summit|conference|congress|festival)\b", re.I),
)

_REFERRER_PATTERN = re.compile(
    r"\b(?:referred by|introduced (?:to us )?by|referral from)\s+([^,.;]+)", re.I
)


def _clean_fragment(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .,-")


def _extract_page(text: str) -> str | None:
    """Name the page or content a note mentions, e.g. 'Pricing page', 'Homepage'."""
    for pattern in _PAGE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        raw = _clean_fragment(match.group(1))
        if not raw:
            continue
        label = raw[:1].upper() + raw[1:]
        if label.lower() == "homepage" or label.lower().startswith("blog post"):
            return label
        return f"{label} page"
    return None


def _extract_event(text: str) -> str | None:
    """Name the event a note refers to, without inventing one."""
    for pattern in _EVENT_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        raw = _clean_fragment(match.group(1))
        # Guard against swallowing a whole clause when the sentence is shaped unusually.
        if raw and len(raw.split()) <= 6:
            return raw
    return None


def _with_page(base: str, text: str) -> str:
    page = _extract_page(text)
    return f"{base} → {page}" if page else base


def _event_detail(text: str) -> str:
    """Describe an event touchpoint, distinguishing a scan from a conversation.

    'Spoke with them at our Web Summit 2026 booth, no QR scan logged' must not be reported
    as a QR scan. Claiming a scan that did not happen is exactly the kind of small
    fabrication that erodes trust in an extraction pipeline.
    """
    lowered = text.lower()
    event = _extract_event(text)
    has_qr = "qr" in lowered
    denies_qr = bool(re.search(r"no qr\b|without (?:a |an )?qr|qr[^.]{0,20}not\b", lowered))

    if denies_qr:
        descriptor = "Booth conversation (no QR scan)"
    elif has_qr:
        descriptor = "Booth QR Code"
    else:
        descriptor = "Booth conversation"
    return f"{event} — {descriptor}" if event else descriptor


def _linkedin_detail(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\bdm\b|direct message|inmail", lowered):
        return "Inbound LinkedIn DM"
    if "comment" in lowered:
        return "Comment on our LinkedIn post"
    if "connect" in lowered:
        return "LinkedIn connection"
    return "LinkedIn"


def _manual_detail(text: str) -> str:
    lowered = text.lower()
    if "cold" in lowered:
        return "Added by sales from a cold outreach list"
    if "phone" in lowered or "call" in lowered:
        return "Added by sales after an inbound phone call"
    return "Added manually by sales"


def _other_explicit_detail(text: str) -> str:
    lowered = text.lower()
    if "walked into" in lowered or "walk-in" in lowered:
        return "Walk-in at our office"
    if "info@" in lowered or "general inbox" in lowered:
        return "General info@ inbox"
    # "Other - <something>": quote what the note actually said rather than paraphrasing.
    match = re.search(r"\bother\s*[-–—:]\s*([^.]+)", text, re.I)
    if match:
        return _clean_fragment(match.group(1))[:MAX_DETAIL_LENGTH].capitalize()
    return "Other"


# --------------------------------------------------------------------------------------
# Rule table
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    channel: str | None  # None => route to the LLM tier
    detail: Callable[[str], str | None] | None = None


# Order matters; see the module docstring. Patterns key on portable vocabulary
# ("booth", "expo", "referred by") rather than on this dataset's sentence templates, so a
# reworded note still resolves.
RULES: tuple[Rule, ...] = (
    Rule(
        # First, because a referral names a person and is never about the surface they used.
        name="referral",
        pattern=re.compile(r"\breferred by\b|\bwarm intro\b|\bintroduced (?:to us )?by\b|\breferral from\b", re.I),
        channel="Referral",
        detail=lambda t: (
            f"Referred by {_clean_fragment(m.group(1))}"
            if (m := _REFERRER_PATTERN.search(t))
            else "Warm introduction"
        ),
    ),
    Rule(
        # Before the web rules: an event lead who later filled in a form is an event lead.
        name="event",
        pattern=re.compile(
            r"\bbooth\b|\bexpo\b|\bsummit\b|\bconference\b|\bcongress\b|\btrade\s?show\b|"
            r"\bfestival\b|\bmeetup\b|\bwe met (?:them|him|her) at\b",
            re.I,
        ),
        channel="Event",
        detail=_event_detail,
    ),
    Rule(
        # Must precede organic search: "clicking a google ad" contains "google".
        # Paid search has no home in the fixed taxonomy, so it maps to Other with the fact
        # preserved in the detail rather than being mislabelled as Organic Search.
        name="paid_ad",
        pattern=re.compile(
            r"\bgoogle ads?\b|\badwords\b|\bpaid (?:ad|ads|search)\b|\bppc\b|\bsponsored\b|"
            r"\bretarget\w*\b|clicking (?:an?|our|the)\s+\w*\s*ad\b",
            re.I,
        ),
        channel="Other",
        detail=lambda t: _with_page("Paid search (Google Ads)", t),
    ),
    Rule(
        # Must precede website: these notes name a landing page too, but search came first.
        name="organic_search",
        pattern=re.compile(
            r"\borganic\b[^.]{0,20}\bsearch\b|\bgoogled us\b|\bgoogle search\b|"
            r"\bfound us\b[^.]{0,30}\b(?:google|search)\b|\bsearch engine\b|\bseo\b",
            re.I,
        ),
        channel="Organic Search",
        detail=lambda t: _with_page("Google search", t),
    ),
    Rule(
        name="linkedin",
        pattern=re.compile(r"\blinked-?in\b", re.I),
        channel="LinkedIn",
        detail=_linkedin_detail,
    ),
    Rule(
        name="manual_sales",
        pattern=re.compile(
            r"\bmanual(?:ly)?\b|\badded by sales\b|\bcold (?:outreach|call|list)\b|"
            r"\binbound phone call\b|\bphone call from\b",
            re.I,
        ),
        channel="Manual/Sales",
        detail=_manual_detail,
    ),
    Rule(
        name="other_explicit",
        pattern=re.compile(
            r"\bwalked into our office\b|\bwalk-in\b|\binfo@\b|\bgeneral inbox\b|"
            r"^\s*other\s*[-–—:]",
            re.I,
        ),
        channel="Other",
        detail=_other_explicit_detail,
    ),
    Rule(
        name="website_form",
        pattern=re.compile(
            r"\bfilled (?:out|in) the form\b|\bform on the\b|\bsubmitted the form\b|"
            r"\bcontact form\b|\bweb form\b|\brequested a demo on\b|\bbook-?a-?demo page\b",
            re.I,
        ),
        channel="Website",
        detail=lambda t: (f"{page} — form submission" if (page := _extract_page(t)) else "Website form submission"),
    ),
    Rule(
        # Recognised as ambiguous, not as a channel: the note says a post was commented on
        # but never says where. Inferring LinkedIn would be invention, and the taxonomy has
        # no generic social bucket. This is the slice the LLM tier exists for.
        name="unattributed_social",
        pattern=re.compile(
            r"(?:our|the)\s+post\b[^.]{0,60}\bcomment|\bcomment\w*\b[^.]{0,60}(?:our|the)\s+post\b",
            re.I,
        ),
        channel=None,
    ),
)


def match_rule(text: str) -> tuple[Rule, str] | None:
    """Return the first matching rule and the text span that triggered it."""
    for rule in RULES:
        match = rule.pattern.search(text)
        if match:
            return rule, match.group(0)
    return None


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def _fallback(evidence: str | None = None, detail: str | None = None) -> SourceExtraction:
    """What we return when nothing can be established: Other, flagged, no invented detail."""
    return SourceExtraction(
        channel="Other",
        detail=detail,
        confidence=config.SOURCE_CONFIDENCE_LOW,
        method="fallback",
        evidence=evidence,
        needs_review=True,
    )


def _from_llm_response(response: dict, evidence: str | None, method: str) -> SourceExtraction | None:
    """Validate a model response against the taxonomy. Anything off-contract is discarded."""
    channel = response.get("channel")
    if channel not in config.SOURCE_CHANNELS:
        return None

    detail = response.get("detail")
    if detail is not None:
        if not isinstance(detail, str):
            return None
        detail = _clean_fragment(detail)[:MAX_DETAIL_LENGTH] or None

    confident = bool(response.get("confident", False))
    return SourceExtraction(
        channel=channel,
        detail=detail,
        confidence=config.SOURCE_CONFIDENCE_MEDIUM if confident else config.SOURCE_CONFIDENCE_LOW,
        method=method,
        evidence=evidence,
        needs_review=not confident,
    )


def extract_source(text: str | None, client: LLMClient | None = None) -> SourceExtraction:
    """Extract a lead source from free text.

    `client` is injected so callers control whether the LLM tier is live. Passing None (the
    default) keeps the function fully deterministic, which is what the loader, the tests and
    a credential-free reviewer all get.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return _fallback()

    matched = match_rule(cleaned)
    if matched is not None and matched[0].channel is not None:
        rule, evidence = matched
        detail = rule.detail(cleaned) if rule.detail else None
        return SourceExtraction(
            channel=rule.channel,
            detail=detail or None,
            confidence=config.SOURCE_CONFIDENCE_HIGH,
            method=f"rule:{rule.name}",
            evidence=evidence,
            needs_review=False,
        )

    # Either a rule recognised the text as ambiguous, or no rule read it at all. Both are
    # escalation cases; both fall back safely when the LLM tier is unavailable.
    evidence = matched[1] if matched else None
    response = llm_module.classify_note(cleaned, client)
    if response is not None:
        method = "llm:cached" if getattr(client, "last_call_was_cached", False) else "llm"
        extraction = _from_llm_response(response, evidence, method)
        if extraction is not None:
            return extraction

    return _fallback(evidence=evidence)


def preserve_confident_source(
    existing: SourceExtraction | None, incoming: SourceExtraction
) -> SourceExtraction:
    """Decide which source survives when a lead is updated.

    A lead's original source is a first-touch fact: once it is confidently known, later
    activity on the record must not rewrite it. A newsletter signup from someone we first
    met at a conference does not make them a website lead. So the incoming value is only
    taken when the stored one is unknown, low-confidence or already flagged for review.
    """
    if existing is None:
        return incoming
    if existing.confidence in config.SOURCE_OVERWRITABLE_CONFIDENCES or existing.needs_review:
        if incoming.confidence not in config.SOURCE_OVERWRITABLE_CONFIDENCES:
            return incoming
    return existing
