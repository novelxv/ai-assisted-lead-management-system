"""Central configuration: taxonomies, thresholds and scoring weights.

Every non-obvious number in this system lives here with the reasoning that produced it.
Nothing in `app/` should hard-code a threshold.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, get_args

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SEED_CSV = DATA_DIR / "leads_seed.csv"
SUBMISSIONS_JSON = DATA_DIR / "website_form_submissions.json"

# Overridable so tests can point at a temporary file.
DB_PATH = Path(os.environ.get("LEADS_DB_PATH", PROJECT_ROOT / "leads.db"))

# Runtime cache of *real* LLM responses. Gitignored: we never commit fabricated model output.
LLM_CACHE_PATH = Path(os.environ.get("LLM_CACHE_PATH", PROJECT_ROOT / ".cache" / "llm_cache.json"))

# --------------------------------------------------------------------------------------
# Lead status taxonomy
# --------------------------------------------------------------------------------------
# The seed file contains 34 raw spellings of these 7 values (casing + surrounding whitespace),
# e.g. "New", "new", "NEW", " New". We canonicalise to Title Case and keep the raw row in
# `raw_record` so nothing is lost.

LeadStatus = Literal[
    "New",
    "Contacted",
    "Connected",
    "Qualified",
    "Opportunity",
    "Closed Won",
    "Closed Lost",
]
LEAD_STATUSES: tuple[str, ...] = get_args(LeadStatus)

# --------------------------------------------------------------------------------------
# Source channel taxonomy (fixed by the assignment — do not extend)
# --------------------------------------------------------------------------------------

SourceChannel = Literal[
    "Website",
    "Event",
    "LinkedIn",
    "Organic Search",
    "Referral",
    "Manual/Sales",
    "Other",
]
SOURCE_CHANNELS: tuple[str, ...] = get_args(SourceChannel)

# Confidence attached to an extraction, by the path that produced it.
# These are ordinal labels, not probabilities.
SOURCE_CONFIDENCE_HIGH = "high"
SOURCE_CONFIDENCE_MEDIUM = "medium"
SOURCE_CONFIDENCE_LOW = "low"

# An extraction at or below this level may be overwritten by a later, better one
# (see assumption #8: a confident first-touch source is immutable).
SOURCE_OVERWRITABLE_CONFIDENCES = frozenset({SOURCE_CONFIDENCE_LOW})

# --------------------------------------------------------------------------------------
# Duplicate detection — blocking
# --------------------------------------------------------------------------------------

# A blocking key that lands more than this many records in one bucket is a bad key for that
# bucket (e.g. a placeholder phone number, or a very common surname in a large country).
# Emitting C(n,2) pairs from it would defeat the point of blocking, so the bucket is skipped
# and logged rather than silently expanded.
MAX_BLOCK_SIZE = 30

# --------------------------------------------------------------------------------------
# Duplicate detection — scoring weights
# --------------------------------------------------------------------------------------
# This is a TRANSPARENT ADDITIVE HEURISTIC, not a calibrated probability model. There is no
# labelled ground truth in this dataset, so any 0-1 "probability" would imply a calibration
# that does not exist. Points are an ordinal ranking aid; the band label is what callers act on.
#
# Weights are assigned from evidence-strength reasoning:
#   * A contact key that identifies a mailbox or a handset is the strongest single signal.
#   * Name agreement is necessary corroboration, never sufficient on its own.
#   * Company and country are weak: in this dataset the company display string disagreed in
#     every single duplicate-looking group, and colleagues share both.
#   * A name CONFLICT is strongly negative, so that no single contact key can carry a pair to
#     `high` on its own (assumption #10: shared handsets and shared mailboxes are real).

W_EMAIL_EXACT = 45           # same mailbox; strongest single evidence available
W_PHONE_EXACT = 40           # same handset; slightly below email (lines get reassigned/shared)
W_PHONE_LAST9 = 30           # agrees ignoring country-code formatting; same number, less certain
W_EMAIL_DOMAIN = 8           # same employer — weak, most companies have many employees
W_CREATE_DATE_EXACT = 10     # same-day re-entry is the classic double-keying signature
W_FAMILY_EXACT = 12
W_FAMILY_SIMILAR = 6         # typo-level agreement
W_FAMILY_CONFLICT = -25      # suppressive, not a veto (see below)
W_GIVEN_EXACT = 15
W_GIVEN_PREFIX = 10          # "Jun" vs "Jun Wei" — truncation, not a different person
W_GIVEN_SIMILAR = 8
W_GIVEN_INITIAL = 5          # "J." vs "Jamal": compatible, but ~1/20 of names share an initial
W_GIVEN_CONFLICT = -25       # suppressive, not a veto (see below)
W_LOCALPART_NAME_DERIVED = 8  # both local parts derive from the same person name
W_LOCALPART_SIMILAR = 4      # weak: punctuation-insensitive equality is NOT assumed (see normalize)
W_COMPANY_SIMILAR = 5
W_COUNTRY_EXACT = 4

# Why the name-conflict weights suppress rather than veto:
# nicknames and transliterations are real ("Mike"/"Michael" scores ~0.55 similarity), so a hard
# veto would silently drop true duplicates. -25 means a conflicting pair cannot reach `high` on
# one contact key alone, but can still be surfaced for review when several signals agree.
# Residual risk: a false NEGATIVE for a nickname pair with weak contact evidence. Documented
# in the README; the fix is a diminutive lexicon or phonetic key.

# Similarity cut-offs for the fuzzy comparisons above (difflib SequenceMatcher ratio).
SIM_FAMILY_SIMILAR = 0.88    # tight: surnames are short, so fuzzy matching them is risky
SIM_GIVEN_SIMILAR = 0.85
SIM_GIVEN_CONFLICT = 0.75    # below this, two *full* given names are treated as conflicting
SIM_LOCALPART_SIMILAR = 0.80
SIM_COMPANY_SIMILAR = 0.60   # token-set overlap, applied after legal-suffix stripping

# --------------------------------------------------------------------------------------
# Duplicate detection — bands
# --------------------------------------------------------------------------------------
# Chosen from what evidence a band should REQUIRE, then sanity-checked against the data:
#
#   high   >= 80  Requires a contact key (email 45 / phone 40) plus corroborating name
#                 agreement. Nothing reaches 80 on name + company + country alone (max 44).
#   medium 40-79  Genuine ambiguity: enough agreement to be worth a human's time, not enough
#                 to act on. Identical name at the same employer with no contact-detail
#                 agreement lands here (44), as does a duplicate whose phone was changed (62).
#   low     < 40  Dropped.
#
# `medium` is never merged and never auto-matched on ingest.

BAND_HIGH_MIN = 80
BAND_MEDIUM_MIN = 40

BandName = Literal["high", "medium", "low"]


def band_for(score: int) -> BandName:
    """Map a raw point total to its ordinal band label."""
    if score >= BAND_HIGH_MIN:
        return "high"
    if score >= BAND_MEDIUM_MIN:
        return "medium"
    return "low"


# Cap on members considered when re-checking a group for internal contradictions. Groups here
# are 2-3 records; the cap only bounds the cost if a pathological chain ever forms.
MAX_GROUP_SIZE_FOR_CONFLICT_CHECK = 10

# --------------------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------------------
# Used for exactly one thing: notes the deterministic rules cannot resolve (see
# app/source_extraction.py), plus adjudicating `medium`-band duplicate pairs. When no API key
# is present the system falls back deterministically and says so via the `method` field.

LLM_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")
LLM_MAX_TOKENS = 256
LLM_TEMPERATURE = 0.0  # classification, not generation: we want reproducible output
LLM_TIMEOUT_SECONDS = 20.0


def llm_api_key() -> str | None:
    """Read the key at call time so tests and reviewers can set it after import."""
    return os.environ.get("ANTHROPIC_API_KEY") or None


# --------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200

# Columns of the CSV export, in order. Kept stable so downstream consumers can rely on it.
EXPORT_COLUMNS = (
    "id",
    "display_name",
    "given_name",
    "family_name",
    "job_title",
    "company",
    "email",
    "phone",
    "country",
    "status",
    "lifecycle_stage",
    "owner",
    "created_at",
    "last_modified_at",
    "source_channel",
    "source_detail",
    "source_confidence",
    "source_needs_review",
    "lead_score",
    "notes",
)
