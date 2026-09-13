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
# Source channel taxonomy. Fixed contract: downstream reporting depends on these exact
# seven values, so extending it is a breaking change, not a config tweak.
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
W_GIVEN_INITIAL = 5          # "J." vs "Jamal": compatible, but ~1/20 of names share an initial
W_GIVEN_CONFLICT = -25       # suppressive, not a veto (see below)
W_LOCALPART_NAME_DERIVED = 8  # both local parts derive from the same person name
W_LOCALPART_SIMILAR = 4      # weak: punctuation-insensitive equality is NOT assumed (see normalize)
W_COMPANY_SIMILAR = 5
W_COUNTRY_EXACT = 4

# Why the name-conflict weights suppress rather than veto:
# nicknames and transliterations are real ("Mike"/"Michael" scores ~0.55 similarity), so a hard
# veto would silently drop genuine matches. -25 means a conflicting pair cannot reach `high` on
# one contact key alone, but can still be surfaced for review when several signals agree.
# Residual risk: a false NEGATIVE for a nickname pair with weak contact evidence. Documented
# in the README; the fix is a diminutive lexicon or phonetic key.

# Similarity cut-offs (difflib SequenceMatcher ratio).
#
# There is no positive "fuzzy given name" band, and that is deliberate. Measured on real
# name pairs, the two populations overlap almost entirely:
#
#   same name, different spelling   Mike/Michael 0.55 · Yusuf/Youssef 0.67 · Sofia/Sophia 0.73
#                                   Sergey/Sergei 0.83 · Katherine/Catherine 0.89
#   different people, similar names Eric/Erik 0.75 · Ana/Anna 0.86 · Jon/John 0.86
#                                   Alan/Allan 0.89 · Sara/Sarah 0.89
#
# String similarity therefore cannot tell a spelling variant from a different name, and a
# positive band would reward Ana/Anna as evidence of a duplicate. So given names get three
# zones: clearly compatible (exact / prefix / initial) scores, clearly different penalises,
# and everything between earns nothing in either direction. An honest zero beats a
# confident guess.
SIM_GIVEN_CONFLICT = 0.60    # below this no reading as the same name survives

# A string prefix only counts as a shortening ("Chris"/"Christopher") when the longer form
# is substantially longer. "Sara"/"Sarah" is a prefix too, but a one-character tail is a
# spelling variant — and those are uninformative, per the ranges above.
MIN_GIVEN_PREFIX_LEN = 4     # shorter forms ("Ben", "Dan") are too ambiguous to score
MIN_GIVEN_PREFIX_GAP = 3     # characters the longer form must add
SIM_FAMILY_SIMILAR = 0.90    # typo-level only: 'Rasmusen'/'Rasmussen' (0.94) yes, 'Smith'/'Smyth' (0.80) no
MIN_FAMILY_LEN_FOR_FUZZY = 4  # 'Oh'/'Koh', 'Li'/'Liu', 'Ng'/'Ang' all score 0.80 yet are
                              # different surnames; on short names only equality counts
SIM_LOCALPART_SIMILAR = 0.80
SIM_COMPANY_SIMILAR = 0.60   # token-set overlap, applied after legal-suffix stripping

# --------------------------------------------------------------------------------------
# Duplicate detection — bands
# --------------------------------------------------------------------------------------
# Each floor is derived from what evidence the band should REQUIRE, then checked against
# the data. The check is a sanity test, not the derivation.
#
#   high   >= 80  Must require a contact key. Email (45) or phone (40) plus corroborating
#                 name agreement clears it; name + company + country + date together reach
#                 only 66, so no amount of soft agreement can get there. That structural
#                 property is the point of the number.
#
#   medium 35-79  "A human should look at this." An exact given + family name match is 27,
#                 so the floor sits just below 27 + two independent weak corroborations
#                 (e.g. both local parts spelling that name, +8, and a shared country, +4
#                 = 39). Requiring 40 would demand three corroborations before a human is
#                 even told, which is too strict for a band whose only action is "review".
#                 Lands here: the same full name at a different employer (a job change, or
#                 two different people), and a duplicate whose phone was changed.
#
#   low     < 35  Dropped.
#
# Checked against the seed data: colleagues who share an employer and a surname peak at 25,
# comfortably clear of the review floor, and the pairs that do land in review are exactly
# the same-name-different-employer cases. `medium` is never merged and never auto-matched
# on ingest.

BAND_HIGH_MIN = 80
BAND_MEDIUM_MIN = 35

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
# Used for exactly two things: notes the deterministic rules cannot resolve (see
# app/source_extraction.py), plus adjudicating `medium`-band duplicate pairs. When no API key
# is present the system falls back deterministically and says so via the `method` field.
#
# Model choice: the deterministic rules already handle ~96% of notes and settle every
# duplicate pair outside the review band, so the model only ever sees a small, genuinely
# ambiguous slice. That makes a fast, inexpensive model the right fit — a larger one would
# cost more to reach the same "I cannot tell from this text" answer.

LLM_PROVIDER = "Google Gemini API"
LLM_SDK = "google-genai"

# Pinned default. This is the model the live validation documented in the README was run
# against, so the shipped configuration and the measured behaviour are the same thing. It
# gave reliable structured-output behaviour on these narrow classification tasks. Override
# with LLM_MODEL; anything with comparable JSON-schema support should work.
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.5-flash")

LLM_MAX_OUTPUT_TOKENS = 512
LLM_TEMPERATURE = 0.0  # classification, not generation: we want reproducible output
LLM_TIMEOUT_SECONDS = 30.0

# Gemini models reason before answering by default. These are short classification calls
# against an explicit schema, so that reasoning buys nothing: measured on this workload,
# "minimal" produced the same verdicts using 0 thinking tokens instead of ~330 per call.
# Set to "" to send no thinking preference at all.
LLM_THINKING_LEVEL = os.environ.get("LLM_THINKING_LEVEL", "minimal")

LLM_API_KEY_ENV = "GEMINI_API_KEY"


def llm_api_key() -> str | None:
    """Read the key at call time rather than at import, so the process environment can
    change without a reload.

    Environment only, so a credential cannot be committed by accident. A local .env is
    loaded into the environment at startup; see app/env.py.
    """
    return os.environ.get(LLM_API_KEY_ENV) or None


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
