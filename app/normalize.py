"""Normalization: the single source of truth for cleaning lead fields.

Loading, searching, filtering, deduplication and ingest all go through this module, so a
record is compared the same way no matter which entry point produced it. Nothing here is
specific to the seed file's generator — the rules are the ones you would want against any
CRM export.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime
from difflib import SequenceMatcher
from typing import Literal

from app import config

# --------------------------------------------------------------------------------------
# Generic text helpers
# --------------------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def collapse_ws(value: str | None) -> str:
    """Trim and collapse internal whitespace runs to a single space."""
    if not value:
        return ""
    return _WS_RE.sub(" ", value).strip()


def normalize_notes(value: str | None) -> str:
    """Tidy note text while preserving line structure.

    Notes are the one field that legitimately spans lines: ingest appends a timestamped
    entry per touchpoint, so collapsing whitespace the way every other field does would
    flatten a lead's entire history into one run-on paragraph.
    """
    if not value:
        return ""
    lines = [_WS_RE.sub(" ", line).strip() for line in value.splitlines()]
    return "\n".join(lines).strip()


def fold_ascii(value: str) -> str:
    """Casefold and strip diacritics, so 'Müller' and 'Muller' compare equal."""
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.casefold()


def similarity(a: str, b: str) -> float:
    """Similarity in [0, 1] for two short strings.

    stdlib `difflib` is used deliberately: the candidate set after blocking is in the
    hundreds of pairs, so speed is irrelevant and a dependency is not worth adding. Every
    fuzzy comparison in the system routes through here, so swapping in `rapidfuzz` later is
    a one-function change.
    """
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


# --------------------------------------------------------------------------------------
# Lead status
# --------------------------------------------------------------------------------------

_STATUS_LOOKUP = {fold_ascii(s): s for s in config.LEAD_STATUSES}


def normalize_status(value: str | None) -> str | None:
    """Canonicalise a lead status, or return None if it is not in the taxonomy.

    The seed file spells these 7 values 35 different ways (casing plus surrounding
    whitespace). Callers decide what an unknown value means: the loader records it as-is in
    `raw_record` and leaves the column null, the API rejects it with a 422.
    """
    key = fold_ascii(collapse_ws(value))
    return _STATUS_LOOKUP.get(key)


# --------------------------------------------------------------------------------------
# Country
# --------------------------------------------------------------------------------------

# Country names that are acronyms and must not be title-cased into 'Uae'.
_COUNTRY_ACRONYMS = {"uae": "UAE", "usa": "USA", "uk": "UK", "us": "US"}


def normalize_country(value: str | None) -> str:
    """Canonicalise country to a single display spelling.

    The seed file has 62 raw spellings that collapse to 35 countries purely through casing.
    Matching and filtering are done case-insensitively on this value.
    """
    cleaned = collapse_ws(value)
    if not cleaned:
        return ""
    folded = cleaned.casefold()
    if folded in _COUNTRY_ACRONYMS:
        return _COUNTRY_ACRONYMS[folded]
    # Title-case each word, leaving already-uppercase short tokens (acronyms) alone.
    words = []
    for word in cleaned.split(" "):
        if word.isupper() and len(word) <= 4:
            words.append(word)
        else:
            words.append(word[:1].upper() + word[1:].lower())
    return " ".join(words)


# --------------------------------------------------------------------------------------
# Email
# --------------------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_email(value: str | None) -> str:
    """Trim and lowercase. Nothing else.

    Deliberately conservative. Stripping '.', '_', '-' or '+' from the local part and
    treating the result as the same mailbox is a *Gmail* behaviour, not a general one:
    at most providers `first.last@acme.com` and `firstlast@acme.com` are different people.
    Local-part punctuation is therefore never used as evidence of identity — only as a weak
    similarity feature, and via the name-derivation check below.

    Lowercasing the local part is technically lossy (RFC 5321 leaves it case-sensitive) but
    is safe against every mainstream provider and prevents trivial casing duplicates.
    """
    return collapse_ws(value).lower()


def is_valid_email(value: str) -> bool:
    return bool(_EMAIL_RE.match(value))


def split_email(email: str) -> tuple[str, str]:
    """Return (local_part, domain); ('', '') if the value is not email-shaped."""
    if "@" not in email:
        return "", ""
    local, _, domain = email.rpartition("@")
    return local, domain


# --------------------------------------------------------------------------------------
# Phone
# --------------------------------------------------------------------------------------

_NON_DIGITS_RE = re.compile(r"\D")


def normalize_phone(value: str | None) -> str:
    """Reduce a phone number to comparable digits.

    The seed file mixes '+86 138 2424 7912', '+82 10-8787-3705' and bare '46704602383'.
    A leading '00' international access code is dropped so '0044...' and '+44...' agree.
    We deliberately do not strip a national trunk '0': without knowing the country's
    numbering plan that would corrupt the number.
    """
    digits = _NON_DIGITS_RE.sub("", value or "")
    if digits.startswith("00"):
        digits = digits[2:]
    return digits


def phone_last9(digits: str) -> str:
    """Last 9 digits — the subscriber portion, ignoring country-code formatting choices.

    Used for blocking, and as weaker evidence than a full match when the two numbers agree
    on the tail but not on the prefix.
    """
    return digits[-9:] if len(digits) >= 9 else ""


# --------------------------------------------------------------------------------------
# Company
# --------------------------------------------------------------------------------------

# Universal legal-entity and connector tokens only. Industry words ('Trading', 'Labs',
# 'Robotics', 'Freight Solutions' ...) are deliberately NOT stripped: those are this
# dataset's vocabulary, and hard-coding them would be overfitting to the generator.
_LEGAL_SUFFIX_TOKENS = frozenset(
    """
    inc incorporated llc lc ltd ltda limited co corp corporation company
    gmbh ag kg mbh sarl sas sa srl spa nv bv ab as oy ap aps plc llp lp
    pte pty pt kk bros and
    """.split()
)

_COMPANY_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)


def normalize_company(value: str | None) -> str:
    """Lowercase, drop punctuation, drop universal legal suffixes.

    'Kim Trading Pte Ltd', 'Kim Trading Co.' and 'Kim Trading & Co' all reduce to
    'kim trading'. If stripping would empty the name (a company literally called 'Ltd'),
    the punctuation-stripped form is kept instead.
    """
    cleaned = collapse_ws(value)
    if not cleaned:
        return ""
    folded = _COMPANY_PUNCT_RE.sub(" ", fold_ascii(cleaned))
    tokens = [t for t in folded.split() if t]
    kept = [t for t in tokens if t not in _LEGAL_SUFFIX_TOKENS]
    return " ".join(kept or tokens)


def company_similarity(a: str, b: str) -> float:
    """Token-set overlap of two normalized company names, in [0, 1].

    Overlap coefficient (shared / smaller set) rather than Jaccard, because the same
    employer is routinely written with different numbers of descriptor words —
    'Lotus Finance Studio' vs 'Lotus Finance Freight Solutions'. This is only ever a weak
    signal (worth +5 against an 80-point bar), so it can never carry a decision on its own.
    """
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


# --------------------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------------------

_INITIAL_RE = re.compile(r"^[^\W\d_]\.?$", re.UNICODE)


def name_key(value: str | None) -> str:
    """Comparable form of a single name part: accent-folded, casefolded, no trailing dot."""
    cleaned = collapse_ws(value).rstrip(".")
    return fold_ascii(cleaned)


def is_initial(value: str) -> bool:
    """True for 'J' or 'J.' — a single letter standing in for a given name."""
    return bool(_INITIAL_RE.match(value.strip()))


def split_full_name(full: str) -> tuple[str, str]:
    """Split one name string into (given, family) by taking the last token as the family name.

    This is a HEURISTIC, not a rule. It holds for this dataset (family names are always a
    single token, while given names are sometimes two: 'Xiu Ying Ba', 'Jun Wei Lim') but it
    breaks on Spanish double surnames, family-name-first orders and particles like
    'van der'. Consequences, enforced elsewhere:
      * the supplied string is stored verbatim as `display_name` and is what users see;
      * the split is used only for blocking and scoring;
      * a match resting on the split alone, with no email or phone agreement, cannot reach
        the `high` band (see app/dedupe/scoring.py and app/ingest.py).
    """
    tokens = collapse_ws(full).split(" ")
    if not tokens or tokens == [""]:
        return "", ""
    if len(tokens) == 1:
        return "", tokens[0]
    return " ".join(tokens[:-1]), tokens[-1]


def resolve_name(
    first: str | None, last: str | None, full: str | None
) -> tuple[str, str, str]:
    """Reconcile the three name columns into (given, family, display).

    The seed file populates either First/Last (1,942 rows) or Full Name (107 rows), never
    both. First/Last wins when present because it is already split by the source system;
    Full Name is preserved verbatim as the display value and split heuristically.
    """
    first, last, full = collapse_ws(first), collapse_ws(last), collapse_ws(full)
    if first or last:
        return first, last, collapse_ws(f"{first} {last}")
    if full:
        given, family = split_full_name(full)
        return given, family, full
    return "", "", ""


GivenRelation = Literal["exact", "prefix", "initial", "conflict", "unknown"]


def given_name_relation(a: str, b: str) -> GivenRelation:
    """Classify how two given names relate.

    Three zones, not a similarity gradient. Measured on real name pairs, "same name spelled
    differently" (Mike/Michael 0.55 ... Katherine/Catherine 0.89) and "different people with
    similar names" (Eric/Erik 0.75 ... Sara/Sarah 0.89) occupy the same similarity range, so
    a fuzzy match in the middle is not evidence of anything. 'unknown' is therefore a real
    outcome that earns neither points nor a penalty.

    An initial is never a conflict against a full name — 'J.' is compatible with 'Jamal' —
    but two disagreeing initials are.
    """
    ka, kb = name_key(a), name_key(b)
    if not ka or not kb:
        return "unknown"
    if ka == kb:
        return "exact"

    a_initial, b_initial = is_initial(ka), is_initial(kb)
    if a_initial or b_initial:
        return "initial" if ka[0] == kb[0] else "conflict"

    # Token-prefix: 'Jun' vs 'Jun Wei' is a truncated record, not a different person.
    ta, tb = ka.replace("-", " ").split(), kb.replace("-", " ").split()
    short, long = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if len(short) < len(long) and long[: len(short)] == short:
        return "prefix"

    # String-prefix: 'Chris' vs 'Christopher' is a shortening. 'Sara' vs 'Sarah' is a prefix
    # too but only adds one character, which makes it a spelling variant — and those carry
    # no signal. A prefix relationship never counts as evidence of *difference* either, so
    # 'Ben'/'Benjamin' falls through to 'unknown' rather than to a penalty.
    s, l = (ka, kb) if len(ka) <= len(kb) else (kb, ka)
    if l.startswith(s):
        substantial = (
            len(s) >= config.MIN_GIVEN_PREFIX_LEN
            and len(l) - len(s) >= config.MIN_GIVEN_PREFIX_GAP
        )
        return "prefix" if substantial else "unknown"

    if similarity(ka, kb) < config.SIM_GIVEN_CONFLICT:
        return "conflict"
    return "unknown"


FamilyRelation = Literal["exact", "similar", "conflict", "unknown"]


def family_name_relation(a: str, b: str) -> FamilyRelation:
    """Classify how two family names relate.

    Stricter than given names: for the same person a family name should agree, so anything
    below typo-level similarity is treated as a conflict. Legitimate name changes exist,
    which is exactly why a conflict suppresses (-25) rather than vetoes — an exact email or
    phone match still outweighs it.

    Short family names get equality only. 'Oh'/'Koh', 'Li'/'Liu' and 'Ng'/'Ang' all score
    0.80 on a string comparison, but a one-character difference in a two-character surname
    is a different surname, not a typo.
    """
    ka, kb = name_key(a), name_key(b)
    if not ka or not kb:
        return "unknown"
    if ka == kb:
        return "exact"
    if min(len(ka), len(kb)) < config.MIN_FAMILY_LEN_FOR_FUZZY:
        return "conflict"
    if similarity(ka, kb) >= config.SIM_FAMILY_SIMILAR:
        return "similar"
    return "conflict"


_LOCALPART_SEPARATORS_RE = re.compile(r"[._\-+]+")


def _compact(value: str) -> str:
    return _LOCALPART_SEPARATORS_RE.sub("", fold_ascii(value).replace(" ", ""))


def localpart_is_name_derived(local: str, given: str, family: str) -> bool:
    """True if an email local part is a plausible rendering of this person's name.

    'f.osei' and 'francescao' are both derivable from 'Francesca Osei', which is real
    evidence that two records describe the same person — and it is evidence about the
    *name*, not about punctuation, which is why it is used instead of the naive
    "strip the dots and compare" rule (see `normalize_email`).
    """
    g, f = _compact(given), _compact(family)
    if not local or (not g and not f):
        return False
    compact = _compact(local)
    candidates = set()
    for a, b in ((g, f), (f, g)):
        if not a or not b:
            continue
        candidates |= {a + b, a[:1] + b, a + b[:1], a[:1] + b[:1]}
    candidates |= {p for p in (g, f) if p}
    return compact in candidates


# --------------------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------------------

_DATE_FORMATS = (
    "%Y-%m-%d",              # 2026-06-02
    "%Y-%m-%dT%H:%M:%SZ",    # 2026-05-20T00:00:00Z
    "%m/%d/%Y",              # 6/4/2026  -- month first, see below
)


def parse_date(value: str | None) -> date | None:
    """Parse the three date formats present in the export; None if absent or unparseable.

    The slash format is month-first. That is not an assumption: across the 599 slash dates
    the first component never exceeds 12 while the second reaches 31, and duplicate records
    pair '12/21/2025' with the ISO '2025-12-21'.
    """
    cleaned = collapse_ws(value)
    if not cleaned:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def to_iso(value: date | None) -> str | None:
    return value.isoformat() if value else None
