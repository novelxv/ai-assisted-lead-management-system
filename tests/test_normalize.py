"""Normalization tests.

Normalization is the foundation every other feature stands on: if `Closed Won` and
`CLOSED WON ` are different values, filtering lies and deduplication misses. These tests
pin the behaviour that the rest of the system assumes.
"""

from __future__ import annotations

from datetime import date

import pytest

from app import config, normalize as nz


# --------------------------------------------------------------------------------------
# Status
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("New", "New"),
        ("new", "New"),
        ("NEW", "New"),
        (" New", "New"),
        ("New ", "New"),
        ("CLOSED LOST", "Closed Lost"),
        ("closed won", "Closed Won"),
        (" Qualified", "Qualified"),
        ("Opportunity ", "Opportunity"),
    ],
)
def test_status_variants_canonicalise(raw: str, expected: str) -> None:
    assert nz.normalize_status(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", None, "Nurturing", "closed-won"])
def test_unknown_status_is_none_not_a_guess(raw: str | None) -> None:
    """An unrecognised status must surface as unknown, never be coerced to a real one."""
    assert nz.normalize_status(raw) is None


def test_every_status_spelling_in_the_seed_file_is_recognised(seed_rows) -> None:
    """The real file spells 7 statuses 35 ways. All of them must land in the taxonomy."""
    raw_spellings = {row["Lead Status"] for row in seed_rows}
    assert len(raw_spellings) > 7, "fixture should still contain the messy spellings"
    normalized = {nz.normalize_status(v) for v in raw_spellings}
    assert None not in normalized
    assert normalized == set(config.LEAD_STATUSES)


# --------------------------------------------------------------------------------------
# Country
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("united kingdom", "United Kingdom"),
        ("United Kingdom", "United Kingdom"),
        ("hong kong", "Hong Kong"),
        ("UAE", "UAE"),
        ("uae", "UAE"),
        ("  south korea ", "South Korea"),
        ("", ""),
    ],
)
def test_country_canonicalisation(raw: str, expected: str) -> None:
    assert nz.normalize_country(raw) == expected


def test_seed_countries_collapse_by_casing_only(seed_rows) -> None:
    raw = {row["Country/Region"] for row in seed_rows}
    canonical = {nz.normalize_country(v) for v in raw}
    assert len(raw) > len(canonical)
    assert len(canonical) == 35


# --------------------------------------------------------------------------------------
# Email
# --------------------------------------------------------------------------------------


def test_email_is_trimmed_and_lowercased() -> None:
    assert nz.normalize_email("  Ama.Asante@KimTrading.com ") == "ama.asante@kimtrading.com"


def test_email_normalization_is_conservative_about_punctuation() -> None:
    """Dot-stripping is a Gmail behaviour, not a general one.

    Treating these as the same mailbox would merge two different people at most providers.
    Local-part punctuation is only ever a weak similarity signal, never proof of identity.
    """
    a = nz.normalize_email("first.last@acme.com")
    b = nz.normalize_email("firstlast@acme.com")
    assert a != b


def test_split_email() -> None:
    assert nz.split_email("f.osei@dialloimports.net") == ("f.osei", "dialloimports.net")
    assert nz.split_email("not-an-email") == ("", "")


# --------------------------------------------------------------------------------------
# Phone
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("+86 138 2424 7912", "8613824247912"),      # spaced international
        ("+82 10-8787-3705", "821087873705"),        # hyphenated
        ("46704602383", "46704602383"),              # bare digits, no '+' (159 rows in seed)
        ("+54 9 11 6370-9043", "5491163709043"),
        ("0044 7798 299448", "447798299448"),        # '00' access code dropped
        ("", ""),
    ],
)
def test_phone_reduces_to_comparable_digits(raw: str, expected: str) -> None:
    assert nz.normalize_phone(raw) == expected


def test_differently_formatted_numbers_compare_equal() -> None:
    """The seed pairs '+34 636 239 494' with the bare '34636239494' for the same person."""
    assert nz.normalize_phone("+34 636 239 494") == nz.normalize_phone("34636239494")


def test_phone_last9_ignores_country_code_prefix() -> None:
    assert nz.phone_last9(nz.normalize_phone("+44 7798 299448")) == "798299448"
    assert nz.phone_last9("1234") == ""


# --------------------------------------------------------------------------------------
# Company
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Kim Trading Pte Ltd", "kim trading"),
        ("Kim Trading Co.", "kim trading"),
        ("Kim Trading & Co", "kim trading"),
        ("Muller GmbH", "muller"),
        ("Santos Ltda", "santos"),
        ("Mensah Trading & and Co", "mensah trading"),
    ],
)
def test_universal_legal_suffixes_are_stripped(raw: str, expected: str) -> None:
    assert nz.normalize_company(raw) == expected


def test_industry_words_are_not_stripped() -> None:
    """Stripping 'Labs'/'Robotics'/'Studio' would be overfitting to this generator.

    They are real distinguishing words in other datasets, so they are kept and handled by
    token-set similarity instead.
    """
    assert nz.normalize_company("Lotus Finance Labs") == "lotus finance labs"
    assert nz.normalize_company("Ang Analytics Freight") == "ang analytics freight"


def test_company_name_that_is_only_a_legal_suffix_is_not_emptied() -> None:
    assert nz.normalize_company("Ltd") == "ltd"


def test_company_similarity_tolerates_differing_descriptors() -> None:
    """The same employer is written many ways; in the seed the display string disagreed in
    every duplicate-looking group."""
    a = nz.normalize_company("Lotus Finance Studio")
    b = nz.normalize_company("Lotus Finance Freight Solutions")
    assert nz.company_similarity(a, b) >= config.SIM_COMPANY_SIMILAR

    c = nz.normalize_company("Kim Trading Group")
    assert nz.company_similarity(a, c) < config.SIM_COMPANY_SIMILAR


# --------------------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------------------


def test_resolve_name_prefers_split_columns() -> None:
    assert nz.resolve_name("Ama", "Asante", "") == ("Ama", "Asante", "Ama Asante")


def test_resolve_name_falls_back_to_full_name_and_keeps_it_verbatim() -> None:
    given, family, display = nz.resolve_name("", "", "Natasha Dubois")
    assert (given, family) == ("Natasha", "Dubois")
    assert display == "Natasha Dubois"


def test_multi_token_given_names_survive_the_split() -> None:
    """'Xiu Ying', 'Jun Wei' and 'Wei Chen' are given names, not two people."""
    assert nz.split_full_name("Jun Wei Lim") == ("Jun Wei", "Lim")
    assert nz.split_full_name("Xiu Ying Ba") == ("Xiu Ying", "Ba")


def test_initial_style_full_name() -> None:
    given, family, display = nz.resolve_name("", "", "J. Diallo")
    assert (given, family) == ("J.", "Diallo")
    assert display == "J. Diallo"
    assert nz.is_initial(nz.name_key("J."))


def test_missing_name_everywhere_is_empty_not_an_error() -> None:
    assert nz.resolve_name("", "", "") == ("", "", "")


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("Ama", "Ama", "exact"),
        ("ama", "AMA", "exact"),          # casing is not a difference
        ("Jun", "Jun Wei", "prefix"),     # truncated record (real: seed phone group)
        ("Chris", "Christopher", "prefix"),
        ("Alex", "Alexander", "prefix"),
        ("J.", "Jamal", "initial"),
        ("L", "Luca", "initial"),
        ("Femi", "Sophia", "conflict"),   # real look-alike pair @acme.sg
        ("Michael", "Amina", "conflict"),
        ("J.", "Sophia", "conflict"),     # initials disagree
        ("Ama", "", "unknown"),
    ],
)
def test_given_name_relation(a: str, b: str, expected: str) -> None:
    assert nz.given_name_relation(a, b) == expected


@pytest.mark.parametrize(
    "a,b",
    [
        ("Sofia", "Sophia"),   # same name, different spelling  (0.73)
        ("Eric", "Erik"),      # usually different people       (0.75)
        ("Ana", "Anna"),       # usually different people       (0.86)
        ("Sara", "Sarah"),     # usually different people       (0.89)
        ("Rania", "Hana"),     # real look-alike pair @simrobotics.biz (0.67)
        ("Ben", "Benjamin"),   # a prefix, but too short to tell a nickname from a name
    ],
)
def test_middling_similarity_is_uninformative_not_evidence(a: str, b: str) -> None:
    """Names in the middle similarity range earn nothing in either direction.

    Spelling variants of one name and genuinely different names occupy the same similarity
    range, so a score here would be a guess dressed as evidence. Returning 'unknown' means
    the pair is decided by contact keys instead.
    """
    assert nz.given_name_relation(a, b) == "unknown"


def test_nickname_pair_is_a_known_blind_spot() -> None:
    """'Mike'/'Michael' (0.55) falls below even the conflict floor.

    Documented as a false-negative risk rather than hidden: the conflict weight suppresses
    rather than vetoes, so such a pair can still clear the bar when a contact key agrees.
    """
    assert nz.given_name_relation("Mike", "Michael") == "conflict"


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("Asante", "Asante", "exact"),
        ("Rasmussen", "Rasmusen", "similar"),  # typo-level (0.94)
        ("Oh", "Koh", "conflict"),             # real distinct-people pair @larsenas.net
        ("Li", "Liu", "conflict"),             # real distinct-people pair @raoventures.biz
        ("Ng", "Ang", "conflict"),             # real distinct-people pair @acme.com.au
        ("Santos", "Sato", "conflict"),        # real distinct-people pair @gohdigital.net
        ("Diallo", "", "unknown"),
    ],
)
def test_family_name_relation(a: str, b: str, expected: str) -> None:
    assert nz.family_name_relation(a, b) == expected


def test_short_family_names_require_exact_equality() -> None:
    """'Oh'/'Koh' and 'Li'/'Liu' both score 0.80 — the same as a genuine typo on a longer
    name. On two- and three-letter surnames a one-character difference is a different
    surname, so fuzzy matching is disabled below `MIN_FAMILY_LEN_FOR_FUZZY`."""
    assert nz.similarity("oh", "koh") == pytest.approx(0.8)
    assert nz.family_name_relation("Oh", "Koh") == "conflict"


@pytest.mark.parametrize(
    "local",
    ["f.osei", "francescao", "francesca.osei", "fosei", "osei"],
)
def test_localpart_recognised_as_derived_from_the_name(local: str) -> None:
    """The duplicate-looking group for Francesca Osei uses three of these spellings."""
    assert nz.localpart_is_name_derived(local, "Francesca", "Osei")


@pytest.mark.parametrize("local", ["sales", "info", "d.johnson", "mia.j"])
def test_localpart_not_derived_from_a_different_name(local: str) -> None:
    assert not nz.localpart_is_name_derived(local, "Francesca", "Osei")


def test_localpart_derivation_handles_hyphenated_given_names() -> None:
    assert nz.localpart_is_name_derived("min-jun.l", "Min-jun", "Lau")


# --------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------


def test_notes_keep_their_line_structure() -> None:
    """Notes are the one field that legitimately spans lines: ingest appends one entry per
    touchpoint, so collapsing them would destroy a lead's history."""
    value = "  Met at   the booth.  \n\n  [2026-06-12] Followed up.  "
    assert nz.normalize_notes(value) == "Met at the booth.\n\n[2026-06-12] Followed up."


def test_notes_normalization_handles_blank_input() -> None:
    assert nz.normalize_notes("") == ""
    assert nz.normalize_notes(None) == ""
    assert nz.normalize_notes("   \n  ") == ""


# --------------------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2026-06-02", date(2026, 6, 2)),
        ("2026-05-20T00:00:00Z", date(2026, 5, 20)),
        ("6/4/2026", date(2026, 6, 4)),
        ("12/21/2025", date(2025, 12, 21)),
        ("", None),
        ("   ", None),
        ("not a date", None),
    ],
)
def test_parse_date_handles_every_format_in_the_export(raw: str, expected: date | None) -> None:
    assert nz.parse_date(raw) == expected


def test_slash_dates_are_month_first(seed_rows) -> None:
    """Evidence, not assumption: across the slash dates the first component never exceeds
    12 while the second reaches 31, so d/m/Y is ruled out."""
    slash = [
        row["Create Date"].strip()
        for row in seed_rows
        if "/" in row["Create Date"]
    ]
    assert slash
    firsts = [int(v.split("/")[0]) for v in slash]
    seconds = [int(v.split("/")[1]) for v in slash]
    assert max(firsts) <= 12
    assert max(seconds) > 12


def test_every_date_in_the_seed_file_parses(seed_rows) -> None:
    for row in seed_rows:
        assert nz.parse_date(row["Create Date"]) is not None
        if row["Last Modified Date"].strip():
            assert nz.parse_date(row["Last Modified Date"]) is not None
