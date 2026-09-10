"""Pair scoring: how much evidence says two records are the same person.

The score is a **transparent additive heuristic, not a probability**. There is no labelled
ground truth in this dataset, so publishing a calibrated-looking 0-1 number would imply a
confidence that has not been earned. Callers get an ordinal point total for ranking, a band
label to act on, and the list of signals that produced it.

Weights live in app/config.py with the reasoning behind each one. The shape that matters:

* No amount of name, company, country and date agreement reaches the `high` band on its own
  (the arithmetic maximum without a contact key is 66 against a bar of 80). Reaching `high`
  structurally requires an email or phone match — not as a special case, but as a property
  of the weights.
* A name conflict is worth -25, so neither a shared handset nor a shared mailbox can carry
  a pair over the bar alone. Colleagues share switchboards and role addresses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app import config, normalize as nz
from app.models import Lead


@dataclass(frozen=True)
class Signal:
    """One piece of evidence, with the points it contributed and why."""

    name: str
    points: int
    reason: str


@dataclass(frozen=True)
class PairScore:
    score: int
    confidence: str
    signals: list[Signal] = field(default_factory=list)
    has_identity_conflict: bool = False
    has_contact_evidence: bool = False

    @property
    def reasons(self) -> list[str]:
        """Human-readable explanation, strongest evidence first."""
        ordered = sorted(self.signals, key=lambda s: -abs(s.points))
        return [f"{s.reason} ({s.points:+d})" for s in ordered]


def _quote(value: str) -> str:
    return f"'{value}'" if value else "(blank)"


def score_pair(left: Lead, right: Lead) -> PairScore:
    """Score one candidate pair."""
    signals: list[Signal] = []

    def add(name: str, points: int, reason: str) -> None:
        signals.append(Signal(name=name, points=points, reason=reason))

    # ---- Contact keys -----------------------------------------------------------------
    email_match = bool(left.email) and left.email == right.email
    if email_match:
        add("email_exact", config.W_EMAIL_EXACT, f"identical email {_quote(left.email)}")
    elif left.email_domain and left.email_domain == right.email_domain:
        # Only counted when the addresses differ, so it is never double-counted with an
        # exact email match. Weak on its own: a domain is an employer, not a person.
        add("email_domain", config.W_EMAIL_DOMAIN, f"same email domain {_quote(left.email_domain)}")

    phone_match = False
    if left.phone_digits and left.phone_digits == right.phone_digits:
        phone_match = True
        add("phone_exact", config.W_PHONE_EXACT, "identical phone number")
    elif left.phone_last9 and left.phone_last9 == right.phone_last9:
        phone_match = True
        add(
            "phone_last9",
            config.W_PHONE_LAST9,
            "phone numbers agree apart from country-code formatting",
        )

    # ---- Name -------------------------------------------------------------------------
    family = nz.family_name_relation(left.family_name, right.family_name)
    if family == "exact":
        add("family_exact", config.W_FAMILY_EXACT, f"family name matches {_quote(left.family_name)}")
    elif family == "similar":
        add(
            "family_similar",
            config.W_FAMILY_SIMILAR,
            f"family names differ by a likely typo ({_quote(left.family_name)} / {_quote(right.family_name)})",
        )
    elif family == "conflict":
        add(
            "family_conflict",
            config.W_FAMILY_CONFLICT,
            f"family names differ ({_quote(left.family_name)} / {_quote(right.family_name)})",
        )

    given = nz.given_name_relation(left.given_name, right.given_name)
    if given == "exact":
        add("given_exact", config.W_GIVEN_EXACT, f"given name matches {_quote(left.given_name)}")
    elif given == "prefix":
        add(
            "given_prefix",
            config.W_GIVEN_PREFIX,
            f"one given name is a shortening of the other ({_quote(left.given_name)} / {_quote(right.given_name)})",
        )
    elif given == "initial":
        add(
            "given_initial",
            config.W_GIVEN_INITIAL,
            f"given name initial is compatible ({_quote(left.given_name)} / {_quote(right.given_name)})",
        )
    elif given == "conflict":
        add(
            "given_conflict",
            config.W_GIVEN_CONFLICT,
            f"given names differ ({_quote(left.given_name)} / {_quote(right.given_name)})",
        )

    # ---- Email local part, reasoned about as a name rather than as punctuation --------
    if not email_match and left.email_local and right.email_local:
        # Use the fuller name available across the pair: one record may only hold an initial.
        given_name = max([left.given_name, right.given_name], key=len)
        family_name = max([left.family_name, right.family_name], key=len)
        both_derived = nz.localpart_is_name_derived(
            left.email_local, given_name, family_name
        ) and nz.localpart_is_name_derived(right.email_local, given_name, family_name)
        if both_derived:
            add(
                "localpart_name_derived",
                config.W_LOCALPART_NAME_DERIVED,
                f"both email local parts spell the same person "
                f"({_quote(left.email_local)} / {_quote(right.email_local)})",
            )
        elif (
            nz.similarity(left.email_local, right.email_local) >= config.SIM_LOCALPART_SIMILAR
        ):
            add("localpart_similar", config.W_LOCALPART_SIMILAR, "email local parts are similar")

    # ---- Weak corroboration ------------------------------------------------------------
    if left.company_norm and right.company_norm:
        if nz.company_similarity(left.company_norm, right.company_norm) >= config.SIM_COMPANY_SIMILAR:
            add("company_similar", config.W_COMPANY_SIMILAR, f"same company {_quote(left.company)}")

    if left.country and left.country == right.country:
        add("country_exact", config.W_COUNTRY_EXACT, f"same country {_quote(left.country)}")

    if left.created_at and left.created_at == right.created_at:
        add(
            "created_same_day",
            config.W_CREATE_DATE_EXACT,
            f"both created on {left.created_at}",
        )

    total = sum(signal.points for signal in signals)
    return PairScore(
        score=total,
        confidence=config.band_for(total),
        signals=signals,
        has_identity_conflict=(given == "conflict" or family == "conflict"),
        has_contact_evidence=email_match or phone_match,
    )
