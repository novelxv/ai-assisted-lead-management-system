"""Duplicate detection tests.

Two halves, and both are necessary.

*Real* cases keep the pipeline honest about the data it was built for. *Synthetic* cases
cover the failure modes this generated dataset does not contain at all — every duplicate in
it kept its phone number, and nobody in it shares a switchboard — which is exactly where an
approach tuned to this file would break in production.
"""

from __future__ import annotations

import itertools

import pytest

from app import config, normalize as nz, repository
from app.dedupe.pipeline import blocking_keys, candidate_pairs, find_duplicates
from app.dedupe.scoring import score_pair
from app.models import Lead


def make_lead(
    lead_id: str,
    name: str,
    email: str,
    phone: str,
    *,
    company: str = "Acme Ltd",
    country: str = "Singapore",
    created_at: str | None = "2026-01-01",
) -> Lead:
    """Build a normalized Lead the same way the loader does."""
    given, family, display = nz.resolve_name("", "", name)
    normalized_email = nz.normalize_email(email)
    local, domain = nz.split_email(normalized_email)
    digits = nz.normalize_phone(phone)
    return Lead(
        id=lead_id,
        display_name=display,
        given_name=given,
        family_name=family,
        family_key=nz.name_key(family),
        company=company,
        company_norm=nz.normalize_company(company),
        job_title="",
        email=normalized_email,
        email_local=local,
        email_domain=domain,
        phone=phone,
        phone_digits=digits,
        phone_last9=nz.phone_last9(digits),
        country=nz.normalize_country(country),
        status="New",
        lifecycle_stage="Lead",
        owner="Test Owner",
        lead_score=None,
        created_at=created_at,
        last_modified_at=None,
        notes="",
        source_channel=None,
        source_detail=None,
        source_confidence=None,
        source_method=None,
        source_needs_review=False,
    )


@pytest.fixture(scope="session")
def leads(seeded_conn) -> list[Lead]:
    return repository.all_leads(seeded_conn)


def _find(leads: list[Lead], lead_id: str) -> Lead:
    return next(lead for lead in leads if lead.id == lead_id)


# --------------------------------------------------------------------------------------
# Candidate generation
# --------------------------------------------------------------------------------------


def test_blocking_replaces_the_quadratic_comparison(leads) -> None:
    """2,049 leads is 2,098,176 brute-force comparisons; that is the wrong shape."""
    pairs, stats = candidate_pairs(leads)
    assert stats["all_pairs_if_brute_forced"] > 2_000_000
    assert len(pairs) < 1_000
    assert stats["reduction_factor"] > 1_000


def test_no_single_blocking_key_carries_the_whole_load(leads) -> None:
    """Recall must not depend on phone numbers agreeing.

    If every candidate came from one key, a source that reformatted phones would silently
    halve recall with no visible failure.
    """
    kinds = {kind for lead in leads[:500] for kind, _ in blocking_keys(lead)}
    assert kinds == {"email", "phone", "domain_family", "family_initial_country"}


def test_blocking_recovers_pairs_that_agree_on_phone_and_domain(leads) -> None:
    """Coverage of a reference set built by an independent construction heuristic.

    Not ground truth: by construction this set cannot contain a duplicate whose phone
    differs, so it bounds candidate-generation recall for contact-identical duplicates only.
    """
    buckets: dict[tuple[str, str], list[int]] = {}
    for index, lead in enumerate(leads):
        if lead.phone_digits and lead.email_domain:
            buckets.setdefault((lead.phone_digits, lead.email_domain), []).append(index)
    reference = [
        pair
        for members in buckets.values()
        if len(members) > 1
        for pair in itertools.combinations(sorted(members), 2)
    ]
    pairs, _ = candidate_pairs(leads)
    assert reference
    assert all(pair in pairs for pair in reference)


# --------------------------------------------------------------------------------------
# Real duplicates
# --------------------------------------------------------------------------------------


def test_exact_duplicate_scores_high(leads) -> None:
    """Same email and phone, only the company suffix differs."""
    result = score_pair(_find(leads, "100234871"), _find(leads, "100234872"))
    assert result.confidence == "high"
    assert any("identical email" in reason for reason in result.reasons)


def test_near_duplicate_with_a_different_local_part_scores_high(leads) -> None:
    """Francesca Osei appears as f.osei, francescao and francesca.osei.

    The addresses are genuinely different mailboxes, so the match rests on the phone plus
    the observation that all three local parts spell the same person.
    """
    left, right = _find(leads, "100236546"), _find(leads, "100236548")
    result = score_pair(left, right)
    assert left.email != right.email
    assert result.confidence == "high"
    assert any("spell the same person" in reason for reason in result.reasons)


def test_initial_abbreviated_duplicate_scores_high(leads) -> None:
    """'J. Yoon' and 'Ji-woo Yoon' are the same person, recorded two ways."""
    result = score_pair(_find(leads, "100234955"), _find(leads, "100234956"))
    assert result.confidence == "high"
    assert any("initial is compatible" in reason for reason in result.reasons)


def test_company_display_name_differences_do_not_block_a_match(leads) -> None:
    """The company string disagreed in every duplicate-looking group in this export."""
    left, right = _find(leads, "100234871"), _find(leads, "100234872")
    assert left.company != right.company
    assert score_pair(left, right).confidence == "high"


def test_triples_form_a_single_group_not_three_pairs(leads) -> None:
    result = find_duplicates(leads)
    sizes = {len(group.leads) for group in result.groups}
    assert sizes == {2, 3}
    triple = next(g for g in result.groups if len(g.leads) == 3)
    assert len({lead.id for lead in triple.leads}) == 3


def test_every_group_member_shares_a_family_name(leads) -> None:
    """A cheap invariant that would catch a grouping bug wiring unrelated records together."""
    for group in find_duplicates(leads).groups:
        keys = {lead.family_key for lead in group.leads}
        assert len(keys) == 1


# --------------------------------------------------------------------------------------
# Real non-duplicates — the precision cases
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "left_id,right_id,description",
    [
        ("100235160", "100235161", "Femi vs Sophia Diallo at acme.sg"),
        ("100236182", "100236183", "Erik vs Antoine Silva at bennett.com.au"),
        ("100235879", "100236318", "Michael vs Amina Johnson at fischertrading.com.au"),
        ("100235513", "100236733", "Rania vs Hana Andersen at simrobotics.biz"),
    ],
)
def test_colleagues_who_share_a_company_and_surname_are_not_duplicates(
    leads, left_id: str, right_id: str, description: str
) -> None:
    """A similar name plus the same employer must never be enough on its own."""
    result = score_pair(_find(leads, left_id), _find(leads, right_id))
    assert result.confidence == "low", f"{description} scored {result.score}"


def test_a_real_duplicate_and_a_real_colleague_at_one_company_are_separated(leads) -> None:
    """boatengdigital.com holds Mia Johnson twice and a distinct Diego Johnson."""
    mia_a, mia_b = _find(leads, "100235417"), _find(leads, "100235418")
    diego = _find(leads, "100235419")
    assert score_pair(mia_a, mia_b).confidence == "high"
    assert score_pair(mia_a, diego).confidence == "low"

    groups = find_duplicates(leads).groups
    containing = [g for g in groups if any(lead.id == mia_a.id for lead in g.leads)]
    assert len(containing) == 1
    assert {lead.id for lead in containing[0].leads} == {mia_a.id, mia_b.id}


def test_no_lookalike_pair_reaches_the_high_band(leads) -> None:
    """The whole stress set at once, so a weight change cannot quietly break precision."""
    buckets: dict[tuple[str, str], list[int]] = {}
    for index, lead in enumerate(leads):
        if lead.email_domain and lead.family_key:
            buckets.setdefault((lead.email_domain, lead.family_key), []).append(index)
    lookalikes = [
        (a, b)
        for members in buckets.values()
        if len(members) > 1
        for a, b in itertools.combinations(sorted(members), 2)
        if leads[a].phone_digits != leads[b].phone_digits
    ]
    assert len(lookalikes) > 50
    for a, b in lookalikes:
        assert score_pair(leads[a], leads[b]).confidence != "high"


def test_same_name_at_a_different_employer_is_reviewed_not_merged(leads) -> None:
    """Two 'Bashir Malik' records at different companies: a job change, or two people.

    This is the shape the review band exists for. It must be surfaced and must not be
    silently merged into the duplicate group.
    """
    result = find_duplicates(leads)
    review_ids = {
        frozenset((pair.left.id, pair.right.id)) for pair in result.review_pairs
    }
    assert frozenset(("100235443", "100236473")) in review_ids

    grouped = [
        g for g in result.groups if any(lead.id == "100236473" for lead in g.leads)
    ]
    assert grouped == []


# --------------------------------------------------------------------------------------
# Synthetic boundaries: risks this dataset does not contain
# --------------------------------------------------------------------------------------


def test_two_people_sharing_one_phone_line_are_not_merged() -> None:
    """Reception desks, spouses and shared team lines all exist.

    Nothing in the seed data exercises this, because every duplicate there kept its own
    phone number — which is exactly why it is worth a synthetic test.
    """
    a = make_lead("1", "Priya Nair", "priya.nair@acme.com", "+65 8000 1111")
    b = make_lead("2", "Rohan Mehta", "rohan.mehta@acme.com", "+65 8000 1111")
    result = score_pair(a, b)
    assert result.confidence == "low"
    assert result.has_identity_conflict
    assert find_duplicates([a, b]).groups == []


def test_colleagues_sharing_a_phone_line_and_a_surname_are_still_not_merged() -> None:
    a = make_lead("1", "Priya Nair", "priya.nair@acme.com", "+65 8000 1111")
    b = make_lead("2", "Rohan Nair", "rohan.nair@acme.com", "+65 8000 1111")
    result = score_pair(a, b)
    assert result.confidence != "high"
    assert find_duplicates([a, b]).groups == []


def test_identical_name_at_one_company_with_no_contact_agreement_is_reviewed() -> None:
    """Genuinely undecidable from the record alone, so it goes to a human, not to a merge."""
    a = make_lead("1", "Wei Chen", "w.chen@acme.com", "+65 8000 1111")
    b = make_lead("2", "Wei Chen", "wei.chen2@acme.com", "+65 9000 2222")
    result = score_pair(a, b)
    assert result.confidence == "medium"
    assert find_duplicates([a, b]).groups == []


def test_a_duplicate_whose_phone_was_changed_is_still_surfaced() -> None:
    """The reference set in the seed cannot contain this case, so it is tested directly."""
    a = make_lead("1", "Ama Asante", "ama.asante@kimtrading.com", "+44 7937 541221")
    b = make_lead("2", "Ama Asante", "a.asante@kimtrading.com", "+44 7700 900123")
    result = score_pair(a, b)
    assert result.confidence == "medium"
    assert len(find_duplicates([a, b]).review_pairs) == 1


def test_a_nickname_pair_with_strong_contact_evidence_still_matches() -> None:
    """'Mike'/'Michael' reads as a name conflict to string comparison.

    The conflict weight suppresses rather than vetoes precisely so that an exact email and
    phone match still wins. This documents the intended balance.
    """
    a = make_lead("1", "Mike Foster", "m.foster@acme.com", "+65 8000 1111")
    b = make_lead("2", "Michael Foster", "m.foster@acme.com", "+65 8000 1111")
    assert score_pair(a, b).confidence == "high"


def test_a_bridged_group_is_flagged_and_demoted() -> None:
    """A-B strong and B-C strong, but A and C contradict each other.

    Union-find would happily present {A, B, C} as one confident group because B bridges
    them. A cluster the system cannot justify internally should be handed to a human, not
    asserted.
    """
    a = make_lead("1", "Jamal Diallo", "jamal.diallo@acme.com", "+65 8000 1111")
    b = make_lead("2", "J. Diallo", "j.diallo@acme.com", "+65 8000 1111")
    c = make_lead("3", "Jun Diallo", "jun.diallo@acme.com", "+65 8000 1111")

    assert score_pair(a, b).confidence == "high"
    assert score_pair(b, c).confidence == "high"
    assert score_pair(a, c).has_identity_conflict

    groups = find_duplicates([a, b, c]).groups
    assert len(groups) == 1
    group = groups[0]
    assert {lead.id for lead in group.leads} == {"1", "2", "3"}
    assert group.has_internal_conflict
    assert group.confidence == "medium"
    assert any("conflict on identity" in reason for reason in group.reasons)


def test_a_consistent_triple_is_not_flagged() -> None:
    """The mirror of the bridge test: no false alarm on a genuinely coherent group."""
    leads = [
        make_lead("1", "Ama Asante", "ama.asante@acme.com", "+65 8000 1111"),
        make_lead("2", "Ama Asante", "a.asante@acme.com", "+65 8000 1111"),
        make_lead("3", "A. Asante", "amaasante@acme.com", "+65 8000 1111"),
    ]
    groups = find_duplicates(leads).groups
    assert len(groups) == 1
    assert not groups[0].has_internal_conflict
    assert groups[0].confidence == "high"


def test_review_pairs_never_become_groups() -> None:
    """Ambiguity must not chain: only `high` edges may merge records."""
    a = make_lead("1", "Wei Chen", "w.chen@acme.com", "+65 8000 1111")
    b = make_lead("2", "Wei Chen", "wei.chen2@acme.com", "+65 9000 2222")
    c = make_lead("3", "Wei Chen", "wchen3@acme.com", "+65 7000 3333")
    result = find_duplicates([a, b, c])
    assert result.groups == []
    assert len(result.review_pairs) == 3


def test_scoring_is_symmetric_and_deterministic() -> None:
    a = make_lead("1", "Ama Asante", "ama.asante@acme.com", "+65 8000 1111")
    b = make_lead("2", "A. Asante", "a.asante@acme.com", "+65 8000 1111")
    assert score_pair(a, b).score == score_pair(b, a).score
    assert score_pair(a, b).score == score_pair(a, b).score


def test_oversized_blocks_are_skipped_rather_than_expanded() -> None:
    """A placeholder phone number shared by hundreds of rows must not explode into pairs."""
    shared = [
        make_lead(str(i), f"Person{i} Smith", f"p{i}@acme.com", "+65 0000 0000")
        for i in range(config.MAX_BLOCK_SIZE + 5)
    ]
    pairs, stats = candidate_pairs(shared)
    assert stats["oversized_blocks_skipped"] >= 1


# --------------------------------------------------------------------------------------
# The LLM adjudication tier
# --------------------------------------------------------------------------------------


class FakeLLM:
    """Test double. Records calls so a test can prove the tier was not reached."""

    def __init__(self, response=None):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, *, system: str, user: str):
        self.calls.append((system, user))
        return self.response


def test_the_llm_is_never_asked_about_a_decided_pair() -> None:
    """Confident pairs and rejected pairs both bypass the model entirely."""
    spy = FakeLLM(response={"same_person": True, "confident": True, "reason": "x"})
    obvious = [
        make_lead("1", "Ama Asante", "ama.asante@acme.com", "+65 8000 1111"),
        make_lead("2", "Ama Asante", "ama.asante@acme.com", "+65 8000 1111"),
        make_lead("3", "Rohan Mehta", "rohan@other.com", "+44 7700 900999"),
    ]
    result = find_duplicates(obvious, client=spy)
    assert spy.calls == []
    assert result.stats["llm_adjudicated_pairs"] == 0


def test_the_llm_adjudicates_only_the_review_band() -> None:
    spy = FakeLLM(
        response={"same_person": False, "confident": True, "reason": "different employers"}
    )
    ambiguous = [
        make_lead("1", "Wei Chen", "w.chen@acme.com", "+65 8000 1111"),
        make_lead("2", "Wei Chen", "wei.chen2@acme.com", "+65 9000 2222"),
    ]
    result = find_duplicates(ambiguous, client=spy)
    assert len(spy.calls) == 1
    assert result.stats["llm_adjudicated_pairs"] == 1
    assert "different people" in result.review_pairs[0].adjudication


def test_an_unavailable_llm_leaves_the_pair_surfaced_rather_than_dropped() -> None:
    """Degrading must never lose a candidate a human should still see."""
    silent = FakeLLM(response=None)
    ambiguous = [
        make_lead("1", "Wei Chen", "w.chen@acme.com", "+65 8000 1111"),
        make_lead("2", "Wei Chen", "wei.chen2@acme.com", "+65 9000 2222"),
    ]
    result = find_duplicates(ambiguous, client=silent)
    assert len(result.review_pairs) == 1
    assert result.review_pairs[0].adjudication is None


def test_the_llm_verdict_never_promotes_a_pair_into_a_group() -> None:
    """Even a confident 'same person' only annotates; merging stays a human decision."""
    spy = FakeLLM(response={"same_person": True, "confident": True, "reason": "same person"})
    ambiguous = [
        make_lead("1", "Wei Chen", "w.chen@acme.com", "+65 8000 1111"),
        make_lead("2", "Wei Chen", "wei.chen2@acme.com", "+65 9000 2222"),
    ]
    result = find_duplicates(ambiguous, client=spy)
    assert result.groups == []
    assert "same person" in result.review_pairs[0].adjudication
