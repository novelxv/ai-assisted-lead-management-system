"""Evaluate the duplicate pipeline against the seed data.

    python -m scripts.evaluate_dedupe

**There is no labelled ground truth in this dataset.** Nobody has told us which records are
really the same person, so every number below is a proxy, and each one prints the limitation
that goes with it. A report that quoted "100% recall" without saying what the denominator
was would be worse than no report at all.

The four checks are deliberately independent of each other, and two of them are independent
of the scorer itself.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import sys
from pathlib import Path

from app import config, db, loader, repository
from app.dedupe.pipeline import candidate_pairs, find_duplicates
from app.dedupe.scoring import score_pair
from app.models import Lead

HUMAN_FLAG = "possible duplicate"


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def _note(text: str) -> None:
    print(f"  ! Limitation: {text}")


def reference_pairs(leads: list[Lead]) -> list[tuple[int, int]]:
    """Pairs that agree on BOTH exact phone digits and email domain.

    Built by a construction heuristic that shares no code with the scorer, so it is not
    simply the scorer grading its own homework. It is still not ground truth.
    """
    buckets: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for index, lead in enumerate(leads):
        if lead.phone_digits and lead.email_domain:
            buckets[(lead.phone_digits, lead.email_domain)].append(index)
    return [
        pair
        for members in buckets.values()
        if len(members) > 1
        for pair in itertools.combinations(sorted(members), 2)
    ]


def lookalike_pairs(leads: list[Lead]) -> list[tuple[int, int]]:
    """Same employer domain and same surname, but different phone numbers.

    A precision stress set: these are overwhelmingly colleagues rather than duplicates, and
    they are exactly the shape that a naive "similar name + same company" rule gets wrong.
    """
    buckets: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for index, lead in enumerate(leads):
        if lead.email_domain and lead.family_key:
            buckets[(lead.email_domain, lead.family_key)].append(index)
    return [
        (a, b)
        for members in buckets.values()
        if len(members) > 1
        for a, b in itertools.combinations(sorted(members), 2)
        if leads[a].phone_digits != leads[b].phone_digits
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    conn = db.connect(args.db)
    loader.ensure_loaded(conn)
    leads = repository.all_leads(conn)
    by_index = {lead.id: i for i, lead in enumerate(leads)}

    result = find_duplicates(leads)
    pairs, _ = candidate_pairs(leads)
    scored = {(a, b): score_pair(leads[a], leads[b]) for a, b in sorted(pairs)}
    high_pairs = {pair for pair, score in scored.items() if score.confidence == "high"}

    print(f"Evaluating {len(leads)} leads from {config.SEED_CSV.name}")
    print("NOTE: this dataset carries no labelled duplicates. Every figure below is a")
    print("      proxy measurement with its own stated limitation.")

    # -- 1 ------------------------------------------------------------------------------
    _rule("1. Tractability (exact, requires no labels)")
    stats = result.stats
    print(f"  Brute-force comparisons : {stats['all_pairs_if_brute_forced']:,}")
    print(f"  Candidate pairs scored  : {stats['candidate_pairs']:,}")
    print(f"  Reduction factor        : {stats['reduction_factor']:,}x")
    print(f"  Oversized blocks skipped: {stats['oversized_blocks_skipped']}")
    _note("this says the approach scales. It says nothing about whether it is correct.")

    # -- 2 ------------------------------------------------------------------------------
    _rule("2. Coverage of constructed reference pairs")
    reference = reference_pairs(leads)
    recovered = sum(1 for pair in reference if pair in pairs)
    confirmed = sum(1 for pair in reference if pair in high_pairs)
    print(f"  Reference pairs (exact phone + same email domain): {len(reference)}")
    print(f"  Recovered by blocking : {recovered}/{len(reference)}")
    print(f"  Scored 'high'         : {confirmed}/{len(reference)}")
    _note(
        "these pairs were constructed, not labelled. By construction the set cannot contain\n"
        "               a duplicate whose phone number differs, so this measures candidate-generation\n"
        "               recall for contact-identical duplicates and nothing wider."
    )

    # -- 3 ------------------------------------------------------------------------------
    _rule("3. Agreement with an independent human signal")
    flagged = [
        lead.id for lead in leads if HUMAN_FLAG in (lead.notes or "").lower()
    ]
    surfaced = {lead.id for group in result.groups for lead in group.leads}
    surfaced |= {
        pair.left.id for pair in result.review_pairs
    } | {pair.right.id for pair in result.review_pairs}
    covered = sum(1 for lead_id in flagged if lead_id in surfaced)
    print(f"  Rows a human annotated '{HUMAN_FLAG}': {len(flagged)}")
    print(f"  Surfaced by the pipeline               : {covered}/{len(flagged)}")
    _note(
        "this annotation is excluded from every scoring feature, so agreement here is\n"
        "               genuinely independent. But it marks only about half the reference set, its own\n"
        "               precision is unverifiable, and it speaks to recall on rows a human already\n"
        "               suspected - not to overall accuracy."
    )

    # -- 4 ------------------------------------------------------------------------------
    _rule("4. False-positive stress set")
    lookalikes = lookalike_pairs(leads)
    as_high = [pair for pair in lookalikes if pair in high_pairs]
    as_review = [
        pair
        for pair in lookalikes
        if pair in scored and scored[pair].confidence == "medium"
    ]
    print(f"  Same employer domain + same surname, different phone: {len(lookalikes)}")
    print(f"  Scored 'high'  (target: 0): {len(as_high)}")
    print(f"  Scored 'medium'           : {len(as_review)}")
    for pair in as_high[:5]:
        print(f"    ! {leads[pair[0]].summary()}  <->  {leads[pair[1]].summary()}")
    _note(
        "'colleague' here is inferred from differing given names, not confirmed. A handful\n"
        "               were inspected by hand; the rest are assumed, so treat this as a strong\n"
        "               indicator rather than a precision figure."
    )

    # -- 5 ------------------------------------------------------------------------------
    _rule("5. Score distribution and band sensitivity")
    bands = collections.Counter(score.confidence for score in scored.values())
    high_scores = [s.score for s in scored.values() if s.confidence == "high"]
    mid_scores = [s.score for s in scored.values() if s.confidence == "medium"]
    low_scores = [s.score for s in scored.values() if s.confidence == "low"]
    print(f"  Bands: {dict(bands)}")
    if low_scores:
        print(f"  Highest 'low'    : {max(low_scores):4d}   (review floor {config.BAND_MEDIUM_MIN})")
    if mid_scores:
        print(f"  'medium' range   : {min(mid_scores):4d}..{max(mid_scores)}")
    if high_scores:
        print(f"  Lowest 'high'    : {min(high_scores):4d}   (high floor {config.BAND_HIGH_MIN})")
    if low_scores and high_scores:
        print(
            f"  Nothing scores between {max(low_scores) + 1} and {min(high_scores) - 1} "
            f"except the {len(mid_scores)} review pair(s)."
        )
    print(f"  LLM adjudications performed: {stats['llm_adjudicated_pairs']} "
          f"(tier available: {stats['llm_available']})")
    _note(
        "scores cluster far from both floors, so the outcome here barely depends on exactly\n"
        "               where the thresholds sit. That separation is a property of how this data was\n"
        "               generated - duplicates kept their phone numbers - and is not evidence that\n"
        "               the thresholds generalise to a messier source."
    )

    # -- 6 ------------------------------------------------------------------------------
    _rule(f"6. Sample for manual inspection ({args.samples} groups)")
    for group in result.groups[: args.samples]:
        print(f"\n  score {group.score} ({group.confidence})")
        for lead in group.leads:
            print(f"    - {lead.id}  {lead.summary()}")
        for reason in group.reasons[:4]:
            print(f"      · {reason}")

    if result.review_pairs:
        _rule(f"7. Pairs needing human review ({len(result.review_pairs)})")
        for pair in result.review_pairs[: args.samples]:
            print(f"\n  score {pair.score.score} ({pair.score.confidence})")
            print(f"    - {pair.left.id}  {pair.left.summary()}")
            print(f"    - {pair.right.id}  {pair.right.summary()}")
            for reason in pair.score.reasons[:4]:
                print(f"      · {reason}")
            if pair.adjudication:
                print(f"      => {pair.adjudication}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
