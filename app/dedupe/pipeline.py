"""Candidate generation, grouping and the overall duplicate pass.

Comparing every pair of 2,049 leads is 2,098,176 comparisons, and that number grows
quadratically — it is the wrong shape regardless of how fast each comparison is. So the
pipeline blocks first: records only get compared when they already share something that a
duplicate would almost certainly share.

The guiding asymmetry: **be aggressive in blocking, conservative in scoring.** Generating a
candidate costs one cheap comparison, so over-generating is nearly free; acting on a wrong
match costs a merged customer record, so scoring stays strict.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any

from app import config
from app.dedupe.scoring import PairScore, score_pair
from app.llm import LLMClient, adjudicate_pair
from app.models import Lead

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidatePair:
    left: Lead
    right: Lead
    score: PairScore
    adjudication: str | None = None


@dataclass(frozen=True)
class CandidateGroup:
    leads: list[Lead]
    score: int
    confidence: str
    reasons: list[str]
    has_internal_conflict: bool = False


@dataclass
class DedupeResult:
    groups: list[CandidateGroup] = field(default_factory=list)
    review_pairs: list[CandidatePair] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Blocking
# --------------------------------------------------------------------------------------


def blocking_keys(lead: Lead) -> list[tuple[str, str]]:
    """The keys a lead is filed under.

    Four keys, chosen so that no single one is load-bearing:

    * `email`  — the same address entered twice.
    * `phone`  — the same handset, ignoring country-code formatting.
    * `domain+family` — same employer and surname; catches re-entry where both the address
      and the phone were retyped differently.
    * `family+initial+country` — no shared employer needed, so it still fires when someone
      changed jobs or email provider. This is the key that keeps recall from depending on
      contact details agreeing at all.
    """
    keys: list[tuple[str, str]] = []
    if lead.email:
        keys.append(("email", lead.email))
    if lead.phone_last9:
        keys.append(("phone", lead.phone_last9))
    if lead.email_domain and lead.family_key:
        keys.append(("domain_family", f"{lead.email_domain}|{lead.family_key}"))
    if lead.family_key:
        initial = lead.given_name[:1].lower()
        keys.append(("family_initial_country", f"{lead.family_key}|{initial}|{lead.country.lower()}"))
    return keys


def candidate_pairs(leads: list[Lead]) -> tuple[set[tuple[int, int]], dict[str, Any]]:
    """Generate index pairs worth scoring, plus stats about how it went."""
    buckets: dict[tuple[str, str], list[int]] = {}
    for index, lead in enumerate(leads):
        for key in blocking_keys(lead):
            buckets.setdefault(key, []).append(index)

    pairs: set[tuple[int, int]] = set()
    skipped: list[tuple[str, str, int]] = []
    for (kind, value), members in buckets.items():
        if len(members) < 2:
            continue
        if len(members) > config.MAX_BLOCK_SIZE:
            # A bucket this large is not a useful key for these records — think a
            # placeholder phone number, or a very common surname in a populous country.
            # Expanding it would cost O(n^2) inside the bucket for almost no signal, so it
            # is skipped and reported rather than silently dropped.
            skipped.append((kind, value, len(members)))
            continue
        pairs.update(itertools.combinations(sorted(members), 2))

    total_possible = len(leads) * (len(leads) - 1) // 2
    stats = {
        "lead_count": len(leads),
        "all_pairs_if_brute_forced": total_possible,
        "candidate_pairs": len(pairs),
        "reduction_factor": round(total_possible / len(pairs), 1) if pairs else None,
        "blocks": len(buckets),
        "oversized_blocks_skipped": len(skipped),
    }
    if skipped:
        logger.info(
            "Skipped %d oversized blocks (> %d members): %s",
            len(skipped),
            config.MAX_BLOCK_SIZE,
            ", ".join(f"{k}={v} ({n})" for k, v, n in skipped[:5]),
        )
    return pairs, stats


# --------------------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------------------


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_b] = root_a


def _build_groups(
    leads: list[Lead], high_pairs: list[tuple[int, int, PairScore]]
) -> list[CandidateGroup]:
    """Merge `high` edges into groups, then re-check each group for self-contradiction.

    Union-find on pairwise edges can chain: A-B and B-C are both strong, so A, B and C
    become one group even though A and C contradict each other. B has acted as a bridge.
    Rather than build a full clustering algorithm for a problem this size, every intra-group
    pair is re-scored; if any of them conflicts on identity the group is flagged and demoted
    out of `high`. A cluster the system cannot justify internally is presented as needing
    review, not as a confident answer.
    """
    union = _UnionFind(len(leads))
    for left, right, _ in high_pairs:
        union.union(left, right)

    members: dict[int, list[int]] = {}
    for left, right, _ in high_pairs:
        members.setdefault(union.find(left), []).extend((left, right))

    edge_scores = {(left, right): score for left, right, score in high_pairs}

    groups: list[CandidateGroup] = []
    for indexes in members.values():
        unique = sorted(set(indexes))
        if len(unique) < 2:
            continue

        conflict = False
        weakest: PairScore | None = None
        conflict_reason: str | None = None

        if len(unique) <= config.MAX_GROUP_SIZE_FOR_CONFLICT_CHECK:
            for a, b in itertools.combinations(unique, 2):
                pair_score = edge_scores.get((a, b)) or score_pair(leads[a], leads[b])
                if weakest is None or pair_score.score < weakest.score:
                    weakest = pair_score
                if pair_score.has_identity_conflict:
                    conflict = True
                    conflict_reason = (
                        f"{leads[a].display_name or leads[a].id} and "
                        f"{leads[b].display_name or leads[b].id} conflict on identity"
                    )
        else:  # pragma: no cover - no group this large occurs at this data size
            weakest = min((s for _, _, s in high_pairs), key=lambda s: s.score)

        assert weakest is not None
        reasons = list(weakest.reasons)
        if conflict and conflict_reason:
            reasons.insert(0, f"demoted for review: {conflict_reason}")

        groups.append(
            CandidateGroup(
                leads=[leads[i] for i in unique],
                # A group is only as strong as its weakest internal link.
                score=weakest.score,
                confidence="medium" if conflict else weakest.confidence,
                reasons=reasons,
                has_internal_conflict=conflict,
            )
        )

    groups.sort(key=lambda g: (-g.score, g.leads[0].id))
    return groups


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------


def find_duplicates(
    leads: list[Lead],
    client: LLMClient | None = None,
    *,
    min_score: int | None = None,
    include_review: bool = True,
) -> DedupeResult:
    """Run the full pass: block, score, group, and escalate only what stays ambiguous."""
    pairs, stats = candidate_pairs(leads)

    high: list[tuple[int, int, PairScore]] = []
    review: list[CandidatePair] = []
    threshold = min_score if min_score is not None else config.BAND_MEDIUM_MIN

    for left, right in sorted(pairs):
        score = score_pair(leads[left], leads[right])
        if score.confidence == "high":
            high.append((left, right, score))
        elif score.confidence == "medium" and score.score >= threshold:
            review.append(CandidatePair(left=leads[left], right=leads[right], score=score))

    # The LLM only ever sees pairs the deterministic rules could not settle. On this dataset
    # that set is usually empty, which is a fact about the data rather than a claim that the
    # tier is unnecessary elsewhere; the stats below report the actual number.
    adjudicated = 0
    if client is not None and include_review:
        resolved: list[CandidatePair] = []
        for pair in review:
            verdict = adjudicate_pair(
                pair.left.summary(), pair.right.summary(), pair.score.reasons, client
            )
            same = verdict.get("same_person") if verdict else None
            confident = verdict.get("confident") if verdict else None
            # Both must be real booleans. `bool("false")` is True, so coercing a malformed
            # response would flip "different people, not confident" into a confident merge
            # suggestion. A response we cannot read is treated as no answer: the pair stays
            # surfaced for a human rather than carrying a misleading verdict.
            if not (isinstance(same, bool) and isinstance(confident, bool)):
                if verdict is not None:
                    logger.warning(
                        "Discarding malformed adjudication for %s/%s: %r",
                        pair.left.id,
                        pair.right.id,
                        verdict,
                    )
                resolved.append(pair)
                continue
            adjudicated += 1
            note = verdict.get("reason") or ""
            label = f"llm: {'same person' if same else 'different people'}"
            label += " (confident)" if confident else " (uncertain)"
            if note:
                label += f" — {note}"
            resolved.append(
                CandidatePair(
                    left=pair.left, right=pair.right, score=pair.score, adjudication=label
                )
            )
        review = resolved

    groups = _build_groups(leads, high)
    if min_score is not None:
        groups = [group for group in groups if group.score >= min_score]

    stats.update(
        {
            "high_confidence_pairs": len(high),
            "review_pairs": len(review),
            "groups": len(groups),
            "leads_in_groups": sum(len(group.leads) for group in groups),
            "llm_adjudicated_pairs": adjudicated,
            "llm_available": client is not None,
        }
    )
    return DedupeResult(
        groups=groups, review_pairs=review if include_review else [], stats=stats
    )
