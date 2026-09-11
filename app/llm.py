"""LLM access: one client, two call sites, and an honest fallback when unavailable.

The LLM is used for exactly two things, both of them genuinely ambiguous judgement calls
that deterministic code cannot settle:

1. `classify_note` — free-text notes the rule pass cannot resolve (app/source_extraction.py).
2. `adjudicate_pair` — duplicate pairs that land in the `medium` band (app/dedupe/scoring.py).

Everything else is deterministic, because deterministic is better: cheaper, reproducible,
explainable and testable. See the README for the measured split.

**Behaviour without credentials.** `get_client()` returns None when `ANTHROPIC_API_KEY` is
unset or the optional `anthropic` package is not installed, and every caller degrades to a
documented deterministic fallback. No model output is ever simulated: the on-disk cache
holds only real responses and is gitignored, and tests inject an explicit test double.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from app import config

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal surface the rest of the app depends on."""

    def complete_json(self, *, system: str, user: str) -> dict[str, Any] | None:
        """Return the parsed JSON object, or None if the call could not be completed."""


class ResponseCache:
    """File-backed cache of real responses, keyed by prompt hash.

    Two reasons it exists: the ambiguous notes repeat heavily, so caching collapses many
    records into far fewer calls; and a cached run is reproducible. It is gitignored — a
    committed cache would be indistinguishable from fabricated model output.

    Note that the key covers the *whole* prompt, so notes differing only by a trailing sales
    remark are separate entries. On the seed file the 91 ambiguous rows reduce to 13 distinct
    strings, not one. Keying on a stripped-down note would cache more aggressively, but it
    would also hand the model less context than the caller actually has.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._data: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._data is None:
            if self.path and self.path.exists():
                try:
                    self._data = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    logger.warning("Could not read LLM cache at %s; ignoring it", self.path)
                    self._data = {}
            else:
                self._data = {}
        return self._data

    @staticmethod
    def key(system: str, user: str) -> str:
        return hashlib.sha256(f"{system}\x00{user}".encode()).hexdigest()[:32]

    def get(self, key: str) -> dict[str, Any] | None:
        return self._load().get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        data = self._load()
        data[key] = value
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        except OSError:
            logger.warning("Could not write LLM cache at %s", self.path)


class AnthropicClient:
    """Anthropic Messages API client returning parsed JSON.

    Failures are logged and turned into None rather than raised: an unavailable model must
    degrade the answer, never break the request.
    """

    def __init__(self, api_key: str, cache: ResponseCache | None = None) -> None:
        from anthropic import Anthropic  # imported lazily: optional [llm] extra

        self._client = Anthropic(api_key=api_key, timeout=config.LLM_TIMEOUT_SECONDS)
        self._cache = cache if cache is not None else ResponseCache(config.LLM_CACHE_PATH)
        self.last_call_was_cached = False

    def complete_json(self, *, system: str, user: str) -> dict[str, Any] | None:
        cache_key = ResponseCache.key(system, user)
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self.last_call_was_cached = True
                return cached
        self.last_call_was_cached = False

        try:
            message = self._client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(
                block.text for block in message.content if getattr(block, "type", "") == "text"
            )
            parsed = _parse_json_object(text)
        except Exception as exc:  # noqa: BLE001 - any client/transport error degrades the same way
            logger.warning("LLM call failed (%s); falling back to deterministic handling", exc)
            return None

        if parsed is not None and self._cache:
            self._cache.put(cache_key, parsed)
        return parsed


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from a model response, tolerating fenced output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(cleaned[start : end + 1])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def get_client() -> LLMClient | None:
    """Build the default client, or None when the LLM tier is unavailable.

    Unavailable is a normal state, not an error: reviewers can run and evaluate the whole
    system without credentials.
    """
    api_key = config.llm_api_key()
    if not api_key:
        return None
    try:
        return AnthropicClient(api_key)
    except ImportError:
        logger.info("ANTHROPIC_API_KEY is set but the 'anthropic' package is not installed "
                    "(pip install -e '.[llm]'); using deterministic fallback")
        return None


# --------------------------------------------------------------------------------------
# Call site 1: ambiguous lead-source notes
# --------------------------------------------------------------------------------------

_SOURCE_SYSTEM = f"""You classify the ORIGINAL lead source described in a free-text CRM note.

Reply with a single JSON object and nothing else:
{{"channel": <one of {list(config.SOURCE_CHANNELS)}>, "detail": <short string or null>, "confident": <true|false>}}

Rules:
- Use only what the note states. Never infer an unstated company, platform, event or person.
- If the note names a social post but not which platform, the channel is "Other" — do not
  guess LinkedIn.
- If the note describes no source at all, return channel "Other", detail null, confident false.
- "detail" must quote or closely paraphrase the note. If you cannot fill it from the note,
  use null.
- Prefer the ORIGINATING channel over where the person eventually landed."""


def classify_note(text: str, client: LLMClient | None) -> dict[str, Any] | None:
    """Ask the model to classify one ambiguous note. None if the tier is unavailable."""
    if client is None:
        return None
    return client.complete_json(system=_SOURCE_SYSTEM, user=f"Note:\n{text.strip()}")


# --------------------------------------------------------------------------------------
# Call site 2: ambiguous duplicate pairs
# --------------------------------------------------------------------------------------

_DEDUPE_SYSTEM = """You decide whether two CRM records describe the SAME PERSON.

Reply with a single JSON object and nothing else:
{"same_person": <true|false>, "confident": <true|false>, "reason": <short string>}

Rules:
- Colleagues at one company often share a surname, a switchboard number and a domain. None
  of those alone makes them the same person.
- Differing given names mean different people unless one is clearly an initial or a
  shortening of the other.
- If the evidence is genuinely balanced, set confident false. A wrong merge is far more
  costly than an unresolved pair."""


def adjudicate_pair(
    left: str, right: str, signals: list[str], client: LLMClient | None
) -> dict[str, Any] | None:
    """Ask the model to settle one `medium`-band pair. None if the tier is unavailable."""
    if client is None:
        return None
    user = (
        f"Record A: {left}\n"
        f"Record B: {right}\n"
        f"Deterministic signals: {'; '.join(signals) if signals else 'none'}"
    )
    return client.complete_json(system=_DEDUPE_SYSTEM, user=user)
