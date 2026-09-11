# AI-Assisted Mini Lead Management System

A small backend that loads a messy 2,049-row CRM export, exposes a lead API over it, and adds
two AI-assisted features: **duplicate detection** and **lead-source extraction** from free-text
notes.

Python 3.10+ · FastAPI · SQLite (stdlib driver) · ~2,400 lines of application code, ~2,200 of
tests · **224 tests, no required external services**.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Open **<http://127.0.0.1:8000/docs>**. That is the intended UI — the brief does not ask for a
frontend, so the effort went into the API contract instead.

The database seeds itself on first run: if `leads.db` does not exist, the app loads
`data/leads_seed.csv` and logs a column-coverage report. There is no separate setup step. To
reload it deliberately:

```bash
python -m app.loader --force
```

```bash
pytest                              # 224 tests, ~3s
python -m scripts.evaluate_dedupe   # duplicate-detection evaluation report
```

Everything works with no API key. See [LLM usage](#llm-usage) for what changes if you add one.

---

## What the data actually is

I profiled both files before designing anything. The messiness is the problem, not an
appendix to it.

| | |
|---|---|
| Rows × columns | 2,049 × 22 |
| **Columns empty in every single row** | `City`, `Original Source Drill-Down 1`, `Annual Revenue`, `Marketing contact status`, `GDPR consent` |
| Sparse columns | `Lead Score` 7.5%, `Job Title` 60%, `Original Source` 50% blank |
| `Lead Status` | **34 raw spellings of 7 values** (casing + surrounding whitespace) |
| `Contact Owner` | 10 people, 20 spellings (66 rows have a trailing space) |
| `Country/Region` | 62 spellings of 35 countries |
| Dates | 3 formats: `2026-06-02`, `2026-05-20T00:00:00Z`, `6/4/2026` |
| Names | 1,942 rows use First+Last; **107 use Full Name only, 53 of those as an initial** (`J. Diallo`) |
| Given names | Multi-token (`Xiu Ying`, `Jun Wei`) and hyphenated (`Ha-eun`) |
| Phones | 1,890 with `+`, **159 as bare digit strings** |
| Company | **Unreliable as a string**: it disagreed in *every* duplicate-looking group |
| Encoding | UTF-8 with em dashes — must be opened explicitly, or Windows' default codepage corrupts it |

**Duplicate-looking records:** 232 groups (201 pairs, 31 triples) covering 495 rows. Within
each, phone digits and email domain are identical while the email local part and company
suffix vary — `Lotus Finance Studio` / `Lotus Finance Freight Solutions` / `Lotus Finance Labs`.

**Records that only look like duplicates:** 64 pairs share an employer domain *and* a surname
but have different given names — `Femi` vs `Sophia Diallo`, `Erik` vs `Antoine Silva`. At
`boatengdigital.com` there are two Mia Johnson records (a real duplicate) sitting beside a
distinct Diego Johnson. **This is the case the system is built to get right.**

**Form submissions** (90): 49 carry an email and phone identical to an existing lead and all
of them carry the same zero-signal message; 41 are new people, 38 at companies already in the
file. `form_id`, `form_name` and `page_url` **contradict each other** — a `form_demo_request`
named "Newsletter Signup" submitted from `/blog` — so none of them is trusted as evidence.

---

## Architecture

```
app/
  config.py             every taxonomy, threshold and weight, with its reasoning
  normalize.py          single source of truth for cleaning + name/company comparison
  db.py                 schema and indexes          models.py    internal record + API schemas
  repository.py         all SQL                     loader.py    CSV -> SQLite + coverage report
  llm.py                LLM client and its two call sites
  source_extraction.py  ordered rule table + escalation
  dedupe/scoring.py     pair features, points, human-readable reasons
  dedupe/pipeline.py    blocking -> candidate pairs -> grouping
  ingest.py             submission -> match (reuses the dedupe scorer) -> merge policy
  main.py               routes
scripts/evaluate_dedupe.py
```

Three decisions worth defending:

**SQLite via the stdlib driver, no ORM.** 2,049 rows and one table. A server would be
infrastructure with no payoff; an ORM would add a dependency and a layer of indirection
without removing any work. A file-backed database still gives real SQL filtering, indexes
that make the blocking keys cheap, and durable results across restarts.

**One table, plus a `raw_record` JSON column.** Rather than duplicating 22 fields into
normalized and `raw_*` twins, each lead stores typed normalized columns *and* the original
CSV row verbatim as JSON. Full provenance in one column, and `GET /leads/{id}` returns it so
you can see exactly what normalization did.

**`difflib`, not `rapidfuzz`.** After blocking there are ~390 pairs to compare, so speed is
irrelevant and a dependency is not worth it. Every fuzzy comparison routes through one
`similarity()` helper, so swapping the implementation is a one-function change.

---

## Deduplication

The brief's constraint is the design driver: 2,049 leads is 2,098,176 pairwise comparisons,
and that number grows quadratically. So the pipeline never enumerates pairs.

The governing asymmetry: **aggressive in blocking, conservative in scoring.** Generating a
candidate costs one cheap comparison. Acting on a wrong match corrupts a customer record.

### 1. Blocking

Four general-purpose keys; a lead is compared only with leads sharing at least one.

| Key | Catches |
|---|---|
| exact normalized email | the same address entered twice |
| phone last-9 digits | the same handset, ignoring country-code formatting |
| email domain + surname | same employer and surname, when both address and phone were retyped |
| surname + given initial + country | **no shared employer needed** — survives a job change |

The fourth key exists so recall never depends on contact details agreeing at all. A block
larger than `MAX_BLOCK_SIZE` (30) is skipped and logged rather than expanded — a placeholder
phone number shared by hundreds of rows would otherwise cost O(n²) inside the bucket for
almost no signal.

**Result: 389 candidate pairs instead of 2,098,176 — a 5,394× reduction.** The whole pass
runs in ~0.03s.

### 2. Scoring — a transparent heuristic, explicitly *not* a probability

There is no labelled ground truth in this dataset. Publishing a calibrated-looking `0.87`
would imply a confidence nobody has earned. Instead each pair gets an **additive point
total** for ranking, a **band label** to act on, and the list of signals that produced it.

Weights live in [`app/config.py`](app/config.py) with the reasoning for each. The shape
matters more than the exact numbers:

| Evidence | pts | | Evidence | pts |
|---|--:|---|---|--:|
| identical email | +45 | | given name matches | +15 |
| identical phone | +40 | | given name is a shortening | +10 |
| phone matches ignoring country code | +30 | | given name initial compatible | +5 |
| same creation date | +10 | | **given names conflict** | **−25** |
| surname matches | +12 | | **surnames conflict** | **−25** |
| both local parts spell the same person | +8 | | same email domain | +8 |
| local parts similar | +4 | | same company / same country | +5 / +4 |

Two properties are deliberate, not incidental:

- **Nothing reaches `high` without a contact key.** Name + company + country + date together
  max out at 66 against a bar of 80. "Similar name at the same company" *cannot* be enough,
  by arithmetic rather than by a special case.
- **A name conflict is −25**, so neither a shared handset nor a shared mailbox carries a pair
  over the bar alone. Reception desks, spouses and `info@` addresses are real.

**Bands:** `high` ≥ 80 · `medium` 35–79 (review) · `low` < 35 (dropped).

The review floor is 35 rather than 40 because an exact given+surname match is 27 points, so
the floor sits just below "full name plus two independent corroborations". At 40, the three
*same name at a different employer* pairs in this file — the classic job-change case — were
dropped instead of being shown to a human.

#### Why no embeddings

The discriminative signal lives in short structured identity strings: email, phone, surname.
Exact keys and edit distance dominate there, blocking already reduces the comparison set by
~3 orders of magnitude, and embeddings would add cost, latency and non-determinism for no
evident gain. I would revisit this for fuzzy *company* resolution across sources, where
semantic similarity actually carries information.

#### Why string similarity is *not* used for given names

I measured it rather than assuming:

```
same name, different spelling   Mike/Michael 0.55 · Sofia/Sophia 0.73 · Katherine/Catherine 0.89
different people, similar names Eric/Erik 0.75 · Ana/Anna 0.86 · Sara/Sarah 0.89
```

The populations overlap almost entirely, so a "fuzzy name match" bonus would reward
`Ana`/`Anna` as evidence of a duplicate. Given names therefore get three zones — clearly
compatible (exact / shortening / initial) scores, clearly different penalises, and everything
in between **earns nothing in either direction**. An honest zero beats a confident guess.

The same measurement drives two smaller rules: short surnames get equality only (`Oh`/`Koh`,
`Li`/`Liu`, `Ng`/`Ang` all score 0.80 yet are different surnames — and all three appear in
this file as distinct people), and a string prefix only counts as a shortening when the
longer form adds ≥3 characters, so `Chris`/`Christopher` scores but `Sara`/`Sarah` does not.

#### Why email punctuation is *not* normalized away

Stripping `.` `_` `-` `+` from a local part and treating the result as the same mailbox is a
**Gmail** behaviour, not a general one. At most providers `first.last@acme.com` and
`firstlast@acme.com` are different people. So local-part punctuation is never evidence of
identity. A principled substitute does that work instead: *do both local parts spell the same
person?* `f.osei` and `francescao` both derive from **Francesca Osei**, which is a claim about
the name, not about punctuation.

#### Why company names are barely used

The company display string disagreed in **every** duplicate-looking group. Only universal
legal suffixes are stripped (`Inc`, `Ltd`, `GmbH`, `SRL`, `Pte`, `& Co`…), never this file's
industry vocabulary (`Labs`, `Robotics`, `Studio`) — stripping those would be overfitting to
the generator. What survives is compared by token-set overlap and worth +5 out of 80.

### 3. Grouping, and the bridge guard

Union-find over **`high` edges only**. `medium` edges are surfaced as pairs and never merge,
so ambiguity cannot chain records together.

Union-find can still produce a self-contradicting cluster: A–B and B–C are both strong while
A and C conflict, with B acting as a bridge. Rather than build a clustering algorithm for a
problem this size, every intra-group pair is re-scored; if any conflicts on identity the
group is flagged `has_internal_conflict`, **demoted out of `high`**, and the contradicting
pair is named in its reasons. A cluster the system cannot justify internally is handed to a
human, not asserted. (Synthetic test: `test_a_bridged_group_is_flagged_and_demoted`.)

**Nothing is ever auto-merged.** The brief does not ask for it, and it is not safe without
review.

### 4. LLM adjudication — and an honest finding

`medium`-band pairs, and only those, can be escalated to the model. On this dataset the review
band holds 3 pairs and the tier runs only when credentials are present.

**Finding, stated plainly: on the provided data the deterministic pipeline settles essentially
everything, and the LLM contributes nothing.** Scores cluster at 87–131 and −9–29, with a
near-empty gap between. That is a fact about *this generated file* — every duplicate in it
kept its phone number — and not evidence that the tier is unnecessary against a messier
source. It is also exactly why the escalation path is covered by synthetic tests rather than
by seed rows.

### Evaluation

```bash
python -m scripts.evaluate_dedupe
```

**This dataset has no labelled duplicates.** Every figure is a proxy, and the script prints
the limitation beside each one rather than in a footnote.

| Check | Result | What it does **not** tell you |
|---|---|---|
| **Tractability** | 2,098,176 → 389 pairs (5,394×) | That the approach scales. Nothing about correctness. |
| **Constructed reference pairs** (exact phone + same domain) | 294/294 recovered by blocking; 294/294 scored `high` | These were constructed, not labelled. By construction the set **cannot contain a duplicate whose phone changed**, so it bounds candidate-generation recall for contact-identical duplicates only. |
| **Independent human signal** — 136 rows a human annotated `possible duplicate` | 136/136 surfaced | Genuinely independent (the annotation is excluded from every scoring feature), but it marks only ~half the reference set, its own precision is unverifiable, and it speaks to recall on rows a human already suspected. |
| **False-positive stress set** — 64 same-domain, same-surname, different-phone pairs | **0 reached `high`, 0 reached `medium`** | "Colleague" is inferred from differing given names. I inspected ~15 by hand; the rest are assumed. Treat as a strong indicator, not a precision figure. |

I also read through 10 accepted groups and the 5 highest-scoring rejected pairs by hand. The
weakest accepted groups (87) are all initial-abbreviation cases — `J. Yoon` / `Ji-woo Yoon`,
same phone, same domain, same creation date. The strongest rejected pairs (39) are the
same-name-different-employer cases, which is why they go to the review band rather than being
dropped.

---

## Source extraction

`extract_source(text)` returns `{channel, detail, confidence, method, evidence, needs_review}`.
`channel` is typed against the 7 allowed values and re-validated after any model call, so an
out-of-taxonomy value can never leave the API.

Three tiers:

**1. Rules — 1,958 of 2,049 notes (95.6%), with no cross-contamination between channels.**
An ordered pattern table. Patterns key on portable vocabulary (`booth|expo|summit|festival`,
`referred by`) rather than this file's sentence templates, so a reworded note still resolves.

*Why rules rather than an LLM over every note:* these notes state their channel in plain
words. For those a rule is better in every dimension that matters — free, instant,
reproducible, and explainable to the sales team who will eventually dispute a
classification. The model is reserved for cases where a rule would have to guess.

**Precedence is the design.** Notes routinely name two surfaces, so the ordering encodes one
principle: *the originating channel wins over the surface the person eventually landed on*,
and the landing surface is preserved in `detail`.

```
Referral → Event → paid ad → Organic Search → LinkedIn → Manual/Sales → explicit Other → Website → ambiguous
```

That ordering is load-bearing:

| Note | Result | Why the order matters |
|---|---|---|
| `Googled us and ended up on the book-a-demo page…` | `Organic Search` · `Google search → Book-a-demo page` | must not be read as `Website` |
| `Booked a demo … after clicking a google ad.` | `Other` · `Paid search (Google Ads) → Book-a-demo page` | must not be read as `Organic Search` |
| `Met at the booth during Retail Asia Expo…` (submitted via a web form) | `Event` | first touch beats the form it arrived through |

**2. LLM — the 4.4% the rules flag as ambiguous**, plus any unseen phrasing. Cached by
normalized text, so on this dataset the ambiguous slice collapses to **one distinct call**.

**3. Fallback — `Other`, `detail: null`, `needs_review: true`.** Never invents anything.

`method` always reports which path ran (`rule:event`, `llm`, `llm:cached`, `fallback`), so you
can audit any record.

### Two taxonomy gaps, and how I resolved them

The allowed channels are fixed by the brief. Two common note families have no home in them,
and both decisions are visible in the data rather than hidden:

- **Paid search (119 rows).** `"…after clicking a google ad"` is explicitly *not* organic, and
  the taxonomy has no paid channel. It maps to **`Other`** with the fact preserved in the
  detail — `Paid search (Google Ads) → Book-a-demo page`. Mislabelling it `Organic Search`
  would silently corrupt every channel-attribution number downstream. If a paid channel is
  ever added, this is a one-line change in the rule table.
- **Un-attributed social (91 rows).** `"Saw our post about replacing hubspot and commented"`
  names no platform. `Original Source` says `Social Media` for 40 of them, but the taxonomy has
  no generic social bucket and the *text* names no platform, so promoting it to `LinkedIn`
  would be invention. It stays `Other` with `needs_review: true`. **This is the slice the LLM
  tier exists for.**

### Not inventing detail

`"Spoke with them at our Mobile World Congress booth, no QR scan logged."` yields
`Mobile World Congress — Booth conversation (no QR scan)`, never `Booth QR Code`. Claiming a
scan that did not happen is the smallest possible fabrication and exactly the kind that
destroys trust in an extraction pipeline. All 30 distinct event details across the file
distinguish a scan, a denied scan, and an unspecified conversation.

The 49 form submissions saying only *"Following up after our earlier conversation, please send
more info."* return `Other` with `detail: null` and `needs_review: true`. There is no signal
there, and saying so is the correct answer.

---

## Ingest policy

`POST /leads/ingest` reuses the deduplication scorer rather than growing a second, subtly
different matcher.

**The asymmetry that drives everything:** attaching a submission to the wrong person corrupts
a record silently and nobody notices. Creating a near-duplicate is visible, reversible, and
already surfaced by `/leads/dedupe-candidates`.

| Tier | Condition | Action |
|---|---|---|
| 1 | exact email **and no name conflict** | update |
| 2 | exact phone **and positive name support** | update |
| 3 | no contact key, but the pair scores `high` | update |
| 4 | anything still plausible, or a contact key we **refused** | **create**, with `possible_duplicates` |
| 5 | nothing plausible | create |

Three rules keep this consistent with the dedupe philosophy instead of short-circuiting it:

- **No contact key is unconditional.** An exact email match whose name explicitly conflicts
  means a role address, a shared inbox, or a typo — so it is refused and reported.
- **When the submitted email or phone points at stored leads, only those leads are eligible.**
  Auto-matching some *other* record while quietly ignoring who owns the submitted address
  would be the worst of both worlds.
- **A refused contact-key match is always reported, whatever it scored.** Its low score is
  precisely *why* it was refused, so filtering the report by score would hide the evidence.

### Field policy on update — nothing is destroyed

| Field | Behaviour |
|---|---|
| `notes` | **Appended**, never replaced: `[2026-06-12 · web form · "Newsletter Signup" · /blog] <message>` |
| names | Fill if missing; upgrade an initial to the full form (`J. Yoon` → `Ji-woo Yoon`); **never** swap one full name for another |
| email, phone, company, country | Fill if blank; otherwise **preserve**, and record the differing value in the appended note |
| `status`, `owner`, `lifecycle_stage` | **Never touched.** An inbound form does not move a lead through the pipeline or reassign its owner |
| source channel | Fill only when the stored one is unknown or flagged — original source is a first-touch fact |
| `created_at` | Earliest known date wins |

Form metadata is recorded as note context, never used to classify: `form_id`, `form_name` and
`page_url` contradict each other throughout the real file.

**Verified over all 90 real submissions: 49 updated, 41 created, no false merges.** A returning
lead who was scanned at a TechCrunch Disrupt booth and later submits a newsletter form keeps
`source_channel: Event`, keeps its status and owner, and gains an appended note.

---

## API

All filters are case-insensitive and compare normalized values. Ordering is deterministic
(`created_at DESC, id ASC`), so pagination cannot repeat or drop rows.

| Endpoint | Notes |
|---|---|
| `GET /leads` | Filters `status`, `owner`, `country`, `q` (name/company/email). `limit` 1–200, `offset`. An unknown `status` is a **422 listing the allowed values**, not an empty result |
| `GET /leads/{id}` | Includes `raw_record`, the untouched source row. 404 if absent |
| `PATCH /leads/{id}` | `status`, `owner`, `notes` only. Empty body or an unknown field → 422 |
| `GET /leads/export` | CSV of the **whole filtered view**, ignoring pagination |
| `POST /leads/ingest` | **201** created / **200** updated |
| `POST /leads/dedupe-candidates` | Ranked groups + review pairs + stats. Nothing is merged |
| `POST /source/extract` | Classify arbitrary text |
| `GET /dashboard` | Counts by status and by **extracted** channel, plus how many need review |
| `GET /health` | Liveness |

```bash
curl 'localhost:8000/leads?status=qualified&country=India&limit=2'
curl 'localhost:8000/leads/100234811'
curl -X PATCH localhost:8000/leads/100234811 -H 'Content-Type: application/json' \
     -d '{"status":"closed won"}'
curl 'localhost:8000/leads/export?country=Singapore&status=Qualified' -o leads.csv
curl -X POST localhost:8000/source/extract -H 'Content-Type: application/json' \
     -d '{"text":"Spoke with them at our Web Summit 2026 booth, no QR scan logged."}'
#   -> {"channel":"Event","detail":"Web Summit 2026 — Booth conversation (no QR scan)", ...}

curl -X POST localhost:8000/leads/ingest -H 'Content-Type: application/json' \
     -d '{"name":"Karim Toure","email":"k.toure@liutrading.biz","phone":"+61 462 210 338",
          "company":"Liu Trading Studio","country":"Australia","form_name":"Newsletter Signup",
          "page_url":"/blog","submitted_at":"2026-06-12T18:17:00Z",
          "message":"Following up after our earlier conversation, please send more info."}'
#   -> 200 {"action":"updated","matched_by":"email","changed_fields":["last_modified_at","notes"], ...}

curl -X POST localhost:8000/leads/dedupe-candidates -H 'Content-Type: application/json' \
     -d '{"limit":5}'
```

`dedupe-candidates` returns the full pass in `stats` while truncating `groups` and
`review_pairs` to `limit`.

---

## LLM usage

| | |
|---|---|
| Provider / model | Anthropic, `claude-haiku-4-5` (optional `[llm]` extra) |
| Where | (1) notes the rules flag as ambiguous; (2) `medium`-band duplicate pairs |
| Why there | Both are genuine judgement calls under missing information. Everything else is deterministic because deterministic is better here |
| Without credentials | Falls back deterministically and reports `method: "fallback"`. **All 224 tests pass and every endpoint works with no key** |
| Enable it | `pip install -e ".[llm]"` and set `ANTHROPIC_API_KEY` |

**Two things stated plainly:**

*The integration was never run against the live API.* No key was available while building
this, so the model path is covered by an explicit test double (`FakeLLM` in the test suite).
I deliberately did **not** commit a "cache" of hand-written responses — that would amount to
simulating model output, and a reviewer could not tell it from the real thing. The runtime
cache (`.cache/llm_cache.json`) is gitignored and only ever holds genuine responses.

*The cost figure is an estimate, not a measurement.* The ambiguous slice of this dataset
collapses to **one distinct note text**, so a full pass is ~1 cached call of a few hundred
tokens — well under $0.01. Actual spend to date: **$0.00**.

The response is re-validated against the taxonomy before use: an out-of-taxonomy channel, a
non-string detail or a malformed body all fall back rather than propagate.

---

## Assumptions and tradeoffs

1. **Paid search maps to `Other`** with the paid fact in the detail (above).
2. **Un-attributed social stays `Other` + `needs_review`** rather than being guessed as LinkedIn.
3. **First touch beats landing surface** — search-then-page is `Organic Search`, and applying
   this consistently is what forces (1).
4. **No source signal → `Other`, `detail: null`, `needs_review: true`.** The taxonomy has no
   `Unknown`, and detail is never fabricated.
5. **Dashboard counts use the extracted channel**, not `Original Source` — that column is blank
   for half the rows and actively misleading on others (21 booth conversations are tagged
   `Other Campaigns`).
6. **`q` searches name, company and email only**, per the brief — deliberately not phone.
7. **Lead `id` is the source `Record ID`**; new leads continue the numeric sequence. A UUID
   would be safer against concurrent writers, but this is a single-process service.
8. **Original source is immutable once confidently known.** Both ingest and `PATCH` may only
   fill it in when the stored value is unknown or flagged.
9. **The `possible duplicate` note is an unverified human hint.** It is excluded from every
   scoring feature and used *only* as an independent evaluation signal — using it as a feature
   would be leakage and would not generalise.
10. **Splitting one name string into given/family is a heuristic, not a fact.** "Last token is
    the surname" holds here but breaks on Spanish double surnames, family-name-first orders and
    particles (`van der`). So the supplied name is stored verbatim as `display_name` and is what
    every response shows; the split is used only for blocking and scoring; and a match resting
    on the split alone can never reach `high`.
11. **Slash dates are month-first.** Evidence, not assumption: across 599 slash dates the first
    component never exceeds 12 while the second reaches 31, and duplicate rows pair `12/21/2025`
    with `2025-12-21`.
12. **The five all-empty columns are reported, not modelled.** The loader logs them, so you
    would notice if a future export started populating one.

---

## Known limitations

- **Nickname pairs are a blind spot.** `Mike`/`Michael` scores 0.55 and reads as a conflict. The
  weight suppresses rather than vetoes, so such a pair still matches when a contact key agrees —
  but with weak contact evidence it would be missed. A diminutive lexicon or a phonetic key
  (Double Metaphone) is the fix.
- **Thresholds are reasoned, not calibrated.** With no labels they cannot be fitted honestly.
  Scores here cluster far from both floors, so the outcome barely depends on where they sit —
  but that separation is a property of this generated file, not proof they generalise.
- **The rule precedence was validated on this file, not proven universal.** It rests on general
  principles and portable vocabulary, but a source with different phrasing would need the rule
  table revisited — and the `method`/`needs_review` fields are there to make that visible.
- **Family-name changes** (e.g. after marriage) score as a conflict; an exact email or phone
  match still outweighs it, but a record with neither would be missed.
- **Single-process, no concurrency control.** Two simultaneous ingests could race on
  `next_lead_id`. Fine for the brief; a real deployment needs a sequence or a UUID.
- **The LLM tier is untested against the live API** (above).
- **`q` is a `LIKE` scan.** Fine at 2,049 rows; at 10⁶ it needs an FTS index.

## What I'd do next

1. **Get labels.** A few hundred human-adjudicated pairs would turn every proxy in the
   evaluation into a real precision/recall number and let the weights be fitted rather than
   argued. This is by far the highest-value next step.
2. **A review UI for the `medium` band** — the pipeline already produces exactly the queue an
   operator would work through, with reasons attached. Decisions fed back become the labels
   from (1).
3. **Phonetic + diminutive name matching** to close the `Mike`/`Michael` gap.
4. **Merge execution with an undo trail**, once review exists. Deliberately not built now:
   irreversible merges without a review step are how CRMs lose data.
5. **Extraction drift monitoring.** Alert when the share of notes hitting `fallback` rises —
   that is the signal the rule table has fallen behind the sales team's vocabulary.
6. Run the LLM tier against the live API and replace the estimated cost with a measured one.

---

## Testing

**224 tests, ~3 seconds, no network.** Real cases and synthetic cases, because they cover
different risks.

*Real cases* run against the actual 2,049 rows — every status spelling, both phone formats,
all three date formats, the real duplicate triples, and the real look-alike pairs
(`Femi`/`Sophia Diallo`, `Mia`/`Diego Johnson`) as named precision tests.

*Synthetic cases* cover what this generated file **does not contain**, which is where an
approach tuned to it would break:

- two different people sharing one phone line, and colleagues sharing a line *and* a surname
- an identical name at one company with no contact-detail agreement → review, never merged
- a duplicate whose phone number was changed — impossible to test from the file, since the
  reference set is built on phone agreement
- a bridged group (A–B and B–C strong, A–C contradictory) → flagged and demoted
- an exact email match with a conflicting name → created with `possible_duplicates`, not merged
- an out-of-taxonomy LLM response → rejected; a silent LLM → pair still surfaced

Tests assert behaviour, not internals: no test asserts a score margin, because fitting a
margin to this dataset is exactly the mistake the evaluation section warns about.
