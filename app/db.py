"""SQLite storage.

Why SQLite and not Postgres or an in-memory dict: the dataset is ~2,000 rows and the query
surface is one table with a handful of filters, so a server would be infrastructure without
a payoff. A file-backed database still gives real SQL filtering, indexes that make the
blocking keys cheap, durable PATCH/ingest results across restarts, and a zero-install
setup. The stdlib driver is used directly rather than an ORM: the schema is a single table
and the queries are short, so an ORM would add a dependency and a layer of indirection
without removing any work.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS leads (
    id                  TEXT PRIMARY KEY,

    -- Identity, normalized. `display_name` is the supplied name verbatim and is what every
    -- response shows; given/family are a heuristic split used only for matching.
    display_name        TEXT NOT NULL DEFAULT '',
    given_name          TEXT NOT NULL DEFAULT '',
    family_name         TEXT NOT NULL DEFAULT '',
    family_key          TEXT NOT NULL DEFAULT '',

    company             TEXT NOT NULL DEFAULT '',
    company_norm        TEXT NOT NULL DEFAULT '',
    job_title           TEXT NOT NULL DEFAULT '',

    email               TEXT NOT NULL DEFAULT '',
    email_local         TEXT NOT NULL DEFAULT '',
    email_domain        TEXT NOT NULL DEFAULT '',

    phone               TEXT NOT NULL DEFAULT '',
    phone_digits        TEXT NOT NULL DEFAULT '',
    phone_last9         TEXT NOT NULL DEFAULT '',

    country             TEXT NOT NULL DEFAULT '',
    -- NULL when the source value is not in the taxonomy, so an unrecognised status is
    -- visible as unknown rather than silently coerced into a real one.
    status              TEXT,
    lifecycle_stage     TEXT NOT NULL DEFAULT '',
    owner               TEXT NOT NULL DEFAULT '',
    lead_score          INTEGER,

    created_at          TEXT,
    last_modified_at    TEXT,
    notes               TEXT NOT NULL DEFAULT '',

    source_channel      TEXT,
    source_detail       TEXT,
    source_confidence   TEXT,
    source_method       TEXT,
    source_needs_review INTEGER NOT NULL DEFAULT 0,

    -- The original source row as JSON: full provenance in one column, so normalization is
    -- auditable without duplicating every field into a raw_* twin.
    raw_record          TEXT,

    created_ts          TEXT NOT NULL,
    updated_ts          TEXT NOT NULL
);

-- Indexes mirror the blocking keys in app/dedupe/pipeline.py and the API filters.
CREATE INDEX IF NOT EXISTS idx_leads_email        ON leads (email);
CREATE INDEX IF NOT EXISTS idx_leads_phone_last9  ON leads (phone_last9);
CREATE INDEX IF NOT EXISTS idx_leads_domain_family ON leads (email_domain, family_key);
CREATE INDEX IF NOT EXISTS idx_leads_family_country ON leads (family_key, country);
CREATE INDEX IF NOT EXISTS idx_leads_status       ON leads (status);
CREATE INDEX IF NOT EXISTS idx_leads_owner        ON leads (owner);
CREATE INDEX IF NOT EXISTS idx_leads_country      ON leads (country);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open a connection with dict-like rows and foreign keys enabled."""
    target = Path(path) if path else config.DB_PATH
    if str(target) != ":memory:":
        target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def is_populated(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT COUNT(*) AS n FROM leads").fetchone()
    return bool(row["n"])
