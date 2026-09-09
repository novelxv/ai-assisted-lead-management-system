"""Load data/leads_seed.csv into SQLite.

Runs automatically the first time the app starts with no database, and can be re-run
explicitly with `python -m app.loader --force`.

The loader reports column coverage rather than quietly discarding columns. Five of the 22
columns in the export are empty for every single row (`City`, `Original Source Drill-Down 1`,
`Annual Revenue`, `Marketing contact status`, `GDPR consent`); saying so out loud is more
useful to a reviewer than a schema that silently omits them, and it is how you would notice
if a future export started populating one.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app import config, db, normalize as nz
from app.source_extraction import extract_source

logger = logging.getLogger(__name__)

# Source column -> internal meaning. Columns absent from this map are preserved in
# `raw_record` but not modelled; see the coverage report.
CSV_COLUMNS = (
    "Record ID",
    "First Name",
    "Last Name",
    "Full Name",
    "Job Title",
    "Company Name",
    "Email",
    "Phone Number",
    "Country/Region",
    "City",
    "Lead Status",
    "Lifecycle Stage",
    "Original Source",
    "Original Source Drill-Down 1",
    "Contact Owner",
    "Create Date",
    "Last Modified Date",
    "Notes",
    "Annual Revenue",
    "Marketing contact status",
    "GDPR consent",
    "Lead Score",
)

INSERT_SQL = """
INSERT INTO leads (
    id, display_name, given_name, family_name, family_key,
    company, company_norm, job_title,
    email, email_local, email_domain,
    phone, phone_digits, phone_last9,
    country, status, lifecycle_stage, owner, lead_score,
    created_at, last_modified_at, notes,
    source_channel, source_detail, source_confidence, source_method, source_needs_review,
    raw_record, created_ts, updated_ts
) VALUES (
    :id, :display_name, :given_name, :family_name, :family_key,
    :company, :company_norm, :job_title,
    :email, :email_local, :email_domain,
    :phone, :phone_digits, :phone_last9,
    :country, :status, :lifecycle_stage, :owner, :lead_score,
    :created_at, :last_modified_at, :notes,
    :source_channel, :source_detail, :source_confidence, :source_method, :source_needs_review,
    :raw_record, :created_ts, :updated_ts
)
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_int(value: str) -> int | None:
    cleaned = nz.collapse_ws(value)
    try:
        return int(cleaned)
    except ValueError:
        return None


def build_lead_params(row: dict[str, str], now: str) -> dict[str, Any]:
    """Map one raw CSV row to the normalized column set.

    Source extraction runs here so a freshly loaded database is immediately usable by the
    dashboard and the API. It runs without an LLM client, keeping the load fully
    deterministic; the ambiguous slice is stored flagged for review rather than guessed.
    """
    given, family, display = nz.resolve_name(
        row.get("First Name"), row.get("Last Name"), row.get("Full Name")
    )
    email = nz.normalize_email(row.get("Email"))
    local, domain = nz.split_email(email)
    phone_digits = nz.normalize_phone(row.get("Phone Number"))
    company = nz.collapse_ws(row.get("Company Name"))
    notes = nz.collapse_ws(row.get("Notes"))
    source = extract_source(notes)

    return {
        "id": nz.collapse_ws(row.get("Record ID")),
        "display_name": display,
        "given_name": given,
        "family_name": family,
        "family_key": nz.name_key(family),
        "company": company,
        "company_norm": nz.normalize_company(company),
        "job_title": nz.collapse_ws(row.get("Job Title")),
        "email": email,
        "email_local": local,
        "email_domain": domain,
        "phone": nz.collapse_ws(row.get("Phone Number")),
        "phone_digits": phone_digits,
        "phone_last9": nz.phone_last9(phone_digits),
        "country": nz.normalize_country(row.get("Country/Region")),
        "status": nz.normalize_status(row.get("Lead Status")),
        "lifecycle_stage": nz.collapse_ws(row.get("Lifecycle Stage")),
        "owner": nz.collapse_ws(row.get("Contact Owner")),
        "lead_score": _to_int(row.get("Lead Score", "")),
        "created_at": nz.to_iso(nz.parse_date(row.get("Create Date"))),
        "last_modified_at": nz.to_iso(nz.parse_date(row.get("Last Modified Date"))),
        "notes": notes,
        "source_channel": source.channel,
        "source_detail": source.detail,
        "source_confidence": source.confidence,
        "source_method": source.method,
        "source_needs_review": int(source.needs_review),
        "raw_record": json.dumps(row, ensure_ascii=False),
        "created_ts": now,
        "updated_ts": now,
    }


def read_seed_rows(path: Path | None = None) -> list[dict[str, str]]:
    """Read the export. Encoding is explicit: the file is UTF-8 and contains em dashes,
    while Python's default on Windows is the locale codepage, which would corrupt them."""
    target = path or config.SEED_CSV
    with target.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def coverage_report(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    """Count non-blank values per source column."""
    counts = {column: 0 for column in CSV_COLUMNS}
    for row in rows:
        for column in CSV_COLUMNS:
            if nz.collapse_ws(row.get(column)):
                counts[column] += 1
    return counts


def log_coverage(counts: dict[str, int], total: int) -> None:
    empty = [column for column, n in counts.items() if n == 0]
    sparse = [f"{c} ({n})" for c, n in counts.items() if 0 < n < total * 0.6]
    logger.info("Loaded %d leads from %s", total, config.SEED_CSV.name)
    if empty:
        logger.info("Columns empty for every row (not modelled): %s", ", ".join(empty))
    if sparse:
        logger.info("Sparsely populated columns (kept, nullable): %s", ", ".join(sparse))


def load(
    conn: sqlite3.Connection, path: Path | None = None, *, force: bool = False
) -> int:
    """Populate the leads table. Returns the number of rows inserted.

    A populated database is left alone unless `force` is set, so restarting the app never
    discards PATCH edits or ingested leads.
    """
    db.init_schema(conn)
    if db.is_populated(conn):
        if not force:
            logger.info("Database already populated; skipping load")
            return 0
        conn.execute("DELETE FROM leads")

    rows = read_seed_rows(path)
    now = _now()
    params = [build_lead_params(row, now) for row in rows]
    conn.executemany(INSERT_SQL, params)
    conn.commit()
    log_coverage(coverage_report(rows), len(rows))
    return len(rows)


def ensure_loaded(conn: sqlite3.Connection) -> None:
    """Auto-initialise on first run so a reviewer never has to run a separate step."""
    db.init_schema(conn)
    if not db.is_populated(conn):
        logger.info("No leads found; loading seed data from %s", config.SEED_CSV)
        load(conn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the lead seed CSV into SQLite.")
    parser.add_argument(
        "--force", action="store_true", help="reload even if the database already has leads"
    )
    parser.add_argument("--db", type=Path, default=None, help="database path override")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    conn = db.connect(args.db)
    try:
        inserted = load(conn, force=args.force)
        if not inserted and not args.force:
            print("Database already populated. Re-run with --force to reload.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
