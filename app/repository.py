"""Query layer.

All SQL lives here so route handlers stay about HTTP and the matching code stays about
matching. Every value is bound as a parameter; nothing is interpolated into SQL.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app import config, normalize as nz
from app.models import Lead

# Deterministic order for every list, export and page: newest first, ties broken by id so
# pagination can never repeat or drop a row.
ORDER_BY = "ORDER BY created_at IS NULL, created_at DESC, id ASC"


@dataclass(frozen=True)
class LeadFilters:
    """The four filters the brief specifies. Empty values mean "no constraint"."""

    status: str | None = None
    owner: str | None = None
    country: str | None = None
    q: str | None = None

    def where(self) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        if self.status:
            clauses.append("status = ?")
            params.append(self.status)
        if self.owner:
            # Owner names carry stray whitespace and casing in the export; compare on the
            # normalized form so "marcus wong" and "Marcus Wong " both match.
            clauses.append("LOWER(owner) = ?")
            params.append(nz.collapse_ws(self.owner).lower())
        if self.country:
            clauses.append("LOWER(country) = ?")
            params.append(nz.collapse_ws(self.country).lower())
        if self.q:
            # Free-text across name, company and email, per the brief. Phone is
            # deliberately excluded: it is not in the specified search surface.
            needle = f"%{nz.collapse_ws(self.q).lower()}%"
            clauses.append(
                "(LOWER(display_name) LIKE ? OR LOWER(company) LIKE ? OR LOWER(email) LIKE ?)"
            )
            params.extend([needle, needle, needle])

        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def count_leads(conn: sqlite3.Connection, filters: LeadFilters) -> int:
    where, params = filters.where()
    return conn.execute(f"SELECT COUNT(*) AS n FROM leads{where}", params).fetchone()["n"]


def list_leads(
    conn: sqlite3.Connection, filters: LeadFilters, limit: int, offset: int
) -> list[Lead]:
    where, params = filters.where()
    rows = conn.execute(
        f"SELECT * FROM leads{where} {ORDER_BY} LIMIT ? OFFSET ?", [*params, limit, offset]
    ).fetchall()
    return [Lead.from_row(row) for row in rows]


def list_all_filtered(conn: sqlite3.Connection, filters: LeadFilters) -> list[Lead]:
    """The whole filtered view, unpaginated — what the CSV export writes."""
    where, params = filters.where()
    rows = conn.execute(f"SELECT * FROM leads{where} {ORDER_BY}", params).fetchall()
    return [Lead.from_row(row) for row in rows]


def get_lead(conn: sqlite3.Connection, lead_id: str) -> Lead | None:
    row = conn.execute("SELECT * FROM leads WHERE id = ?", (lead_id,)).fetchone()
    return Lead.from_row(row) if row else None


def all_leads(conn: sqlite3.Connection) -> list[Lead]:
    """Every lead, for the deduplication pass."""
    rows = conn.execute("SELECT * FROM leads ORDER BY id ASC").fetchall()
    return [Lead.from_row(row) for row in rows]


def find_by_email(conn: sqlite3.Connection, email: str) -> list[Lead]:
    rows = conn.execute("SELECT * FROM leads WHERE email = ?", (email,)).fetchall()
    return [Lead.from_row(row) for row in rows]


def find_by_phone_last9(conn: sqlite3.Connection, last9: str) -> list[Lead]:
    if not last9:
        return []
    rows = conn.execute("SELECT * FROM leads WHERE phone_last9 = ?", (last9,)).fetchall()
    return [Lead.from_row(row) for row in rows]


def find_by_domain_and_family(
    conn: sqlite3.Connection, domain: str, family_key: str
) -> list[Lead]:
    if not domain or not family_key:
        return []
    rows = conn.execute(
        "SELECT * FROM leads WHERE email_domain = ? AND family_key = ?", (domain, family_key)
    ).fetchall()
    return [Lead.from_row(row) for row in rows]


def find_by_family_and_country(
    conn: sqlite3.Connection, family_key: str, country: str
) -> list[Lead]:
    if not family_key:
        return []
    rows = conn.execute(
        "SELECT * FROM leads WHERE family_key = ? AND LOWER(country) = ?",
        (family_key, country.lower()),
    ).fetchall()
    return [Lead.from_row(row) for row in rows]


def _column_names(conn: sqlite3.Connection) -> set[str]:
    return {row["name"] for row in conn.execute("PRAGMA table_info(leads)")}


def update_fields(
    conn: sqlite3.Connection, lead_id: str, changes: dict[str, Any]
) -> Lead | None:
    """Apply a column-level update. Callers decide the policy; this just writes."""
    if not changes:
        return get_lead(conn, lead_id)
    # Column names are the one part of the statement that cannot be a bound parameter, so
    # they are checked against the real schema. Today every caller passes internal
    # constants; this makes that a guarantee rather than a convention.
    unknown = set(changes) - _column_names(conn)
    if unknown:
        raise ValueError(f"unknown lead columns: {sorted(unknown)}")
    payload = dict(changes)
    payload["updated_ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    assignments = ", ".join(f"{column} = :{column}" for column in payload)
    conn.execute(
        f"UPDATE leads SET {assignments} WHERE id = :lead_id", {**payload, "lead_id": lead_id}
    )
    conn.commit()
    return get_lead(conn, lead_id)


def insert_lead(conn: sqlite3.Connection, params: dict[str, Any]) -> None:
    from app.loader import INSERT_SQL  # imported here to avoid a circular import at module load

    conn.execute(INSERT_SQL, params)
    conn.commit()


def next_lead_id(conn: sqlite3.Connection) -> str:
    """Allocate the next id.

    Ids come from the source system's `Record ID` and are numeric, so new leads continue the
    sequence. That keeps ids stable, human-readable and sortable; a UUID would be safer
    against concurrent writers but this is a single-process service and the brief has no
    multi-writer requirement.
    """
    row = conn.execute("SELECT MAX(CAST(id AS INTEGER)) AS m FROM leads").fetchone()
    return str((row["m"] or 100_000_000) + 1)


def dashboard_counts(conn: sqlite3.Connection) -> dict[str, Any]:
    """Counts by status and by extracted source channel.

    Channel counts come from the extracted source, not the `Original Source` column: that
    column is blank for half the rows and misleading on others (booth conversations tagged
    "Other Campaigns"), so counting it would produce a confidently wrong dashboard.
    """
    by_status = {
        (row["status"] or "Unknown"): row["n"]
        for row in conn.execute(
            "SELECT status, COUNT(*) AS n FROM leads GROUP BY status ORDER BY n DESC"
        )
    }
    by_channel = {
        (row["source_channel"] or "Unknown"): row["n"]
        for row in conn.execute(
            "SELECT source_channel, COUNT(*) AS n FROM leads GROUP BY source_channel ORDER BY n DESC"
        )
    }
    total = conn.execute("SELECT COUNT(*) AS n FROM leads").fetchone()["n"]
    review = conn.execute(
        "SELECT COUNT(*) AS n FROM leads WHERE source_needs_review = 1"
    ).fetchone()["n"]
    return {
        "total_leads": total,
        "by_status": by_status,
        "by_source_channel": by_channel,
        "needs_source_review": review,
    }


def export_rows(leads: list[Lead]) -> list[dict[str, Any]]:
    """Flatten leads into the stable export column order."""
    rows = []
    for lead in leads:
        rows.append(
            {
                column: (
                    "" if getattr(lead, column, None) is None else getattr(lead, column)
                )
                for column in config.EXPORT_COLUMNS
            }
        )
    return rows
