"""FastAPI application.

Route handlers stay thin: validation is Pydantic's job (app/models.py), SQL is the
repository's (app/repository.py) and policy is the matching modules'. The interactive docs
at /docs are the intended UI — the brief does not ask for a frontend, so the effort went
into the API contract instead.
"""

from __future__ import annotations

import csv
import io
import logging
import sqlite3
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Response, status as http_status

from app import config, db, loader, normalize as nz, repository
from app.models import (
    DashboardResponse,
    LeadDetail,
    LeadListResponse,
    LeadOut,
    LeadPatch,
    SourceExtractRequest,
    SourceExtractResponse,
)
from app.repository import LeadFilters
from app.source_extraction import extract_source, preserve_confident_source

logger = logging.getLogger(__name__)

_connection: sqlite3.Connection | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the database and seed it on first run, so setup is `install` then `run`."""
    global _connection
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _connection = db.connect()
    loader.ensure_loaded(_connection)
    try:
        yield
    finally:
        _connection.close()
        _connection = None


app = FastAPI(
    title="Lead Management System",
    version="0.1.0",
    summary="Lead store with AI-assisted duplicate detection and source extraction.",
    lifespan=lifespan,
)


def get_conn() -> sqlite3.Connection:
    if _connection is None:  # pragma: no cover - only reachable outside the lifespan
        raise RuntimeError("database connection is not initialised")
    return _connection


Conn = Annotated[sqlite3.Connection, Depends(get_conn)]


def _validated_status(raw: str | None) -> str | None:
    """Accept any casing, reject anything outside the taxonomy with a helpful 422."""
    if raw is None or not raw.strip():
        return None
    canonical = nz.normalize_status(raw)
    if canonical is None:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unknown status {raw!r}. Allowed values (case-insensitive): "
                f"{list(config.LEAD_STATUSES)}"
            ),
        )
    return canonical


def _filters(status_: str | None, owner: str | None, country: str | None, q: str | None) -> LeadFilters:
    return LeadFilters(
        status=_validated_status(status_),
        owner=nz.collapse_ws(owner) or None,
        country=nz.collapse_ws(country) or None,
        q=nz.collapse_ws(q) or None,
    )


@app.get("/health", tags=["meta"])
def health(conn: Conn) -> dict[str, object]:
    return {"status": "ok", "leads": repository.count_leads(conn, LeadFilters())}


# --------------------------------------------------------------------------------------
# Leads
# --------------------------------------------------------------------------------------


@app.get("/leads", response_model=LeadListResponse, tags=["leads"])
def list_leads(
    conn: Conn,
    status: Annotated[str | None, Query(description="Lead status; case-insensitive")] = None,
    owner: Annotated[str | None, Query(description="Contact owner")] = None,
    country: Annotated[str | None, Query(description="Country/Region")] = None,
    q: Annotated[str | None, Query(description="Free text across name, company and email")] = None,
    limit: Annotated[int, Query(ge=1, le=config.MAX_PAGE_SIZE)] = config.DEFAULT_PAGE_SIZE,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> LeadListResponse:
    filters = _filters(status, owner, country, q)
    total = repository.count_leads(conn, filters)
    leads = repository.list_leads(conn, filters, limit, offset)
    return LeadListResponse(
        items=[LeadOut.from_lead(lead) for lead in leads],
        total=total,
        limit=limit,
        offset=offset,
    )


# Declared before /leads/{lead_id} so "export" is never read as an id.
@app.get("/leads/export", tags=["leads"], response_class=Response)
def export_leads(
    conn: Conn,
    status: Annotated[str | None, Query()] = None,
    owner: Annotated[str | None, Query()] = None,
    country: Annotated[str | None, Query()] = None,
    q: Annotated[str | None, Query()] = None,
) -> Response:
    """CSV of the current filtered view.

    Pagination is intentionally ignored: an export is for the whole filtered set, and a
    50-row file would be a surprising thing to hand someone who filtered to 300 leads.
    """
    filters = _filters(status, owner, country, q)
    leads = repository.list_all_filtered(conn, filters)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(config.EXPORT_COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(repository.export_rows(leads))

    return Response(
        content=buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="leads.csv"'},
    )


@app.get("/leads/{lead_id}", response_model=LeadDetail, tags=["leads"])
def get_lead(conn: Conn, lead_id: str) -> LeadDetail:
    lead = repository.get_lead(conn, lead_id)
    if lead is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND, detail=f"Lead {lead_id!r} not found"
        )
    return LeadDetail.from_lead(lead)


@app.patch("/leads/{lead_id}", response_model=LeadDetail, tags=["leads"])
def patch_lead(conn: Conn, lead_id: str, patch: LeadPatch) -> LeadDetail:
    """Update status, owner or notes.

    Editing notes re-runs source extraction, but only fills the source in when the stored
    one is unknown or flagged: a lead's original source is a first-touch fact, and a later
    note about a phone call should not rewrite the conference where we actually met them.
    """
    lead = repository.get_lead(conn, lead_id)
    if lead is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND, detail=f"Lead {lead_id!r} not found"
        )

    changes: dict[str, object] = {}
    if patch.status is not None:
        changes["status"] = patch.status
    if patch.owner is not None:
        changes["owner"] = patch.owner
    if patch.notes is not None:
        notes = nz.collapse_ws(patch.notes)
        changes["notes"] = notes
        stored = _stored_source(lead)
        chosen = preserve_confident_source(stored, extract_source(notes))
        if chosen is not stored:
            changes.update(
                {
                    "source_channel": chosen.channel,
                    "source_detail": chosen.detail,
                    "source_confidence": chosen.confidence,
                    "source_method": chosen.method,
                    "source_needs_review": int(chosen.needs_review),
                }
            )

    updated = repository.update_fields(conn, lead_id, changes)
    assert updated is not None  # the lead existed a moment ago and nothing deletes rows
    return LeadDetail.from_lead(updated)


def _stored_source(lead):
    """Rebuild the stored extraction so it can be compared with a fresh one."""
    from app.source_extraction import SourceExtraction

    if lead.source_channel is None:
        return None
    return SourceExtraction(
        channel=lead.source_channel,
        detail=lead.source_detail,
        confidence=lead.source_confidence or config.SOURCE_CONFIDENCE_LOW,
        method=lead.source_method or "unknown",
        evidence=None,
        needs_review=lead.source_needs_review,
    )


# --------------------------------------------------------------------------------------
# Source extraction
# --------------------------------------------------------------------------------------


@app.post("/source/extract", response_model=SourceExtractResponse, tags=["source"])
def extract(payload: SourceExtractRequest) -> SourceExtractResponse:
    """Classify arbitrary note text. Exposed so the extractor is directly demonstrable."""
    from app.llm import get_client

    result = extract_source(payload.text, client=get_client())
    return SourceExtractResponse(**vars(result))


# --------------------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------------------


@app.get("/dashboard", response_model=DashboardResponse, tags=["meta"])
def dashboard(conn: Conn) -> DashboardResponse:
    return DashboardResponse(**repository.dashboard_counts(conn))
