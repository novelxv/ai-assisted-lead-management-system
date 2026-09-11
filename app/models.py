"""Internal record type and the API request/response schemas.

`Lead` is the shape every internal component agrees on (repository, dedupe, ingest). The
Pydantic models are the external contract and do the validation, so route handlers never
hand-roll checks.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app import config, normalize as nz

# --------------------------------------------------------------------------------------
# Internal record
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Lead:
    """One lead in its normalized form, plus the source row it came from."""

    id: str
    display_name: str
    given_name: str
    family_name: str
    family_key: str
    company: str
    company_norm: str
    job_title: str
    email: str
    email_local: str
    email_domain: str
    phone: str
    phone_digits: str
    phone_last9: str
    country: str
    status: str | None
    lifecycle_stage: str
    owner: str
    lead_score: int | None
    created_at: str | None
    last_modified_at: str | None
    notes: str
    source_channel: str | None
    source_detail: str | None
    source_confidence: str | None
    source_method: str | None
    source_needs_review: bool
    raw_record: dict[str, Any] | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Lead":
        raw = row["raw_record"] if "raw_record" in row.keys() else None
        return cls(
            id=row["id"],
            display_name=row["display_name"],
            given_name=row["given_name"],
            family_name=row["family_name"],
            family_key=row["family_key"],
            company=row["company"],
            company_norm=row["company_norm"],
            job_title=row["job_title"],
            email=row["email"],
            email_local=row["email_local"],
            email_domain=row["email_domain"],
            phone=row["phone"],
            phone_digits=row["phone_digits"],
            phone_last9=row["phone_last9"],
            country=row["country"],
            status=row["status"],
            lifecycle_stage=row["lifecycle_stage"],
            owner=row["owner"],
            lead_score=row["lead_score"],
            created_at=row["created_at"],
            last_modified_at=row["last_modified_at"],
            notes=row["notes"],
            source_channel=row["source_channel"],
            source_detail=row["source_detail"],
            source_confidence=row["source_confidence"],
            source_method=row["source_method"],
            source_needs_review=bool(row["source_needs_review"]),
            raw_record=json.loads(raw) if raw else None,
        )

    def summary(self) -> str:
        """One-line description, used in dedupe explanations and LLM prompts."""
        parts = [self.display_name or "(no name)", self.company or "(no company)", self.email, self.phone]
        return " | ".join(p for p in parts if p)

    def stored_source(self):
        """The saved extraction, rebuilt so it can be compared against a fresh one.

        Both PATCH and ingest need this to honour the first-touch rule, so it lives here
        rather than being reimplemented at each call site.
        """
        from app.source_extraction import SourceExtraction

        if self.source_channel is None:
            return None
        return SourceExtraction(
            channel=self.source_channel,
            detail=self.source_detail,
            confidence=self.source_confidence or config.SOURCE_CONFIDENCE_LOW,
            method=self.source_method or "unknown",
            evidence=None,
            needs_review=self.source_needs_review,
        )


# --------------------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------------------


class LeadOut(BaseModel):
    """A lead as returned by the API. `display_name` is the name exactly as supplied."""

    id: str
    display_name: str
    given_name: str
    family_name: str
    job_title: str
    company: str
    email: str
    phone: str
    country: str
    status: str | None
    lifecycle_stage: str
    owner: str
    lead_score: int | None
    created_at: str | None
    last_modified_at: str | None
    notes: str
    source_channel: str | None
    source_detail: str | None
    source_confidence: str | None
    source_method: str | None
    source_needs_review: bool

    @classmethod
    def from_lead(cls, lead: Lead) -> "LeadOut":
        return cls(**{field: getattr(lead, field) for field in cls.model_fields})


class LeadDetail(LeadOut):
    """Detail view. Adds the untouched source row so normalization is auditable."""

    raw_record: dict[str, Any] | None = None

    @classmethod
    def from_lead(cls, lead: Lead) -> "LeadDetail":
        data = {field: getattr(lead, field) for field in LeadOut.model_fields}
        return cls(**data, raw_record=lead.raw_record)


class LeadListResponse(BaseModel):
    items: list[LeadOut]
    total: int
    limit: int
    offset: int


# --------------------------------------------------------------------------------------
# PATCH
# --------------------------------------------------------------------------------------


class LeadPatch(BaseModel):
    """Only the three fields the brief allows to be edited.

    `extra="forbid"` turns a typo'd or unsupported field into a 422 rather than a silently
    ignored write, which matters when the caller believes it changed something.
    """

    model_config = ConfigDict(extra="forbid")

    status: str | None = None
    owner: str | None = None
    notes: str | None = None

    @field_validator("status")
    @classmethod
    def _canonical_status(cls, value: str | None) -> str | None:
        if value is None:
            return None
        canonical = nz.normalize_status(value)
        if canonical is None:
            raise ValueError(
                f"status must be one of {list(config.LEAD_STATUSES)} (case-insensitive)"
            )
        return canonical

    @field_validator("owner")
    @classmethod
    def _clean_owner(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = nz.collapse_ws(value)
        if not cleaned:
            raise ValueError("owner must not be blank")
        return cleaned

    @model_validator(mode="after")
    def _at_least_one_field(self) -> "LeadPatch":
        if self.status is None and self.owner is None and self.notes is None:
            raise ValueError("provide at least one of: status, owner, notes")
        return self


# --------------------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------------------


class FormSubmission(BaseModel):
    """A website form submission, shaped like an entry of website_form_submissions.json.

    Only name/email are required: a real form can omit the rest, and the matching policy is
    built to cope with partial records rather than reject them.
    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    email: str = Field(min_length=3)
    phone: str = ""
    company: str = ""
    country: str = ""
    message: str = ""
    form_id: str = ""
    form_name: str = ""
    page_url: str = ""
    submitted_at: str = ""

    @field_validator("email")
    @classmethod
    def _valid_email(cls, value: str) -> str:
        normalized = nz.normalize_email(value)
        if not nz.is_valid_email(normalized):
            raise ValueError("email must be a valid address")
        return normalized

    @field_validator("name", "company", "country", "message", "form_id", "form_name", "page_url")
    @classmethod
    def _trim(cls, value: str) -> str:
        return nz.collapse_ws(value)

    @model_validator(mode="after")
    def _name_not_blank(self) -> "FormSubmission":
        if not self.name:
            raise ValueError("name must not be blank")
        return self


class DuplicateRef(BaseModel):
    """A lead the incoming submission might be, but not confidently enough to merge into."""

    lead_id: str
    display_name: str
    score: int
    confidence: str
    reasons: list[str]


class IngestResult(BaseModel):
    action: Literal["created", "updated"]
    lead_id: str
    matched_by: str | None = None
    score: int | None = None
    confidence: str | None = None
    reasons: list[str] = Field(default_factory=list)
    changed_fields: list[str] = Field(default_factory=list)
    possible_duplicates: list[DuplicateRef] = Field(default_factory=list)
    lead: LeadDetail


# --------------------------------------------------------------------------------------
# Deduplication
# --------------------------------------------------------------------------------------


class DedupeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_score: int | None = None
    include_review: bool = True
    limit: Annotated[int, Field(ge=1, le=500)] = 50


class PairOut(BaseModel):
    lead_ids: tuple[str, str]
    summaries: tuple[str, str]
    score: int
    confidence: str
    reasons: list[str]
    adjudication: str | None = None


class GroupOut(BaseModel):
    lead_ids: list[str]
    summaries: list[str]
    score: int
    confidence: str
    reasons: list[str]
    has_internal_conflict: bool = False


class DedupeResponse(BaseModel):
    groups: list[GroupOut]
    review_pairs: list[PairOut]
    stats: dict[str, Any]


# --------------------------------------------------------------------------------------
# Source extraction
# --------------------------------------------------------------------------------------


class SourceExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = ""


class SourceExtractResponse(BaseModel):
    # Typed against the taxonomy, so an out-of-contract channel can never leave the API.
    channel: config.SourceChannel
    detail: str | None
    confidence: str
    method: str
    evidence: str | None
    needs_review: bool


# --------------------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------------------


class DashboardResponse(BaseModel):
    total_leads: int
    by_status: dict[str, int]
    by_source_channel: dict[str, int]
    needs_source_review: int
