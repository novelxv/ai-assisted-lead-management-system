"""Shared fixtures.

Tests draw on the real dataset wherever a real case exists, and fall back to synthetic
records only for risks the provided data does not contain (a shared phone line, a duplicate
whose phone changed, a bridged cluster). Both kinds matter: real cases keep the system
honest about the input it was built for, synthetic ones probe where it is most likely wrong.
"""

from __future__ import annotations

import csv
import json

import pytest

from app import config


@pytest.fixture(scope="session")
def seed_rows() -> list[dict[str, str]]:
    """Every row of data/leads_seed.csv, unmodified."""
    with config.SEED_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="session")
def submissions() -> list[dict[str, str]]:
    """Every entry of data/website_form_submissions.json, unmodified."""
    return json.loads(config.SUBMISSIONS_JSON.read_text(encoding="utf-8"))
