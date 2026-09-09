"""Shared fixtures.

Tests draw on the real dataset wherever a real case exists, and fall back to synthetic
records only for risks the provided data does not contain (a shared phone line, a duplicate
whose phone changed, a bridged cluster). Both kinds matter: real cases keep the system
honest about the input it was built for, synthetic ones probe where it is most likely wrong.
"""

from __future__ import annotations

import csv
import json
import shutil
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import config, db, loader
from app.main import app, get_conn


@pytest.fixture(scope="session")
def seed_rows() -> list[dict[str, str]]:
    """Every row of data/leads_seed.csv, unmodified."""
    with config.SEED_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="session")
def submissions() -> list[dict[str, str]]:
    """Every entry of data/website_form_submissions.json, unmodified."""
    return json.loads(config.SUBMISSIONS_JSON.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# Database and API fixtures
# --------------------------------------------------------------------------------------
# The full seed is loaded once per session, then copied per test for anything that writes.
# Tests therefore run against the real 2,049 rows without paying to reload them each time,
# and a mutating test can never leak state into another.


@pytest.fixture(scope="session")
def seed_db_file(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("seed") / "leads.db"
    conn = db.connect(path)
    loader.load(conn)
    conn.close()
    return path


@pytest.fixture(scope="session")
def seeded_conn(seed_db_file: Path) -> sqlite3.Connection:
    conn = db.connect(seed_db_file)
    yield conn
    conn.close()


@pytest.fixture
def client(seeded_conn: sqlite3.Connection) -> TestClient:
    """Read-only API client over the shared seeded database."""
    app.dependency_overrides[get_conn] = lambda: seeded_conn
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def mutable_client(seed_db_file: Path, tmp_path: Path) -> TestClient:
    """API client over a private copy, for tests that PATCH or ingest."""
    private = tmp_path / "leads.db"
    shutil.copy(seed_db_file, private)
    conn = db.connect(private)
    app.dependency_overrides[get_conn] = lambda: conn
    yield TestClient(app)
    app.dependency_overrides.clear()
    conn.close()
