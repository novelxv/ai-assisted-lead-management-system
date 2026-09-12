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
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import config, db, loader
from app.main import app, get_conn


@pytest.fixture(autouse=True, scope="session")
def _isolate_llm_cache(tmp_path_factory):
    """Point the LLM response cache at a throwaway path for the whole test session.

    Without this, a test that builds a real client with the default cache writes into the
    developer's `.cache/llm_cache.json` and then reads its own answer back on the next run —
    which is exactly how a test that mocks the network can still end up depending on it.
    """
    original = config.LLM_CACHE_PATH
    config.LLM_CACHE_PATH = tmp_path_factory.mktemp("llm_cache") / "cache.json"
    yield
    config.LLM_CACHE_PATH = original


class FakeLLM:
    """An explicit test double for the LLM tier.

    It fabricates nothing: it returns whatever a test tells it to, and records the prompts
    it was given so a test can assert the tier was *not* reached, or inspect what the model
    would have been shown. Real model responses are never committed to this repo.
    """

    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []
        self.schemas: list[type] = []

    def complete_json(self, *, system: str, user: str, schema: type) -> dict[str, Any] | None:
        self.calls.append((system, user))
        self.schemas.append(schema)
        return self.response

    @property
    def last_user_prompt(self) -> str:
        return self.calls[-1][1]

    @property
    def last_system_prompt(self) -> str:
        return self.calls[-1][0]


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
