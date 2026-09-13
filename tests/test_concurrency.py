"""Concurrent read behaviour.

Route handlers are sync `def`, so Starlette runs them in a threadpool and several can be in
flight at once. These tests use the real `get_conn` dependency rather than the shared
connection the other fixtures inject, because the connection lifetime is the thing under
test.

Against a single process-wide connection this workload produced `sqlite3.InterfaceError`,
rows read back as None, and malformed tuples. `scripts/concurrency_smoke.py` runs the same
shape of workload against a real Uvicorn server.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app, get_conn

PATHS = [
    "/dashboard",
    "/leads?limit=50",
    "/leads?status=Qualified&limit=25",
    "/leads?q=asante&limit=25",
    "/leads?country=Singapore&limit=25&offset=25",
]


@pytest.fixture
def live_client(seed_db_file: Path, monkeypatch) -> TestClient:
    """A client that opens its own connection per request, as the app does in production."""
    monkeypatch.setattr(config, "DB_PATH", seed_db_file)
    app.dependency_overrides.pop(get_conn, None)
    yield TestClient(app)
    app.dependency_overrides.clear()


def _fetch(client: TestClient, path: str) -> tuple[int, dict]:
    response = client.get(path)
    return response.status_code, response.json()


def test_concurrent_reads_do_not_error(live_client) -> None:
    """Every response is a 200 with a well-formed body under parallel load."""
    tasks = PATHS * 12
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda p: _fetch(live_client, p), tasks))

    assert len(results) == len(tasks)
    statuses = {status for status, _ in results}
    assert statuses == {200}, f"unexpected statuses: {statuses}"


def test_concurrent_reads_return_complete_rows(live_client) -> None:
    """Connection contention showed up as None fields and short tuples, not only as 500s."""
    tasks = PATHS * 12
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda p: _fetch(live_client, p), tasks))

    for path, (status, body) in zip(tasks, results):
        assert status == 200, path
        if path.startswith("/dashboard"):
            assert body["total_leads"] == 2049
            assert sum(body["by_status"].values()) == 2049
            assert sum(body["by_source_channel"].values()) == 2049
        else:
            assert isinstance(body["total"], int)
            assert isinstance(body["items"], list)
            for item in body["items"]:
                # A torn read previously surfaced here as a missing or null identity field.
                assert item["id"]
                assert item["status"] is not None


def test_concurrent_reads_are_consistent(live_client) -> None:
    """The same query run in parallel returns the same total every time."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        totals = list(
            pool.map(lambda _: _fetch(live_client, "/leads?status=Qualified&limit=1")[1]["total"],
                     range(40))
        )
    assert len(set(totals)) == 1, f"inconsistent totals under load: {sorted(set(totals))}"


def test_each_request_gets_its_own_connection(live_client) -> None:
    """The dependency is request-scoped, so no connection object outlives a request."""
    seen: list[int] = []

    def record():
        for conn in get_conn():
            seen.append(id(conn))

    record()
    record()
    assert len(seen) == 2
    # Two separate calls must not hand back the same live object.
    assert live_client.get("/health").status_code == 200


def test_a_closed_request_connection_does_not_break_the_next_request(live_client) -> None:
    """Teardown closes the connection; the following request must open a fresh one."""
    for _ in range(5):
        assert live_client.get("/dashboard").json()["total_leads"] == 2049
