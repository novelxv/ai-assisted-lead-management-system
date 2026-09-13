"""Console route tests.

Deliberately thin. The console is a thin client over the JSON API, so the API tests remain
the correctness suite; what is worth pinning here is that the page and its assets are served,
that the shell references what it loads, and that adding the UI did not disturb the API
surface it sits on top of.
"""

from __future__ import annotations

import pytest


def test_console_is_served_at_the_root(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>" in response.text


@pytest.mark.parametrize(
    "path,content_type",
    [("/static/styles.css", "text/css"), ("/static/app.js", "javascript")],
)
def test_static_assets_resolve(client, path: str, content_type: str) -> None:
    response = client.get(path)
    assert response.status_code == 200
    assert content_type in response.headers["content-type"]


def test_the_shell_references_the_assets_it_loads(client) -> None:
    """A renamed asset would otherwise fail silently in the browser, not in the suite."""
    page = client.get("/").text
    for asset in ("/static/styles.css", "/static/app.js"):
        assert asset in page
        assert client.get(asset).status_code == 200


def test_the_console_links_to_the_api_documentation(client) -> None:
    assert '"/docs"' in client.get("/").text


def test_api_docs_remain_available(client) -> None:
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_the_console_route_is_not_part_of_the_api_contract(client) -> None:
    """The UI is not an API endpoint, so it stays out of the OpenAPI schema."""
    schema = client.get("/openapi.json").json()
    assert "/" not in schema["paths"]
    assert "/static" not in schema["paths"]


def test_adding_the_ui_did_not_shadow_any_api_route(client) -> None:
    """`/` and the `/static` mount sit alongside the API, not in front of it."""
    schema = client.get("/openapi.json").json()["paths"]
    for path in ("/leads", "/leads/{lead_id}", "/leads/export", "/dashboard", "/health"):
        assert path in schema
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/leads", params={"limit": 1}).status_code == 200


def test_an_unknown_static_asset_is_a_404_not_the_shell(client) -> None:
    """A mount that fell through to the SPA shell would turn typos into silent blank pages."""
    assert client.get("/static/does-not-exist.js").status_code == 404
