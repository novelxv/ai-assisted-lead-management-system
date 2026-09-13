"""Concurrent read smoke test against a running server.

    uvicorn app.main:app --port 8000
    python -m scripts.concurrency_smoke http://127.0.0.1:8000 24 15

Hits /dashboard and /leads from several threads at once and reports any non-200 or
malformed response. Exits non-zero if any request fails.

This exists because route handlers are sync `def`, so Starlette runs them in a threadpool
and several can be in flight at once. TestClient exercises that too (tests/test_concurrency.py),
but this runs the same workload through a real Uvicorn server and HTTP stack. Stdlib only.
"""

import collections
import concurrent.futures
import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 12

PATHS = [
    "/dashboard",
    "/leads?limit=50",
    "/leads?status=Qualified&limit=25",
    "/leads?q=asante&limit=25",
    "/dashboard",
    "/leads?country=Singapore&limit=25&offset=25",
]


def fetch(path):
    url = BASE + path
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
            if path.startswith("/dashboard"):
                ok = isinstance(body.get("total_leads"), int) and body["total_leads"] > 0
            else:
                ok = isinstance(body.get("total"), int) and isinstance(body.get("items"), list)
            return (response.status, path, ok, None)
    except urllib.error.HTTPError as exc:
        return (exc.code, path, False, exc.read().decode("utf-8", "replace")[:200])
    except Exception as exc:  # noqa: BLE001
        return ("EXC", path, False, f"{type(exc).__name__}: {exc}")


def main():
    tasks = [p for _ in range(ROUNDS) for p in PATHS]
    codes = collections.Counter()
    bad = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for status, path, ok, detail in pool.map(fetch, tasks):
            codes[status] += 1
            if status != 200 or not ok:
                bad.append((status, path, detail))

    print(f"requests={len(tasks)} workers={WORKERS}")
    print(f"status codes: {dict(codes)}")
    print(f"failed or malformed: {len(bad)}")
    for status, path, detail in bad[:8]:
        print(f"  {status}  {path}  {detail}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
