"""Operators-view auth: unconfigured must 404, wrong token must 401.

The page shows endpoint URLs, client addresses and failure rates, so the gate
in front of it is a security boundary and gets pinned like one. The database
query behind /admin/state needs a database and isn't covered here; what is
covered is every path by which someone reaches it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import HTTPException  # noqa: E402

from app import admin  # noqa: E402
from app.config import settings  # noqa: E402

FAILURES = []


def check(name: str, got, want) -> None:
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL  {name}  (got {got!r}, want {want!r})")
    else:
        print(f"  ok    {name}")


class FakeRequest:
    """Only what _require_admin touches."""
    def __init__(self, headers=None, query=None):
        self.headers = headers or {}
        self.query_params = query or {}


def status_of(request) -> int:
    """The HTTP status _require_admin raises, or 200 if it allows through."""
    try:
        admin._require_admin(request)
        return 200
    except HTTPException as e:
        return e.status_code


original = settings.admin_token
try:
    print("\nunconfigured: the routes must not exist")
    settings.admin_token = ""
    # 404 rather than 401 on purpose: an operator who never set ADMIN_TOKEN
    # should look like a gateway without the feature, not one hiding a login.
    check("no token set -> 404 even with a guess",
          status_of(FakeRequest(query={"token": "guess"})), 404)
    check("no token set -> 404 with nothing",
          status_of(FakeRequest()), 404)

    print("\nconfigured: only the right password gets in")
    settings.admin_token = "s3cret-operators-token"
    check("even a correct token in the URL is rejected",
          status_of(FakeRequest(query={"token": "s3cret-operators-token"})), 401)
    check("correct token in header (curl/scripts)",
          status_of(FakeRequest(headers={"x-common-admin-token": "s3cret-operators-token"})), 200)
    check("wrong token -> 401",
          status_of(FakeRequest(query={"token": "wrong"})), 401)
    check("missing token -> 401",
          status_of(FakeRequest()), 401)
    check("empty token -> 401",
          status_of(FakeRequest(query={"token": ""})), 401)
    # A prefix of the real password must not pass; compare_digest is length-safe
    # but a naive startswith() refactor would break exactly here.
    check("prefix of the real token -> 401",
          status_of(FakeRequest(query={"token": "s3cret"})), 401)
    check("header wins nothing when wrong -> 401",
          status_of(FakeRequest(headers={"x-common-admin-token": "nope"})), 401)
finally:
    settings.admin_token = original

print("\nshipped default is off")
check("admin_token defaults to empty (routes 404 until set)", original, "")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED\n")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all operators-view auth tests passed")
