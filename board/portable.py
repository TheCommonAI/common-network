#!/usr/bin/env python3
"""Run the Common message board with Python only: one SQLite file, a password.

    python3 portable.py --password YOUR_PASSWORD
    # then open http://127.0.0.1:8080/admin-mes/

Same page as the Netlify version. Messages are kept in board.db next to this
file. Standard library only (Python 3.8+).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
PUBLIC = HERE / "public"
NAME_MAX = 32
BODY_MAX = 500
PAYLOAD_MAX = 4096
LIST_DEFAULT = 100
LIST_MAX = 200
DEFAULT_NAME = "anon"
REALM = 'Basic realm="common message board", charset="UTF-8"'
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; "
    "img-src 'self' data:; connect-src 'self'; form-action 'self'; "
    "base-uri 'none'; frame-ancestors 'none'"
)
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
}


def clean(value: object, limit: int, *, required: bool) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ValueError("must be text")
    text = value.strip()
    if "\x00" in text:
        raise ValueError("that character is not allowed")
    if required and not text:
        raise ValueError("write something first")
    if len(text) > limit:
        raise ValueError(f"keep it under {limit} characters")
    return text


class Board:
    def __init__(self, path: str | os.PathLike = ":memory:") -> None:
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " name TEXT NOT NULL,"
                " body TEXT NOT NULL,"
                " created_at TEXT NOT NULL)"
            )
            self._db.commit()

    def post(self, name: object, body: object) -> dict:
        name = clean(name, NAME_MAX, required=False) or DEFAULT_NAME
        body = clean(body, BODY_MAX, required=True)
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO messages (name, body, created_at) VALUES (?, ?, ?)",
                (name, body, created),
            )
            self._db.commit()
            msg_id = cur.lastrowid
        return {"id": msg_id, "name": name, "body": body, "created_at": created}

    def latest(self, limit: int = LIST_DEFAULT) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, name, body, created_at FROM messages ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"id": i, "name": n, "body": b, "created_at": c}
            for i, n, b, c in rows
        ]


def is_authorized(header: str | None, expected: str) -> bool:
    if not header or not header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(header[6:], validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return False
    separator = decoded.find(":")
    if separator < 0:
        return False
    got = decoded[separator + 1 :].encode("utf-8")
    want = expected.encode("utf-8")
    if len(got) != len(want):
        return False
    diff = 0
    for a, b in zip(got, want):
        diff |= a ^ b
    return diff == 0


def strip_prefix(path: str) -> str:
    if path == "/admin-mes":
        return "/"
    if path.startswith("/admin-mes/"):
        return path[len("/admin-mes") :] or "/"
    return path


class Handler(BaseHTTPRequestHandler):
    server_version = "common-board"
    board: Board
    password: str

    def do_HEAD(self) -> None:
        self._dispatch(send_body=False)

    def do_GET(self) -> None:
        self._dispatch(send_body=True)

    def do_POST(self) -> None:
        self._dispatch(send_body=True)

    def _dispatch(self, send_body: bool) -> None:
        if not is_authorized(self.headers.get("Authorization"), self.password):
            self._send(
                401,
                b"password required",
                "text/plain; charset=utf-8",
                extra={"WWW-Authenticate": REALM},
                send_body=send_body,
            )
            return

        parsed = urlparse(self.path)
        path = strip_prefix(parsed.path)

        if path == "/api/messages" and self.command in ("GET", "HEAD"):
            try:
                limit = int(parse_qs(parsed.query).get("limit", [LIST_DEFAULT])[0])
            except ValueError:
                self._json(400, {"error": "limit must be a number"}, send_body=send_body)
                return
            limit = max(1, min(limit, LIST_MAX))
            self._json(200, {"messages": self.board.latest(limit)}, send_body=send_body)
            return

        if path == "/api/messages" and self.command == "POST":
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._json(400, {"error": "bad content length"}, send_body=send_body)
                return
            if length <= 0 or length > PAYLOAD_MAX:
                status = 413 if length > PAYLOAD_MAX else 400
                error = "message is too large" if length > PAYLOAD_MAX else "bad content length"
                self._json(status, {"error": error}, send_body=send_body)
                return
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                self._json(400, {"error": "send json"}, send_body=send_body)
                return
            if not isinstance(data, dict):
                self._json(400, {"error": "send an object with name and body"}, send_body=send_body)
                return
            try:
                msg = self.board.post(data.get("name"), data.get("body"))
            except ValueError as exc:
                self._json(400, {"error": str(exc)}, send_body=send_body)
                return
            self._json(201, {"message": msg}, send_body=send_body)
            return

        if path == "/health":
            self._json(200, {"ok": True}, send_body=send_body)
            return

        if self.command in ("GET", "HEAD") and path in STATIC:
            name, ctype = STATIC[path]
            self._send(200, (PUBLIC / name).read_bytes(), ctype, send_body=send_body)
            return

        if self.command == "POST":
            self._json(404, {"error": "not found"}, send_body=send_body)
            return
        self._json(404, {"error": "not found"}, send_body=send_body)

    def _json(self, status: int, payload: dict, send_body: bool = True) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json", cache="no-store", send_body=send_body)

    def _send(
        self,
        status: int,
        body: bytes,
        ctype: str,
        cache: str = "public, max-age=60",
        extra: dict | None = None,
        send_body: bool = True,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        if extra:
            for key, value in extra.items():
                self.send_header(key, value)
        self.end_headers()
        if send_body and self.command != "HEAD":
            self.wfile.write(body)


def make_server(board: Board, password: str, host: str, port: int) -> ThreadingHTTPServer:
    handler = type("BoardHandler", (Handler,), {"board": board, "password": password})
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Common message board (Python, no Netlify).")
    parser.add_argument("--host", default=os.environ.get("BOARD_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument("--db", default=os.environ.get("BOARD_DB", str(HERE / "board.db")))
    parser.add_argument("--password", default=os.environ.get("BOARD_PASSWORD", ""))
    args = parser.parse_args()
    if not args.password:
        raise SystemExit("set --password or BOARD_PASSWORD")

    board = Board(args.db)
    server = make_server(board, args.password, args.host, args.port)
    print(f"message board on http://{args.host}:{server.server_port}/admin-mes/  (db: {args.db})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
