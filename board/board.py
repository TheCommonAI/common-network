#!/usr/bin/env python3
"""The Common message board: one page, one SQLite file, no accounts.

Run `python3 board/board.py` and open http://127.0.0.1:8080. Environment:
BOARD_DB (where messages are kept, default board/board.db), BOARD_HOST
(default 127.0.0.1) and PORT (default 8080). Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

HERE = Path(__file__).resolve().parent
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
}
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; "
    "img-src 'self' data:; connect-src 'self'; form-action 'self'; "
    "base-uri 'none'; frame-ancestors 'none'"
)
NAME_MAX = 32
BODY_MAX = 500
PAYLOAD_MAX = 4096
LIST_DEFAULT = 100
LIST_MAX = 200
DEFAULT_NAME = "anon"


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


class Handler(BaseHTTPRequestHandler):
    server_version = "common-board"
    board: Board

    def do_GET(self) -> None:
        path, _, query = self.path.partition("?")
        if path in STATIC:
            name, ctype = STATIC[path]
            self._send(200, (HERE / name).read_bytes(), ctype)
        elif path == "/api/messages":
            try:
                limit = int(parse_qs(query).get("limit", [LIST_DEFAULT])[0])
            except ValueError:
                self._json(400, {"error": "limit must be a number"})
                return
            limit = max(1, min(limit, LIST_MAX))
            self._json(200, {"messages": self.board.latest(limit)})
        elif path == "/health":
            self._json(200, {"ok": True})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.partition("?")[0]
        if path != "/api/messages":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._json(400, {"error": "bad content length"})
            return
        if length > PAYLOAD_MAX:
            self._json(413, {"error": "message is too large"})
            return
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._json(400, {"error": "send json"})
            return
        if not isinstance(data, dict):
            self._json(400, {"error": "send an object with name and body"})
            return
        try:
            msg = self.board.post(data.get("name"), data.get("body"))
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        self._json(201, {"message": msg})

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json", cache="no-store")

    def _send(self, status: int, body: bytes, ctype: str, cache: str = "public, max-age=60") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.end_headers()
        self.wfile.write(body)


def make_server(board: Board, host: str, port: int) -> ThreadingHTTPServer:
    handler = type("BoardHandler", (Handler,), {"board": board})
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Common message board.")
    parser.add_argument("--host", default=os.environ.get("BOARD_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument("--db", default=os.environ.get("BOARD_DB", str(HERE / "board.db")))
    args = parser.parse_args()

    board = Board(args.db)
    server = make_server(board, args.host, args.port)
    print(f"message board on http://{args.host}:{server.server_port}  (db: {args.db})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
