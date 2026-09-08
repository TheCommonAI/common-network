#!/usr/bin/env python3
"""The Common worker: the only thing that should ever face the network.

The Common Network Alpha (v0.1.2).

Copyright (C) 2026 Common AI Inc. Licensed under AGPL-3.0; see LICENSE at
https://github.com/TheCommonAI/common-network. This program comes with
ABSOLUTELY NO WARRANTY.

Before this existed, `common join` tunnelled straight to Ollama and the
gateway published that URL in GET /nodes. Ollama's API has no authentication,
so anyone who read the registry could talk to a contributor's machine
directly: bypass the gateway's contribution gate and rate limit, run
inference on someone else's hardware for free, and reach model management --
`/api/pull`, `/api/delete`, `/api/create` -- not just inference.

So the tunnel now points here instead, and this process enforces three rules
before anything reaches Ollama:

  1. **Bearer token.** Every request must carry the worker token the gateway
     was issued for this node. No token, no reply.
  2. **Two routes.** GET /v1/models and POST /v1/chat/completions. Everything
     else 404s -- there is no path from the public internet to /api/delete.
  3. **One model.** The model this machine donated. A request naming anything
     else is refused rather than quietly pulling a 40GB download onto a
     contributor's laptop.

What it deliberately does NOT do: hide request content from the person
running it. Prompts routed here are readable by this machine's owner, and no
amount of code changes that -- it is inherent to donated compute. Say so
plainly rather than implying otherwise.

Stdlib only, like the rest of the CLI: a contributor should never have to
install anything to donate a machine.
"""
from __future__ import annotations

import json
import secrets
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_OLLAMA = "http://localhost:11434"

# The complete set of routes reachable from outside. Ollama's own management
# API is not on it and must never be: this list is the security boundary.
ROUTE_MODELS = "/v1/models"
ROUTE_CHAT = "/v1/chat/completions"

# A slow model on a cold laptop can take ~57s for the first token; the
# gateway's own member timeout is 90s. Sit just above it so the gateway's
# timeout is the one that fires, and a stuck upstream cannot pin a thread
# forever.
UPSTREAM_TIMEOUT = 120


class WorkerConfig:
    def __init__(self, token: str, model: str, ollama_url: str = DEFAULT_OLLAMA):
        self.token = token
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")


class Handler(BaseHTTPRequestHandler):
    config: WorkerConfig  # set by serve()

    # --- plumbing ---------------------------------------------------------

    protocol_version = "HTTP/1.1"          # keep-alive, so streaming works
    server_version = "common-worker"
    sys_version = ""                       # don't advertise the Python version

    def log_message(self, fmt, *args):
        """Silence per-request logging.

        The default handler writes the request line to stderr. That is a log
        of what a contributor's machine was asked to do, sitting in their
        terminal and in any file the process is redirected to. Content-free
        logging is the default here; `common join` prints its own summary.
        """
        return

    def _json(self, status: int, payload: dict, close: bool = False) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if close:
            # For a refusal issued before the body was read. A client that has
            # finished sending reads the status cleanly; one still streaming a
            # huge body sees a connection reset instead, because that is what
            # TCP does when the far end closes mid-send. Refusing without
            # buffering is the point, so that trade is deliberate -- a
            # legitimate request never reaches this branch.
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Peer gave up mid-refusal. Nothing to do and nothing worth
            # logging -- the request was being denied anyway.
            pass

    def _authorised(self) -> bool:
        """Constant-time bearer check.

        compare_digest rather than == so the comparison does not leak the
        token's prefix through timing to something that can retry quickly.
        """
        header = self.headers.get("Authorization", "")
        scheme, _, supplied = header.partition(" ")
        if scheme.lower() != "bearer" or not supplied:
            return False
        return secrets.compare_digest(supplied.strip(), self.config.token)

    def _deny(self) -> None:
        # 401 with no detail: a prober learns that credentials are required,
        # and nothing about what a valid one looks like.
        self._json(401, {"error": {
            "message": "this worker serves the Common Network gateway. "
                       "Requests need the gateway's worker token.",
            "type": "unauthorized",
        }})

    # --- routes -----------------------------------------------------------

    def do_GET(self):  # noqa: N802  (stdlib naming)
        if self.path.rstrip("/") != ROUTE_MODELS:
            self._json(404, {"error": {"message": "not found", "type": "not_found"}})
            return
        if not self._authorised():
            self._deny()
            return
        # Answered locally rather than proxied. The gateway's health check only
        # needs to know this node is alive and serving the model it declared,
        # and generating the reply here means a health check never reaches
        # Ollama at all.
        self._json(200, {
            "object": "list",
            "data": [{
                "id": self.config.model,
                "object": "model",
                "owned_by": "common-network",
            }],
        })

    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != ROUTE_CHAT:
            self._json(404, {"error": {"message": "not found", "type": "not_found"}})
            return
        if not self._authorised():
            self._deny()
            return

        # Bound the body before reading it: an unauthenticated caller cannot
        # get here, but an authenticated one should still not be able to make
        # this machine buffer an arbitrary amount of memory.
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 2_000_000:
            # Refused without reading it -- the point is not to buffer whatever
            # someone decided to send.
            self._json(413, {"error": {
                "message": "request body must be present and under 2MB",
                "type": "invalid_request_error",
            }}, close=True)
            return

        try:
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, ValueError):
            self._json(400, {"error": {"message": "body must be JSON",
                                       "type": "invalid_request_error"}})
            return

        # This machine donated one model. Anything else is refused rather than
        # forwarded -- Ollama would happily start pulling a model nobody here
        # agreed to host.
        requested = body.get("model")
        if requested and requested not in (self.config.model, "auto"):
            self._json(400, {"error": {
                "message": f"this node serves {self.config.model!r}, not {requested!r}",
                "type": "invalid_request_error",
            }})
            return
        body["model"] = self.config.model

        self._proxy_chat(body)

    def _proxy_chat(self, body: dict) -> None:
        """Forward to Ollama and stream the reply back byte for byte.

        Streaming is the part that has to be exact: the gateway relays these
        bytes to a client parsing server-sent events, so re-chunking or
        re-encoding here truncates answers downstream.
        """
        req = urllib.request.Request(
            f"{self.config.ollama_url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            upstream = urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT)
        except urllib.error.HTTPError as e:
            # Pass Ollama's own error through -- the gateway surfaces it, and a
            # contributor debugging a bad model name should see the real
            # reason rather than a generic 502.
            detail = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(detail)))
            self.end_headers()
            self.wfile.write(detail)
            return
        except (urllib.error.URLError, OSError):
            self._json(502, {"error": {
                "message": "this node's model backend is not responding",
                "type": "upstream_error",
            }})
            return

        with upstream:
            self.send_response(200)
            content_type = upstream.headers.get("Content-Type", "application/json")
            self.send_header("Content-Type", content_type)
            # Length is unknown while streaming, so switch to chunked framing
            # rather than guessing; keep-alive then works for both cases.
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                while True:
                    chunk = upstream.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(f"{len(chunk):X}\r\n".encode())
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # The gateway hung up (client cancelled, timeout). Stop
                # writing; the upstream response closes with the `with`.
                return


def serve(token: str, model: str, port: int = 11435,
          ollama_url: str = DEFAULT_OLLAMA) -> ThreadingHTTPServer:
    """Start the worker on a background thread and return the server.

    Bound to 0.0.0.0 because the whole point is to be reachable -- by the
    tunnel, or by other machines in LAN mode. What protects it is the token,
    not the bind address.
    """
    handler = type("BoundHandler", (Handler,), {
        "config": WorkerConfig(token, model, ollama_url),
    })
    httpd = ThreadingHTTPServer(("0.0.0.0", port), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main() -> int:
    """Run standalone: `python3 worker.py --token ... --model ...`.

    `common join` calls serve() directly; this exists for debugging and for
    anyone wanting to run the worker under their own supervisor.
    """
    import argparse

    p = argparse.ArgumentParser(prog="common-worker", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--token", required=True, help="worker token issued by the gateway")
    p.add_argument("--model", required=True, help="the one model this node serves")
    p.add_argument("--port", type=int, default=11435)
    p.add_argument("--ollama", default=DEFAULT_OLLAMA)
    args = p.parse_args()

    serve(args.token, args.model, args.port, args.ollama)
    print(f"common-worker on :{args.port} → {args.ollama} (model: {args.model})")
    print("only /v1/models and /v1/chat/completions are reachable; both need the token.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
