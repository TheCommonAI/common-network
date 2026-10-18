#!/usr/bin/env python3
"""The Common worker: the only thing that should ever face the network.

The Common Network Alpha (v0.1.13).

Copyright (C) 2026 Common AI Inc. Licensed under AGPL-3.0; see LICENSE at
https://github.com/TheCommonAI/common-network. This program comes with
ABSOLUTELY NO WARRANTY.

Before this existed, `common join` tunnelled straight to the model backend and
the gateway published that URL in GET /nodes. Ollama's API has no
authentication, so anyone who read the registry could talk to a contributor's
machine directly: bypass the gateway's contribution gate and rate limit, run
inference on someone else's hardware for free, and reach model management --
`/api/pull`, `/api/delete`, `/api/create` -- not just inference. llama-server is
the same story with a different name: no authentication either, plus a web UI
that browses the directory the weights are in.

So the tunnel now points here instead, and this process enforces three rules
before anything reaches the model:

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
import os
import secrets
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 127.0.0.1 instead of localhost: on Windows, localhost may resolve to ::1 (IPv6)
# while the model backend binds only to IPv4, causing requests to fail silently.
DEFAULT_UPSTREAM = "http://127.0.0.1:11436"

# The complete set of routes reachable from outside. The model backend's own
# management surface -- llama-server's /props, /slots and web UI -- is not on it
# and must never be: this list is the security boundary.
ROUTE_MODELS = "/v1/models"
ROUTE_CHAT = "/v1/chat/completions"

# A slow model on a cold laptop can take ~57s for the first token; the
# gateway's own member timeout is 90s. Sit just above it so the gateway's
# timeout is the one that fires, and a stuck upstream cannot pin a thread
# forever.
UPSTREAM_TIMEOUT = 120

# The ceiling on one generation, and the value used when the caller names none.
#
# Nothing upstream sets one. The client sends {model, messages, stream}; the
# gateway rewrites the model name and nothing else; this file stamps the fields
# below and nothing else. So every request runs on the backend's own default, and both
# backends this has served have the same one -- Ollama's num_predict and
# llama-server's n_predict both default to -1, "generate until the model emits
# EOS" -- so a model that does not emit EOS has nothing stopping it but the
# context window.
#
# This fleet serves a maths specialisation, and routing hands it whatever the
# network is asked. Asked off-distribution -- a roleplay prompt, a question in
# mixed scripts -- it intermittently never stops. Measured on one node
# 2026-10-13: two such prompts were still generating at 150s, against 121 and 56
# tokens in under 31s with a cap; and that node's own Ollama log holds 4m24s,
# 5m0s and 14m56s rows for /v1/chat/completions. The tail of a run like that is
# repetition salad, and the client is billed the whole wait for it.
#
# 1024 is chosen against the timeouts rather than picked round. At the ~13 tok/s
# that generation measured, a capped worst case is about 80s -- inside the
# gateway's 180s read timeout, so a pathological answer still arrives as an
# answer rather than as a client-side timeout with nothing to show. It is also
# several times the longest legitimate answer observed here; asked ordinarily,
# this model stops at 20-200 tokens of its own accord.
#
# This bounds the blast radius; it does not make the model behave. A capped run
# can still degenerate, just sooner. What it removes is minutes of someone's
# wait, and the case where the answer is not merely bad but late.
#
# Overridable per node with COMMON_MAX_TOKENS (see main()).
DEFAULT_MAX_TOKENS = 1024

# A mirror of runtime.DEFAULT_SLEEP_IDLE_SECONDS, duplicated on purpose: this
# file is the one piece of a node that must run stdlib-only with no sibling
# imports, and a standalone worker (see main()) still wants keep-warm behaviour
# against whatever backend it fronts. Overridable with the same
# COMMON_SLEEP_IDLE_SECONDS join.py reads, so one env var steers both layers.
DEFAULT_SLEEP_IDLE_SECONDS = 1800

# The catalogue's non-thinking sampling, stamped per request as defaults the
# caller may override (see the stamp in do_POST for why, and for why these are
# defaults rather than forced values). The values are the Bonsai 2 cards' own
# non-thinking configuration, not picks of ours.
DEFAULT_SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "presence_penalty": 1.5,
}


def bounded_max_tokens(requested, ceiling: int = DEFAULT_MAX_TOKENS) -> int:
    """The value to send as max_tokens, honouring a sane caller value.

    A caller may ask for less and is taken at its word; it may not ask for more,
    because the unbounded case is the whole bug and it arrives disguised as an
    ordinary large number.

    This sits on a public HTTP boundary, so the body is whatever a stranger's
    client sent. A limit that is absent, zero, negative, a bool or a string has
    to land on the ceiling rather than raise inside the request path or reach
    the model as junk.
    """
    if isinstance(requested, bool) or not isinstance(requested, (int, float)):
        return ceiling
    if requested < 1:
        return ceiling
    return min(int(requested), ceiling)


class WorkerConfig:
    def __init__(self, token: str, model: str, upstream_url: str = DEFAULT_UPSTREAM,
                 max_tokens: int = DEFAULT_MAX_TOKENS):
        self.token = token
        self.model = model
        self.upstream_url = upstream_url.rstrip("/")
        self.max_tokens = max_tokens


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
        # and generating the reply here means a health check never reaches the
        # model backend at all.
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
        # forwarded -- llama-server serves whatever single model it was started
        # with, so an unknown name would come back as an opaque 500 from the
        # backend rather than as the clear refusal a caller can act on.
        requested = body.get("model")
        if requested and requested not in (self.config.model, "auto"):
            self._json(400, {"error": {
                "message": f"this node serves {self.config.model!r}, not {requested!r}",
                "type": "invalid_request_error",
            }})
            return
        body["model"] = self.config.model

        # The node owns its own ceiling on generation, for the same reason it owns
        # the model. A caller asking for less is taken at its word; one asking for
        # more -- or for nothing at all -- gets this node's limit. See
        # DEFAULT_MAX_TOKENS for why one has to exist.
        #
        # There used to be a keep_alive stamp beside this, asking Ollama to hold
        # the model in memory for half an hour. That intent has not gone away, but
        # the mechanism has: llama-server is started with --sleep-idle-seconds and
        # decides for itself, which is strictly better than a hint repeated on
        # every request by a process that does not own the memory. The process
        # that does own it keeps the model warm instead -- see _keep_warm, which
        # makes sure the sleep of an idle server is only ever reached after this
        # node has stopped serving, not in the middle of a session.
        body["max_tokens"] = bounded_max_tokens(body.get("max_tokens"),
                                                self.config.max_tokens)

        # One user's exchange must not be context for the next user's request.
        # This node runs one inference slot (llama-server is started -np 1 -- a
        # memory decision, like the context size: every parallel slot allocates
        # its own KV cache), and a slot outlives the request that filled it: the
        # server prefix-matches each new prompt against whatever the slot last
        # held, generated tokens included, and reuses what matches. On a machine
        # one person uses that is a free speed-up; on a machine a network shares
        # it is one user's conversation sitting in the context the next user's
        # request is processed against, and whatever survives the matching shows
        # up as an answer to a question this user never asked. Forced rather than
        # honoured, for the same reason the model name is rewritten above: it is
        # this node's rule, and not a caller's optimisation to trade everyone
        # else's isolation for.
        body["cache_prompt"] = False

        # The catalogue's models default to thinking mode -- a long chain of
        # thought before the answer -- which is incompatible with the bounded
        # replies above: at the 1024-token ceiling, a thinking preamble can
        # eat the whole budget and leave nothing for the answer. So the node
        # stamps the model cards' non-thinking configuration: thinking off via
        # the chat template's own switch, then the cards' sampling defaults.
        #
        # Unlike the model name, the token ceiling and cache_prompt, these are
        # defaults rather than overrides -- a caller that knows what it wants
        # keeps it. setdefault, not assignment: the node's job is to make an
        # unstamped request produce a usable answer, not to confiscate control
        # from a caller that sets things deliberately. A junk value is passed
        # through for the backend to reject in its own words (which _proxy_chat
        # relays), matching how an unknown model name is handled above.
        kwargs = body.get("chat_template_kwargs")
        if not isinstance(kwargs, dict):
            kwargs = {}
        kwargs.setdefault("enable_thinking", False)
        body["chat_template_kwargs"] = kwargs
        for field, value in DEFAULT_SAMPLING.items():
            body.setdefault(field, value)

        self._proxy_chat(body)

    def _proxy_chat(self, body: dict) -> None:
        """Forward to the model backend and stream the reply back byte for byte.

        Streaming is the part that has to be exact: the gateway relays these
        bytes to a client parsing server-sent events, so re-chunking or
        re-encoding here truncates answers downstream.
        """
        req = urllib.request.Request(
            f"{self.config.upstream_url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            upstream = urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT)
        except urllib.error.HTTPError as e:
            # Pass the backend's own error through -- the gateway surfaces it, and
            # a contributor debugging a bad model name should see the real reason
            # rather than a generic 502.
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
                    # read1, not read: read(n) blocks until n bytes or the end of
                    # the stream, so a short answer arrives in one piece at the
                    # end and a long one in 8 KB lumps. read1 returns whatever has
                    # arrived, so each token is forwarded as the model writes it.
                    chunk = upstream.read1(8192)
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


def warm_model(model: str, upstream_url: str) -> None:
    """Ask the model backend for one token before any user needs it.

    The cold load used to be the whole story. Ollama deferred loading a model
    until a request touched it, so the first token cost 53-77s on this fleet and
    whichever user arrived first paid it. llama-server loads at start-up
    instead, and `common join` waits for /health before it starts this worker at
    all -- so most of that cost is now gone before this function is reached.

    What is left is the first batch. The weights are memory-mapped, so the first
    inference faults them in and builds its graph, and that is still seconds on a
    cold machine rather than milliseconds. There is no reason for a user to be
    the one who pays even that: `common join` has already spent tens of seconds
    on a tunnel and on registration, and this runs on a background thread inside
    that window, so registration is not held up behind it and every request after
    it starts warm.

    Failure is deliberately silent. If the backend is not up yet, or this model
    name is not one it serves, the first real request pays exactly what this
    would have -- precisely the behaviour that existed before this function, so
    there is nothing to report and nothing to break.
    """
    req = urllib.request.Request(
        f"{upstream_url}/v1/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "max_tokens": 1,
            # Keep the warm-up off the slot's cache too -- see the stamp in
            # do_POST. A warm model with someone else's residue in it would be
            # the warm-up unmaking the isolation the next real request expects.
            "cache_prompt": False,
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT):
            pass
    except (urllib.error.URLError, OSError):
        return


def _keep_warm(model: str, upstream_url: str, interval_seconds: float) -> None:
    """Never let the model sleep while this node is serving.

    llama-server unloads the weights after --sleep-idle-seconds of quiet, and
    what wakes a slept server is the next request to arrive -- which is to say,
    a user, who pays the reload: tens of seconds on a laptop, the cold start
    the warm-up above exists to keep off the *first* request and which would
    otherwise land on every thirty-minutes-quiet first request after it. That
    trade is right for a node that has gone home and wrong for one in the
    middle of a session; a node running this process has already said its
    hardware is for this.

    So the interval is derived from the sleep window the server was started
    with (three quarters of it, floored at a minute) rather than being a second
    setting to drift out of step, and the ping is the warm-up itself: one
    token, straight to the backend, never through the token-checked route. Each
    ping resets the idle timer, so the window never closes while this thread
    lives -- and this thread is a daemon, so when the node leaves, both die
    together and --sleep-idle-seconds goes back to doing exactly what it is
    for: a lingering backend nobody pinged gives the RAM back.
    """
    while True:
        time.sleep(interval_seconds)
        warm_model(model, upstream_url)


def serve(token: str, model: str, port: int = 11435,
          upstream_url: str = DEFAULT_UPSTREAM,
          max_tokens: int = DEFAULT_MAX_TOKENS,
          warm: bool = True,
          sleep_idle_seconds: int | None = None) -> ThreadingHTTPServer:
    """Start the worker on a background thread and return the server.

    Bound to 0.0.0.0 because the whole point is to be reachable -- by the
    tunnel, or by other machines in LAN mode. What protects it is the token,
    not the bind address.

    `warm=False` skips the background model load: for a caller that is not
    really serving, and cannot afford an unannounced inference against whatever
    happens to be behind `upstream_url`. The test suite is the caller that
    matters -- a warm-up thread firing mid-suite would land an extra request in
    the stub's hit log and turn a real assertion into a coin toss. Keep-warm is
    gated the same way, for the same reason: a caller that does not want one
    background inference does not want a repeating one either.

    `sleep_idle_seconds` is the window the backend was started with
    (--sleep-idle-seconds). Given once, keep-warm derives its interval from it
    and there is no second knob to keep in step with a flag that lives in
    another process's argv. Omitted or zero, no keep-warm thread runs: a
    backend that never sleeps needs no alarm clock, and a caller that says
    nothing gets exactly the old behaviour.
    """
    handler = type("BoundHandler", (Handler,), {
        "config": WorkerConfig(token, model, upstream_url, max_tokens),
    })
    httpd = ThreadingHTTPServer(("0.0.0.0", port), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    if warm:
        threading.Thread(target=warm_model, args=(model, upstream_url),
                         daemon=True).start()
        if sleep_idle_seconds:
            threading.Thread(
                target=_keep_warm,
                args=(model, upstream_url, max(60.0, sleep_idle_seconds * 0.75)),
                daemon=True,
            ).start()
    return httpd


def main() -> int:
    """Run standalone: `python3 worker.py --token ... --model ...`.

    `common join` calls serve() directly; this exists for debugging and for
    anyone wanting to run the worker under their own supervisor.
    """
    import argparse

    # The banner line below prints an arrow. On Windows a redirected stream --
    # a pipe, or the file a supervisor sends output to -- gets the locale
    # encoding, which is cp1252 on most installs and cannot encode it, so the
    # worker would die on its own first line. `common join` calls serve() in
    # process and has already done this, but standing alone it has not.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass

    p = argparse.ArgumentParser(prog="common-worker", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--token", required=True, help="worker token issued by the gateway")
    p.add_argument("--model", required=True, help="the one model this node serves")
    p.add_argument("--port", type=int, default=11435)
    p.add_argument("--upstream", default=os.environ.get("COMMON_UPSTREAM", DEFAULT_UPSTREAM),
                   help="base URL of the model backend this worker fronts "
                        "(default: %(default)s)")
    p.add_argument("--max-tokens", type=int,
                   default=os.environ.get("COMMON_MAX_TOKENS", DEFAULT_MAX_TOKENS),
                   help="ceiling on the tokens generated for one request "
                        "(default: %(default)s)")
    args = p.parse_args()

    # Standalone use manages its own backend, so it reads the same
    # COMMON_SLEEP_IDLE_SECONDS join.py passes and derives keep-warm from it;
    # see DEFAULT_SLEEP_IDLE_SECONDS above for why the constant is mirrored
    # here instead of imported.
    sleep_idle = os.environ.get("COMMON_SLEEP_IDLE_SECONDS")
    try:
        sleep_idle = int(sleep_idle) if sleep_idle else DEFAULT_SLEEP_IDLE_SECONDS
    except ValueError:
        sleep_idle = DEFAULT_SLEEP_IDLE_SECONDS
    serve(args.token, args.model, args.port, args.upstream, args.max_tokens,
          sleep_idle_seconds=sleep_idle)
    print(f"common-worker on :{args.port} → {args.upstream} (model: {args.model})")
    print("the backend unloads the model itself, once idle for longer than the "
          "sleep timeout it was started with.")
    print(f"one answer is capped at {args.max_tokens} tokens.")
    print("only /v1/models and /v1/chat/completions are reachable; both need the token.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
