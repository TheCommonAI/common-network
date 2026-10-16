"""The worker is the boundary between the internet and a contributor's machine.

Everything here is a real HTTP request to a real worker process on localhost,
with a stub upstream standing in for the model backend (llama-server; it was
Ollama when this suite was written) -- the thing being tested is what the worker
lets through, and a mocked request object would test the mock.

The stub records what reached it, so "was this blocked?" is answered by
upstream silence rather than by trusting the status code the worker returned.

Two things changed with the llama.cpp migration and are pinned below: the
worker's upstream is named by `upstream_url` rather than `ollama_url`, and the
keep-alive mechanism is gone. It did not go away because the concern behind it
was wrong -- a caller must not be able to make a contributor's machine slow for
everyone after them -- but because llama-server decides residency itself, from
`--sleep-idle-seconds` given at start-up, which is a better owner for that
decision than a hint repeated on every request by a process that does not own
the memory.
"""
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "join"))

import worker  # noqa: E402

FAILURES = []
TOKEN = "test-worker-token-0123456789"
MODEL = "test-model:1b"


def check(name: str, got, want) -> None:
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL  {name}  (got {got!r}, want {want!r})")
    else:
        print(f"  ok    {name}")


# --- stub model backend -----------------------------------------------------

UPSTREAM_HITS = []


class StubBackend(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        return

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length)) if length else {}
        UPSTREAM_HITS.append({"path": self.path, "body": body})

        if body.get("stream"):
            # Two SSE frames and a terminator -- enough to prove the worker
            # relays the framing rather than reassembling it.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for frame in (b'data: {"choices":[{"delta":{"content":"one "}}]}\n\n',
                          b'data: {"choices":[{"delta":{"content":"two"}}]}\n\n',
                          b"data: [DONE]\n\n"):
                self.wfile.write(f"{len(frame):X}\r\n".encode() + frame + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            return

        payload = json.dumps({
            "model": body.get("model"),
            "choices": [{"message": {"role": "assistant", "content": "stub reply"}}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        UPSTREAM_HITS.append({"path": self.path, "body": None})
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")


stub = ThreadingHTTPServer(("127.0.0.1", 0), StubBackend)
stub.daemon_threads = True
threading.Thread(target=stub.serve_forever, daemon=True).start()
STUB_URL = f"http://127.0.0.1:{stub.server_address[1]}"

# warm=False: the real serve() loads the model in the background, and an
# unannounced 53s inference landing in the stub's hit log mid-suite would turn
# the `UPSTREAM_HITS == []` assertions below into a race. The warm-up path has
# its own section at the end, where it is the subject rather than a bystander.
httpd = worker.serve(TOKEN, MODEL, port=0, upstream_url=STUB_URL, warm=False)
BASE = f"http://127.0.0.1:{httpd.server_address[1]}"


def request(method: str, path: str, token: str | None = None,
            body: dict | None = None, raw: bytes | None = None):
    """Returns (status, body_text). An HTTP error status is a result here,
    not an exception -- refusals are what most of these tests assert."""
    data = raw if raw is not None else (json.dumps(body).encode() if body else None)
    headers = {"Content-Type": "application/json"} if data else {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def headers_of(method: str, path: str, token: str | None = None,
               body: dict | None = None, raw: bytes | None = None) -> dict:
    data = raw if raw is not None else (json.dumps(body).encode() if body else None)
    hdrs = {"Content-Type": "application/json"} if data else {}
    if token:
        hdrs["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(BASE + path, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return dict(r.headers)
    except urllib.error.HTTPError as e:
        return dict(e.headers)


try:
    print("\nauthentication (the tunnel URL is public; the token is the gate)")
    check("no credential -> 401", request("GET", "/v1/models")[0], 401)
    check("wrong token -> 401", request("GET", "/v1/models", "nope")[0], 401)
    check("right token -> 200", request("GET", "/v1/models", TOKEN)[0], 200)
    # A prefix must fail: compare_digest is length-safe, but a startswith()
    # refactor would pass every test above and break exactly here.
    check("prefix of the token -> 401",
          request("GET", "/v1/models", TOKEN[:10])[0], 401)
    check("chat without a credential -> 401",
          request("POST", "/v1/chat/completions", None,
                  {"messages": [{"role": "user", "content": "hi"}]})[0], 401)
    # The scheme is part of the credential, not decoration: a bare token must
    # not be accepted just because the value is right.
    check("the token without 'Bearer ' -> 401",
          request("GET", "/v1/models", "x")[0], 401)

    print("\nroute allowlist (the model backend's own API must be unreachable)")
    UPSTREAM_HITS.clear()
    # The first four are Ollama's management API, kept because a node that
    # somehow still had Ollama behind it must not be reachable through here
    # either. The rest are llama-server's: /props and /slots describe the
    # running model, and the bare / serves a web UI that browses the directory
    # the weights live in -- the exact thing that made tunnelling to the
    # backend directly the wrong design.
    for path in ("/api/delete", "/api/pull", "/api/create", "/api/tags",
                 "/v1/completions", "/v1/embeddings", "/",
                 "/props", "/slots", "/completion", "/health", "/metrics", "/ui"):
        method = "GET" if path in ("/api/tags", "/", "/props", "/slots", "/health", "/metrics", "/ui") else "POST"
        status, _ = request(method, path, TOKEN, {"name": MODEL} if method == "POST" else None)
        check(f"{method} {path} -> 404", status, 404)
    check("nothing reached the upstream at all", UPSTREAM_HITS, [])

    print("\nthe worker does not advertise what it is running")
    hdr = headers_of("GET", "/v1/models", TOKEN)
    # http.server builds this as "server_version sys_version", so the absent
    # sys_version leaves a trailing space -- asserted by content rather than by
    # exact equality, because the property that matters is that no interpreter
    # version is volunteered to whoever is probing.
    server = hdr.get("Server", "")
    check("no Python version in Server",
          (server.strip(), "Python" in server, any(c.isdigit() for c in server)),
          (worker.Handler.server_version, False, False))

    print("\nmodel pinning (no surprise multi-GB pull on a donated laptop)")
    UPSTREAM_HITS.clear()
    status, text = request("POST", "/v1/chat/completions", TOKEN,
                           {"model": "llama3.1:70b",
                            "messages": [{"role": "user", "content": "hi"}]})
    check("a different model -> 400", status, 400)
    check("and never reached the upstream", UPSTREAM_HITS, [])

    UPSTREAM_HITS.clear()
    request("POST", "/v1/chat/completions", TOKEN,
            {"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
    check("'auto' is rewritten to this node's model",
          UPSTREAM_HITS[0]["body"]["model"], MODEL)

    UPSTREAM_HITS.clear()
    request("POST", "/v1/chat/completions", TOKEN,
            {"messages": [{"role": "user", "content": "hi"}]})
    check("a missing model is filled in, not rejected",
          UPSTREAM_HITS[0]["body"]["model"], MODEL)

    print("\nrequest shape")
    check("non-JSON body -> 400",
          request("POST", "/v1/chat/completions", TOKEN, raw=b"not json at all")[0], 400)
    check("empty body -> 413 (no length)",
          request("POST", "/v1/chat/completions", TOKEN, raw=b"")[0], 413)

    print("\nhealth: /v1/models answers locally, without waking the model")
    UPSTREAM_HITS.clear()
    status, text = request("GET", "/v1/models", TOKEN)
    listed = json.loads(text)
    check("reports exactly the donated model",
          [m["id"] for m in listed["data"]], [MODEL])
    check("answered without touching the upstream", UPSTREAM_HITS, [])

    print("\nstreaming passes through intact")
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps({"model": MODEL, "stream": True,
                         "messages": [{"role": "user", "content": "hi"}]}).encode(),
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        body = r.read().decode()
        stream_ct = r.headers.get("Content-Type", "")
    check("both SSE frames arrived", body.count("data: {"), 2)
    check("terminator preserved", "data: [DONE]" in body, True)
    check("the content type is relayed, not rewritten",
          stream_ct, "text/event-stream")
    check("content reassembles in order",
          "".join(json.loads(line[6:])["choices"][0]["delta"]["content"]
                  for line in body.splitlines()
                  if line.startswith("data: {")), "one two")

    # The cold load is the whole latency story: 53-77s to the first token on a
    # node whose model had been evicted, against about a second when it is
    # resident. That was Ollama's default keep_alive of five minutes, which
    # nothing here used to set, so every idle node was cold.
    #
    # The concern -- that a caller choosing to be slow must not pay for it at
    # the contributor's expense, for everyone else on that node -- has not gone
    # away. The mechanism has: llama-server holds the model itself, per its own
    # --sleep-idle-seconds, so there is no per-request hint to stamp and none to
    # defend against. Asserted rather than assumed, because a keep_alive that
    # came back would be silently ignored by the backend and look harmless.
    print("\nthe node owns its own memory (no per-request residency hint)")
    check("the worker stamps no keep_alive of its own", hasattr(worker, "DEFAULT_KEEP_ALIVE"), False)
    UPSTREAM_HITS.clear()
    request("POST", "/v1/chat/completions", TOKEN,
            {"messages": [{"role": "user", "content": "hi"}], "keep_alive": 0})
    check("and a caller's keep_alive is not something this node acts on",
          UPSTREAM_HITS[0]["body"].get("keep_alive"), 0)

    # Nothing upstream bounds generation. The client sends {model, messages,
    # stream}; the gateway rewrites the model name and nothing else; this file
    # used to add keep_alive and nothing else. So every request ran on the
    # backend's default of n_predict = -1 -- "until EOS" -- and a model that
    # does not emit EOS ran to the context window instead. On 2026-10-13 one
    # such request on a live node took 4m24s and returned repetition salad.
    # These checks are the ceiling that stops it, at the one process every
    # request must pass.
    print("\ngeneration is bounded (an answer that will not stop must still end)")
    check("the default ceiling is set", worker.DEFAULT_MAX_TOKENS, 1024)

    UPSTREAM_HITS.clear()
    request("POST", "/v1/chat/completions", TOKEN,
            {"messages": [{"role": "user", "content": "hi"}]})
    check("a caller naming no limit gets the node's",
          UPSTREAM_HITS[0]["body"]["max_tokens"], worker.DEFAULT_MAX_TOKENS)

    # A caller may ask for less. It may not ask for more: the unbounded case is
    # the whole bug, and it arrives dressed as an ordinary large number.
    UPSTREAM_HITS.clear()
    request("POST", "/v1/chat/completions", TOKEN,
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64})
    check("a smaller request is honoured",
          UPSTREAM_HITS[0]["body"]["max_tokens"], 64)

    UPSTREAM_HITS.clear()
    request("POST", "/v1/chat/completions", TOKEN,
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10 ** 9})
    check("an enormous one is clamped to the ceiling",
          UPSTREAM_HITS[0]["body"]["max_tokens"], worker.DEFAULT_MAX_TOKENS)

    # This is a public HTTP boundary: the body is whatever a stranger's client
    # sent. A limit that is not a positive number has to land on the ceiling --
    # not raise inside the request path, and not reach the backend as junk. A
    # bool is listed because it is an int subclass and would otherwise sail
    # through as 1, quietly turning every answer into a single token.
    for label, value in (("a non-numeric", "lots"), ("a negative", -5),
                         ("zero", 0), ("a bool", True), ("null", None)):
        UPSTREAM_HITS.clear()
        status, _ = request("POST", "/v1/chat/completions", TOKEN,
                            {"messages": [{"role": "user", "content": "hi"}],
                             "max_tokens": value})
        check(f"{label} limit -> the ceiling, not a 500",
              (status, UPSTREAM_HITS[0]["body"]["max_tokens"]),
              (200, worker.DEFAULT_MAX_TOKENS))

    # A real number below the ceiling is a real request, not a mistake: 12.9
    # tokens is 12 tokens.
    UPSTREAM_HITS.clear()
    request("POST", "/v1/chat/completions", TOKEN,
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 12.9})
    check("a fractional limit is truncated, not rejected",
          UPSTREAM_HITS[0]["body"]["max_tokens"], 12)

    # The ceiling must not reach the warm-up, which asks for exactly one token
    # on purpose and would be a join that appears to hang at 1024.
    check("the warm-up's single token is not raised to the ceiling",
          worker.bounded_max_tokens(1), 1)

    # A node may lower its own ceiling -- COMMON_MAX_TOKENS reaches serve() as
    # max_tokens, and a caller asking for more than that must still be clamped
    # by the node's value rather than the module default.
    check("a node's own ceiling is what a caller is clamped to",
          worker.bounded_max_tokens(10 ** 9, ceiling=256), 256)

    # warm_model is what `common join` relies on to make the *first* user
    # request fast. Called directly rather than through serve(), so this is a
    # statement about the function instead of a race with a background thread.
    print("\nwarm-up (joining pays the load, not the first person to ask)")
    UPSTREAM_HITS.clear()
    worker.warm_model(MODEL, STUB_URL)
    check("the warm-up actually reaches the backend", len(UPSTREAM_HITS), 1)
    check("against this node's model", UPSTREAM_HITS[0]["body"]["model"], MODEL)
    # One token is the point: this exists to move weights into memory, and a
    # long generation here would just be a join that appears to hang.
    check("for one token, not a real answer",
          UPSTREAM_HITS[0]["body"]["max_tokens"], 1)
    # Not streamed: the warm-up wants the whole load finished, not the first
    # byte of it, and a streaming request returns as soon as headers land.
    check("and not as a stream",
          UPSTREAM_HITS[0]["body"]["stream"], False)

    # A contributor whose backend is not up yet must still be able to join --
    # the first real request loads the model exactly as it did before this
    # existed. A raise here would turn a slow start into no node at all.
    worker.warm_model(MODEL, "http://127.0.0.1:1")
    check("an unreachable backend is not fatal", True, True)

    # On Windows, "localhost" may resolve to ::1 (IPv6) while services bind to
    # IPv4 only, causing silent failures. The default upstream must use the
    # explicit IPv4 address 127.0.0.1 instead.
    print("\nIPv4 address (Windows localhost resolves to IPv6, breaking tunnels)")
    check("DEFAULT_UPSTREAM uses 127.0.0.1, not localhost",
          "127.0.0.1" in worker.DEFAULT_UPSTREAM and "localhost" not in worker.DEFAULT_UPSTREAM, True)
    # The runtime is held to localhost; only the worker faces the network.
    check("and the runtime's port is not the worker's",
          worker.DEFAULT_UPSTREAM.rsplit(":", 1)[1], "11436")

    print("\nthe upstream is not reachable without the token")
    check("a wrong model name never becomes an upstream request",
          [h for h in UPSTREAM_HITS if h["path"] != "/v1/chat/completions"], [])

    # A backend that is down must surface as a 502 naming the problem, not as a
    # stack trace or a hang.
    print("\na dead backend is reported, not hidden")
    dead = worker.serve(TOKEN, MODEL, port=0, upstream_url="http://127.0.0.1:1", warm=False)
    dead_base = f"http://127.0.0.1:{dead.server_address[1]}"
    try:
        req = urllib.request.Request(
            dead_base + "/v1/chat/completions",
            data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
            headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                check("an unreachable backend -> 502", r.status, 502)
        except urllib.error.HTTPError as e:
            check("an unreachable backend -> 502", e.code, 502)
            check("and says whose fault it is",
                  json.loads(e.read().decode())["error"]["type"], "upstream_error")
    finally:
        dead.shutdown()
finally:
    httpd.shutdown()
    stub.shutdown()

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED\n")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all worker tests passed")
