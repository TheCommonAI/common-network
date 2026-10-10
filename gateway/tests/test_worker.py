"""The worker is the boundary between the internet and a contributor's machine.

Everything here is a real HTTP request to a real worker process on localhost,
with a stub upstream standing in for Ollama -- the thing being tested is what
the worker lets through, and a mocked request object would test the mock.

The stub records what reached it, so "was this blocked?" is answered by
upstream silence rather than by trusting the status code the worker returned.
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
# The base a served blend was built from -- the one thing the allowlist is for.
BASE_MODEL = "test-base:1b"


def check(name: str, got, want) -> None:
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL  {name}  (got {got!r}, want {want!r})")
    else:
        print(f"  ok    {name}")


# --- stub Ollama ------------------------------------------------------------

UPSTREAM_HITS = []


class StubOllama(BaseHTTPRequestHandler):
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


stub = ThreadingHTTPServer(("127.0.0.1", 0), StubOllama)
stub.daemon_threads = True
threading.Thread(target=stub.serve_forever, daemon=True).start()
STUB_URL = f"http://127.0.0.1:{stub.server_address[1]}"

# warm=False: the real serve() loads the model in the background, and an
# unannounced 53s inference landing in the stub's hit log mid-suite would turn
# the `UPSTREAM_HITS == []` assertions below into a race. The warm-up path has
# its own section at the end, where it is the subject rather than a bystander.
httpd = worker.serve(TOKEN, MODEL, port=0, ollama_url=STUB_URL, warm=False)
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

    print("\nroute allowlist (Ollama's management API must be unreachable)")
    UPSTREAM_HITS.clear()
    for path in ("/api/delete", "/api/pull", "/api/create", "/api/tags",
                 "/v1/completions", "/v1/embeddings", "/"):
        method = "GET" if path in ("/api/tags", "/") else "POST"
        status, _ = request(method, path, TOKEN, {"name": MODEL})
        check(f"{method} {path} -> 404", status, 404)
    check("nothing reached the upstream at all", UPSTREAM_HITS, [])

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

    # The optional allowlist. It exists so a node serving a blend can still be
    # asked for the base it was built from, without a worker restart -- and the
    # assertions that matter are the ones proving it did NOT widen the pin: a
    # third name, a prefix of an allowed name, and a name that differs by case
    # must all still be refused. A regression here would turn a one-model
    # donation into an open proxy for whoever can reach the tunnel.
    print("\nmodel allowlist (a closed set, not a wildcard)")
    base_worker = worker.serve(TOKEN, MODEL, port=0, ollama_url=STUB_URL,
                               warm=False, extra_models=frozenset({BASE_MODEL}))
    base_url = f"http://127.0.0.1:{base_worker.server_address[1]}"

    def request_allowed(body):
        data = json.dumps(body).encode()
        req = urllib.request.Request(base_url + "/v1/chat/completions", data=data,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {TOKEN}"},
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    for allowed in (MODEL, BASE_MODEL, "auto"):
        UPSTREAM_HITS.clear()
        status, _ = request_allowed({"model": allowed,
                                     "messages": [{"role": "user", "content": "hi"}]})
        check(f"{allowed!r} (allowed) -> 200", status, 200)

    UPSTREAM_HITS.clear()
    status, _ = request_allowed({"model": MODEL,
                                 "messages": [{"role": "user", "content": "hi"}]})
    check("the primary is passed through unchanged",
          UPSTREAM_HITS[0]["body"]["model"], MODEL)

    UPSTREAM_HITS.clear()
    request_allowed({"model": BASE_MODEL,
                     "messages": [{"role": "user", "content": "hi"}]})
    check("an allowed extra is passed through, not rewritten",
          UPSTREAM_HITS[0]["body"]["model"], BASE_MODEL)

    for refused in ("llama3.1:70b", MODEL.split(":")[0], MODEL.upper(),
                    BASE_MODEL + ":latest"):
        UPSTREAM_HITS.clear()
        status, _ = request_allowed({"model": refused,
                                     "messages": [{"role": "user", "content": "hi"}]})
        check(f"{refused!r} (not allowed) -> 400", status, 400)
        check(f"{refused!r} never reached the upstream", UPSTREAM_HITS, [])

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
    check("both SSE frames arrived", body.count("data: {"), 2)
    check("terminator preserved", "data: [DONE]" in body, True)
    check("content reassembles in order",
          "".join(json.loads(line[6:])["choices"][0]["delta"]["content"]
                  for line in body.splitlines()
                  if line.startswith("data: {")), "one two")

    # The cold load is the whole latency story: 53-77s to the first token on a
    # node whose model has been evicted, against about a second when it is
    # resident. Ollama's default keep_alive is five minutes and nothing in this
    # repository used to set it, so every node fell back to that default and
    # every idle node was cold. These checks exist because the failure mode is
    # invisible -- the request still succeeds, it just takes a minute.
    print("\nthe model stays resident (a cold node costs 53-77s of someone's wait)")
    check("the default outlasts Ollama's own five minutes",
          worker.DEFAULT_KEEP_ALIVE, "30m")

    UPSTREAM_HITS.clear()
    request("POST", "/v1/chat/completions", TOKEN,
            {"messages": [{"role": "user", "content": "hi"}]})
    check("every request carries the node's keep_alive",
          UPSTREAM_HITS[0]["body"]["keep_alive"], worker.DEFAULT_KEEP_ALIVE)

    # The node owns its own memory, not the caller. Without this, one request
    # asking for keep_alive=0 would evict the model and make every request
    # after it pay the reload -- a caller choosing to be slow, at the
    # contributor's expense, for everyone else on that node.
    UPSTREAM_HITS.clear()
    request("POST", "/v1/chat/completions", TOKEN,
            {"messages": [{"role": "user", "content": "hi"}], "keep_alive": 0})
    check("a caller cannot evict the model out from under the next caller",
          UPSTREAM_HITS[0]["body"]["keep_alive"], worker.DEFAULT_KEEP_ALIVE)

    # warm_model is what `common join` relies on to make the *first* user
    # request fast. Called directly rather than through serve(), so this is a
    # statement about the function instead of a race with a background thread.
    print("\nwarm-up (joining pays the load, not the first person to ask)")
    UPSTREAM_HITS.clear()
    worker.warm_model(MODEL, STUB_URL, "7m")
    check("the warm-up actually reaches Ollama", len(UPSTREAM_HITS), 1)
    check("and asks Ollama to hold the model afterwards",
          UPSTREAM_HITS[0]["body"]["keep_alive"], "7m")
    check("against this node's model", UPSTREAM_HITS[0]["body"]["model"], MODEL)
    # One token is the point: this exists to move weights into memory, and a
    # long generation here would just be a join that appears to hang.
    check("for one token, not a real answer",
          UPSTREAM_HITS[0]["body"]["max_tokens"], 1)

    # A contributor whose Ollama is not up yet must still be able to join --
    # the first real request loads the model exactly as it did before this
    # existed. A raise here would turn a slow start into no node at all.
    worker.warm_model(MODEL, "http://127.0.0.1:1", "7m")
    check("an unreachable Ollama is not fatal", True, True)
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
