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

httpd = worker.serve(TOKEN, MODEL, port=0, ollama_url=STUB_URL)
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
