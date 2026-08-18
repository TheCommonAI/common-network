"""End-to-end test of the panel flow against real HTTP nodes.

Everything except the database and the embedding model is real here: actual
sockets, actual OpenAI-shaped request and response bodies, the actual
`upstream.forward` code path. Stub nodes are `http.server` instances on
localhost, so this catches the class of bug a mocked test cannot — a malformed
outgoing body, a header that doesn't survive the wire, a response the parser
chokes on.

Which is the class of bug that actually bit this project before. From
seam-findings.md §4: *"An earlier round of this project concluded entailment
'needs a bigger model'; it was an Ollama transport bug."* A transport failure
that looks like a capability failure costs weeks, and the only defence is
exercising the transport.

Run: `python tests/test_panel_flow.py` from `gateway/`.
"""
import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FAILURES: list[str] = []


def check(name: str, got, want) -> None:
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL  {name}  (got {got!r}, want {want!r})")
    else:
        print(f"  ok    {name}")


# --- Stub nodes -----------------------------------------------------------

class StubNode:
    """An OpenAI-compatible node that returns a canned answer.

    Records the request bodies it received, so the test can assert on what the
    gateway actually sent rather than on what it meant to send.
    """

    def __init__(self, reply: str, *, status: int = 200, delay: float = 0.0):
        self.reply = reply
        self.status = status
        self.delay = delay
        self.received: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.received.append(body)
                if outer.delay:
                    import time as _t
                    _t.sleep(outer.delay)
                if outer.status >= 400:
                    self.send_response(outer.status)
                    self.end_headers()
                    self.wfile.write(b"{}")
                    return
                payload = json.dumps({
                    "id": "chatcmpl-stub", "object": "chat.completion", "created": 0,
                    "model": body.get("model", "stub"),
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": outer.reply}}],
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_port
        self._stopped = False
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def stop(self):
        # Idempotent on purpose. `HTTPServer.shutdown()` blocks until the
        # serve_forever loop acknowledges it, so calling it a second time on an
        # already-stopped server waits for a loop that will never run again and
        # hangs forever -- after every assertion has already passed, which is a
        # miserable thing to debug.
        if self._stopped:
            return
        self._stopped = True
        self.server.shutdown()
        self.server.server_close()


# --- Fake embedder, as in test_compose ------------------------------------

from app import embedder, router  # noqa: E402

VECTORS = {
    "math": [1, 0, 0, 0], "legal": [0, 0, 1, 0], "general": [0.5, 0.5, 0.5, 0.5],
}
embedder.embed = lambda text: VECTORS.get(text, [0.25, 0.25, 0.25, 0.25])
router._tag_embed_cache.clear()

from app import compose  # noqa: E402
from app.compose import build_aggregation_body, extract_question, plan_panel, run_panel  # noqa: E402
from app.router import score_nodes  # noqa: E402
from app.verify import verify  # noqa: E402


def node(name, tags, embed, url):
    return {"id": name, "name": name, "domain_tags": tags, "capability_embed": embed,
            "cost_per_1k": 0, "avg_latency_ms": 100, "model_name": f"{name}-model",
            "endpoint_url": url, "can_aggregate": False, "api_key_ref": None}


SPANNING = [0.707, 0, 0.707, 0]
BODY = {"model": "auto", "messages": [
    {"role": "user", "content": "My landlord kept my bond of $1,200. "
                               "I paid 8 weeks rent at $340/week. What am I owed?"}]}


async def main() -> None:
    maths = StubNode("The rent paid is 340 * 8 = 2700. The bond is 1200.")
    legal = StubNode("Under s.63 of the Residential Tenancies Act 1995 (SA) the bond "
                     "must be returned. Bond is 1200. Writing is outside my domain.")
    agg = StubNode("You are owed your $1,200 bond back under s.63.")

    try:
        nodes = [
            node("mathstral", ["math"], VECTORS["math"], maths.url),
            node("cgla", ["legal"], VECTORS["legal"], legal.url),
            node("generalist", ["general"], VECTORS["general"], agg.url),
        ]
        scored = score_nodes(nodes, SPANNING)
        plan = plan_panel(scored, SPANNING)

        print("\nplanning")
        check("composes", plan.compose, True)
        check("panel members", sorted(m.name for m in plan.members), ["cgla", "mathstral"])
        check("aggregator", plan.aggregator["name"], "generalist")

        print("\nfan-out over real HTTP")
        answers = await run_panel(plan, BODY)
        check("both specialists answered", sorted(answers), ["cgla", "mathstral"])
        check("maths node was called once", len(maths.received), 1)
        check("legal node was called once", len(legal.received), 1)

        # The gateway must rewrite the model name to the node's own, or a node
        # serving 'mathstral' gets asked for 'auto' and 404s.
        check("model name rewritten per node",
              maths.received[0]["model"], "mathstral-model")
        check("streaming disabled for panel members",
              maths.received[0].get("stream"), False)

        # The specialist framing must be an extra leading system turn, with the
        # user's own messages preserved untouched.
        msgs = maths.received[0]["messages"]
        check("role prepended as system turn", msgs[0]["role"], "system")
        check("specialist told its domain", "math specialist" in msgs[0]["content"], True)
        check("user message preserved", msgs[-1]["content"], BODY["messages"][0]["content"])
        check("legal node got its own domain, not maths",
              "legal specialist" in legal.received[0]["messages"][0]["content"], True)

        print("\nverification across the panel")
        report = verify(answers)
        check("caught the wrong arithmetic",
              [(c.excerpt, c.ok) for c in report.checks], [("340 * 8 = 2700", False)])
        check("agreeing bond figure not flagged", report.disagreements, [])
        check("recomputed value is in the report", "2720" in report.as_prompt_section(), True)

        print("\naggregation request")
        agg_body = build_aggregation_body(
            BODY, extract_question(BODY), answers, report, plan.members, False)
        sys_prompt = agg_body["messages"][0]["content"]
        user_prompt = agg_body["messages"][1]["content"]
        check("aggregator gets its own system prompt", "aggregator" in sys_prompt, True)
        check("aggregator not told to hide conflicts", "do not silently pick one" in sys_prompt.lower(), True)
        check("both specialists' answers included",
              all(a in user_prompt for a in answers.values()), True)
        check("verification section reaches the aggregator",
              "INDEPENDENT VERIFICATION" in user_prompt, True)
        check("original question reaches the aggregator",
              "landlord" in user_prompt, True)

        print("\ndegradation: one member down")
        legal.stop()
        answers2 = await run_panel(plan, BODY)
        check("failed member drops out rather than failing the request",
              list(answers2), ["mathstral"])

        print("\ndegradation: whole panel down")
        maths.stop()
        check("empty panel returns nothing to compose", await run_panel(plan, BODY), {})

        print("\ntimeout is bounded")
        slow = StubNode("too late", delay=1.5)
        try:
            from app.config import settings
            original = settings.compose_member_timeout_seconds
            settings.compose_member_timeout_seconds = 0.3
            slow_plan = plan_panel(
                score_nodes([node("slow", ["math"], VECTORS["math"], slow.url),
                             node("cgla2", ["legal"], VECTORS["legal"], agg.url),
                             node("gen2", ["general"], VECTORS["general"], agg.url)], SPANNING),
                SPANNING)
            out = await run_panel(slow_plan, BODY)
            check("slow member dropped, fast member kept", "slow" in out, False)
            check("the request still produced an answer", len(out) >= 1, True)
        finally:
            settings.compose_member_timeout_seconds = original
            slow.stop()
    finally:
        for stub in (maths, legal, agg):
            try:
                stub.stop()
            except Exception:
                pass

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED\n")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("all panel-flow tests passed")


asyncio.run(main())
