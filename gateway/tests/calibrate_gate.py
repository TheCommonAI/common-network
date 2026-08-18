#!/usr/bin/env python3
"""Does auto-composition fire on the right requests? Check against the REAL embedder.

    python tests/calibrate_gate.py        # from gateway/

Separate from the other suites because it loads BAAI/bge-small-en-v1.5 (~130MB
on first run) and is slow. Run it whenever COMPOSE_DOMAIN_GAP, the quantitative
cues, or the embedding model change — the thresholds in `app/config.py` were
derived from this output and are meaningless if the model underneath changes.

WHAT THIS EXISTS TO PREVENT
---------------------------
The gate was first written with an absolute similarity floor, and the unit
tests passed. Against the real embedder it composed 3 of 3 off-topic requests,
because cosine between a sentence and a short domain tag sits at 0.42-0.64 for
*everything*: "write me a poem about the sea" scores 0.553 on `math`, above an
actual arithmetic question's 0.537. On-topic and off-topic ranges overlap
completely. That is invisible to a test with a stubbed embedder, and it would
have shipped a gateway that composed nearly every request — Experiment 2
(seam-findings.md §2) rebuilt as a default.

WHAT WAS MEASURED, AND WHAT IT MEANS
------------------------------------
- The embedder reliably identifies the ONE domain a request is about (the
  drop-off after rank 1 is 0.172 on-topic vs 0.026-0.038 off-topic).
- It does NOT identify a second. The gap between ranks 2 and 3 is 0.010-0.038
  whether or not the request genuinely spans two domains. Tested with bare tags
  and with full capability paragraphs; neither separates.
- So the second lane comes from a deterministic check instead
  (`compose.has_quantitative_component`), which is why a legal question with
  money in it composes and a poem does not.

Expect ~1 of 12 conservative misses. A miss falls back to v0.1 single-routing;
a false positive is the failure that matters. If false positives appear here,
do not ship.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import embedder  # noqa: E402

embedder.load()

from app.compose import has_quantitative_component, plan_panel  # noqa: E402
from app.router import score_nodes  # noqa: E402


def node(name: str, tags: list[str], capability: str) -> dict:
    return {"id": name, "name": name, "domain_tags": tags,
            "capability_embed": embedder.embed(capability),
            "cost_per_1k": 0, "avg_latency_ms": 100, "model_name": name,
            "endpoint_url": "http://x/v1", "can_aggregate": "general" in tags}


NODES = [
    node("mathstral", ["math", "arithmetic"],
         "Mathstral 7B. Mathematical and quantitative reasoning, arithmetic, algebra, "
         "multi-step word problems and calculations involving money, rates and totals."),
    node("cgla", ["legal", "tenancy"],
         "CGLA-Legal. Statute-grounded South Australian and federal Australian law: "
         "residential tenancy, bonds, employment, consumer rights and legal deadlines."),
    node("coder", ["code", "programming"],
         "Qwen2.5 Coder 7B. Code generation, debugging and multi-file reasoning."),
    node("sqlcoder", ["sql", "databases"],
         "SQLCoder 7B. Plain-English questions into SQL against a given schema."),
    node("qwen3", ["general", "conversation"],
         "Qwen3 8B. General conversation, writing, summarisation and world knowledge."),
]

# (label, request, should_compose)
CASES = [
    ("poem", "Write me a short friendly poem about the sea.", False),
    ("chit-chat", "Hey, how has your week been going?", False),
    ("recipe", "How do I make a good carbonara?", False),
    ("recipe w/ numbers", "How do I make carbonara? I have 2 eggs and 100g of guanciale.", False),
    ("pure legal", "Can my landlord enter without notice in South Australia?", False),
    ("pure code", "Why does my Python list comprehension throw a TypeError?", False),
    # One domain and a calculation, but the SAME node leads both -- the
    # domination gate must refuse rather than compose a node with itself.
    ("pure maths", "What is 17 percent of 4,500?", False),
    ("thanks", "Thanks, that was really helpful!", False),
    ("rent + maths",
     "I'm 8 weeks behind on rent of $340 a week in South Australia and my bond was 4 weeks. "
     "How much do I owe in arrears, and can my landlord end the tenancy?", True),
    ("dismissal + maths",
     "I'm paid $1425 a week and was dismissed with 3 weeks notice plus 2.5 weeks accrued "
     "leave. What is that worth and do I have an unfair dismissal claim?", True),
    ("invoice + legal",
     "A client owes me $5920 on an invoice 60 days overdue, terms say 5% annual interest. "
     "What's the total owing and how do I recover it in SA?", True),
    ("consumer + maths",
     "I paid a 15% deposit on $1899 of furniture that was never delivered. How much is "
     "outstanding and what are my rights?", True),
]

print(f"\n{'request':22} {'quant':7} {'composes':9} panel")
print("-" * 82)
false_positives, misses = [], []
for label, text, want in CASES:
    e = embedder.embed(text)
    plan = plan_panel(score_nodes(NODES, e), e, request_text=text)
    panel = ", ".join(f"{m.name}({m.domain})" for m in plan.members) if plan.compose else "—"
    flag = ""
    if plan.compose and not want:
        flag, _ = "  <-- FALSE POSITIVE", false_positives.append(label)
    elif want and not plan.compose:
        flag, _ = "  <-- missed", misses.append(label)
    print(f"{label:22} {str(has_quantitative_component(text)):7} {str(plan.compose):9} {panel}{flag}")

print()
print(f"false positives: {len(false_positives)}  {false_positives or ''}")
print(f"conservative misses: {len(misses)}  {misses or ''}")
print()
if false_positives:
    print("FALSE POSITIVES PRESENT — do not ship. Composing an off-topic request is the")
    print("failure mode seam-findings.md §2 measured; a miss merely falls back to v0.1.")
    sys.exit(1)
print(f"no false positives. {len(misses)} conservative miss(es), which single-route safely.")
