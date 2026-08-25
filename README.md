# Common Network

The AI intelligence layer is being enclosed by a handful of corporations — the
same way English common land was enclosed and privatised. **Common** is the
counter-enclosure: a permissionless network where anyone can contribute a
model as a node, and requests are answered by the best available capability,
not by a corporate gatekeeper.

Common speaks the OpenAI API. Point any existing OpenAI SDK client at the
gateway and it works unchanged — except every response tells you exactly
which machines answered it, and why. The commons should be legible.
Or just use `common` — a terminal client, no API knowledge required:

```bash
curl -fsSL https://raw.githubusercontent.com/robot-time/common-network/main/install.sh | sh
common ask "What's a good way to learn recursion?"
```

**v0.1.1 is about composition.** v0.1 sent one request to one node, which made
the network a load balancer with good manners: it could never answer better
than its own best machine. This version sends a request that spans domains to
**several specialists at once**, checks their arithmetic deterministically, and
has a third model synthesise one answer.

```
$ common ask "I'm 8 weeks behind on $340/week rent in SA. What do I owe and
              can my landlord evict me?"

  2 specialists answering in parallel
     ├─ mathstral-node
     ├─ cgla-legal-node
     └─ qwen3-node combining
     ⚠ 1 of 3 calculations were wrong — recomputed and corrected
```

## Does it actually answer better?

**Unknown, and this repository is careful not to claim otherwise.**

The honest status: v0.1's experiments (`testing/seam-findings.md`) tested
composition and found **no gain** — then diagnosed exactly why, and the
diagnosis is what this version is built from. Three findings, three design
decisions:

| Finding (v0.1, measured) | What v0.1.1 does |
|---|---|
| Composing models where one dominates the other cannot help — you can only dilute the stronger one. Experiment 2 did this without checking. | Composition is **gated**. The gateway refuses to compose unless different nodes genuinely lead different domains. Every refusal is explained in `X-Common-Compose-Reason`. |
| Chaining specialists *damages* output — routing a draft through a maths model dropped the prose win rate to 0.188. Parallel + aggregator was neutral (0.438). | **Parallel fan-out only.** Sequential chaining is out of scope for a measured reason. |
| The one intervention that beat the noise floor (+0.203): recompute derivable values at the receiver. Four other plausible fixes were all noise. | The aggregator never sees raw specialist output alone — it gets a **deterministic verification report** alongside it. |

What is still unproven is whether a *genuinely* non-dominated panel beats its
own best member. v0.1 could not test it: the pre-flight searched all 411
OpenRouter model IDs and found **no maths-tuned model hosted anywhere on it**.
Common runs local Ollama weights, where `mathstral`, `phi4-mini-reasoning` and
`sqlcoder` all resolve today. The blocker was one hosted catalogue's economics,
not the world — narrow specialists are unprofitable to serve as an API and free
to serve on donated hardware, which is the network's whole thesis.

`testing/compose-test/` is the instrument. **It has not been run against live
models.** Until it has, composition here is a well-founded hypothesis with a
gate on it — not a result.

## How it works

1. An operator registers a node — an OpenAI-compatible endpoint plus a short
   capability profile and domain tags.
2. A client sends a standard `POST /v1/chat/completions`.
3. The gateway embeds the request and scores every healthy node.
4. **It then decides whether to compose:**
   - Does the request span two or more declared domains?
   - Is a *different* node best at each of them?
   - If one node is best at all of them, it is not dominated by anything — route
     to it alone. Composing could only dilute it.
5. **Single route** → forward, fall back to the runner-up once on failure.
   **Panel** → ask every member in parallel, verify, aggregate.
6. Response headers say what happened: `X-Common-Topology`, `X-Common-Panel`,
   `X-Common-Aggregator`, `X-Common-Checks-Failed`, `X-Common-Compose-Reason`.

### Verification

The panel's answers are checked by Python before a model ever sees them:

- **Arithmetic is recomputed.** Any calculation a specialist showed its working
  for is re-derived in `Decimal` and compared. Rounding is not an error; a
  genuine mistake is. The aggregator is handed the *corrected* value.
- **Cross-specialist disagreement is surfaced.** When two specialists state
  different values for the same named quantity, the aggregator is told, and told
  not to silently pick one.

No model verifies another model — a model checking a model reproduces the exact
failure this is meant to catch. It also **flags rather than rewrites**: the
specialist's text is never silently edited, because a corrected answer nobody
can audit is not legible.

It only checks what is genuinely derivable from the text. Where hand-written
derivation rules would be needed, it does nothing, because where those rules
come from at scale is the honest open research question — not something to
paper over.

## Quickstart

Requirements: Python 3.11+, PostgreSQL with `pgvector`.

```bash
cd gateway
uv venv --python 3.11 .venv          # or: python3.11 -m venv .venv
uv pip install -p .venv/bin/python -r requirements.txt

createdb common_network
psql -d common_network -c "create extension if not exists vector;"
DATABASE_URL=postgresql://localhost/common_network python -m app.migrate

cp .env.example .env                 # edit DATABASE_URL / OPENROUTER_API_KEY
.venv/bin/uvicorn app.main:app --reload
```

Upgrading from v0.1: run `python -m app.migrate` — it applies every migration
in order and is idempotent. Existing decision rows read back as
`topology: "single"`, which is what they were.

```bash
curl http://localhost:8000/v1/chat/completions -i \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Explain recursion"}]}'
```

Or open **`/dashboard`**.

### Tests

```bash
cd gateway && python tests/run_all.py
```

No pytest, no database, no network beyond localhost, no embedding model — the
suites stub what they need, so there is no reason not to run them.

## Contributing a node

```bash
curl -fsSL https://raw.githubusercontent.com/robot-time/common-network/main/install.sh | sh
common join                 # over a Cloudflare tunnel
common join --lan           # over the local network — no tunnel, nothing exposed
```

`--lan` is for computer labs and anywhere the gateway is on the same network.
It skips `cloudflared` entirely, and refuses to register if Ollama is bound to
localhost only — otherwise you get a node that health-checks green from its own
machine and is invisible to every other one. See
[`SCHOOL-NETWORK-REQUIREMENTS.md`](SCHOOL-NETWORK-REQUIREMENTS.md).

### What should I install?

```bash
common recommend                   # what the network is short of
common recommend --machines 20     # plan a whole lab at once
```

The lab planner matters more than it looks. Twenty machines each installing the
best model they can fit produces twenty copies of one generalist — a network
that cannot beat its own best node however large it grows. `--machines` spreads
them across lanes and allocates one aggregator, because a panel needs members
who are each best at *something*, not members who are each pretty good at
everything.

## Endpoints

| Endpoint | What it gives you |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible. `X-Common-Compose: never\|auto\|always` overrides composition per request. |
| `GET /nodes`, `POST /nodes`, `DELETE /nodes/{id}` | The registry. Registration is permissionless. |
| `GET /decisions/recent?topology=panel` | The routing log, filterable by topology. |
| `GET /decisions/composition` | How often each topology runs, and what the verifier caught. |
| `GET /demand/gaps` | Under-served domains, and demand nothing in the catalogue covers. |
| `GET /demand/plan?machines=20&ram_gb=8` | An install plan for a set of machines. |
| `GET /catalogue`, `POST /assign` | The specialist catalogue, and what a given machine should run. |

## The catalogue

`catalogue/catalogue.seed.yaml` is the source of truth for what the network will
auto-install. v0.1 listed five general-purpose models; **five generalists cannot
compose**, so v0.1.1 is built around narrow specialists — models clearly worse
than a generalist at most things and clearly better at one. Every `ollama:` tag
was verified against `registry.ollama.ai` before being listed.

Generalists are still there for two jobs, neither of which is being on a panel:
answering requests that don't span domains, and **aggregating**. The `general`
tag is excluded from panel seats entirely — a generalist sitting next to a
specialist in that specialist's own lane is the pairing that lost in
Experiment 2.

**CGLA-Legal is temporarily commented out** of `catalogue.seed.yaml` (2026-08-25).
Uncomment the block to restore it; nothing else needs changing. While it is out,
the network has **no legal coverage at all** — a legal question routes to a
generalist that will answer confidently and without grounding, which is the
failure CGLA exists to prevent.

It is worth restoring, because it is the catalogue's only genuinely
non-dominated specialist. It is not a plain model download and not an API: it is
a causal graph over South Australian and federal statute plus a local Llama model
that only extracts facts and narrates outcomes. Its own benchmark shows it
answering *fewer* legal questions correctly than a frontier model (35/46 vs
Claude Sonnet's 40/46) while refusing **7/7** out-of-scope questions where Claude
refused **0/7**.

Worse on one axis, better on another is exactly what *non-dominated* means — the
precondition v0.1 concluded no available model pair satisfied. Note that
`testing/compose-test/` is built around SA-law cases and needs this entry back
before its result means what it is designed to mean.

## Scope (v0.1.1)

**In scope:** everything in v0.1, plus parallel multi-specialist composition,
gated on non-domination; deterministic arithmetic re-derivation and
cross-specialist disagreement detection; demand-gap and unserved-cluster
analysis; fleet install planning; LAN joining; graph-overlay catalogue entries.

**Still explicitly out of scope:** no DHT/peer-to-peer/consensus, no token or
incentive mechanism, no weight merging or Soup of Experts, no sequential
specialist chaining (measured harmful), no learned router, no vector-native
model-to-model communication, no production-grade auth.

`/demand/gaps` reports demand clusters the catalogue cannot serve — the signal
Soup of Experts would need. It does not merge weights, and says so where the
theory would claim otherwise.

## Licence

AGPL-3.0. Chosen deliberately: copyleft means anyone running a modified version
of Common as a network service must release their changes back to the commons —
structurally preventing this from being taken closed, in keeping with the
project's anti-enclosure thesis.
