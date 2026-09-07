# Common Network

![Common.](assets/common-banner.png)

**The Common Network Alpha — v0.1.2**

## Install and set up

**1. Install [Ollama](https://ollama.com/download)** — the program that runs the
AI model on your machine. Install it before step 2 so the installer can find it.

**2. Install Common.**

macOS / Linux — paste into Terminal:

```bash
curl -fsSL https://commonai.com.au/install.sh | sh
```

Windows — paste into PowerShell:

```powershell
irm https://commonai.com.au/install.ps1 | iex
```

**3. Close your terminal and open a new one.** The old window can't see the new
command yet.

**4. Donate a machine.** This picks a model that fits your computer, downloads
it, and puts you on the network:

```bash
common join
```

Leave that window open — it's your machine being part of the network. (Or run
`common join --permanent` to have it run quietly in the background and start
itself at login.)

**5. Ask the network something.** In a second window:

```bash
common ask "What's a good way to learn recursion?"
```

Or run `common` on its own for a back-and-forth session.

**Why join first?** The network answers its contributors — asking requires a
machine that's donating. `common join` sets that up for you and stores the
credential; you don't have to handle it yourself.

### Useful commands

```bash
common peers          # who else is on the network right now
common status         # your node: health, position, requests served
common recommend      # what specialist the network is short of
common leave          # take your machine off the network
common help           # everything else
```

Using an OpenAI SDK instead? Point it at the gateway and pass your node token
(printed by `common join`, stored in `~/.common-network/identity.json`) as the
API key.

---

The AI intelligence layer is being enclosed by a handful of corporations — the
same way English common land was enclosed and privatised. **Common** is the
counter-enclosure: a permissionless network where anyone can contribute a
model as a node, and requests are answered by the best available capability,
not by a corporate gatekeeper.

Common speaks the OpenAI API. Point any existing OpenAI SDK client at the
gateway and pass your node token as the API key — every response tells you
exactly which machines answered it, and why. The commons should be legible.

**This is the Alpha: a model-donation platform.** Anyone can donate a machine
and a model — a school lab, a spare laptop, a desktop with a spare GPU — and
the network answers every request from the best available donation. One
request, one machine, full visibility into which machine answered and why.
And the network answers its **contributors**: a request must carry the node
token of a registered node, so donated compute serves donors rather than
anonymous bulk traffic.

```
$ common ask "I'm 8 weeks behind on $340/week rent in SA. What do I owe and
              can my landlord evict me?"

  answered by   cgla-legal-node
  chosen from   3 nodes, margin 0.21
```

**Composition is built, but off by default.** v0.1.1 also contains the next
step — sending a request that spans domains to **several specialists at once**,
checking their arithmetic deterministically, and having a third model
synthesise one answer. That is the mechanism a future version will need to be
competitive with a frontier model, but it has not been proven yet, so Alpha
ships with `COMPOSE_MODE=never` and does not claim it. It can be turned on
per-request with `common ask --compose`, per-gateway with `COMPOSE_MODE=auto`,
or tested end-to-end with `testing/compose-test/` — see
["Does it actually answer better?"](#does-it-actually-answer-better) below.

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
2. A client sends a standard `POST /v1/chat/completions`, carrying the token
   of a node it registered. With `REQUIRE_CONTRIBUTION` on — the default — a
   request without a registered node's token is refused with 401. OpenAI SDK
   clients pass the token as the API key; `common` and the chat client pick
   it up from `~/.common-network/identity.json`, which `common join` writes.
3. The gateway embeds the request and scores every healthy node.
4. **In Alpha, the best node answers.** Forward, fall back to the runner-up
   once on failure. If the best specialist's match is weak, a confident
   generalist answers instead.
5. **If composition has been turned on** (`COMPOSE_MODE=auto`/`always`, or the
   `X-Common-Compose` header), the gateway first decides whether a panel is
   worth forming:
   - Does the request span two or more declared domains?
   - Is a *different* node best at each of them?
   - If one node is best at all of them, it is not dominated by anything — route
     to it alone. Composing could only dilute it.
   A panel asks every member in parallel, verifies, and aggregates.
6. Response headers say what happened: `X-Common-Topology`, `X-Common-Node`,
   `X-Common-Panel`, `X-Common-Aggregator`, `X-Common-Checks-Failed`,
   `X-Common-Compose-Reason` — including why a request was *not* composed.

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

## Running your own gateway

Everything above joins the shared network. This section is for running a
gateway of your own — a school, a lab, a fork.

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

The bare curl works locally because the demo `.env` sets
`REQUIRE_CONTRIBUTION=false` (its seed nodes have no tokens). On a gateway
running the shipped default, add `-H "X-Common-Node-Token: <what common join
printed>"` — or pass the token as the API key from any OpenAI SDK client.

Or open **`/dashboard`**.

### Tests

```bash
cd gateway && python tests/run_all.py
```

No pytest, no database, no network beyond localhost, no embedding model — the
suites stub what they need, so there is no reason not to run them.

## Contributing a node: the other options

[Install and set up](#install-and-set-up) covers the normal path. `common join`
also takes flags for labs and long-running machines:

```bash
common join                 # over a Cloudflare tunnel (the default)
common join --lan           # over the local network — no tunnel, nothing exposed
common join --permanent     # run in the background, start at login
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
| `POST /v1/chat/completions` | OpenAI-compatible. `X-Common-Compose: never\|auto\|always` overrides composition per request. Contribution-gated by default: send a registered node's token (`X-Common-Node-Token`, or the `Authorization: Bearer` API-key slot). |
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

## Security

Registration is permissionless — that is the thesis — which makes the security
model worth stating rather than assuming. Using the network is gated on
contributing to it (`REQUIRE_CONTRIBUTION`, on by default), requests are
rate-limited per client, node registration requires the node's private token,
and endpoint URLs are re-validated against cloud-metadata ranges on every
health pass. What a stranger can and cannot do, what an operator of a public
gateway should set, and the known limits (`common test` executes model code;
a node is trusted for identity, not behaviour): see
[SECURITY.md](SECURITY.md).

## Scope (Alpha)

**What Alpha ships as:** a model-donation platform. Permissionless node
contribution, contribution-gated access (the network answers its donors),
per-client rate limiting, legible routing to the best donated machine, health
checking, demand-gap and unserved-cluster analysis, fleet install planning,
LAN joining, graph-overlay catalogue entries.

**Built but off by default:** parallel multi-specialist composition, gated on
non-domination, with deterministic arithmetic re-derivation and cross-specialist
disagreement detection. It runs only where deliberately enabled
(`COMPOSE_MODE`, or `X-Common-Compose` per request) until compose-test has
proven a non-dominated panel beats its own best member against live models.

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
