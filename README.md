# Common Network

![Common.](assets/common-banner.png)

**v0.4**

Common Network is a distributed LLM inference network. It splits one large
model's layers across volunteer machines, in the style of
[Petals](https://github.com/bigscience-workshop/petals), so a model too big for
one computer can run on many ordinary ones. There is no central inference
server: peers find each other, build their own routes, and replace a peer that
drops out mid-generation.

A South Australian nonprofit building decentralised AI. [commonai.com.au](https://commonai.com.au)

## Install and join

macOS / Linux:

```bash
curl -fsSL https://commonai.com.au/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://commonai.com.au/install.ps1 | iex
```

Open a new terminal, then:

```bash
common join                      # serve a slice of the default model
common ask "What's a good way to learn recursion?"
```

`common join` sets itself up on first run (private Python environment, model
runtime, code), picks the layers the network needs most that fit your memory,
downloads only those layers, and starts serving. Ctrl+C leaves cleanly. Add
`--permanent` to keep serving after reboots.

Backends: Apple silicon (MLX), NVIDIA GPUs on Windows/Linux (NF4), and CPU-only
machines (PyTorch, slower; routes prefer GPU peers). The NVIDIA path has not yet
been verified on real hardware; see the status notes in
[docs/layer-network.md](docs/layer-network.md).

### Choosing a network

```bash
common join --network bonsai-8b        # also: bonsai-2-9b, bonsai-27b, bonsai-2-27b, qwen3-8b
common join --network URL              # a network a friend runs (any peer's URL)
common join --network manifest.json
```

A network is defined by a manifest: the model and the fixed layer boundaries
every peer's window is made of. See [docs/bonsai-networks.md](docs/bonsai-networks.md).

### Useful `join` flags

| Flag | Effect |
|---|---|
| `--local-front N\|auto\|on` | Hold the first layers on your machine and run them in your own chat client, so peers never see your prompt tokens (they get layer-N hidden states, which are not encrypted) and one network hop disappears. Costs memory (about 1 GB for 9 layers of Qwen3-8B 4-bit). Off by default. |
| `--direct` | Experimental. Serve over TLS on your own public address and be found through a Kademlia DHT, with the Cloudflare tunnel only as fallback. A plain `common join` already tries this automatically; `--no-direct` forces tunnel only. |
| `--no-draft` | Turn off speculative decoding in the local chat API. |
| `--layers A-B` | Serve exactly this window instead of the one the network picks. |
| `--no-api` / `--api PORT` | Skip or move the local OpenAI-compatible API (default `http://127.0.0.1:11440/v1`). |

## How it works

- **Token-ID chain.** The client sends token IDs to the first peer, peers pass
  hidden states along the chain, and the last peer returns the next token ID.
  Only small activations travel between peers. See
  [docs/token-id-chain.md](docs/token-id-chain.md).
- **Link compression.** Hidden states between peers are sent as int8 by default
  (one scale per 64 values). `COMMONNET_ACTIVATION_CODEC=int4_group` selects int4
  (smaller, with a quality cost); `float` disables compression. Older peers fall
  back automatically.
- **Common Flash speculative decoding.** On by default for greedy requests
  (`temperature` 0; `common ask` sends 0). A small local drafter guesses several
  tokens and the network verifies them in one pass, with output identical to
  non-speculative greedy decoding. The drafter is MLX on Macs and PyTorch
  elsewhere. Not available for networks whose model cannot roll back (Bonsai 2
  9B/27B). Gains on a real wide-area network vary and are still being measured.
  See [docs/flash-drafter.md](docs/flash-drafter.md).
- **Resilience.** If another peer holds the same layers, the client switches to
  it and replays what it sent; if a stage loses every peer, a spare peer moves
  over.
- **Discovery.** Peers gossip signed records. The production gateway also runs a
  small directory (rendezvous) so `common join` needs no typed address; it never
  has to be trusted, since readers verify signatures. Direct mode adds a DHT.

Everything is documented in [docs/layer-network.md](docs/layer-network.md).

### Honest limits

- The network is open. Peers see activations, which reveal a lot about the text.
  Do not send private prompts over a public network.
- Nothing proves a peer computes honestly; treat it as cooperative, like early Petals.
- Single-stream speed is bounded by the number of stages and the network. More
  peers add capacity and resilience, not speed.
- Direct mode and the NVIDIA and CPU paths are experimental or lightly tested.

## Commands

```bash
common join           # put this machine on the network
common ask "..."      # ask the network (or `common` for an interactive session)
common ask --temperature 0.7 "..."   # sampled instead of greedy (no speculation)
common peers          # who else is on the network right now
common status         # your node: health and requests served
common report         # an answer was bad: send the evidence (only after you approve it)
common privacy        # what the network keeps, and what it doesn't
common leave          # take your machine off the network
common help           # everything else
```

Any OpenAI-compatible client can use the local API at `http://127.0.0.1:11440/v1`
while `common join` is running.

## Running your own gateway

This section is for running a gateway of your own — a school, a lab, a fork.

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

Three suites, all stdlib-only, all run from their own directory:

```bash
cd gateway && python tests/run_all.py    # the gateway: 14 suites
cd join    && python test_join.py        # the worker and the runtime: 66 checks
cd common  && python test_common.py      # the CLI: 128 checks
```

No pytest, no database, no network beyond localhost, no embedding model — the
suites stub what they need, so there is no reason not to run them. There is no
CI yet, which is exactly why they are listed here: a test nobody runs is
decoration.

All three are safe to run on a working machine. `test_common.py` points the CLI
at a temporary install directory and a gateway that does not exist; it never
reads or writes `~/.common-network`.

## Endpoints

| Endpoint | What it gives you |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible. `X-Common-Compose: never\|auto\|always` overrides composition per request. Contribution-gated by default: send a registered node's token (`X-Common-Node-Token`, or the `Authorization: Bearer` API-key slot). |
| `GET /nodes`, `POST /nodes`, `DELETE /nodes/{id}` | The registry. Registration is permissionless. |
| `GET /dashboard` | Public status: nodes, coverage, catalogue, routing, recent decisions. |
| `GET /admin?token=…` | Operators view: failing nodes, error rates, limiter state. 404s unless `ADMIN_TOKEN` is set. |
| `GET /source` | The repository this instance was built from (AGPL §13). |
| `GET /decisions/recent?topology=panel` | The routing log, filterable by topology. |
| `GET /decisions/composition` | How often each topology runs, and what the verifier caught. |
| `GET /demand/gaps` | Under-served domains, and demand nothing in the catalogue covers. |
| `GET /demand/plan?machines=20&ram_gb=8` | An install plan for a set of machines. |
| `GET /catalogue`, `POST /assign` | The specialist catalogue, and what a given machine should run. |
| `WS /nodes/{id}/connect` | Authenticated outbound-only Android/iOS worker channel; see [the protocol](gateway/MOBILE-NODES.md). |

## The catalogue

`catalogue/catalogue.seed.yaml` is the source of truth for what the network will
auto-install. The live catalogue currently contains Bonsai 2 2B/4B/9B and the
distinct first-generation Bonsai 8B. Bonsai 2 uses ternary `Q2_0` GGUF; the 8B
entry uses the upstream-compatible `Q2_0_g64` GGUF. Stock upstream llama.cpp
needs build b11555 or later for Bonsai 2's Hadamard-activation runtime and does
not support the smaller `PQ2_0`/`PTQ1_0` packings. The non-`g64` Bonsai 8B
layout is fork-only. Catalogue requirements and format caveats are exposed by
`GET /catalogue` and `POST /assign`; do not substitute standard Ollama models.
Mobile contribution eligibility and its protocol are documented in
[gateway/MOBILE-NODES.md](gateway/MOBILE-NODES.md).

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

## Legacy gateway network (v0.1)

The original network routed each request to one whole-model node chosen by a
gateway (`common join --legacy`, `--lan`, `--model`, the catalogue, and the
optional multi-specialist composition mode, which is off by default and has not
been shown to answer better than a single node). It is kept for compatibility
and for operators who run their own gateway. The layer network above is the
current design; the earlier operator-managed RPC-group prototype has been removed.

## Licence

Copyright (C) 2026 Common AI Inc. Licensed under [AGPL-3.0](LICENSE).

Chosen deliberately: copyleft means anyone running a modified version of Common
as a network service must release their changes back to the commons —
structurally preventing this from being taken closed, in keeping with the
project's anti-enclosure thesis.

That obligation is section 13, and this gateway honours it: **`GET /source`**
returns the repository a running instance was built from. If you fork Common
and run it as a service, set `SOURCE_URL` to your own repository — your users
are entitled to the code you are actually running, not to ours.

The **name** is not covered by the AGPL. Fork the code freely; give your fork
its own name. See [TRADEMARK.md](TRADEMARK.md) — it is short, and permissive
about everything except calling a modified version "Common".
