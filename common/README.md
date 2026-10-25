# common — the COMMON. CLI

The Common Network CLI (v0.4): join the distributed layer network as a peer, ask it
questions, and inspect it. `common join` serves a slice of a model's layers
(`--network NAME` picks bonsai-8b, bonsai-2-9b, bonsai-27b, bonsai-2-27b or
qwen3-8b); `common ask` sends greedy (`temperature` 0) requests so Common Flash
speculative decoding applies, and needs `common join` running for its local API.
See [docs/layer-network.md](../docs/layer-network.md) and the
[top-level README](../README.md). Commands that talk to a gateway (`demand`,
`contrib`, `test`, `--region`) belong to the legacy v0.1 gateway network.

## Commands

```
common                    interactive session (just type a question, or a /command)
common ask "<prompt>"     route one question through the network
common join               serve layers of the shared model (--network NAME, --local-front, --direct, --no-draft)
common serve <model>      legacy: contribute a specific whole model to a gateway
common leave              take this machine off the commons
common status             this node: health, position, requests served
common demand             live domain coverage gaps (real data, not a mockup)
common peers              connected nodes and their coverage
common peers --all        the same, including machines that have left
common contrib            your contribution ledger
common whoami             your node identity
common config             settings, and exactly what the network retains
common help [verb]        help, per verb
```

Short alias: `cmn` does the same thing as `common`.

## Flags

```
--gateway <url>     talk to a different network
--region <id>       bias routing to a region
--model <id>         pin a specific model (ask: refuses rather than silently
                     rerouting if nothing serves it; join/serve: pick from
                     the catalogue or a raw Ollama tag)
--auto              accept join's recommended model without confirming
--local             ask only — never leaves this machine, talks to your
                     local Ollama directly
--json               machine-readable output, no colour, stable-ish schema
-q / --quiet         answer only, no banners/footers
-v / --verbose       extra routing detail
--no-color           also honours the NO_COLOR env var and non-tty output
--no-update          skip the self-update check (for local development)

```

## What's not built yet

- **`common map`** — a vector-space position view. Needs a new gateway
  endpoint exposing embeddings (none is public today) plus a 2D projection.
  `common help map` explains what's missing.
- **Local keypair identity** — `whoami` currently tracks a name-based local
  identity file (`~/.common-network/identity.json`, written by `join.py` on
  registration), not a cryptographic keypair. That's a real architecture
  decision, not something to assume into existence — flagged plainly in
  `whoami`'s own output.

## Design

Built to the COMMON. CLI design system: a four-colour palette (charcoal /
paper / dim / blue+red for attention), a fixed glyph vocabulary (`·` `→`
`←` `✓` `⚠` `✗` `#`), lowercase direct copy, and a transparency footer on
every `ask` — which node answered, how long routing took, time to the first
token, total time, and what was actually retained (an embedding for demand
analytics, not the raw question — the footer says so honestly rather than
claiming blanket "no data retained").

The three timings are separate because they answer different questions.
Routing is the gateway choosing a node and is usually tens of milliseconds;
time-to-first-token is dominated by the chosen machine loading its model
(~57s for a cold 7B on a laptop); total is generation. A single number
labelled "routed in" hid the second inside the first and made a cold node
look like a slow router. The footer also names the split when the wait is
long enough to be a cold start, since that is the one a user can act on.

The blended routing score is deliberately **not** printed. It is
similarity+cost+latency, and the similarity term barely moves: measured
live, the gibberish `asdfgh qwerty zxcvbn` scored above a real Python
question. Shown as a match percentage it tells the user the network assessed
their request and was fairly confident — which is false in a way they cannot
detect. `common ask -v` still prints the raw value for debugging.
