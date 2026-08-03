# common — the COMMON. CLI

The full network client: ask questions, contribute a node, watch demand,
see who's on the commons. Installed by the same one-liner as everything
else — see the [top-level README](../README.md#contributing-a-node-for-friends).

## Commands

```
common                    interactive session (just type a question, or a /command)
common ask "<prompt>"     route one question through the network
common join               put this machine on the commons as a node
common serve <model>      contribute a specific model
common leave              take this machine off the commons
common status             this node: health, position, requests served
common demand             live domain coverage gaps (real data, not a mockup)
common peers              connected nodes and their coverage
common contrib            your contribution ledger
common whoami             your node identity
common config             settings, and exactly what the network retains
common test               benchmark every node + routing, log to jsonl
common help [verb]        help, per verb
```

Short alias: `cmn` does the same thing as `common`.

## `common test`

A network measurement, not a quality benchmark — `bench/` owns scoring
against real datasets. This answers the three things you need while
standing in a room with several machines:

1. **Is every node reachable?** Sweeps every healthy node, forcing each one
   via `X-Common-Node` (which also disables gateway fallback, so a failure
   surfaces as *that node's* failure rather than being masked by the
   runner-up).
2. **How fast is each one?** Time-to-first-token and total latency, reported
   as p50/p90 — never means, because the distribution is long-tailed and a
   mean hides exactly the cold-start behaviour that matters. Plus an
   approximate tok/s.
3. **Does routing work?** Runs the same probes through auto-routing and
   checks whether each domain landed on a node tagged for that domain.

```bash
common test                      # one pass over every healthy node
common test --full               # 3 repeats per probe
common test --repeats 5
common test --nodes-only         # skip the routing check
common test --routing-only       # skip the per-node sweep
common test --out results.jsonl
```

Every measurement is written to `~/.common-network/tests/<run>.jsonl`
(flushed per record, so a Ctrl+C mid-run still leaves usable data). The
first line is a manifest: client machine, OS, gateway, gateway RTT, and the
full node roster at run time.

**Run it on every machine and merge the files.** Each one records its own
client-side view — network asymmetry between participants is invisible from
any single vantage point. Work is interleaved with a fixed seed rather than
run node-by-node, so a slow node isn't confounded with time-of-day
variation on someone's home uplink.

Known gaps, both blocked on gateway-side work: token counts are approximated
from SSE chunk counts (the gateway doesn't forward usage yet), and fallback
events are invisible in the decisions log (a successful retry is recorded
identically to a first-attempt success).

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

test only:
--full               3 repeats per probe
--repeats <n>        repeats per probe per node (default 1)
--nodes-only         skip the routing check
--routing-only       skip the per-node sweep
--out <path>         results jsonl path
```

## What's not built yet

- **`common synth <region>`** — would trigger Soup-of-Experts weight merging
  to fill a coverage gap. Real weight merging doesn't exist yet (explicitly
  out of scope through v0.3). Run `common help synth` for the honest
  explanation rather than a fake simulation.
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
every `ask` — node, score, latency, and what was actually retained (an
embedding for demand analytics, not the raw question — the footer says so
honestly rather than claiming blanket "no data retained").
