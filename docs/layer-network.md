# The layer network

A peer-to-peer network that runs one large model by splitting its layers across everyone's machines.
Each peer holds a few layers; only the small activations travel between peers (about 20 KB per token).
There is no central server: peers find each other by gossip, build their own routes, and replace a peer
that drops out in the middle of a generation.

`common join` joins this network by default. The original whole-model gateway node is still there as
`common join --legacy` (and anything that only makes sense for it, such as `--model` or `--lan`, implies
it), and a machine that cannot serve layers (no Apple silicon or NVIDIA GPU) keeps running as a
whole-model node. A background service keeps the mode it was installed with.

## Join (any machine with a supported GPU)

```bash
common join                      # the built-in network (same as --network)
common join --network URL        # a network a friend runs: any peer's URL
common join --network manifest.json
common join --permanent          # keep serving after reboots (records the mode)
common join --legacy             # the original whole-model gateway node instead
```

The first run sets itself up: it makes a private Python environment, installs the model runtime, fetches
the code, picks the layers the network needs most that fit your memory, downloads only those layers, opens a
Cloudflare tunnel (no ports to forward), and starts serving. Ctrl+C leaves cleanly. Running it again
resumes. Nothing is registered with any gateway.

Supported: Apple silicon Macs (MLX) and NVIDIA GPUs on Windows/Linux (NF4). Machines without a GPU are
refused. **The NVIDIA path is written but has not been run on real hardware yet.**

## Chat with it

```bash
python -m commonnet.serve --network URL     # OpenAI-compatible API on http://127.0.0.1:11440/v1
```

Any OpenAI-compatible app can point at it. Each request builds a fresh route from the live peers.

## Start a network (be a seed)

Run `common join --network` with nobody else around and you are the network's first peer. Share the
tunnel URL it prints; others join with `common join --network <that URL>`. To make a peer the default
for everyone, list its URL in `commonnet/networks/seeds.json` on `main` (fetched at join time, no release
needed). A network is defined by its manifest (`commonnet/networks/main.json`): the model, and the fixed
stage boundaries every peer's window is made of.

## What survives a dropout

- **A peer dies while others hold the same layers** — the client switches to a replica and replays what it
  already sent, so the generation continues with identical output. Measured on Qwen3-8B with a peer
  killed mid-generation: 1.4 s pause, output identical to an unbroken run.
- **Nobody else holds those layers** — the client waits (30 s by default) for a peer to come back or
  join with that window, then continues.
- **A stage loses every peer** — a spare peer from a better-covered stage moves over, one at a time. Measured
  on Qwen3-8B with five local peers: the only holder of the last stage was killed and a spare was serving
  it again 111 s later (peers re-check every 60 s), after which generation worked.
- **Tunnel addresses change** — peers re-publish their current address every 15 s, so nothing needs
  re-entering.

Replicas should run the same backend: switching between MLX and NF4 mid-generation continues but can
drift slightly because their 4-bit weights differ.

## Honest limits

- **It is open.** A peer's block token is published in the mesh, so anyone can send it work. Peers see
  activations, which reveal a lot about the text. Do not put private prompts on a public network.
- **Nothing proves a peer computes honestly.** Treat it as cooperative, like early Petals.
- **Speed is bounded by the number of stages and the network.** Each token passes every stage and the
  client; more peers add resilience and capacity, not single-stream speed.
- **No default seeds yet.** The seed list in the repo is empty until an operator lists a peer.
- The first peer of a stage needs enough memory for its window; the network tells you how much if you
  have too little.

Design details: `commonnet/NETWORK.md`.
