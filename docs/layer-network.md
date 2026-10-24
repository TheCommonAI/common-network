# The layer network

A peer-to-peer network that runs one large model by splitting its layers across everyone's machines.
Each peer holds a few layers; only the small activations travel between peers (about 20 KB per token).
There is no central server: peers find each other by gossip, build their own routes, and replace a peer
that drops out in the middle of a generation.

`common join` joins this network by default. The original whole-model gateway node is still there as
`common join --legacy` (and anything that only makes sense for it, such as `--model` or `--lan`, implies
it). A background service keeps the mode it was installed with.

## Join (Apple silicon, NVIDIA, or any machine with enough RAM)

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

Supported: Apple silicon Macs (MLX), NVIDIA GPUs on Windows/Linux (NF4), and **CPU-only machines** (Windows,
Linux, Intel Macs) running the unquantised model in bfloat16 on the CPU with plain PyTorch. A CPU peer is
supported but slower than a GPU peer: it offers `min(60% of free RAM, 50% of total RAM)`, and a 9-layer window
of Qwen3-8B (`Qwen/Qwen3-8B`, about 0.36 GiB per layer) needs about 3.6 GiB, so a PC with 8 GB of free RAM can
serve one stage and a 31 GB PC can serve several. Routes prefer GPU peers when the number of hops is the same
(a CPU peer is never the cheapest choice for a hop a GPU peer also serves), and a CPU peer still adds capacity
and replicas. Replicas are same-backend only (see below). A machine with too little RAM for even one stage is
told how much it needs and exits. **Measured on an M-series Mac's CPU (4 torch threads, 9 Qwen3-8B layers): bf16 decode 58 ms/token, but bf16
prefill of 64 tokens took 9.3 s because that CPU has no fast bf16 matmul (float32: 0.40 s prefill, 120 ms/token
and twice the RAM). Speed on x86 depends on AVX512-BF16/AMX support and has not been measured; `COMMONNET_DTYPE=float32`
switches a CPU peer to float32 (and its `gib` cost doubles, so only with RAM to spare). The NVIDIA path is written but has not been run on real hardware yet, and
the CPU path has been run on a Mac's CPU only, not on Windows.**

## Chat with it

```bash
python -m commonnet.serve --network URL     # OpenAI-compatible API on http://127.0.0.1:11440/v1
```

Any OpenAI-compatible app can point at it. Each request builds a fresh route from the live peers.

## Status file (for apps that show what a peer is doing)

`commonnet.node` (so `common join --network`) writes a JSON status document every 2 s and on every change,
atomically (temp file + rename), to `~/.common-network/net/status.json` — or `--status-file PATH`, or the
`COMMONNET_STATUS_FILE` environment variable. The directory is private (0700), the file 0600. Read it; never
write it. It never contains tokens or keys.

```json
{"v":1,"state":"up","network":"common-main-8b-v1","model":"Qwen3-8B","peer":"<32 hex>","window":[9,18],
 "backend":"mlx","gib":9.92,"url":"https://....trycloudflare.com","download":null,
 "stages":[[0,9],[9,18],[18,27],[27,36]],"coverage":[1,1,1,1],
 "peers":[{"peer":"<32 hex>","window":[9,18],"backend":"mlx","state":"up","age_s":0.4,"me":true}],
 "error":null,"api":{"port":11440,"ready":true},"pid":4242,"ts":1761200000.5}
```

- `state`: `starting` -> `downloading` -> `loading` -> `up`; also `moving` (switching to a stage that lost every
  peer, through its own download and load), `leaving`, `stopped` (clean exit) and `error`.
- `window` is the layer range this peer serves, or is about to (it is set while downloading); `null` until chosen.
- `download` is `{"bytes","total"}` while `downloading`/`moving`, from the hub's own progress lines, else `null`.
- `coverage[i]` is how many live peers hold stage `stages[i]`; `peers` lists live peers (fresh in the last 60 s,
  not leaving), this one first (`me`). 0 in `coverage` means the network cannot answer until someone joins.
- `error` is one sentence meant for a person, or `null`. With `state: "error"` the peer is down; a failed chat API
  also fills it while `state` stays `up`.
- Before the network is known `network`, `model` and `peer` are `""` and `backend`, `window` and `url` are `null`.
- **A crashed peer** never writes `stopped`: treat the file as dead when `ts` is older than 15 s and `pid` is not a
  running process.

`scripts/fake_status.py` plays a realistic changing file (download 0-100%, peers joining and leaving, a stage losing its
last peer, a move, the API loading; `--scenario error|crash`) for developing a UI without a model.

## Local chat API (`--api`)

```bash
common join --network --api          # peer + OpenAI-compatible API on http://127.0.0.1:11440/v1
common join --network --api 12000    # another port
```

The same process that serves layers also answers `GET /v1/models`, `GET /health` and `POST /v1/chat/completions`
(stream and non-stream). It binds 127.0.0.1 only and uses this peer's own view of the network: no second gossip loop,
no second copy of any weights, and requests for this peer's own layers go to its local block server rather than out
through the tunnel. The port answers straight away; chat requests get `503 loading` (and `/health` says
`"ready": false`) until the client part of the model — the embedding, final norm and head, **about 1 GB for
Qwen3-8B on a 16 GB Mac, on top of the layers this peer serves** — has downloaded and loaded in the background.
`status.json` shows `api: {"port", "ready"}`. If the port is taken or the model part fails to load, the peer keeps
serving and the status `error` says so. On NVIDIA and CPU peers the client part runs from CPU in fp16; that path is untested.
`python -m commonnet.serve` still runs the API alone, without serving layers.

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
