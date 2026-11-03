# The layer network

A peer-to-peer network that runs one large model by splitting its layers across everyone's machines.
Each peer holds a few layers; only the small activations travel between peers (about 20 KB per token).
There is no central server: peers find each other by gossip, build their own routes, and replace a peer
that drops out in the middle of a generation.

`common join` joins this network. `--model NAME` for a layer-network name (`qwen3-8b`, `bonsai-8b`,
`bonsai-27b`, `bonsai-2-9b`, `bonsai-2-27b`) is the same as `--network NAME`. The old whole-model
gateway (`--legacy`, `--lan`, or a `--model` that is not one of those names) has been retired: those
flags still parse, then say so and exit.

## Join (Apple silicon, NVIDIA, or any machine with enough RAM)

### Node-count presets

Run the same preset on each participating machine:

```bash
common join --1                  # Qwen3-8B, one-node milestone
common join --2                  # Qwen3-8B, slices capped at half the model
common join --3                  # Qwen3.5-35B-A3B, up to 16 layers per peer
common join --4                  # same 35B network, add capacity and replicas
common join --5                  # same 35B network
common join --6                  # same 35B network
common join --7                  # Qwen3.5-122B-A10B, up to 7 layers per peer
common join --nodes 7            # long form of --7
common join --7 --permanent      # preserve the preset after a reboot
```

The count is the planned deployment size, not a peer number. Every machine uses the same command;
there is no manual assignment of node 1, node 2, or layers. The joiner checks the directory and gossip
for online peers on that model, then chooses a missing slice that fits its memory. Once a complete
route exists, extra peers prefer exact replicas of existing slices for failover. Presets 1/2 share
the default 8B network; presets 3–6 share the 35B network. The 122B model has its own network. Models
are not switched automatically when peers arrive or leave.

All expected layers must be hosted before chat can work. The progress view waits for both a complete
route and the local chat API; debug logs identify missing layers and show which model was selected.
If two machines choose the same slice while joining concurrently, normal gossip and rebalancing can
move a redundant peer into a gap. Peers with `--layers` explicitly set remain pinned.

Capacity still depends on the hardware. Seven arbitrary machines do not guarantee enough memory
for 122B. Placement uses conservative per-backend estimates, including endpoint tables and runtime
headroom, and refuses a machine that cannot hold even one stage before downloading weights.
Apple silicon uses the MLX 4-bit checkpoint. NVIDIA downloads only its assigned BF16 tensors and
quantises projections and routed experts to NF4 on load, so its disk download/cache is larger than
its GPU weight footprint. CPU peers use BF16 weights and need considerably more RAM. The manifest
memory estimates are planning estimates, not measured full-model benchmarks.

The Qwen3.5 presets serve text, with speculative decoding disabled because their recurrent attention
cache cannot be rolled back. Tiny real MoE checkpoints are checked against Transformers, including
seven local HTTP servers and the packed MLX format. Full-size 35B/122B deployment and CUDA throughput
still require testing on the intended machines. `--check` is a general machine check; model-specific
memory checks happen during join. Presets cannot be combined with `--network` or `--model`; use a
model name directly for a custom deployment.

```bash
common join                      # the built-in network (same as --network)
common join --network URL        # a network a friend runs: any peer's URL
common join --network manifest.json
common join --network bonsai-8b          # Ternary Bonsai 8B
common join --network bonsai-27b         # Ternary Bonsai 27B
common join --network bonsai-2-9b        # Ternary Bonsai 2 9B
common join --network bonsai-2-27b       # Ternary Bonsai 2 27B
common join --network qwen3-8b           # default model
common join --model bonsai-8b            # the same as --network bonsai-8b, named as a model
common join --layers 27-36       # serve exactly this window (a stage boundary pair) instead of the one the network picks
common join --check              # can this machine join? reads only: nothing is installed, downloaded or started
common join --debug              # show the full setup logs from the start
common join --permanent          # keep serving after reboots (records the mode)
```

Joining shows the Common ASCII banner above one overall percentage for setup, model downloads, and
connecting. The setup stages are weighted estimates; download progress uses the bytes received. Press **d** (no Enter needed) to reveal
the recent setup details and live logs, then **d** again to return to the percentage. Warnings and errors
stay visible. Full logs are saved under `~/.common-network/logs/`; when output is redirected, use
`--debug` for details. Ctrl+C leaves the network.

### Laptop sleep and wake

Keep the `common join` terminal open. When the laptop sleeps, the join process is suspended by the OS;
other peers stop receiving renewals and mark this node offline after its lease expires. On wake, the
running process refreshes peer discovery and its DHT connection, checks the tunnel, and replaces it if
it did not survive sleep. The model stays loaded, so it does not need to download its layers again.
This reconnect happens automatically; it does not prevent the laptop from sleeping.

### Switching the model

The model a machine serves is the network it joins. Name a different one with `--model` (or `--network`):

- **In a terminal:** press Ctrl+C, then run the new command, for example `common join --model bonsai-2-9b`. Each
  network keeps its own identity under `~/.common-network/net/`, so switching does not reuse another network's node.
- **As a background service:** run the new command with `--permanent`, for example
  `common join --model bonsai-2-9b --permanent`. It replaces the installed service and restarts it, so the old model
  stops. On Linux this is `systemctl --user restart`; on macOS launchd reloads the agent. Windows (Task Scheduler) has
  not been checked for an in-place switch: run `common leave` first, then the command with `--permanent`.
- `--model` takes a layer-network name (`qwen3-8b`, `bonsai-8b`, `bonsai-27b`, `bonsai-2-9b`, `bonsai-2-27b`). Any other
  value was a catalogue id for the retired whole-model gateway and now exits with that explanation.

The first run sets itself up: it makes a private Python environment, installs the model runtime, fetches
the code, picks the layers the network needs most that fit your memory, downloads only those layers, opens a
Cloudflare tunnel (no ports to forward), and starts serving. Ctrl+C leaves cleanly. Running it again
resumes. Nothing is registered with any gateway.

Before the first join, `common join --check` looks at this machine and reports each thing a join needs: the
platform, GitHub and pypi.org (the code and packages), disk space, memory (GPU memory on NVIDIA), and the
rendezvous directory. A failure in any of those except the directory stops `common join` before it downloads
anything, and says what to do. The directory is only a warning: a node still finds its peers through the seed
list without it, and the node log says whenever it is listed on the directory or cannot be.

Supported: Apple silicon Macs (MLX), NVIDIA GPUs on Windows/Linux (NF4), and **CPU-only machines** (Windows,
Linux, Intel Macs) running the unquantised model in bfloat16 on the CPU with plain PyTorch. A CPU peer is
supported but slower than a GPU peer: it offers `min(60% of free RAM, 50% of total RAM)`, and a 9-layer window
of Qwen3-8B (`Qwen/Qwen3-8B`, about 0.36 GiB per layer) needs about 3.6 GiB, so a PC with 8 GB of free RAM can
serve one stage and a 31 GB PC can serve several. Routes prefer GPU peers when the number of hops is the same
(a CPU peer is never the cheapest choice for a hop a GPU peer also serves), and a CPU peer still adds capacity
and replicas. Replicas are same-backend only (see below). A machine with too little RAM for even one stage is
told how much it needs and exits. **Measured on an M-series Mac's CPU (4 torch threads, 9 Qwen3-8B layers): bf16 decode 58 ms/token, but bf16
prefill of 64 tokens took 9.3 s because that CPU has no fast bf16 matmul (float32: 0.40 s prefill, 120 ms/token
and twice the RAM). Speed on x86 depends on AVX512-BF16/AMX support and has not been measured; A CPU peer picks its dtype itself: bfloat16
when the CPU has AVX512-BF16 or AMX, otherwise float32 if the smallest stage still fits at twice the memory (it then offers half as
much to the network, in bfloat16 units), else bfloat16. `COMMONNET_DTYPE` overrides the choice. The NVIDIA path is written but has not been run on real hardware yet, and
the CPU path has been run on a Mac's CPU only, not on Windows.**

## Chat with it

```bash
python -m commonnet.serve --network URL     # OpenAI-compatible API on http://127.0.0.1:11440/v1
```

Any OpenAI-compatible app can point at it. Each request builds a fresh route from the live peers.

### Faster decoding with a drafter (on by default for greedy requests)

```bash
python -m commonnet.serve --network URL --draft          # Qwen3-0.6B drafter on your machine, 8 tokens per round
```

Common Flash is on by default in the node chat API (`/v1/chat/completions`). A request uses the drafter only when it is
greedy (`temperature` 0; a request with no `temperature` gets the API default 0.7 and is sampled, so it does not), the
network's model allows it (Qwen3-8B and Bonsai 8B yes; Bonsai 2 9B/27B no: hybrid linear attention cannot roll back),
every hop on the route advertises `chain_v1`, and the Qwen3-0.6B-4bit drafter loads (mlx_lm, memory). Anything else
falls back to normal decoding silently; output is identical to non-speculative greedy, k=4. The drafter loads on the
first eligible request and is then cached. `common ask` now sends `temperature: 0` (override: `common ask --temperature 0.7 "..."`).
Opt out with `common join --no-draft` (or `commonnet.node --no-draft`) or `COMMONNET_DRAFT=0`. Not yet tested on a real WAN.

A small local model guesses 8 tokens, the network verifies them in one pass and rolls back the rest, so per-hop latency is
shared across several tokens: about 1.7x on chat and 2.3 to 3.1x on code and quoted text at 50 ms hops, with identical greedy
output. Greedy only (`temperature: 0`); it switches itself off against older peers. See [flash-drafter.md](flash-drafter.md).

## Local first layers for chat

`common join` now tries to keep the first model stage on the same device as the chat client. When enabled, the client
embeds and runs layers 0..F locally, then sends the layer-F hidden state straight to the peer that holds layer F onward.
That removes the first network hop from each request and keeps prompt tokens off the network. Speculative decoding helps
later tokens; this local stage is the change intended to reduce time to the first token. Replies are token IDs as before; drafts,
speculative verify, rollback and replica failover work unchanged (a failed remote peer is replayed from retained layer-F
hidden states).

- **Saves** the client-to-first-peer hop (about one round trip per token) and means no peer ever sees your prompt tokens: peers
  only receive layer-F hidden states (still activations, not encrypted, so this is less exposure, not privacy).
- **Costs** memory: about 1 GB for 9 layers of Qwen3-8B 4-bit, about 0.5 GB for Bonsai 8B, about 1.5 GiB NF4. If this node
  does not already serve the first stage, it may download an extra copy. It helps only the machine that asks; other clients
  still use the network as before. A node that serves from layer 0 reuses that block (loaded once).
- The default `auto` benchmarks the first stage once at startup and enables it only if one decode token through it takes
  under 25 ms (median) and at least 1.5x its weight size is free. Otherwise it leaves that stage on the network and says why.
- Needs a peer whose window starts at F. If none does, the route silently uses the whole model on peers (the front is idle).
- Use `common join --local-front off` or set `COMMONNET_LOCAL_FRONT=off` to opt out. Use `--local-front on` to request the
  first stage without the speed test, or pass a stage boundary such as `--local-front 18` to run more layers locally.

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
common join --network                # peer + OpenAI-compatible API on http://127.0.0.1:11440/v1 (the API is on by default)
common join --no-api                 # peer only: skips the ~1 GB client part, so `common ask` and chat apps will not work
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

## See the layer assignment

```bash
python -m commonnet.coordinator --network URL --plan    # JSON: the chain, every peer's window, idle peers
```

The plan covers every layer exactly once (the chain), puts the other peers on windows as replicas, and changes as few
peers' windows as it can when one joins or drops. If no chain of live peers covers the model it exits 1 and names the
layers nobody reaches. Peers report their memory (`gib`), device (`backend`) and a measured rough `speed` in their signed
record. See [network-contract.md](network-contract.md).

## Start a network (be a seed)

Run `common join --network` with nobody else around and you are the network's first peer. Share the
tunnel URL it prints; others join with `common join --network <that URL>`. To make a peer the default
for everyone, list its URL in `commonnet/networks/seeds.json` on `main` (fetched at join time, no release
needed). A network is defined by its manifest: the model, and the fixed stage boundaries every peer's window is made of. `--network` accepts the built-in names `qwen3-8b`, `bonsai-8b`, `bonsai-27b`, `bonsai-2-9b`, and `bonsai-2-27b`, as well as a manifest path or peer URL. See [the Bonsai manifests](bonsai-networks.md) for their runtime and hardware details.

## Direct mode (experimental)

Off by default. Without `--direct` nothing in this section applies and the Cloudflare tunnel plus gossip path
runs exactly as before. With it, a peer serves over TLS on its own public address and is found through a
Kademlia DHT (UDP), so a client needs no seed URL.

```
common join --direct --bootstrap BOOT.HOST:7480 --direct-port 7471 --public-ip YOUR.IP --external-port 7471
python -m commonnet.node --direct --bootstrap BOOT.HOST:7480 --stage-gib 4      # without the launcher
python -m commonnet.serve --bootstrap BOOT.HOST:7480                              # a client: no --network, no URL
python -m commonnet.bench --live --bootstrap BOOT.HOST:7480                       # per-hop kind: direct / tunnel / plain
```

- **Address.** `--public-ip` / `--external-port` (a port you forwarded by hand, a first-class path) else an automatic
  router mapping (UPnP, PCP, NAT-PMP) else the public address STUN reports. Two or three DHT peers are asked to dial
  back; only a successful dial-back (or, for a manual forward when nobody can be asked, your word) publishes
  `https://IP:PORT`. A private, loopback or carrier-grade address is never published.
- **Not reachable** (school or office network, carrier-grade NAT): the peer opens the Cloudflare tunnel as today and
  joins the same mesh through it. `--no-tunnel-fallback` makes it refuse instead and print the port to forward.
- **Trust.** Each peer has a self-signed certificate; its fingerprint (`fp`) and the identity's signature over it
  (`fpsig`) travel in the signed record. Clients pin that fingerprint per peer; a record with a missing or forged
  binding is dropped. Tunnel URLs are unpinned, as before.
- **Mixed meshes.** Direct and tunnel peers gossip and share one table. A node that predates this change still
  verifies and relays the new records (extra fields are ignored), but cannot connect to a direct peer (the certificate
  does not verify), so it fails over to its tunnel peers. Run direct-capable clients to use direct peers.
- **Cold start needs a bootstrap node.** `python -m commonnet.dht --bind 0.0.0.0:7480` on any always-on machine with
  UDP 7480 open; its address goes in `commonnet/networks/bootstrap.json` or `--bootstrap`. None is listed yet.
- **Not built:** relaying for unreachable peers (they use the tunnel), IPv6, automatic switch from direct to tunnel if a
  working address stops working later (the node logs a warning), and any of this verified over the real Internet.

How to test it on real machines: [direct-test.md](direct-test.md). Design notes: [direct-mode.md](direct-mode.md).

## Link compression (hidden states between peers)

Token IDs go to the first peer and come back from the last; the hidden state between two peers is the only large
payload. By default it is sent as **int8, one scale per 64 values** (about 4.3 KB per token instead of 16 KB as float32).
`COMMONNET_ACTIVATION_CODEC=int4_group` on the client switches to **int4, one float16 scale per 32 values** (about
2.3 KB per token); `float` turns compression off. The sender never uses a codec some peer on the route lacks: it steps
down from the one asked for (int4, then int8, then float), so older peers keep working.

Quality (Bonsai 8B at the layer-27 boundary, greedy 40-token outputs on 12 prompts): int8 matched the uncompressed text on
11 and the top token 99.7% of the time; int4 matched on 7 and 96.6%. Neither changes per-token decode time (the payload is
a few KB); the saving is on long-prompt prefill. int8 is the default for that reason; int4 is for people who want the
smaller payload and accept the quality cost.

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

Design details: [network-contract.md](network-contract.md).

## Gateway directory (rendezvous)

The production gateway also runs a small in-memory directory so `common join` needs no typed address. It stores
signed peer records (90 s TTL, newest per peer, capped per network and per source IP) and never trusts them: readers
verify signatures themselves. `POST /announce` (a signed record), `GET /directory?network=ID` (up to 100 live records)
and `POST /check {port, fp?}` (the gateway opens a TLS connection back to the caller's own IP only, ports 1024-65535,
to tell a peer whether it is reachable from outside). Announcements and checks are rate limited per IP. Code: `gateway/app/directory.py`.

On Railway, caller identification uses the edge's documented `X-Real-IP` header
([Railway request headers](https://docs.railway.com/networking/public-networking/specs-and-limits)),
with exactly one public IPv4 or IPv6 literal required. Missing, duplicate or invalid headers return HTTP 400;
the gateway never selects an address from `X-Forwarded-For` or a request body. Outside Railway it uses the socket
peer and ignores forwarding headers. Reachability checks refuse private, loopback, reserved and multicast targets.
Only `COMMON_GATEWAY_TEST=1` outside Railway permits the single-address test header and loopback probes.
Correct caller identification does not open a router's ports: TCP 7471 and UDP 7472 still need reachable mappings
and local firewall rules for direct mode; otherwise the peer uses its tunnel fallback.

## Zero-config join (matrix)

| command | behaviour |
|---|---|
| `common join` | auto-direct: directory lookup, address decision (flags, router mapping, STUN; ports 7471/7472, free port if busy), verified by the gateway `/check` and the DHT; verified => direct, else Cloudflare tunnel. Announces to the gateway directory every ~15 s either way |
| `common join --no-direct` | tunnel only (still uses the directory) |
| `common join --direct ...` | as before: forced direct (tunnel fallback unless `--no-tunnel-fallback`) |
| `COMMONNET_GATEWAY=off` | no directory; `=URL` another gateway |

Gateway down or old: everything behaves as before.
