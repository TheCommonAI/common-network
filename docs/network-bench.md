# Layer-network latency benchmark

`python -m commonnet.bench` measures every latency component of the layer network without a real LLM and with
dials for compute and wire delay. It runs the real path: each node is its own process running the real
`commonnet.node.Node` (gossip, placement, signed records) and `BlockServer` (HTTP/1.1, JSON + base64). The client is
the real `Network` + `HttpTransport` + `BlockClient` (keep-alive, replica failover). Only two things are synthetic:

- **The model** (`commonnet/bench/synth.py`). Each layer is a residual low-rank MLP over a causal running mean of the
  session's earlier positions, so a block holds per-session history like a KV cache. Weights come from a seed, so
  runs are repeatable. `--layer-ms` sleeps per layer, so block compute is a dial. The client side (embedding, final
  norm, head, greedy sampling) is synthetic too: nothing is downloaded. It is also registered as backend `synth` in
  `commonnet.net.factory` (`COMMONNET_BACKEND=synth`, `COMMONNET_SYNTH_LAYER_MS`), and `commonnet/networks/test-synth.json`
  is a ready manifest.
- **The wire** (`commonnet/bench/proxy.py`). A TCP delay proxy sits in front of every node and advertises its own
  URL in the node's record, so all block traffic crosses it. It delays each direction by RTT/2 (plus up to jitter/2),
  charges one extra RTT for a new connection (the TCP handshake), and models loss as one 200 ms retransmission
  timeout per hit chunk. Gossip bootstrap goes direct, not through the proxy.

## What the columns mean

Per decode step (one new token through every hop), taken from `BlockClient.timings` plus client-side timers:

| column | meaning |
|---|---|
| `stages` | hops in the route (one HTTP call per hop per token) |
| `rtt` | emulated round trip per hop, ms |
| `hid` | hidden size, which sets the bytes on the wire |
| `ka` | HTTP keep-alive on or off (off closes the connection after every request) |
| `ms/tok`, `p50`, `p95` | wall time of a decode step: mean, median, 95th percentile |
| `tok/s` | 1000 / mean decode ms |
| `compute` | sum of `server_ms` over hops: time inside `block.forward` (the `Server-Timing` header) |
| `network` | sum of `total_ms - server_ms` over hops: wire delay, TCP, HTTP, JSON and base64 on both ends, server thread wake-up |
| `client` | the rest of the step: embedding, final norm + head, sampling, and tensor conversion in the transport |
| `reuse%` | share of forward calls that reused a kept connection |
| `KB/tok` | request + reply bytes per decode step, all hops |
| `prefill ms` | wall time of the first step. It also includes opening the session on every hop, so with RTT > 0 it pays extra round trips and connection setups (its `client` bucket absorbs these) |

`compute + network + client = ms/tok` by construction. The check that means something is the comparison with the
knobs: `compute` should be near layers x `--layer-ms`, and `network` near hops x RTT plus a small serialisation term.

## Results (this Mac, 2026-10-24)

Apple silicon laptop, loopback, Python 3.14, 12 layers, `--layer-ms 3` (so 36 ms of nominal compute per token), hidden 256
unless stated, 16 prompt tokens, 20 new tokens, 1 warm-up + 2 measured generations. Raw data:
`bench-results/sweep-2026-10-24.json`. All times in ms.

Stages x RTT (the formula `time/token = compute + network + client`):

| stages | RTT | ms/tok | p95 | tok/s | compute | network | client | network - stages x RTT |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 41.8 | 43.2 | 23.9 | 40.9 | 0.6 | 0.3 | 0.6 |
| 1 | 25 | 68.7 | 70.6 | 14.5 | 40.9 | 26.9 | 0.9 | 1.9 |
| 1 | 50 | 93.5 | 95.9 | 10.7 | 40.2 | 52.2 | 1.2 | 2.2 |
| 1 | 100 | 143.4 | 146.0 | 7.0 | 41.1 | 101.6 | 0.8 | 1.6 |
| 2 | 0 | 49.1 | 52.6 | 20.4 | 43.5 | 4.3 | 1.4 | 4.3 |
| 2 | 25 | 101.0 | 106.8 | 9.9 | 44.0 | 55.4 | 1.6 | 5.4 |
| 2 | 50 | 149.5 | 154.2 | 6.7 | 44.5 | 103.8 | 1.2 | 3.8 |
| 2 | 100 | 251.4 | 254.4 | 4.0 | 43.9 | 205.5 | 2.0 | 5.5 |
| 3 | 0 | 50.0 | 54.3 | 20.0 | 43.8 | 4.8 | 1.4 | 4.8 |
| 3 | 25 | 126.3 | 131.0 | 7.9 | 44.1 | 80.5 | 1.7 | 5.5 |
| 3 | 50 | 203.8 | 211.1 | 4.9 | 44.2 | 157.7 | 2.0 | 7.7 |
| 3 | 100 | 354.9 | 359.7 | 2.8 | 44.4 | 308.4 | 2.1 | 8.4 |
| 4 | 0 | 52.9 | 58.1 | 18.9 | 44.5 | 6.6 | 1.8 | 6.6 |
| 4 | 25 | 156.5 | 161.7 | 6.4 | 44.7 | 109.4 | 2.4 | 9.4 |
| 4 | 50 | 255.1 | 266.1 | 3.9 | 44.2 | 208.8 | 2.1 | 8.8 |
| 4 | 100 | 456.8 | 462.3 | 2.2 | 45.1 | 409.4 | 2.3 | 9.4 |

Hidden size (2 stages): bytes grow linearly, time barely moves at these sizes.

| hidden | RTT | ms/tok | network | client | KB/tok |
|---|---|---|---|---|---|
| 256 | 0 | 47.5 | 3.4 | 1.2 | 3.0 |
| 1024 | 0 | 49.8 | 3.9 | 1.7 | 11.0 |
| 4096 | 0 | 51.7 | 3.1 | 3.7 | 43.0 |
| 8192 | 0 | 53.7 | 2.6 | 4.7 | 85.6 |
| 256 | 50 | 149.3 | 105.0 | 1.5 | 3.0 |
| 1024 | 50 | 149.9 | 104.3 | 2.0 | 11.0 |
| 4096 | 50 | 154.2 | 104.5 | 4.4 | 43.0 |
| 8192 | 50 | 159.8 | 105.2 | 7.4 | 85.6 |

Keep-alive, jitter and loss (2 stages, 50 ms RTT, hidden 256):

| case | ms/tok | p50 | p95 | network | reuse% |
|---|---|---|---|---|---|
| keep-alive on | 149.9 | 150.0 | 152.6 | 105.2 | 100 |
| keep-alive off | 254.2 | 254.3 | 259.2 | 208.6 | 0 |
| jitter 20 ms | 174.6 | 174.4 | 183.3 | 129.9 | 100 |
| loss 2% (200 ms timeout per hit) | 177.2 | 151.5 | 349.5 | 132.3 | 100 |

Failover (2 stages x 2 replicas, 50 ms RTT, the in-use replica of hop 0 killed, process and proxy, before decode step 10 of 24):
tokens identical to an undisturbed run, 1 failover, steady step 114 ms, the interrupted step took 1286 ms, so the
pause is about **1.17 s**. That is the client's retry backoff (0.5 s), opening the session on the replica and replaying
the whole history to it, with every message crossing the emulated 50 ms link.

Reading the table:

- Network is the bottleneck as soon as RTT is non-zero: at 50 ms RTT each extra stage costs about 50 ms per token, while
  45 ms of nominal compute per token is small next to it. Tokens per second is roughly 1 / (compute + stages x RTT).
- Keep-alive matters: without it every hop pays an extra RTT for the handshake (+104 ms for 2 hops here). On a real
  tunnel the TLS handshake costs more RTTs than TCP alone, which this emulation does not model.
- The stack's own cost is small: with no delay and no compute a token takes 0.7 ms through 2 hops (about 0.3 ms per hop for HTTP, JSON and
  base64 at hidden 256), and the client's own work stays under 8 ms even at hidden 8192.
- The emulated hop costs RTT plus a few ms (about 2 ms per hop at 1 stage, up to 9 ms per hop at 4 stages): proxy thread
  scheduling and the stack's per-hop work.
- `compute` reads about 8 ms above the 36 ms the knob says: `sleep()` overshoots on macOS, the numpy layers add about 1 ms and the server hashes each step for its retry cache.

## What this does and does not prove

- It shows the real code path adds little and that per-token time is set by hops x RTT plus compute. It validates
  the harness itself: the tests assert that injected delay shows up in `network` and injected compute in `compute`.
- Loopback plus an emulated delay is not the internet. There is no bandwidth limit, no real congestion, no
  TLS or Cloudflare tunnel (its extra hops, handshakes and per-request overhead), no packet reordering, no NAT or path
  changes. Loss here is a fixed timeout, not TCP's real behaviour. The proxy only sleeps, so it adds none of the variance a real path has.
- Synthetic compute is a sleep plus a small numpy MLP. It is not GPU compute: it has no memory bandwidth limit, no batch effects, no
  growing KV cache, no thermal throttling, and it shares CPU with the client and the other nodes.
  Real per-token compute depends on the model, quantisation and device: measure it with a real node and put the figure in `--layer-ms`.
- All processes run on one machine, so they compete for CPU. Hidden sizes up to 8192 and 20-token generations are small, and 2
  measured runs per cell are enough to see the trends above, not to give confidence intervals.
- Failover covers a killed node (connection refused). It does not cover a hung node, a slow node or a tunnel that returns 502.

## Reproduce

```
python3 -m commonnet.bench --sweep --out bench-results/sweep-2026-10-24.json    # about 5 minutes
python3 -m commonnet.bench --stages 2 --layer-ms 40 --rtt-ms 50                  # one scenario
python3 -m commonnet.bench --stages 3 --rtt-ms 50 --jitter-ms 20 --loss 0.02 --hidden 4096
python3 -m commonnet.bench --stages 4 --rtt-ms 100 --no-keepalive
python3 -m commonnet.bench --failover --stages 2 --rtt-ms 50 --layer-ms 3
python3 -m pytest -q commonnet/tests/test_bench_harness.py                       # the harness validation, about 30 s
```

By default `--layers` equals `--stages` (one layer per stage), so `compute` should be stages x `--layer-ms`. The sweep uses 12 layers.

## Benchmark your own network

The synthetic harness above emulates a network. `--live` measures a real one: your peers, your tunnels, the real model,
through the same client path `commonnet.serve` uses (gossip, `plan()`, `NetworkBackend`/`Client4`). It needs the
client checkpoint (embedding and head) like `serve`: downloaded from the manifest's `client_repo`, or `--checkpoint DIR`.

```
python3 -m commonnet.bench --live https://one-of-your-peers.trycloudflare.com     # a peer URL, or a manifest file
python3 -m commonnet.bench --live URL --prompt-lens 16,256 --new-tokens 48 --repeats 3
python3 -m commonnet.bench --live URL --draft --draft-k 8                          # also plain vs Common Flash speculative
python3 -m commonnet.bench --live URL --checkpoint ~/models/qwen-client --out bench-results/mine.json
```

To split the model across several peers on one machine, give each peer a small `--stage-gib` and its own identity
directory, then point the others at the first one with `--seed` (use the URL the first prints):

```
python3 -m commonnet.node --stage-gib 2 --identity-dir ~/.common-network/net/split-a
python3 -m commonnet.node --stage-gib 2 --identity-dir ~/.common-network/net/split-b --seed URL_OF_A
python3 -m commonnet.node --stage-gib 2 --identity-dir ~/.common-network/net/split-c --seed URL_OF_A
python3 -m commonnet.node --stage-gib 2 --identity-dir ~/.common-network/net/split-d --seed URL_OF_A
```

(Add `--port` to each if the defaults clash.) Loopback has no real link, so this shows the stack's own cost per hop;
peers on other machines or behind tunnels show the real network.

What it prints:

- The route `plan()` picked: stages, the peer (short id), backend and host of each hop. Secrets are never printed or saved.
  If one peer holds every layer it says so plainly: there is nothing to split and no hop to measure. If every hop is on
  one host it says the network column is loopback.
- Per token (decode) and per prefill, for fixed prompts of 16, 256 and 1024 tokens (default cases: 16/16, 16/64, 256/32,
  1024/16 prompt/new tokens): ms/token, p50, p95, tok/s, and the split `compute + network + client = ms/token`.
  **compute** is the sum of the blocks' own `server_ms` (Server-Timing header), **network** is the sum of
  `total_ms - server_ms` over the hops (link, TLS/tunnel, serialisation, queueing), **client** is everything else in the
  token (embedding, head, sampling, tensor conversion). Prefill ms and prefill tokens/s are reported the same way.
  A peer that sends no Server-Timing header counts as 0 compute, so its time shows up as network (flagged in the JSON as `missing_server_ms`).
- Per hop: peer, layers, backend, host, calls, `total_ms`, `server_ms`, network ms, connection reuse %, KB per call, and
  **RTT**: the bare `GET /health` round trip on a warm connection (median of 5, first call in the JSON). Network ms well
  above RTT means serialisation, queueing or a slow tunnel, not the link itself.
- `--draft`: the same prompts run plain and speculative (greedy, temperature 0); tok/s of each, acceptance, and whether
  the output is identical. It is skipped, with the reason, if the drafter does not load or a server cannot roll back.
- JSON in `bench-results/live-<timestamp>.json` (hosts without paths or query strings, no tokens).

Read it with the caveats of one run on one connection: a few repeats per case, the first call opens the tunnels (a
warm-up generation runs first, untimed), and the client's MLX work shares the machine with any local peers.
