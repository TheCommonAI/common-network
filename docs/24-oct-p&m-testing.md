# 24 October 2026: testing log (Common Network)

Everything measured or found while testing the layer network on 24 Oct 2026, in the order it happened, with the numbers.
Raw files are in `bench-results/` (see the last section). Where a number is an estimate or a guess it is marked **estimate**.
Nothing here was run on a Windows machine by us: Windows results are what the user pasted back.

## Devices

| Device | Hardware | Role in testing |
|---|---|---|
| Mac | MacBook Air M3, 16 GB, Apple silicon (MLX) | Main peer. Ran every benchmark. On AC power, Low Power Mode off. |
| CPU laptop | School Windows laptop, 16 cores, about 31 GB RAM, no NVIDIA GPU (integrated graphics), disk nearly full | CPU peer for layers 27-36 |
| Patrick's PC | Ryzen 5 5500, RTX 4060 8 GB, 32 GB RAM (Windows) | First real NVIDIA test. At the time of writing it had joined with the same layers as the Mac (0-27) instead of 27-36. |

Model: Qwen3-8B, 36 layers, stage boundaries `[0, 9, 18, 27, 36]`, network `common-main-8b-v1`.

## 1. What was merged during testing

| PR | What | Why |
|---|---|---|
| #74 | Unpack `join/` with the runtime; fetch it if missing | Fresh install died at the tunnel step with `No module named 'join'` |
| #75 | Clear message for machines with no GPU | Windows PC fell back to the retired gateway and crashed on HTTP 410 |
| #76 | Synthetic test model + benchmark harness (`python -m commonnet.bench`) | Measure every latency stat without a real LLM |
| #77 | Common Flash drafter (speculative decoding, `--draft`, k=8 default) | Hide per-hop latency by checking 8 drafted tokens in one pass |
| #78 | CPU backend | Let machines with no GPU serve layers |
| #79 | CPU peers pick float32 or bfloat16 | bfloat16 prefill is about 23x slower than float32 on a CPU without fast bf16 |
| #80 | `common join` serves the chat API by default (`--no-api` to skip) | `common ask` failed with "your peer is running without the chat API" |
| #81 | Windows: wait for the child process instead of `os.execv` | The prompt came back immediately and Enter looked like it exited the node |
| #82 | Chat API client part from the float repo on CPU/NVIDIA peers | Size mismatch loading a 4-bit client part as floats |
| #83 | `python -m commonnet.bench --live` | Benchmark a real running network |
| #84 | Never publish `api.trycloudflare.com` as a peer's address | A peer published Cloudflare's own endpoint, so every request to it answered 405 |
| #85 | NVIDIA peers compute in bfloat16 by default | float16 overflows to NaN on some Qwen3 prompts; the NF4 path had never run |

Merged after: #86 (this log) and #87 (`--layers A-B`, pin a peer to a window).

## 2. Synthetic benchmark (fake layers, emulated delay)

Run on the Mac with `python -m commonnet.bench --sweep`. 12 layers, 3 ms of fake compute per layer (36 ms nominal), hidden size 256, loopback with a
delay proxy in front of every peer. Times in ms per decoded token. This is a measurement of the network stack, not of any real model or link.

| Stages | RTT per hop | ms/token | tok/s | Compute | Network | Client |
|---|---|---|---|---|---|---|
| 1 | 0 | 42.2 | 23.7 | 39.7 | 1.6 | 0.9 |
| 1 | 25 | 68.3 | 14.6 | 40.7 | 26.7 | 0.9 |
| 1 | 50 | 92.9 | 10.8 | 39.8 | 52.0 | 1.1 |
| 1 | 100 | 143.7 | 7.0 | 40.6 | 102.1 | 1.1 |
| 2 | 0 | 47.4 | 21.1 | 43.0 | 3.2 | 1.2 |
| 2 | 25 | 98.3 | 10.2 | 44.4 | 52.8 | 1.1 |
| 2 | 50 | 149.2 | 6.7 | 43.8 | 104.0 | 1.4 |
| 2 | 100 | 249.7 | 4.0 | 43.7 | 204.4 | 1.6 |
| 3 | 0 | 50.1 | 19.9 | 44.5 | 4.3 | 1.3 |
| 3 | 25 | 123.8 | 8.1 | 44.9 | 78.1 | 0.8 |
| 3 | 50 | 200.5 | 5.0 | 43.6 | 155.2 | 1.7 |
| 3 | 100 | 352.1 | 2.8 | 43.9 | 306.3 | 1.9 |
| 4 | 0 | 52.2 | 19.2 | 44.5 | 6.4 | 1.4 |
| 4 | 25 | 151.8 | 6.6 | 45.3 | 105.8 | 0.7 |
| 4 | 50 | 257.7 | 3.9 | 45.1 | 210.3 | 2.3 |
| 4 | 100 | 459.1 | 2.2 | 45.5 | 410.9 | 2.7 |

Other cells (2 stages, 50 ms RTT unless stated):

| Variation | ms/token | Note |
|---|---|---|
| hidden 256 | 149.2 | 3.0 KB per token |
| hidden 1024 | 151.7 | 11.0 KB per token |
| hidden 4096 | 154.7 | 43.0 KB per token |
| hidden 8192 | 157.6 | 85.6 KB per token |
| keep-alive off | 252.6 | 0% connection reuse; one extra round trip per hop |
| jitter 20 ms | 175.4 (p95 182) | |
| loss 2% | p50 151, p95 352 | modelled as one 200 ms timeout per hit chunk |
| Bare stack, 0 ms delay, 0 compute | 0.7 | 2 hops |

Findings:
- Network time is about stages x RTT plus 2-9 ms per hop of emulation overhead.
- Client overhead is 1-2 ms per token (up to 6 ms at hidden 8192). The earlier "340 ms client overhead" in the latency PDF was a residual from subtraction and was wrong.
- Payload size barely matters: hidden 256 to 8192 moved ms/token from 149 to 158.
- Keep-alive is worth about 100 ms per token at 50 ms RTT with two hops.
- Failover (2 stages x 2 replicas, kill the in-use replica before decode step 10 of 24): output identical to an undisturbed run, 1 failover, steady step 114 ms, interrupted step 1286 ms, so a pause of about 1.17 s (0.5 s retry backoff plus replaying the history over the emulated link).

Caveats: one machine, no Cloudflare or TLS, no bandwidth cap, fake compute reads about 8 ms above its setting, two measured runs per cell.

Raw: `synthetic-sweep.json`, `synthetic-sweep.log`, `synthetic-one-run.json` (and `sweep-2026-10-24.json` from the harness author).

## 3. Common Flash speculative decoding (greedy)

A local Qwen3-0.6B drafter (`mlx-community/Qwen3-0.6B-4bit`) proposes k tokens; the network checks all of them in one pass and every stage rolls back the rejected part.
Tokenizer match with Qwen3-8B was checked in code (identical ids). Output must equal plain greedy decoding.

Synthetic cluster (3 stages, 36 layers at 1 ms/layer, hidden 256, 48 tokens, oracle drafter with a set per-token accuracy). Speedup over plain decoding:

| RTT per hop | Plain tok/s | k=4, 50% | k=4, 70% | k=4, 90% | k=8, 50% | k=8, 70% | k=8, 90% |
|---|---|---|---|---|---|---|---|
| 0 | 20.1 | 2.05x | 2.54x | 3.46x | 2.15x | 2.96x | 4.93x |
| 25 | 7.7 | 2.01x | 2.42x | 3.18x | 2.10x | 2.81x | 4.31x |
| 50 | 4.7 | 1.99x | 2.38x | 3.12x | 2.07x | 2.77x | 4.14x |
| 100 | 2.6 | 1.79x | 2.07x | 2.60x | 1.85x | 2.34x | 3.25x |

These flatter the real speedup because synthetic compute does not grow with tokens per pass.

Real Qwen3-8B (MLX 4-bit, bfloat16 compute, float32 on the wire, 3 stages of 12 layers behind 50 ms delay proxies, 64 new tokens, one run per cell, so noisy; all three stages in one process on one machine):

| Prompt | Plain tok/s | k=4 | k=8 |
|---|---|---|---|
| chat (explanation) | 4.01 | 7.14 (1.78x, accept 0.46, 2.2 tok/round) | 6.74 (1.68x, accept 0.37, 2.1) |
| chat (tips) | 4.09 | 7.16 (1.75x, accept 0.52, 2.2) | 6.70 (1.64x, accept 0.40, 2.2) |
| code | 4.02 | 9.12 (2.26x, accept 0.67, 3.0) | 9.58 (2.38x, accept 0.61, 3.6) |
| copy/quote | 3.83 | 10.89 (2.84x, accept 1.00, 4.9) | 12.04 (3.14x, accept 0.88, 7.8) |

- The project's bar of at least 2x at 50 ms hops was met for code and copy/quote and not met for free-form chat (1.6-1.8x), where the 0.6B drafter agrees with the 8B on under half of its guesses.
- Output matched plain decoding in 7 of 8 real runs. One (chat tips, k=8) diverged at token 51: the bfloat16 verify pass over 9 positions is a different computation from one-token decoding, so a near tie can flip.
- Not tested: sampling (temperature above 0 falls back to plain), real peers over a real link, `--draft` on live peers. Qwen3.5/3.6 linear-attention layers cannot roll back.

Raw: `spec-synth.json`, `spec-real.json`.

## 4. CPU backend measurements (the Mac's CPU, not the laptop)

9 real layers of Qwen3-8B on torch CPU (4 threads), one hop, on the Mac:

| dtype | Decode (1 token, KV cached) | Prefill (64 tokens) |
|---|---|---|
| bfloat16 | 58 ms (median; range 54-158) | 9.3 s (8.7-9.8 over 5 runs) |
| float32 | 120 ms | 0.40 s |

The Mac CPU has no fast bf16 matmul: bf16 prefill was about 23x slower than float32, while bf16 decode was 2x faster because it is memory-bound.
float32 needs twice the memory. This is why #79 chooses the dtype automatically (bfloat16 when the CPU has AVX512-BF16 or AMX; float32 when the smallest stage fits at twice the memory; else bfloat16).

## 5. Live network runs (`python -m commonnet.bench --live`)

### 5a. Wrong peer (a mistake, kept as data)

The benchmark was seeded from `status.json`. Two peers on the Mac share that file, so it pointed at an old single-peer node that holds all 36 layers and knows no one else.
It measured one peer through its own Cloudflare tunnel (prompt 16 tokens, 16 new, one run):

| Part | ms/token |
|---|---|
| Compute (36 layers, MLX) | 60.7 |
| Network (1 hop; bare RTT 100.4 ms, first call 196.6 ms) | 72.3 |
| Client | 10.6 |
| Total | 143.6 (7.0 tok/s) |

Prefill: 837.8 ms for 16 tokens. Reuse 100%. 42.8 KB per call.
Lesson: a hop to a peer on the same machine still goes out through the Cloudflare edge and back: about 100 ms round trip.

### 5b. Mac (0-27) + CPU laptop (27-36), the real two-machine split

Route: hop 0 layers 0-27 on the Mac (MLX), hop 1 layers 27-36 on the CPU laptop (`ba7d4aea`, CPU). Prompt 16 tokens, 16 new, one run.

| Part | ms/token | Share |
|---|---|---|
| CPU laptop compute (9 layers) | 377.8 | 65% |
| Network (both hops beyond the server time) | 138.6 | 24% |
| Mac compute (27 layers) | 49.9 | 9% |
| Client | 14.1 | 2% |
| Total | 580.4 (p50 568.0, p95 649.5), 1.7 tok/s | |

Per hop (decode):

| Hop | Layers | Peer | Backend | Calls | Total ms | Server ms | Network ms | Bare RTT ms | Reuse |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0-27 | Mac | mlx | 15 | 119.4 | 49.9 | 69.6 | 48.3 (first call 382.1) | 100% |
| 1 | 27-36 | CPU laptop | cpu | 15 | 446.9 | 377.8 | 69.0 | 72.2 (first call 156.7) | 100% |

Prefill: 3311.6 ms for 16 tokens (4.8 tok/s). Per hop: Mac 500.7 ms total (169.5 server), CPU laptop 2521.8 ms total (2287.1 server).
42.8 KB per call decode, 682.8 KB prefill.

Reading:
- The user's own hand measurement with two laptops over the tunnel was about 2 tokens/s; this agrees (1.7).
- The CPU laptop's 9 layers take 378 ms (about 42 ms per layer); the Mac does 27 layers in 50 ms (about 2 ms per layer). The laptop has no fast bfloat16 and chose bfloat16 because float32 did not fit its memory offer.
- The tunnel adds about 100-120 ms of bare round trips per token for two hops, plus about 70 ms per hop of serialisation and queueing.
- **Estimate (not measured):** if layers 27-36 ran on a GPU peer at tens of ms, a token would cost roughly 250-300 ms (3-4 tok/s), with the network the largest part.

Limits: one run of 16 tokens, so noisy; the two peers are on different networks and very different hardware; Server-Timing came from the peers.

Raw: `live-wrong-peer-single-mac-36-layers.json`, `live-mac-0-27-plus-cpu-laptop-27-36.json`.

### 5c. Mac (0-27) + Patrick's RTX 4060 (27-36, NF4): the first real NVIDIA run

Route: hop 0 layers 0-27 on the Mac (MLX), hop 1 layers 27-36 on Patrick's PC (`6f7d25cb`, backend nf4; the CPU laptop is its replica). Pinned/placed so the GPU holds the slow stage.
Prompts of 16 and 256 tokens, 24 new tokens each, 2 runs per case, over the real internet through Cloudflare tunnels. This is the first time the NVIDIA path has run on real hardware.

| Part | ms/token (mean of both cases) | Note |
|---|---|---|
| Compute: Mac, 27 layers | 46.6-48.2 | MLX |
| Compute: RTX 4060, 9 layers | 31.3-37.8 | NF4, bfloat16 |
| Network, both hops beyond server time | 92.5-94.1 | bare RTT 52.2 ms (Mac), 29.6 ms (Patrick) |
| Client | 11.7-11.8 | |
| Total | **186.9 ms (5.3 tok/s)**; prompt 16: 188.5 (p50 175.1, p95 270.7), prompt 256: 185.4 (p50 175.3, p95 233.3) | |

Per hop, decode (prompt 16): Mac 105.1 ms total (46.6 server, 58.6 network); Patrick 71.7 ms total (37.8 server, 33.9 network). Connection reuse 100%, 42.8 KB per call.

Prefill: 16 tokens 852.5 ms (18.8 tok/s); 256 tokens 7186 ms (35.6 tok/s). In the 256-token case each hop carries 10.9 MB per call: Mac hop 4540 ms total (1180 server, 3360 network), Patrick hop 2094 ms total (308 server, 1786 network). Prefill is dominated by moving the activations, not by compute.

Same route with the CPU laptop holding 27-36 (section 5b): 580.4 ms/token (1.7 tok/s). Moving that stage to the RTX 4060 made decode about 3.1x faster; that stage's compute went from 378 ms to about 31-38 ms.

Limits: this checks speed only. The benchmark does not print or compare the generated text, so correctness of the NF4 output (and absence of NaN) is **not verified** by this run. Two runs per case; the machines are on different home/school networks; one 256-token prompt length.

Raw: `live-mac-0-27-plus-patrick-rtx4060-27-36.json`, `.log`.

## 6. Why one Mac peer holds all 36 layers

A peer takes the longest window that fits. The Mac offers about 9.9 GiB and the whole 4-bit model costs about 4.8 GiB (`0.3 + 36 x 0.125`), so it serves 0-36 alone;
clients choose the route with the fewest hops, so nobody else is used. The network splits only when no single peer can hold the model, or when memory is capped with `--stage-gib`.
`--stage-gib 2` gives 9-layer windows (a quarter), `--stage-gib 3` gives 18-layer windows (a half), `--stage-gib 4` gave this Mac 0-27 (27 layers, about 3.7 GiB).

Placement score: uncovered stages dominate, then a longer window, then a window that already has the fewest replicas. It does not look at speed.
Observed: Patrick's PC (offering about 5.5-6 GiB on NF4, so up to 27 layers) was assigned 0-27, the same as the Mac, because covering three stages once more outscores replacing the single CPU holder of 27-36.
Fix in progress: `--layers A-B` to pin a window.

## 7. Problems found while testing (and what fixed them)

| Symptom | Cause | Fix |
|---|---|---|
| `No module named 'join'` at the tunnel step on a fresh Mac install | Runtime tarball unpacked only `commonnet/`; dev runs with `COMMONNET_SRC` hid it | #74 |
| Windows `common join` returned to the prompt immediately, output arrived later, Enter looked like exit, Ctrl+C did nothing | `os.execv` on Windows starts a new process and exits | #81 |
| Windows PC without GPU: HTTP 410 traceback | Fell back to the retired gateway | #75, then #78 (CPU backend) |
| `common ask`: "your peer is running without the chat API" | API was opt-in; also hand-started `commonnet.node` peers have no API unless `--api` | #80 (for `common join`); use `--api` on hand-started nodes |
| Chat API on CPU peer: `size mismatch for weight ... [151936, 512] vs [151936, 4096]` | The manifest's client part is the 4-bit MLX checkpoint (packed embedding) | #82 |
| Bench route failed with HTTP 405 | A peer published `https://api.trycloudflare.com` (cloudflared names it in error lines; first-match search took it) | #84 |
| Windows laptop: `OSError: [Errno 28] No space left on device` while writing `join.py` | Disk full. Model cache held only about 7.4 GB (shards 3 and 4 done, shard 1 `.part` at 0.7 GB of 3.7 GB for the chat client part) | Free space; `--no-api` skips shards 1 and 5; `COMMONNET_CACHE` can move the cache |
| NVIDIA block computed in float16 | Default dtype; overflows to NaN on some Qwen3 prompts | #85 |
| Two peers on the Mac write one `status.json` | Shared default path | Use `--status-file` per node and benchmark with an explicit URL |

Disk facts: on the Mac `~/.common-network` was 39 GB (`commonnet/models/Qwen/Qwen3-8B` 12 GB, `Qwen3.6-27B-4bit` 9.2 GB, `Qwen3-8B-4bit` 4.3 GB, old gateway `models/` 12 GB).
Sharded repos (Qwen3-8B) download whole shards that contain needed tensors; single-file repos download a compact copy per window. Nothing stacks per retry; changing the window can add shards.

## 8. Not verified

- Correctness of the NF4 output on Patrick's RTX 4060 (it ran and was fast, but generated text was not compared with a reference). Wheels for `torch 2.11.0+cu128` and `bitsandbytes` exist for Windows (checked with pip dry runs, not installed).
- Any Windows result produced by us; the Windows install steps, `COMMONNET_CACHE` on Windows, the CPU wheel index.
- Speculative decoding on live peers over a real link, sampling (temperature above 0), and the chat client part on NF4.
- Multi-run statistics: every live number is a single run.
- Replica failover mid-benchmark on the live network.

## 9. Raw data

`bench-results/2026-10-24/`
- `synthetic-sweep.json`, `synthetic-sweep.log`, `synthetic-one-run.json`
- `live-wrong-peer-single-mac-36-layers.json`
- `live-mac-0-27-plus-cpu-laptop-27-36.json`
- `live-mac-0-27-plus-patrick-rtx4060-27-36.json` and `.log`

`bench-results/` (from the harness and drafter authors)
- `sweep-2026-10-24.json`, `spec-synth.json`, `spec-real.json`

Reproduce: see `docs/network-bench.md` and `docs/flash-drafter.md`.
