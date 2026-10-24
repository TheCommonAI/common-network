# Flash drafter: speculative decoding for the layer network

Every token normally needs one trip through every stage, so at 50 ms per hop a 3-stage route tops out near 6 tokens a
second whatever the GPUs do. Speculative decoding hides that latency: a small drafter on the client guesses the next
`k` tokens (default **8**), the network checks all of them (plus one) in a **single** pass through the stages, the client
keeps the longest prefix the network agrees with, and the stages roll their KV caches back past the rest.

The drafter is Common Flash's validated small autoregressive drafter, Qwen3-0.6B (default
`mlx-community/Qwen3-0.6B-4bit`, about 0.4 GB RAM on the client), run with `mlx_lm`. It shares Qwen3-8B's tokenizer.
The client compares the two tokenizers' vocabularies id for id and refuses to speculate if they differ.

## Run it

```bash
python -m commonnet.serve --network URL --draft                 # default drafter, k=8
python -m commonnet.serve --network URL --draft MODEL --draft-k 4
```

Off by default. `--draft` also works with the node's `--api` path through the same `NetworkBackend` (`draft=`, `draft_k=`).
Plain decoding is always the fallback, chosen automatically when: the drafter fails to load or its tokenizer differs,
any server on the route is an older version without rollback, or the request samples (`temperature > 0`). The API
default temperature is 0.7, so send `"temperature": 0` to get speculation through the API.

## How it works

- **Rollback.** New block-server op `POST /truncate {"id", "length"}` (idempotent: a session already at or under `length`
  is untouched) and an optional `truncate_to` field on `/forward`, so the client's rollback rides on its next forward and
  costs no round trip. Qwen3 is pure attention, so cutting each layer's KV cache is exact. `HttpTransport` cuts the history
  it keeps for failover to the same length, so a replacement server replays exactly the rolled-back session. Old servers
  answer 404 "no endpoint /truncate"; `HttpTransport.supports_truncate` detects that and speculation turns itself off.
  A block that cannot truncate (the Qwen3.6 linear-attention layers) answers 501.
- **Round.** Feed `[last, d1..dk]` at the current position; the head gives the network's greedy token after each. Accept
  `d1..dm` while they equal it; emit them plus the network's own token after the last accepted one (1 to k+1 tokens per
  round). Output is exactly plain greedy decoding. Adaptive mode halves `k` after a round that accepted under 30 percent of its
  drafts and doubles it back after a fully accepted round.
- Code: `commonnet/speculative.py` (generator, `MlxDrafter`, tokenizer check), rollback in `net/{protocol,server,client,transport,factory}.py`,
  `backends/mlx_block.py`, `block.py`.

## Results

Synthetic cluster (`python -m commonnet.bench.spec_bench`): 3 stages, 36 layers at 1 ms each, hidden 256, 48 new tokens,
delay proxy per hop, oracle drafter whose per-token accuracy is the first column (acceptance counts the accepted prefix only;
tokens per round is what matters). Speedup = tokens/s over plain at the same RTT; every speculative output was asserted equal to
plain. The synthetic compute does not grow with tokens per pass, so it flatters the speedup somewhat (a real 9-token verify
pass costs more than a 1-token pass).

| RTT per hop | plain tok/s | k=4 acc 0.5 | k=4 acc 0.7 | k=4 acc 0.9 | k=8 acc 0.5 | k=8 acc 0.7 | k=8 acc 0.9 |
|---|---|---|---|---|---|---|---|
| 0 ms | 20.1 | 2.05x | 2.54x | 3.46x | 2.15x | 2.96x | 4.93x |
| 25 ms | 7.7 | 2.01x | 2.42x | 3.18x | 2.10x | 2.81x | 4.31x |
| 50 ms | 4.7 | 1.99x | 2.38x | 3.12x | 2.07x | 2.77x | 4.14x |
| 100 ms | 2.6 | 1.79x | 2.07x | 2.60x | 1.85x | 2.34x | 3.25x |

Real Qwen3-8B (MLX 4-bit, bf16 compute, float32 on the wire) in 3 stages of 12 layers behind 50 ms delay proxies, real
Qwen3-0.6B-4bit drafter, 64 new tokens (one run each, so noisy), Apple silicon:

| prompt | plain tok/s | k=4 tok/s (speedup, draft acceptance, tokens/round) | k=8 tok/s (speedup, acceptance, tokens/round) |
|---|---|---|---|
| chat (explanation) | 4.01 | 7.14 (1.78x, 0.46, 2.2) | 6.74 (1.68x, 0.37, 2.1) |
| chat (tips) | 4.09 | 7.16 (1.75x, 0.52, 2.2) | 6.70 (1.64x, 0.40, 2.2), differs from plain at token 51 |
| code | 4.02 | 9.12 (2.26x, 0.67, 3.0) | 9.58 (2.38x, 0.61, 3.6) |
| copy / quote | 3.83 | 10.89 (2.84x, 1.00, 4.9) | 12.04 (3.14x, 0.88, 7.8) |

The 2x bar at 50 ms hops is **met for code and copy/quote, not for free-form chat** (1.6 to 1.8x), where a 0.6B drafter
agrees with the 8B model on under half of its guesses. Acceptance depends on the workload.

## Limits

- Greedy only: `temperature > 0` falls back to plain decoding (rejection sampling is not implemented).
- Exactness: the network's math is the same, but verifying 9 positions in one pass is a different bf16 computation from 1
  position at a time, so on the real model a near-tie argmax can flip. 7 of 8 real runs matched plain exactly; one diverged at
  token 51. On the synthetic model outputs are always identical.
- The drafter costs the client RAM (about 0.4 to 0.6 GB) and a little compute per round; it runs serially with the network
  (no overlap).
- Prefill is not accelerated; only decoding is.
- Rollback needs pure-attention blocks (Qwen3). Qwen3.6 blocks refuse it and the client then decodes plainly.
- All the real-model numbers come from one machine running every stage (GPU shared), with emulated delay, not a real
  internet path. The 3 stages were in one process.
