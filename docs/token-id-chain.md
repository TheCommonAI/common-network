# Token IDs at the edges, hidden states between peers

A client negotiates the route before generation. When every selected peer supports
`chain_v1`, the first peer supports `ids_in`, and the last supports `head_out`, the
client sends token IDs to the first peer. The machine serving the first decoder
layer (a window starting at layer index 0) holds the target embedding table. It
embeds the IDs on that machine and runs its assigned decoder layers.

Each peer forwards the resulting hidden states straight to the next selected peer.
The machine serving the final decoder layer holds the final norm and output head,
then returns token IDs. Intermediate peers hold neither end table. On Metal, end
tables run on the decoder's worker thread. Torch end tables use the decoder's
configured device and dtype. Stored MLX affine/Hadamard end tables remain packed
on Metal; CPU and NVIDIA peers expand affine end tables in bounded chunks. A tied
embedding/head on one torch device shares its weight storage.

For speculative decoding, the local drafter still proposes token IDs. The target
chain receives the last committed token plus the draft. The last peer applies its
head, compares the draft with the target's greedy choices, and returns the accepted
prefix plus one correction or bonus token. It does not return a vocabulary-sized
logit array or all speculative hidden states. Rollback cuts the rejected suffix
before the next round. Greedy speculation remains the supported verification rule;
nonzero-temperature generation uses ordinary sampling on the final peer. Sampling
is seeded by absolute position so a replay is repeatable; its draws are not
promised to match the legacy client's sequential random-number stream.

## Compatibility and failures

Health capabilities select the path before a session starts. Routes without the
chain capability can use the token-ID edge protocol when their end peers support
it; older routes retain the hidden-state protocol. A supported chain does not
silently change protocols halfway through a session. Recovery replays retained
input token IDs into a new chain session, rather than retaining intermediate
hidden-state histories on the client.

The serving client initially needs model metadata and tokenizer files, not the
target embedding and head weights. It downloads/loads those weights lazily only
if a route requires the legacy path. `COMMONNET_IDS=0` disables end-stage token-ID
capabilities and their additional downloads; normal placement remains conservative
about end-window memory.

Chain forwarding can use existing pinned TLS addresses or tunnel addresses. This
is an application-protocol improvement; it does not make a router accept inbound
traffic. Real internet latency and throughput still depend on reachability,
per-link bandwidth, placement, and the speed of the slowest stage. Peers can read
prompts and answers, including the token IDs and activations they process.

## Between-peer activation compression

Compatible chain peers use per-group 8-bit activation compression by default.
Each sequence position has one signed int8 value per hidden component and one
little-endian float32 scale per group of 64 components. The original shape and
float16/float32 dtype are restored before the next peer computes. Token IDs and
model-weight precision are unchanged. Peers without the compression capability
retain the existing floating-point format. Set `COMMONNET_ACTIVATION_CODEC=float`
on the client to use uncompressed activations.

This quantization is lossy. At the user's request, only basic codec/protocol smoke
checks were run for this change; model quality was not compared. Earlier limited
per-group int8 agreement measurements do not guarantee unchanged output on every
prompt. The saved `token-id-chain-loopback.json` byte report measures the prior
float32 chain, before this compression; it is not an int8 performance result.

## End-window memory budgets

The manifest's optional `io_gib` maps a backend to an additional resident-weight
allowance **per end**. A window starting at 0 pays one allowance, a window ending
at the model's last layer pays one, and a whole-model window pays two. Middle
windows pay none. Unknown backends conservatively use the largest stated allowance.
Missing `io_gib` preserves the accounting of older custom manifests. Values must
be finite, nonnegative numbers.

The shipped allowances are rounded up from the cached checkpoint configurations:

| Network | Vocabulary × hidden width | Metal per end | CPU/NVIDIA per end |
| --- | --- | --- | --- |
| Qwen3-8B | 151,936 × 4,096 | 0.35 GiB | 1.2 GiB |
| Ternary Bonsai 8B | 151,669 × 4,096 | 0.20 GiB | 1.2 GiB |
| Ternary Bonsai 2 9B | 248,320 × 4,096 | 0.30 GiB | 2.0 GiB |

CPU/NVIDIA allowances use two bytes per vocabulary weight. Metal uses four-bit
Qwen weights with group-64 scale/bias overhead, or two-bit Bonsai weights with
group-128 scale/bias overhead. Norm and sign vectors are small relative to the
vocabulary matrices. These allowances describe resident end weights; the existing
`fixed_gib` and per-layer costs still cover the rest of the placement estimate.
They are estimates, not a guarantee against allocator overhead or larger KV caches.

## Validation and limits

Offline tests compare real tiny-checkpoint end embeddings and greedy head IDs with
the existing client, verify tied weights and exact torch dtype/device placement,
exercise Metal packed and mixed-format tables when Metal is available, and reject
out-of-vocabulary IDs before device execution. Metadata-only download tests use a
local HTTP server and assert no weight payload is fetched until a legacy fallback.

Loopback chain tests validate the protocol and byte placement. They do not establish
a real-WAN speed-up or validate the new NVIDIA end-table path on physical CUDA.
Speculative performance must be measured separately from greedy correctness,
including the previously observed `k=8` mismatch on the live heterogeneous network.
