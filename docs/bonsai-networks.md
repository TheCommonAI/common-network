# Bonsai networks

Two more models run on the layer network next to the default Qwen3-8B. Each is a manifest file in
`commonnet/networks/`; the default network (`main.json`) is untouched.

| Network | Manifest | Model | Layers | Stages | Backends |
| --- | --- | --- | ---: | --- | --- |
| Bonsai 1 8B | `bonsai-8b.json` | Ternary Bonsai 8B (Qwen3-8B architecture and tokenizer) | 36 | 0-9-18-27-36 | Apple silicon (MLX), CPU |
| Bonsai 2 9B | `bonsai-2-9b.json` | Ternary Bonsai 2 9B (Qwen3.5 lineage, hybrid) | 32 | 0-8-16-24-32 | Apple silicon (MLX) only |

```bash
common join --network /path/to/commonnet/networks/bonsai-8b.json     # serve a slice of Bonsai 1 8B
common join --network /path/to/commonnet/networks/bonsai-2-9b.json   # serve a slice of Bonsai 2 9B
python -m commonnet.bench --live /path/to/bonsai-8b.json ...          # measure it (see network-bench.md)
```

The first peer of a network starts it (the manifest lists no seeds); a friend joins with
`common join --network http://<that peer's url>` as for any network. Peers of different networks never mix:
records carry the network id.

## Where the weights come from

Repo ids were read from the Hugging Face API (organisation `prism-ml`), not guessed.

- Bonsai 1 8B, MLX: `prism-ml/Ternary-Bonsai-8B-mlx-2bit`. One 2.30 GB `model.safetensors` plus an index.
  Stock MLX affine 2-bit, group size 128, ternary values {-1, 0, +1}. Standard `model.layers.N.*` names.
- Bonsai 1 8B, CPU: `prism-ml/Ternary-Bonsai-8B-unpacked`. The same weights as float16 in a plain
  `Qwen3ForCausalLM` checkpoint (4 shards, 16.4 GB). Listed so CPU machines can join. **Not run here.**
- Bonsai 2 9B, MLX: `prism-ml/Ternary-Bonsai-2-9B-mlx-2bit`. One 3.46 GB file: language model (2-bit, group 128)
  plus an FP16 vision tower that the network never loads. Names are `language_model.model.layers.N.*`.
- No NVIDIA (NF4) repo exists for either model, and no CPU repo exists for Bonsai 2 (its weights are only
  published folded into the Hadamard basis, see below). A machine whose backend a network does not list is
  refused at start, before anything downloads:

  ```
  error: Ternary Bonsai 2 9B (network common-bonsai-2-9b-v1) has no CPU checkpoint, so this machine cannot serve it.
  Supported here: Apple silicon (MLX). Join the default network instead (`common join --network`), or use a supported machine.
  ```

`gib_per_layer` was measured from the safetensors headers (sum of each layer's tensors): Bonsai 1 8B MLX 0.0506 GiB
per layer (every layer equal), CPU 0.359 GiB (float16); Bonsai 2 9B MLX 0.0551 GiB for a full-attention layer and
0.0584 GiB for a linear-attention one (mean 0.0576). The manifests round up (0.051, 0.36, 0.058) and keep the
default 0.3 GiB fixed overhead. The embedding and head (the client part) are 0.33 GiB for Bonsai 1 and 0.53 GiB for
Bonsai 2 and are not counted in a window. A peer fetches only its own layers: the one-file packs are read with
HTTP range requests (`hub.py` now does that for any repo whose index names a single shard, not only for qwen3_5).

## How a window loads

**Bonsai 1 8B loads as is.** `MlxBlockRange` already uses a checkpoint's own quantization when it stores `.scales`
next to the layer weights; this pack says `bits 2, group_size 128` in `config.json`, and `mlx_lm`'s Qwen3 block
runs it with the stock `QuantizedLinear`. No format change was needed. The client side (`Client4`, `mlx4`) reads
the pack's own 2-bit embedding and head the same way.

**Bonsai 2 9B needs the pack's Hadamard transform.** Its `config.json` has `model_type: prism_hadamard_qwen35`,
which stock `mlx_lm` rejects. The projections are stored in a rotated basis (blockwise Walsh-Hadamard with an explicit
sign vector, block 4096) so that the ternary codes are accurate; the runtime must transform each activation before
the 2-bit matmul, and the embedding's rows after the lookup. The change, all small:

- `commonnet/backends/hadamard.py`: `Packed` (ported from the pack's own `runtime/runtime.py`, MIT) and `install()`.
- `mlx_block.py`: a `prism_hadamard_qwen35` config selects the existing qwen3_5 path (hybrid layers, linear-attention
  state) and swaps each listed projection of the window's layers for `Packed`, reading `weight/scales/biases/signs`.
  Only the window's tensors are read. It does not need `mlx-vlm`; the layers are `mlx_lm`'s `qwen3_5` blocks.
- `weights.py` / `client4.py`: the decoder config comes from the nested `text_config`, and the client's embedding and
  head use `Packed` with the inverse / forward transform.

## Speculative decoding (Common Flash)

- **Bonsai 1 8B**: the tokenizer is identical to Qwen3's. Checked programmatically with the same function the drafter
  uses (`speculative.check_tokenizers_match`): `get_vocab()` has 151,669 entries (151,643 plus 26 added tokens) in both
  Bonsai 1 8B and Qwen3-0.6B, zero ids differ, and the added-token tables are equal, so the check accepts the
  Qwen3-0.6B drafter and was not changed. (`config.json` says `vocab_size 151669`, which is the tokenizer size; Qwen3
  pads the embedding to 151,936.) That the drafter is *compatible* is not that it is *fast* on this model: acceptance
  against the ternary model was not measured here.
- **Bonsai 2 9B**: speculation stays off. Its vocabulary has 248,077 entries and only 281 of them sit at the same id as
  in Qwen3-0.6B, so the check refuses; and its linear-attention layers hold a recurrent state that cannot be rolled
  back, so a block answers "not supported" to the trim that verification needs. The manifest carries a
  `no_speculation` reason and `commonnet.serve --draft` prints it (`speculative decoding is off: ...`).
  A drafter for it needs retraining on the Qwen3.5 tokenizer (see `experiments/common-flash/mlx/README.md`).

## Caveats

- Quality: these are 2-bit ternary models; their quality is Prism ML's, not something the network changes.
- A mixed network (MLX peers and CPU peers on Bonsai 1) works like the default one, but the CPU repo is the unpacked
  float16 file and was not run here.
- Bonsai 2 hidden size is 4096 like Qwen3-8B, so hidden states cross the wire the same size (about 16 KB per token).
- Compute is bfloat16 (the node's default for MLX), the pack's reference runtime uses float16.
- The relative path `commonnet/networks/<name>.json` works from a checkout; otherwise give the full path.

## Results (Apple silicon, 16 GB Mac, 2026-10-24)

The Mac was busy (other peers running, swap nearly full), so treat all timings as rough upper bounds.

- **Correctness**: `correctness/bonsai_split.py`, greedy 24 tokens on 3 chat prompts, bfloat16 compute, float32 hidden
  state, a 2-stage split (8B cut at 18, 9B cut at 16) against a single block holding every layer: identical tokens on
  3/3 prompts for both models, and the text is coherent. For Bonsai 2 the first two answers were also compared with the
  pack's own loader (mlx-vlm, `vision_artifact.load_vl_model`): the same text for the 24 tokens checked.
  The split was in-process (no HTTP) for this check.
- **Window download**: `Hub.fetch(layers=(0,9))` of the 8B pack and `(0,8)` of the 9B pack fetched 0.455 and 0.461 GiB.
- **Speed**: one local peer holding every layer, `python -m commonnet.bench --live` (prompt 16 / 256, 32 new tokens, 1 repeat):
  Bonsai 1 8B about 227-250 ms/token (4.0-4.4 tok/s; compute about 190-212 ms); Bonsai 2 9B about 115-124 ms/token
  (8.0-8.7 tok/s; compute about 98-105 ms). The two were measured minutes apart under different load, so do not read
  the 8B/9B ordering as real; the prism-ml reports roughly 30 ms/token on a quiet machine.
  Memory was not measured.
- **Not verified**: the CPU repo for Bonsai 1 (never run), more than one peer on these networks, the chat API (`--api`)
  with these models, drafter acceptance on Bonsai 1.
