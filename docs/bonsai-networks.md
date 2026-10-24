# Bonsai networks

Three more models run on the layer network next to the default Qwen3-8B. Each is a manifest file in
`commonnet/networks/`; the default network (`main.json`) is untouched.

| Network | Manifest | Model | Layers | Stages | Backends |
| --- | --- | --- | ---: | --- | --- |
| Bonsai 1 8B | `bonsai-8b.json` | Ternary Bonsai 8B (Qwen3-8B architecture and tokenizer) | 36 | 0-9-18-27-36 | Apple silicon (MLX), NVIDIA (NF4, quantised on load), CPU |
| Bonsai 2 9B | `bonsai-2-9b.json` | Ternary Bonsai 2 9B (Qwen3.5 lineage, hybrid) | 32 | 0-8-16-24-32 | Apple silicon (MLX), NVIDIA (NF4 of the dequantised pack), CPU (bf16) |
| Bonsai 2 27B | `bonsai-2-27b.json` | Ternary Bonsai 2 27B (Qwen3.8 lineage, hybrid) | 64 | every 4 layers | Apple silicon (MLX), NVIDIA (NF4 of the dequantised pack), CPU (bf16) |

```bash
common join --network /path/to/commonnet/networks/bonsai-8b.json     # serve a slice of Bonsai 1 8B
common join --network /path/to/commonnet/networks/bonsai-2-9b.json   # serve a slice of Bonsai 2 9B
common join --network bonsai-2-27b                                # serve a slice of Bonsai 2 27B
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
- Bonsai 2 27B, all backends: `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`, one 8.60 GB file. The manifest
  uses the same pack for MLX and the torch paths (which dequantise its 2-bit matrices on load).
- NVIDIA (NF4): Bonsai 1 uses the same `Ternary-Bonsai-8B-unpacked` float16 repo as the CPU (plain `Qwen3ForCausalLM`,
  4 shards, 399 tensors, `model.layers.N.*`, float16 throughout, 8.19 B parameters). Bonsai 2 uses its one MLX pack
  (see "NVIDIA and CPU" below). A network that lists no repo for a machine's backend is still refused at start, before
  anything downloads (`has no NVIDIA GPU (4-bit) checkpoint, ... Supported here: ...`).

`gib_per_layer` was measured from the safetensors headers (sum of each layer's tensors): Bonsai 1 8B MLX 0.0506 GiB
per layer (every layer equal), CPU 0.359 GiB (float16), NVIDIA 0.101 GiB computed (192.9 M weights x (0.5 B NF4 + 4 B/64 absmax)); Bonsai 2 9B MLX 0.0551 GiB for a full-attention layer and
0.0584 GiB for a linear-attention one (mean 0.0576), and for the torch paths 0.110-0.115 GiB NF4 / 0.39-0.41 GiB bf16 (218 M / 210 M packed weights per layer). Bonsai 2 27B MLX is 0.0977 GiB per full-attention layer and 0.1025 GiB per linear-attention layer (from the single-file safetensors header); its manifest rounds the layer cost up to 0.103 GiB and budgets 0.204 GiB NF4 / 0.81 GiB bf16 for torch. The manifests round up (0.051, 0.36, 0.058, nf4 0.105 and 0.115, cpu 0.41) and keep the
default 0.3 GiB fixed overhead. The embedding and head (the client part) are 0.33 GiB for Bonsai 1 and 0.53 GiB for
Bonsai 2 9B and about 0.67 GiB for Bonsai 2 27B; these are not counted in a window. A peer fetches only its own layers: the one-file packs are read with
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

## NVIDIA and CPU (torch paths)

```bash
common join --network commonnet/networks/bonsai-8b.json --layers 0-9 --no-api     # a 4060: 9 layers, about 1.3 GiB VRAM
common join --network commonnet/networks/bonsai-2-9b.json --layers 0-8 --no-api   # Bonsai 2, 8 layers, about 1.3 GiB VRAM
```

Use a manifest path or URL, and `--layers A-B` on a stage boundary (8B: 0-9-18-27-36, 9B: 0-8-16-24-32). With about 4 GiB
offered a peer can take up to 27 layers of the 8B (3.1 GiB) or 24 of the 9B (3.1 GiB); all 32 of the 9B would need 3.98 GiB.

- **Quantise on load** (`nf4_block.py`): per Linear, a tensor with `.absmax` beside it is a pre-quantised bitsandbytes
  tensor and loads as before; a float tensor is read to the CPU, moved to the GPU as a bitsandbytes `Params4bit`
  (`nf4`, bf16 compute, no double quantisation) which quantises it there, one matrix at a time. So any plain HF repo runs.
  Statistics stay fp32 on purpose: the Bonsai weights are ternary (-s, 0, +s per group of 128) and NF4 has the levels
  -1, 0, +1, so they are reproduced exactly.
- **Download**: the 8B repo is float16, so a window downloads at 0.36 GiB per layer (9 layers about 3.2 GiB, only the
  window's tensors by range requests, not whole 5 GB shards). Bonsai 2 downloads its 2-bit pack: 0.46 GiB for 8 layers.
- **Bonsai 2** (`hadamard_torch.py`): each packed projection is dequantised in torch (affine 2-bit, group 128) to
  W' and wrapped in `HadamardLinear`, which applies sign flip plus normalised Walsh-Hadamard (block 4096, float32)
  to the activation before W', exactly as `hadamard.Packed` does. W' keeps its three levels per group, so the NF4
  step is exact on it. The pack's norms (+1) and conv1d layout are converted to transformers' layout.
  CPU peers use the same layers with W' in bf16 (0.41 GiB per layer).
- **Client part / `--api`**: Bonsai 1 on NVIDIA/CPU reads the fp16 embedding and head from the unpacked repo (about
  2.5 GB). Bonsai 2's embedding and head are Hadamard-packed and only the MLX client reads them, so `--api` on a
  non-Mac peer reports that and the peer still serves layers: use `--no-api`.

### What is verified

On CPU here (Mac, tests in `test_nf4_quantise_on_load.py`, `test_hadamard_torch.py`):
- NF4 (blocksize 64, fp32 absmax, 16 levels, simulated in plain torch, not bitsandbytes) reproduces ternary weights
  exactly: synthetic ternary, the real `layers.0.self_attn.k_proj` of the unpacked 8B (32,768 groups, all of the
  form {-s, 0, +s}; max error 0.0, also after bf16), and the dequantised Bonsai 2 W' of 6 real matrices (max 3 levels per
  group, error 0.0).
- Selection logic (pre-quantised vs float) with a fake bitsandbytes; manifests; refusal messages.
- Bonsai 2 torch block vs the MLX block on the real pack, layers 0-8 (linear and full attention), random hidden input,
  prefill 7 tokens then a decode step, float32: relative L2 error 2.0e-6 (prefill) and 1.3e-6 (decode), max abs 1e-4 on
  values up to 33. In bfloat16 both sides round differently: relative L2 1.0%, cosine 0.99995 (layers 0-8), 1.9% and
  0.9998 for layers 28-32. A tiny random pack against its float reference runs in the test suite (needs Metal to build).

Only on real CUDA (unverified; the friend's first run is the test): `bitsandbytes` quantising from `Params4bit` built on a
meta-device `Linear4bit` (written against the bnb 0.4x API), the peak VRAM (expected: NF4 size plus one fp16 matrix,
at most about 0.1 GiB extra), bf16 on an RTX 4060, the speed of the float32 Hadamard matmul on the GPU, the
`ranged` window download of the float16 repo on Windows, the 8B and 9B end to end on a real GPU, and a mixed Mac plus
NVIDIA pipeline. NF4 on Bonsai 2 is exact on W' only if bitsandbytes matches the simulated quantiser (it should: the same 16 levels).

## Speculative decoding (Common Flash)

- **Bonsai 1 8B**: the tokenizer is identical to Qwen3's. Checked programmatically with the same function the drafter
  uses (`speculative.check_tokenizers_match`): `get_vocab()` has 151,669 entries (151,643 plus 26 added tokens) in both
  Bonsai 1 8B and Qwen3-0.6B, zero ids differ, and the added-token tables are equal, so the check accepts the
  Qwen3-0.6B drafter and was not changed. (`config.json` says `vocab_size 151669`, which is the tokenizer size; Qwen3
  pads the embedding to 151,936.) That the drafter is *compatible* is not that it is *fast* on this model: acceptance
  against the ternary model was not measured here.
- **Bonsai 2 9B and 27B**: speculation stays off. Its vocabulary has 248,077 entries and only 281 of them sit at the same id as
  in Qwen3-0.6B, so the check refuses; and its linear-attention layers hold a recurrent state that cannot be rolled
  back, so a block answers "not supported" to the trim that verification needs. The manifest carries a
  `no_speculation` reason and `commonnet.serve --draft` prints it (`speculative decoding is off: ...`).
  A drafter for it needs retraining on the Qwen3.5 tokenizer (see `experiments/common-flash/mlx/README.md`).

## Caveats

- Quality: these are 2-bit ternary models; their quality is Prism ML's, not something the network changes.
- A mixed network (MLX peers and CPU/NVIDIA peers) works like the default one; the CPU repo is the unpacked
  float16 file and the full CPU path on a real 8B window was not run here.
- Bonsai 2 9B hidden size is 4096; the 27B hidden size is 5120, so its hidden states are larger (about 20 KB per token). The 27B chat client reads the Hadamard block size from checkpoint metadata.
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
- **Not verified**: the CPU repo for Bonsai 1 (never run), Bonsai 2 27B on any Common backend, more than one peer on these networks, the chat API (`--api`) with these models, drafter acceptance on Bonsai 1.
