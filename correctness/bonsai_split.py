"""Greedy tokens through a 2-stage local split vs a single block, on a real Bonsai pack (Apple silicon).

    python correctness/bonsai_split.py CHECKPOINT_DIR [CUT] [NEW_TOKENS]

The checkpoint is a full pack directory (for example the Hugging Face snapshot of
prism-ml/Ternary-Bonsai-8B-mlx-2bit). Both runs use the node's settings: bfloat16 compute, float32 hidden state
between stages, the client's embedding and head from the same pack. Prints one line per prompt and the verdict.
"""
import gc
import json
import os
import sys
import time

import mlx.core as mx
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from commonnet.backends.mlx_block import MlxBlockRange  # noqa: E402
from commonnet.client import BlockEndpoint, LocalTransport  # noqa: E402
from commonnet.client4 import Client4  # noqa: E402

ck = sys.argv[1]
cfg = json.load(open(os.path.join(ck, "config.json")))
n = (cfg.get("text_config") or cfg)["num_hidden_layers"]
cut = int(sys.argv[2]) if len(sys.argv) > 2 else n // 2
new = int(sys.argv[3]) if len(sys.argv) > 3 else 24
tok = AutoTokenizer.from_pretrained(ck)
PROMPTS = ["What is the capital of France? Answer in one sentence.",
           "Write a Python function that returns the n-th Fibonacci number.",
           "Explain in two sentences why the sky is blue."]
ids = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False,
                               **({"enable_thinking": False} if "qwen3" in str(cfg.get("model_type", "")) or True else {}))
       for p in PROMPTS]
ids = [tok(t, add_special_tokens=False)["input_ids"] for t in ids]


def run(bounds):
    blocks = {f"b{i}": MlxBlockRange(ck, a, b, dtype=mx.bfloat16) for i, (a, b) in enumerate(zip(bounds, bounds[1:]))}
    eps = [BlockEndpoint(f"b{i}", a, b) for i, (a, b) in enumerate(zip(bounds, bounds[1:]))]
    client = Client4(ck, LocalTransport(blocks), fmt="mlx4", hidden_dtype=torch.float32)
    out, t0 = [], time.time()
    for p in ids:
        out.append(client.generate(eps, p, new))
    dt = time.time() - t0
    del client, blocks
    gc.collect()
    mx.clear_cache()
    return out, dt


ref, t_ref = run([0, n])
got, t_split = run([0, cut, n])
exact = 0
for p, a, b in zip(PROMPTS, ref, got):
    same = a == b
    exact += same
    print(("MATCH " if same else "DIFF  ") + f"{p[:48]!r} -> {tok.decode(a)[:90]!r}")
print(f"{n} layers, cut at {cut}: split == single-block greedy on {exact}/{len(PROMPTS)} prompts "
      f"({new} tokens each; single {t_ref:.1f}s, split {t_split:.1f}s)")
sys.exit(0 if exact == len(PROMPTS) else 1)
