"""Single-machine reference: mlx_lm greedy decoding of the same token ids. Writes reference.json."""
import json, sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from transformers import AutoTokenizer
sys.path.insert(0, ".")
from prompts import PROMPTS, N_NEW
ck = sys.argv[1]
model, _ = load(ck)
tok = AutoTokenizer.from_pretrained(ck)
out = []
for p in PROMPTS:
    ids = tok(p)["input_ids"]
    gen = []
    for t, _lp in generate_step(mx.array(ids), model, max_tokens=N_NEW):
        gen.append(int(t))
    # teacher-forced top-1 over the whole sequence in one pass
    full = mx.array([ids + gen])
    logits = model(full)
    top1 = mx.argmax(logits[0], axis=-1).tolist()
    out.append({"prompt": p, "ids": ids, "gen": gen, "top1": top1})
    print("ref", len(out), flush=True)
json.dump(out, open("reference.json", "w"))
