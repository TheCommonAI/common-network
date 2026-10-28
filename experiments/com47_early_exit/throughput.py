"""COM-47 throughput: wall-clock tokens/s of plain greedy decoding vs self-speculative decoding.

The drafter is the same model truncated to its first L layers plus the final norm and lm_head (the logit lens
measured in feasibility.py). Drafts are verified by the full model with KV-cache rollback, the same accept/bonus
rule as commonnet/speculative.py. Outputs are checked against plain greedy decoding.
"""

import argparse
import contextlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

DATA_DIR = Path(__file__).resolve().parents[2] / "bench" / "data"


@contextlib.contextmanager
def truncated(model, full_layers, L):
    N = len(full_layers)
    model.model.layers = nn.ModuleList(full_layers[:L])
    model.config.num_hidden_layers = L
    try:
        yield
    finally:
        model.model.layers = nn.ModuleList(full_layers)
        model.config.num_hidden_layers = N


def set_len(cache, length):
    for layer in cache.layers:
        cur = layer.get_seq_length()
        if cur > length:
            layer.crop(-(cur - length))


@torch.inference_mode()
def plain_decode(model, ids, n_new):
    cache = DynamicCache()
    logits = model(ids, past_key_values=cache, use_cache=True).logits[0, -1]
    tok = logits.argmax().view(1, 1)
    gen = [tok.item()]
    t0 = time.perf_counter()
    for _ in range(n_new - 1):
        logits = model(tok, past_key_values=cache, use_cache=True).logits[0, -1]
        tok = logits.argmax().view(1, 1)
        gen.append(tok.item())
    return gen, time.perf_counter() - t0, {}


@torch.inference_mode()
def spec_decode(model, full_layers, ids, n_new, L, K, tau):
    cache = DynamicCache()
    logits = model(ids, past_key_values=cache, use_cache=True).logits[0, -1]
    last = logits.argmax().view(1, 1)
    gen = [last.item()]
    n = ids.shape[1]
    rounds = drafted = accepted = 0
    t0 = time.perf_counter()
    while len(gen) < n_new:
        drafts = []
        cur = last
        with truncated(model, full_layers, L):
            for _ in range(K):
                lg = model(cur, past_key_values=cache, use_cache=True).logits[0, -1]
                if tau > 0 and torch.softmax(lg.float(), -1).max().item() < tau:
                    break
                d = lg.argmax().view(1, 1)
                drafts.append(d)
                cur = d
        set_len(cache, n)
        m = len(drafts)
        block = torch.cat([last] + drafts, dim=1) if m else last
        preds = model(block, past_key_values=cache, use_cache=True).logits[0].argmax(-1).tolist()
        a = 0
        while a < m and drafts[a].item() == preds[a]:
            a += 1
        new_tokens = [d.item() for d in drafts[:a]] + [preds[a]]
        gen.extend(new_tokens)
        set_len(cache, n + a + 1)
        n = n + a + 1
        last = torch.tensor([[preds[a]]])
        rounds += 1
        drafted += m
        accepted += a
    elapsed = time.perf_counter() - t0
    return gen[:n_new], elapsed, {"rounds": rounds, "drafted": drafted, "accepted": accepted}


@torch.inference_mode()
def time_components(model, full_layers, ids, L, K, reps=20):
    cache = DynamicCache()
    model(ids, past_key_values=cache, use_cache=True)
    n = ids.shape[1]
    tok = torch.tensor([[1000]])
    block = torch.tensor([[1000] * (K + 1)])

    def t(fn):
        vals = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            vals.append(time.perf_counter() - t0)
            set_len(cache, n)
        return statistics.median(vals)

    t_full = t(lambda: model(tok, past_key_values=cache, use_cache=True))
    t_verify = t(lambda: model(block, past_key_values=cache, use_cache=True))

    def draft_step():
        with truncated(model, full_layers, L):
            model(tok, past_key_values=cache, use_cache=True)

    t_draft = t(draft_step)
    return {"full_step_s": t_full, "verify_block_s": t_verify, "draft_step_s": t_draft}


def load_prompts(n):
    with open(DATA_DIR / "math_gsm8k.jsonl") as f:
        rows = [json.loads(line) for line in f][:n]
    return [r["question"] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--out", default="/tmp/com47/throughput.json")
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 4)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    full_layers = list(model.model.layers)
    N = len(full_layers)
    prompts = [tok(p, return_tensors="pt", truncation=True, max_length=256).input_ids for p in load_prompts(args.prompts)]

    configs = [
        ("plain", None, None, None),
        ("spec L=24 K=4 ungated", 24, 4, 0.0),
        ("spec L=24 K=8 ungated", 24, 8, 0.0),
        ("spec L=24 K=4 pmax>=0.5", 24, 4, 0.5),
        ("spec L=20 K=4 ungated", 20, 4, 0.0),
        ("spec L=12 K=4 ungated", 12, 4, 0.0),
    ]

    for ids in prompts:
        plain_decode(model, ids, 8)

    plain_gen = []
    results = []
    for name, L, K, tau in configs:
        total_tok = 0
        total_time = 0.0
        stats = {"rounds": 0, "drafted": 0, "accepted": 0}
        exact = 0
        for p_i, ids in enumerate(prompts):
            if L is None:
                gen, elapsed, st = plain_decode(model, ids, args.new)
                plain_gen.append(gen)
            else:
                gen, elapsed, st = spec_decode(model, full_layers, ids, args.new, L, K, tau)
                exact += int(gen == plain_gen[p_i])
                for k in stats:
                    stats[k] += st[k]
            total_tok += len(gen)
            total_time += elapsed
        row = {
            "config": name,
            "tokens_per_s": round(total_tok / total_time, 3),
            "tokens_per_round": round(total_tok / stats["rounds"], 3) if stats["rounds"] else 1.0,
            "draft_precision": round(stats["accepted"] / stats["drafted"], 3) if stats["drafted"] else None,
            "exact_match_vs_plain": f"{exact}/{len(prompts)}" if L is not None else "reference",
        }
        results.append(row)
        print(json.dumps(row), flush=True)

    comp = {}
    for L in (20, 24):
        comp[L] = time_components(model, full_layers, prompts[0], L, 8)
        print("components", L, json.dumps(comp[L]), flush=True)

    with open(args.out, "w") as f:
        json.dump({"model": args.model, "prompts": len(prompts), "new_tokens": args.new, "N": N,
                    "configs": results, "components": {str(k): v for k, v in comp.items()}}, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
