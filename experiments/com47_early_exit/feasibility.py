"""COM-47 feasibility: can a gate predict whether a draft from the first L layers matches the full model?

Protocol (greedy decoding, measured on the full model's own trajectory):
- Generate GEN_LEN greedy tokens per prompt with the full model.
- One forward pass over prompt+continuation gives the full model's argmax y[q] at each decision position q.
- For each layer L, the early-exit draft is the logit lens d_L[q] = argmax lm_head(norm(h_L[q])).
  c_L[q] = (d_L[q] == y[q]). Because the context is the full model's own trajectory, c_L[q] is exactly
  "this draft is accepted given that earlier drafts in the round were accepted".
- A gate (pmax threshold, logistic regression, or small MLP on the layer-L hidden state) is trained on
  the train split and evaluated on held-out sequences.
- Speculative rounds are simulated on held-out sequences with the same accept/bonus rule as
  commonnet/speculative.py: draft up to K tokens while the gate passes, accept the longest correct prefix,
  emit one verifier token after it. Cost per round is one verify pass plus the drafted layers' share of
  the model, using parameter counts (memory-bandwidth-bound decode assumption).
"""

import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = Path(__file__).resolve().parents[2] / "bench" / "data"


def load_prompts(n_per_set):
    prompts = []
    for name in ("math_gsm8k.jsonl", "general_mmlu.jsonl"):
        with open(DATA_DIR / name) as f:
            rows = [json.loads(line) for line in f][:n_per_set]
        prompts += [r["question"] for r in rows]
    return prompts


@torch.no_grad()
def collect_sequence(model, tok, prompt, gen_len, layers, final_normed):
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=256)
    P = enc.input_ids.shape[1]
    ids = model.generate(
        **enc,
        max_new_tokens=gen_len,
        min_new_tokens=gen_len,
        do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    T = ids.shape[1]
    out = model(ids, output_hidden_states=True)
    N = model.config.num_hidden_layers
    sl = slice(P - 1, T - 1)
    y = out.logits[0, sl].argmax(-1).numpy()

    rec = {"P": P, "T": T, "y": y, "layers": {}}
    for L in layers:
        h = out.hidden_states[L][0, sl]
        if L == N and final_normed:
            logits = model.lm_head(h)
        else:
            logits = model.lm_head(model.model.norm(h))
        probs = torch.softmax(logits.float(), -1)
        top2 = probs.topk(2, -1).values
        rec["layers"][L] = {
            "H": h.float().numpy().astype(np.float16),
            "pmax": top2[:, 0].numpy(),
            "margin": (top2[:, 0] - top2[:, 1]).numpy(),
            "ent": (-(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)).numpy(),
            "d": logits.argmax(-1).numpy(),
        }
    return rec


def detect_final_normed(model, tok, prompt):
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(enc.input_ids, output_hidden_states=True)
    N = model.config.num_hidden_layers
    y = out.logits[0].argmax(-1)
    raw = model.lm_head(out.hidden_states[N][0]).argmax(-1)
    return bool((raw == y).float().mean() > 0.99)


def build_records(args, model, tok, layers):
    prompts = load_prompts(args.n_per_set)
    final_normed = detect_final_normed(model, tok, prompts[0])
    print(f"final hidden state already normalized: {final_normed}", flush=True)
    recs = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        recs.append(collect_sequence(model, tok, p, args.gen_len, layers, final_normed))
        if (i + 1) % 10 == 0:
            print(f"  collected {i + 1}/{len(prompts)} sequences ({time.time() - t0:.0f}s)", flush=True)
    return recs


def split(recs, seed, test_frac=0.25):
    idx = list(range(len(recs)))
    random.Random(seed).shuffle(idx)
    n_test = int(len(idx) * test_frac)
    return [recs[i] for i in idx[n_test:]], [recs[i] for i in idx[:n_test]]


def features(rec, L):
    d = rec["layers"][L]
    X = np.hstack([d["H"].astype(np.float32), np.stack([d["pmax"], d["ent"], d["margin"]], 1)])
    c = (d["d"] == rec["y"]).astype(np.int64)
    return X, c


def stack(recs, L):
    parts = [features(r, L) for r in recs]
    X = np.concatenate([p[0] for p in parts])
    c = np.concatenate([p[1] for p in parts])
    return X, c


def simulate(probs_per_seq, c_per_seq, K, tau, r):
    tokens = cost = drafted = accepted = rounds = 0
    for prob, c in zip(probs_per_seq, c_per_seq):
        G = len(c)
        j0 = 0
        while j0 < G:
            m = 0
            while m < K and j0 + m < G and prob[j0 + m] >= tau:
                m += 1
            a = 0
            while a < m and c[j0 + a]:
                a += 1
            produced = min(a + 1, G - j0)
            tokens += produced
            cost += 1.0 + m * r
            drafted += m
            accepted += a
            rounds += 1
            j0 += produced
    return {
        "tau": tau,
        "tokens_per_round": tokens / rounds,
        "draft_precision": accepted / drafted if drafted else float("nan"),
        "drafted_per_round": drafted / rounds,
        "cost_per_token": cost / tokens,
        "speedup": tokens / cost,
    }


def run_analysis(train, test, layers, K, r_by_layer):
    results = {}
    taus = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    for L in layers:
        Xtr, ctr = stack(train, L)
        Xte, cte = stack(test, L)
        lens = np.cumsum([0] + [len(r["y"]) for r in test])
        te_lens = [len(r["y"]) for r in test]
        split_at = lambda v: [v[lens[i]:lens[i + 1]] for i in range(len(test))]
        c_test_seq = split_at(cte)

        logreg = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
        logreg.fit(Xtr, ctr)
        mlp = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-3, early_stopping=True, max_iter=300, random_state=0),
        )
        mlp.fit(Xtr, ctr)

        pmax_te = Xte[:, -3]
        gates = {
            "pmax": pmax_te,
            "logreg": logreg.predict_proba(Xte)[:, 1],
            "mlp": mlp.predict_proba(Xte)[:, 1],
        }
        single_class = len(np.unique(cte)) < 2
        L_res = {
            "train_positions": int(len(ctr)),
            "test_positions": int(len(cte)),
            "base_accept_rate": float(cte.mean()),
            "top1_agreement_train": float(ctr.mean()),
            "auc": {k: (float("nan") if single_class else float(roc_auc_score(cte, v))) for k, v in gates.items()},
            "r_cost": r_by_layer[L],
            "sim": {},
        }
        for name, prob in gates.items():
            prob_seq = split_at(prob)
            L_res["sim"][name] = [simulate(prob_seq, c_test_seq, K, tau, r_by_layer[L]) for tau in taus]
        oracle = [c.astype(float) for c in c_test_seq]
        L_res["sim"]["oracle"] = [simulate(oracle, c_test_seq, K, 0.5, r_by_layer[L])]
        results[L] = L_res
    return results


def print_report(results, K):
    print(f"\n=== COM-47 feasibility (K={K} drafts per round) ===")
    for L, res in results.items():
        print(f"\nLayer {L}: draft cost ratio r={res['r_cost']:.3f}")
        print(f"  train positions={res['train_positions']}  test positions={res['test_positions']}")
        print(f"  base accept rate (test, per-position given correct prefix): {res['base_accept_rate']:.3f}")
        print("  AUC for predicting acceptance: " + ", ".join(f"{k}={v:.3f}" for k, v in res["auc"].items()))
        print("  tau   gate    tok/round  draft_prec  drafted/round  cost/token  speedup")
        for name in ("pmax", "logreg", "mlp"):
            for row in res["sim"][name]:
                print(
                    f"  {row['tau']:<5} {name:<7} {row['tokens_per_round']:>9.3f}  {row['draft_precision']:>10.3f}"
                    f"  {row['drafted_per_round']:>13.3f}  {row['cost_per_token']:>10.3f}  {row['speedup']:>7.3f}"
                )
        o = res["sim"]["oracle"][0]
        print(
            f"  oracle gate upper bound: tok/round={o['tokens_per_round']:.3f} "
            f"cost/token={o['cost_per_token']:.3f} speedup={o['speedup']:.3f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--n-per-set", type=int, default=60)
    ap.add_argument("--gen-len", type=int, default=128)
    ap.add_argument("--layers", default="4,8,12,16,20,24,28")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--cache", default="/tmp/com47/records.pkl")
    ap.add_argument("--out", default="/tmp/com47/results.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 4)
    layers = [int(x) for x in args.layers.split(",")]
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    N = model.config.num_hidden_layers
    if N not in layers:
        layers.append(N)
    layer_params = sum(p.numel() for p in model.model.layers[0].parameters())
    head_params = model.lm_head.weight.numel()
    r_by_layer = {L: (L * layer_params + head_params) / (N * layer_params + head_params) for L in layers}
    print(f"model={args.model} layers={N} layer_params={layer_params} head_params={head_params}", flush=True)

    if os.path.exists(args.cache):
        with open(args.cache, "rb") as f:
            recs = pickle.load(f)
        print(f"loaded {len(recs)} cached sequences", flush=True)
    else:
        recs = build_records(args, model, tok, layers)
        with open(args.cache, "wb") as f:
            pickle.dump(recs, f)

    sanity = {L: float(np.mean(np.concatenate([(r["layers"][L]["d"] == r["y"]) for r in recs]))) for L in layers}
    print("sanity: draft agreement with full model per layer (layer N must be ~1.0):")
    for L in layers:
        print(f"  L={L}: {sanity[L]:.3f}")
    assert sanity[N] > 0.99, "final-layer logit lens does not reproduce the model output"

    train, test = split(recs, args.seed)
    draft_layers = [L for L in layers if L != N]
    results = run_analysis(train, test, draft_layers, args.K, r_by_layer)
    print_report(results, args.K)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "K": args.K, "n_seqs": len(recs), "gen_len": args.gen_len,
                   "layer_sanity": sanity, "results": {str(k): v for k, v in results.items()}}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
