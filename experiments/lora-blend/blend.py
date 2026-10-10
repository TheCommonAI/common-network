#!/usr/bin/env python3
"""LoraHub-style blending in plain torch, and the gradient-free search over blend weights.

The arithmetic
--------------

Applying one adapter to a frozen weight is `W_eff = W + (alpha/r) * B @ A`. LoraHub
observes that you can blend several adapters by weighting the *factors* rather than the
products:

    blend = W + (alpha/r) * (sum_i w_i B_i) @ (sum_i w_i A_i)

with `w` on the simplex -- non-negative, summing to one. The parenthesised sums are each
still rank r, so the blend is an adapter of the same shape as its inputs and costs the
same to apply. That is the entire trick, and it is why nothing here needs `peft`: forming
those sums is tensor addition, and this module does it with `torch` and writes the result
with `safetensors`.

There is one thing worth being explicit about, because it is the mistake that makes a
blend look right and be wrong: the product is taken of the *sums*, not summed over
products. `sum_i w_i (B_i @ A_i)` is a different matrix -- a sum of rank-r terms, so rank
up to n*r -- and it is not what any of these adapters does. `blend_sums` forms the sums
and `LoRALinear.set_blend` takes the product; the two halves of the expression live in
different files and neither is correct alone.

The search
-----------

The weights are chosen on the held-out slice only, never on the 50 scored items, under a
fixed candidate budget. Two consequences of a 20-item held-out slice are stated in the
code rather than hidden:

  * Accuracy on it moves in steps of 1/20 (and 1/18 for the maths-only items), so ties
    are common and the search can return a candidate that is not uniquely best.
  * Ties are broken toward *sparser* weights -- fewer adapters active. That is the
    conservative direction: on a flat objective it collapses the blend toward a single
    adapter, so it can only make the blend's eventual score worse, never better.

The full candidate trace is returned and written to results.json, so the tie structure is
visible to the reader rather than summarised away.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent


# --- the blend itself ---------------------------------------------------------------

def blend_sums(matrices_list: list[dict], weights: list[float]) -> dict:
    """`sum_i w_i A_i` and `sum_i w_i B_i` per module. Raw, unscaled.

    The output is deliberately left unscaled so it can be written in PEFT's layout, where
    `r` and `lora_alpha` live in `adapter_config.json` and the `alpha/r` factor is applied
    by whatever loads the adapter. Folding the scaling into `A` here would be silently
    undone -- or worse, applied twice -- by a PEFT loader.

    The arithmetic itself is `harness.weighted_factor_sums`, shared with the wrapper that
    applies a blend at inference time. It was written out twice before, here and there,
    and the two copies drifted -- see that function. Imported inside the call because
    `harness` imports nothing from this module until `_score` needs it.
    """
    from harness import weighted_factor_sums

    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for path, (a0, b0) in matrices_list[0].items():
        if any(path not in m for m in matrices_list):
            continue
        A, B = weighted_factor_sums([m[path] for m in matrices_list], weights)
        if A is None:  # all weights zero; only reachable from a caller error
            A, B = torch.zeros_like(a0), torch.zeros_like(b0)
        out[path] = (A.to(torch.float32), B.to(torch.float32))
    return out


def save_peft_adapter(blend: dict, out_dir: Path, base_model: str, r: int,
                      lora_alpha: float, target_modules) -> Path:
    """Write a blend as a loadable PEFT adapter directory.

    Exists to measure something concrete and to be independently checkable: anyone can
    point `PeftModel.from_pretrained` at this directory and get the same model the
    harness scored. That is a stronger claim than "the numbers in results.json came from
    a blend", and it costs one file write.
    """
    from safetensors.torch import save_file

    tensors = {}
    for path, (A, B) in blend.items():
        # `path` is a named_modules() path such as model.layers.0.self_attn.q_proj, which
        # is exactly what PEFT's own save prefixes with base_model.model.
        tensors[f"base_model.model.{path}.lora_A.weight"] = A.contiguous()
        tensors[f"base_model.model.{path}.lora_B.weight"] = B.contiguous()

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out_dir / "adapter_model.safetensors", metadata={"format": "pt"})
    (out_dir / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": r,
        "target_modules": list(target_modules),
        "task_type": "CAUSAL_LM",
    }, indent=2) + "\n")
    return out_dir


def build_blend(matrices_list: list[dict], weights: list[float], out_dir: Path,
                base_model: str, entry: dict) -> tuple[Path, float, dict]:
    """Time the tensor averaging and the write, separately from search and inference.

    This is the "how long does it take to build a blend on a CPU" number, and it is
    deliberately measured with nothing else in the timed region -- no model load, no
    generation, no weight search. The search is the expensive part of this experiment by
    orders of magnitude and reporting the two together would make the build number
    meaningless.
    """
    started = time.perf_counter()
    blend = blend_sums(matrices_list, weights)
    path = save_peft_adapter(blend, out_dir, base_model, entry["r"],
                             entry["lora_alpha"], entry["target_modules"])
    elapsed = time.perf_counter() - started
    total_bytes = sum(t.numel() * t.element_size()
                      for pair in blend.values() for t in pair)
    return path, elapsed, {"blend_bytes": total_bytes, "modules": len(blend)}


def verify_artifact(path: Path, matrices_list: list[dict], weights: list[float],
                    harness, scaling: float, prompt: str = "What is 2 + 2?") -> dict:
    """Check the written artifact reproduces the in-memory blend, two ways.

    Tensor equality catches a layout or naming mistake in the save. The logits comparison
    catches everything the tensor check cannot see -- a key written where nothing reads
    it, a scaling applied on one path and not the other, a weight that survives the write
    but not the reload. A blend that scores 31% in the harness and 12% through PEFT would
    invalidate every number in results.json, and this is the cheap way to rule it out:
    apply the in-memory matrices and the reloaded ones to the same model, and require the
    next-token distributions to be identical.
    """
    from safetensors.torch import load_file

    raw = load_file(str(path / "adapter_model.safetensors"))
    reloaded = {}
    for key in raw:
        if not key.endswith(".lora_A.weight"):
            continue
        stem = key[len("base_model.model."):-len(".lora_A.weight")]
        reloaded[stem] = (raw[key], raw[f"base_model.model.{stem}.lora_B.weight"])

    in_memory = blend_sums(matrices_list, weights)
    if set(reloaded) != set(in_memory):
        missing = set(in_memory) - set(reloaded)
        extra = set(reloaded) - set(in_memory)
        return {"ok": False, "reason": "module sets differ",
                "missing": sorted(missing)[:3], "extra": sorted(extra)[:3]}

    max_matrix_diff = max(
        max((reloaded[p][i] - in_memory[p][i]).abs().max().item() for i in (0, 1))
        for p in in_memory)

    harness.use_blend(matrices_list, weights, scaling)
    logits_blend = harness.next_token_logits(prompt)
    harness.use_single(reloaded, scaling)
    logits_reloaded = harness.next_token_logits(prompt)
    harness.clear()
    max_logit_diff = (logits_blend - logits_reloaded).abs().max().item()

    return {
        "ok": max_matrix_diff == 0.0 and max_logit_diff < 1e-3,
        "max_matrix_diff": max_matrix_diff,
        "max_logit_diff": max_logit_diff,
        "modules": len(reloaded),
    }


# --- the weight search --------------------------------------------------------------

def _simplex_candidates(rng: random.Random, n: int, count: int) -> list[list[float]]:
    """Anchors first, then uniform draws on the simplex.

    The anchors are the n single-adapter one-hots plus the uniform blend. Starting from
    them is not a heuristic nicety: it guarantees the search never returns something
    *worse* than the best single adapter it was given, so a degenerate answer is a real
    finding about the held-out slice rather than an artefact of where the random draws
    happened to land.
    """
    candidates = []
    for i in range(n):
        weights = [0.0] * n
        weights[i] = 1.0
        candidates.append(weights)
    candidates.append([1.0 / n] * n)
    for _ in range(count):
        draws = [rng.expovariate(1.0) for _ in range(n)]
        total = sum(draws)
        candidates.append([d / total for d in draws])
    return candidates


def _prefer(new_score: float, new_weights: list[float],
            best_score: float, best_weights: list[float]) -> bool:
    """Higher score wins; on a tie, fewer active adapters wins.

    See the module docstring: this tie-break points the search toward a single adapter,
    which is the direction that can only cost the blend accuracy. Ties are not a corner
    case here -- at 20 held-out items a score can only take 21 values, so on a slice the
    models find equally hard most candidates land on the same number.
    """
    if new_score != best_score:
        return new_score > best_score
    new_active = sum(1 for w in new_weights if w > 0)
    best_active = sum(1 for w in best_weights if w > 0)
    return new_active < best_active


def search_weights(harness, matrices_list: list[dict], adapter_ids: list[str], items: list[dict],
                   scaling: float, *, seed: int, random_candidates: int = 10,
                   coordinate_rounds: int = 1, step: float = 0.2,
                   max_new_tokens: int = 200, log=print) -> dict:
    """Find blend weights on the held-out slice under a fixed candidate budget.

    Gradient-free by design. The objective is a count of correct answers out of 20, so it
    is piecewise constant and has no useful gradient to follow; a random search over the
    simplex followed by coordinate moves is the honest tool for a surface like that, and
    it has the property that every evaluation is a real measurement.

    Deterministic given (seed, budget). Greedy decoding means candidate k scores the same
    on every run, and the random draws come from a seeded `random.Random`, so re-running
    reproduces the weights exactly.
    """
    n = len(matrices_list)
    rng = random.Random(seed)
    cache: dict[tuple, float] = {}
    trace: list[dict] = []
    evaluations = 0

    def evaluate(weights: list[float], label: str) -> float:
        nonlocal evaluations
        key = tuple(round(w, 6) for w in weights)
        if key in cache:
            return cache[key]
        harness.use_blend(matrices_list, weights, scaling)
        correct = 0
        for item in items:
            completion, _ = harness.generate(item["question"], item["answer_type"],
                                             max_new_tokens)
            if _score(item, completion):
                correct += 1
        score = correct / len(items)
        evaluations += 1
        cache[key] = score
        trace.append({"label": label, "weights": [round(w, 4) for w in weights],
                      "score": score, "correct": correct})
        log(f"    [{evaluations:>2}] {label:<22} score={score:.3f} "
            f"({correct}/{len(items)})  {_format_weights(adapter_ids, weights)}")
        return score

    best_weights = [1.0 / n] * n
    best_score = evaluate(best_weights, "uniform")
    for weights in _simplex_candidates(rng, n, random_candidates):
        # Anchors are re-evaluated through the cache, so this costs no generations
        label = ("single:" + adapter_ids[weights.index(1.0)]) if sum(
            1 for w in weights if w > 0) == 1 else "random"
        score = evaluate(weights, label)
        if _prefer(score, weights, best_score, best_weights):
            best_weights, best_score = weights, score

    for round_index in range(coordinate_rounds):
        improved = False
        for i in range(n):
            # Move weight to adapter i from whichever other adapter currently holds the
            # least. One evaluation per recipient rather than one per ordered pair: the
            # full n*(n-1) sweep is 30 generations of a 20-item slice for six adapters,
            # two thirds of which are moves off an adapter the search has already decided
            # is not pulling its weight. The budget is the scarce resource here.
            donors = [j for j in range(n) if j != i and best_weights[j] > 0]
            if not donors:
                continue
            j = min(donors, key=lambda k: best_weights[k])
            shift = min(step, best_weights[j])
            weights = list(best_weights)
            weights[i] += shift
            weights[j] -= shift
            score = evaluate(weights, f"coord r{round_index + 1} {adapter_ids[j]}->{adapter_ids[i]}")
            if _prefer(score, weights, best_score, best_weights):
                best_weights, best_score = weights, score
                improved = True
        if not improved:
            break

    active = [a for a, w in zip(adapter_ids, best_weights) if w > 0]
    return {
        "weights": {a: round(w, 6) for a, w in zip(adapter_ids, best_weights)},
        "active": active,
        "degenerate": len(active) == 1,
        "heldout_score": best_score,
        "heldout_n": len(items),
        "evaluations": evaluations,
        "distinct_candidates": len(cache),
        "budget": {"random_candidates": random_candidates,
                   "coordinate_rounds": coordinate_rounds, "step": step, "seed": seed},
        "trace": trace,
    }


def _score(item: dict, completion: str) -> bool:
    from harness import score as score_completion
    return score_completion(completion, item)[1]


def _format_weights(adapter_ids: list[str], weights: list[float]) -> str:
    return " ".join(f"{a}={w:.2f}" for a, w in zip(adapter_ids, weights) if w > 0.001)


def check_apply_matches_write() -> dict:
    """Assert the wrapper's blend and the writer's blend are the same arithmetic.

    No model, no adapter download: one tiny `nn.Linear` and three random factor pairs. The
    two entry points -- `LoRALinear.set_blend`, which scores a configuration, and
    `blend_sums`, which produces the artifact -- are a single implementation now, and this
    is what keeps them one.

    It exists because they were two, and the second omitted the weight on its first term:
    the harness scored one blend and wrote a different one, both silently, with plausible
    numbers on each side. Only `verify_artifact` noticed, and only after a 50-minute weight
    search had been spent scoring a blend that was not the one being recorded. This check
    costs milliseconds and runs before anything expensive.
    """
    import torch.nn as nn

    from harness import LoRALinear

    generator = torch.Generator().manual_seed(11)
    pairs = [(torch.randn(2, 5, generator=generator),
              torch.randn(4, 2, generator=generator)) for _ in range(3)]
    weights = [0.21, 0.35, 0.44]

    wrapper = LoRALinear(nn.Linear(5, 4, bias=False))
    wrapper.set_blend(pairs, weights, 1.0)
    written_A, written_B = blend_sums([{"p": pair} for pair in pairs], weights)["p"]

    return {"A_equal": torch.equal(wrapper.A, written_A),
            "B_equal": torch.equal(wrapper.B, written_B)}


if __name__ == "__main__":
    # Build a blend out of the first two blendable adapters and check that what lands on
    # disk is what the harness would score. No model download needed beyond the adapters.
    import sys

    from harness import Harness, load_adapter_matrices, load_registry, scaling_for

    match = check_apply_matches_write()
    print(f"apply/write agreement: {match}")
    if not (match["A_equal"] and match["B_equal"]):
        print("the wrapper and the artifact writer disagree on the blend; refusing to run")
        sys.exit(1)

    registry = load_registry()
    blendable = [a for a in registry["adapters"] if a.get("blendable")]
    picked = blendable[:2]
    print(f"blending {[a['id'] for a in picked]}")

    matrices = [load_adapter_matrices(a) for a in picked]
    weights = [0.5, 0.5]
    out = HERE / "blends" / "selftest"
    path, seconds, info = build_blend(matrices, weights, out, registry["base_model"], picked[0])
    print(f"built in {seconds * 1000:.0f} ms, {info['blend_bytes'] / 1e6:.1f} MB, "
          f"{info['modules']} modules -> {path}")

    harness = Harness(registry["base_model"]).load()
    check = verify_artifact(path, matrices, weights, harness, scaling_for(picked[0]))
    print(f"artifact check: {check}")
    sys.exit(0 if check.get("ok") else 1)
