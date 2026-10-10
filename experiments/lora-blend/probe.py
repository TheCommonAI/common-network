#!/usr/bin/env python3
"""Scores base, the maths adapters and the blend on GSM8K and MMLU, as a contamination probe.

Why this exists
---------------

The ten maths families in `questions_maths.json` are the GSM8K template space -- unit
prices, ratios, rates, percentages. Generating fresh *numbers* does not remove *template*
memorisation: a model that has seen ten thousand GSM8K unit-price problems has seen this
kind of question whatever the figures are.

So a maths adapter can score well on the 50 for two different reasons, and the scores
alone cannot tell them apart:

  * it is genuinely better at this kind of arithmetic, or
  * it memorised the shape and the training distribution leaked into the test set.

Scoring the same configurations on GSM8K's own held-out test split separates them. If an
adapter scores much higher on GSM8K than on my set, it was tuned toward that distribution
and its score on mine is partly recall. If the two move together, the adapter transferred.

The confound cuts in both directions and this probe is honest about which way each half
goes. A memorising adapter *inflates* (b), the best single, which makes the blend's job
harder -- a conservative error. But a blend that inherits a memorising adapter inherits
the inflation too, which is anti-conservative. Both are visible in the table rather than
assumed away.

MMLU is the control *within* the probe. It is maths-adjacent but not GSM8K-shaped, so an
adapter that jumps on GSM8K and not on MMLU is showing distribution fit, while one that
jumps on both is showing a broader maths gain.

This file never contributes a number to the headline result. It is reported beside it.

Usage
-----

    python probe.py                       # base + maths adapters + blend
    python probe.py --gsm8k 60 --mmlu 40
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from harness import Harness, extract_number, load_adapter_matrices, load_registry, scaling_for

HERE = Path(__file__).parent
SEED = 90210
_LETTER = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")


def gsm8k_items(count: int) -> list[dict]:
    """GSM8K test items. Ground truth is the number after the `####` marker."""
    import datasets

    data = datasets.load_dataset("openai/gsm8k", "main", split="test")
    index = random.Random(SEED).sample(range(len(data)), count)
    items = []
    for position, i in enumerate(sorted(index)):
        row = data[i]
        marker = row["answer"].rsplit("####", 1)[-1].strip().replace(",", "")
        items.append({
            "id": f"gsm8k-{position:03d}",
            "question": row["question"],
            "answer": marker,
        })
    return items


def mmlu_items(count: int) -> list[dict]:
    """MMLU high-school mathematics. Maths-adjacent, but not GSM8K-shaped."""
    import datasets

    data = datasets.load_dataset("cais/mmlu", "high_school_mathematics", split="test")
    index = random.Random(SEED + 1).sample(range(len(data)), count)
    items = []
    for position, i in enumerate(sorted(index)):
        row = data[i]
        letters = "ABCD"
        options = "\n".join(
            f"{letters[k]}) {choice}" for k, choice in enumerate(row["choices"]))
        items.append({
            "id": f"mmlu-{position:03d}",
            "question": f"{row['question']}\n\n{options}",
            "answer": letters[row["answer"]],
        })
    return items


def gsm8k_correct(completion: str, expected: str) -> bool:
    """GSM8K's own convention: the number after `####`, falling back to the last number."""
    if "####" in completion:
        value = extract_number(completion.rsplit("####", 1)[-1])
    else:
        value = extract_number(completion)
    if value is None:
        return False
    try:
        return abs(value - float(expected)) < 1e-6
    except ValueError:
        return False


def mmlu_correct(completion: str, expected: str) -> bool:
    found = _LETTER.findall(completion)
    return bool(found) and found[-1] == expected


def probe_set(harness, items: list[dict], kind: str, max_new_tokens: int, label: str) -> dict:
    instruction = ("Give your final answer as a single number." if kind == "gsm8k"
                   else "Answer with a single letter, A, B, C or D.")
    records = []
    for item in items:
        prompt = harness.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{item['question']}\n\n{instruction}"}],
            tokenize=False, add_generation_prompt=True)
        inputs = harness.tokenizer(prompt, return_tensors="pt").to(harness.device)
        import torch

        with torch.no_grad():
            output = harness.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=harness.tokenizer.pad_token_id or harness.tokenizer.eos_token_id)
        completion = harness.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        correct = (gsm8k_correct if kind == "gsm8k" else mmlu_correct)(completion, item["answer"])
        records.append({"id": item["id"], "expected": item["answer"],
                        "completion": completion[:400], "correct": correct})
    hit = sum(r["correct"] for r in records)
    print(f"    {label:<15} {kind:<6} {hit:>3}/{len(records)} = {hit / len(records):.3f}",
          flush=True)
    return {"n": len(records), "correct": hit, "accuracy": hit / len(records), "items": records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsm8k", type=int, default=60)
    parser.add_argument("--mmlu", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--out", default="probe.json")
    args = parser.parse_args()

    registry = load_registry()
    maths_adapters = [a for a in registry["adapters"]
                      if a["domain"] == "maths" and a.get("blendable")]

    weights = None
    results_path = HERE / "results.json"
    if results_path.exists():
        weights = json.loads(results_path.read_text())["blend"]["weights"]
        print(f"blend weights from results.json: {weights}")
    else:
        print("no results.json yet -- probing base and the maths adapters only")

    print(f"\nloading {args.gsm8k} GSM8K and {args.mmlu} MMLU items")
    gsm8k = gsm8k_items(args.gsm8k)
    mmlu = mmlu_items(args.mmlu)

    matrices = {a["id"]: load_adapter_matrices(a) for a in maths_adapters}
    harness = Harness(registry["base_model"]).load()
    print(f"{harness.device} / {harness.dtype}\n")

    configs: dict[str, dict] = {}

    print("base")
    harness.clear()
    configs["base"] = {
        "gsm8k": probe_set(harness, gsm8k, "gsm8k", args.max_new_tokens, "base"),
        "mmlu": probe_set(harness, mmlu, "mmlu", args.max_new_tokens, "base"),
    }

    for entry in maths_adapters:
        print(entry["id"])
        harness.use_single(matrices[entry["id"]], scaling_for(entry))
        configs[entry["id"]] = {
            "gsm8k": probe_set(harness, gsm8k, "gsm8k", args.max_new_tokens, entry["id"]),
            "mmlu": probe_set(harness, mmlu, "mmlu", args.max_new_tokens, entry["id"]),
        }

    if weights:
        blendable = [a for a in registry["adapters"] if a.get("blendable")]
        available = [a for a in blendable if a["id"] in matrices]
        if len(available) == len(blendable):
            print("blend")
            harness.use_blend([matrices[a["id"]] for a in blendable],
                               [weights[a["id"]] for a in blendable],
                               scaling_for(blendable[0]))
            configs["blend"] = {
                "gsm8k": probe_set(harness, gsm8k, "gsm8k", args.max_new_tokens, "blend"),
                "mmlu": probe_set(harness, mmlu, "mmlu", args.max_new_tokens, "blend"),
            }
    harness.clear()

    # The number the probe is actually for: how each configuration's GSM8K score compares
    # with its score on the authored maths set. A large positive gap is the fingerprint of
    # having been tuned on GSM8K-shaped data.
    mine = {}
    if results_path.exists():
        for name, config in json.loads(results_path.read_text())["configs"].items():
            mine[name] = config.get("maths", {}).get("accuracy")

    print("\n=== contamination probe ===")
    print(f"  {'config':<15} {'GSM8K':>7} {'MMLU':>7} {'my maths':>9} {'GSM8K - mine':>13}")
    for name, config in configs.items():
        gap = "" if name not in mine else f"{100 * (config['gsm8k']['accuracy'] - mine[name]):+7.1f}"
        mine_text = "" if name not in mine else f"{mine[name]:.3f}"
        print(f"  {name:<15} {config['gsm8k']['accuracy']:>7.3f} "
              f"{config['mmlu']['accuracy']:>7.3f} {mine_text:>9} {gap:>13}")

    (HERE / args.out).write_text(json.dumps({
        "meta": {
            "seed": SEED,
            "gsm8k_items": len(gsm8k),
            "mmlu_items": len(mmlu),
            "gsm8k_split": "openai/gsm8k main test (held-out, never trained on by us)",
            "mmlu_split": "cais/mmlu high_school_mathematics test",
            "note": "reported beside the headline result, never as part of it",
        },
        "configs": configs,
        "author_maths_accuracy": mine,
    }, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
