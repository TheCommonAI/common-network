#!/usr/bin/env python3
"""Frozen-weight replication of Stage 1 on 200 fresh questions.

What this does and does not do
------------------------------

It runs exactly three configurations — the base model, the single adapter, the blend — on
`questions_maths_200.json`, and it runs them under weights that are **read from
results.json and never searched**. There is no held-out slice here, no candidate trace, no
tie-break. The one number that matters is whether a blend fitted on 20 items still beats
its baselines on 200 it has never seen.

Two things are deliberately frozen, and freezing them is the whole design:

  * **The weights.** Re-searching on the new set would convert an out-of-sample test back
    into an in-sample one, which is the error the replication exists to rule out.
  * **The comparator.** `math-adaanchor` was the best single adapter *on the 50*. If the 200
    would promote a different adapter, we do not switch to it. Switching would hand the
    baseline the same hindsight the blend was denied, and the comparison would flatter the
    blend. The script asserts the comparator has not moved rather than assuming it.

Statistics are `compare` and `verdict` imported from `run_eval`, not reimplemented, so the
McNemar and bootstrap here are literally the same code that produced the Stage 1 numbers.
The five-point rule and the CI tripwire are applied unchanged.

Checkpointing
-------------

Per *item*, not per configuration. At 200 items a configuration is about 35 minutes, so a
per-config checkpoint would put half an hour of work at risk on every kill; per-item puts
about ten seconds at risk. Written atomically, and discarded if the frozen weights, the
question file or the configuration list has changed, so a resumed run cannot splice two
different experiments into one table.

Run: python replicate.py [--resume]
Writes replication.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from blend import build_blend, verify_artifact
from harness import Harness, load_adapter_matrices, load_registry, score, scaling_for
from run_eval import compare, log, verdict, POINTS_TO_PASS

HERE = Path(__file__).parent
ITEMS_FILE = "questions_maths_200.json"
SOURCE_FILE = "results.json"        # where the frozen weights come from
OUT_FILE = "replication.json"
COMPARATOR = "math-adaanchor"       # frozen: best single on the 50, never re-picked
CONFIGS = ("base", COMPARATOR, "blend")


def checkpoint_path() -> Path:
    return HERE / "blends" / "replication-checkpoint.json"


def run_signature(items_bytes: bytes, weights: dict[str, float], max_new_tokens: int) -> dict:
    return {
        "items_sha1": hashlib.sha1(items_bytes).hexdigest(),
        "weights": weights,
        "comparator": COMPARATOR,
        "configs": list(CONFIGS),
        "max_new_tokens": max_new_tokens,
    }


def save_checkpoint(signature: dict, configs: dict) -> None:
    path = checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written to a temporary name and renamed into place: a kill mid-write must not leave a
    # truncated file that the next run would either refuse to trust or, worse, parse as a
    # valid checkpoint with the tail of a configuration missing.
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"signature": signature, "configs": configs}, indent=2) + "\n")
    temporary.replace(path)


def load_checkpoint(signature: dict) -> dict:
    path = checkpoint_path()
    if not path.exists():
        return {}
    try:
        saved = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        log(f"  checkpoint unreadable ({type(error).__name__}: {error}); starting fresh")
        return {}
    if saved.get("signature") != signature:
        log("  checkpoint was written for different weights or questions; ignoring it rather "
            "than mixing two experiments")
        return {}
    return saved.get("configs", {})


def run_config_resumable(harness, items: list[dict], max_new_tokens: int, label: str,
                         signature: dict, configs: dict) -> dict:
    """`run_eval.run_config`, but checkpointed after every item.

    Kept as a local copy rather than a change to run_eval, because run_eval is the file
    that produced the Stage 1 evidence and its checkpoint format is per-configuration. The
    record shape and the log line are identical so `compare` can pair these results with
    any other run's.
    """
    previous = configs.get(label, {}).get("records", [])
    records = list(previous)
    seconds = configs.get(label, {}).get("seconds", 0.0)
    if records:
        log(f"  resuming {label} at item {len(records) + 1}/{len(items)}")

    started = time.perf_counter()
    for index, item in enumerate(items, 1):
        if index <= len(records):
            continue
        completion, elapsed = harness.generate(
            item["question"], item["answer_type"], max_new_tokens)
        extracted, correct = score(completion, item)
        seconds += elapsed
        records.append({
            "id": item["id"],
            "family": item["family"],
            "question": item["question"],
            "expected": item["answer"],
            "completion": completion,
            "extracted": extracted if not isinstance(extracted, str) else extracted[:200],
            "correct": correct,
        })
        configs.setdefault(label, {})["records"] = records
        configs[label]["seconds"] = seconds
        save_checkpoint(signature, configs)

        running = sum(r["correct"] for r in records) / len(records)
        log(f"    {label:<14} {index:>3}/{len(items)}  "
            f"{'ok ' if correct else '   '} {elapsed:>5.1f}s  acc={running:.3f}  {item['id']}")
    _ = started

    correct = sum(r["correct"] for r in records)
    return {
        "n": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "seconds": seconds,
        "items": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0,
                        help="cap items, for a smoke run before committing 2.5 hours")
    parser.add_argument("--resume", action="store_true",
                        help="continue a killed replication from its per-item checkpoint")
    args = parser.parse_args()

    source = json.loads((HERE / SOURCE_FILE).read_text())
    frozen = {k: float(v) for k, v in source["blend"]["weights"].items()}
    frozen_active = sorted(k for k, v in frozen.items() if v > 0)
    source_best = source["comparison"]["best_single"]

    log(f"frozen weights from {SOURCE_FILE}: {frozen}")
    log(f"  active {frozen_active}  (searched on {source['blend']['heldout_n']} held-out items)")

    # The comparator is an assertion, not a default. If results.json ever says a different
    # adapter was the best single, this replication no longer means what it claims to and
    # must not run silently.
    if source_best != COMPARATOR:
        log(f"refusing to run: {SOURCE_FILE} names {source_best} as the best single, but this "
            f"replication is pinned to {COMPARATOR}. Re-decide the comparator deliberately "
            f"rather than letting it drift with the data.")
        return 1
    if abs(sum(frozen.values()) - 1.0) > 1e-6:
        log(f"refusing to run: frozen weights sum to {sum(frozen.values())}, not 1")
        return 1

    items_bytes = (HERE / ITEMS_FILE).read_bytes()
    items = json.loads(items_bytes)["items"]
    log(f"\n{ITEMS_FILE}: {len(items)} items")
    if args.limit:
        items = items[:args.limit]
        log(f"  smoke: restricted to the first {len(items)}")

    signature = run_signature(items_bytes, frozen, args.max_new_tokens)
    configs = load_checkpoint(signature) if args.resume else {}
    if args.resume:
        log(f"  resumed: {sorted(configs)}")
    else:
        checkpoint_path().unlink(missing_ok=True)

    registry = load_registry()
    log("\nloading adapters")
    matrices_by_id = {}
    for entry in registry["adapters"]:
        try:
            matrices_by_id[entry["id"]] = load_adapter_matrices(entry)
        except Exception as error:  # a fragile adapter must not take the run down
            log(f"  {entry['id']:<15} UNAVAILABLE ({type(error).__name__}: {error})")

    blendable = [a for a in registry["adapters"]
                 if a.get("blendable") and a["id"] in matrices_by_id]
    blendable_ids = [a["id"] for a in blendable]
    if COMPARATOR not in matrices_by_id or len(blendable) < 2:
        log(f"refusing to run: need {COMPARATOR} plus at least two blendable adapters")
        return 1

    weights = []
    for entry in blendable:
        if entry["id"] not in frozen:
            log(f"refusing to run: {entry['id']} is blendable but has no frozen weight")
            return 1
        weights.append(frozen[entry["id"]])
    log(f"  blend over {blendable_ids} -> {[round(w, 4) for w in weights]}")

    log("\nloading base model")
    harness = Harness(registry["base_model"]).load()
    log(f"  {harness.device} / {harness.dtype}  loaded in {harness.load_seconds:.1f}s")

    # Rebuild the artifact from the frozen weights and re-verify it against the in-memory
    # path the scores come from. Cheap, and it means a number in replication.json can be
    # reproduced by anyone pointing PEFT at the files this writes.
    log("\nrebuilding the blend artifact from the frozen weights")
    artifact_dir = HERE / "blends" / "replication-blend"
    path, build_seconds, build_info = build_blend(
        [matrices_by_id[i] for i in blendable_ids], weights, artifact_dir,
        registry["base_model"], blendable[0])
    log(f"  wrote {path.name} in {build_seconds * 1000:.0f} ms "
        f"({build_info['blend_bytes'] / 1e6:.1f} MB, {build_info['modules']} modules)")
    artifact_check = verify_artifact(
        path, [matrices_by_id[i] for i in blendable_ids], weights, harness,
        scaling_for(blendable[0]))
    log(f"  verification: {artifact_check}")
    if not artifact_check.get("ok"):
        log("  the artifact does not reproduce the in-memory blend; stopping before the "
            "scored run so no number depends on which path produced it")
        return 1

    results = {}
    for name in CONFIGS:
        log(f"\n{name} on the {len(items)} replication questions")
        if name == "base":
            harness.clear()
        elif name == "blend":
            harness.use_blend([matrices_by_id[i] for i in blendable_ids], weights,
                              scaling_for(blendable[0]))
        else:
            entry = next(e for e in registry["adapters"] if e["id"] == name)
            harness.use_single(matrices_by_id[name], scaling_for(entry))
        results[name] = run_config_resumable(
            harness, items, args.max_new_tokens, name, signature, configs)
    harness.clear()

    vs_base = compare("blend", results["blend"], "base", results["base"])
    vs_single = compare("blend", results["blend"], COMPARATOR, results[COMPARATOR])
    decision = verdict(vs_base, vs_single)

    log(f"\n=== replication: accuracy on the {len(items)} fresh questions ===")
    for name in sorted(results, key=lambda n: -results[n]["accuracy"]):
        marker = {"base": "(a)", "blend": "(c)"}.get(name, "(b)")
        log(f"  {marker} {name:<15} {results[name]['accuracy']:.3f}  "
            f"({results[name]['correct']}/{results[name]['n']})")

    log("")
    for label, comparison in (("(c) - (a)", vs_base), ("(c) - (b)", vs_single)):
        log(f"  {label} = {comparison['delta_points']:+.1f} points    "
            f"McNemar p={comparison['mcnemar']['p_value']:.3f}    "
            f"CI [{comparison['bootstrap']['lo']:+.3f}, {comparison['bootstrap']['hi']:+.3f}]"
            f"    discordant {comparison['mcnemar']['discordant']} "
            f"({comparison['mcnemar']['first_only']} blend-only, "
            f"{comparison['mcnemar']['second_only']} baseline-only)")

    log(f"\nverdict: {'PROCEED to Stage 2' if decision['proceed'] else 'STOP'}"
        f" -- {decision['reason']}")
    if decision["contradiction"]:
        log("\nThe point estimate clears the threshold while the paired evidence does not "
            "support it. This comes back to a human rather than resolving itself.")

    payload = {
        "meta": {
            "what": "frozen-weight replication of Stage 1 on fresh questions",
            "base_model": registry["base_model"],
            "device": harness.device,
            "greedy": True,
            "max_new_tokens": args.max_new_tokens,
            "items": len(items),
            "items_file": ITEMS_FILE,
            "items_sha1": signature["items_sha1"],
            "weights_source": SOURCE_FILE,
            "frozen_weights": frozen,
            "frozen_active": frozen_active,
            "comparator": COMPARATOR,
            "comparator_note": "the best single adapter on the 50, held fixed. If the 200 "
                               "would promote a different adapter it is not switched to; "
                               "see the module docstring.",
            "search": "none -- weights were not re-searched on this set",
            "build_seconds": build_seconds,
            "notes": [
                "Fresh instances of the same ten families, not a second domain. This "
                "generalises across instances of a family, not across domains.",
                "Pairs with results.json only in the sense that the weights come from it. "
                "The 200 items are disjoint from both the 50 scored and the 20 held out.",
            ],
        },
        "configs": results,
        "comparison": {"frozen_comparator": COMPARATOR,
                       "blend_vs_base": vs_base,
                       "blend_vs_comparator": vs_single},
        "verdict": decision | {"points_required": POINTS_TO_PASS},
        "artifact_check": artifact_check,
        "artifact": str(path.relative_to(HERE)),
    }
    (HERE / OUT_FILE).write_text(json.dumps(payload, indent=2) + "\n")
    log(f"\nwrote {OUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
