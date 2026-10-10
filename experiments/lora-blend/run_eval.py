#!/usr/bin/env python3
"""Scores every configuration on both question sets and writes results.json.

The comparison being made
-------------------------

  (a) base        -- Qwen2.5-1.5B-Instruct, untouched
  (b) best single -- whichever single adapter scores highest on the 50 maths questions
                     *themselves*
  (c) blend       -- LoraHub weights, searched on the 20 held-out items only

(b) is deliberately chosen with hindsight. Picking the strongest adapter by its score on
the very questions the blend is then judged on makes it the hardest available baseline: it
already knew the answers. A weaker baseline would flatter the blend. The harder comparison
is the one worth reporting.

The verdict rule
----------------

Stage 2 proceeds when (c) beats both (a) and (b) by at least 5 points. That is the
threshold that was set, and it is a crude one by design -- at n=50 a single question is
worth 2 points, so 5 points is two and a half questions and anything smaller is inside the
noise floor.

The paired statistics are reported against every baseline and are a tripwire rather than a
veto. At n=50 the bootstrap half-width is roughly +-14 points, so a real 6-point gain will
essentially never reach significance; making the paired test a hard gate would throw away
genuine wins. Making it the only thing reported would let a 5-point wobble be called a
result. So: the 5-point rule gates, and if a gain clears 5 points while the paired CI
comfortably includes zero, the run stops and reports that contradiction instead of
resolving it.

Usage
-----

    python run_eval.py --smoke     # 5 questions, all config kinds, hard gate
    python run_eval.py             # the full run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from blend import build_blend, check_apply_matches_write, search_weights, verify_artifact
from harness import (Harness, load_adapter_matrices, load_registry, score, scaling_for)

HERE = Path(__file__).parent
SMOKE_N = 5
POINTS_TO_PASS = 5.0  # percentage points, (c) versus both baselines


def load_items(name: str) -> list[dict]:
    return json.loads((HERE / name).read_text())["items"]


def log(message: str) -> None:
    print(message, flush=True)


# --- scoring one configuration ------------------------------------------------------

def run_config(harness, items: list[dict], max_new_tokens: int,
               label: str = "") -> dict:
    """Generate and mark every item under the configuration currently loaded."""
    records = []
    started = time.perf_counter()
    for index, item in enumerate(items, 1):
        completion, seconds = harness.generate(
            item["question"], item["answer_type"], max_new_tokens)
        extracted, correct = score(completion, item)
        records.append({
            "id": item["id"],
            "family": item["family"],
            "question": item["question"],
            "expected": item["answer"],
            "completion": completion,
            "extracted": extracted if not isinstance(extracted, str) else extracted[:200],
            "correct": correct,
        })
        running = sum(r["correct"] for r in records) / len(records)
        log(f"    {label:<14} {index:>3}/{len(items)}  "
            f"{'ok ' if correct else '   '} {seconds:>5.1f}s  acc={running:.3f}  "
            f"{item['id']}")
    correct = sum(r["correct"] for r in records)
    return {
        "n": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "seconds": time.perf_counter() - started,
        "items": records,
    }


def parsed_rate(result: dict, items: list[dict]) -> float:
    """Share of completions from which the extractor got *anything* at all.

    Separate from accuracy on purpose. A configuration scoring 0% because it answered
    wrongly and one scoring 0% because the extractor returned None are different failures,
    and the second one would mean the parser, not the model, is being measured.
    """
    by_id = {i["id"]: i for i in items}
    hits = 0
    for record in result["items"]:
        if by_id[record["id"]]["answer_type"] == "number":
            hits += record["extracted"] is not None
        else:
            hits += bool(record["extracted"])
    return hits / len(result["items"])


# --- the paired statistics -----------------------------------------------------------

def mcnemar_exact(first: list[bool], second: list[bool]) -> dict:
    """Exact McNemar on paired outcomes. Only the discordant pairs carry information."""
    from scipy.stats import binomtest

    first_only = sum(1 for a, b in zip(first, second) if a and not b)
    second_only = sum(1 for a, b in zip(first, second) if b and not a)
    discordant = first_only + second_only
    p_value = 1.0 if discordant == 0 else binomtest(
        min(first_only, second_only), discordant, 0.5, alternative="two-sided").pvalue
    return {
        "first_only": first_only,
        "second_only": second_only,
        "discordant": discordant,
        "p_value": p_value,
    }


def bootstrap_ci(first: list[bool], second: list[bool], iterations: int = 10000,
                 seed: int = 4242) -> dict:
    """Percentile bootstrap CI on the paired accuracy difference, resampling items."""
    rng = random.Random(seed)
    n = len(first)
    differences = []
    for _ in range(iterations):
        total_first = total_second = 0
        for _ in range(n):
            index = rng.randrange(n)
            total_first += first[index]
            total_second += second[index]
        differences.append((total_first - total_second) / n)
    differences.sort()
    return {
        "point": sum(first) / n - sum(second) / n,
        "lo": differences[int(0.025 * iterations)],
        "hi": differences[int(0.975 * iterations)],
        "iterations": iterations,
        "seed": seed,
    }


def compare(name_a: str, result_a: dict, name_b: str, result_b: dict) -> dict:
    """Pair by item id rather than by position, so a reordering cannot silently misalign."""
    by_id_b = {r["id"]: r for r in result_b["items"]}
    a = [r["correct"] for r in result_a["items"]]
    b = [by_id_b[r["id"]]["correct"] for r in result_a["items"]]
    return {
        "a": name_a,
        "b": name_b,
        "delta_points": 100 * (sum(a) - sum(b)) / len(a),
        "mcnemar": mcnemar_exact(a, b),
        "bootstrap": bootstrap_ci(a, b),
    }


def verdict(comparison_vs_base: dict, comparison_vs_single: dict) -> dict:
    """Apply the 5-point rule, then check the paired statistics do not contradict it.

    Expect the tripwire to fire on any gain under about 15 points, and that is not a
    flaw in it. At n=50 the bootstrap half-width is roughly 14 points, so a 5-to-10 point
    gain sits comfortably inside an interval that spans zero -- the point estimate is
    two or three questions better and the paired evidence genuinely cannot separate that
    from noise. The chosen handling is to stop and hand that back rather than pick a
    side, because "clears the threshold on a point estimate" and "the evidence supports
    it" are different claims and only the first one is true in that case.
    """
    passes = (comparison_vs_base["delta_points"] >= POINTS_TO_PASS
              and comparison_vs_single["delta_points"] >= POINTS_TO_PASS)
    if not passes:
        return {"proceed": False, "contradiction": False,
                "reason": "does not clear 5 points against both baselines"}

    for comparison in (comparison_vs_base, comparison_vs_single):
        interval = comparison["bootstrap"]
        if interval["lo"] <= 0 <= interval["hi"]:
            return {"proceed": False, "contradiction": True,
                    "reason": f"reads {comparison['delta_points']:+.1f} points versus "
                              f"{comparison['b']}, which clears the 5-point rule, but the "
                              f"paired bootstrap CI spans zero "
                              f"({100 * interval['lo']:+.1f}, {100 * interval['hi']:+.1f} "
                              f"points) and the exact McNemar p is "
                              f"{comparison['mcnemar']['p_value']:.3f}, so the gain and "
                              f"the evidence disagree"}
    return {"proceed": True, "contradiction": False,
            "reason": "clears 5 points against both baselines, with the paired evidence "
                      "agreeing"}


# --- smoke gate ---------------------------------------------------------------------

def run_smoke(harness, registry, matrices_by_id, maths: list[dict],
              max_new_tokens: int) -> bool:
    """Five questions through base, one adapter and one blend. Hard gate.

    The two assertions that matter are structural rather than numerical, because a broken
    adapter load produces a plausible-looking zero rather than an error:

      * base and adapter must differ somewhere. If a LoRA silently failed to attach, every
        completion would be identical and the run would quietly measure the base model
        three times.
      * the blend must differ from both singles. If the weighted sum collapsed to one
        adapter -- weights not applied, or applied to the wrong modules -- the blend would
        be a copy of a single and the experiment would be comparing an adapter with
        itself.

    This is the same failure the Ollama spike hit from a different direction: a loader
    that accepts an artifact and ignores it.
    """
    items = maths[:SMOKE_N]
    blendable = [a for a in registry["adapters"] if a.get("blendable")]
    if not blendable:
        log("smoke: no blendable adapters")
        return False

    smoke_blend = blendable[:2]
    results = {}
    for name in ("base", "single", "blend"):
        if name == "base":
            harness.clear()
        elif name == "single":
            entry = blendable[0]
            applied = harness.use_single(matrices_by_id[entry["id"]], scaling_for(entry))
            log(f"  [{name}] {entry['id']} attached to {applied} projections")
        else:
            applied = harness.use_blend(
                [matrices_by_id[a["id"]] for a in smoke_blend], [0.5, 0.5],
                scaling_for(smoke_blend[0]))
            log(f"  [{name}] {'+'.join(a['id'] for a in smoke_blend)} "
                f"attached to {applied} projections")
        results[name] = run_config(harness, items, max_new_tokens, label=("smoke-" + name))

    log("")
    for name, result in results.items():
        log(f"  {name:<7} acc={result['accuracy']:.2f}  "
            f"parsed={parsed_rate(result, items):.2f}  {result['seconds']:.1f}s")
    per_question = max(result["seconds"] for result in results.values()) / len(items)
    log(f"  slowest config {per_question:.1f}s per question")

    failures = []
    for name, result in results.items():
        if not all(record["completion"].strip() for record in result["items"]):
            failures.append(f"{name} produced an empty completion")
    if sum(parsed_rate(result, items) for result in results.values()) == 0:
        failures.append("no configuration parsed a single answer -- the extractor or the "
                        "chat template is wrong, not the models")

    def completions(name):
        return [record["completion"] for record in results[name]["items"]]

    if completions("base") == completions("single"):
        failures.append("base and single adapter produced identical output on all 5 "
                        "questions -- the adapter did not attach")
    if (completions("blend") == completions("single")
            or completions("blend") == completions("base")):
        failures.append("the blend produced identical output to a single configuration on "
                        "all 5 questions -- the weights were not applied")

    harness.clear()
    if failures:
        log("\nSMOKE FAILED:")
        for line in failures:
            log("  " + line)
        return False
    log("\nSMOKE PASSED")
    return True


# --- checkpointing ------------------------------------------------------------------
#
# The scored phase takes roughly ninety minutes and results.json is written only at the
# very end, so anything that kills the process -- a stray Ctrl-C, the laptop sleeping, the
# harness reaping a background task -- discards every configuration that had already run.
# That is not hypothetical: it happened once, and an hour of search and scoring went with
# it and left no results.json behind.
#
# So the run checkpoints as it goes, one cell at a time. The file lives under blends/,
# which .gitignore already excludes, because it is build state rather than evidence:
# results.json stays the single artifact every number traces back to, and a resumed run
# writes it exactly as an uninterrupted one would.
#
# The checkpoint carries a signature of the parameters that produced it. Reusing a search
# run under a different seed, budget or item count would silently splice two experiments
# together, so a mismatch discards the file rather than trusting it.

def checkpoint_path():
    return HERE / "blends" / "checkpoint.json"


def run_signature(args, maths, control, heldout, base_model):
    return {
        "base_model": base_model,
        "search_seed": args.search_seed,
        "search_random": args.search_random,
        "search_rounds": args.search_rounds,
        "search_step": args.search_step,
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "maths": len(maths), "control": len(control), "heldout": len(heldout),
    }


def load_checkpoint(signature, log):
    """The checkpointed search and config cells, or empty state if it does not apply."""
    path = checkpoint_path()
    if not path.exists():
        return {"search": None, "configs": {}}
    try:
        saved = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        log(f"  checkpoint unreadable ({type(error).__name__}: {error}); starting fresh")
        return {"search": None, "configs": {}}
    if saved.get("signature") != signature:
        log("  checkpoint was written for different settings; ignoring it rather than "
            "mixing two runs")
        return {"search": None, "configs": {}}
    return {"search": saved.get("search"), "configs": saved.get("configs", {})}


def save_checkpoint(signature, state):
    path = checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written under a temporary name and renamed into place, so a kill mid-write cannot
    # leave a truncated file that the next run would either refuse to trust or, worse,
    # parse as a valid checkpoint with cells missing.
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"signature": signature, **state}, indent=2) + "\n")
    temporary.replace(path)


# --- main ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="5 questions, all config kinds, hard gate")
    parser.add_argument("--limit", type=int, default=0, help="cap items per set")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--search-seed", type=int, default=20261010)
    parser.add_argument("--search-random", type=int, default=8,
                        help="random simplex candidates after the anchors")
    parser.add_argument("--search-rounds", type=int, default=2)
    parser.add_argument("--search-step", type=float, default=0.2)
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--resume", action="store_true",
                        help="reuse the checkpointed search and any configurations already "
                             "measured; a kill then costs the cell in flight, not the run")
    args = parser.parse_args()

    registry = load_registry()
    maths = load_items("questions_maths.json")
    control = load_items("questions_control.json")
    heldout = load_items("questions_heldout.json")
    if args.limit:
        maths, control, heldout = maths[:args.limit], control[:args.limit], heldout[:args.limit]

    log(f"base model   {registry['base_model']}")
    log(f"maths {len(maths)} | control {len(control)} | held out {len(heldout)}")

    # Pre-flight, before anything expensive. The wrapper that *applies* a blend and the
    # writer that *saves* one once held separate copies of the LoraHub sum, and the copies
    # drifted -- so the search spent fifty minutes scoring a blend that was not the blend
    # being recorded, and only the artifact check at the end caught it. This costs a
    # millisecond and fails in the first second instead.
    agreement = check_apply_matches_write()
    log(f"apply/write agreement {agreement}")
    if not (agreement["A_equal"] and agreement["B_equal"]):
        log("the wrapper and the artifact writer disagree on the blend; refusing to run")
        return 1

    log("\nloading adapters")
    matrices_by_id = {}
    unavailable = {}
    for entry in registry["adapters"]:
        try:
            matrices_by_id[entry["id"]] = load_adapter_matrices(entry)
            log(f"  {entry['id']:<15} {len(matrices_by_id[entry['id']])} modules  "
                f"{entry['size_mb']} MB  blendable={entry.get('blendable', False)}")
        except Exception as error:  # a fragile adapter must not take the run down
            unavailable[entry["id"]] = f"{type(error).__name__}: {error}"
            log(f"  {entry['id']:<15} UNAVAILABLE ({unavailable[entry['id']]})")

    log("\nloading base model")
    harness = Harness(registry["base_model"]).load()
    log(f"  {harness.device} / {harness.dtype}  loaded in {harness.load_seconds:.1f}s  "
        f"{harness.describe_active()}")

    if args.smoke:
        ok = run_smoke(harness, registry, matrices_by_id, maths, args.max_new_tokens)
        return 0 if ok else 1

    # --- checkpoint state ------------------------------------------------------------
    state = {"search": None, "configs": {}}
    signature = run_signature(args, maths, control, heldout, registry["base_model"])
    if args.resume:
        state = load_checkpoint(signature, log)
        log(f"  resumed: search={'reused' if state['search'] else 'not checkpointed'}, "
            f"cells={sorted(state['configs'])}")
    else:
        checkpoint_path().unlink(missing_ok=True)

    def save():
        save_checkpoint(signature, state)

    def cell(name, set_name, compute):
        """One configuration x set result: from the checkpoint, or freshly measured.

        Saved the moment it is measured rather than at the end, so the most a kill can now
        cost is the single cell in flight -- about eight minutes -- instead of the run.
        """
        existing = state["configs"].get(name, {}).get(set_name)
        if existing is not None:
            log(f"  reusing checkpointed {name}/{set_name}: {existing['accuracy']:.3f} "
                f"({existing['correct']}/{existing['n']})")
            return existing
        result = compute()
        state["configs"].setdefault(name, {})[set_name] = result
        save()
        return result

    blendable = [a for a in registry["adapters"]
                 if a.get("blendable") and a["id"] in matrices_by_id]
    singles = [a for a in registry["adapters"] if a["id"] in matrices_by_id]
    if len(blendable) < 2:
        log("fewer than two blendable adapters available; nothing to blend")
        return 1

    # --- weight search on the held-out slice only -----------------------------------
    if state["search"] is not None:
        # The search is deterministic given its seed and budget, so a checkpointed one is
        # the same measurement, not an approximation of it.
        search = state["search"]
        log(f"\nreusing the checkpointed weight search: {search['heldout_score']:.3f} "
            f"({int(round(search['heldout_score'] * search['heldout_n']))}/"
            f"{search['heldout_n']} on the held-out slice)")
    else:
        log(f"\nsearching blend weights on {len(heldout)} held-out items "
            f"(never on the {len(maths)} scored items)")
        started = time.perf_counter()
        search = search_weights(
            harness, [matrices_by_id[a["id"]] for a in blendable], [a["id"] for a in blendable],
            heldout, scaling_for(blendable[0]), seed=args.search_seed,
            random_candidates=args.search_random, coordinate_rounds=args.search_rounds,
            step=args.search_step, max_new_tokens=args.max_new_tokens, log=log)
        search["seconds"] = time.perf_counter() - started
        state["search"] = search
        save()
    weights = [search["weights"][a["id"]] for a in blendable]
    log(f"  chosen: {search['weights']}")
    if search["degenerate"]:
        log(f"  note: the search collapsed to a single adapter ({search['active'][0]}). "
            f"On {search['heldout_n']} items that is a real finding, not a failure -- it "
            f"means no combination beat the best single on the held-out slice.")

    # --- build and verify the artifact ----------------------------------------------
    log("\nbuilding the blend artifact")
    artifact_dir = HERE / "blends" / ("blend-" + "-".join(search["active"]))
    path, build_seconds, build_info = build_blend(
        [matrices_by_id[a["id"]] for a in blendable], weights, artifact_dir,
        registry["base_model"], blendable[0])
    log(f"  wrote {path.name} in {build_seconds * 1000:.0f} ms "
        f"({build_info['blend_bytes'] / 1e6:.1f} MB, {build_info['modules']} modules)")
    artifact_check = verify_artifact(
        path, [matrices_by_id[a["id"]] for a in blendable], weights, harness,
        scaling_for(blendable[0]))
    log(f"  verification: {artifact_check}")
    if not artifact_check.get("ok"):
        log("  the artifact does not reproduce the in-memory blend; stopping before the "
            "scored run so no number depends on which path produced it")
        return 1

    # --- the scored run --------------------------------------------------------------
    configs = state["configs"]

    log(f"\n(a) base on maths")
    harness.clear()
    configs.setdefault("base", {"kind": "base"})["maths"] = cell(
        "base", "maths",
        lambda: run_config(harness, maths, args.max_new_tokens, label="base"))

    for entry in singles:
        log(f"\n(b) {entry['id']} on maths")
        harness.use_single(matrices_by_id[entry["id"]], scaling_for(entry))
        config = configs.setdefault(entry["id"], {"kind": "single",
                                                  "domain": entry["domain"]})
        config["maths"] = cell(
            entry["id"], "maths",
            lambda e=entry: run_config(harness, maths, args.max_new_tokens, label=e["id"]))

    log(f"\n(c) blend on maths")
    harness.use_blend([matrices_by_id[a["id"]] for a in blendable], weights,
                      scaling_for(blendable[0]))
    configs.setdefault("blend", {"kind": "blend", "weights": search["weights"]})["maths"] = cell(
        "blend", "maths",
        lambda: run_config(harness, maths, args.max_new_tokens, label="blend"))

    # The control set is run for three configurations only. The question it answers is
    # whether the treatment cost general ability, and base / best-single / blend is
    # exactly the set needed for that; running all seven would triple the cost of the
    # run to answer a question nobody asked.
    best_single = max((e["id"] for e in singles), key=lambda i: configs[i]["maths"]["accuracy"])
    log(f"\n(b) best single on the 50 is {best_single} "
        f"({configs[best_single]['maths']['accuracy']:.3f})")
    for name in ("base", best_single, "blend"):
        log(f"\ncontrol set: {name}")
        if name == "base":
            harness.clear()
        elif name == "blend":
            harness.use_blend([matrices_by_id[a["id"]] for a in blendable], weights,
                              scaling_for(blendable[0]))
        else:
            entry = next(e for e in singles if e["id"] == name)
            harness.use_single(matrices_by_id[name], scaling_for(entry))
        configs[name]["control"] = cell(
            name, "control",
            lambda n=name: run_config(harness, control, args.max_new_tokens, label=n))
    harness.clear()

    # --- report ----------------------------------------------------------------------
    vs_base = compare("blend", configs["blend"]["maths"], "base", configs["base"]["maths"])
    vs_single = compare("blend", configs["blend"]["maths"], best_single,
                        configs[best_single]["maths"])
    decision = verdict(vs_base, vs_single)

    log("\n=== maths accuracy on the %d scored questions ===" % len(maths))
    for name, config in sorted(configs.items(), key=lambda kv: -kv[1]["maths"]["accuracy"]):
        marker = {"base": "(a)", "blend": "(c)"}.get(name, "(b)" if name == best_single else "   ")
        log(f"  {marker} {name:<15} {config['maths']['accuracy']:.3f}  "
            f"({config['maths']['correct']}/{config['maths']['n']})")
    log(f"\n  (c) - (a) = {vs_base['delta_points']:+.1f} points    "
        f"McNemar p={vs_base['mcnemar']['p_value']:.3f}    "
        f"CI [{vs_base['bootstrap']['lo']:+.3f}, {vs_base['bootstrap']['hi']:+.3f}]")
    log(f"  (c) - (b) = {vs_single['delta_points']:+.1f} points    "
        f"McNemar p={vs_single['mcnemar']['p_value']:.3f}    "
        f"CI [{vs_single['bootstrap']['lo']:+.3f}, {vs_single['bootstrap']['hi']:+.3f}]")
    log(f"  blend build {build_seconds * 1000:.0f} ms, "
        f"{build_info['blend_bytes'] / 1e6:.1f} MB")

    log("\n=== control set ===")
    for name in ("base", best_single, "blend"):
        result = configs[name]["control"]
        log(f"  {name:<15} {result['accuracy']:.3f}  "
            f"({result['correct']}/{result['n']})")

    log(f"\nverdict: {'PROCEED to Stage 2' if decision['proceed'] else 'STOP'}"
        f" -- {decision['reason']}")
    if decision["contradiction"]:
        log("\nThis is the case that comes back to you rather than resolving itself: the "
            "point estimate clears the threshold but the paired evidence does not "
            "support it. Nothing has been written to a PR.")

    payload = {
        "meta": {
            "base_model": registry["base_model"],
            "device": harness.device,
            "dtype": str(harness.dtype),
            "greedy": True,
            "max_new_tokens": args.max_new_tokens,
            "extractor": "last number in the completion / word-boundary phrase match",
            "tolerance": "half a unit in the item's own stated decimal place",
            "sets": {"maths": len(maths), "control": len(control), "heldout": len(heldout)},
            "loaded_seconds": harness.load_seconds,
            "search_seconds": search["seconds"],
            "build_seconds": build_seconds,
            "build_bytes": build_info["blend_bytes"],
            "notes": [
                "Every score is reproducible from this file: greedy decoding, seeded "
                "search. The timing fields are machine-dependent and are not.",
                "Adapter revisions are pinned in adapters.json; the numbers below belong "
                "to those exact bytes.",
            ],
        },
        "adapters": [
            {k: a[k] for k in ("id", "hf_repo", "revision", "domain", "size_mb") if k in a}
            | {"blendable": a.get("blendable", False),
               "scaling": scaling_for(a),
               "unavailable": unavailable.get(a["id"])}
            for a in registry["adapters"]
        ],
        "blend": search | {"artifact": str(path.relative_to(HERE)),
                           "artifact_check": artifact_check},
        "configs": configs,
        "comparison": {"best_single": best_single, "blend_vs_base": vs_base,
                       "blend_vs_best_single": vs_single},
        "verdict": decision | {"points_required": POINTS_TO_PASS},
    }
    (HERE / args.out).write_text(json.dumps(payload, indent=2) + "\n")
    log(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
