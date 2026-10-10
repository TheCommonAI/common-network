#!/usr/bin/env python3
"""Authors the 200-item replication set: fresh maths items, same generator, new seed.

Why a second set exists
-----------------------

The 50 scored questions produced a blend that beat the base model by 12 points and the
best single adapter by 8, with a paired bootstrap interval that still spanned zero. The
five-point rule said yes and the paired evidence said not proven, so the run stopped
rather than declaring a win. This set is the response to that: the *same* blend weights,
frozen, applied to 200 questions that neither the weight search nor the choice of single
adapter had ever seen.

What "fresh" means here, and what it does not
---------------------------------------------

Fresh means new numbers from the same ten families, from the same generator seeded
differently, with every prompt proven absent from the 50 scored and the 20 held-out items.
So this generalises the result across *instances* of a family. It is not a second domain,
and the write-up must not let 200 items imply that it is.

The weights were fitted on 18 held-out maths items drawn from these same families. That is
why this is a replication of "does the blend beat the single adapter", not an independent
test of "does blending transfer across domains".

Validation
----------

The checks that guard the original sets run here too, through the same functions and in the
same words: every answer re-derived from the item's own stored parameters, every parameter
proven load-bearing by perturbation, every parameter's value shown to appear in the
question text, positivity, and the unit-price-change bound. Self-consistency alone is not
enough — check_questions.py records the -21.20 item that passed every internal check and
was still an impossible question.

Run: python make_replication_set.py
Writes questions_maths_200.json. Exits non-zero with a list of failures.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from make_questions import MATHS_COMPUTE, build_maths, quantise
# The two substantive validators are imported rather than rewritten. They are the same
# functions that guard the scored sets, so a bug cannot be fixed in one and left in the
# other -- the failure mode that made the blend writer and the blend applier disagree.
from check_questions import perturbations, rendered

HERE = Path(__file__).parent
REPLICATION_N = 200

# Far from BASE_SEED (20261010) and SEED_HELDOUT (771). build_maths derives one RNG per
# family as Random(seed + i), so a seed close to an existing one would re-walk that set's
# stream for some families. The prompt-level `seen` guard would still prevent duplicates,
# but only by rejection -- burning attempts and hiding which items are genuinely new.
REPLICATION_SEED = 20261010 + 10000

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, got, want) -> None:
    global CHECKS
    CHECKS += 1
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")


def check_true(name: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(f"{name}{': ' + detail if detail else ''}")


def load(name: str) -> list[dict]:
    return json.loads((HERE / name).read_text())["items"]


def main() -> int:
    maths = load("questions_maths.json")
    control = load("questions_control.json")
    heldout = load("questions_heldout.json")

    # Seed `seen` from the committed prompts rather than from a regeneration of them. The
    # files on disk are what was scored; regenerating would prove disjointness against
    # what the generator *would* produce, which is a different claim.
    seen = {i["question"] for i in maths + control + heldout}
    prior = set(seen)

    fresh = build_maths(REPLICATION_N, REPLICATION_SEED, seen)

    # --- counts and ids ---------------------------------------------------------------
    check("replication count", len(fresh), REPLICATION_N)
    check("ids unique", len({i["id"] for i in fresh}), REPLICATION_N)
    prompts = [i["question"] for i in fresh]
    check("prompts unique within the set", len(set(prompts)), len(prompts))
    if len(set(prompts)) != len(prompts):
        repeated = [q for q, n in Counter(prompts).items() if n > 1]
        FAILURES.append(f"  repeated: {repeated[:3]}")

    # --- disjointness: the whole point of the set -------------------------------------
    overlap = [q for q in prompts if q in prior]
    check("no prompt shared with the 50 scored or the 20 held out", len(overlap), 0)
    if overlap:
        FAILURES.append(f"  overlaps: {overlap[:3]}")

    # --- re-derivation, load-bearing parameters, stated numbers ----------------------
    for item in fresh:
        family, params = item["family"], item["params"]
        expected = quantise(MATHS_COMPUTE[family](params), item["places"])
        check(f"{item['id']} ({family}) re-derives", item["answer"], expected)

        for text in rendered(params, family):
            check_true(f"{item['id']} question states {text!r}",
                       text in item["question"], item["question"][:90])

        for key in params:
            moved = False
            for trial in perturbations(params, key):
                try:
                    if quantise(MATHS_COMPUTE[family](trial), item["places"]) != item["answer"]:
                        moved = True
                        break
                except ZeroDivisionError:
                    continue
            check_true(f"{item['id']} answer depends on {key!r}", moved,
                       f"perturbing {key} never changed the answer")

        try:
            float(item["answer"])
        except ValueError:
            FAILURES.append(f"{item['id']} answer is not numeric: {item['answer']!r}")

    # --- positivity, and the note bound -----------------------------------------------
    for item in fresh:
        value = float(item["answer"])
        check_true(f"{item['id']} ({item['family']}) answer is positive",
                   value > 0, item["answer"])
        if item["family"] == "unit_price_change":
            note_pounds = item["params"]["paid_cents"] / 100
            check_true(f"{item['id']} change is under the note",
                       value < note_pounds, f"{item['answer']} vs £{note_pounds:g}")

    # --- family coverage ---------------------------------------------------------------
    spread = Counter(i["family"] for i in fresh)
    check("families covered", len(spread), 10)
    check_true("families even", max(spread.values()) - min(spread.values()) <= 1,
               str(dict(spread)))

    print(f"{CHECKS} checks run")
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILED:")
        for line in FAILURES[:40]:
            print("  " + line)
        if len(FAILURES) > 40:
            print(f"  ... and {len(FAILURES) - 40} more")
        return 1

    out = HERE / "questions_maths_200.json"
    out.write_text(json.dumps({
        "_note": "Replication set. Fresh instances from the same ten families as "
                 "questions_maths.json, generated with a seed far from the scored and "
                 "held-out seeds, and verified disjoint from both. Used with frozen blend "
                 "weights -- no search is run against it.",
        "generator": "make_replication_set.py",
        "replication_seed": REPLICATION_SEED,
        "items": fresh,
    }, indent=2, ensure_ascii=False) + "\n")

    print("all passed")
    print(f"  {len(fresh)} items, {len(spread)} families, "
          f"{sorted(spread.values())[0]}-{sorted(spread.values())[-1]} per family")
    print(f"  seed {REPLICATION_SEED} (scored {20261010}, held out 771)")
    print(f"  wrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
