#!/usr/bin/env python3
"""Verifies the generated question sets before anything is scored on them.

The point of generating the sets rather than quoting them was that a question and its
answer should not be able to disagree. This script is what makes that claim checkable
rather than a promise, and it is run before the evaluation rather than after.

What it asserts, and why each one is not a tautology:

  1. Counts and ids -- 50 scored per set, 20 held out. Catches a generator that silently
     produced 49 because a collision loop gave up.

  2. No duplicate prompts within or across any set. A duplicate would double-weight one
     item in a 50-item accuracy and quietly move the headline number.

  3. Every maths answer re-derives from the item's own stored parameters. This is a real
     check: the answer was computed when the item was made, and is recomputed here from
     the parameters by the same public function. A wording change that stopped matching
     the parameters would show up as a mismatch the moment the two drifted apart.

  4. Every maths parameter is load-bearing. For each parameter the script perturbs it and
     requires the answer to move. Recomputation alone would not catch a generator whose
     question mentions a number its answer ignores -- which is exactly the bug that makes
     a question unanswerable, and it would look like a model failure rather than a
     question failure.

  5. Every parameter's value appears in the question text, in the form the family renders
     it. Together with (3) and (4) this closes the loop: the question states the numbers,
     and the answer is a function of exactly those numbers.

  6. Every control answer equals a fresh table lookup for its key, and every control key
     is distinct across a set -- a repeated key would measure the same fact twice.

  7. No held-out prompt appears in either scored set. If it did, the blend weights would
     have been fitted on scored items and every number in the README would be inflated.

  8. The scored sets' family distribution is even, so one family cannot dominate accuracy.

  9. Every maths answer is a positive quantity. This is the only check here that is not
     about internal consistency, and it exists because internal consistency is not
     enough. The first generated maths set contained "6 markers at £5.20 each, paid with
     a £10 note -- how much change?" with the answer -£21.20. Every check above passed on
     that item: the answer re-derived, both parameters were load-bearing, both appeared
     in the text. It was simply an impossible question, because the generator had mixed
     pounds and pence when choosing the note. Re-derivation cannot catch a wrong premise.
     A blunt "the answer is not a negative quantity of money" can.

Run: python check_questions.py
Exits non-zero with a list of failures.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

from make_questions import (
    CAPITALS,
    COMPOUNDS,
    CONTINENTS,
    CURRENCIES,
    ELEMENT_NUMBER,
    ELEMENT_SYMBOL,
    MATHS_COMPUTE,
    PLANET_ORDER,
    SI_UNITS,
    quantise,
)

HERE = Path(__file__).parent
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


# --- 5. how each family renders its parameters into the question -------------------
#
# Written out per family rather than inferred, because the point is to state what the
# question is supposed to contain and then check the question actually contains it. An
# inferred version would agree with whatever the generator happened to do.

def rendered(p: dict, family: str) -> list[str]:
    if family == "unit_price_change":
        return [f"£{quantise(Fraction(p['cents'], 100), 2)}", str(p["n"]),
                f"£{p['paid_cents'] // 100}"]
    if family == "percent_discount_tax":
        return [str(p["pounds"]), f"{p['discount_pct']}%", f"{p['tax_pct']}%"]
    if family == "rate_time_distance":
        return [str(p["speed"]), str(p["hours"]), str(p["minutes"])]
    if family == "average":
        return [str(s) for s in p["scores"]]
    if family == "ratio_sharing":
        return [str(p["total"]), f"{p['parts'][0]}:{p['parts'][1]}:{p['parts'][2]}"]
    if family == "compound_growth":
        return [str(p["principal"]), f"{p['rate_pct']}%", str(p["years"])]
    if family == "work_rate":
        return [str(p["a"]), str(p["b"])]
    if family == "pct_increase_then_decrease":
        return [str(p["value"]), f"{p['up_pct']}%", f"{p['down_pct']}%"]
    if family == "mixture":
        return [str(p["l1"]), f"{p['c1']}%", str(p["l2"]), f"{p['c2']}%"]
    if family == "series_sum":
        return [str(p["n"]), str(p["a"]), str(p["d"])]
    raise KeyError(family)


def perturbations(params: dict, key: str):
    """Yield copies of params with one entry nudged, by amounts large enough to matter.

    The amounts have to be generous, and that is the point rather than a fudge. Several
    families quantise to a whole number, so nudging a parameter by one can leave the
    rounded answer identical and the check would report a load-bearing parameter as
    inert. A ratio-sharing total, for instance, only moves the answer once the nudge is a
    noticeable fraction of the sum being shared. A list parameter is nudged one element
    at a time, because adding a constant to every part of a ratio barely changes it.
    """
    value = params[key]
    if isinstance(value, list):
        for index in range(len(value)):
            for amount in (1, 2, 3, 5, -1, -2, 7, 10, 25, -11):
                copy = json.loads(json.dumps(params))
                if copy[key][index] + amount <= 0:
                    continue
                copy[key][index] += amount
                yield copy
        for index in range(len(value)):
            copy = json.loads(json.dumps(params))
            copy[key][index] *= 2
            yield copy
    elif isinstance(value, int):
        for amount in (1, 2, 3, 5, -1, -2, 7, 10, 13, 25, -11, value, -(value // 2)):
            if amount == 0 or value + amount <= 0:
                continue
            copy = json.loads(json.dumps(params))
            copy[key] = value + amount
            yield copy


# --- the tables the control answers are looked up in --------------------------------

def control_rendered(family: str, key: str) -> str:
    """What the question is supposed to say about its key.

    Not always the key itself: the planet question asks for an ordinal ("which planet is
    number 4 counting outward from the Sun?"), not for the planet's name, so checking
    that the name appears would fail on a perfectly good item. Each family states what it
    states.
    """
    if family == "planet_order":
        return str(PLANET_ORDER.index(key) + 1)
    return key


CONTROL_TABLES = {
    "element_symbol": lambda k: ELEMENT_SYMBOL[k],
    "element_atomic_number": lambda k: str(ELEMENT_NUMBER[k]),
    "country_capital": lambda k: CAPITALS[k],
    "country_currency": lambda k: CURRENCIES[k],
    "country_continent": lambda k: CONTINENTS[k],
    "si_unit": lambda k: SI_UNITS[k],
    "planet_order": lambda k: PLANET_ORDER[PLANET_ORDER.index(k)],
    "compound_formula": lambda k: COMPOUNDS[k],
}


def main() -> int:
    maths = load("questions_maths.json")
    control = load("questions_control.json")
    heldout = load("questions_heldout.json")
    held_maths = [i for i in heldout if i["answer_type"] == "number"]
    held_control = [i for i in heldout if i["answer_type"] == "text"]

    # --- 1. counts -----------------------------------------------------------------
    check("maths count", len(maths), 50)
    check("control count", len(control), 50)
    check("heldout total", len(heldout), 20)
    check("heldout maths", len(held_maths), 18)
    check("heldout control", len(held_control), 2)
    check("maths ids unique", len({i["id"] for i in maths}), 50)
    check("control ids unique", len({i["id"] for i in control}), 50)
    check("heldout ids unique", len({i["id"] for i in heldout}), 20)
    for item in maths + control + heldout:
        check_true(f"{item['id']} has a question", bool(item["question"].strip()))
        check_true(f"{item['id']} has an answer", bool(str(item["answer"]).strip()))

    # --- 2. no duplicate prompts ---------------------------------------------------
    for label, items in [("maths", maths), ("control", control), ("heldout", heldout)]:
        prompts = [i["question"] for i in items]
        check(f"{label} prompts unique", len(set(prompts)), len(prompts))
        if len(set(prompts)) != len(prompts):
            repeated = [q for q, n in Counter(prompts).items() if n > 1]
            FAILURES.append(f"  repeated in {label}: {repeated[:3]}")
    all_prompts = [i["question"] for i in maths + control + heldout]
    check("prompts unique across all sets", len(set(all_prompts)), len(all_prompts))

    # --- 3/4/5. maths answers re-derive, every parameter is load-bearing, and the
    #            question states the numbers the answer is a function of ---------------
    for item in maths + held_maths:
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

    # --- 6. control answers are table lookups, on distinct keys ---------------------
    for items, label in [(control, "control"), (held_control, "held control")]:
        # Distinctness is per (family, key), not per key. The capital, currency and
        # continent families all index by country name, so the same country appearing
        # once in each is three different facts about it, not a fact measured twice.
        # What must not repeat is the same family asking about the same key twice.
        pairs = [(i["family"], i["params"]["key"]) for i in items]
        check(f"{label} (family, key) pairs distinct", len(set(pairs)), len(pairs))
        for item in items:
            family, key = item["family"], item["params"]["key"]
            check(f"{item['id']} ({family}/{key}) matches table",
                  str(item["answer"]), str(CONTROL_TABLES[family](key)))
            stated = control_rendered(family, key)
            check_true(f"{item['id']} question states {stated!r}",
                       stated in item["question"], item["question"][:90])

    # --- 7. held-out items are genuinely absent from the scored sets ----------------
    scored = {i["question"] for i in maths + control}
    for item in heldout:
        check_true(f"held-out {item['id']} not in a scored set",
                   item["question"] not in scored)
        check_true(f"held-out {item['id']} flagged", item["held_out"] is True)
    for item in maths + control:
        check_true(f"{item['id']} not flagged held-out", item["held_out"] is False)

    # --- 8. even family distribution -------------------------------------------------
    maths_spread = Counter(i["family"] for i in maths)
    check("maths families covered", len(maths_spread), 10)
    check_true("maths families even",
               max(maths_spread.values()) - min(maths_spread.values()) <= 1,
               str(dict(maths_spread)))
    control_spread = Counter(i["family"] for i in control)
    check("control families covered", len(control_spread), 8)
    check_true("control families even",
               max(control_spread.values()) - min(control_spread.values()) <= 1,
               str(dict(control_spread)))

    # --- 9. every maths answer is a positive quantity --------------------------------
    #
    # All ten families ask "how much/many/far", and every one of those is a positive
    # quantity. An answer at or below zero means the item's premise is broken even if its
    # arithmetic is not -- see the docstring. The unit_price_change family gets a second,
    # tighter bound: the change must be strictly less than the note handed over, or the
    # shopper is being given back more money than they paid.
    for item in maths + held_maths:
        value = float(item["answer"])
        check_true(f"{item['id']} ({item['family']}) answer is positive",
                   value > 0, item["answer"])
        if item["family"] == "unit_price_change":
            note_pounds = item["params"]["paid_cents"] / 100
            check_true(f"{item['id']} change is under the note",
                       value < note_pounds, f"{item['answer']} vs £{note_pounds:g}")

    # --- report ---------------------------------------------------------------------
    print(f"{CHECKS} checks run")
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILED:")
        for line in FAILURES[:40]:
            print("  " + line)
        if len(FAILURES) > 40:
            print(f"  ... and {len(FAILURES) - 40} more")
        return 1
    print("all passed")
    print(f"  maths   50 items, {len(maths_spread)} families, "
          f"{sorted(maths_spread.values())[0]}-{sorted(maths_spread.values())[-1]} per family")
    print(f"  control 50 items, {len(control_spread)} families")
    print(f"  heldout 20 items (18 maths, 2 control), never scored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
