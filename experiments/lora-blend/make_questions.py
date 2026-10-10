#!/usr/bin/env python3
"""Authors the Stage 1 question sets, computing every answer rather than asserting it.

Why this exists at all: the experiment asks whether a *blended* adapter beats the base
model and the best single adapter on a domain that is genuinely in between the domains
the blend was built from. If the ground truth is wrong, or if the base model has simply
memorised the questions, the comparison measures nothing. So:

  * The maths set is generated. Numbers come from a seeded RNG and every answer is the
    generator's own arithmetic in exact rational form, quantised at a stated precision.
    A question and its answer therefore *cannot* disagree -- there is no second place
    where a human typed the answer.

  * The control set is rendered from data tables encoded below. Same property: the
    answer is the table lookup the question was built from.

  * Each item carries its generator's parameters, so `check_questions.py` can re-derive
    the answer from the parameters and separately confirm the parameters actually
    appear in the question text. That is a real check rather than a tautology: it would
    catch a question whose wording drifted away from the numbers it was built from.

  * Twenty items are held out -- eighteen maths, two control -- and are written to
    questions_heldout.json. They are used only for the blend-weight search and are never
    scored. check_questions.py asserts they share no prompt with either scored set.
    Eight would have been nearly a coin flip as a search objective: accuracy on eight
    items moves in 12.5-point steps, so almost every weight vector ties and the search
    degenerates to picking whichever candidate came first. Eighteen gives 5.6-point
    steps, and costs nothing to generate.

  * The maths families below ARE the GSM8K template space -- unit price, ratios, rates,
    percentages. That is deliberate (word problems needing programmatic decomposition are
    the point of the maths-plus-code pairing) but it is also this experiment's sharpest
    threat: a maths adapter trained on GSM8K would score well here for the wrong reason,
    and a blend containing it would inherit that. Fresh numbers do not remove template
    memorisation. probe.py exists to measure it -- it scores every config on a GSM8K
    test subsample and prints the result next to the same config's score on this set, so
    a memorised adapter is visible rather than assumed away.

The tables in the control set are the one place human knowledge enters, and they are
plainly checkable. They are listed as data rather than prose precisely so they can be
read at a glance and audited; nothing about them is computed.

Run: python make_questions.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction
from pathlib import Path

BASE_SEED = 20261010
SEED_HELDOUT = 771

# ---------------------------------------------------------------------------
# Exact arithmetic
# ---------------------------------------------------------------------------


def quantise(value: Fraction, places: int) -> str:
    """Render an exact rational as a decimal string, rounding half up.

    Doing this on a Fraction rather than a float means the stored answer is the one
    and only correct rendering of the generator's arithmetic at the stated precision --
    there is no representable-but-wrong neighbour for a float to land on.
    """
    scale = 10 ** places
    scaled = value * scale
    n, d = scaled.numerator, scaled.denominator
    whole = (2 * n + d) // (2 * d)  # floor(n/d + 1/2), for positive n
    if places == 0:
        return str(whole)
    sign = "-" if whole < 0 else ""
    whole = abs(whole)
    return f"{sign}{whole // scale}.{whole % scale:0{places}d}"


# ---------------------------------------------------------------------------
# Maths set -- ten word-problem families, answers computed
# ---------------------------------------------------------------------------

NAMES = ["Alex", "Priya", "Sam", "Mei", "Omar", "Lena", "Tom", "Aisha", "Ben", "Nina"]


def gen_unit_price_change(rng: random.Random) -> tuple[str, dict]:
    """Buy n items at a 2dp unit price, pay with a note, work out the change.

    Everything here is in pence, and deliberately so. The first version of this function
    mixed pounds and pence when choosing the note, which produced items like "6 markers
    at £5.20 each, paid with a £10 note, change -£21.20". The arithmetic was internally
    consistent, so the re-derivation check passed happily: the question was simply
    impossible. Fixing the units is the fix; the positivity check in check_questions.py
    is there so a future rewording cannot reintroduce it quietly.

    The note is chosen first and the unit price is then bounded by it, so the total is
    always strictly below the note handed over -- and the change is always positive.
    """
    note_cents = rng.choice([500, 1000, 2000, 5000])
    n = rng.randrange(2, 12)
    hi = ((note_cents - 5) // n) // 5 * 5
    if hi < 25:  # note too small for this many items; the caller re-rolls
        hi = 25
    cents = rng.randrange(25, hi + 1, 5)
    price = Fraction(cents, 100)
    name = rng.choice(NAMES)
    item = rng.choice(["notebooks", "pens", "mugs", "pencils", "folders", "markers"])
    q = (
        f"A shop sells {item} at £{quantise(price, 2)} each. {name} buys {n} of them "
        f"and pays with a £{note_cents // 100} note. How much change, in pounds, does "
        f"{name} receive?"
    )
    return q, {"cents": cents, "n": n, "paid_cents": note_cents}


def compute_unit_price_change(p: dict) -> Fraction:
    return Fraction(p["paid_cents"], 100) - Fraction(p["cents"], 100) * p["n"]


def gen_percent_discount_tax(rng: random.Random) -> tuple[str, dict]:
    """A percentage discount followed by a percentage tax on the discounted price."""
    pounds = rng.randrange(40, 400)
    p = Fraction(pounds)
    d = rng.choice([10, 12, 15, 20, 25, 30])
    tax = rng.choice([5, 10, 15, 20])
    q = (
        f"A jacket is priced at £{pounds}. In a sale it is discounted by {d}%. "
        f"{tax}% VAT is then added to the discounted price. "
        f"What is the final price, in pounds?"
    )
    return q, {"pounds": pounds, "discount_pct": d, "tax_pct": tax}


def compute_percent_discount_tax(p: dict) -> Fraction:
    discounted = Fraction(p["pounds"]) * (100 - p["discount_pct"]) / 100
    return discounted * (100 + p["tax_pct"]) / 100


def gen_rate_time_distance(rng: random.Random) -> tuple[str, dict]:
    """Constant speed for a whole number of hours plus a number of minutes."""
    speed = rng.randrange(35, 130, 5)
    hours = rng.randrange(1, 7)
    minutes = rng.choice([5, 10, 15, 20, 25, 30, 40, 45, 50])
    hour_word = "hour" if hours == 1 else "hours"
    q = (
        f"A train travels at a steady {speed} km/h for {hours} {hour_word} and "
        f"{minutes} minutes. How far does it travel, in kilometres?"
    )
    return q, {"speed": speed, "hours": hours, "minutes": minutes}


def compute_rate_time_distance(p: dict) -> Fraction:
    return Fraction(p["speed"]) * (Fraction(p["hours"]) + Fraction(p["minutes"], 60))


def gen_average(rng: random.Random) -> tuple[str, dict]:
    """Average of five scores, with the scores chosen so the mean is a whole number."""
    base = rng.randrange(40, 90)
    offsets = [rng.randrange(-12, 13) for _ in range(5)]
    drift = sum(offsets) % 5
    offsets[0] -= drift  # make the total a multiple of five, so the mean is exact
    scores = [base + o for o in offsets]
    name = rng.choice(NAMES)
    q = (
        f"{name} scores {scores[0]}, {scores[1]}, {scores[2]}, {scores[3]} and "
        f"{scores[4]} in five tests. What is the average score?"
    )
    return q, {"scores": scores}


def compute_average(p: dict) -> Fraction:
    return Fraction(sum(p["scores"]), len(p["scores"]))


def gen_ratio_sharing(rng: random.Random) -> tuple[str, dict]:
    """Split a total in a three-part ratio; the total is a whole multiple of the parts."""
    parts = [rng.randrange(1, 8) for _ in range(3)]
    if len(set(parts)) < 3:
        parts[2] = parts[0] + parts[1] + 1
    unit = rng.randrange(5, 40)
    total = sum(parts) * unit
    q = (
        f"£{total} is shared between three people in the ratio "
        f"{parts[0]}:{parts[1]}:{parts[2]}. How much, in pounds, does the first person "
        f"receive?"
    )
    return q, {"parts": parts, "total": total}


def compute_ratio_sharing(p: dict) -> Fraction:
    return Fraction(p["total"]) * p["parts"][0] / sum(p["parts"])


def gen_compound_growth(rng: random.Random) -> tuple[str, dict]:
    """Compound interest over a small number of years."""
    principal = rng.randrange(500, 9000, 100)
    rate = rng.choice([2, 3, 4, 5, 6, 8])
    years = rng.randrange(2, 5)
    q = (
        f"A sum of £{principal} is invested at {rate}% compound interest per year. "
        f"What is its value after {years} years, in pounds?"
    )
    return q, {"principal": principal, "rate_pct": rate, "years": years}


def compute_compound_growth(p: dict) -> Fraction:
    value = Fraction(p["principal"])
    for _ in range(p["years"]):
        value = value * (100 + p["rate_pct"]) / 100
    return value


def gen_work_rate(rng: random.Random) -> tuple[str, dict]:
    """Two machines working together: the classic reciprocal-sum rate problem."""
    a = rng.randrange(3, 15)
    b = rng.randrange(3, 15)
    q = (
        f"One machine can complete a job in {a} hours, and a second machine can "
        f"complete the same job in {b} hours. Working together, how long do they take? "
        f"Give your answer in hours."
    )
    return q, {"a": a, "b": b}


def compute_work_rate(p: dict) -> Fraction:
    return Fraction(p["a"] * p["b"], p["a"] + p["b"])


def gen_pct_increase_then_decrease(rng: random.Random) -> tuple[str, dict]:
    """The classic non-cancelling increase-then-decrease, which is not a no-op."""
    value = rng.randrange(200, 9000, 50)
    up = rng.choice([10, 15, 20, 25, 30, 40])
    down = rng.choice([10, 15, 20, 25, 30, 40])
    q = (
        f"A population of {value} increases by {up}% in one year and then decreases "
        f"by {down}% the next year. What is the population after the two years? "
        f"Give your answer to the nearest whole number."
    )
    return q, {"value": value, "up_pct": up, "down_pct": down}


def compute_pct_increase_then_decrease(p: dict) -> Fraction:
    return Fraction(p["value"]) * (100 + p["up_pct"]) / 100 * (100 - p["down_pct"]) / 100


def gen_mixture(rng: random.Random) -> tuple[str, dict]:
    """Weighted-average concentration of two mixed solutions."""
    l1 = rng.randrange(2, 20)
    l2 = rng.randrange(2, 20)
    c1 = rng.randrange(5, 45, 5)
    c2 = rng.randrange(50, 95, 5)
    q = (
        f"{l1} litres of a solution that is {c1}% salt is mixed with {l2} litres of a "
        f"solution that is {c2}% salt. What percentage of the resulting mixture is salt? "
        f"Give your answer to 2 decimal places."
    )
    return q, {"l1": l1, "l2": l2, "c1": c1, "c2": c2}


def compute_mixture(p: dict) -> Fraction:
    return Fraction(
        p["l1"] * p["c1"] + p["l2"] * p["c2"], p["l1"] + p["l2"]
    )


def gen_series_sum(rng: random.Random) -> tuple[str, dict]:
    """Sum of an arithmetic series, from the closed form."""
    a = rng.randrange(2, 30)
    d = rng.randrange(2, 12)
    n = rng.randrange(8, 40)
    q = (
        f"What is the sum of the first {n} terms of an arithmetic sequence whose "
        f"first term is {a} and whose common difference is {d}?"
    )
    return q, {"a": a, "d": d, "n": n}


def compute_series_sum(p: dict) -> Fraction:
    return Fraction(p["n"] * (2 * p["a"] + (p["n"] - 1) * p["d"]), 2)


MATHS_FAMILIES = [
    ("unit_price_change", gen_unit_price_change, compute_unit_price_change, 2),
    ("percent_discount_tax", gen_percent_discount_tax, compute_percent_discount_tax, 2),
    ("rate_time_distance", gen_rate_time_distance, compute_rate_time_distance, 2),
    ("average", gen_average, compute_average, 0),
    ("ratio_sharing", gen_ratio_sharing, compute_ratio_sharing, 0),
    ("compound_growth", gen_compound_growth, compute_compound_growth, 2),
    ("work_rate", gen_work_rate, compute_work_rate, 2),
    ("pct_increase_then_decrease", gen_pct_increase_then_decrease,
     compute_pct_increase_then_decrease, 0),
    ("mixture", gen_mixture, compute_mixture, 2),
    ("series_sum", gen_series_sum, compute_series_sum, 0),
]

MATHS_PLACES = {name: places for name, _, _, places in MATHS_FAMILIES}
MATHS_COMPUTE = {name: fn for name, _, fn, _ in MATHS_FAMILIES}


# ---------------------------------------------------------------------------
# Control set -- rendered from data tables, answers looked up
# ---------------------------------------------------------------------------

ELEMENTS = [
    ("Hydrogen", "H", 1), ("Helium", "He", 2), ("Lithium", "Li", 3),
    ("Beryllium", "Be", 4), ("Boron", "B", 5), ("Carbon", "C", 6),
    ("Nitrogen", "N", 7), ("Oxygen", "O", 8), ("Fluorine", "F", 9),
    ("Neon", "Ne", 10), ("Sodium", "Na", 11), ("Magnesium", "Mg", 12),
    ("Aluminium", "Al", 13), ("Silicon", "Si", 14), ("Phosphorus", "P", 15),
    ("Sulfur", "S", 16), ("Chlorine", "Cl", 17), ("Argon", "Ar", 18),
    ("Potassium", "K", 19), ("Calcium", "Ca", 20), ("Scandium", "Sc", 21),
    ("Titanium", "Ti", 22), ("Vanadium", "V", 23), ("Chromium", "Cr", 24),
    ("Manganese", "Mn", 25), ("Iron", "Fe", 26), ("Cobalt", "Co", 27),
    ("Nickel", "Ni", 28), ("Copper", "Cu", 29), ("Zinc", "Zn", 30),
    ("Gallium", "Ga", 31), ("Germanium", "Ge", 32), ("Arsenic", "As", 33),
    ("Selenium", "Se", 34), ("Bromine", "Br", 35), ("Krypton", "Kr", 36),
    ("Rubidium", "Rb", 37), ("Strontium", "Sr", 38), ("Yttrium", "Y", 39),
    ("Zirconium", "Zr", 40), ("Niobium", "Nb", 41), ("Molybdenum", "Mo", 42),
    ("Technetium", "Tc", 43), ("Ruthenium", "Ru", 44), ("Rhodium", "Rh", 45),
    ("Palladium", "Pd", 46), ("Silver", "Ag", 47), ("Cadmium", "Cd", 48),
    ("Indium", "In", 49), ("Tin", "Sn", 50), ("Antimony", "Sb", 51),
    ("Tellurium", "Te", 52), ("Iodine", "I", 53), ("Xenon", "Xe", 54),
    ("Caesium", "Cs", 55), ("Barium", "Ba", 56), ("Lanthanum", "La", 57),
    ("Cerium", "Ce", 58), ("Tungsten", "W", 74), ("Platinum", "Pt", 78),
    ("Gold", "Au", 79), ("Mercury", "Hg", 80), ("Thallium", "Tl", 81),
    ("Lead", "Pb", 82), ("Bismuth", "Bi", 83), ("Radon", "Rn", 86),
    ("Radium", "Ra", 88), ("Thorium", "Th", 90), ("Uranium", "U", 92),
]

ELEMENT_SYMBOL = {name: sym for name, sym, _ in ELEMENTS}
ELEMENT_NUMBER = {name: num for name, _, num in ELEMENTS}

CAPITALS = {
    "France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid",
    "Portugal": "Lisbon", "Belgium": "Brussels", "Austria": "Vienna",
    "Switzerland": "Bern", "Sweden": "Stockholm", "Norway": "Oslo",
    "Denmark": "Copenhagen", "Finland": "Helsinki", "Poland": "Warsaw",
    "Czechia": "Prague", "Hungary": "Budapest", "Greece": "Athens",
    "Ireland": "Dublin", "Iceland": "Reykjavik", "Canada": "Ottawa",
    "Mexico": "Mexico City", "Brazil": "Brasilia", "Argentina": "Buenos Aires",
    "Chile": "Santiago", "Peru": "Lima", "Colombia": "Bogota", "Japan": "Tokyo",
    "China": "Beijing", "India": "New Delhi", "South Korea": "Seoul",
    "Thailand": "Bangkok", "Vietnam": "Hanoi", "Indonesia": "Jakarta",
    "Malaysia": "Kuala Lumpur", "Philippines": "Manila", "Australia": "Canberra",
    "New Zealand": "Wellington", "Egypt": "Cairo", "Morocco": "Rabat",
    "Kenya": "Nairobi", "Nigeria": "Abuja", "Ethiopia": "Addis Ababa",
    "Ghana": "Accra", "Turkey": "Ankara", "Saudi Arabia": "Riyadh",
    "Iran": "Tehran", "Iraq": "Baghdad", "Russia": "Moscow", "Ukraine": "Kyiv",
}
# Answers whose conventional spelling has more than one accepted form. Without this,
# a correct reply would be marked wrong for writing "Bogotá" or "Kyiv" differently from
# the table, and the control set would measure spelling rather than knowledge.
CAPITAL_ALIASES = {
    "Bogota": ["Bogotá"], "Kyiv": ["Kiev"], "Brasilia": ["Brasília"],
    "Reykjavik": ["Reykjavík"], "New Delhi": ["Delhi"], "Mexico City": ["Ciudad de Mexico"],
}

CURRENCIES = {
    "Japan": "yen", "China": "renminbi", "India": "Indian rupee",
    "United Kingdom": "pound sterling", "United States": "United States dollar",
    "France": "euro", "Germany": "euro", "Russia": "Russian ruble",
    "Brazil": "Brazilian real", "Mexico": "Mexican peso", "Switzerland": "Swiss franc",
    "Sweden": "Swedish krona", "Poland": "Polish zloty", "South Korea": "South Korean won",
    "Thailand": "Thai baht", "Vietnam": "Vietnamese dong", "Turkey": "Turkish lira",
    "Nigeria": "Nigerian naira", "Kenya": "Kenyan shilling", "Canada": "Canadian dollar",
    "Australia": "Australian dollar", "Norway": "Norwegian krone",
    "Denmark": "Danish krone", "Czechia": "Czech koruna", "Hungary": "Hungarian forint",
    "Iceland": "Icelandic krona", "Egypt": "Egyptian pound",
    "South Africa": "South African rand",
}
CURRENCY_ALIASES = {
    "renminbi": ["yuan", "RMB"], "pound sterling": ["pound", "sterling", "GBP"],
    "United States dollar": ["dollar", "US dollar", "USD"],
    "Russian ruble": ["ruble", "rouble", "RUB"],
    "Polish zloty": ["zloty", "złoty", "PLN"],
    "Vietnamese dong": ["dong", "đồng", "VND"],
    "Icelandic krona": ["krona", "króna", "ISK"],
    "Brazilian real": ["real", "BRL"],
}

CONTINENTS = {
    "France": "Europe", "Germany": "Europe", "Italy": "Europe", "Spain": "Europe",
    "Portugal": "Europe", "Belgium": "Europe", "Austria": "Europe", "Poland": "Europe",
    "Czechia": "Europe", "Hungary": "Europe", "Greece": "Europe", "Ireland": "Europe",
    "Iceland": "Europe", "Finland": "Europe", "Sweden": "Europe", "Norway": "Europe",
    "Denmark": "Europe", "Switzerland": "Europe", "Ukraine": "Europe",
    "Japan": "Asia", "China": "Asia", "India": "Asia", "South Korea": "Asia",
    "Thailand": "Asia", "Vietnam": "Asia", "Indonesia": "Asia", "Malaysia": "Asia",
    "Philippines": "Asia", "Iran": "Asia", "Iraq": "Asia", "Saudi Arabia": "Asia",
    "Turkey": "Asia",
    "Egypt": "Africa", "Morocco": "Africa", "Kenya": "Africa", "Nigeria": "Africa",
    "Ethiopia": "Africa", "Ghana": "Africa",
    "Canada": "North America", "Mexico": "North America",
    "Brazil": "South America", "Argentina": "South America", "Chile": "South America",
    "Peru": "South America", "Colombia": "South America",
}

SI_UNITS = {
    "length": "metre", "mass": "kilogram", "time": "second",
    "electric current": "ampere", "thermodynamic temperature": "kelvin",
    "amount of substance": "mole", "luminous intensity": "candela",
    "force": "newton", "energy": "joule", "power": "watt", "pressure": "pascal",
    "frequency": "hertz", "electric charge": "coulomb", "voltage": "volt",
    "electrical resistance": "ohm", "capacitance": "farad",
    "magnetic flux": "weber", "inductance": "henry",
}
SI_ALIASES = {
    "metre": ["meter"], "kilogram": ["kg", "kilogramme"], "ampere": ["amp", "A"],
    "kelvin": ["K"], "coulomb": ["C"], "joule": ["J"], "watt": ["W"],
    "pascal": ["Pa"], "hertz": ["Hz"], "newton": ["N"], "volt": ["V"],
    "farad": ["F"], "weber": ["Wb"], "henry": ["H"], "ohm": ["Ω"],
}

PLANET_ORDER = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus",
                "Neptune"]

COMPOUNDS = {
    "water": "H2O", "carbon dioxide": "CO2", "methane": "CH4", "ammonia": "NH3",
    "sodium chloride": "NaCl", "sulfuric acid": "H2SO4", "nitric acid": "HNO3",
    "hydrochloric acid": "HCl", "glucose": "C6H12O6", "ethanol": "C2H5OH",
    "acetic acid": "CH3COOH", "sodium bicarbonate": "NaHCO3",
    "sucrose": "C12H22O11", "hydrogen peroxide": "H2O2", "ozone": "O3",
    "nitrogen dioxide": "NO2", "carbon monoxide": "CO", "calcium carbonate": "CaCO3",
    "potassium permanganate": "KMnO4", "sodium hydroxide": "NaOH",
    "calcium hydroxide": "Ca(OH)2", "magnesium oxide": "MgO",
    "iron(III) oxide": "Fe2O3", "silicon dioxide": "SiO2",
    "hydrogen sulfide": "H2S", "phosphoric acid": "H3PO4",
}
COMPOUND_ALIASES = {
    "C2H5OH": ["C2H6O", "CH3CH2OH"], "CH3COOH": ["C2H4O2", "CH3CO2H"],
    "Ca(OH)2": ["Ca(OH)₂"], "C6H12O6": ["C6H12O6"], "NaHCO3": ["NaHCO₃"],
}


def _norm(text: str) -> str:
    """Lower-case, drop punctuation and collapse whitespace, for answer matching."""
    keep = []
    for ch in text.lower():
        if ch.isalnum() or ch.isspace():
            keep.append(ch)
        else:
            keep.append(" ")
    return " ".join("".join(keep).split())


def gen_element_symbol(rng: random.Random, key: str) -> tuple[str, str, list[str]]:
    return (f"What is the chemical symbol for the element {key}?",
            ELEMENT_SYMBOL[key], [])


def gen_element_number(rng: random.Random, key: str) -> tuple[str, str, list[str]]:
    return (f"What is the atomic number of the element {key}?",
            str(ELEMENT_NUMBER[key]), [])


def gen_country_capital(rng: random.Random, key: str) -> tuple[str, str, list[str]]:
    answer = CAPITALS[key]
    return (f"What is the capital city of {key}?", answer,
            CAPITAL_ALIASES.get(answer, []))


def gen_country_currency(rng: random.Random, key: str) -> tuple[str, str, list[str]]:
    answer = CURRENCIES[key]
    return (f"What is the official currency of {key}?", answer,
            CURRENCY_ALIASES.get(answer, []))


def gen_country_continent(rng: random.Random, key: str) -> tuple[str, str, list[str]]:
    return (f"On which continent is {key}?", CONTINENTS[key], [])


def gen_si_unit(rng: random.Random, key: str) -> tuple[str, str, list[str]]:
    answer = SI_UNITS[key]
    return (f"What is the SI unit of {key}?", answer, SI_ALIASES.get(answer, []))


def gen_planet_order(rng: random.Random, key: str) -> tuple[str, str, list[str]]:
    n = PLANET_ORDER.index(key) + 1
    return (f"Which planet is number {n} counting outward from the Sun?",
            key, [])


def gen_compound_formula(rng: random.Random, key: str) -> tuple[str, str, list[str]]:
    answer = COMPOUNDS[key]
    return (f"What is the chemical formula for {key}?", answer,
            COMPOUND_ALIASES.get(answer, []))


CONTROL_FAMILIES = [
    ("element_symbol", gen_element_symbol, sorted(ELEMENT_SYMBOL)),
    ("element_atomic_number", gen_element_number, sorted(ELEMENT_NUMBER)),
    ("country_capital", gen_country_capital, sorted(CAPITALS)),
    ("country_currency", gen_country_currency, sorted(CURRENCIES)),
    ("country_continent", gen_country_continent, sorted(CONTINENTS)),
    ("si_unit", gen_si_unit, sorted(SI_UNITS)),
    ("planet_order", gen_planet_order, list(PLANET_ORDER)),
    ("compound_formula", gen_compound_formula, sorted(COMPOUNDS)),
]

CONTROL_RENDER = {name: fn for name, fn, _ in CONTROL_FAMILIES}
CONTROL_KEYS = {name: keys for name, _, keys in CONTROL_FAMILIES}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_maths(n: int, seed: int, seen: set[str] | None = None) -> list[dict]:
    """Round-robin across the ten families so no family dominates the set."""
    seen = seen if seen is not None else set()
    items: list[dict] = []
    rngs = {name: random.Random(seed + i) for i, (name, _, _, _) in
            enumerate(MATHS_FAMILIES)}
    family_index = 0
    attempts = 0
    while len(items) < n:
        name, gen, _, places = MATHS_FAMILIES[family_index % len(MATHS_FAMILIES)]
        attempts += 1
        if attempts > n * 200:
            raise RuntimeError("could not generate enough distinct maths items")
        question, params = gen(rngs[name])
        if question in seen:
            # Retry the same family rather than stepping to the next one. Advancing the
            # pointer here lets a collision cost that family its turn, and the set drifts
            # out of balance -- at n=200 that was 18 to 21 per family instead of 20. The
            # committed 50 and held-out 18 regenerate identically either way, so this
            # changes nothing that has already been scored.
            continue
        family_index += 1
        seen.add(question)
        answer = MATHS_COMPUTE[name](params)
        items.append({
            "id": f"m{len(items) + 1:03d}",
            "family": name,
            "question": question,
            "answer": quantise(answer, places),
            "answer_type": "number",
            "places": places,
            "params": params,
            "held_out": False,
        })
    return items


def build_control(n: int, seed: int, keys: dict[str, list[str]],
                  seen: set[str] | None = None) -> list[dict]:
    """Round-robin across the control families, drawing each family's keys once."""
    seen = seen if seen is not None else set()
    rng = random.Random(seed)
    pools = {name: list(ks) for name, ks in keys.items()}
    for name in pools:
        rng.shuffle(pools[name])
    items: list[dict] = []
    family_index = 0
    while len(items) < n:
        name, fn, _ = CONTROL_FAMILIES[family_index % len(CONTROL_FAMILIES)]
        family_index += 1
        pool = pools[name]
        if not pool:
            # Every key in this family is used up. Skip it rather than repeat one.
            if all(not p for p in pools.values()):
                raise RuntimeError("control tables exhausted")
            continue
        key = pool.pop()
        question, answer, aliases = fn(rng, key)
        if question in seen:
            continue
        seen.add(question)
        items.append({
            "id": f"c{len(items) + 1:03d}",
            "family": name,
            "question": question,
            "answer": answer,
            "answer_type": "text",
            "aliases": aliases,
            "params": {"key": key},
            "held_out": False,
        })
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).parent))
    args = ap.parse_args()
    out = Path(args.out)

    maths_seen: set[str] = set()
    control_seen: set[str] = set()

    maths = build_maths(50, BASE_SEED, maths_seen)
    control = build_control(50, BASE_SEED + 101, CONTROL_KEYS, control_seen)

    # Held out: eighteen maths and two control, generated with a different seed so the
    # numbers differ from the scored set, and pooled with it to guarantee no prompt is
    # shared. Eighteen is what the blend-weight search is fitted against; see the module
    # docstring for why eight was not enough.
    held_maths = build_maths(18, SEED_HELDOUT, maths_seen)
    for i, item in enumerate(held_maths):
        item["id"] = f"h{i + 1:03d}"
        item["held_out"] = True

    held_control = build_control(2, SEED_HELDOUT + 5,
                                 {k: [x for x in v] for k, v in CONTROL_KEYS.items()},
                                 control_seen)
    for i, item in enumerate(held_control):
        item["id"] = f"h{len(held_maths) + i + 1:03d}"
        item["held_out"] = True

    heldout = held_maths + held_control

    for path, payload, note in [
        (out / "questions_maths.json", maths,
         "Stage 1 maths set. Every answer is the generator's own exact arithmetic."),
        (out / "questions_control.json", control,
         "Stage 1 control set. Every answer is a lookup in a table in make_questions.py."),
        (out / "questions_heldout.json", heldout,
         "Held out for the blend-weight search only. Never scored."),
    ]:
        path.write_text(json.dumps({
            "_note": note,
            "generator": "make_questions.py",
            "base_seed": BASE_SEED,
            "items": payload,
        }, indent=2, ensure_ascii=False) + "\n")

    print(f"maths   {len(maths):>3} items -> questions_maths.json")
    print(f"control {len(control):>3} items -> questions_control.json")
    print(f"heldout {len(heldout):>3} items -> questions_heldout.json "
          f"({len(held_maths)} maths, {len(held_control)} control)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
