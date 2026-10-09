"""Which node answers: the decision below the confidence threshold.

`gateway.choose_primary` is the rule that decides where a request goes when the
embedder's ranking cannot be trusted, and it is where a measured routing failure
lived: a clear maths question with a maths specialist online was answered by the
generalist instead. Two things caused it, both pinned below.

1. The generalist fallback swapped on the incumbent being *unconfident*, never
   checking that the generalist was *better*. On "A train travels 240 km in
   3 hours. What is its average speed?" it chose a node scoring 0.392 over one
   scoring 0.400 -- worse by the gateway's own measure, on a margin of 0.008.
   That small margin is why the same question routed correctly some days and
   not others.

2. The deterministic signal that answers the question the embedder cannot -- is
   this a request for a calculation? -- existed but was only wired into the
   panel gate, which is off by default (COMPOSE_MODE=never). And its cue
   vocabulary was finance-only, so it missed the maths lane entirely.

No embedder here on purpose. The decision rule is asserted against hand-built
scores, so the suite pins the *rule* rather than today's embedding of a
sentence, and `tests/run_all.py` stays runnable with no model downloaded.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.compose import is_calculation_request  # noqa: E402
from app.config import settings  # noqa: E402
from app.gateway import choose_primary  # noqa: E402
from app.router import ScoredNode  # noqa: E402

FAILURES = []


def check(name: str, got, want) -> None:
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL  {name}  (got {got!r}, want {want!r})")
    else:
        print(f"  ok    {name}")


def node(name: str, tags: list[str], topical: float) -> ScoredNode:
    """A candidate whose topical_score is exactly `topical`.

    topical_score is w_sim * sim + tag_term, so the similarity is set to carry
    the whole of it.
    """
    return ScoredNode(
        node={"id": name, "name": name, "domain_tags": tags},
        score=topical, sim=topical / settings.w_sim, cost_term=0.0,
        lat_term=0.0, region_term=0.0, tag_term=0.0,
    )


MATHS = ["math", "arithmetic", "quantitative"]
CODER = ["code", "programming", "debugging"]
GENERAL = ["general", "conversation", "writing"]
LEGAL = ["legal", "tenancy"]

print("\nis_calculation_request: the vocabulary gap that caused the bug")
# The three the original finance-only cue list caught.
for label, text in [
    ("money owed", "I'm 8 weeks behind at $340 a week, how much do I owe?"),
    ("percent of", "What is 17 percent of 4,500?"),
    ("cost/total", "If 7 pencils cost $3 and I buy 12, how much do I pay?"),
    # The maths lane, missed entirely before: two figures but no cue word --
    # nothing in the old list matched "average" or "speed", and \brate\b does
    # not match "average" either.
    ("distance/rate word problem",
     "A train travels 240 km in 3 hours. What is its average speed?"),
    ("derivative", "What is the derivative of x^3 + 2x with respect to x?"),
    ("quadratic", "Solve for x: 2x^2 - 8x + 6 = 0"),
    ("fractions", "What is 3/4 divided by 2/5?"),
    ("algebra", "Factorise x^2 - 5x + 6"),
    ("unit conversion", "Convert 45 miles per hour into metres per second"),
]:
    check(f"fires on {label}", is_calculation_request(text), True)

# The negative case the two-figure requirement exists for. It must stay out --
# this was chosen deliberately in compose.py and is not mine to relax.
for label, text in [
    ("ingredients, no calculation",
     "How do I make carbonara? I have 2 eggs and 100g of guanciale."),
    ("poem", "Write me a short friendly poem about the sea."),
    ("chit-chat", "Hey, how has your week been going?"),
    ("code question", "Why does my Python list comprehension throw a TypeError?"),
    ("prose with 'simplify'", "Help me simplify my morning routine a bit."),
    ("pure legal", "Can my landlord enter without notice in South Australia?"),
    ("empty", ""),
]:
    check(f"stays quiet on {label}", is_calculation_request(text), False)

print("\nabove the confidence threshold: similarity still decides")
# 0.55 and 0.50 both clear 0.40, so neither fallback may fire -- even though the
# request is a calculation and a maths node is available.
above = [node("coder", CODER, 0.55), node("maths", MATHS, 0.50), node("gen", GENERAL, 0.45)]
primary, backup = choose_primary(above, "A train travels 240 km in 3 hours. What is its average speed?")
check("highest score wins", primary.node["id"], "coder")
check("runner-up is the backup", backup.node["id"], "maths")

print("\nbelow the threshold: a calculation request takes the arithmetic lane")
# The reported bug, with the live numbers it produced: the maths node ranked
# LAST for a word problem (0.387) and the generalist second (0.392). The
# incumbent is 0.399 rather than the 0.400 that measurement printed, because it
# printed three decimals and the gate only fires *below* 0.40 -- so the real
# value was some fraction under, which is precisely how a rounding-boundary
# score turned into a wrong answer some of the time and not others.
TRAIN = "A train travels 240 km in 3 hours. What is its average speed?"
bug = [node("coder", CODER, 0.399), node("gen", GENERAL, 0.392), node("maths", MATHS, 0.387)]
primary, backup = choose_primary(bug, TRAIN)
check("maths specialist answers a maths question", primary.node["id"], "maths")
check("what similarity chose becomes the backup", backup.node["id"], "coder")

# Same ranking, but nothing in it is a maths node: the lane cannot fire, and the
# generalist may only take it by actually outscoring the incumbent.
no_lane = [node("coder", CODER, 0.399), node("gen", GENERAL, 0.392), node("legal", LEGAL, 0.380)]
primary, _ = choose_primary(no_lane, TRAIN)
check("no lane holder -> no lane promotion", primary.node["id"], "coder")

print("\nthe threshold boundary itself")
# `>=` is the shipped comparison: a score exactly at the threshold is treated as
# confident and similarity decides. Asserted because the whole failure lived in
# the half-point under it.
at = [node("coder", CODER, settings.routing_confidence_threshold),
      node("maths", MATHS, settings.routing_confidence_threshold - 0.05)]
check("exactly at the threshold counts as confident",
      choose_primary(at, TRAIN)[0].node["id"], "coder")

print("\nbelow the threshold: a generalist must actually be better")
# The exact regression. 0.392 does not beat 0.399, so the old rule's swap to a
# strictly-worse node must not happen. The 0.007 gap is the one that made the
# same question route correctly some days and not others.
check("lower-scoring generalist does NOT override",
      choose_primary(no_lane, "Tell me about the history of trains.")[0].node["id"], "coder")

# A generalist that genuinely outscores may take over -- the original intent,
# which is still right when the gap is real.
better_gen = [node("coder", CODER, 0.36), node("gen", GENERAL, 0.44)]
check("clearly better generalist DOES override",
      choose_primary(better_gen, "Tell me about the history of trains.")[0].node["id"], "gen")

# Inside the margin it must not flip: two nodes 0.010 apart is a coin toss, and
# the answer should not change with the toss.
tied = [node("coder", CODER, 0.38), node("gen", GENERAL, 0.39)]
check("generalist inside the margin does NOT override",
      choose_primary(tied, "Tell me about the history of trains.")[0].node["id"], "coder")
check("margin is what separates them", 0.39 > 0.38 + settings.generalist_override_margin, False)

print("\nno fallback available")
only = [node("legal", LEGAL, 0.20)]
primary, backup = choose_primary(only, "A train travels 240 km in 3 hours. What is its average speed?")
check("a lone off-topic node still answers", primary.node["id"], "legal")
check("and has no backup", backup, None)

# A single generalist, unconfident, is still the only candidate -- and must not
# be swapped for itself.
solo_gen = [node("gen", GENERAL, 0.20)]
primary, _ = choose_primary(solo_gen, "Anything at all.")
check("generalist is not swapped for itself", primary.node["id"], "gen")

# The incumbent already holds the lane: promoting it would be a no-op, and the
# backup must stay the runner-up rather than the incumbent.
incumbent_is_lane = [node("maths", MATHS, 0.39), node("gen", GENERAL, 0.35)]
primary, backup = choose_primary(incumbent_is_lane, "Solve for x: 2x^2 - 8x + 6 = 0")
check("lane holder already leading stays", primary.node["id"], "maths")
check("backup is the runner-up", backup.node["id"], "gen")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED\n")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all routing tests passed")
