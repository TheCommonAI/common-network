"""Regression tests for the deterministic verifier.

Plain Python, no pytest — `python tests/test_verify.py` from `gateway/`. The
gateway's requirements are the gateway's runtime deps and nothing else; a test
suite that needs an install step is a test suite that stops being run.

Every case here is one that actually broke during development. Four of them
were bugs that would have produced *confident wrong verification* — the
verifier flagging a correct figure, or manufacturing a disagreement between two
specialists that agreed. That failure mode is worse than no verifier: it
teaches the aggregator to hedge on figures that were never in doubt, and it is
the same class of error the seam findings caught three times in the harness
(`seam-findings.md` §4, "three measurement bugs caught before they became
findings").
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.verify import parse_number, verify  # noqa: E402

FAILURES: list[str] = []


def check(name: str, got, want) -> None:
    if got != want:
        FAILURES.append(f"{name}\n     got:  {got!r}\n     want: {want!r}")
        print(f"  FAIL  {name}")
    else:
        print(f"  ok    {name}")


def arith(text: str) -> list[tuple[str, bool]]:
    return [(c.excerpt, c.ok) for c in verify({"n": text}).checks]


def disagreements(answers: dict[str, str]) -> list[str]:
    return [d.label for d in verify(answers).disagreements]


print("\nnumber parsing")
check("plain", parse_number("1200"), parse_number("1200"))
check("thousands separator", str(parse_number("1,200.50")), "1200.50")
check("currency", str(parse_number("$605")), "605")
check("percent resolves to fraction", str(parse_number("15%")), "0.15")
check("not a number", parse_number("clause"), None)

print("\narithmetic re-derivation")
check("correct sum passes", arith("150 + 250 = 400"), [("150 + 250 = 400", True)])
check("wrong sum fails", arith("150 + 250 = 500"), [("150 + 250 = 500", False)])
check("precedence respected", arith("2 + 3 * 4 = 14"), [("2 + 3 * 4 = 14", True)])
check("parenthesised", arith("(100 + 20) * 3 = 360"), [("(100 + 20) * 3 = 360", True)])
check("parenthesised wrong", arith("(100 + 20) * 3 = 400"), [("(100 + 20) * 3 = 400", False)])
check("currency and commas", arith("$1,200.50 + $300 = $1,500.50"),
      [("1,200.50 + $300 = $1,500.50", True)])
check("percent inside expression", arith("2000 * 15% = 300"), [("2000 * 15% = 300", True)])
check("percent as result", arith("300 / 2000 = 15%"), [("300 / 2000 = 15%", True)])
check("percent-of prose form", arith("GST at 10% of 4500 is 450"),
      [("10% of 4500 is 450", True)])
check("percent-of prose form, wrong", arith("10% of 4500 is 460"),
      [("10% of 4500 is 460", False)])

print("\nrounding is not error")
# The findings recorded a model faithfully reproducing a 16-significant-figure
# value. A verifier stricter than the models are is a verifier that cries wolf.
check("2dp rounding accepted", arith("10 / 3 = 3.33"), [("10 / 3 = 3.33", True)])
check("real case: daily interest", arith("5920 * 0.05 / 365 = 0.81"),
      [("5920 * 0.05 / 365 = 0.81", True)])
check("but a real error still fails", arith("5920 * 0.05 / 365 = 0.92"),
      [("5920 * 0.05 / 365 = 0.92", False)])

print("\nrefusing to guess")
check("unbalanced parens skipped", arith("f(x) + 3 = 7"), [])
check("division by zero skipped", arith("5 / 0 = 0"), [])
check("phone numbers and clauses untouched",
      arith("Call 555 1234 or see clause 4.2 = section"), [])

print("\ncross-specialist disagreement")
check("identical figures agree", disagreements(
    {"a": "The total is 400.", "b": "The total is 400."}), [])
check("genuine conflict caught", disagreements(
    {"a": "The total is 400.", "b": "The total is 450."}), ["total"])
check("agreeing figure not flagged alongside a conflicting one", disagreements(
    {"a": "Deadline is 28 days. Fee is $605.",
     "b": "Deadline is 21 days. Fee is $605."}), ["deadline"])
check("rounding across specialists is not disagreement", disagreements(
    {"a": "Daily interest: 0.8104", "b": "Daily interest: 0.81"}), [])
check("single answer means nothing to cross-check", disagreements(
    {"a": "The total is 400."}), [])

# The bug this guards against: "daily interest is 5920 * 0.05 / 365 = 0.81"
# captured `daily interest = 5920` -- the first operand of the working read as
# the result -- and reported a disagreement against a specialist who had
# written the correct 0.81.
check("operand of an expression is not a stated value", disagreements(
    {"a": "The daily interest is 5920 * 0.05 / 365 = 0.81",
     "b": "Daily interest: 0.81"}), [])

# The bug this guards against: a guard against partial-number matches also
# rejected any figure ending a sentence, which silently disabled the whole
# cross-check on ordinary prose.
check("figure ending a sentence is still seen", disagreements(
    {"a": "Total payable: $5,920.00.", "b": "Total payable: $5,290.00."}),
    ["total payable"])

print("\nreport")
report = verify({"a": "Total: 400. And 150 + 250 = 500.", "b": "Total: 450."})
check("counts", (len(report.checks), len(report.failed), len(report.disagreements)),
      (1, 1, 1))
check("fire rate", report.fire_rate, 1.0)
check("clean report produces no prompt section", verify({"a": "No numbers here."}).as_prompt_section(), "")
section = report.as_prompt_section()
check("prompt section names the recomputed value", "400" in section, True)
check("prompt section flags the disagreement", "total" in section, True)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED\n")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all verifier tests passed")
