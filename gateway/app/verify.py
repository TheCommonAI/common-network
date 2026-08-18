"""Deterministic verification of what specialists hand to the aggregator.

This module implements the one intervention that measurably worked in the v0.1
seam experiments: recompute derivable values at the receiving end and flag
disagreement, instead of letting an unverified figure arrive at the aggregator
carrying full confidence. See `testing/seam-findings.md` §1 — "the one fix that
works", +0.203 against a ±0.061 noise floor, where four other plausible fixes
(schema enforcement, symmetric verification, JSON-vs-prose, terse contracts) all
came in under the noise floor.

The mechanism it is built on, in the findings' own words:

    The failure is *not* information degrading in transit. It is unverified
    values arriving with full confidence at a receiver with no means to check
    them.

Three rules, each of which is a direct consequence of those findings:

1. **No model is involved in verification.** Every figure produced here comes
   from Python arithmetic on `Decimal`. Asking a model to check a model
   reproduces the exact failure this is meant to catch.

2. **It flags; it never silently rewrites.** The aggregator is *told* that a
   figure failed to recompute and is given the recomputed value. Rewriting the
   specialist's text in place would destroy the audit trail, and Common's
   whole claim is that routing decisions are legible.

3. **It only checks what is genuinely derivable from the text.** The findings
   are explicit that hand-writing derivation rules for 16 cases "proves the
   mechanism and nothing about shipping it", and that where rules come from at
   scale is the central open research question. So this module deliberately
   covers only the subset that needs no hand-written rules at all: arithmetic a
   specialist *showed its working for*, and figures two specialists both
   asserted. Anything else is left alone rather than guessed at.

The residual is real and is not papered over: the findings note ~11 failures
that were the receiving model's own arithmetic on values it computed itself.
Nothing here reaches those.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from decimal import Decimal, DivisionByZero, InvalidOperation, getcontext

# Enough precision that re-deriving a figure never fails for a reason of our
# own making. The findings recorded a model faithfully reproducing
# `daily_interest: 8.107876712328768` — 16 significant figures — so a
# verifier working to fewer digits than the models do would manufacture
# disagreements that aren't there.
getcontext().prec = 40


# --- Number parsing -------------------------------------------------------

# Currency symbols, thousands separators and trailing percent signs all appear
# in ordinary model output. They are notation, not disagreement.
_CURRENCY = "$£€¥"
# Built by concatenation rather than %-formatting throughout this module: the
# patterns themselves contain literal `%` (percentages are the whole point),
# and %-formatting them is a silent trap.
_NUM_RE = r"[-+]?[" + re.escape(_CURRENCY) + r"]?\s?\d[\d,]*(?:\.\d+)?%?"


def parse_number(raw: str) -> Decimal | None:
    """Parse a number as it appears in prose. Returns None if it isn't one.

    A trailing `%` is resolved to its value (`15%` -> `0.15`) rather than
    dropped, so that `300 / 2000 = 15%` verifies correctly instead of being
    scored as a factor-of-100 error.
    """
    s = raw.strip()
    is_pct = s.endswith("%")
    if is_pct:
        s = s[:-1]
    s = s.strip().lstrip("+")
    for sym in _CURRENCY:
        s = s.replace(sym, "")
    s = s.replace(",", "").strip()
    if not s or s in {"-", "."}:
        return None
    try:
        value = Decimal(s)
    except InvalidOperation:
        return None
    return value / 100 if is_pct else value


def _decimal_places(raw: str) -> int:
    """How precisely a figure was *stated*, so rounding isn't read as error."""
    s = raw.strip().rstrip("%")
    if "." not in s:
        return 0
    return len(s.split(".")[-1].strip())


def _agrees(stated: Decimal, computed: Decimal, stated_raw: str) -> bool:
    """Does a stated figure agree with an independently computed one?

    Rounding is not an error. A specialist that writes `$8.11` for a value that
    recomputes to `8.107876712328768` has done nothing wrong, and flagging it
    would bury the real failures in noise — the findings are emphatic that a
    check firing on everything is as useless as one firing on nothing.
    """
    if stated == computed:
        return True
    dp = _decimal_places(stated_raw)
    try:
        quant = Decimal(1).scaleb(-dp)
        if computed.quantize(quant) == stated.quantize(quant):
            return True
    except (InvalidOperation, ValueError):
        pass
    # Relative tolerance, for figures large enough that absolute equality is
    # unreasonable and small enough that quantize would overflow.
    if computed != 0:
        try:
            if abs((stated - computed) / computed) <= Decimal("1e-9"):
                return True
        except (InvalidOperation, DivisionByZero):
            pass
    return False


# --- Arithmetic re-derivation --------------------------------------------

# An equality whose left-hand side is arithmetic and nothing else. Restricting
# the LHS character class this tightly is the point: it means we only ever
# re-derive a calculation the specialist actually showed, never one we inferred
# it might have meant.
_EQUATION_RE = re.compile(
    r"(?<![\w.])"
    # Optional leading paren so "(100 + 20) * 3 = 360" is reachable; the
    # balance check in _normalise_expression rejects anything lopsided.
    r"(?P<lhs>\(?[\d,.]+\s*(?:[-+*/×÷]|\bx\b)\s*[\d,.()%\s+\-*/×÷$£€¥]*[\d,.%)])"
    r"\s*=\s*"
    r"(?P<rhs>" + _NUM_RE + r")"
)

# "15% of 2000 is 300" — extremely common in costing prose and not an equation,
# so the equality matcher above will never see it.
_PERCENT_OF_RE = re.compile(
    r"(?P<pct>\d[\d,]*(?:\.\d+)?)\s*%\s+of\s+"
    r"(?P<base>" + _NUM_RE + r")"
    r"\s*(?:=|is|comes to|equals)\s*"
    r"(?P<result>" + _NUM_RE + r")",
    re.IGNORECASE,
)

_ALLOWED_AST = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd)


def _normalise_expression(lhs: str) -> str | None:
    """Turn prose arithmetic into something `ast` can parse, or give up.

    Giving up is a perfectly good outcome. A verifier that guesses at what an
    ambiguous expression meant will eventually flag a correct figure as wrong,
    and a false alarm costs more than a missed check: it teaches the aggregator
    to distrust a value that was fine.
    """
    s = lhs
    for sym in _CURRENCY:
        s = s.replace(sym, "")
    s = s.replace(",", "")
    s = s.replace("×", "*").replace("÷", "/")
    s = re.sub(r"\bx\b", "*", s)
    # `15%` inside an expression means the fraction, not the integer.
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", s)
    s = s.strip()
    if not re.fullmatch(r"[\d.()*/+\-\s]+", s):
        return None
    if s.count("(") != s.count(")"):
        return None
    return s


def _eval_decimal(node: ast.AST) -> Decimal:
    """Evaluate a whitelisted arithmetic AST in Decimal.

    `ast` rather than `eval` because this runs on text that arrived from an
    arbitrary node on a permissionless network. `Decimal` rather than `float`
    because binary floating point would introduce exactly the kind of
    small-magnitude disagreement this module exists to distinguish from real
    error.
    """
    if isinstance(node, ast.Expression):
        return _eval_decimal(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("non-numeric constant")
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp):
        operand = _eval_decimal(node.operand)
        return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.BinOp):
        left, right = _eval_decimal(node.left), _eval_decimal(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("division by zero")
            return left / right
    raise ValueError("disallowed expression")


@dataclass
class ArithmeticCheck:
    """One calculation a specialist showed, re-derived independently."""
    source_node: str
    expression: str
    stated: Decimal
    computed: Decimal
    ok: bool
    excerpt: str


def check_arithmetic(text: str, source_node: str) -> list[ArithmeticCheck]:
    """Re-derive every calculation the text shows its working for."""
    checks: list[ArithmeticCheck] = []
    seen: set[str] = set()

    for m in _EQUATION_RE.finditer(text or ""):
        excerpt = m.group(0).strip().rstrip(",;.")
        if excerpt in seen:
            continue
        expr = _normalise_expression(m.group("lhs"))
        stated = parse_number(m.group("rhs"))
        if expr is None or stated is None:
            continue
        try:
            computed = _eval_decimal(ast.parse(expr, mode="eval"))
        except (SyntaxError, ValueError, InvalidOperation, DivisionByZero, RecursionError):
            continue
        seen.add(excerpt)
        checks.append(ArithmeticCheck(
            source_node=source_node,
            expression=m.group("lhs").strip(),
            stated=stated,
            computed=computed,
            ok=_agrees(stated, computed, m.group("rhs")),
            excerpt=excerpt,
        ))

    for m in _PERCENT_OF_RE.finditer(text or ""):
        excerpt = m.group(0).strip().rstrip(",;.")
        if excerpt in seen:
            continue
        pct = parse_number(m.group("pct"))
        base = parse_number(m.group("base"))
        stated = parse_number(m.group("result"))
        if pct is None or base is None or stated is None:
            continue
        computed = (pct / Decimal(100)) * base
        seen.add(excerpt)
        checks.append(ArithmeticCheck(
            source_node=source_node,
            expression=f"{m.group('pct')}% of {m.group('base')}",
            stated=stated,
            computed=computed,
            ok=_agrees(stated, computed, m.group("result")),
            excerpt=excerpt,
        ))

    return checks


# --- Cross-specialist disagreement ---------------------------------------

# `label: value` and `label is value`. The label is capped at four words
# because longer captures start swallowing sentence fragments, which produces
# labels that never match between two specialists and so silently disable the
# whole cross-check.
_LABELLED_RE = re.compile(
    r"(?P<label>(?:[A-Za-z][A-Za-z\-]*\s+){0,3}[A-Za-z][A-Za-z\-]*)"
    r"\s*(?::|\bis\b|\bof\b|=)\s*"
    r"(?P<value>" + _NUM_RE + r")"
    # Must be a whole number, not a prefix of one. Without this guard the
    # engine backtracks around the operator lookahead below by shortening the
    # match -- "5920 * 0.05" yields the value `592`, which is not a figure
    # anyone wrote.
    # A trailing `.` or `,` only continues the number if a digit follows it --
    # otherwise it is the end of the sentence. Excluding them unconditionally
    # silently drops every figure that ends a sentence, which is most of them.
    r"(?!\d|[.,]\d)"
    # Not followed by an operator. Without this, "daily interest is 5920 *
    # 0.05 / 365 = 0.81" yields `daily interest = 5920` — the first operand of
    # the working, captured as though it were the result. Two specialists doing
    # the same correct sum different ways would then be reported as disagreeing,
    # which is worse than no cross-check at all: it trains the aggregator to
    # hedge on figures that were never in doubt.
    r"(?!\s*(?:[-+*/×÷]|\bx\b)\s*[\d,.$£€¥(])"
)

# Words that make a "label" meaningless on its own -- two specialists both
# saying "total: 500" about different totals is not agreement, and both saying
# "step 2" is not a quantity at all.
_STOP_LABELS = {
    "step", "stage", "part", "section", "item", "no", "number", "question",
    "answer", "example", "figure", "note", "point", "line", "page", "option",
}


def _normalise_label(label: str) -> str:
    words = [w.lower().strip("-") for w in label.split()]
    words = [w for w in words if w and w not in {"the", "a", "an", "your", "its", "this", "that"}]
    if not words or words[-1] in _STOP_LABELS:
        return ""
    if len(words[-1]) > 3 and words[-1].endswith("s") and not words[-1].endswith("ss"):
        words[-1] = words[-1][:-1]
    return " ".join(words[-3:])


@dataclass
class Disagreement:
    """The same named quantity, asserted differently by two specialists."""
    label: str
    values: dict[str, Decimal]  # node name -> value

    def describe(self) -> str:
        parts = ", ".join(f"{node} says {value}" for node, value in self.values.items())
        return f"{self.label}: {parts}"


def find_disagreements(answers: dict[str, str]) -> list[Disagreement]:
    """Find quantities that two or more specialists labelled the same and
    valued differently.

    This is the check that only becomes possible once more than one node
    answers — v0.1, routing to a single node, had no way to notice a wrong
    figure at all. It is cheap, deterministic, and needs no domain rules,
    which is precisely why it is here and hand-written derivation rules are
    not.
    """
    by_label: dict[str, dict[str, Decimal]] = {}
    raw_by_label: dict[str, dict[str, str]] = {}

    for node_name, text in answers.items():
        for m in _LABELLED_RE.finditer(text or ""):
            label = _normalise_label(m.group("label"))
            if not label:
                continue
            value = parse_number(m.group("value"))
            if value is None:
                continue
            # First mention wins per node: a later restatement of the same
            # quantity by the same node is a summary, not a second opinion.
            by_label.setdefault(label, {}).setdefault(node_name, value)
            raw_by_label.setdefault(label, {}).setdefault(node_name, m.group("value"))

    out: list[Disagreement] = []
    for label, values in by_label.items():
        if len(values) < 2:
            continue
        distinct = list(values.items())
        conflict = False
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                (node_a, a), (node_b, b) = distinct[i], distinct[j]
                if not (_agrees(a, b, raw_by_label[label][node_a])
                        or _agrees(b, a, raw_by_label[label][node_b])):
                    conflict = True
        if conflict:
            out.append(Disagreement(label=label, values=values))

    out.sort(key=lambda d: d.label)
    return out


# --- Report ---------------------------------------------------------------

@dataclass
class VerificationReport:
    checks: list[ArithmeticCheck] = field(default_factory=list)
    disagreements: list[Disagreement] = field(default_factory=list)

    @property
    def failed(self) -> list[ArithmeticCheck]:
        return [c for c in self.checks if not c.ok]

    @property
    def fire_rate(self) -> float:
        """Fraction of re-derived calculations that disagreed.

        Worth watching rather than just recording. The seam findings measured
        overrides firing on 42% of handoffs; a rate near 0% here more likely
        means the extractor stopped matching than that the models became
        correct. A check nobody audits is a check that quietly dies.
        """
        return len(self.failed) / len(self.checks) if self.checks else 0.0

    def as_prompt_section(self) -> str:
        """The block handed to the aggregator.

        Phrased as instruction rather than suggestion, and it hands over the
        recomputed value rather than merely announcing a problem — the findings
        showed receivers accept handed values faithfully and unrounded, so
        giving the aggregator the right number is the reliable move.
        """
        if not self.failed and not self.disagreements:
            return ""

        lines = ["INDEPENDENT VERIFICATION (computed by the gateway, not by a model):"]

        if self.failed:
            lines.append("")
            lines.append("These calculations did not check out when recomputed. Use the")
            lines.append("recomputed value. Do not repeat the stated one.")
            for c in self.failed:
                lines.append(f"  - {c.source_node} wrote \"{c.excerpt}\" — "
                             f"{c.expression} actually equals {c.computed}, not {c.stated}.")

        if self.disagreements:
            lines.append("")
            lines.append("These specialists disagree with each other. Do not silently pick")
            lines.append("one. Either resolve it by showing the working, or state plainly in")
            lines.append("your answer that the figure is uncertain and give both.")
            for d in self.disagreements:
                lines.append(f"  - {d.describe()}")

        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "checks_run": len(self.checks),
            "checks_failed": len(self.failed),
            "disagreements": len(self.disagreements),
            "fire_rate": round(self.fire_rate, 4),
            "failures": [
                {
                    "node": c.source_node,
                    "excerpt": c.excerpt,
                    "stated": str(c.stated),
                    "computed": str(c.computed),
                }
                for c in self.failed
            ],
            "disagreement_detail": [
                {"label": d.label, "values": {k: str(v) for k, v in d.values.items()}}
                for d in self.disagreements
            ],
        }


def verify(answers: dict[str, str]) -> VerificationReport:
    """Verify a panel's answers. `answers` maps node name -> answer text."""
    checks: list[ArithmeticCheck] = []
    for node_name, text in answers.items():
        checks.extend(check_arithmetic(text, node_name))
    return VerificationReport(
        checks=checks,
        disagreements=find_disagreements(answers) if len(answers) > 1 else [],
    )
