"""Composition: many specialists, one answer.

This is what v0.1.1 exists for. v0.1 routed one request to one node, which made
the network a load balancer with good manners — it could never beat its own
best node. Composition is the claim that a panel of narrow specialists can
answer better than any one of them alone.

That claim is not assumed here. It was tested in v0.1 and it *failed*
(`testing/seam-findings.md` §2: no gain on either axis, and one topology
actively made things worse). The failure was diagnosed precisely, and this
module is built around the diagnosis rather than around the original design.

Three findings, three design decisions:

1. **"Routing only pays when specialists are genuinely non-dominated."**
   Experiment 2 composed two models where one was better at both axes; at
   70-72B the *writing* model was the better mathematician. Combining a model
   with one it dominates cannot help — you can only dilute the stronger one.
   So composition here is *gated*, not default: `plan_panel` refuses to compose
   unless the request genuinely spans domains that different nodes are best at,
   and it records why in every decision. A gateway that composes everything
   would reproduce Experiment 2 at scale.

2. **Chaining damages output; parallel does not.** Routing a draft *through*
   the maths specialist first dropped the prose win rate to 0.188 — the writer
   inherited the specialist's framing along with its figures. Parallel + an
   aggregator came in at 0.438, indistinguishable from baseline. So parallel
   fan-out is the only topology implemented. Sequential chaining is out of
   scope for a measured reason, not an unexamined one.

3. **"Unverified values arriving with full confidence at a receiver with no
   means to check them."** The one intervention that beat the noise floor
   (+0.203) was recomputing derivable values at the receiver. The aggregator
   here never receives raw specialist output alone: it receives it alongside a
   deterministic verification report (`app/verify.py`).

What is still unproven, stated plainly: whether a *genuinely* non-dominated
panel beats its own best member. v0.1 could not test it because no maths-tuned
model is hosted on OpenRouter at all. Common runs local Ollama weights, where
those models do exist — so the experiment is now runnable, and
`testing/compose-test/` is the instrument for running it. Until it does, this
module is a well-founded hypothesis with a gate on it, and `COMPOSE_MODE`
defaults to `auto` so it only fires where the preconditions hold.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from app import upstream
from app.config import settings
from app.router import ScoredNode, domain_similarities
from app.verify import VerificationReport


# --- Plan -----------------------------------------------------------------

@dataclass
class PanelMember:
    """One specialist on the panel, and the domain it earned its seat with."""
    scored: ScoredNode
    domain: str
    domain_sim: float

    @property
    def node(self) -> dict:
        return self.scored.node

    @property
    def name(self) -> str:
        return self.scored.node["name"]


@dataclass
class PanelPlan:
    compose: bool
    members: list[PanelMember] = field(default_factory=list)
    aggregator: dict | None = None
    aggregator_in_panel: bool = False
    reason: str = ""
    matched_domains: list[tuple[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        """The explanation attached to the decisions row and the response
        headers. Common's pitch is that routing is legible; a panel that formed
        for reasons nobody can reconstruct is worse than no panel."""
        return {
            "compose": self.compose,
            "reason": self.reason,
            "matched_domains": [{"domain": d, "similarity": round(s, 4)}
                                for d, s in self.matched_domains],
            "panel": [
                {
                    "node": m.name,
                    "domain": m.domain,
                    "domain_similarity": round(m.domain_sim, 4),
                    "score": round(m.scored.score, 4),
                    "topical_score": round(m.scored.topical_score, 4),
                }
                for m in self.members
            ],
            "aggregator": self.aggregator["name"] if self.aggregator else None,
            "aggregator_in_panel": self.aggregator_in_panel,
        }


def _pick_aggregator(scored: list[ScoredNode], panel_ids: set) -> tuple[dict | None, bool]:
    """Choose who synthesises the panel's answers.

    Preference order, and the reasoning behind it:

    1. A node that opted in via `can_aggregate` and is *not* on the panel.
       Aggregating is a different job from answering in a lane — it wants
       breadth and instruction-following, not depth — and a member grading its
       own contribution is a conflict of interest we can simply avoid.
    2. The best-scoring generalist not on the panel.
    3. A panel member, as a last resort. On a small network (a school lab on
       its first day, three machines up) refusing to compose for want of a
       spare node would mean composition never runs at all. This is allowed but
       recorded — `aggregator_in_panel` lands in the decision row so any
       measured result can be split on it, rather than a self-favouring
       aggregator quietly inflating the numbers.
    """
    off_panel = [s for s in scored if s.node["id"] not in panel_ids]

    volunteers = [s for s in off_panel if s.node.get("can_aggregate")]
    if volunteers:
        return volunteers[0].node, False

    generalists = [s for s in off_panel if "general" in (s.node.get("domain_tags") or [])]
    if generalists:
        return generalists[0].node, False

    if off_panel:
        return off_panel[0].node, False

    on_panel_generalists = [s for s in scored if "general" in (s.node.get("domain_tags") or [])]
    if on_panel_generalists:
        return on_panel_generalists[0].node, True
    return (scored[0].node, True) if scored else (None, False)


def plan_panel(scored: list[ScoredNode], request_embed: list[float],
               mode_override: str | None = None) -> PanelPlan:
    """Decide whether this request should be composed, and by whom.

    The gate is the whole point. Every `compose=False` branch below corresponds
    to a condition under which the v0.1 experiments showed composition cannot
    help — refusing to compose is a *result* being applied, not a limitation.

    `mode_override` comes from the `X-Common-Compose` request header, so a
    harness can run both arms against one gateway without restarting it. It
    cannot override the domination gate: 'always' relaxes the heuristics about
    when composing is *worth* it, not the finding about when it is actively
    counterproductive.
    """
    mode = mode_override or settings.compose_mode
    if mode == "never":
        return PanelPlan(compose=False, reason="composition disabled (COMPOSE_MODE=never)")

    if len(scored) < 2:
        return PanelPlan(compose=False,
                         reason="only one healthy node — nothing to compose with")

    nodes = [s.node for s in scored]
    excluded = {t.strip().lower() for t in settings.compose_excluded_domains.split(",") if t.strip()}
    sims = [(d, s) for d, s in domain_similarities(nodes, request_embed)
            if d.lower() not in excluded]
    # 'always' ignores the similarity floor as well as the gates below, so that
    # the experimental arm really is "compose wherever two different nodes lead
    # two different declared domains". A mode that still silently declined on a
    # threshold would make an A/B look like a null result when the treatment
    # never actually ran.
    matched = (sims[: settings.compose_max_panel] if mode == "always"
               else [(d, s) for d, s in sims if s >= settings.compose_domain_floor])

    if len(matched) < settings.compose_min_domains and mode != "always":
        top = f"{matched[0][0]} ({matched[0][1]:.2f})" if matched else "none"
        return PanelPlan(
            compose=False,
            matched_domains=sims[:4],
            reason=(
                f"single-domain request — only {len(matched)} domain cleared the "
                f"{settings.compose_domain_floor:.2f} floor (best: {top}). One "
                f"specialist covers this; a panel would add cost and dilution, not coverage."
            ),
        )

    # Best node per matched domain. A node only represents a domain it actually
    # declares -- capability_text similarity is not enough to seat someone.
    best_per_domain: list[PanelMember] = []
    for domain, sim in matched[: settings.compose_max_panel]:
        holders = [s for s in scored if domain in (s.node.get("domain_tags") or [])]
        if not holders:
            continue
        best_per_domain.append(PanelMember(scored=holders[0], domain=domain, domain_sim=sim))

    if len(best_per_domain) < settings.compose_min_domains:
        return PanelPlan(
            compose=False, matched_domains=sims[:4],
            reason=("the request spans domains this network has no declared specialist for — "
                    "routing to the single best general node instead"),
        )

    # --- The domination gate --------------------------------------------
    #
    # The lesson of Experiment 2, enforced. If one node is the best available
    # answer for *every* domain this request touches, then no other node beats
    # it at anything relevant, and composing can only dilute it. The findings
    # are unambiguous that this is not a hypothetical: it is what happened, and
    # nothing in that experiment's design checked for it in advance.
    distinct_nodes = {m.node["id"] for m in best_per_domain}
    if len(distinct_nodes) < 2:
        dominant = best_per_domain[0].name
        return PanelPlan(
            compose=False, matched_domains=sims[:4],
            reason=(
                f"'{dominant}' is the best node for every domain this request touches, so it is "
                f"not dominated by anything on the network. Composing could only dilute it — "
                f"see seam-findings.md §2, 'routing only pays when specialists are genuinely "
                f"non-dominated'."
            ),
        )

    # Deduplicate: a node that is best at two matched domains sits once, under
    # the domain it matched most strongly.
    members: list[PanelMember] = []
    seen: set = set()
    for m in best_per_domain:
        if m.node["id"] in seen:
            continue
        seen.add(m.node["id"])
        members.append(m)

    # Don't compose on a guess. If the strongest topical match is weak, the
    # network doesn't really serve this request and a confident generalist is
    # the better answer -- the same threshold the single-route path uses.
    strongest = max(m.scored.topical_score for m in members)
    if strongest < settings.routing_confidence_threshold and mode != "always":
        return PanelPlan(
            compose=False, matched_domains=sims[:4],
            reason=(
                f"low confidence — best topical score {strongest:.2f} is under the "
                f"{settings.routing_confidence_threshold:.2f} threshold. Preferring a confident "
                f"generalist over a panel of guessed specialists."
            ),
        )

    aggregator, in_panel = _pick_aggregator(scored, seen)
    if aggregator is None:
        return PanelPlan(compose=False, matched_domains=sims[:4],
                         reason="no node available to aggregate")

    domains = ", ".join(f"{m.domain} → {m.name}" for m in members)
    return PanelPlan(
        compose=True, members=members, aggregator=aggregator,
        aggregator_in_panel=in_panel, matched_domains=sims[:4],
        reason=(
            f"request spans {len(members)} domains with a different best node for each "
            f"({domains}); no single node dominates, so a panel can add what one cannot."
        ),
    )


# --- Prompts --------------------------------------------------------------

def specialist_prompt(member: PanelMember) -> str:
    """The instruction given to each panel member.

    Two things it deliberately does:

    * **Licenses declining.** A specialist asked to answer everything will
      answer everything, including the parts it is worst at, and the aggregator
      then has to guess which contributions to trust. Telling it another
      specialist is covering the rest is what makes a panel different from
      three copies of the same request.

    * **Asks for the working.** The verifier can only recompute arithmetic that
      was actually shown (`app/verify.py` refuses to guess at what an implicit
      calculation meant). So the prompt that makes verification possible and the
      verification itself are two halves of one mechanism.

    What it does *not* do is impose a terse output contract. The v0.1 terse arm
    scored 0.550 against 0.587 for the same handoff with reasoning allowed — the
    worst arm measured. Forbidding a model to think costs real accuracy.
    """
    return (
        f"You are the {member.domain} specialist on a network answering this request "
        f"together with other specialists, working in parallel.\n\n"
        f"Answer the parts of the request that fall within {member.domain}. Work carefully "
        f"and show your reasoning.\n\n"
        f"If part of the request falls outside {member.domain}, say so in one line and move "
        f"on — another specialist is covering it. Do not guess outside your domain; a "
        f"confident wrong answer is worse here than an acknowledged gap, because whoever "
        f"combines these replies cannot tell the difference.\n\n"
        f"Show the working for any calculation you do, as an explicit expression "
        f"(for example: 5920 * 0.05 / 365 = 0.81). Your arithmetic is independently "
        f"recomputed before it reaches the final answer."
    )


AGGREGATOR_PROMPT = """You are the aggregator on a network of specialist models. Several \
specialists have each answered part of one request, working in parallel and without seeing \
each other's replies.

Write the single best answer to the user's request, drawing on their replies.

- Use each specialist for what it was asked about. Where one declined a part as outside its \
domain, ignore that decline — another specialist covered it.
- Where they overlap and agree, say it once.
- Where they conflict, do not silently pick one. Resolve it by reasoning it through and \
showing why, or say plainly that the point is uncertain.
- Add what is needed to make the reply coherent, but do not introduce facts or figures that \
no specialist provided and you cannot derive from what they did.

Answer the user directly. Do not mention the specialists, the panel, or this process — the \
user asked a question, not for a report on how it was handled."""


def build_aggregation_body(
    original_body: dict[str, Any],
    question: str,
    answers: dict[str, str],
    report: VerificationReport,
    members: list[PanelMember],
    stream: bool,
) -> dict[str, Any]:
    """Assemble the aggregator's request.

    Verification goes *after* the specialist answers and is labelled as coming
    from the gateway rather than from a model. The findings showed receivers
    reproduce handed values faithfully and unrounded — so the reliable move is
    to hand over the corrected number explicitly, not to hint that something is
    wrong and hope it re-derives.
    """
    domain_by_node = {m.name: m.domain for m in members}

    parts = [f"The user asked:\n\n{question}\n"]
    for node_name, text in answers.items():
        domain = domain_by_node.get(node_name, "general")
        parts.append(f"--- {domain} specialist ({node_name}) ---\n{text}\n")

    verification = report.as_prompt_section()
    if verification:
        parts.append(f"--- {verification}\n")

    body = dict(original_body)
    body["messages"] = [
        {"role": "system", "content": AGGREGATOR_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]
    body["stream"] = stream
    return body


# --- Execution ------------------------------------------------------------

def extract_question(body: dict[str, Any]) -> str:
    messages = body.get("messages") or []
    user_turns = [str(m.get("content", "")) for m in messages if m.get("role") == "user"]
    return "\n".join(t for t in user_turns if t)


def _specialist_body(original_body: dict[str, Any], member: PanelMember) -> dict[str, Any]:
    """Prepend the specialist's role to the conversation.

    The user's own messages are passed through untouched, including any system
    message they sent. The specialist framing is inserted as a separate leading
    system turn rather than merged into theirs, so a client that set its own
    system prompt still gets it honoured.
    """
    body = dict(original_body)
    body["messages"] = ([{"role": "system", "content": specialist_prompt(member)}]
                        + list(original_body.get("messages") or []))
    body["stream"] = False
    return body


async def _call_specialist(member: PanelMember, original_body: dict[str, Any]) -> str | None:
    resp = None
    try:
        resp = await upstream.forward(member.node, _specialist_body(original_body, member), stream=False)
        if resp.status_code >= 400:
            return None
        payload = json.loads(await resp.aread())
        content = (payload.get("choices") or [{}])[0].get("message", {}).get("content")
        return content or None
    finally:
        if resp is not None:
            try:
                await resp.aclose()
                await resp.extensions["_client"].aclose()
            except (httpx.HTTPError, KeyError):
                pass


async def _ask_specialist(member: PanelMember, original_body: dict[str, Any]) -> tuple[str, str | None]:
    """Ask one panel member. Returns (node_name, answer-or-None).

    A failure is a returned None, never a raised exception: one laptop closing
    its lid mid-request must degrade the panel, not fail the user's request.
    Redundancy and quality being the same mechanism is the original
    architecture note's central claim, and this is where it has to hold.
    """
    try:
        return member.name, await asyncio.wait_for(
            _call_specialist(member, original_body),
            timeout=settings.compose_member_timeout_seconds,
        )
    except (httpx.HTTPError, asyncio.TimeoutError, ValueError, KeyError, IndexError):
        return member.name, None


async def run_panel(plan: PanelPlan, original_body: dict[str, Any]) -> dict[str, str]:
    """Ask every panel member at once. Returns node name -> answer, successes only.

    Parallel is not an optimisation here, it is the finding: the sequential arm
    of Experiment 2 was the only composition topology that made output measurably
    *worse* than the baseline, because the second model inherited the first's
    framing. Fan-out has no seam to inherit across.
    """
    results = await asyncio.gather(
        *(_ask_specialist(m, original_body) for m in plan.members)
    )
    return {name: answer for name, answer in results if answer}
