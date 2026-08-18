import json
import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from app import compose, db, embedder, upstream
from app.compose import PanelPlan
from app.config import settings
from app.router import ScoredNode, best_matched_domain, score_nodes
from app.verify import VerificationReport, verify

router = APIRouter()


def _extract_routing_text(body: dict[str, Any]) -> str:
    messages = body.get("messages") or []
    user_turns = [m.get("content", "") for m in messages if m.get("role") == "user"]
    text = "\n".join(str(t) for t in user_turns if t)
    return text or str(body)


async def _fetch_healthy_nodes() -> list[dict]:
    async with db.pool().acquire() as conn:
        rows = await conn.fetch("select * from nodes where healthy = true")
    return [dict(r) for r in rows]


async def _record_decision(
    request_embed: list[float],
    chosen: ScoredNode | None,
    runner_up: ScoredNode | None,
    latency_ms: int,
    ok: bool,
    matched_domain: str | None = None,
    topology: str = "single",
    panel: list[uuid.UUID] | None = None,
    aggregator_node: uuid.UUID | None = None,
    compose_reason: dict | None = None,
    report: VerificationReport | None = None,
) -> None:
    """Log what happened, in enough detail to argue with later.

    Composition and single-routing land in the same table, distinguished by
    `topology`. That is deliberate: whether composing actually beats routing to
    the best single node is the open question this version exists to answer, and
    the two arms have to be comparable in one query to answer it.
    """
    async with db.pool().acquire() as conn:
        await conn.execute(
            """
            insert into decisions
                (request_embed, chosen_node, score, runner_up, latency_ms, ok,
                 matched_domain, topology, panel, aggregator_node, compose_reason,
                 checks_run, checks_failed, disagreements)
            values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """,
            request_embed,
            chosen.node["id"] if chosen else None,
            chosen.score if chosen else None,
            runner_up.node["id"] if runner_up else None,
            latency_ms,
            ok,
            matched_domain,
            topology,
            panel,
            aggregator_node,
            json.dumps(compose_reason) if compose_reason else None,
            len(report.checks) if report else None,
            len(report.failed) if report else None,
            len(report.disagreements) if report else None,
        )


async def _update_latency(node_id, latency_ms: int) -> None:
    # Simple rolling average (equal weight to history and this sample).
    async with db.pool().acquire() as conn:
        await conn.execute(
            """
            update nodes
            set avg_latency_ms = case
                when avg_latency_ms = 0 then $2
                else (avg_latency_ms + $2) / 2
            end
            where id = $1
            """,
            node_id, latency_ms,
        )


def _header_safe(value: str, limit: int = 400) -> str:
    """HTTP headers are latin-1 and must not contain newlines.

    The composition reason is written for humans and can be long; the full
    version is on the decisions row. This is the version you can see with
    `curl -i`, which is where most people will actually look.
    """
    flat = " ".join(str(value).split())
    if len(flat) > limit:
        flat = flat[: limit - 1] + "…"
    return flat.encode("ascii", "replace").decode("ascii")


# --- Synthesised OpenAI envelopes ----------------------------------------
#
# The composition path has already consumed each specialist's response, so when
# it returns one of those answers directly there is no upstream response left to
# proxy. Rather than spend a second inference re-asking a node for text we are
# already holding, we wrap the text we have in the envelope the client expects.

def _completion_envelope(content: str, model: str) -> dict:
    return {
        "id": f"chatcmpl-common-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        # No usage figures invented here. A client that budgets on token counts
        # would rather see the field absent than see a number we made up.
    }


def _sse_from_text(content: str, model: str):
    """Emit a complete answer as a one-chunk SSE stream.

    Clients that asked for `stream: true` get the shape they asked for. It
    arrives in a single chunk because the text was already complete before this
    was called -- dribbling it out word by word to simulate generation would be
    theatre, and would misreport where the latency actually went.
    """
    created = int(time.time())
    chunk_id = f"chatcmpl-common-{uuid.uuid4().hex[:24]}"

    async def gen():
        first = {
            "id": chunk_id, "object": "chat.completion.chunk", "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": content},
                         "finish_reason": None}],
        }
        yield f"data: {json.dumps(first)}\n\n".encode()
        last = {
            "id": chunk_id, "object": "chat.completion.chunk", "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(last)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return gen()


def _proxy_response(resp: httpx.Response, headers: dict[str, str], stream: bool):
    if stream:
        async def body_iter():
            client = resp.extensions["_client"]
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

        return StreamingResponse(
            body_iter(),
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "text/event-stream"),
            headers=headers,
        )
    return None


# --- Composition path -----------------------------------------------------

async def _handle_composed(
    plan: PanelPlan,
    body: dict[str, Any],
    stream: bool,
    request_embed: list[float],
    matched_domain: str | None,
    scored: list[ScoredNode],
    start: float,
):
    """Fan out to the panel, verify, aggregate.

    Returns None if the panel produced nothing usable, so the caller can fall
    back to ordinary single-node routing. Composition must never be able to turn
    a request that v0.1 would have answered into a failure — that would trade a
    measured, working behaviour for an unproven one.
    """
    answers = await compose.run_panel(plan, body)
    panel_ids = [m.node["id"] for m in plan.members]

    if not answers:
        return None

    base_headers = {
        "X-Common-Panel": _header_safe(", ".join(answers.keys())),
        "X-Common-Compose-Reason": _header_safe(plan.reason),
    }

    # One survivor: hand it back as-is. Aggregating a single answer adds a
    # seam -- the exact thing measured to cost +0.144 -- and buys nothing,
    # since there is no second opinion to reconcile it against. Recorded as
    # 'degraded' so it never gets counted as evidence for composition.
    if len(answers) == 1:
        node_name, content = next(iter(answers.items()))
        member = next(m for m in plan.members if m.name == node_name)
        latency_ms = int((time.monotonic() - start) * 1000)
        await _update_latency(member.node["id"], latency_ms)
        await _record_decision(
            request_embed, member.scored, None, latency_ms, True, matched_domain,
            topology="degraded", panel=panel_ids, compose_reason={
                **plan.as_dict(),
                "degraded": f"only {node_name} returned; passed through unaggregated",
            },
        )
        headers = {
            **base_headers,
            "X-Common-Node": node_name,
            "X-Common-Topology": "degraded",
            "X-Common-Score": f"{member.scored.score:.4f}",
        }
        model_name = member.node["model_name"]
        if stream:
            return StreamingResponse(_sse_from_text(content, model_name),
                                     media_type="text/event-stream", headers=headers)
        return JSONResponse(_completion_envelope(content, model_name), headers=headers)

    # The real path: two or more specialists, verified, then synthesised.
    report = verify(answers)
    agg_body = compose.build_aggregation_body(
        body, compose.extract_question(body), answers, report, plan.members, stream,
    )

    resp = None
    try:
        resp = await upstream.forward(plan.aggregator, agg_body, stream)
        if resp.status_code >= 400:
            raise httpx.HTTPStatusError("aggregator failed", request=resp.request, response=resp)
    except (httpx.HTTPError, httpx.HTTPStatusError):
        if resp is not None:
            try:
                await resp.aclose()
                await resp.extensions["_client"].aclose()
            except (httpx.HTTPError, KeyError):
                pass
        # The panel answered but the aggregator did not. Returning the
        # strongest specialist's answer is strictly better than failing:
        # the user gets a real answer from the node that would have won under
        # v0.1 routing anyway.
        best = max(plan.members, key=lambda m: m.scored.score)
        content = answers.get(best.name) or next(iter(answers.values()))
        latency_ms = int((time.monotonic() - start) * 1000)
        await _record_decision(
            request_embed, best.scored, None, latency_ms, True, matched_domain,
            topology="degraded", panel=panel_ids, compose_reason={
                **plan.as_dict(), "degraded": "aggregator unreachable; returned strongest panel member",
            }, report=report,
        )
        headers = {**base_headers, "X-Common-Node": best.name, "X-Common-Topology": "degraded"}
        if stream:
            return StreamingResponse(_sse_from_text(content, best.node["model_name"]),
                                     media_type="text/event-stream", headers=headers)
        return JSONResponse(_completion_envelope(content, best.node["model_name"]), headers=headers)

    latency_ms = int((time.monotonic() - start) * 1000)
    await _update_latency(plan.aggregator["id"], latency_ms)
    await _record_decision(
        request_embed, None, None, latency_ms, True, matched_domain,
        topology="panel", panel=panel_ids, aggregator_node=plan.aggregator["id"],
        compose_reason=plan.as_dict(), report=report,
    )

    headers = {
        **base_headers,
        "X-Common-Topology": "panel",
        "X-Common-Node": _header_safe(plan.aggregator["name"]),
        "X-Common-Aggregator": _header_safe(plan.aggregator["name"]),
        # The verification counters are in the response, not just the log,
        # because a check nobody can see is a check nobody audits.
        "X-Common-Checks": f"{len(report.checks)}",
        "X-Common-Checks-Failed": f"{len(report.failed)}",
        "X-Common-Disagreements": f"{len(report.disagreements)}",
    }

    streamed = _proxy_response(resp, headers, stream)
    if streamed is not None:
        return streamed

    content = await resp.aread()
    await resp.aclose()
    await resp.extensions["_client"].aclose()
    return Response(
        content=content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
        headers=headers,
    )


# --- Entry point ----------------------------------------------------------

@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    stream = bool(body.get("stream", False))

    nodes = await _fetch_healthy_nodes()
    if not nodes:
        raise HTTPException(status_code=503, detail="no healthy nodes available")

    routing_text = _extract_routing_text(body)
    request_embed = embedder.embed(routing_text)
    region_hint = request.headers.get("X-Common-Region")
    forced_node_name = request.headers.get("X-Common-Node")
    matched_domain = best_matched_domain(nodes, request_embed)

    # Per-request override of COMPOSE_MODE, so a client can A/B the two
    # topologies against each other without restarting the gateway. This is
    # what testing/compose-test drives.
    compose_header = (request.headers.get("X-Common-Compose") or "").strip().lower()

    if forced_node_name:
        matches = [n for n in nodes if n["name"] == forced_node_name]
        if not matches:
            raise HTTPException(status_code=404, detail=f"node '{forced_node_name}' not found or not currently healthy")
        # No fallback candidate -- if you asked for this node specifically,
        # a failure should surface as a failure, not silently reroute.
        candidates = [ScoredNode(node=matches[0], score=1.0, sim=0.0, cost_term=0.0, lat_term=0.0, region_term=0.0)]
        scored = candidates
        plan = PanelPlan(compose=False, reason="a specific node was requested")
    else:
        scored = score_nodes(nodes, request_embed, region_hint)

        plan = compose.plan_panel(
            scored, request_embed,
            mode_override=compose_header if compose_header in {"auto", "always", "never"} else None,
            # The quantitative check reads the request itself, not its embedding.
            request_text=routing_text,
        )

        if plan.compose:
            start = time.monotonic()
            composed = await _handle_composed(
                plan, body, stream, request_embed, matched_domain, scored, start,
            )
            if composed is not None:
                return composed
            # Every panel member failed. Fall through to single-node routing
            # rather than failing the request.
            plan = PanelPlan(compose=False,
                             reason="panel selected but no member answered — fell back to single routing")

        primary, backup = scored[0], (scored[1] if len(scored) > 1 else None)

        # Low confidence on a narrow specialist match loses to frontier
        # instantly -- prefer a confident generalist over a guessed specialist.
        if scored[0].topical_score < settings.routing_confidence_threshold:
            generalists = [s for s in scored if "general" in (s.node.get("domain_tags") or [])]
            if generalists and generalists[0].node["id"] != scored[0].node["id"]:
                primary, backup = generalists[0], scored[0]

        candidates = [primary] + ([backup] if backup else [])

    last_error: Exception | None = None

    for attempt, candidate in enumerate(candidates):
        start = time.monotonic()
        try:
            resp = await upstream.forward(candidate.node, body, stream)
            latency_ms = int((time.monotonic() - start) * 1000)
            if resp.status_code >= 500:
                raise httpx.HTTPStatusError("upstream error", request=resp.request, response=resp)

            await _update_latency(candidate.node["id"], latency_ms)
            runner_up = next((c for i, c in enumerate(candidates) if i != attempt), None)
            await _record_decision(
                request_embed, candidate, runner_up, latency_ms, True, matched_domain,
                topology="single", compose_reason={"compose": False, "reason": plan.reason},
            )

            headers = {
                "X-Common-Node": candidate.node["name"],
                "X-Common-Score": "forced" if forced_node_name else f"{candidate.score:.4f}",
                "X-Common-Topology": "single",
                # Why this request was *not* composed. The negative case is the
                # one worth explaining -- "why did only one node answer this?"
                # is the question the composition feature invites.
                "X-Common-Compose-Reason": _header_safe(plan.reason),
            }

            streamed = _proxy_response(resp, headers, stream)
            if streamed is not None:
                return streamed

            content = await resp.aread()
            await resp.aclose()
            await resp.extensions["_client"].aclose()
            return Response(
                content=content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
                headers=headers,
            )
        except (httpx.HTTPError, httpx.HTTPStatusError) as exc:
            last_error = exc
            latency_ms = int((time.monotonic() - start) * 1000)
            if attempt == len(candidates) - 1:
                await _record_decision(request_embed, None, None, latency_ms, False, matched_domain)

    raise HTTPException(status_code=502, detail=f"all candidate nodes failed: {last_error}")


@router.get("/v1/models")
async def list_models():
    nodes = await _fetch_healthy_nodes()
    seen = {}
    for n in nodes:
        seen[n["model_name"]] = n
    data = [
        {"id": model_name, "object": "model", "owned_by": n.get("operator") or "common-network"}
        for model_name, n in seen.items()
    ]
    return JSONResponse({"object": "list", "data": data})
