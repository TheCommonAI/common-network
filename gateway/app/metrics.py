"""What the network costs, measured rather than asserted.

Two halves, and they are deliberately in one file because they share a
definition of what each column means.

**Recording.** Every request the gateway answers writes one row to
`request_metrics` (see migrations/012): the gateway's own routing time, the
wait for the node's first token, and the wait for the last byte. The write is
detached from the request that caused it, for the reason spelled out on
`spawn`.

**Reading.** `/metrics/summary` aggregates a rolling window into percentile
blocks, and `/metrics/recent` returns the rows themselves. Both are the same
shape the dashboard reads.

**A note on honesty, because this file is where it would be easiest to lose.**
The latency figures the gateway measures are its own and are what they claim to
be. The `node_*` figures -- model load, prompt evaluation, generation, token
counts -- are *reported by the node*, and a node is a stranger's machine that
registered itself on a permissionless registry. Nothing here verifies them. A
node that wanted to look fast could send whatever it liked. They are collected
anyway because they are the only account of model load and throughput that
exists anywhere in the system, and a labelled claim beats no measurement -- but
`node_reported` is a column rather than an assumption, `/metrics/summary`
reports its own coverage, and every surface that renders one of these numbers
says where it came from.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app import db

router = APIRouter()


# --- Recording -------------------------------------------------------------

# Detached metric writes. A strong reference is required or the event loop is
# free to garbage-collect a task that nothing awaits, which would drop exactly
# the writes nobody is waiting on -- all of them.
_pending: set[asyncio.Task] = set()

# No single request may have taken longer than this. A node is free to send
# whatever it likes in the headers below, and one node claiming a load time of
# two billion milliseconds would move a p95 on its own. Anything past this is
# discarded rather than clamped, so it cannot enter an aggregate at all.
_MAX_MS = 24 * 60 * 60 * 1000   # one day, in milliseconds
_MAX_TOKENS = 10_000_000


def spawn(coro) -> None:
    """Run a metrics write on its own task.

    Deliberately detached from the request. The write must survive the case
    that is most worth recording: a client that gave up. When a request is
    cancelled -- client hung up, or the gateway's own read timeout fired -- the
    task running the streaming generator is cancelled too, and an `await` in
    its `finally` would raise CancelledError before reaching the database.
    The slowest requests would be the ones missing from the distribution, and
    the missing tail is the tail anyone actually cares about.

    A task created here is not awaited by anyone, so failures have nowhere to
    propagate. `_record` catches its own exceptions for that reason.
    """
    task = asyncio.create_task(coro)
    _pending.add(task)
    task.add_done_callback(_pending.discard)


def _int_header(headers: httpx.Headers | dict, name: str,
                maximum: int = _MAX_MS) -> int | None:
    """One node-reported integer, or None if the node did not send a usable one.

    This parses input from an unauthenticated stranger's machine, so it is
    strict on purpose. `isdigit` rather than a try/except on `int()`: the latter
    accepts "+5" and "5_0", and a value that parses is not the same as a value
    that was meant.

    `isascii` as well as `isdigit`, and the reason is worth stating because it
    is not obvious: `isdigit` is true of Arabic-Indic and other Unicode digits,
    and `int()` converts them, so "٥" was being read as 5. Nothing legitimate
    sends those in a custom header -- HTTP field values are ASCII by
    construction -- so a non-ASCII digit is malformed by definition and is
    dropped rather than decoded.

    Whitespace is stripped rather than rejected. Every HTTP parser removes
    optional whitespace around a field value before this function sees it, so
    the strip is belt-and-braces rather than a loosening -- there is no way for
    a node to smuggle a padded number past `httpx.Headers` to begin with, and
    rejecting padding here would reject well-formed input.

    Anything non-numeric, negative, or larger than `maximum` is treated as
    absent -- a lie is dropped rather than trusted, and a typo is dropped rather
    than corrected into a wrong number.
    """
    raw = headers.get(name)
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or len(text) > 12 or not (text.isascii() and text.isdigit()):
        return None
    value = int(text)
    return value if value <= maximum else None


def _bool_header(headers: httpx.Headers | dict, name: str) -> bool | None:
    """One node-reported boolean. Only the two spellings, nothing clever.

    Not `bool(raw)`: every non-empty string is truthy, so a node sending "false"
    would have been recorded as cold.
    """
    raw = headers.get(name)
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    return None


def node_report(headers: httpx.Headers | dict | None) -> dict[str, Any]:
    """The node-reported timing block, read off a node's response headers.

    Header names rather than a body field because the response body is relayed
    to the client byte for byte and must not be touched (see `_proxy_chat` in
    join/worker.py) -- and because the numbers are only known once generation
    has finished, by which time the headers are sent but the body is still
    streaming.

    Everything here is unverified. See the module docstring.

    Always returns the same eight keys, with None wherever the node sent
    nothing usable. The two ways of having nothing to report -- no headers at
    all, and headers carrying nothing that parses -- therefore produce the same
    shape, so a caller can index the result rather than `.get` it without the
    no-headers path raising KeyError.
    """
    headers = headers or {}
    load = _int_header(headers, "X-Common-Node-Load-Ms")
    prompt_eval = _int_header(headers, "X-Common-Node-Prompt-Eval-Ms")
    eval_ms = _int_header(headers, "X-Common-Node-Eval-Ms")
    prompt_tokens = _int_header(headers, "X-Common-Node-Prompt-Tokens", _MAX_TOKENS)
    cached = _int_header(headers, "X-Common-Node-Prompt-Cached-Tokens", _MAX_TOKENS)
    completion = _int_header(headers, "X-Common-Node-Completion-Tokens", _MAX_TOKENS)
    cold = _bool_header(headers, "X-Common-Node-Cold")

    reported = any(v is not None for v in
                   (load, prompt_eval, eval_ms, prompt_tokens, cached, completion, cold))
    return {
        "node_reported": reported,
        "load_ms": load,
        "prompt_eval_ms": prompt_eval,
        "eval_ms": eval_ms,
        "prompt_tokens": prompt_tokens,
        "prompt_cached_tokens": cached,
        "completion_tokens": completion,
        "cold": cold,
    }


async def _record(row: dict[str, Any]) -> None:
    """Insert one metrics row. Never raises into anything.

    Called from a detached task, so an exception here has no caller to surface
    in; it would reach the event loop's default handler and print a traceback
    that says nothing about which request it belonged to. Printed explicitly
    instead, and the request itself is unaffected either way -- a network that
    fails to answer because it could not write a latency row would have its
    priorities badly wrong.
    """
    try:
        async with db.pool().acquire() as conn:
            await conn.execute(
                """
                insert into request_metrics
                    (node_id, node_name, model_name, topology, stream,
                     route_ms, ttft_ms, total_ms,
                     node_reported, load_ms, prompt_eval_ms, eval_ms,
                     prompt_tokens, prompt_cached_tokens, completion_tokens, cold,
                     ok, error)
                values ($1, $2, $3, $4, $5, $6, $7, $8,
                        $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
                """,
                row.get("node_id"),
                row.get("node_name"),
                row.get("model_name"),
                row.get("topology", "single"),
                bool(row.get("stream", False)),
                row.get("route_ms"),
                row.get("ttft_ms"),
                row.get("total_ms"),
                bool(row.get("node_reported", False)),
                row.get("load_ms"),
                row.get("prompt_eval_ms"),
                row.get("eval_ms"),
                row.get("prompt_tokens"),
                row.get("prompt_cached_tokens"),
                row.get("completion_tokens"),
                row.get("cold"),
                bool(row.get("ok", True)),
                (row.get("error") or None) and str(row["error"])[:200],
            )
    except Exception as exc:  # noqa: BLE001 -- see docstring
        print(f"metrics: could not record request metrics "
              f"({type(exc).__name__}: {exc})", flush=True)


def record_request(**row: Any) -> None:
    """Queue one metrics row. Returns immediately; the request proceeds.

    The synchronous front door to `_record` -- callers in the request path
    should not have to know that the write is detached, and should not be able
    to accidentally await it.
    """
    spawn(_record(dict(row)))


# --- Reading ---------------------------------------------------------------

# asyncpg hands back Decimal for numeric and int for bigint; JSON needs neither.
# percentile_cont comes back as float, min/max/count as int or Decimal depending
# on the input type, so every numeric that leaves this module goes through here.
def _num(value, cast):
    if value is None:
        return None
    return cast(value)


def _block(prefix: str, row, keys: dict[str, str]) -> dict[str, Any]:
    return {label: _num(row.get(f"{prefix}{suffix}"), cast)
            for label, (suffix, cast) in keys.items()}


_LATENCY_KEYS = {
    "p50": ("_p50", float),
    "p95": ("_p95", float),
    "min": ("_min", int),
    "max": ("_max", int),
}


@router.get("/metrics/summary")
async def metrics_summary(
    window_hours: int = Query(default=24, ge=1, le=8760),
    topology: str | None = Query(default=None, description="Filter to 'single', 'panel' or 'degraded'"),
):
    """Everything the dashboard needs about latency, in one call.

    Percentiles rather than means, throughout. A mean latency across a network
    of donated laptops is close to meaningless: the distribution is a fast warm
    cluster and a slow cold tail, and the mean sits in the gap between them,
    describing a request that never happens. p50 says what a typical user gets
    and p95 says what the unlucky one gets, and the two together are the whole
    story. (The old `nodes.avg_latency_ms` is a running `(avg + sample) / 2`,
    which is neither a mean nor a median -- it is fixed-weight and biased toward
    whatever arrived most recently. It is left alone because routing scores on
    it, and changing routing is not what this endpoint is for.)

    `latency_ms.ttft` covers streaming requests only, and not by filtering: a
    non-streaming request has no observable first token at all, so its
    `ttft_ms` is null and `percentile_cont` skips it. `latency_ms.total` is the
    one figure that means the same thing in both modes -- how long until the
    user has the whole answer -- and is split by mode beneath it, since that
    comparison is the entire case for streaming.

    `node_reports.coverage` is the fraction of requests a node actually
    reported on. It is returned rather than implied so that a summary showing
    model load for 3% of traffic cannot be read as covering all of it.
    """
    where = ["created_at > now() - make_interval(hours => $1::int)"]
    args: list[Any] = [window_hours]
    if topology:
        args.append(topology)
        where.append(f"topology = ${len(args)}")
    clause = " and ".join(where)

    async with db.pool().acquire() as conn:
        row = await conn.fetchrow(
            f"""
            select
              count(*)                                              as requests,
              count(*) filter (where ok)                            as succeeded,
              count(*) filter (where not ok)                        as failed,
              count(*) filter (where stream)                        as streaming,
              count(*) filter (where not stream)                    as non_streaming,
              count(*) filter (where node_reported)                 as reported,
              count(*) filter (where cold)                          as cold_count,
              count(*) filter (where cold is false)                 as warm_count,

              percentile_cont(0.5)  within group (order by ttft_ms)  as ttft_p50,
              percentile_cont(0.95) within group (order by ttft_ms)  as ttft_p95,
              min(ttft_ms) as ttft_min, max(ttft_ms) as ttft_max,

              percentile_cont(0.5)  within group (order by total_ms) as total_p50,
              percentile_cont(0.95) within group (order by total_ms) as total_p95,
              min(total_ms) as total_min, max(total_ms) as total_max,

              percentile_cont(0.5)  within group (order by route_ms) as route_p50,
              percentile_cont(0.95) within group (order by route_ms) as route_p95,
              min(route_ms) as route_min, max(route_ms) as route_max,

              -- The ttft block above is streaming-only already: ttft_ms is null
              -- on non-streaming rows, and percentile_cont skips nulls. Total
              -- is the one figure that means the same thing in both modes --
              -- how long until the user has the whole answer -- so it is the
              -- one worth splitting, to see whether streaming actually helps.
              percentile_cont(0.5)  within group (order by total_ms)
                filter (where stream)                                as total_stream_p50,
              percentile_cont(0.5)  within group (order by total_ms)
                filter (where not stream)                            as total_nonstream_p50,

              percentile_cont(0.5)  within group (order by load_ms)  as load_p50,
              percentile_cont(0.95) within group (order by load_ms)  as load_p95,
              percentile_cont(0.5)  within group (order by load_ms)
                filter (where cold)                                  as cold_load_p50,
              percentile_cont(0.95) within group (order by load_ms)
                filter (where cold)                                  as cold_load_p95,
              -- Warm requests should be near zero. A warm load time that is not
              -- means the node is reporting its keep_alive window, not a load.
              percentile_cont(0.5)  within group (order by load_ms)
                filter (where cold is false)                         as warm_load_p50,

              percentile_cont(0.5)  within group (order by prompt_eval_ms) as prompt_eval_p50,
              percentile_cont(0.5)  within group (order by eval_ms)        as eval_p50,

              -- Tokens per second, from the node's own two numbers. Guarded on
              -- eval_ms > 0 because the case expression would otherwise divide
              -- by zero, and a node reporting tokens but no duration would drop
              -- the whole query rather than one sample.
              percentile_cont(0.5) within group (order by
                case when eval_ms > 0
                     then completion_tokens::numeric * 1000 / eval_ms end) as tps_p50,
              percentile_cont(0.95) within group (order by
                case when eval_ms > 0
                     then completion_tokens::numeric * 1000 / eval_ms end) as tps_p95
            from request_metrics
            where {clause}
            """,
            *args,
        )

        by_node = await conn.fetch(
            f"""
            select coalesce(node_name, 'unknown') as node_name,
                   model_name,
                   count(*)                              as requests,
                   count(*) filter (where ok)            as succeeded,
                   count(*) filter (where node_reported) as reported,
                   count(*) filter (where cold)          as cold_count,
                   count(*) filter (where cold is false) as warm_count,
                   percentile_cont(0.5)  within group (order by ttft_ms)  as ttft_p50,
                   percentile_cont(0.95) within group (order by ttft_ms)  as ttft_p95,
                   percentile_cont(0.5)  within group (order by total_ms) as total_p50,
                   percentile_cont(0.5)  within group (order by route_ms) as route_p50,
                   percentile_cont(0.5)  within group (order by load_ms)  as load_p50,
                   percentile_cont(0.5)  within group (order by
                     case when eval_ms > 0
                          then completion_tokens::numeric * 1000 / eval_ms end) as tps_p50
            from request_metrics
            where {clause}
            group by node_name, model_name
            order by count(*) desc
            limit 50
            """,
            *args,
        )

    requests = int(row["requests"] or 0)
    reported = int(row["reported"] or 0)
    cold_count = int(row["cold_count"] or 0)
    warm_count = int(row["warm_count"] or 0)
    stream_count = int(row["streaming"] or 0)

    return JSONResponse({
        "window_hours": window_hours,
        "topology_filter": topology,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "requests": requests,
            "succeeded": int(row["succeeded"] or 0),
            "failed": int(row["failed"] or 0),
            "streaming": stream_count,
            "non_streaming": int(row["non_streaming"] or 0),
        },
        "latency_ms": {
            "ttft": _block("ttft", row, _LATENCY_KEYS),
            "total": _block("total", row, _LATENCY_KEYS),
            "routing": _block("route", row, _LATENCY_KEYS),
            "total_by_mode": {
                "streaming_p50": _num(row["total_stream_p50"], float),
                "non_streaming_p50": _num(row["total_nonstream_p50"], float),
            },
        },
        # Node-reported and unverified. `coverage` is the fraction of requests
        # this block describes; everywhere else on this response is measured.
        "node_reports": {
            "coverage": round(reported / requests, 3) if requests else 0.0,
            "reported": reported,
            "model_load_ms": {
                "p50": _num(row["load_p50"], float),
                "p95": _num(row["load_p95"], float),
                "cold_p50": _num(row["cold_load_p50"], float),
                "cold_p95": _num(row["cold_load_p95"], float),
                # Near zero on a healthy fleet: Ollama keeps the model resident.
                "warm_p50": _num(row["warm_load_p50"], float),
            },
            "prompt_eval_ms": {"p50": _num(row["prompt_eval_p50"], float)},
            "generation_ms": {"p50": _num(row["eval_p50"], float)},
            "tokens_per_second": {
                "p50": _num(row["tps_p50"], float),
                "p95": _num(row["tps_p95"], float),
            },
            "cold_fraction": (
                round(cold_count / (cold_count + warm_count), 3)
                if (cold_count + warm_count) else None
            ),
            "cold_requests": cold_count,
            "warm_requests": warm_count,
        },
        "by_node": [
            {
                "node_name": r["node_name"],
                "model_name": r["model_name"],
                "requests": int(r["requests"] or 0),
                "succeeded": int(r["succeeded"] or 0),
                "reported": int(r["reported"] or 0),
                "cold_requests": int(r["cold_count"] or 0),
                "warm_requests": int(r["warm_count"] or 0),
                "ttft_p50_ms": _num(r["ttft_p50"], float),
                "ttft_p95_ms": _num(r["ttft_p95"], float),
                "total_p50_ms": _num(r["total_p50"], float),
                "route_p50_ms": _num(r["route_p50"], float),
                "load_p50_ms": _num(r["load_p50"], float),
                "tokens_per_second_p50": _num(r["tps_p50"], float),
            }
            for r in by_node
        ],
        "caveat": (
            "Everything under latency_ms is measured by this gateway. Everything "
            "under node_reports and the load/tokens columns of by_node is reported "
            "by the nodes themselves, which are strangers' machines on a "
            "permissionless registry -- unverified, and absent (null) wherever a "
            "node did not send it. Check node_reports.coverage before quoting any "
            "of it. ttft is the wait for the node's first byte, which for a "
            "streaming request is the first token: Ollama sends no response "
            "headers until the model is in memory, so ttft includes model load."
        ),
    })


@router.get("/metrics/recent")
async def metrics_recent(
    limit: int = Query(default=100, ge=1, le=1000),
    only_reported: bool = Query(default=False, description="Only rows a node reported timings for"),
):
    """The rows themselves, newest first.

    The distribution endpoints answer "how is the network doing". This answers
    "what happened to that one request", which is a different question and the
    one you have when a p95 is bad and you want to know which node caused it.
    """
    async with db.pool().acquire() as conn:
        rows = await conn.fetch(
            """
            select created_at, node_name, model_name, topology, stream,
                   route_ms, ttft_ms, total_ms,
                   node_reported, load_ms, prompt_eval_ms, eval_ms,
                   prompt_tokens, prompt_cached_tokens, completion_tokens, cold,
                   ok, error
            from request_metrics
            where (not $2::boolean or node_reported)
            order by created_at desc
            limit $1
            """,
            limit, only_reported,
        )

    out = []
    for r in rows:
        load_ms, eval_ms, completion = r["load_ms"], r["eval_ms"], r["completion_tokens"]
        out.append({
            "created_at": r["created_at"].isoformat(),
            "node_name": r["node_name"],
            "model_name": r["model_name"],
            "topology": r["topology"],
            "stream": r["stream"],
            "route_ms": r["route_ms"],
            "ttft_ms": r["ttft_ms"],
            "total_ms": r["total_ms"],
            "node_reported": r["node_reported"],
            "load_ms": load_ms,
            "prompt_eval_ms": r["prompt_eval_ms"],
            "eval_ms": eval_ms,
            "prompt_tokens": r["prompt_tokens"],
            "prompt_cached_tokens": r["prompt_cached_tokens"],
            "completion_tokens": completion,
            "cold": r["cold"],
            # Derived here rather than stored: it is a ratio of two reported
            # numbers, and storing it would let the three drift apart when a
            # node sends one and not the other.
            "tokens_per_second": (
                round(float(completion) * 1000 / eval_ms, 1)
                if eval_ms and completion else None
            ),
            "ok": r["ok"],
            "error": r["error"],
        })
    return out
