"""Bounded client intake, private operator access and authenticated node probes.

Never log request bodies here. The transport necessarily sees a source address;
operators must disable proxy/access-log IP retention for these intake routes.
"""
import asyncio
import hashlib
import json
import math
import re
import secrets
import time
from uuid import UUID

import httpx
from fastapi import APIRouter, Header, HTTPException, Request

from app import db
from app.admin import _require_admin
from app.health import _check_one

router = APIRouter()
EVENTS = set('app_started app_shutdown app_crashed renderer_failed ollama_detected ollama_start_failed gateway_reachable gateway_unreachable tunnel_starting tunnel_connected tunnel_failed registration_started registration_completed registration_failed worker_ready worker_disconnected worker_reconnecting job_received job_started first_token job_completed job_failed chat_started chat_completed chat_failed contribution_paused setup_started setup_completed setup_failed model_checked benchmark_completed update_failed'.split())
CODES = set('OLLAMA_NOT_INSTALLED OLLAMA_NOT_RUNNING MODEL_NOT_INSTALLED MODEL_HEALTH_FAILED GATEWAY_UNREACHABLE GATEWAY_AUTH_FAILED GATEWAY_MALFORMED TUNNEL_RATE_LIMITED TUNNEL_START_FAILED TUNNEL_TIMEOUT REGISTRATION_FAILED REGISTRATION_EXPIRED WORKER_UNREACHABLE INFERENCE_TIMEOUT INFERENCE_FAILED INFERENCE_CANCELLED UPDATE_FAILED SECURE_STORAGE_UNAVAILABLE REPORT_FAILED APP_CRASHED UNKNOWN'.split())
NUMBERS = set('timestamp durationMs requestStart requestDispatched inferenceStart queueDelayMs ttftMs generationMs completionTokens promptTokens totalTokens tokensPerSecond totalMs httpStatus cores ramGB jobsCompleted jobsFailed tokensGenerated inferenceMs contributionMs uptimeMs ttftCount ttftSum rateCount rateSum startedAt checkedAt active'.split())
BOOLS = set('success online installed accepting unexpectedExit available'.split())
ENUMS = {
    'platform': ['win32', 'darwin', 'linux'], 'arch': ['x64', 'arm64', 'ia32'],
    'target': ['local', 'network'], 'event': EVENTS, 'code': CODES,
    'status': 'healthy unhealthy ready missing connected disconnected active expired reachable unreachable paused unknown pending running passed failed skipped'.split() + ['accepting jobs'],
    'category': 'network ollama model tunnel registration inference update application'.split(),
    'stage': 'application network gateway ollama model localInference helper publicEndpoint registration worker networkInference'.split(),
    'causeType': 'Error TypeError SyntaxError AbortError TimeoutError RangeError'.split(),
    'causeCode': 'ECONNREFUSED ENOTFOUND ECONNRESET ETIMEDOUT ENOENT EPIPE UND_ERR_CONNECT_TIMEOUT'.split(),
}
CONTAINERS = set('application computer ollama models gateway node tunnel worker contribution recentEvents performance update health preferences events statistics session today allTime activity diagnostics error benchmark'.split())
MODEL = re.compile(r'^(?:llama[0-9.]*|qwen[0-9.]*|gemma[0-9.]*|phi[0-9.]*|mistral|deepseek-r1)(?::[0-9]+(?:\.[0-9]+)?[bm])?$')

# Why an answer was reported. A closed list, deliberately: the point of the
# field is to be countable, and free text is already available in `text`.
# `category` is a different axis and stays 'chat' for these.
REASONS = set('made-up wrong off-topic truncated refused format slow other'.split())

# An exchange is the question and the answer the user chose to attach. Caps are
# generous enough for a real 7B response (a long one is ~8k characters, well
# under 32KiB) and small enough that a report stays a report rather than a
# transcript dump. `read_body` already refuses a body over 128KiB, so these
# bounds exist to keep one exchange from crowding out the other fields.
EXCHANGE_MAX_BYTES = 65536
QUESTION_MAX_CHARS = 4000
ANSWER_MAX_BYTES = 32768
ROUTING_STR_MAX = 200
# The numeric half of `routing`. Finite and bounded, like NUMBERS above.
ROUTING_NUMBERS = set('score candidates margin_over_runner_up checks_run checks_failed '
                      'disagreements routing_ms ttft_ms total_ms'.split())
# The string half. `panel` and `compose_reason` are handled separately below.
ROUTING_STRINGS = set('node topology aggregator'.split())


def _valid_routing(value):
    """Keep the routing metadata the CLI read off the response headers.

    This is metadata the network already recorded on the `decisions` row — it is
    not new text, and it is what lets an operator tell "the router picked the
    wrong node" apart from "the right node answered badly" *before* opening the
    decision. Unknown sub-keys are dropped rather than stored, so a client
    cannot use this to smuggle arbitrary content into the column.
    """
    if not isinstance(value, dict):
        raise HTTPException(400, 'Invalid routing')
    out = {}
    for k, v in value.items():
        if k in ROUTING_NUMBERS:
            if v is not None and (type(v) in (int, float) and math.isfinite(v) and abs(v) <= 1e15):
                out[k] = v
        elif k in ROUTING_STRINGS:
            if v is not None:
                if not isinstance(v, str) or len(v) > ROUTING_STR_MAX:
                    raise HTTPException(400, 'Invalid routing')
                out[k] = v
        elif k == 'panel':
            if v is None:
                continue
            if not isinstance(v, list) or len(v) > 32:
                raise HTTPException(400, 'Invalid routing')
            if any(not isinstance(p, str) or len(p) > ROUTING_STR_MAX for p in v):
                raise HTTPException(400, 'Invalid routing')
            out[k] = v
        elif k == 'compose_reason':
            # One level deep and scalars only. The CLI sends the gateway's own
            # reason dict here; it is small and read-only, and flattening it any
            # further would lose the thing it is for.
            if v is None:
                continue
            if not isinstance(v, dict) or len(v) > 32:
                raise HTTPException(400, 'Invalid routing')
            flat = {}
            for rk, rv in v.items():
                if not isinstance(rk, str) or len(rk) > ROUTING_STR_MAX:
                    raise HTTPException(400, 'Invalid routing')
                if isinstance(rv, (dict, list)):
                    continue
                if isinstance(rv, str) and len(rv) > ROUTING_STR_MAX:
                    raise HTTPException(400, 'Invalid routing')
                if rv is not None and not isinstance(rv, (str, int, float, bool)):
                    continue
                flat[rk] = rv
            out[k] = flat
        elif k == 'decision_id':
            if v is None:
                continue
            try:
                out[k] = str(UUID(v))
            except (ValueError, TypeError, AttributeError):
                raise HTTPException(400, 'Invalid routing')
    return out


def _valid_exchange(value):
    """An exchange is stored VERBATIM and must not go through `sanitise()`.

    `sanitise()` is a strict allowlist over NUMBERS/BOOLS/ENUMS/CONTAINERS. Not
    one of 'question', 'answer' or 'exchange' appears in any of them, so
    `sanitise(exchange)` returns `{}` — it would delete this entire feature while
    looking like it had worked. Text the user deliberately attached, previewed
    and confirmed is not the same kind of value as an automatically-gathered
    diagnostic, and the allowlist is there to stop the second kind. Validate
    structurally instead, and bound the size.

    `tests/test_client_observability.py` pins this: it asserts attached text
    comes back out of the row unchanged, which is the inverse of the
    `assertNotIn('SECRET', ...)` guards that must keep protecting `diagnostics`.

    No migration accompanies this. `client_reports.payload` is jsonb and this
    route builds it as a Python dict, so a new key needs no DDL.
    """
    if not isinstance(value, dict):
        raise HTTPException(400, 'Invalid exchange')
    question = value.get('question')
    if not isinstance(question, str) or not 1 <= len(question) <= QUESTION_MAX_CHARS:
        raise HTTPException(400, 'Invalid exchange question')
    answer = value.get('answer')
    # Empty is allowed on purpose: a node that returned nothing at all is a
    # reportable failure, and it is one of the cases this feature exists for.
    if not isinstance(answer, str) or len(answer.encode('utf-8')) > ANSWER_MAX_BYTES:
        raise HTTPException(400, 'Invalid exchange answer')
    out = {'question': question, 'answer': answer}
    if 'answer_truncated' in value:
        if type(value['answer_truncated']) is not bool:
            raise HTTPException(400, 'Invalid exchange')
        out['answer_truncated'] = value['answer_truncated']
    if value.get('routing') is not None:
        out['routing'] = _valid_routing(value['routing'])
    if len(json.dumps(out)) > EXCHANGE_MAX_BYTES:
        raise HTTPException(400, 'Exchange too large')
    return out



def sanitise(value, depth=0):
    if depth > 8:
        return {}
    if isinstance(value, list):
        return [sanitise(v, depth + 1) for v in value[-200:]]
    if not isinstance(value, dict):
        return {}
    out = {}
    for k, v in value.items():
        if k in NUMBERS and (v is None or (type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1e15)):
            out[k] = v
        elif k in BOOLS and type(v) is bool:
            out[k] = v
        elif k in ENUMS and isinstance(v, str) and v in ENUMS[k]:
            out[k] = v
        elif k == 'model':
            out[k] = v if isinstance(v, str) and MODEL.fullmatch(v) else 'custom-model'
        elif k in ('version', 'ollamaVersion') and isinstance(v, str) and re.fullmatch(r'\d+\.\d+\.\d+(?:-[a-z0-9.]+)?', v):
            out[k] = v
        elif k == 'build' and isinstance(v, str) and re.fullmatch(r'[a-f0-9]{7,40}', v):
            out[k] = v
        elif k in ('installationId', 'workerId') and isinstance(v, str) and re.fullmatch(r'[a-f0-9-]{36}', v):
            out[k] = v
        elif k == 'nodeId' and isinstance(v, str) and re.fullmatch(r'[a-f0-9]{24}', v):
            out[k] = v
        elif k in CONTAINERS:
            out[k] = sanitise(v, depth + 1)
    return out


_buckets = {}
_salt = secrets.token_bytes(32)


def limit(request):
    # Global brake bounds aggregate intake even when a caller rotates addresses.
    # No forwarded headers are trusted and no raw IP is kept, even in this map.
    now = time.monotonic()
    key = hashlib.sha256(_salt + (request.client.host if request.client else '').encode()).hexdigest()
    if len(_buckets) > 10000:
        _buckets.clear()
    for name, capacity in [('global', 120), (key, 20)]:
        tokens, last = _buckets.get(name, (capacity, now))
        tokens = min(capacity, tokens + max(0, now - last) * capacity / 60)
        _buckets[name] = (max(0, tokens - 1), now)
        if tokens < 1:
            raise HTTPException(429, 'Please retry later', headers={'Retry-After': '60'})


async def read_body(request):
    limit(request)
    async def read():
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > 131072:
                raise HTTPException(413, 'Request too large')
        try:
            body = json.loads(data)
        except (ValueError, UnicodeError, RecursionError):
            raise HTTPException(400, 'Invalid JSON')
        if not isinstance(body, dict):
            raise HTTPException(400, 'Expected an object')
        return body
    try:
        return await asyncio.wait_for(read(), 10)
    except asyncio.TimeoutError:
        raise HTTPException(408, 'Request timed out')


@router.post('/client/telemetry')
async def telemetry(request: Request):
    body = await read_body(request)
    try:
        installation = UUID(body.get('installationId', ''))
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(400, 'Invalid installation ID')
    events = body.get('events')
    if not isinstance(events, list) or not 1 <= len(events) <= 50:
        raise HTTPException(400, 'Expected 1–50 events')
    clean = [sanitise(e) for e in events]
    if any(e.get('event') not in EVENTS for e in clean):
        raise HTTPException(400, 'Invalid event')
    async with db.pool().acquire() as conn:
        async with conn.transaction():
            await conn.execute('insert into client_telemetry(installation_id,payload) values($1,$2::jsonb)', installation, json.dumps(clean))
            await conn.execute("delete from client_telemetry where received_at < now() - interval '30 days' or id < (select coalesce(max(id),0)-50000 from client_telemetry)")
    return {'accepted': True}


@router.post('/client/reports')
async def report(request: Request):
    b = await read_body(request)
    if not isinstance(b.get('id'), str) or not re.fullmatch(r'COMMON-\d{8}-\d{4}-[A-F0-9]{8}', b['id']):
        raise HTTPException(400, 'Invalid report ID')
    if b.get('mode') not in ('feedback', 'problem') or b.get('category') not in ('general', 'setup', 'chat', 'contribution', 'bug'):
        raise HTTPException(400, 'Invalid report type')
    if not isinstance(b.get('text'), str) or not 1 <= len(b['text'].strip()) <= 5000:
        raise HTTPException(400, 'Expected feedback text')
    if b.get('contact') is not None and (not isinstance(b['contact'], str) or len(b['contact']) > 200):
        raise HTTPException(400, 'Invalid contact')
    if b.get('rating') is not None and (type(b['rating']) is not int or b['rating'] not in range(1, 6)):
        raise HTTPException(400, 'Invalid rating')
    # isinstance before the membership test, deliberately. REASONS is a set, and
    # `x not in set` on an unhashable x -- a list, a dict -- raises TypeError and
    # becomes a 500. `mode` and `category` above get away with a bare `in`
    # because they test against a tuple, which compares rather than hashes. On a
    # public boundary junk has to be a 400, never a 500.
    if b.get('reason') is not None and (not isinstance(b['reason'], str) or b['reason'] not in REASONS):
        raise HTTPException(400, 'Invalid reason')
    payload = {k: b.get(k) for k in ('mode', 'category', 'text', 'rating', 'contact')}
    # `reason` is not coupled to mode. It answers "why was this bad?", which is
    # a separate axis from `category` ("what part of the product?") and from
    # mode itself, and a feedback report carrying one is not an error.
    payload['reason'] = b.get('reason')
    # `diagnostics` is gathered automatically by the client, so it goes through
    # the allowlist; `exchange` is attached deliberately by the user after a
    # preview, so it does not. See _valid_exchange -- running it through
    # sanitise() would silently empty it.
    payload['diagnostics'] = sanitise(b.get('diagnostics')) if b['mode'] == 'problem' else None
    # Coupled to `problem` for the same reason `diagnostics` is: an exchange is
    # evidence about an answer that went wrong, and a `feedback` report has no
    # answer to be evidence about. Validation still runs first, so a malformed
    # exchange is a 400 rather than something silently dropped.
    payload['exchange'] = (_valid_exchange(b['exchange'])
                           if b['mode'] == 'problem' and b.get('exchange') is not None else None)
    async with db.pool().acquire() as conn:
        async with conn.transaction():
            result = await conn.execute('insert into client_reports(id,payload) values($1,$2::jsonb) on conflict do nothing', b['id'], json.dumps(payload))
            if result != 'INSERT 0 1':
                raise HTTPException(409, 'Report ID already exists')
            await conn.execute("delete from client_reports where received_at < now() - interval '90 days' or id in (select id from client_reports order by received_at desc offset 10000)")
    return {'accepted': True, 'id': b['id']}


@router.get('/admin/client-reports')
async def reports(request: Request):
    _require_admin(request)
    async with db.pool().acquire() as conn:
        rows = await conn.fetch('select id,received_at,payload from client_reports order by received_at desc limit 100')
    return [dict(r) for r in rows]


@router.get('/admin/client-telemetry')
async def telemetry_summary(request: Request):
    _require_admin(request)
    async with db.pool().acquire() as conn:
        rows = await conn.fetch('select installation_id,received_at,payload from client_telemetry order by received_at desc limit 200')
    return [dict(r) for r in rows]


@router.post('/nodes/{node_id}/health')
async def node_health(node_id: UUID, request: Request, x_common_node_token: str | None = Header(default=None)):
    limit(request)
    async with db.pool().acquire() as conn:
        row = await conn.fetchrow('select id,endpoint_url,api_key_ref,worker_token,node_token,model_name from nodes where id=$1', node_id)
    if not row or not x_common_node_token or not secrets.compare_digest((row['node_token'] or '').encode(), x_common_node_token.encode()):
        raise HTTPException(404, 'Registration not found')
    async with httpx.AsyncClient(follow_redirects=False) as client:
        healthy = await _check_one(client, dict(row))
    async with db.pool().acquire() as conn:
        await conn.execute('update nodes set healthy=$1,last_heartbeat=now(),last_seen_healthy=case when $1 then now() else last_seen_healthy end where id=$2 and not paused and endpoint_url=$3', healthy, node_id, row['endpoint_url'])
    return {'healthy': healthy}


@router.get('/network/overview')
async def overview():
    async with db.pool().acquire() as conn:
        row = await conn.fetchrow("select count(*) as endpoints, count(*) filter (where node_token is not null) as contributors from nodes where healthy and last_seen_healthy > now() - interval '90 seconds'")
    # No reliable busy count exists yet. Never infer it from stale routing data.
    return dict(row)


async def prune_client_data():
    """Time retention must hold even when nobody submits another event."""
    async with db.pool().acquire() as conn:
        await conn.execute("delete from client_telemetry where received_at < now() - interval '30 days'")
        await conn.execute("delete from client_reports where received_at < now() - interval '90 days'")


async def retention_loop():
    while True:
        try:
            await prune_client_data()
        except Exception:
            # Do not print database errors or payloads into process logs.
            pass
        await asyncio.sleep(3600)


@router.post('/nodes/{node_id}/pause')
async def pause_node(node_id: UUID, x_common_node_token: str | None = Header(default=None)):
    """Stop routing immediately without revoking chat access on an idle pause."""
    async with db.pool().acquire() as conn:
        result = await conn.execute('update nodes set healthy=false,paused=true where id=$1 and node_token=$2', node_id, x_common_node_token)
    if result == 'UPDATE 0':
        raise HTTPException(404, 'Registration not found')
    return {'paused': True}
