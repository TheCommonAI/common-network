# Desktop observability and verification API

Apply migration `010_client_observability.sql` before deploying this code. The
Dockerfile does not automatically apply migrations: use the existing migration
command (`python -m app.migrate`) with the deployment DATABASE_URL.

- `POST /client/reports`: explicitly submitted feedback/problem reports. Returns
  `{accepted:true,id}` only after the database transaction commits. A duplicate ID
  returns 409. Reports contain user-entered text/contact and optional allowlisted
  diagnostics, never automatically attached conversations or credentials.
  A report may also carry an **`exchange`** — the question and the answer it is
  about. That is still not automatic: it is attached only when the person
  reports a specific answer, and the client shows them the exact gateway, the
  question, the answer and the routing line and asks before anything is sent.
  A user who declines keeps nothing locally either.
  Worth stating plainly, because it is a real change in exposure: a report's
  `exchange` is read by the gateway operator, who could not read it before. The
  node that answered could already read the question — that is inherent in
  asking a stranger's machine to answer you — so the report adds a *second*
  reader, not the first. It does not widen who can read anything else.
- `POST /client/telemetry`: opt-in desktop event batches, 1–50 events, random
  installation UUID. Server-side allowlist independently drops unknown fields.
- `POST /nodes/{uuid}/health`: requires that node's contributor token; probes the
  stored endpoint with its worker credential, validates models and updates health.
  It does not accept an arbitrary URL supplied in the probe request.
- `POST /nodes/{uuid}/pause`: authenticated ownership check, immediately marks
  a stopped worker unhealthy without revoking its chat credential. The desktop
  closes its public endpoint first; resume re-registers and verifies reachability.
- `GET /network/overview`: healthy endpoints and contributors seen within 90s.
  A “contributor” is a registered credential-bearing node, not a unique person.
  No busy count is published.
- `GET /admin/client-reports`, `GET /admin/client-telemetry`: bounded recent
  operator views protected by the existing ADMIN_TOKEN gate. No credentials or
  conversation content in automatic telemetry.

  ```
  curl -s -H "X-Common-Admin-Token: $ADMIN_TOKEN" \
       "$GATEWAY/admin/client-reports" | python -m json.tool
  ```

  Each row carries the full `payload`, including `exchange` when one was
  attached. Read it with that in mind: it is somebody's question and somebody's
  answer, kept because they asked for it to be looked at.

## The report ID

`COMMON-YYYYMMDD-HHMM-XXXXXXXX` — UTC date, UTC time, eight uppercase hex
characters. The client mints it (`secrets.token_hex(4).upper()`, stdlib) and
retries while a local file already exists. It is the primary key here, so a
client that resends the same report gets a 409 rather than a duplicate row; the
client treats that as "already received" and stops.

## Exchange shape and caps

`exchange` is validated structurally and stored **verbatim**. It is
deliberately *not* run through `sanitise()`, which is an allowlist over
`NUMBERS`/`BOOLS`/`ENUMS`/`CONTAINERS` and contains none of `exchange`,
`question` or `answer` — `sanitise(exchange)` returns `{}` and would delete the
feature while looking like it had worked. Text someone attached on purpose is
not the same kind of value as an automatically-gathered diagnostic, and the
allowlist exists to stop the second kind. `tests/test_client_observability.py`
asserts attached text survives a round trip, and asserts that `sanitise()` would
have emptied it.

| Field | Bound | Notes |
|---|---|---|
| `question` | 1–4000 chars | characters, matching the client's own cap |
| `answer` | ≤32768 bytes UTF-8 | bytes, not characters; empty is allowed |
| `routing` | allowlisted sub-keys | `node`, `topology`, `aggregator` ≤200 chars; `panel` ≤32 entries; `decision_id` a UUID; `compose_reason` one level deep; unknown sub-keys dropped |
| whole object | ≤64 KiB as JSON | well under the 128 KiB body cap |

`exchange` is coupled to `mode='problem'`, like `diagnostics` and for the same
reason: an exchange is evidence about an answer, and a `feedback` report has no
answer to be evidence about. Validation runs first either way, so a malformed
exchange is a 400 and not something quietly dropped. `reason` is *not* coupled
to mode — it answers "why was this bad?", which is a different axis from
`category` ("what part of the product?") and from mode itself.

## No migration

The `exchange` and `reason` fields needed none. `client_reports.payload` is
`jsonb` and this route builds it as a Python dict, so a new key is not DDL.
Migration `010_client_observability.sql` remains the only one this surface has.

Intake limits: 128 KiB body, ten-second body-read timeout, process-local global
and source-bucket rate limits. Source bucket uses an ephemeral salted hash of the
socket address, not a persisted IP. Proxies may make many clients share a bucket;
this is intentionally conservative. Multi-instance production abuse protection
still needs a shared edge limit. Random installation IDs are pseudonymous and
must not be represented as proof that users cannot be identified.

Telemetry expires after 30 days, reports after 90 days. Hourly pruning runs even
without new submissions; expiry can lag by that interval. Row ceilings are
50,000 telemetry batches and 10,000 reports. Do not log intake request bodies.
Configure provider/proxy access-log IP retention separately; application-level
sanitisation cannot control infrastructure logs.

Validation: `python tests/run_all.py` from gateway/. Intake tests use an in-memory
DB adapter; staging must additionally validate migration, real PostgreSQL
persistence/retention and access control before desktop rollout.
