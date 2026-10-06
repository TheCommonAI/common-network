# Desktop observability and verification API

Apply migration `010_client_observability.sql` before deploying this code. The
Dockerfile does not automatically apply migrations: use the existing migration
command (`python -m app.migrate`) with the deployment DATABASE_URL.

- `POST /client/reports`: explicitly submitted feedback/problem reports. Returns
  `{accepted:true,id}` only after the database transaction commits. A duplicate ID
  returns 409. Reports contain user-entered text/contact and optional allowlisted
  diagnostics, never automatically attached conversations or credentials.
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
