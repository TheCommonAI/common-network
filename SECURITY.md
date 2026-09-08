# Security and privacy

Common is a trusted-compute prototype, not a confidential-computing system.
A contributor's administrator can inspect any prompt or answer processed on
that ordinary PC. The gateway also sees plaintext to route and compose it.
TLS and content-free logs do not change either fact. Use trusted participants
and non-sensitive inputs; no claim of end-to-end encryption is made.

## Implemented boundaries

- Workers authenticate each inference request with a random credential distinct
  from the node's ownership/chat token. Only `GET /v1/models` and
  `POST /v1/chat/completions` are exposed. Only the contributed model is allowed.
- Tunnel workers bind loopback. `--lan` explicitly binds the worker to the LAN;
  Ollama stays on loopback. Old externally bound Ollama installations must be
  reconfigured: Common cannot revoke a separately opened port or old tunnel.
- Workers accept at most 16 connections and one inference job by default.
  Headers/body have a 15-second overall read deadline, and input is limited to
  2 MB. Inference has socket timeouts, an output-byte cap, an output-token cap
  and an overall relay deadline. Content is not logged and backend error
  bodies are not reflected. Responses stream promptly instead of waiting for
  an 8 KB buffer. A missing Ollama/model fails health checks.
- Gateway registration and outbound requests reject non-public addresses by
  default, HTTP on public destinations, URL credentials, query strings,
  fragments and ambiguous paths. Link-local/metadata, unspecified and multicast
  addresses are always forbidden. Every resolved address must pass policy.
- Outbound HTTP connects to a validated numeric address; the original hostname
  is preserved for Host, TLS SNI and certificate verification. Fresh HTTP/1.1
  connections prevent reuse of one hostname's TLS session for another hostname
  sharing an IP. Redirects and environment-configured proxies are disabled on
  gateway-to-node requests. Health checks use the same boundary and a deadline.
- A gateway environment key is released only for its exact configured HTTPS
  endpoint. A name allowlist alone never authorises a credential.
- Clients attach saved tokens only to their issuing gateway. Credential-bearing
  requests and request bodies cannot follow redirects. Internet gateways must
  use HTTPS; trusted LAN HTTP must use a private IP (or loopback).
- Node credentials are stored as SHA-256 verification digests. Random tokens are
  high-entropy secrets, not human passwords. Migration 008 hashes existing
  node tokens without changing client credentials. Hash values themselves are
  not accepted as credentials. Legacy plaintext tokens remain supported during
  an upgrade and are hashed on successful re-registration.
- Worker tokens must remain recoverable by the gateway to authenticate outgoing
  jobs. They are not returned in public node listings. Keep the database and
  backups private. Name claims are serialised in a database transaction.
- Chat quotas use verified contributor identities. IP-based registration/admin
  limits accept forwarding headers only from configured trusted proxies, walking
  the chain from the trusted end. In-memory bucket counts are bounded; bucket
  labels use process-local keyed fingerprints rather than tokens or raw IPs.
- Gateway input and output bytes, message counts, output tokens, concurrent
  requests and request duration are bounded. Text chat, system messages,
  streaming, ordinary sampling controls and JSON response-format requests are
  supported. Unsupported options, remote image inputs and non-text messages
  are rejected explicitly rather than silently forwarded.
- `X-Common-Allowed-Nodes: name1,name2` restricts all eligible workers, including
  primary selection, retries, specialists and aggregation. An empty/unavailable
  allowed group fails closed. `X-Common-No-Retry: true` disables fallback after
  a failed single route or failed panel. `X-Common-Compose: never` prevents
  composition. A node name expresses a user's trust choice, not attested identity.
- Admin data requires `X-Common-Admin-Token`. `/admin` has a password input;
  the password remains in page memory and is never put in URLs, cookies or
  browser storage. Lock or reload clears it. Query-string passwords are refused.
- Responses have no-store, no-referrer and nosniff headers. Dashboard/admin pages
  prohibit framing, external resource loads and external form destinations.
  Validation errors omit submitted values (which could contain secrets).
- Contributor identity files are atomically replaced with restrictive Unix
  permissions. The Windows installer restricts its directory ACL to the current
  user and SYSTEM. Failure to save identity is reported and registration is
  rolled back on a best-effort basis. Background Windows tasks no longer request
  highest privileges. The gateway container runs as a non-root user.

## Retention and public statistics

Request embeddings and detailed composition reasons are not retained by default.
An hourly cleanup clears previous copies when retention is disabled, and deletes
routing metadata older than `DECISION_RETENTION_DAYS` (default 7, minimum 1).
Cleanup failures produce a content-free warning; they must be investigated.
Deleting rows does not erase database backups, WAL, disk remnants or provider
logs. Operators must configure the corresponding backup/log expiry themselves.

`/decisions/recent` is admin-only by default. `/decisions/mine` gives a contributor
only their own aggregate contribution counts. `/decisions/summary` and the public
dashboard use delayed aggregate counts with small groups suppressed. Public
worker endpoints are hidden, although aliases, capabilities and available models
remain visible. Each requester still receives their own routing receipt headers.

Demand planning continues from delayed domain counts and model coverage. Without
stored embeddings, clustering of uncategorised request vectors is unavailable;
`embedding_analysis_enabled` reports this. Public clustering work is bounded to
500 vectors and cached. Counts suppressed for privacy are not proof of zero demand.
These measures reduce incidental disclosure; they are not differential privacy,
anonymity, or protection against a determined observer of a small network.

`RETAIN_REQUEST_EMBEDDINGS=true`, `PUBLIC_DECISION_DETAILS=true` and
`PUBLIC_NODE_ENDPOINTS=true` are explicit research/development exceptions. Do not
enable them for sensitive traffic. Enabling embedding retention is an operator
choice, not a substitute for informed consent from the people using that gateway.

## Upgrading an existing deployment

1. Review this PR and stage it with a private test gateway first. Apply
   `python -m app.migrate` with the deployed version; migration 008 is idempotent.
   Rollback to an older gateway requires a plan for hashed node credentials;
   old code cannot authenticate migrated tokens. Clients keep the same raw token.
2. Leave public endpoints on HTTPS and set `ALLOW_LOOPBACK_NODE_ENDPOINTS=false`.
   For a trusted LAN gateway, set `ALLOWED_NODE_CIDRS` to the actual node subnet,
   for example `192.168.1.0/24`. For a same-machine demo, explicitly enable
   loopback. Never use `0.0.0.0/0` as an exception on a public service.
3. Replace `ALLOWED_API_KEY_REFS` with exact destination mappings, e.g.
   `API_KEY_DESTINATIONS={"OPENROUTER_API_KEY":["https://openrouter.ai/api/v1"]}`.
   The old name-only setting is accepted for configuration compatibility but
   does not authorise any key. A changed destination requires operator approval.
4. Keep `REQUIRE_CONTRIBUTION=true` and configure quota values for your test
   population. Use a single gateway process for the built-in quotas. Configure
   `TRUSTED_PROXY_CIDRS` only from your hosting provider's actual proxy network.
   Do not combine this with an ASGI server that blindly rewrites client addresses.
   The Docker command disables uvicorn proxy-header rewriting and access logs.
5. Reinstall/restart contributor tools so tunnels point at the restricted worker.
   New/re-registered nodes become eligible after a successful health check.
   Remove old raw-Ollama tunnels and reverse old `OLLAMA_HOST=0.0.0.0` settings.
   LAN firewall access is for worker port 11435, not Ollama port 11434.
6. Open `/admin` and enter the password in the page. Retire saved `?token=` links.
   Rotate any password previously used in URLs; proxies may already have logged it.
7. Review retention settings and confirm cleanup succeeds. Configure expiry for
   database backups and infrastructure logs separately. No real prompts should
   appear in diagnostics. Windows/macOS installations and real-model performance
   still need smoke testing on those target operating systems.

## Updates and benchmark execution

Automatic fetch-and-execute updates are disabled. `common join` uses installed
code and does not independently fetch a replacement join script. Re-run the
installer to update deliberately; installers resolve one immutable repository
commit for all Common files. `COMMON_INSTALL_COMMIT` can pin a reviewed full SHA.
This is revision consistency, **not signed release verification**. Initial
installation still trusts GitHub/repository control and upstream dependency
installers. The opt-in `COMMON_ALLOW_UNVERIFIED_UPDATES=1` retains the old unsafe
update path for developers, with a warning. Do not enable it on donated PCs.
A signed release process with separately managed signing keys remains future work.

Benchmark code execution is off by default in both implementations. The CLI's
`--allow-unsafe-exec` and the library's `allow_unsafe_exec=True` are explicit,
dangerous opt-ins. They execute generated Python with the caller's permissions.
Use only in a disposable isolated environment. CPU/memory limits and timeouts
are not a security sandbox. `--no-exec` is still accepted and overrides the opt-in.

## Remaining trust and operational limits

- Ordinary PC administrators, the gateway operator, swap/crash dumps and hostile
  backend modifications can expose content. No secure memory-erasure promise.
- LAN HTTP is plaintext, including credentials. Use an encrypted VPN or properly
  terminated TLS when network interception is in scope. Tunnel TLS terminates at
  its provider; this is not requester-to-worker end-to-end encryption.
- Registration does not prove honest contribution or stop Sybil identities.
  Admission policy, abuse monitoring, global shared quotas and fair scheduling
  remain necessary before broad untrusted public use.
- Worker cancellation closes upstream connections best-effort. Ollama controls
  when GPU work actually stops. Resource limits are not a hard GPU/RAM sandbox.
- No attestation, confidential computing, signed updater, secret-management
  service, reproducible-build guarantee or comprehensive dependency audit is
  provided by this change. Containers do not hide content from the host owner.
- Limits are per process. Multiple gateway processes need shared limits; hosting
  firewalls must also restrict ingress/egress and protect the database.

## Verification

Run `python tests/run_all.py` from `gateway/` with gateway dependencies installed.
The lightweight suite does not download or load an embedding model. It covers
routing/composition, real local HTTP forwarding, worker auth/route/model boundaries,
streaming, gateway chat, recipient restrictions, redirects, DNS pinning, TLS
hostname verification, retention writes, quotas, bounds and credential handling.
TLS fixture tests need OpenSSL and report a skip if it is missing.

These tests do not certify live deployment configuration, database migrations on
production data, GPU resource isolation, or platform installers. Test those in
staging before merging/deployment. Report vulnerabilities through a private GitHub
security advisory rather than posting real secrets or exploit details publicly.
