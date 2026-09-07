# Security notes

Common is **permissionless by design**: anyone can register a node, and
anyone can send requests. That is the thesis — but it means the security
model has to be stated, not assumed. This document says what an unauthenticated
stranger can and cannot do, what an operator of a *publicly reachable* gateway
should do, and what is deliberately out of scope for Alpha.

## What a stranger can do (by design)

- Register a node (it must be reachable, and it will be health-checked).
- Send `POST /v1/chat/completions` requests and use the network's compute.
- Read `/nodes` (names, models, endpoint URLs, operators), `/decisions/*`,
  `/demand/*`, `/catalogue` and the dashboard.

Legibility is part of the pitch — every response says which machine answered
it — so the registry is public on purpose. If you operate a gateway, know that
node names, operators and endpoint URLs are visible to anyone. `common join`
defaults `--name` to `<hostname>-<random-suffix>` and `--operator` to
`friend` — it never publishes your OS username unless you pass it.

## What a stranger cannot do

- **Use the network without contributing to it.** With
  `REQUIRE_CONTRIBUTION=true` (the shipped default), a request to
  `/v1/chat/completions` must carry the node token of a currently registered
  node. Donated compute is for donors. The gate checks *registration*, not
  health — a donor whose laptop flaps offline keeps asking; a token whose
  node is deregistered stops working immediately.
- **Burn a contributor's machine with bulk traffic.** A per-client token
  bucket (`RATE_LIMIT_REQUESTS_PER_MINUTE`, default 20) throttles
  `/v1/chat/completions` after the burst. Best-effort, not a boundary — the
  contribution gate is the real control.
- **Take over or delete a node they didn't register.** Registration of an
  existing name requires that node's `X-Common-Node-Token`, issued once at
  first registration and returned only to whoever holds it. (Names are
  public; tokens are not — see `gateway/app/registry.py`.)
- **Make the gateway fetch internal addresses.** An endpoint URL must be
  `http(s)`, must resolve, and is refused outright if it resolves to a
  link-local address — the cloud-metadata range every SSRF write-up walks
  through. Loopback is allowed only when the operator opts in, and the check
  re-runs on every health pass, so a hostname that re-points at a metadata
  service after registering goes dark instead of being fetched.
- **Harvest credentials through the registry.** A node stores the *name* of
  an environment variable (`api_key_ref`), never a key. The gateway resolves
  the name against its own environment; no key value ever enters the
  database or an API response.
- **Read other people's requests.** The decisions log records topology, nodes,
  scores and latencies. It does not expose request text or embeddings.

## Running a public gateway

The shipped defaults are the public posture: contribution-gated
(`REQUIRE_CONTRIBUTION=true`) and rate-limited
(`RATE_LIMIT_REQUESTS_PER_MINUTE=20`). The local demo `.env` switches both
off because its seeded nodes have no tokens — **do not copy those overrides
to a public deployment**. On a public gateway:

1. **Set `ALLOW_LOOPBACK_NODE_ENDPOINTS=false`** so a stranger can't point the
   network at services only visible from the gateway's own machine.
2. **Leave `REQUIRE_CONTRIBUTION=true`** unless you deliberately want to give
   compute away to anonymous traffic.
3. **Keep the database private.** Request embeddings live in the `decisions`
   table; the API doesn't expose them, but a public Postgres would.
4. **Pin the CLI install to a tag** if you fork this. `install.sh` and the
   CLI's self-update fetch from `main` by default; `main` is fine while it is
   this repository, but it means whoever controls the repo controls every
   installed machine's CLI. The short install URL (`commonai.com.au/install.sh`)
   is a redirect to this repository, not a mirror — keep it that way, so the
   repo stays the single source of truth and the domain can never serve
   something the repo doesn't contain.

## Known limits, stated rather than papered over

- **The gate checks registration, not identity or fair shares.** Anyone who
  can register a node can use the network — that is still the thesis — and
  one person can register several. Usage proportional to *how much* you
  donate (capacity, model size, uptime) is a v0.2 problem; Alpha's gate is
  binary: in or out.
- **The node token now doubles as the access credential.** Leaking it costs
  more than it did yesterday: before it only controlled your own node, now it
  also grants network access while that node is registered. It is still
  scoped (one node, one gateway) and revocable (deregister the node).
- **Rate limiting is per-IP and trusts X-Forwarded-For.** A client rotating
  source addresses gets a bucket each. It is a brake on scripts in loops,
  not a defence against a determined bulk attacker — the gate is.
- **A contributor machine flapping offline keeps access while registered.**
  Deliberate (health is a routing signal, not membership), but it means a
  node that is registered-but-never-answering still lets its owner ask.
- **`common test` executes model-generated code** with process-level
  isolation only (`bench/sandbox.py` — subprocess, timeout, memory/CPU caps;
  not a container). Fine against machines you chose; run it with `--no-exec`
  against a gateway full of strangers' nodes. A node can feed you code to
  run; don't point the scorer at a network you don't trust.
- **A malicious node can answer requests**, including as a panel aggregator
  if composition is enabled. The network trusts node *identity*, not node
  *behaviour*. Responses are model output and should be treated as untrusted
  text by any client, the same as any LLM.
- **No CORS configuration.** Browsers cannot call the API cross-origin (the
  default is deny), which is the safe direction; the CLI and curl are
  unaffected.

## Reporting a problem

Open a private GitHub security advisory on this repository (Security tab →
"Report a vulnerability") rather than a public issue. If you can't, a plain
issue titled "security" with no reproduction details works — we'll follow up
for the details privately.

## History

- 2026-09-07 (pre-Alpha public release): re-registration of an existing node
  name returned that node's stored token to the caller — anyone who read a
  node's name from `GET /nodes` could take it over or delete it. Fixed before
  first publish: token required on name conflict, and endpoint URLs
  validated against metadata/link-local ranges.