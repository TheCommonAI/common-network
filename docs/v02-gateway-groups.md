# DESIGN: Gateway groups for the v0.2 distributed model

**Summary and recommendation.** A distributed group is already one registered node. `join --distributed-config` without `--local-only` registers the coordinator under its `model_name`, through the normal Common worker and tunnel (join/join.py, registration payload near line 1850). The gateway cannot see the group itself: its workers, their health, or their credit. So v1 is smaller than "make the gateway group-aware." Recommendation: keep the coordinator as the only public, registered node; keep push routing to its endpoint; have the coordinator report group state through its authenticated worker; restart the runtime in place instead of deregistering; keep workers private behind an authenticated outbound transport; credit only the coordinator in v1. Workers never face the internet in any version.

## 1. Registration

Today `nodes` has one row per `name` (unique, `gateway/migrations/001_init.sql`), one `endpoint_url`, one `model_name`, one `worker_token` and one `node_token`. Routing already works for a group: `score_nodes` and `choose_primary` (`gateway/app/router.py`, `gateway/app/gateway.py`) treat the coordinator as an ordinary node.

Changes:
- Migration `015`: add nullable `group_status` (`ready | restarting`, null for single nodes), `group_members_total` and `group_members_up`. The coordinator writes them. They are shown on `GET /nodes` and are display only, never a routing input. Self-reported values are data, not authorisation, as with `client`.
- Distributed registration currently hardcodes `domain_tags = ["general"]` (join.py, distributed branch). A group registered as "general" can be picked by the generalist rule in `choose_primary`. Give the group its model's own domain tag instead.
- No `groups` table and no group id in v1. Add them when a second group or discovery exists.

## 2. Routing

- **A. Push (current).** The gateway POSTs to the coordinator's `endpoint_url` through the Common worker (`upstream.forward`). It reuses the 30s health check (`health.py`, `GET /v1/models` with the worker token), metrics, decisions, the SSRF checks in `registry.py`, and the candidate fallback loop in `gateway.py`.
- **B. Pull.** The coordinator holds an outbound long-poll open; the gateway queues requests and relays streamed responses back. This removes the need for an inbound tunnel. It needs a work queue, a streamed response relay, cancellation, and a rewrite of the forward step in `chat_completions`.

**Recommend A.** B is a second serving architecture. Revisit it only if NAT or tunnel limits block real coordinators.

One gap in A. The health check probes only `/v1/models`, which needs no worker. A coordinator whose RPC worker has died could still answer `/models` and stay routable. `DISTRIBUTED.md` says a dead RPC connection "may terminate" the runtime. Whether llama-server exits or hangs is unverified. Fix: add an authenticated `GET /v1/status` route to `join/worker.py` (today it has two routes), returning runtime state and member counts. `health.py` reads it when present and falls back to `/models` for older nodes.

## 3. Trust

RPC has no authentication. `runtime.py` already binds `ggml-rpc-server` to `127.0.0.1` (its `main`), and the distributed config requires `host:port` endpoints.

Model:
- Each worker runs `ggml-rpc-server` on loopback only, plus a small worker agent.
- The worker agent dials out to the coordinator. The connection is mTLS, keyed to the member's enrolled key. The coordinator exposes each member on a loopback port, so `rpc_servers` stays `127.0.0.1:port` and `DistributedConfig` does not change.
- Workers accept no inbound connections. Only the coordinator is public, and only through the authenticated Common worker.
- For operator-owned workers, the existing `ssh -L` forwards meet this bar. The agent is needed when members are not operator-owned.

Admission:
- The coordinator's operator owns the group key and is the only party who can admit members.
- The operator issues a one-time join token (HMAC over group id, member name, expiry). The worker presents it once and receives a member key bound to its public key. The coordinator stores a hash of that key.
- The coordinator dials only admitted members. Revocation deletes the row.
- The gateway never sees worker addresses, tokens or a member list. It sees counts only, in line with `GET /nodes` not exposing `endpoint_url`.

## 4. Failure

Today a dead worker can take the runtime down. `join.py` then sees `llama_proc.poll()` in its main loop, calls `cleanup()`, deregisters and exits (around line 2000). Recovery is manual. Only `run_local_coordinator` restarts with backoff (`LOCAL_RESTART_DELAYS_SECONDS`), and only in local-only mode.

Proposed states, shown on `GET /nodes`, in the CLI and on the gateway:
- **ready.** Runtime serving, all members up. Routed normally.
- **restarting.** A member or the runtime is down. The coordinator restarts the runtime with backoff and keeps its registration and tunnel. The node is `healthy=false`, so it is not routed. A request whose only candidate is restarting gets a 503 with `Retry-After` and a detail naming the group status. Today it gets a generic "no healthy nodes available" (`gateway.py`, around line 585).
- **down.** The restart budget is exhausted. The coordinator deregisters, as it does today. `/v1/models` omits the model, and the public list shows `last_seen_healthy`.

Rules:
- **No "degraded" in v1.** The llama.cpp split is fixed at start, and preflight requires every device (`runtime.py`, device discovery). A partial group cannot serve, so showing "degraded" would promise service that does not exist.
- **Keep the registration during restarts.** Deregistering on every blip would force a rejoin, and the node token is the thing that must survive restarts.
- **Streams.** In-flight streams end with an error. Nothing silently switches to a different model. The candidate loop may fall back to another node serving the same model, as it does today.
- **Worker return.** Returning members trigger a restart with a full weight transfer. Show that as restarting, with its cost, rather than as a quick recovery.

## 5. Contribution

`node_token` is issued per registered node. `_require_contributor` (`gateway.py`, lines 43–64) checks only that a row exists with that token. Workers have no row and no token, so they cannot be credited.

`common contrib` (`common/common.py`, `cmd_contrib`) counts recent decisions whose `chosen_node` equals the node's id. `request_metrics` also keys on `node_id`. Credit therefore lands on the coordinator only.

Options:
- **(a) Coordinator-only credit.** Simple, and it needs no ledger change. Worker operators give compute and get nothing, which is the wrong incentive for a group.
- **(b) Member rows with declared shares.** Credit `completion_tokens` times each member's share of the tensor split. The share is declared, not measured, so the gateway cannot verify it. It is a coordinator claim, weaker than a node's own counters.

Recommend (a) in v1. Record member enrolment now, so (b) can be added without a backfill. Exclude the automatic one-token warm-up from credit. `DISTRIBUTED.md` notes that registered joins send it.

External access: a user must present a node token from some registered node (`REQUIRE_CONTRIBUTION`). The gate is network-wide, not per group. See decision 6.

## 6. Scope for a first version

**Must have:**
1. Authenticated `GET /v1/status` on `join/worker.py`, reporting runtime state and member count.
2. `health.py` reads that status and marks a non-serving runtime unhealthy. It falls back to `/models` for older nodes.
3. Registered distributed mode restarts the runtime in place, with the backoff already used locally, and keeps the registration. This replaces the `cleanup()` at exit (join.py, around line 2000).
4. Migration `015` with the group fields, shown on `GET /nodes` and in the CLI.
5. The group's own domain tag in place of `["general"]`.
6. A decision on the external access gate (decision 6).
7. Coordinator-only credit, with no change to `contrib`.
8. A test: kill one RPC worker, see `restarting`, see 503 with `Retry-After` on a request, see `ready` again after the worker returns.

**Can wait:** worker agent and mTLS (SSH is enough for operator-owned workers); member join tokens and admission flow; pull routing; per-member credit and its ledger view; block replication and state recovery (the release gates in `DISTRIBUTED.md`); multiple groups and discovery; adaptive draft length; partial-group serving.

## Decisions for you

1. **Group representation.** Group columns on `nodes` (default) or a `groups` table now.
2. **Routing.** Push through the tunnel (default) or a pull queue.
3. **Worker transport for v1.** Operator-run SSH forwards (default, since you own the workers) or a worker agent with mTLS now.
4. **Membership.** Operator-issued one-time join tokens (default) or open join.
5. **Failure policy.** Restart in place and keep the registration (default), or deregister and rejoin as today.
6. **External access.** Keep the contributor-token requirement (default) or add a group-level open flag.
7. **Credit.** Coordinator-only in v1 (default), or declared per-member shares now.
8. **Group domain tag.** The model's own tag (default), not `general`.
