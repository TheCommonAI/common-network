# Retiring the old (v0.1) network

Status of this document: an inventory and an ordering, not a plan that has been executed. Nothing described
under "Not removed yet" has been touched. The `retire/` branches only remove the unreleased v0.2 RPC-group
prototype and flip the `common join` default; they do not delete the v0.1 system.

## What "the old network" is

v0.1 is **one whole model per node, routed by a central gateway**:

- A node (`common join`) downloads a complete GGUF, runs `llama-server` locally, puts a worker in front of it,
  opens a Cloudflare tunnel and **registers with the gateway**.
- The gateway scores nodes, picks one (or fans out to several "specialists" and composes one answer),
  and proxies an OpenAI-style request to it.
- Clients (`common ask`, the desktop app, `chat/chat.py`) talk to the gateway, never to nodes.

The layer network replaces this with **one model split across peers, no central server**: peers gossip
signed records, clients build routes themselves, and a dropped peer is replaced mid-generation
(`docs/layer-network.md`).

## Already done on the `retire/` branches

| Piece | Where | State |
|---|---|---|
| v0.2 RPC-group prototype (`join/group.py`, `group_config.py`, tests, `DISTRIBUTED.md`, trial configs, `bench/distributed.py`, `DistributedConfig` in `runtime.py`, the join flags, `common rpc`) | `retire/v02-rpc-group` | Removed. About 4,800 lines. Never released ("not a stable release" in its own docs). The flags still parse and now say `removed: ... use common join --network`. |
| `common join` default | `retire/layer-first-join` | A plain `common join` runs the layer network if the machine can (Apple silicon or NVIDIA); `--legacy`, `--model`, `--lan`, `--list-catalogue` select the old node. Services record their mode. |

Not touched on either branch, although they mention "distributed groups": `gateway/app/gateway.py`,
`health.py`, `models.py`, `config.py`, `gateway/tests/test_groups.py`, migration `015_group_status.sql`.
That is production and stays until the gateway itself is retired.

## Not removed yet: the inventory

Line counts are from this tree. "Depends on it" means something would break if it were deleted today.

### Gateway (`gateway/`, deployed on Railway from `gateway/Dockerfile`)

| Piece | Lines | Depends on it | Replaced by | Removable when |
|---|---|---|---|---|
| `registry.py` (node registry, `POST/GET/DELETE /nodes`) | 398 | every legacy node; `common status/peers/leave/contrib`; the dashboard | the peer table (`commonnet/mesh`): signed records, gossip | no legacy nodes remain |
| `router.py`, `upstream.py`, `gateway.py` (`POST /v1/chat/completions` routing and proxy) | 130 + 111 + 863 | `common ask`, the desktop app (runs `common ask --json` for every question), `chat/chat.py` | `python -m commonnet.serve` (OpenAI-compatible, built); **`common ask` and the desktop app are not yet pointed at it** | `common ask` and the desktop app use the layer network |
| `health.py` (health loop, mobile leases) | 188 | registry, mobile nodes | peers re-sign their own record every 15 s; stale records expire | with the registry |
| `compose.py`, `verify.py` (parallel specialists + verification report) | 644 + 463 | `gateway.py` only (flag `X-Common-Compose`) | nothing: composition is a different idea (many models, one answer), not a layer-network feature. Its own docstring records that the v0.1 experiments found no gain | when the gateway stops routing; or earlier if composition is switched off |
| `demand.py`, `decisions.py`, `catalogue.py` + `catalogue/catalogue.seed.yaml` | 489 + 136 + 309 | `common demand/recommend`, `join` model assignment (`POST /assign`), dashboard | the manifest names the one model; stage coverage is `placement.coverage` | with the registry |
| `metrics.py`, `client_observability.py`, `admin.py` | 1078 + 357 + 324 | dashboards, operators | per-hop timings exist in the client (`BlockClient.timings`); no network-wide metrics yet | decide what to keep for the new network first |
| `mobile_transport.py` (`WS /nodes/{id}/connect`, Android/iOS workers) | 534 | the mobile apps/workers | **nothing yet**: the layer network has no mobile peer | do not remove until a mobile peer exists |
| `gateway/migrations/001-017`, `gateway/tests/*` | n/a | the database | n/a | with the gateway |

### Node side (`join/`, `common/`)

| Piece | Lines | Depends on it | Replaced by | Removable when |
|---|---|---|---|---|
| `join/worker.py` (auth'd front for llama-server, warm-up) | 560 | legacy `join.py` | the block server (`commonnet/net/server.py`) | `--legacy` is dropped |
| `join/runtime.py` (llama.cpp fetch, GGUF fetch, `llama-server` argv) | about 1,260 | legacy `join.py`, `common ask --local` | `hub.py` + MLX/NF4 blocks | `--legacy` is dropped and `--local` is re-done |
| legacy path in `join/join.py` (registration, tunnel, model assignment, services) | about 1,600 of 2,000 | `--legacy`, `--model`, existing services | `join/layers.py` + `commonnet/node.py` | after the services below have migrated |
| `common/common.py` verbs `ask/status/peers/demand/recommend/contrib/report/test/leave` | about 4,100 | users and the desktop app | `serve`-based `ask`; other verbs have no layer equivalent yet | one verb at a time |
| `chat/chat.py` | 438 | users of the terminal chat | `commonnet.serve` | after `common ask` moves |
| `bench/` (v0.1 scoring: GSM8K, HumanEval, MMLU) | n/a | `common test` | still valid for quality testing the layer model | keep |

## Order that works

1. **Done:** remove the unreleased v0.2 prototype; flip the `join` default; keep `--legacy`.
2. **Point asking at the layer network.** Give `common ask` (and so the desktop app) a mode that talks to
   `commonnet.serve` or an in-process client, with the gateway as the fallback. This is the real gate:
   until it exists, retiring the gateway removes the only way users ask questions.
3. **Run both side by side** long enough to see the layer network carry real use (more than one operator's
   machines, a real seed list, the NVIDIA path actually run).
4. **Stop accepting new legacy nodes** (registration returns a message pointing to `common join`), leave the
   gateway serving existing ones.
5. **Remove composition** (`compose.py`, `verify.py`, their tests, migration 006's use). It has the smallest
   dependency surface and the weakest evidence behind it.
6. **Retire the gateway routing and registry**, then `worker.py` / `runtime.py` and the legacy `join` path.
7. **Decide the mobile story separately.** `mobile_transport.py` has no replacement.
8. **Shut down the Railway service**, last.

## Migration risks

- **Installed nodes self-update from `main`.** `common join` fetches `join.py` (and `worker.py`, `runtime.py`)
  from `raw.githubusercontent.com/.../main`, and `common` fetches `common.py` the same way. Merging a change to
  these files reaches every installed node on its next start. The default flip is therefore a change to what
  every plain `common join` does after they update. This is why the flip is its own branch.
- **Existing services keep their mode.** A background service records its full command line, which always
  carried `--model`, so services installed before the flip still run the legacy node. New legacy services
  record `--legacy`; new layer services record `--network`. A service installed by the *old* `join.py`
  and then updated is not rewritten.
- **A machine that cannot serve layers stays legacy.** CPU-only machines (and Intel Macs) cannot join the layer
  network ("GPU only"), so they fall back to the whole-model node with a notice rather than breaking.
- **The gateway builds from `main` on Railway** (`railway.json` -> `gateway/Dockerfile`). A merge that touches
  `gateway/` can redeploy production. None of the retire branches touch it.
- **Users lose what the old network gave them.** The layer network serves one model (Qwen3-8B in the shipped
  manifest). It does not do specialist routing, composition, mobile nodes, a dashboard or the contribution
  gate. Whether that trade is wanted is a product decision, not a code one.
- **Open network.** The layer network publishes each peer's access token in the mesh and peers see
  activations; the v0.1 gateway had token gating and rate limits. Do not migrate private use until that is decided.
- **Untested on real hardware:** the NVIDIA (Windows/Linux) path of the layer network has never run, so in
  practice Windows users are only served by the old network until it has.
