# Website dashboard API

The gateway exposes a public, read-only snapshot for websites:

```text
GET https://gateway-production-b820.up.railway.app/api/network-stats
```

No API key is needed. The endpoint allows cross-origin browser GETs (`Access-Control-Allow-Origin: *`).
Poll every five seconds. Responses are not cached. To show one model, add `?network=NETWORK_ID`:

| Preset | Network ID | Model |
| --- | --- | --- |
| 1–2 | `common-main-8b-v1` | Qwen3-8B |
| 3–6 | `common-qwen3-5-35b-a3b-v1` | Qwen3.5-35B-A3B |
| 7 | `common-qwen3-5-122b-a10b-v1` | Qwen3.5-122B-A10B |

Bonsai networks are included too. Bundled manifests provide the model names, layer counts, and stage
boundaries. Custom network IDs appear when observed, with their ID as the name and unknown coverage
until their manifest is installed on the gateway.

## Response fields

| Field | Meaning |
| --- | --- |
| `v` | Public snapshot schema version, currently 1 |
| `updated_at` | UTC timestamp of the snapshot |
| `online_nodes` | Recently seen peer instances, including nodes that are still loading |
| `serving_nodes` | Online peers advertising an active layer slice |
| `reporting_nodes` | Online peers sending the new telemetry; older directory-only nodes still appear |
| `metrics` | Weighted timing summary across the selected model(s) |
| `metrics.speculative_acceptance_rate` | Accepted draft tokens divided by proposed draft tokens (`0`–`1`, or `null` when unavailable) |
| `metrics.speculative_accepted_tokens`, `metrics.speculative_drafted_tokens` | Five-minute window totals used to compute the acceptance rate |
| `models[]` | Model summaries, coverage, timings, and their nodes |
| `models[].network`, `.model`, `.num_layers` | Network ID, model name, and decoder-layer count |
| `models[].route_ready` | Online advertised slices form a continuous route from layer 0 to the final layer |
| `models[].stages[]` | `{layers: [start, end], peers: count}` for each required stage |
| `models[].nodes[]` | Public node status; fields below |

Each node includes `peer`, `backend` (`mlx`, `nf4`, `cpu`), `gib` (offered memory), `phase`,
`layers` (actively served slice), `assigned_layers` (the slice being prepared), `online`, `last_seen`
(Unix seconds), `telemetry_reporting`, and `metrics`. Layer ranges exclude their end: `[0, 7]`
means layers 0 through 6. A node's `avg_stage_ms` and `avg_network_ms` describe inference sessions
that used its layer slice and peer link. TTFT and generation-rate fields describe chats initiated through
that node's local API.

The mesh status fields map to the existing CSS classes:

| `state` | `display_state` / CSS | Meaning |
| --- | --- | --- |
| `offline` | `off` | Left, failed, or not heard from for 60 seconds |
| `linking` | `lnk` | Preparing, downloading, loading, or moving a slice |
| `unassigned` | `idl` | Online without an active layer assignment |
| `partial` | `grp` | Serving part of a model that has no complete route yet |
| `full` | `ful` | Serving a slice of a model whose cluster has a complete route |

`full` describes the cluster's coverage. A complete route can involve several nodes. Merely covering
every layer with incompatible overlapping windows does not count. Readiness is based on current
announcements, not an active inference probe or a guarantee that every peer computes honestly.

## Performance definitions

`metrics` is present at the top level, per model, and per node:

```json
{
  "sample_count": 0,
  "failed_requests": 0,
  "cancelled_requests": 0,
  "output_tokens": 0,
  "avg_tokens_per_second": null,
  "avg_ms_per_token": null,
  "avg_time_to_first_token_ms": null,
  "avg_network_ms": null,
  "speculative_acceptance_rate": null,
  "speculative_accepted_tokens": 0,
  "speculative_drafted_tokens": 0
}
```

- `sample_count` counts completed generation attempts through updated nodes' local chat APIs.
- `avg_tokens_per_second` is total generated decode intervals divided by their total duration.
- `avg_ms_per_token` uses that same duration divided by the number of intervals. First-token wait is
  excluded from both. A generation of N tokens contributes N−1 decode intervals.
- `avg_time_to_first_token_ms` is the mean time from the local API receiving a request to observing its
  first token, including local queueing and prompt processing. It excludes browser-to-node network time.
- `speculative_acceptance_rate` is the accepted draft-token count divided by the proposed draft-token
  count across the five-minute window (a ratio from `0` to `1`). The matching
  `speculative_accepted_tokens` and `speculative_drafted_tokens` fields are included for context.
  It is `null` until at least one draft token has been proposed. These values are populated only by
  nodes whose local chat API is generating with speculative decoding enabled; plain generation adds
  no speculative samples.
- `avg_network_ms` is the total reported transit time across the participating peer links, averaged over
  completed model requests. Per-link transit is estimated from each successful RPC's elapsed time minus
  the server-reported compute time. A null means no updated peer has reported a usable sample yet.
- `models[].nodes[].metrics.avg_stage_ms` is the mean time that node spent computing its own layer slice
  for each closed inference session. It includes embedding/head work when those run on that node.
- Failed and cancelled generations are counted separately and do not contribute speed samples.
  Invalid requests that never create a generation stream are not counted.
- A zero-token completion has no first-token sample; a single-token completion has no decode-rate
  sample. Missing/unmeasurable values are `null`, not zero. Render `—` in that case.

The rolling window is five minutes, aggregated in ten-second completion-time buckets. A boundary
bucket may fall out up to ten seconds early. Nodes send signed snapshots every 15 seconds; updates
are therefore eventual. Bucket replacement prevents retries from counting the same chat twice.
Different process sessions retain distinct samples across a node restart. These are self-reported
observations, not independently attested benchmarks. `reporting_nodes` indicates the current reporting
coverage. External clients that bypass updated nodes' chat APIs are not measured. Historical speed
samples can remain visible for five minutes after their source node goes offline.

Dashboard report schema v2 adds the stage and network counters. Schema v3 adds speculative draft and
acceptance counters. Schema v4 adds `served_tokens` and `served_compute_ms` (work done for routes) and
names the machine with a device key (`device`, `device_pub`, `device_sig`; see below). The gateway
continues to accept v1–v3 reports; fields absent from older schemas remain null until newer telemetry arrives.
Deploy the gateway before nodes running v4: an older gateway rejects v4 reports, which only costs those
nodes their dashboard visibility. An older peer that does not accept the
optional timing value on session close still receives the original close request as a compatibility fallback.

The public endpoint never returns bearer tokens, peer URLs, IP addresses, keys, prompts, or answers.
The report schema contains only identifiers, node state, and numeric aggregates. Nodes can disable
the new heartbeat/timing upload by setting `COMMONNET_TELEMETRY=0`; ordinary peer discovery remains
public and continues to populate basic node/model status.

## Connect the supplied HTML tiles

Give the two values IDs:

```html
<h3 id="avg-tps">—</h3>
<h3 id="avg-token-latency">—</h3>
```

Then poll this API from `mesh.js` or another script:

```js
const statsUrl = 'https://gateway-production-b820.up.railway.app/api/network-stats';
const format = (value, suffix = '') =>
  value == null ? '—' : `${value.toFixed(1)}${suffix}`;

async function refreshNetwork() {
  try {
    const response = await fetch(statsUrl, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Network status: ${response.status}`);
    const stats = await response.json();
    document.getElementById('avg-tps').textContent =
      format(stats.metrics.avg_tokens_per_second);
    document.getElementById('avg-token-latency').textContent =
      format(stats.metrics.avg_ms_per_token, ' ms');
    document.getElementById('mesh-sum').textContent = `${stats.online_nodes} nodes online`;

    // Your SVG renderer can group by stats.models, then draw each model.nodes array.
    // Use node.display_state for off / lnk / idl / grp / ful and node.layers for slices.
    window.dispatchEvent(new CustomEvent('common:network-stats', { detail: stats }));
  } catch (error) {
    document.getElementById('avg-tps').textContent = '—';
    document.getElementById('avg-token-latency').textContent = '—';
    document.getElementById('mesh-sum').textContent = 'Status unavailable';
    // Keep the previous mesh visually distinct as stale; a failed fetch is not zero nodes.
    console.warn(error);
  } finally {
    setTimeout(refreshNetwork, 5000);
  }
}
refreshNetwork();
```

Always render names/IDs with `textContent`, rather than inserting network data as HTML. Use
`last_seen * 1000` with `new Date()` for JavaScript timestamps.

## Deployment and retention

Deploy the gateway from `gateway/Dockerfile`; its existing requirements cover the new endpoint.
Updated nodes automatically report when joined with a gateway configured. Update and restart
`common join` to enable their timing feed. Older gateway versions can reject reports without
breaking the node's chat or contribution.

Both the existing directory and the new dashboard are bounded in-memory services intended for one
gateway worker/instance. Recently seen nodes are retained for up to an hour; at most 5,000 peers
and 5,000 reporting sessions are held. A gateway restart resets offline history, and active node
reports rebuild their recent metrics. `history_since` exposes when this gateway process started.
Multiple gateway workers/replicas require a shared store before their snapshots can be combined.

## Contribution log (closed beta)

`GET /api/contributions` returns running totals per machine, for the closed beta and as the measured
basis for a later credit system. Each machine is named by its **device**: a key kept at
`~/.common-network/net/device.key`, separate from the per-network peer key, so a machine keeps one name
across restarts, updates and model switches. `common status` prints it (from `net/device.id`). The
device key signs for each peer key it reports with (`device_sig`), so one machine cannot claim another's
peer. Older nodes appear under their peer ID.

| Field | Meaning | Measured by |
|---|---|---|
| `tokens` | Positions this machine's layers processed for routes: prefill, decode and speculative verification. A retried step counts once. | the node (self-reported, signed) |
| `compute_ms` | Time those layer steps took | the node |
| `requests_served` | Route sessions this machine served to a clean close | the node |
| `chat_completed`, `chat_failed`, `output_tokens` | Chat requests made through this machine's own local API (use, not contribution) | the node |
| `uptime_s` | Seconds serving layers, summed between heartbeats at most 60 s apart | the gateway |
| `first_seen`, `last_seen`, `networks`, `backends` | When and where the machine reported | the gateway |

Work done by `--local-front` (a machine running the first layers of its own chats) is not counted.
Requests a machine routes through its own layers are counted like anyone else's.

Every accepted report appends one JSON line to the file named by `COMMON_CONTRIBUTIONS_LOG`, holding
only what grew since that node's previous report. The totals are the sum of the lines; on start the
gateway replays the file, so resent buckets are never counted twice across a restart. **Put the file
on a persistent volume** (on Railway, attach a volume and set, for example,
`COMMON_CONTRIBUTIONS_LOG=/data/contributions.jsonl`). Without the variable the totals live only in
memory and reset on deploy. Each line is also printed to stdout, prefixed `contribution `, as a backup in
the host's logs. The file grows by about 200 bytes per node per 15 seconds and is kept until deleted.

These numbers are not proof of work: nothing yet checks that a peer computed what it claims. Uptime is
the only figure the gateway measures itself.
