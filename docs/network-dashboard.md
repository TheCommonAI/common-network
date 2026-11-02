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
| `v` | Schema version, currently 1 |
| `updated_at` | UTC timestamp of the snapshot |
| `online_nodes` | Recently seen peer instances, including nodes that are still loading |
| `serving_nodes` | Online peers advertising an active layer slice |
| `reporting_nodes` | Online peers sending the new telemetry; older directory-only nodes still appear |
| `metrics` | Weighted timing summary across the selected model(s) |
| `models[]` | Model summaries, coverage, timings, and their nodes |
| `models[].network`, `.model`, `.num_layers` | Network ID, model name, and decoder-layer count |
| `models[].route_ready` | Online advertised slices form a continuous route from layer 0 to the final layer |
| `models[].stages[]` | `{layers: [start, end], peers: count}` for each required stage |
| `models[].nodes[]` | Public node status; fields below |

Each node includes `peer`, `backend` (`mlx`, `nf4`, `cpu`), `gib` (offered memory), `phase`,
`layers` (actively served slice), `assigned_layers` (the slice being prepared), `online`, `last_seen`
(Unix seconds), `telemetry_reporting`, and `metrics`. Layer ranges exclude their end: `[0, 7]`
means layers 0 through 6. A node's timing summary describes chats initiated through that node's
local API, including their full distributed route; it is not a benchmark of that GPU alone.

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
  "avg_time_to_first_token_ms": null
}
```

- `sample_count` counts completed generation attempts through updated nodes' local chat APIs.
- `avg_tokens_per_second` is total generated decode intervals divided by their total duration.
- `avg_ms_per_token` uses that same duration divided by the number of intervals. First-token wait is
  excluded from both. A generation of N tokens contributes N−1 decode intervals.
- `avg_time_to_first_token_ms` is the mean time from the local API receiving a request to observing its
  first token, including local queueing and prompt processing. It excludes browser-to-node network time.
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
