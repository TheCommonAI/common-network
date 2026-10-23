# Decentralised layer network (design contract)

Goal: `common join` turns any machine into a peer that serves a slice of one shared model. No central
server: peers find each other by gossip, build routes themselves, and a dropped peer is replaced
mid-generation. The old v0.1 whole-model-per-node + gateway-composition system is NOT touched here.

## Terms
- **manifest**: JSON naming the model and its fixed *stage boundaries*. Every peer serves a window that is a
  union of consecutive stages, so windows always start/end on a boundary.
- **peer record**: a small signed JSON a peer publishes about itself. Peers gossip records. No directory.
- **seed**: any reachable peer URL. A joining peer needs one seed (default list + `--network URL|FILE`).
- **replica**: another peer with the *identical* window. Failover only ever goes to a replica (or to a peer that
  joins with that window while we wait), because the client replays exactly what it sent to the dead peer.

## Manifest (`commonnet/mesh/manifest.py`)
```json
{"id": "common-8b-v1", "version": 1,
 "model": {"name": "Qwen3-8B", "num_layers": 36,
           "repos": {"mlx": "mlx-community/Qwen3-8B-4bit", "nf4": "unsloth/Qwen3-8B-bnb-4bit"},
           "client_repo": "mlx-community/Qwen3-8B-4bit"},
 "boundaries": [0, 9, 18, 27, 36],
 "gib_per_layer": {"mlx": 0.125, "nf4": 0.17}, "fixed_gib": 0.3,
 "min_replicas": 2, "seeds": ["https://example.trycloudflare.com"]}
```
`Manifest.load(path_or_dict)`, `.stages()` -> [(a,b)...], `.window_gib(a,b,backend)`, `.windows_fitting(gib, backend)`.
Any peer serves its manifest at `GET /manifest`. A manifest's `id` is the network id; records for another id are dropped.

## Peer record (`commonnet/mesh/records.py`)
```json
{"v":1,"network":"common-8b-v1","peer":"<32 hex = sha256(pub)[:16]>","pub":"<b64 ed25519 public key>",
 "url":"https://x.trycloudflare.com","token":"<bearer token for this peer's block server>",
 "window":[9,18],"backend":"mlx","state":"up","gib":10.5,"seq":1761000000123,"ts":1761000000.1,"sig":"<b64>"}
```
- `state`: `loading` (not serving yet) | `up` | `leaving` (tombstone). `window` may be null while loading.
- Signed (ed25519) over canonical JSON (sorted keys, no spaces) of every field except `sig`. `peer` must equal the hash
  of `pub`. A record with a newer `seq` for the same `peer` replaces an older one. Re-signed every 15 s (fresh `ts`).
- Live = verified, same network, `state=="up"`, `now - ts <= 60`, `ts <= now + 120`.
- The block `token` is public to anyone who reads the mesh: it keeps random scanners off a tunnel URL, nothing more.
  Honest limit: the network is open; do not put private prompts on it. (Activations also reveal content to peers.)
API: `Identity.load_or_create(path)` (private key file mode 0600), `Identity.sign_record(fields)->record`,
`verify_record(record)->bool`, `is_live(record, now)`.

## Gossip (`commonnet/mesh/gossip.py`)
`PeerTable(network_id, now=time.time)`: `merge(record)->bool`, `merge_many(list)`, `live(now=None)`, `get(peer)`,
`export(limit=200)` (live + tombstones < 5 min old), `sample(k, exclude)`.
Server endpoints on every peer's block server, **unauthenticated** (public data, signed):
`GET /manifest`, `GET /peers` -> {"records":[...]}, `POST /gossip` body {"records":[...]} -> merges, replies with ours.
`Gossiper(table, manifest, self_record_fn, seeds)`: thread, every 5-10 s (jitter) exchanges with 3 random live peers
plus one seed; drops a peer from its table's *candidates* after 3 failed exchanges.

## Placement (`commonnet/mesh/placement.py`, pure functions)
- `coverage(manifest, live)` -> per-stage count of live peers whose window contains the stage.
- `choose_window(manifest, live, gib, backend, me=None)` -> (a,b): among windows that fit `gib`, maximise
  sum over covered stages of 1/(1+coverage) (uncovered stages dominate), then prefer more stages (fewer hops), then prefer a
  window identical to an existing one with the fewest replicas.
- `should_move(manifest, live, me)` -> new window or None: only when some stage has zero live coverage, `me` is in a
  window whose every stage has coverage >= 2, and `me` can fit the uncovered window; cooldown handled by the caller (5 min).
- `best_route(manifest, live, prefer_backend=None)` -> list[Hop(window, primary, alternates)]: shortest path over
  boundaries by hop count, then lowest penalty (older record, `loading`, backend differing from the previous hop). `alternates` are other live
  peers with the identical window (same backend first). Raises `NoRoute(boundary)` naming the first uncovered boundary.

## Node (`commonnet/node.py`, what `common join` runs)
1. Load/create identity (`~/.common-network/net/<network>/identity.key`).
2. Fetch manifest from a seed; gossip once to learn peers; detect hardware: macOS arm64 -> `mlx`, NVIDIA CUDA -> `nf4`,
   else refuse ("GPU only"); `gib` = usable memory (Mac: 0.62*RAM; CUDA: free VRAM - 1.5).
3. `choose_window`; publish record `loading`; download only that window (`hub.Hub(repo).fetch(layers=window)`);
   start the block server (+ Cloudflare quick tunnel via `net/tunnel.py`) with its own random token; publish `up`.
4. Loop: re-sign record every 15 s with the *current* tunnel URL (tunnels change URL; the mesh learns it). Every 60 s run
   `should_move`; if it says move, publish `loading`, restart the block on the new window.
5. Ctrl+C / SIGTERM: publish `leaving`, stop block and tunnel. Local test mode: `--advertise URL` and `--no-tunnel`
   (serve on 127.0.0.1:PORT and publish that URL) and `--stub` (identity block, no weights).
`python -m commonnet.node --network URL|FILE [--port N] [--backend B] [--stub] [--no-tunnel --advertise URL]`.

## Client (`commonnet/netclient.py`)
`Network(seeds|manifest)`: `refresh()` (gossip pull), `route()`, `transport()`.
`HttpTransport` gains `on_exhausted(name, window)` -> fresh list of (url, token) replicas and `wait_s` (default 30): when every known
server of a hop is dead the transport refreshes the mesh and waits for a peer with that window, then replays and carries on.
`python -m commonnet.serve --network ... [--port 11440]`: OpenAI-compatible `/v1/models`, `/v1/chat/completions` (stream
and non-stream; greedy and temperature/top-p sampling); builds the route per request, so the next request uses current peers.

## Join (`join/layers.py`, `join/join.py --network`)
`common join --network [URL|FILE]` ensures a runtime (venv + pinned deps + commonnet source; env `COMMONNET_SRC` uses a
checkout), then runs `commonnet.node`. Without `--network`, `common join` behaves exactly as today.

## Out of scope for this pass
Gateway/desktop integration, removing the v0.1 system (separate PR), per-token payments/credits, NAT-less direct links,
sub-window routing, proof that a peer computes honestly.
