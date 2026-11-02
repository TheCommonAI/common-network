# The retired v0.1 network

v0.1 was **one whole model per node, routed by a central gateway**:

- A node (`common join`) downloaded a complete GGUF, ran `llama-server` locally,
  opened a Cloudflare tunnel, and **registered with the gateway**.
- The gateway scored nodes, picked one (or composed several "specialists"),
  and proxied an OpenAI-style request to it.
- Clients (`common ask`, the desktop app, `chat/chat.py`) talked to the
  gateway, never to nodes.

The layer network replaced this: one model split across peers, no central
inference server. Peers gossip signed records, clients build routes themselves,
and a dropped peer is replaced mid-generation. See
[layer-network.md](layer-network.md).

## What shipped in its place (v0.4)

- A plain `common join` is the layer network. `--legacy`, `--lan`, and a
  `--model` that is not a layer-network name still *parse*, then tell you the
  old gateway is gone (it answers 410 Gone) rather than crashing old scripts.
- `gateway/` is a signpost: `/health`, `/manifest`, `/seeds`, and the
  rendezvous directory. Everything else returns 410 Gone.
- `common ask` talks to the node's local API at `127.0.0.1:11440`, not to the
  gateway.
- The unreleased v0.2 RPC-group prototype (`join/group.py`, `common rpc`, and
  friends) was removed. Those flags still parse and say `removed: … use
  common join --network`.

Code that only the old node needed (`join/worker.py`, `join/runtime.py`, the
legacy path in `join/join.py`, and the leftover verbs in `common/common.py`)
is still in the tree so installed services that recorded `--legacy` get a
clear failure instead of a missing file. It is not how a new join works.

## What was not rebuilt

Specialist composition, the contribution ledger, mobile peers, and the old
dashboard have no layer-network equivalent. Whether to bring any of that back
is a product decision, not a leftover TODO in this folder.
