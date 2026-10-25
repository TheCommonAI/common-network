# Common Network

![Common.](assets/common-banner.png)

**v0.4**

Common Network is a distributed LLM inference network. It splits one large
model's layers across volunteer machines, in the style of
[Petals](https://github.com/bigscience-workshop/petals), so a model too big for
one computer can run on many ordinary ones. There is no central inference
server: peers find each other, build their own routes, and replace a peer that
drops out mid-generation.

A South Australian nonprofit building decentralised AI. [commonai.com.au](https://commonai.com.au)

## Install and join

macOS / Linux:

```bash
curl -fsSL https://commonai.com.au/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://commonai.com.au/install.ps1 | iex
```

Open a new terminal, then:

```bash
common join                      # serve a slice of the default model
common ask "What's a good way to learn recursion?"
```

`common join` sets itself up on first run (private Python environment, model
runtime, code), picks the layers the network needs most that fit your memory,
downloads only those layers, and starts serving. Ctrl+C leaves cleanly. Add
`--permanent` to keep serving after reboots.

Backends: Apple silicon (MLX), NVIDIA GPUs on Windows/Linux (NF4), and CPU-only
machines (PyTorch, slower; routes prefer GPU peers). The NVIDIA path has not yet
been verified on real hardware; see the status notes in
[docs/layer-network.md](docs/layer-network.md).

### Choosing a network

```bash
common join --network bonsai-8b        # also: bonsai-2-9b, bonsai-27b, bonsai-2-27b, qwen3-8b
common join --network URL              # a network a friend runs (any peer's URL)
common join --network manifest.json
```

A network is defined by a manifest: the model and the fixed layer boundaries
every peer's window is made of. See [docs/bonsai-networks.md](docs/bonsai-networks.md).

### Useful `join` flags

| Flag | Effect |
|---|---|
| `--local-front N\|auto\|on` | Hold the first layers on your machine and run them in your own chat client, so peers never see your prompt tokens (they get layer-N hidden states, which are not encrypted) and one network hop disappears. Costs memory (about 1 GB for 9 layers of Qwen3-8B 4-bit). Off by default. |
| `--direct` | Experimental. Serve over TLS on your own public address and be found through a Kademlia DHT, with the Cloudflare tunnel only as fallback. A plain `common join` already tries this automatically; `--no-direct` forces tunnel only. |
| `--no-draft` | Turn off speculative decoding in the local chat API. |
| `--layers A-B` | Serve exactly this window instead of the one the network picks. |
| `--no-api` / `--api PORT` | Skip or move the local OpenAI-compatible API (default `http://127.0.0.1:11440/v1`). |

## How it works

- **Token-ID chain.** The client sends token IDs to the first peer, peers pass
  hidden states along the chain, and the last peer returns the next token ID.
  Only small activations travel between peers. See
  [docs/token-id-chain.md](docs/token-id-chain.md).
- **Link compression.** Hidden states between peers are sent as int8 by default
  (one scale per 64 values). `COMMONNET_ACTIVATION_CODEC=int4_group` selects int4
  (smaller, with a quality cost); `float` disables compression. Older peers fall
  back automatically.
- **Common Flash speculative decoding.** On by default for greedy requests
  (`temperature` 0; `common ask` sends 0). A small local drafter guesses several
  tokens and the network verifies them in one pass, with output identical to
  non-speculative greedy decoding. The drafter is MLX on Macs and PyTorch
  elsewhere. Not available for networks whose model cannot roll back (Bonsai 2
  9B/27B). Gains on a real wide-area network vary and are still being measured.
  See [docs/flash-drafter.md](docs/flash-drafter.md).
- **Resilience.** If another peer holds the same layers, the client switches to
  it and replays what it sent; if a stage loses every peer, a spare peer moves
  over.
- **Discovery.** Peers gossip signed records. The production gateway also runs a
  small directory (rendezvous) so `common join` needs no typed address; it never
  has to be trusted, since readers verify signatures. Direct mode adds a DHT.

Everything is documented in [docs/layer-network.md](docs/layer-network.md).

### Honest limits

- The network is open. Peers see activations, which reveal a lot about the text.
  Do not send private prompts over a public network.
- Nothing proves a peer computes honestly; treat it as cooperative, like early Petals.
- Single-stream speed is bounded by the number of stages and the network. More
  peers add capacity and resilience, not speed.
- Direct mode and the NVIDIA and CPU paths are experimental or lightly tested.

## Commands

```bash
common join           # put this machine on the network
common ask "..."      # ask the network (or `common` for an interactive session)
common ask --temperature 0.7 "..."   # sampled instead of greedy (no speculation)
common peers          # who else is on the network right now
common status         # your node: health and requests served
common report         # an answer was bad: send the evidence (only after you approve it)
common privacy        # what the network keeps, and what it doesn't
common leave          # take your machine off the network
common help           # everything else
```

Any OpenAI-compatible client can use the local API at `http://127.0.0.1:11440/v1`
while `common join` is running.

## Gateway

The v0.1 gateway (node registry, request routing, specialist composition, the
catalogue) has been retired. `gateway/` is now a small signpost service: `/health`,
`/manifest`, `/seeds`, and the rendezvous directory (`POST /announce`,
`GET /directory`, `POST /check`) that lets `common join` find peers without a
typed address. Everything else returns 410 Gone. Details:
[docs/layer-network.md](docs/layer-network.md#gateway-directory-rendezvous).

```bash
cd join && python test_join.py
cd common && python test_common.py
cd gateway && python -m unittest discover tests
```

## Security

The network is open and cooperative: peers see activations and nothing proves a
peer computes honestly. What a stranger can and cannot do, and the known limits,
are in [SECURITY.md](SECURITY.md).

## Licence

Copyright (C) 2026 Common AI Inc. Licensed under [AGPL-3.0](LICENSE).

Chosen deliberately: copyleft means anyone running a modified version of Common
as a network service must release their changes back to the commons —
structurally preventing this from being taken closed, in keeping with the
project's anti-enclosure thesis.

That obligation is section 13, and this gateway honours it: **`GET /source`**
returns the repository a running instance was built from. If you fork Common
and run it as a service, set `SOURCE_URL` to your own repository — your users
are entitled to the code you are actually running, not to ours.

The **name** is not covered by the AGPL. Fork the code freely; give your fork
its own name. See [TRADEMARK.md](TRADEMARK.md) — it is short, and permissive
about everything except calling a modified version "Common".
