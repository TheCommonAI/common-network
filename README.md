# Common Network

![Common.](assets/common-banner.png)

**v0.5.1 Beta** · [Release notes](RELEASE-v0.5.md)

Common is an open, peer-to-peer network for running a language model across multiple computers. Instead of requiring every machine to hold the whole model, each participating peer downloads and serves a window of consecutive model layers. A request is processed by routing it through peers that collectively cover the model.

There is no central inference server. Peers discover one another, exchange signed peer records, select routes from the live network, and can recover when a peer disappears if another peer can serve the same layer window. Anyone can join; there are no accounts, allowlists, or contribution gates on the current layer network.

Common is built by Common AI Inc., a South Australian nonprofit. [commonai.com.au](https://commonai.com.au)

## What the network is — and is not

- **A distributed inference network.** Participating machines contribute memory and compute by serving part of a shared model.
- **A peer mesh, not a central model API.** Each node runs locally; clients build a route through currently available peers.
- **A shared model, not model training.** The network distributes inference across machines. It does not train or fine-tune a model.
- **Open and cooperative.** Peers are not cryptographically proven to compute honestly. Treat model output and peer behaviour accordingly.
- **Not private by default.** Peers processing a request can see the token IDs or hidden-state activations that pass through them. Do not send confidential prompts over a public network.

The former v0.1 whole-model gateway — with a central registry, specialist routing, contribution gate and catalogue — has been retired. The current system is the layer network described here. The small gateway service is only used for manifests, seed information and a rendezvous directory; it does not route inference requests.

## Install and join

macOS / Linux:

```bash
curl -fsSL https://commonai.com.au/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://commonai.com.au/install.ps1 | iex
```

Check the machine before joining, then start a peer:

```bash
common join --check
common join
```

The built-in network is **Nanbeige4.3-33B**. A peer chooses a layer window that fits its available memory, downloads only the weights for that window, and starts serving it. `common join` starts a background service and returns to your terminal. Then ask the network:

```bash
common ask "What's a good way to learn recursion?"
common peers
common status
```

The local OpenAI-compatible API is available at `http://127.0.0.1:11440/v1` while the node is running. Any OpenAI-compatible client can use that endpoint.

Nanbeige can search automatically in its default chat mode. Try `common ask "Search the web for the latest Python release and link the release notes."` The local chat API sends the model's search query to DuckDuckGo and gives Nanbeige up to four result snippets to answer from. It needs internet access and uses no API key. To turn search off, send `"web_search": false` in the API request or set `COMMONNET_WEB_SEARCH=0` in the chat service environment.

`common join --check` is a read-only preflight: it checks the platform, memory, disk and required network access without installing or downloading anything. The first actual join sets up its private runtime and downloads the selected layer weights. Joining runs in the background by default, recovers from crashes, and resumes at login. Use `common leave` to stop, or `common join --foreground` for a terminal session that ends with Ctrl+C.

## Background contribution and resource controls

```bash
common join                         # background service is the default
common join --foreground            # optional terminal session
common join --controls             # reopen menu-bar / tray controls
common leave                       # stop and remove automatic startup
```

Click the Common icon in the macOS menu bar for a glass contribution panel.
Choose **Quiet**, **Balanced**, **Maximum**, or **Automatic**, then **Apply changes**.
Automatic lets the runtime size the contribution; Maximum keeps its memory reserve
while offering all CPU threads and full inference duty cycle. Quiet and Balanced
use smaller hardware-relative budgets. Use **Custom** to reveal the labeled sliders.
Choosing a preset only stages a change: the node restarts when you apply it.

The native macOS panel uses a compiled SwiftUI menu-bar app on macOS 14+,
standard system controls and automatic layout, with macOS 26 glass button styles.
It keeps the supplied Common tray artwork. Its universal executable is bundled,
so Xcode and PyObjC are not needed on supported Macs. Older Macs retain the
legacy AppKit fallback. See [native build notes](commonnet/native/README.md).
 On
Windows/Linux, choose **Resource controls…** from the tray to open the local
control panel. Set model memory, CPU threads, compute share and bandwidth,
then **Apply changes**. This restarts the node and reconnects with the saved
limits. **Stop contributing** removes automatic startup. The panel refreshes
live node status every two seconds: it distinguishes model loading, stopped or
unresponsive nodes, and online nodes waiting for enough peers to cover the
model. Recent model-session counts include local work and are not a count of
requests served to other people.

The same settings are available from the terminal:

```bash
common join --memory-gib 4 --cpu-threads 2 --compute-percent 50 --bandwidth-mbps 20
common join --no-controls --no-api --local-front off --no-draft  # lean headless node
```

Limits are cooperative. `--memory-gib` (alias `--stage-gib`) is the estimated
model-layer allocation budget, not a hard cap on total RAM/VRAM. The chat API,
local front, drafter, runtime and request caches need additional memory; the
second example disables optional model components. `--cpu-threads` limits PyTorch
and common BLAS thread pools. Compute share inserts idle time after served
inference steps on CPU/GPU; it does not cap instantaneous GPU utilization,
model loading, or local chat/draft work. Bandwidth is a shared average budget
for model downloads and HTTP/WebRTC peer request/response payloads, excluding
protocol overhead, discovery and dependency installation. Individual requests
can burst; very low limits can make peers time out. Defaults are automatic
memory/threads, full compute share, and unlimited bandwidth.

macOS uses a LaunchAgent; Windows uses a per-user scheduled task. Both resume
at login after a restart. Linux uses `systemd --user`; enable lingering with
`loginctl enable-linger "$USER"` to resume at boot before login and stay online
after logout. Windows retries failures every minute, up to 999 retries;
macOS/Linux keep retrying. A sleeping or offline computer cannot serve.

Settings are stored in `~/.common-network/service.json`. Re-run
`common join` with new flags to replace them. Desktop controls
install their small UI dependencies on first use. Linux requires a supported
tray (such as an AppIndicator-capable desktop); when unavailable, the sliders
open in a browser. Use `--no-controls` for headless installations.

Logs: `~/.common-network/join.log` on macOS/Windows, or
`journalctl --user -u common-join.service -f` on Linux. Detailed join logs also
live under `~/.common-network/logs/`.

## Models and network selection

The default network is Nanbeige4.3-33B. Other available layer-network manifests include:

| Network name | Model |
|---|---|
| `nanbeige4.3-33b` | Nanbeige4.3-33B (default) |
| `qwen3.5-35b-a3b` | Qwen3.5-35B-A3B |
| `qwen3.5-122b-a10b` | Qwen3.5-122B-A10B |
| `qwen3-8b` | Qwen3-8B (smaller fallback) |
| `bonsai-8b` | Ternary Bonsai 8B |
| `bonsai-27b` | Ternary Bonsai 27B |
| `bonsai-2-9b` | Ternary Bonsai 2 9B |
| `bonsai-2-27b` | Ternary Bonsai 2 27B |

For Nanbeige, two 4-bit MLX peers need about 10.5 GiB each in Common’s offered model budget. Three peers can cover the model with about 7.7 GiB each. Common aims for a complete three-peer route and can use four or more when memory is tighter. If the current route needs more than three peers, joining peers prefer a memory-fitting slice that shortens it. These are placement estimates. Leave room for runtime memory and the KV cache. On separate machines, join one at a time and wait for each to appear before starting the next.

Join a named network or a network someone else is hosting:

```bash
common join --network qwen3-8b
common join --network bonsai-8b
common join --network https://<peer-url>
common join --network /path/to/manifest.json
```

The `--model NAME` spelling is an alias for `--network NAME`. Peers must use the same network manifest; peers belonging to different network IDs do not mix. See [docs/bonsai-networks.md](docs/bonsai-networks.md) for model and backend details.

Node-count presets are deployment plans, not guarantees that a given collection of machines has enough memory:

```bash
common join --1
common join --2
common join --3
common join --4
common join --5
common join --6
common join --7
# equivalent long form:
common join --nodes 7
```

Presets 1–2 target Qwen3-8B, presets 3–6 target Qwen3.5-35B-A3B, and preset 7 targets Qwen3.5-122B-A10B. These larger preset deployments require suitable hardware and further real-world validation; the preset does not establish that the model will fit or perform well on arbitrary machines. Use the named manifest options above for the currently documented Qwen3-8B and Bonsai networks.

## How it works

1. **A manifest defines the network.** It identifies the model, its layer count, stage boundaries, weight sources and memory estimates.
2. **A peer joins and selects a window.** It checks live coverage and chooses a boundary-aligned range of layers that fits its available memory. It downloads only the tensors needed for that range.
3. **Peers discover and update one another.** Nodes publish signed records and exchange them through gossip. A small rendezvous directory helps new peers find initial contacts; peers verify signatures themselves.
4. **A client builds a route.** The route covers the model's layers in order. The first and last stages handle token embedding and output-token generation; intermediate peers forward hidden states.
5. **The route adapts to churn.** If a peer becomes unavailable, the client can use another peer serving the identical window or wait for one to appear. A network cannot answer if required layers have no available host.

Compatible routes send token IDs at the edges and hidden states between peers. Intermediate activations are compressed to int8 by default (one scale per group of 64 values); this is lossy compression. Older peers can use the compatible legacy path. See [docs/token-id-chain.md](docs/token-id-chain.md) and [docs/network-contract.md](docs/network-contract.md).

Peers normally connect without router configuration or port forwarding. HTTPS/tunnel connectivity is available as a fallback, while compatible peers can establish direct WebRTC data channels through NAT traversal. Direct connectivity is not guaranteed on every network. See [docs/automatic-connectivity.md](docs/automatic-connectivity.md).

## Hardware and performance

The documented backends are:

- **Apple silicon:** MLX.
- **NVIDIA GPUs on Windows/Linux:** PyTorch with NF4 quantisation.
- **CPU-only machines:** PyTorch; slower and substantially more memory-hungry.

Hardware support and validation are not uniform. The NVIDIA path and several CPU/model combinations still need real-hardware end-to-end testing; consult the current caveats in [docs/layer-network.md](docs/layer-network.md) and [docs/bonsai-networks.md](docs/bonsai-networks.md).

A larger number of peers primarily adds capacity, layer coverage and replicas. It does not automatically make a single request faster: latency depends on the number of stages, the slowest peers and links, available bandwidth, hardware and route quality. Published benchmark or model-quality claims should be treated as specific measurements, not general guarantees.

## Commands

```bash
common join --check                 # inspect whether this machine can join
common join                         # join the default Qwen3-8B network
common ask "..."                    # ask through this machine's local API
common peers                        # see peers currently known to this node
common status                       # inspect this node and its served window
common report                       # review and optionally submit evidence of a bad answer
common privacy                      # explain what the client and network retain
common leave                        # leave the network
common help                         # show available commands
```

Run `common help` for the full command and flag reference. Developer and operational documentation starts at [docs/README.md](docs/README.md).

## Honest limits and security

- **Prompts are not private.** Peers can inspect the token IDs and activations they process. Local-first options can keep the original prompt tokens on the client machine, but the hidden states sent to peers are still sensitive and are not encrypted from those peers.
- **Peer computation is not verified.** Signatures establish the origin and integrity of peer records, not that a peer ran the advertised model correctly.
- **Availability is not guaranteed.** Every required layer must be hosted by an online peer. Replicas improve resilience only when another peer has the same layer window.
- **Open participation has trade-offs.** Only join machines and networks you are willing to expose to other participants. Do not treat output from an untrusted peer as verified.
- **Transport encryption is not end-to-end privacy.** HTTPS and WebRTC/DTLS protect traffic in transit; they do not hide activations from the peers that process them.

Read [SECURITY.md](SECURITY.md) before joining a public network.

## For developers

Start with [docs/README.md](docs/README.md) for the documentation index and repository map. The main implementation is in `commonnet/`; `join/` contains the node launcher, `common/` contains the CLI, and `gateway/` contains the small discovery/rendezvous service.

Useful starting points:

- [Layer network guide](docs/layer-network.md)
- [Network design contract](docs/network-contract.md)
- [Token-ID chain](docs/token-id-chain.md)
- [Bonsai model manifests](docs/bonsai-networks.md)
- [Security notes](SECURITY.md)
- [Common network library map](commonnet/README.md)

Run tests from the repository root or the relevant package directory:

```bash
cd join && python3 test_join.py
cd common && python3 test_common.py
cd gateway && python3 -m unittest discover tests
python3 -m unittest discover -s commonnet/tests -t .
```

End-to-end model checks and manual test plans are documented in [e2e/](e2e/README.md) and [docs/](docs/README.md).

## Licence

Copyright (C) 2026 Common AI Inc. Licensed under [AGPL-3.0](LICENSE).

The AGPL-3.0 is intended to keep network-deployed modifications available under the same licence. Section 13 applies when you run a modified version as a network service. The gateway's `GET /source` endpoint identifies the source repository used by a running instance; forks should set `SOURCE_URL` to their own source.

The Common name is not covered by the AGPL. Fork the code freely, but give a modified fork its own name. See [TRADEMARK.md](TRADEMARK.md).
