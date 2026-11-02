# commonnet: the layer network library

This is the code behind the layer network: a model's decoder layers are split across peers, and the peers pass
small activations to each other. Start with [docs/network-contract.md](../docs/network-contract.md) (the design
contract), then [docs/layer-network.md](../docs/layer-network.md) (how it is used).

The `common` and `common join` commands in `common/` and `join/` run this library. You rarely need to run
these modules by hand.

## Where to start reading

1. `node.py`: one peer. This is what `common join --network` runs: it loads its layers, joins the mesh and serves.
2. `mesh/`: how peers find each other and agree on who serves what (signed records, gossip, placement).
3. `netclient.py`: the client side. It builds a route through the peers and survives one dropping out.
4. `net/server.py` and `backends/`: the block server and the per-hardware code that runs the layers.

## The layout

**Running the network (current)**

| Module | What it does |
|---|---|
| `node.py` | A peer: `common join --network`. |
| `netclient.py` | The client's view: find peers, build a route, fail over. |
| `serve.py` | The local OpenAI-compatible API that `common ask` and chat talk to. |
| `speculative.py` | Speculative decoding: a small local drafter proposes tokens, the network checks them in one pass. |
| `localfront.py` | Optional: this machine holds the first layers and runs them in the chat client's process. |
| `direct.py` | Experimental direct mode: how a node picks the address it publishes without a tunnel. |
| `preflight.py` | Checks before a download: disk and memory for the layers chosen (`common join --check`). |

**Shared pieces**

| Module | What it does |
|---|---|
| `block.py` | The core of a block server: decoder layers `[start, end)` of a checkpoint. |
| `backends/` | The per-hardware block code: `mlx_block.py` (Apple silicon), `nf4_block.py` (NVIDIA), `cpu_block.py` (PyTorch on CPU), and the Hadamard packs used by Bonsai 2. |
| `weights.py` | Reads only the named tensors from a Hugging Face safetensors checkpoint. |
| `hub.py` | Downloads only the tensors a machine needs from Hugging Face. |
| `wire.py` | How hidden states are encoded on the wire. |
| `net/` | Transport: the block server, the HTTP client, tunnels, NAT and TLS, WebRTC links. See [docs/direct-mode.md](../docs/direct-mode.md). |
| `mesh/` | Discovery: signed peer records, gossip, the manifest, placement, layer assignment, and the rendezvous directory client. |
| `dht/` | Seedless peer discovery (experimental). See [dht/README.md](dht/README.md). |

**The older fixed-member flow**

These drive a fixed set of machines listed in a `members.json` file. They predate the layer network. The
end-to-end scripts in `../e2e/` still use them. How to run them: [../docs/fixed-route.md](../docs/fixed-route.md).

| Module | What it does |
|---|---|
| `member.py` | Starts one member: the block server plus its tunnel, and writes its endpoint JSON. |
| `coordinator.py` | Reads the member endpoints and generates through them. |
| `plan.py` | Decides which layers each member serves, from memory. |
| `client.py`, `client4.py` | Generation clients for the fixed members. |

**Measurement**

| Module | What it does |
|---|---|
| `bench/` | The latency harness, the speculative and batching benchmarks, and the GPU profiler. See [../docs/network-bench.md](../docs/network-bench.md). Its output goes to `bench-results/`, which is not committed. |

**Tests**

`tests/` holds the unit tests. Many need PyTorch or MLX, so they skip or fail on a machine without them.
From the repository root: `python3 -m unittest discover -s commonnet/tests -t .`
