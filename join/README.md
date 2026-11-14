# `common join`

Turns this machine into a peer of the layer network: it serves a slice of the
shared model's layers. Other peers find it, send it hidden states, and get
hidden states back.

This folder is what the installer puts on a user's machine (`join.py` plus its
siblings). Installed clients fetch these files by path from `main`, so they
stay here.

```bash
common join --check              # can this machine join? checks only
common join                      # serve a slice of the default model
common join --model bonsai-8b    # a different layer-network model
common join --permanent          # keep serving after reboots
```

`common join` keeps running in its terminal. From a second terminal:

```bash
common ask "What's a good way to learn recursion?"
```

`common ask` talks to this node's local API at `http://127.0.0.1:11440/v1`.
Ctrl+C leaves cleanly.

The first run sets up a private Python environment, picks the layers the
network needs most that fit this machine's memory, and downloads only those
layers. Other peers reach the node through a Cloudflare quick tunnel
(`cloudflared`) unless a direct path is available.

Details, flags, and backends: the [top-level README](../README.md) and
[docs/layer-network.md](../docs/layer-network.md).

### Python setup

The installer accepts Python 3.10 or newer and automatically downloads a private Python 3.12 if none is available. On first `common join`, if the installed Python cannot run the model dependencies and no compatible interpreter is found, Common downloads Python 3.12 using uv and creates its own environment. No manual Python downgrade, pip installation, or administrator access is required. Downloads need internet access to astral.sh, GitHub, and the package indexes.
