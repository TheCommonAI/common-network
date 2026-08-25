# Chat with the Common Network

A terminal client for asking the network questions — no Ollama, no
cloudflared, just Python. If you already ran the [installer](../install.sh),
you have this as `common-chat`.

## Quick install (chat only, no node contribution)

**Mac / Linux:**
```bash
curl -fsSL https://raw.githubusercontent.com/TheCommonAI/common-network/main/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/TheCommonAI/common-network/main/install.ps1 | iex
```

The installer always sets up `common-chat` — Ollama and cloudflared are
only needed if you also want to contribute a node with `common-join`.

## Use it

One-shot question:

```bash
common-chat "What's a good way to learn recursion?"
```

Interactive chat (keeps conversation context across turns):

```bash
common-chat
```

Every reply is followed by a dim line showing which node answered and its
routing score, e.g. `via ollama-qwen-coder-local (score 0.681)` — the
network tells you exactly where your request went.

## Manual install (advanced / no installer)

Just needs Python 3.8+:

```bash
python3 chat.py "your question"
```

## Options

```
common-chat --gateway https://your-gateway.example   # talk to a different network
common-chat --region au-adelaide                      # hint your region for routing
common-chat --no-update                                # skip the self-update check
```
