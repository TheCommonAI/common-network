# Chat with the Common Network

A terminal client for asking the network questions — no Ollama, no
cloudflared, just Python. If you already ran the [installer](../install.sh),
you have this as `common-chat`.

## Quick install (chat only, no node contribution)

**Mac / Linux:**
```bash
curl -fsSL https://commonai.com.au/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://commonai.com.au/install.ps1 | iex
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

Every reply is followed by a dim footer: which node answered, and its three
timings kept apart, e.g.

```
served by   ollama-qwen-coder-local
routed in 14ms   ·   first token 57.1s   ·   total 58.4s
# 57.1s of that was ollama-qwen-coder-local loading its model — the next one is fast.
```

Routing is the gateway choosing a node, and on its own it is tens of
milliseconds. The wait a user actually feels is the node loading its model,
which is why the footer names it rather than folding it into a "routed in"
number — the network tells you exactly where your request went, *and* where
the time went.

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
