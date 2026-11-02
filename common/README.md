# `common` — the CLI

The Common Network CLI (v0.4): join the layer network as a peer, ask it
questions, and inspect it.

```bash
common join --check
common join
common ask "What's a good way to learn recursion?"
common peers
common status
common help
```

`common join` keeps running in its terminal; `common ask` needs that node's
local API (`http://127.0.0.1:11440/v1`). Greedy asks (`temperature` 0, the
default) use Common Flash speculative decoding.

`--model NAME` on `join` is the same as `--network NAME` (`qwen3-8b`,
`bonsai-8b`, `bonsai-27b`, `bonsai-2-9b`, `bonsai-2-27b`). See
[docs/layer-network.md](../docs/layer-network.md) and the
[top-level README](../README.md).

Short alias: `cmn` does the same thing as `common`.

This folder is fetched by path from `main` by installed clients, so it stays
here.

## Design

Built to the COMMON. CLI design system: a four-colour palette (charcoal /
paper / dim / blue+red for attention), a fixed glyph vocabulary (`·` `→`
`←` `✓` `⚠` `✗` `#`), lowercase direct copy, and a transparency footer on
every `ask` — which peers answered, time to the first token, total time.

Retired verbs from the old whole-model gateway (`demand`, `contrib`,
`recommend`, `serve`, `test`, `--legacy`) still parse so old scripts get a
clear message. They are not how the network works.
