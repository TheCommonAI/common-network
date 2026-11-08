# `common` — the CLI

The Common Network CLI (v0.4): join the layer network as a peer, ask it
questions, and inspect it.

```bash
common join --check
common join
common ask "What's a good way to learn recursion?"
common                         # then type /benchmark for ten sequential sample requests
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

Installed copies check the complete launcher on startup. `common update` checks the same files
explicitly, even when the version number has not changed, and reports a build ID. Downloads are
validated before any files are replaced; failed installs restore the previous files. Restart
`common join` after updating a running node. Its original shaded Common ASCII banner appears above
the overall progress percentage, with setup details behind **d**. Model and network code are fetched
when joining; `COMMONNET_SRC` and `COMMONNET_REF` still select a custom checkout or pinned network.

In the interactive session, `/benchmark` sends ten varied-length prompts one at a time through this
machine's joined layer network. It uses greedy decoding so speculative decoding can run, caps each
answer at 96 tokens, and reports per-request TTFT and local draft acceptance when available. Each
completed request is recorded by the normal chat metrics reporter; refresh the public stats API after
the next heartbeat (usually within 15 seconds) to see the rolling aggregate. The first request may
include cold-start delay.

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


On a terminal, CLI failures offer `Would you like to report this bug? [y/N]`.
Choose `y` to open a GitHub issue draft with the error and Common, OS and Python
versions. Review and submit it in your browser. Empty input or `n` skips it;
scripts, invalid command syntax and Ctrl+C do not prompt. Common credential
patterns are redacted, but review the draft for private information.

