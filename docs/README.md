# Documentation

Start with **layer-network.md**. It describes how the network works today and how to use it.

The product version is **v0.5**. User-facing scripts (`common`, `join`, `chat`) and the
installers all report that same number.

## Using the network

- [layer-network.md](layer-network.md): the main guide. Joining, the commands, switching models, the gateway directory.
- [bonsai-networks.md](bonsai-networks.md): the Bonsai models, and how a network is defined by a manifest file.
- [automatic-connectivity.md](automatic-connectivity.md): how peers connect when they sit behind NAT.
- [fixed-route.md](fixed-route.md): the older way to run one model across two machines you list yourself (`members.json`).

## How it works

- [token-id-chain.md](token-id-chain.md): a request sends token IDs to the first peer; peers pass hidden states along.
- [flash-drafter.md](flash-drafter.md): speculative decoding, where a small local model drafts tokens for the network to check.
- [network-contract.md](network-contract.md): the design contract for the layer network (manifest, records, gossip, placement).
- [direct-mode.md](direct-mode.md): how `--direct` publishes a TLS address and finds peers through a DHT.

## Testing and measuring

- [network-bench.md](network-bench.md): the latency benchmark that runs without a real model.
- [direct-test.md](direct-test.md): a manual test plan for direct mode across three real machines.

## History

- [retiring-the-old-network.md](retiring-the-old-network.md): what the v0.1 whole-model network was, and what replaced it.
- [archive/](archive/): older documents kept as history. [SECURITY-REVIEW-2026-10-18.md](archive/SECURITY-REVIEW-2026-10-18.md) reviews the retired v0.1 gateway.

## Where the code is

Folders the installer fetches by path from `main` (`join/`, `common/`, `chat/`,
`install.sh`, `install.ps1`, `commonnet/networks/`) stay where they are on purpose.

| Folder | What it is |
|---|---|
| `join/` | `common join`: the node that serves layers on a machine you own. |
| `common/` | The `common` command: `ask`, `join`, `status`, and the rest. |
| `chat/` | The terminal chat client, `common-chat`. |
| `commonnet/` | The layer-network library. Its [README](../commonnet/README.md) is a module map. |
| `gateway/` | The small signpost service: `/manifest`, `/seeds` and the rendezvous directory. |
| `e2e/` | End-to-end checks against real models, run by hand. |
| `bench/` | Leftover v0.1 quality scoring (GSM8K, HumanEval, MMLU), not the layer-network latency harness. |
| `scripts/` | Small developer tools, such as a fake status file for building a UI. |
| `docs/` | Everything a person reads: this index, the guides, and the design contract. |

Two files outside this folder matter for contributors: the top-level [README](../README.md), and [SECURITY.md](../SECURITY.md) for what the network does and does not protect.
