# Quality benches (legacy)

Scoring helpers and small public subsets (GSM8K, HumanEval, MMLU) used by the
retired v0.1 `common test` command. They are not the layer-network latency
harness.

For measuring the layer network, use `python -m commonnet.bench` — see
[docs/network-bench.md](../docs/network-bench.md). That output goes to
`bench-results/`, which is not committed.
