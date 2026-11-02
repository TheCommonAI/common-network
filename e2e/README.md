# End-to-end checks (run by hand)

These scripts run the real model path: real checkpoints, real HTTP between block servers, and real peers that
find each other over gossip. They need a model checkpoint on disk and, for most of them, Apple silicon with MLX.
The unit tests in `commonnet/tests/` do not run them, and nothing runs them automatically.

Run them from this folder, because they read and write `reference.json` in the current directory:

```bash
cd e2e
python3 reference.py CHECKPOINT   # 1. write reference.json: single-machine greedy output (mlx_lm)
python3 split.py CHECKPOINT CUT   # 2. two local block servers, compared with reference.json
```

| Script | What it checks |
|---|---|
| `reference.py` | Writes `reference.json`, the single-machine greedy output for the prompts in `prompts.py`. Run this first. |
| `split.py` | Splits the model at one layer across two local block servers and compares the output with the reference. |
| `bonsai_split.py` | The same kind of split on a Bonsai pack (Apple silicon). Run from anywhere: `python3 e2e/bonsai_split.py CHECKPOINT_DIR [CUT] [NEW_TOKENS]`. |
| `network_e2e.py` | Real `commonnet.node` peers find each other by gossip. (1) One peer of a replicated stage is killed mid-generation; the output must match the unbroken run. (2) The only live holder of a stage is killed; a spare must take the stage over, and generation must still work. |
| `failover_e2e.py` | Kills the second-half member mid-generation (`CHECKPOINT KILL_AT`); a replica takes over and the output must match. |
| `ctx.py` | Sweeps context length over two local block servers. |

The `gens_cut*_bf16.json` files are saved outputs from earlier `split.py` runs, kept for comparison.
