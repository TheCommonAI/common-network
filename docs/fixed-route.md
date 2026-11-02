# Running a fixed two-machine route

This is the older, hand-wired way to run one model across two machines. You list the machines yourself, in a
`members.json` file, and a coordinator chains them. It predates the layer network. For normal use, join the
layer network with `common join` instead; see [layer-network.md](layer-network.md). Use this when you want to
run a specific split between machines you control, or to reproduce a result.

The example below is Qwen3-8B split in two. Machine A (the coordinator, and the first half of the layers) is
an Apple silicon Mac. Machine B (the second half) has an NVIDIA GPU. Any two machines work with the same
commands, using the backend that matches each one.

## 1. Make a token

Each member needs a shared bearer token. Every request has to present it. Make one, and send it to the other
person over a private channel, never in a chat or on a command line:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))" > member.token
chmod 600 member.token
```

The member reads it with `--token-file`. The coordinator reads it from each member's `token_file` in
`members.json`. To rotate it, restart the member with a new file.

## 2. Start the member on machine B

```bash
python3 -m commonnet.member --repo unsloth/Qwen3-8B-bnb-4bit --layers 20 36 --backend nf4 \
    --token-file member.token --out member.json
```

The member downloads only the layers it serves, about 1.6 GB of the 6.1 GB file, from Hugging Face. The files
go to `~/.common-network/commonnet/models/`. Set `COMMONNET_CACHE` to change that, and `HF_TOKEN` for gated repos.
If you already have a full checkpoint, pass `--checkpoint DIR` instead of `--repo`.

When the tunnel answers, `member.json` is written with `tunnel_url`, `layers` and `device`. Send machine B's
`tunnel_url` to the coordinator, and never the token. Ctrl+C stops the server and the tunnel. A quick-tunnel
hostname changes every time the member restarts, and a new one can take a few minutes to resolve.

## 3. Start the first half on machine A

Machine A serves layers 0 to 20 locally, with no tunnel:

```bash
COMMONNET_CHECKPOINT=$HOME/models/hf/Qwen3-8B-4bit COMMONNET_BACKEND=mlx \
  python3 -m commonnet.net.server --port 18701 --layers 0 20 \
  --block commonnet.net.factory:FACTORY --token-file member.token &
```

Before you pick the split, `commonnet.plan` works out how many layers each machine can hold:

```bash
python3 -m commonnet.plan --model 8b --mac-gib <GiB free on A> --pc-gib <GiB free on B> --ctx 2048
```

## 4. Write `members.json`

```json
[{"url": "http://127.0.0.1:18701", "layers": [0, 20],  "token_file": "member.token"},
 {"url": "https://B-TUNNEL-URL",   "layers": [20, 36], "token_file": "member.token"}]
```

Replace `B-TUNNEL-URL` with the `tunnel_url` from step 2. If machine B restarts, its URL changes, so update the file.

## 5. Generate

```bash
python3 -m commonnet.coordinator --endpoints members.json --checkpoint $HOME/models/hf/Qwen3-8B-4bit \
  --prompt "Hello" --max-new-tokens 64 --client-format mlx4
```

The prompt is raw text, with no chat template. `--client-format mlx4` reads the pre-quantized embedding and
output head from the MLX checkpoint. To fetch only those tensors, pass `--repo mlx-community/Qwen3-8B-4bit`
instead of `--checkpoint`.

## 6. Stop

Ctrl+C the coordinator. Stop machine A's server with `kill`, using the PID that `&` printed. Ctrl+C machine B's
member. Then check that nothing is left running: `pgrep -fl "commonnet|cloudflared"`.

## Limits

- Generation is greedy. The coordinator has no sampling option.
- The route is fixed. If a member drops out mid-generation, that generation fails and has to be restarted. Replicas
  and failover only apply to the live layer network (`--network`), not to `--endpoints`.
- Each request is limited to 8 MiB. Long prompts are sent in chunks to stay under that.
- Hidden states travel as float16 (float32 for the float32 client).
- The `mlx4` client is approximate, because the embedding and output head are 4-bit, so its tokens can differ
  from the full-precision model.
- Context length is limited by the memory left after the weights. Pass the context you need to `commonnet.plan
  --ctx` to check that the split still fits.
- The NVIDIA (NF4) path has not been verified on real CUDA hardware. Expect to debug it on your first run.

## Experimental: Qwen3.6-27B

The same commands work for the larger 27B model, which is split at layer 39. Its layers are hybrid
(linear and full attention), so the block servers keep a small recurrent state as well as the attention cache.

```bash
# Machine A (Apple silicon), layers 0 to 39
python3 -m commonnet.member --repo mlx-community/Qwen3.6-27B-4bit --layers 0 39 --backend mlx \
    --token-file member.token --out mac.json
# Machine B (NVIDIA), layers 39 to 64
python -m commonnet.member --repo samajlouis/Qwen3.6-27B-bnb-nf4 --layers 39 64 --backend nf4 \
    --token-file member.token --out member.json
# Coordinator
python3 -m commonnet.coordinator --endpoints members.json --repo mlx-community/Qwen3.6-27B-4bit \
    --client-format mlx4 --prompt "Hello"
```

This has been tested only on a tiny random model with the same architecture, and not on the real 27B weights.
If it fails, fall back to the 8B commands above.
