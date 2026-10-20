# Common Network (Petals-style) runbook

Layer-sharded Qwen3 inference. Members each serve a contiguous range of decoder layers over HTTP
behind a tunnel. The coordinator holds the embedding, final norm and lm_head and chains the members.

## Token handling

One shared bearer token per member (every request needs it). Make one and send it to the member
over a private channel, not in chat history you cannot delete:

    python3 -c "import secrets;print(secrets.token_urlsafe(32))" > member.token && chmod 600 member.token

The member reads it from `COMMONNET_BLOCK_TOKEN` or `--token-file`. The coordinator reads it from each
member's `token_file` in the endpoints JSON (or `COMMONNET_BLOCK_TOKEN`). Never pass it on a command line.
Nothing in commonnet prints it. Rotate by restarting the member with a new file.

## Patrick: start a member

Needs Python with torch, transformers, safetensors, the model checkpoint on disk, and `cloudflared`.

    export COMMONNET_BLOCK_TOKEN="$(cat member.token)"      # or use --token-file member.token
    python3 -m commonnet.member --layers START END --checkpoint /path/to/Qwen3-8B --out member.json
    # NVIDIA 4-bit (32B):  add  --backend nf4     (needs CUDA; refuses otherwise)
    # Mac:                 add  --backend mlx
    # default backend is torch on cpu; COMMONNET_DEVICE=cuda COMMONNET_DTYPE=float16 for a GPU

`member.json` gets the public tunnel URL and layer range once the tunnel answers. Send Miles the
`url` and `layers` only (not the token). Ctrl+C or SIGTERM stops the server and tunnel and blanks the URL.

## Miles: start the coordinator

Write `members.json`:

    [{"url": "https://....trycloudflare.com", "layers": [0, 18],  "token_file": "patrick.token"},
     {"url": "http://127.0.0.1:PORT",          "layers": [18, 36], "token_file": "mine.token"}]

Run (own member first, same `commonnet.member` command on this Mac):

    python3 -m commonnet.coordinator --endpoints members.json --checkpoint /path/to/Qwen3-8B \
        --prompt "Hello" --max-new-tokens 64 --client-format fp16

Layers must cover the model contiguously with no gap or overlap or the coordinator refuses to start.

### Qwen3-8B first
36 layers. Split e.g. 0-18 / 18-36 (torch backend, fp16 on GPU, fp32/fp16 on CPU).

### Qwen3-32B later
64 layers; plan with `python3 -m commonnet.plan --mac-gib 11 --pc-gib 6.5 --ctx 2048` gives
Mac (MLX 4-bit, plus the 4-bit client) layers 0-38, PC (NF4) layers 38-64. Start Mac with
`--backend mlx --layers 0 38`, PC with `--backend nf4 --layers 38 64`, coordinator with
`--client-format mlx4`. The Mac checkpoint can be a full-precision one; weights are quantized at load.

## Known limits

- 32B memory headroom is about 0.2 GiB on each side at context 2048 (0.181 Mac, 0.205 PC). Longer context will not fit.
- NF4 backend is untested on CUDA (only its refusal without CUDA is tested). Expect to debug it on first run.
- No keep-alive: every call opens a new HTTP connection through the tunnel; quick tunnels have no SLA
  and a restarted tunnel changes the URL (member.json is rewritten, the coordinator must be re-pointed).
  A member dropping mid-generation loses its KV cache; generation fails and must be restarted.
- Hidden states travel as fp16 (fp32 for the fp32 client), prefill is chunked to stay under the 8 MiB request limit.
- Greedy decoding only; the mlx4 client is approximate (4-bit embedding/head), so tokens may differ from the full model.
- The coordinator prints decoded text only, never tokens ids or secrets.
