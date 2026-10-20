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

Windows step-by-step is in `commonnet/PATRICK.md`. In short (Qwen3-8B, NF4, layers 20-36):

    python -m commonnet.member --layers 20 36 --backend nf4 --checkpoint C:\\path\\Qwen3-8B-bnb-4bit \
        --token-file member.token --out member.json

`member.json` is written once the tunnel answers: `{"tunnel_url": ..., "layers": ..., "device": ...}`.
Patrick sends Miles the `tunnel_url` only (never the token). Ctrl+C or SIGTERM stops the server and the
tunnel and sets `tunnel_url` to null. A fresh quick-tunnel hostname can take up to 3 minutes to resolve.

## Miles: tonight (Qwen3-8B, planner split)

Split from `python3 -m commonnet.plan --model 8b --mac-gib 11 --pc-gib 6.5 --ctx 2048`:
Mac layers 0-20 (client + 20 layers, 2.83 GiB), Patrick (NF4) layers 20-36 (16 layers, 1.61 GiB).
Mac checkpoint: `mlx-community/Qwen3-8B-4bit` (4.6 GB) in `~/models/hf/Qwen3-8B-4bit`.
Patrick checkpoint: `unsloth/Qwen3-8B-bnb-4bit` (6.1 GB, single file; embed/lm_head are bf16 and unused by his block).

1. One token for both members, sent to Patrick privately as a file:
       python3 -c "import secrets;print(secrets.token_urlsafe(32))" > member.token && chmod 600 member.token
2. Start your member (local only is fine; no tunnel needed for your own half):
       COMMONNET_CHECKPOINT=$HOME/models/hf/Qwen3-8B-4bit COMMONNET_BACKEND=mlx \
         python3 -m commonnet.net.server --port 18701 --layers 0 20 --block commonnet.net.factory:FACTORY --token-file member.token &
3. Patrick starts his member (PATRICK.md) and sends you the `https://....trycloudflare.com` URL.
4. Write `members.json` with his URL:
       [{"url": "http://127.0.0.1:18701", "layers": [0, 20],  "token_file": "member.token"},
        {"url": "https://PATRICKS-URL",   "layers": [20, 36], "token_file": "member.token"}]
5. Generate (mlx4 client reads the pre-quantized embed/head from the MLX checkpoint):
       python3 -m commonnet.coordinator --endpoints members.json --checkpoint $HOME/models/hf/Qwen3-8B-4bit \
         --prompt "Hello" --max-new-tokens 64 --client-format mlx4
   The prompt is raw text (no chat template). If Patrick restarts, his URL changes: edit `members.json`.
6. Stop: Ctrl+C the coordinator; `kill` your server (pid is whatever `&` printed); tell Patrick to Ctrl+C.
   Check nothing is left: `pgrep -fl "commonnet|cloudflared"`.

Measured on this Mac (Qwen3-8B-4bit, mlx4 client, 64 new tokens, includes prefill): both members on localhost
12-15 tok/s; one member (layers 0-20) behind a real cloudflared quick tunnel 4.8-5.9 tok/s, about +105 ms per
token for one tunnel hop. Greedy vs mlx-lm on the same checkpoint: free-running output matches the first 6 of 48
tokens, then diverges at a near-tie (both are fluent); teacher-forced (same context fed to both) top-1 agrees on
48 of 48 positions, max logit difference 0.92 against logit std 4.0.

DNS caveat: on this Mac (Tailscale DNS) a brand-new trycloudflare hostname did not resolve locally for minutes
while 1.1.1.1 resolved it in about 12 s. The tools now fall back to asking 1.1.1.1 (then 8.8.8.8) directly when the system
resolver fails (commonnet/net/dns.py), so this should resolve in about 12 s on its own. Backup if it still says it cannot
reach the tunnel: wait a few minutes and retry, or set the Mac's DNS to 1.1.1.1. The tunnel itself was fine (verified with curl --resolve).

### Qwen3-32B later
64 layers; plan with `python3 -m commonnet.plan --mac-gib 11 --pc-gib 6.5 --ctx 2048` gives
Mac (MLX 4-bit, plus the 4-bit client) layers 0-38, PC (NF4) layers 38-64. Start Mac with
`--backend mlx --layers 0 38`, PC with `--backend nf4 --layers 38 64`, coordinator with
`--client-format mlx4`. The Mac checkpoint can be a full-precision one; weights are quantized at load.

## Known limits

- 32B memory headroom is about 0.2 GiB on each side at context 2048 (0.181 Mac, 0.205 PC). Longer context will not fit.
- NF4 backend is untested on CUDA (only its refusal without CUDA is tested). Expect to debug it on first run. Tonight is that first run.
- No keep-alive: every call opens a new HTTP connection through the tunnel; quick tunnels have no SLA
  and a restarted tunnel changes the URL (member.json is rewritten, the coordinator must be re-pointed).
  A member dropping mid-generation loses its KV cache; generation fails and must be restarted.
- Hidden states travel as fp16 (fp32 for the fp32 client), prefill is chunked to stay under the 8 MiB request limit.
- Greedy decoding only; the mlx4 client is approximate (4-bit embedding/head), so tokens may differ from the full model.
- The coordinator prints decoded text only, never tokens ids or secrets.
