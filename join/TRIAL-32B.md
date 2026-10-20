# Qwen3-32B two-machine distributed trial

A runbook for one coordinator and one member. It is an operator-managed trial,
not a release. Read `join/DISTRIBUTED.md` for the release gates and the
guarantees this does not give. RPC has no authentication between peers, so
admit only a member you trust with the weights and the compute.

## The trial

| Device | Machine | Role | Endpoint |
|---|---|---|---|
| RPC0 | Miles's Mac, 16 GB unified memory, Apple Metal | target split 0.65, plus the coordinator and the drafter (MTL0) | `127.0.0.1:50051`, local worker |
| RPC1 | Patrick's PC, 8 GB NVIDIA VRAM | target split 0.35, remote member | `127.0.0.1:<52100-52199>` on the coordinator, through the relay |

The electricity-guzzler machine is CPU only and is not part of this trial.

Files:

| File | Size | sha256 |
|---|---|---|
| `Qwen_Qwen3-32B-Q3_K_M.gguf` (bartowski/Qwen_Qwen3-32B-GGUF), target | 15,971,777,856 bytes (about 16 GB) | `26f662712fa72a0649c5578a7e29afb34e7b73650181338e8c3a507cbd7bec50` |
| `Qwen3-0.6B-Q8_0.gguf` (Qwen/Qwen3-0.6B-GGUF), drafter | 639,446,688 bytes (about 0.64 GB) | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` |

Check each download before use, on macOS:

```bash
shasum -a 256 /path/to/Qwen_Qwen3-32B-Q3_K_M.gguf
shasum -a 256 /path/to/Qwen3-0.6B-Q8_0.gguf
```

The drafter filename is assumed from the repo. Confirm the sha256 matches before
you rely on it.

Memory and disk, with a 0.65 / 0.35 split and `context` 4096. These are estimates
from the file sizes and the Qwen3-32B architecture, not measurements:

- Mac, RPC0 plus drafter: about 10.4 GB of target weights, 0.64 GB of drafter, and
  about 0.7 GB of KV cache, plus compute buffers. That is about 11.7 GB before
  buffers, against a Metal working set of about 12.7 GB. This is tight. If the
  coordinator fails to load, lower `context` or move the split toward 0.6 / 0.4
  with `--tensor-split`.
- Patrick, RPC1: about 5.6 GB of weights and about 0.4 GB of KV cache, so about
  6 GB before compute buffers. That leaves room on his 8 GB card.
- Miles's disk: at least 18 GB free for the 16 GB target, the 0.64 GB drafter, and
  the llama.cpp build.
- Patrick's disk: no target file is needed, because the weights are sent to his
  RPC server over the link. He needs only the llama.cpp build that `join.py` installs.

## Miles (coordinator)

**1. Check out the branch.** Run this from your checkout:

```bash
cd /path/to/common-network
git fetch origin
git checkout feat/trial-32b-config
```

**2. Build or locate an RPC-enabled llama-server.** It must have `--rpc`,
`--spec-type`, and `--spec-draft-*`. Confirm that it sees the local worker and the
Metal device:

```bash
/path/to/llama-server --rpc 127.0.0.1:50051 --list-devices
```

You should see `RPC0` and `MTL0` in the output. Note the binary path. It goes in
`--binary` below.

**3. Start the local RPC worker** in its own foreground terminal. It must be an
RPC-enabled `ggml-rpc-server` for the Metal build. The command binds to loopback:

```bash
python3 join/runtime.py --rpc-worker /path/to/ggml-rpc-server --rpc-port 50051
```

**4. Create the group, once.** `init` generates the group secret and keeps it in
an owner-only file. No address is needed here:

```bash
python3 join/group.py init --group qwen3-32b-trial
```

**5. Start the relay**, in its own foreground terminal, and leave it running:

```bash
python3 join/group.py relay --group qwen3-32b-trial
```

The relay listens on 127.0.0.1 only, and opens a Cloudflare quick tunnel to it, so
`cloudflared` must be on your PATH. No router port forward is needed. Wait for the
line `tunnel URL: https://...`. The tokens you issue name that URL, so issue them
only after it appears. If the relay restarts, the URL changes and earlier tokens
stop working.

To stop the relay, press Ctrl+C. SIGTERM (`kill <pid>`) also stops it cleanly. Either
one stops the tunnel and clears the tunnel URL in `group.json`, so `issue` refuses
until the relay runs again. Enrolled members rejoin the restarted relay with `--relay`
(see Patrick's step 5), not with a new token.

**6. Issue Patrick's one-time token**, then send it to him over a private channel:

```bash
python3 join/group.py issue --group qwen3-32b-trial --member patrick
```

The token works once and expires after 24 hours. Do not paste it into chat, a
ticket, or a shared log.

**7. Wait for Patrick to enrol and connect** (section below). Then check that his
endpoint is listed:

```bash
python3 join/group.py members --group qwen3-32b-trial
```

**8. Generate the network config.** The generator reads the enrolled members, so
you do not copy addresses by hand:

```bash
python3 join/group_config.py --group qwen3-32b-trial \
  --target /path/to/Qwen_Qwen3-32B-Q3_K_M.gguf \
  --drafter /path/to/Qwen3-0.6B-Q8_0.gguf \
  --binary /path/to/llama-server \
  --local-backend MTL0 \
  --member-backend CUDA0
```

Use `--member-backend Vulkan0` if Patrick's build reports a Vulkan device, not
CUDA. Use the name that his build reports. It is not verified for his card.

This writes `~/.common-network/groups/qwen3-32b-trial/network.json` with
`rpc_servers` set to `127.0.0.1:50051` and Patrick's port, `devices` set to
`RPC0` and `RPC1`, `tensor_split` set to `[0.65, 0.35]`, and the drafter on `MTL0`
with `draft_tokens` 4. It refuses to write unless there are exactly two endpoints.
It also refuses any CPU device, any endpoint without a GPU backend (MTL, CUDA or
Vulkan), and a drafter on the CPU. To change the split, pass `--tensor-split 0.6,0.4`.
The generator does not check that Patrick is online. The runtime checks that at start.
At start, the runtime preflight also fails if discovery reports a CPU backend for
either configured device.

**9. Start the coordinator** in its own foreground terminal. `--local-only` keeps it
off the gateway, the worker, the tunnel, and self-updates:

```bash
python3 join/join.py --distributed-config ~/.common-network/groups/qwen3-32b-trial/network.json --local-only --port 11437
```

The first start is slow. The coordinator sends about 10 GB of weights to RPC0 and
about 5.6 GB to Patrick's machine over the internet. If startup reports that an RPC
worker is not accepting connections, Patrick is not connected yet, or the relay or
its port is not reachable. Fix that, then start again.

**10. Check health:**

```bash
curl -s http://127.0.0.1:11437/health
curl -s http://127.0.0.1:11437/v1/models
```

**11. Send a chat request.** The `model` value is the `model_name` in the config,
`Qwen3-32B`:

```bash
curl -s http://127.0.0.1:11437/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-32B","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":64,"temperature":0}'
```

Stop the coordinator with Ctrl+C. The RPC worker and the relay keep running, so
restart the coordinator to load a new config.

## Patrick (member)

**1. Check out the branch.** Run this from your checkout:

```bash
cd /path/to/common-network
git fetch origin
git checkout feat/trial-32b-config
```

**2. Use a GPU RPC build on your PC.** Your machine needs an RPC server that uses
your NVIDIA card, through CUDA or Vulkan. A CPU-only RPC server will not work,
because the network is GPU-only. The runtime download is not verified to include
a GPU build for your card, so check it before you join. The `ggml-rpc-server` that
`join.py` installs is the one that runs. Check the RPC log after the first start
for a CUDA or Vulkan device line, in `~/.common-network/groups/qwen3-32b-trial/rpc-server.log`.
The exact wording depends on the build and is not verified here. If the log shows
only CPU, do not continue. Tell Miles. The coordinator's preflight refuses a CPU
backend too.

**3. Join the group with the token Miles sent you.** Run this in its own foreground
terminal. It enrols once, then starts your RPC worker and keeps the link open. Use
`--no-update`, because this checkout is not released. The token already names the
relay's tunnel URL, so the first join needs only the token:

```bash
python3 join/join.py --no-update --group-token '<token>'
```

Your key is stored in `~/.common-network/groups/qwen3-32b-trial/member.json`, which
only you can read. Ctrl+C stops the worker and its RPC server together. If the link
drops, the worker reconnects on its own. The RPC server stays up across reconnects.

**4. Stay connected** until Miles reports that the coordinator is ready. You need no
model file, and you do not run a model yourself. The weights come to you over the
link from Miles's machine.

If you see "already enrolled", your machine has already enrolled in this group. Run
the same command again, and it reconnects at the relay address in `member.json`. If
Miles has restarted the relay, that address is out of date, so use step 5.

**5. After a relay restart, rejoin at the new URL.** When Miles restarts the relay,
the tunnel URL changes. Stop your worker with Ctrl+C if it is still running. Then
run the same command as step 3, with `--relay` set to the new `tunnel URL` line
that Miles sends you, and the same original token:

```bash
python3 join/join.py --no-update --group-token '<original token>' --relay 'https://<new tunnel URL>'
```

You keep your member key. It stays in `member.json`, and only the tunnel URL in that
file changes. Your name and RPC port are unchanged. The token is only read for your
group and machine name, so an expired or already used token works here. `--relay` is
refused on a machine that has not enrolled yet.
