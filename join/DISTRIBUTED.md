# Common Network v0.2 development backend

The v0.2 path serves one target model across an explicitly configured group of
llama.cpp RPC devices. A small model on the coordinator drafts tokens; the
distributed target verifies them using llama.cpp's native speculative decoder.
This is an operator-managed prototype, not a stable release or an autonomous
Petals swarm. Drafting currently runs on the coordinator, not each desktop client.

## Initial network model

**Qwen3-32B is the first network target model**, with public alias `Qwen3-32B`.
The example configuration reflects that choice. Select the actual GGUF artifact
and quantization before provisioning; the example filename is a placeholder,
not an automatic download. The draft model remains a separate choice whose
compatibility, acceptance rate, and output correctness must be validated against
Qwen3-32B.

The existing Bonsai-based configurations under `experiments/` and saved smoke
results describe earlier test fixtures. They have not been converted into
Qwen3-32B deployments, and their performance/correctness results do not qualify
the new target. In particular, the Bonsai-trained draft has not been qualified
against Qwen3-32B. Run fresh plain and speculative tests with the selected pair.

## Desktop compatibility

The coordinator registers through the existing join flow, presents the configured
model alias at `/v1/models`, and serves the existing `/v1/chat/completions` endpoint
through the authenticated Common worker. Gateway registration, node identity,
contribution tokens, streaming, and the CLI appearance retain their existing
contracts. Desktop clients need no new inference protocol. Existing single-model
`common join` and `common serve` remain available.

## Prepare hosts

Use the same llama.cpp revision and compatible model files across the group.
Build llama.cpp with `-DGGML_RPC=ON` plus the accelerator backend for each host.
The binaries must support `--rpc`, `--spec-type`, and the `--spec-draft-*` options
used in `runtime.py`. Common deliberately does not download the ordinary runtime
for this mode: its default CPU release may not support this RPC configuration.
The current installed build is checked for required flags and available devices.
Choose a compatible target/draft tokenizer and architecture; successful startup
is not a substitute for validating output correctness.

On each contributing host, launch the RPC worker in the foreground:

```bash
python3 common/common.py --no-update rpc --rpc-worker /path/to/ggml-rpc-server
```

RPC has no Common authentication and is not safe to expose publicly. This command
binds exclusively to loopback. Reach each worker through an authenticated private
transport, for example two SSH forwards from the coordinator:

```bash
ssh -N -L 127.0.0.1:50051:127.0.0.1:50052 host-a
ssh -N -L 127.0.0.1:50052:127.0.0.1:50052 host-b
```

These are two separate foreground sessions. Do not expose raw RPC through the
public Common worker or a Cloudflare HTTP tunnel. Remote workers expose compute
devices, not independent HTTP model endpoints, and are not registered as complete
models in the legacy gateway roster.

## Enrol members without SSH (automatic setup)

The RPC build, worker launch and SSH forwards above are the manual route. For
members running their own machines, a group can enrol them instead. A member
needs Python and nothing else: its machine listens on nothing, its worker dials
out to the coordinator over a WebSocket (wss://), and its `ggml-rpc-server`
stays on 127.0.0.1. The coordinator's relay is reached through a Cloudflare quick
tunnel, the same one the worker uses, so there is no port to open and no address
to know. The network is GPU-only: a member serves one GPU device, and a machine
with no GPU refuses to join before it spends its token. The RPC build is the one
the runtime already installs. Each archive is checked against the sha256 GitHub
records for it before it is run.

On the coordinator, once per group and then while the group is up:

```bash
python3 join/group.py init --group lab                  # once per group
python3 join/group.py relay --group lab                 # keep running; prints the tunnel URL
python3 join/group.py issue --group lab --member node-a # prints a one-time token, valid 24 hours
python3 join/group.py members --group lab               # prints the rpc_servers list
```

On each member:

```bash
python3 join/join.py --group-token '<token>'     # or: common join --group-token '<token>'
```

The member enrols, starts its RPC worker and stays connected in the foreground.
Ctrl+C or a service stop also stops its RPC server. The member's key is kept in
`~/.common-network/groups/<group>/member.json`, readable only by its owner.

The member's RPC server runs with its tensor cache on. The weights it is sent are
kept under `~/.common-network/tensor-cache/rpc`, so a later load of the same model
does not upload them over the link again. The cache is not bounded and is never
pruned, and the server does not verify it, so a cache file cut short by a kill
during its first write is served later as if it were complete.

Still manual: copy the `rpc_servers` list from `members` into the coordinator's
distributed config and restart the coordinator, because the runtime's split is
fixed at start. A token works once, and it names the tunnel URL the relay had
when it was issued.

Rejoining after the relay's tunnel URL changes (a relay restart, or cloudflared
restarting). An enrolled member needs no new token. The relay prints its
current tunnel URL; the member runs the same command with that address:

```bash
python3 join/join.py --group-token '<the token it enrolled with>' --relay 'https://<new-tunnel>.trycloudflare.com'
```

The member proves its key from `member.json` to the relay, which checks it
against the stored SHA-256 verifier, then keeps the new address in `member.json`
and runs its worker. Its name and rpc port do not change, so the `rpc_servers`
list stays the same. A one-time token is never a rejoin credential: the coordinator
refuses one, spent, expired or fresh. The `--group-token` value is read only for
its group and member name, and it must match `member.json`. A wrong key is
refused and leaves `member.json` as it was. `--relay` is only for a machine that is
already enrolled; a new member still enrols with a token from `issue`.

Limit: the new address still has to reach the member by hand, because a quick
tunnel's URL is not known until the relay runs. A member whose process was
already running keeps retrying its old address until it is restarted with
`--relay`. Revoking a member, and adding members to a running coordinator without
a restart, are not built yet. RPC has no authentication between peers, so admit
only members you trust with the group's weights and compute.

## Configure and join

Copy `join/distributed.example.json` to your own configuration file. Set absolute
paths to the RPC-enabled `llama-server`, target GGUF, and draft GGUF. Relative paths
resolve against the JSON file's directory. Target weights are loaded by the
coordinator and transferred to the remote devices; remote hosts do not need their
own target GGUF download. This may make initial startup expensive.

Discover exact device names with your installed build:

```bash
/path/to/llama-server --rpc 127.0.0.1:50051,127.0.0.1:50052 --list-devices
```

Use those names in `devices`. Every configured endpoint must have a selected
device. Layer splitting and positive tensor proportions are explicit; local
target acceleration is excluded. `draft_device` is `none` for local CPU drafting,
or a local accelerator such as `MTL0` or `CUDA0`. Start with `draft_tokens: 4`.
`context` is the total context allocation across `parallel` slots.

Validate configuration and peer discovery, then run an isolated test:

```bash
python3 common/common.py --no-update rpc --distributed-config /path/to/network.json
python3 common/common.py --no-update join --distributed-config /path/to/network.json --local-only
```

`--local-only` serves the model API at `http://127.0.0.1:11436/v1` without
gateway registration, a Common worker, automatic warm-up requests, a tunnel,
self-updates, or identity-file changes. Only explicit client requests generate
tokens. Ctrl+C stops the coordinator's model process and leaves existing RPC
workers/tunnels alone. This endpoint is for direct local API tests; the desktop's
normal network path still requires a gateway and has not been connected to this
standalone endpoint. The benchmark below also never registers with a gateway.

**`--lan` is not an isolated-test mode.** It still registers a live node with
`--gateway`, which defaults to the production gateway, and advertises its LAN
worker address. For an intentionally registered development node, select an
explicit development gateway:

```bash
python3 common/common.py --no-update join --distributed-config /path/to/network.json --gateway http://development-gateway:8000 --lan
```

Omit `--lan` when a registered coordinator should use the existing Common HTTP tunnel.
That exposes only the authenticated worker, never raw RPC. Use the existing
gateway selection flags for a development gateway; registration otherwise uses
the normal gateway. `--model` cannot be combined with `--distributed-config`.
Permanent services preserve the absolute config path. Keep `--no-update` while
testing this unreleased checkout. Runtime diagnostics go to
`~/.common-network/runtime.log`; they can contain model/request diagnostics.

Registered joins also run an automatic one-token warm-up using the user message
`hi`. A short prompt appearing in runtime logs therefore does not by itself
establish that a gateway user submitted it. Use request-level gateway/worker logs
to establish provenance rather than inferring it from the prompt-token count.

## Validate performance and correctness

First resolve the existing chat-04 greedy divergence. Compare token IDs and
target logits at the first mismatch with the identical preceding token sequence;
do not assume numerical error without checking verification and cache rollback.

Run the benchmark against the same real RPC group:

```bash
python3 bench/distributed.py /path/to/network.json /path/to/txt-prompts --out results.jsonl --repeats 3
```

The benchmark starts independent plain/k=4/k=8/k=12 runtimes, warms each one, and
records end-to-end tokens/sec (including prefill), first-content latency, maximum
SSE content gap, usage, and generated text. Chunk gaps are not token gaps. Runtime
logs are retained per arm for native draft/acceptance counters. Output files are
created exclusively; use a new output path for each run. Generated text is stored
locally. Compare output text across arms, and inspect token-level divergence
separately. Repeated arms are sequential, so repeat with different arm ordering
before drawing close performance comparisons. No latency is simulated by this
tool. Run on two physical hosts before making WAN claims.

The preliminary 30-prompt simulation measured 2.60x/2.83x/2.94x mean speedups at
k=4/8/12 and 87/90 exact greedy matches. Hindsight best-k (3.05x) is not an
implemented adaptive policy. Those figures do not establish distributed speed.

## Remaining release gates

- Validate startup, streaming, cancellation, and request isolation on physical hosts.
- Resolve greedy divergence and validate non-greedy sampling with the chosen build.
- Test concurrent requests, long contexts, disconnects, and actual WAN conditions.
- Add automatic discovery, placement, block replication, and state recovery.
- Integrate individual block contributions into gateway/desktop contribution UI.

A failed RPC connection may terminate the underlying runtime; join detects a dead
runtime, deregisters the coordinator, and stops its children. Active streams may
end with transport errors. This version does not recover a session across block
loss, promise uninterrupted streams, or intentionally serve unverified drafts.
