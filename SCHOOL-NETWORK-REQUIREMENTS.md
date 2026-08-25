# What the school network needs to unblock

Everything below is derived from what the code actually calls — not a generic
list. Each entry says what breaks if it stays blocked, so IT can approve the
minimum rather than the maximum.

**Read this first:** there are two ways to run this on a school network, and
the easy one needs almost nothing unblocked. Take Option A to IT first.

---

## Option A — LAN mode (recommended, minimal asks)

Every machine in the lab is already on the same network. Nodes can talk to the
gateway directly over the LAN, so **no tunnels, no inbound firewall rules, no
public hostnames**. Run the gateway on one machine in the room.

| Need | Detail | Why |
|---|---|---|
| **Outbound HTTPS (443) to `registry.ollama.ai` and `*.ollama.com`** | One-time per model | Pulling model weights. Without this nothing can be installed at all. |
| **Outbound HTTPS (443) to `github.com` + `raw.githubusercontent.com`** | One-time | The installer, the CLI, and the CGLA-Legal graph repo. |
| **Outbound HTTPS (443) to `pypi.org` + `files.pythonhosted.org`** | One-time | Python dependencies for the gateway. |
| **Intra-LAN TCP on ports `11434` and `8000`** | Continuous | `11434` is Ollama on each node; `8000` is the gateway. Many school networks have **client isolation** (AP isolation) on, which blocks machine-to-machine traffic on the same subnet — this is the single most likely thing to break the build. |
| **Ollama binding beyond localhost** | Per node | Set `OLLAMA_HOST=0.0.0.0:11434` so other machines can reach it. Default is localhost-only. |

That's it. No external service needs to accept connections, and no student
traffic leaves the building at query time.

### The one thing to check before the lab day

Client isolation. From one machine, with another machine's IP:

```bash
curl http://<other-machine-ip>:11434/api/tags
```

If that returns JSON, you're fine. If it hangs or is refused while
`curl http://localhost:11434/api/tags` works on that machine, client isolation
is on and IT needs to disable it **for that subnet or VLAN only** — that's a
much smaller ask than disabling it network-wide.

---

## Option B — Public gateway via Cloudflare tunnels

This is what `common join` does by default: it opens an outbound tunnel so a
node behind NAT is reachable from a gateway hosted on Railway. It needs more
unblocked, and school filtering commonly blocks exactly these.

| Need | Detail | Likely blocked? |
|---|---|---|
| **Outbound HTTPS to `*.trycloudflare.com`** | The tunnel hostname each node gets | **Very likely** — free tunnel subdomains are a standard filtering category (they're used to bypass filters). |
| **Outbound to `*.argotunnel.com` / Cloudflare edge on 443 and 7844** | `cloudflared`'s control channel | **Likely** — port 7844 is non-standard and usually closed. |
| **`cloudflared` binary allowed to run** | Installed via brew/winget | Application allowlisting may block an unsigned tunnelling binary, and IT may object on principle. |
| **Outbound HTTPS to `gateway-production-b820.up.railway.app`** | The hosted gateway | Possible — `*.up.railway.app` is a generic app-hosting domain. |
| Everything in Option A as well | | |

**Expect pushback on this option, and it is reasonable pushback.** A tunnel
that makes a school machine reachable from the public internet is precisely
what a network administrator is employed to prevent. Don't lead with it.

---

## What to actually ask for

A short version to forward:

> We're running a school project that installs open-source AI models locally on
> lab machines and has them answer questions collaboratively. Nothing is
> published to the internet and no student data leaves the building.
>
> We need:
> 1. Outbound HTTPS (443) to `registry.ollama.ai`, `ollama.com`, `github.com`,
>    `raw.githubusercontent.com`, `pypi.org`, `files.pythonhosted.org` — for
>    downloading the software and the models, one time.
> 2. Machine-to-machine TCP on ports `8000` and `11434` **within the lab subnet
>    only** — the machines need to talk to each other. If client/AP isolation is
>    enabled on that subnet, we need it off for that subnet.
>
> We do not need any inbound access from outside the school, any public
> hostname, or any firewall changes at the perimeter.

---

## Non-network things that will bite

Worth settling before the day, because each one stops a machine dead:

- **Disk space.** Each specialist is 2–9 GB. A 20-machine lab pulling different
  models is fine, but a machine with 5 GB free cannot hold a 7B model. Check
  free space first; `common join` reports it.
- **RAM.** The catalogue's smallest entry (Llama 3.2 3B) needs ~4 GB available.
  8 GB machines can run a 7B specialist; 4 GB machines can only run the 3B
  fallback and should be aggregators or left out.
- **Admin rights** to install Ollama and Python packages. Usually the real
  blocker in a managed SOE, and it is a different conversation from firewalls.
- **Sleep/power settings.** A node that sleeps drops out of the network mid-
  request. The gateway health-checks every 30s and marks it unhealthy, so
  nothing breaks — but the lab loses that lane. Set the machines to stay awake.
- **Roaming profiles / disk wipe on logout.** If the SOE resets machines
  between sessions, the model downloads vanish and every session re-pulls
  several GB. Ask whether `~/.ollama` can be excluded from the reset, or point
  `OLLAMA_MODELS` at a persistent local path.

---

## Bandwidth, so nobody is surprised

First run, per machine, is a one-time model download:

| Model | Approx download |
|---|---|
| Llama 3.2 3B | ~2 GB |
| Mathstral 7B / Qwen2.5 Coder 7B / SQLCoder 7B | ~4 GB each |
| Llama 3.1 8B (CGLA base) | ~4.7 GB |
| Qwen2.5 Coder 14B / DeepSeek-R1 14B | ~9 GB each |

Twenty machines pulling ~4 GB each is **~80 GB**, and doing that simultaneously
during class will saturate a school link and make you unpopular. Two mitigations:

1. Stage it — pull models the day before, or in batches of four or five.
2. Better: pull once on one machine and copy `~/.ollama/models` to the others
   over the LAN. Same bytes, one internet download.

After that first pull, running the network uses almost no internet at all —
inference is local, and only the gateway's routing traffic crosses the LAN.
That is the whole point of the project, and it's a fair thing to say to IT.
