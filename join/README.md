# Join the Common Network

Contribute your computer's spare capacity as a node. Setup is meant to be
as easy as installing a chat app.

## Quick install

**Mac / Linux:**

```bash
curl -fsSL https://commonai.com.au/install.sh | sh
```

**Windows (PowerShell):**

```powershell
irm https://commonai.com.au/install.ps1 | iex
```

This installs `cloudflared` if you don't have it, downloads the join
script, and puts a `common-join` command on your PATH. That is everything
you have to install — the model runtime and the weights are fetched by
`common-join` itself, on first run.

## Run it

```bash
common-join
```

No password needed — joining is permissionless, anyone can contribute a
node. It will:

1. Check GitHub for a newer version of itself and update in place if found
   (pass `--no-update` to skip).
2. Download the model runtime if this machine doesn't have it yet: a small
   `llama-server` archive, 12–20 MB depending on the platform, from
   llama.cpp's GitHub releases. This is what runs the model, and it is
   fetched once per machine, not once per run.
3. Download the model's weights if they aren't already here, then start the
   runtime on them. This is the big one: 1.3 GB for the smallest model in the
   catalogue, around 4 GB for a 7B, and it resumes if the download is
   interrupted.
4. Open a free Cloudflare tunnel to your local worker.
5. Register your node with the shared gateway.
6. Keep running so your node stays online.

Press `Ctrl+C` to leave the network — it deregisters your node cleanly.
Every future run auto-updates itself, so friends never need to reinstall to
get fixes or improvements.

There is no default model. `common-join` probes what this machine can run
and asks the gateway which of those models the network needs most, then
confirms with you. Pass `--auto` to accept that choice without the prompt,
or `--model` to pick one yourself — a catalogue id, or any model tag the
network knows about:

```bash
common-join --model qwen2.5-coder-7b
common-join --list-catalogue      # what this machine can run
```

Keeping it to 7B or smaller is a good idea unless your machine can
comfortably handle more, and the same free-tunnel time limit described at
the bottom of this file applies.

Run `common-join --help` for all options (custom gateway, node name,
region, etc).

## Running it permanently (servers, always-on machines)

Don't want to keep a terminal window open? Install it as a real background
service — starts at login/boot and restarts automatically if it crashes:

```bash
common-join --permanent
```

This installs a LaunchAgent (macOS), a `systemd --user` service (Linux), or
a Scheduled Task (Windows) that runs `common-join` for you. It still
checks for updates and auto-updates itself every 30 minutes, same as
running it in a terminal. Logs go to `~/.common-network/join.log` (macOS)
or `journalctl --user -u common-join.service` (Linux).

On a headless Linux server, also run `loginctl enable-linger $USER` so it
keeps running after you log out of your SSH session.

To stop and remove it:

```bash
common-join --remove-permanent
```

## Manual install (advanced / no installer)

If you'd rather not run the installer, you only need Python 3.8+ and
[cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
on your PATH, plus the three scripts side by side in one directory:

```
join.py  worker.py  runtime.py
```

All three are in this directory — copy them out of a clone, or download
them from
`https://raw.githubusercontent.com/TheCommonAI/common-network/main/join/`.
`join.py` imports the other two as siblings, so they have to sit next to
it. Then:

```bash
python3 join.py
```

`runtime.py` is stdlib-only like the rest, and needs nothing preinstalled.
Its downloads live under `~/.common-network/` — `models/` for weights,
`runtime/` for the model runtime, `cache/` for the resolved build number —
so removing that directory is a complete uninstall.

## Notes

- Your node only serves the model you specify. It's stateless — the gateway
  never sees your files, just chat requests.
- Ollama is not involved and does not need to be installed. If it happens
  to be, and it has already pulled the model your node is serving, the
  weights are hardlinked out of its store rather than downloaded again — so
  nothing is fetched twice, and removing Ollama afterwards doesn't disturb
  them.
- The tunnel URL changes every time you run the script, which is fine — the
  gateway is told your endpoint at registration time.
- If your laptop goes to sleep or loses internet, the gateway's health check
  will just mark your node unhealthy and stop routing to it until you're
  back.
- **The free Cloudflare quick tunnel has a hard ~100 second response
  ceiling** — if your model hasn't replied by then, the request fails with
  a Cloudflare 524 and the gateway falls back to another node. Stick to
  small, fast, non-reasoning models (the default `llama3.2:3b` is fine).
  "Thinking"/extended-reasoning models (e.g. some Qwen3 variants) can burn
  well past 100 seconds on modest hardware before producing any visible
  output — avoid those unless you've confirmed they respond quickly on
  your machine.
