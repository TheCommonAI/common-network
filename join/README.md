# Join the Common Network

Contribute your computer's spare capacity as a node. Setup is meant to be
as easy as installing Ollama itself.

## Quick install

**Mac / Linux:**

```bash
curl -fsSL https://raw.githubusercontent.com/TheCommonAI/common-network/main/install.sh | sh
```

**Windows (PowerShell):**

```powershell
irm https://raw.githubusercontent.com/TheCommonAI/common-network/main/install.ps1 | iex
```

This installs `cloudflared` if you don't have it, downloads the join
script, and puts a `common-join` command on your PATH. You'll still need
[Ollama](https://ollama.com/download) installed and open — the installer
checks for it and tells you if it's missing.

## Run it

```bash
common-join
```

No password needed — joining is permissionless, anyone can contribute a
node. It will:

1. Check GitHub for a newer version of itself and update in place if found
   (pass `--no-update` to skip).
2. Make sure Ollama is running and pull a default small model if needed.
3. Open a free Cloudflare tunnel to your local Ollama.
4. Register your node with the shared gateway.
5. Keep running so your node stays online.

Press `Ctrl+C` to leave the network — it deregisters your node cleanly.
Every future run auto-updates itself, so friends never need to reinstall to
get fixes or improvements.

By default it serves `llama3.2:3b` (small, fast, works on most laptops).
Pass `--model` to use a different one you have pulled in Ollama — keep it
to 7B or smaller unless your machine can comfortably handle more:

```bash
common-join --model qwen2.5:7b
```

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

If you'd rather not run the installer, you only need Python 3.8+,
[Ollama](https://ollama.com/download), and
[cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
on your PATH, then run the script directly:

```bash
python3 join.py
```

## Notes

- Your node only serves the model you specify. It's stateless — the gateway
  never sees your files, just chat requests.
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
