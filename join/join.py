#!/usr/bin/env python3
"""Join the Common Network as a node operator.

Runs your local Ollama as a contributing node: opens a Cloudflare quick
tunnel to your local Ollama, registers it with the shared gateway, and
keeps running until you press Ctrl+C (at which point it deregisters
cleanly). No account or signup needed for the tunnel.

Requires: Python 3.8+, Ollama (https://ollama.com/download), cloudflared
(https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/).

Usage:
    python3 join.py
    python3 join.py --permanent   # run as a background service
    python3 join.py --remove-permanent                   # undo the above

Joining is permissionless -- no shared password needed, anyone can
contribute a node (see README "Scope"). The gateway hands back a
per-node token on registration, used only to deregister that node later;
it's held in memory for the life of this process, never typed by you.

Joins the default shared Common Network gateway unless --gateway overrides
it. It checks GitHub for a newer version of itself on startup, and again
every 30 minutes while running — if one is found it deregisters, stops
the tunnel, updates, and restarts cleanly (pass --no-update to disable
both checks).

--permanent installs this as a real background service (LaunchAgent on
macOS, systemd --user on Linux, a Scheduled Task on Windows) that starts
at login and restarts automatically if it crashes — for servers or
computers you don't want to babysit with an open terminal. Everything
else has a sensible default — see --help.
"""
import argparse
import json
import os
import platform
import plistlib
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

BANNER = r"""
 ░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓██████████████▓▒░░▒▓██████████████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░         
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░        
░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░        
░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░        
░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░        
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓██▓▒░ 
 ░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓██▓▒░ 
                                                                                                     
                                                                                                     
"""

OLLAMA_URL = "http://localhost:11434"
TUNNEL_URL_PATTERN = re.compile(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com")

# The default, shared Common Network gateway. Override with --gateway to
# join (or run) a different network entirely.
DEFAULT_GATEWAY = "https://gateway-production-b820.up.railway.app"

REPO = "robot-time/common-network"
UPDATE_URL = f"https://raw.githubusercontent.com/{REPO}/main/join/join.py"


def _enable_windows_ansi() -> None:
    if platform.system() != "Windows":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass


# COMMON. brand palette -- exactly four colours (see the CLI design doc).
# No green, no purple, no gradients: restraint is the point. blue is the
# only "active/positive" colour; red carries both emphasis and danger.
PALETTE = {
    "paper": (0xED, 0xE9, 0xE1),
    "dim":   (0x8A, 0x86, 0x81),
    "blue":  (0x92, 0xB4, 0xC8),
    "red":   (0xC8, 0x44, 0x2A),
}


def _color_enabled() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def fg(text: str, color: str, bold: bool = False) -> str:
    if not _color_enabled():
        return text
    r, g, b = PALETTE[color]
    prefix = ("\033[1m" if bold else "") + f"\033[38;2;{r};{g};{b}m"
    return f"{prefix}{text}\033[0m"


def dim(text: str) -> str:
    return fg(text, "dim")


def blue(text: str, bold: bool = False) -> str:
    return fg(text, "blue", bold)


def red(text: str, bold: bool = False) -> str:
    return fg(text, "red", bold)


def paper(text: str, bold: bool = False) -> str:
    return fg(text, "paper", bold)


# Fixed glyph vocabulary -- consistency over cleverness, see design doc 1.5.
GLYPH_WORK = dim("·")       # working / neutral step
GLYPH_ROUTE = blue("→")     # routing to / next / suggested action
GLYPH_DONE = blue("✓")      # done / healthy / served
GLYPH_FORMING = red("⚠")    # degraded / attention
GLYPH_FAILED = red("✗")     # failed / offline / refused


def comment(text: str) -> str:
    return dim(f"# {text}")


def working(text: str) -> None:
    print(f"{GLYPH_WORK} {dim(text)}")


def die(msg: str, code: int = 1) -> None:
    print(f"{GLYPH_FAILED} {red(msg)}", file=sys.stderr)
    sys.exit(code)


IDENTITY_PATH = Path.home() / ".common-network" / "identity.json"


def write_identity(gateway: str, name: str, node_id: str, catalogue_id: str | None, domain_tags: list[str] | None) -> None:
    """Local record of the most recent node this machine registered -- read by
    `common status` / `common whoami` / `common contrib`. Not a cryptographic
    identity (see COMMON. CLI design doc's local-keypair vision) -- just a
    name, deliberately simple until that's a considered decision, not a guess."""
    try:
        IDENTITY_PATH.parent.mkdir(parents=True, exist_ok=True)
        IDENTITY_PATH.write_text(json.dumps({
            "gateway": gateway, "name": name, "node_id": node_id,
            "catalogue_id": catalogue_id, "domain_tags": domain_tags,
            "joined_at": time.time(),
        }))
    except OSError:
        pass  # best-effort -- never block joining the network over this


def clear_identity() -> None:
    try:
        IDENTITY_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def http_json(method: str, url: str, body: dict | None = None, headers: dict | None = None, timeout: float = 10.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


UPDATE_CHECK_INTERVAL_SECONDS = 1800  # re-check every 30 min while a node sits running


def fetch_update() -> bytes | None:
    """Return the latest join.py source from GitHub if it differs from the local copy, else None."""
    try:
        with urllib.request.urlopen(UPDATE_URL, timeout=5) as resp:
            remote = resp.read()
    except (urllib.error.URLError, socket.timeout):
        return None  # offline or GitHub unreachable — carry on with the current version

    if not remote.strip():
        return None

    local_path = os.path.abspath(__file__)
    try:
        with open(local_path, "rb") as f:
            local = f.read()
    except OSError:
        return None

    return remote if remote != local else None


def apply_update_and_restart(remote: bytes) -> None:
    """Overwrite this script with `remote` and re-exec. Never returns on success."""
    local_path = os.path.abspath(__file__)
    try:
        with open(local_path, "wb") as f:
            f.write(remote)
    except OSError as e:
        print(f"{GLYPH_FORMING} {red(f'could not self-update ({e}), continuing with current version')}", file=sys.stderr)
        return
    os.execv(sys.executable, [sys.executable, local_path] + sys.argv[1:])


def die_with_fix(msg: str, fix: str) -> None:
    print(f"{GLYPH_FAILED} {red(msg)}", file=sys.stderr)
    print(comment(fix), file=sys.stderr)
    sys.exit(1)


def check_binaries(need_tunnel: bool = True) -> None:
    if shutil.which("ollama") is None:
        die_with_fix("ollama not found.", "install it from https://ollama.com/download, then run this again.")
    # --lan needs no tunnel, which is most of the point of it: cloudflared is
    # both an extra install and the thing a school network is most likely to
    # block outright.
    if not need_tunnel:
        return
    if shutil.which("cloudflared") is None:
        system = platform.system()
        hint = {
            "Darwin": "brew install cloudflared",
            "Linux": "see https://pkg.cloudflare.com/index.html for your distro",
            "Windows": "winget install --id Cloudflare.cloudflared",
        }.get(system, "see https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/")
        die_with_fix("cloudflared not found.", f"install it with: {hint}")


def ensure_ollama_running() -> None:
    try:
        http_json("GET", f"{OLLAMA_URL}/api/tags", timeout=3)
        return
    except (urllib.error.URLError, socket.timeout):
        pass

    working("ollama doesn't seem to be running — trying `ollama serve`...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(20):
        time.sleep(0.5)
        try:
            http_json("GET", f"{OLLAMA_URL}/api/tags", timeout=3)
            return
        except (urllib.error.URLError, socket.timeout):
            continue
    die_with_fix("can't reach ollama at localhost:11434.", "start it manually and try again.")


def ensure_model(model: str) -> None:
    tags = http_json("GET", f"{OLLAMA_URL}/api/tags")
    names = {m["name"] for m in tags.get("models", [])}
    if model in names:
        return
    working(f"model '{model}' not found locally — pulling it now (this may take a while)...")
    result = subprocess.run(["ollama", "pull", model])
    if result.returncode != 0:
        die(f"failed to pull model '{model}'")


def probe_hardware() -> dict:
    """Best-effort hardware profile: total/available RAM, GPU, disk, CPU, OS.

    Deliberately stdlib + platform-native commands only, no psutil -- this
    script's whole design point is zero pip dependencies (curl one-liner
    install, no venv). A little more platform-branch code here is the right
    trade for keeping that property.
    """
    system = platform.system()
    total_ram_gb = available_ram_gb = 0.0
    gpu_present = False
    vram_gb = 0.0

    if system == "Darwin":
        try:
            total_bytes = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout.strip())
            total_ram_gb = total_bytes / (1024 ** 3)
        except Exception:
            pass
        try:
            vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page_size = 4096
            m = re.search(r"page size of (\d+) bytes", vm)
            if m:
                page_size = int(m.group(1))
            free_pages = 0
            for label in ("Pages free", "Pages inactive"):
                m = re.search(rf"{label}:\s+(\d+)\.", vm)
                if m:
                    free_pages += int(m.group(1))
            available_ram_gb = (free_pages * page_size) / (1024 ** 3)
        except Exception:
            available_ram_gb = total_ram_gb * 0.5  # unknown -- assume half free
        # Apple Silicon has a unified-memory GPU always available for Metal.
        gpu_present = platform.machine() == "arm64"
        vram_gb = 0  # unified memory, no separate VRAM figure

    elif system == "Linux":
        try:
            with open("/proc/meminfo") as f:
                meminfo = f.read()
            total_kb = int(re.search(r"MemTotal:\s+(\d+)", meminfo).group(1))
            avail_kb = int(re.search(r"MemAvailable:\s+(\d+)", meminfo).group(1))
            total_ram_gb = total_kb / (1024 ** 2)
            available_ram_gb = avail_kb / (1024 ** 2)
        except Exception:
            pass
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip().splitlines()
                if out:
                    vram_gb = int(out[0]) / 1024
                    gpu_present = True
            except Exception:
                pass

    elif system == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total_ram_gb = stat.ullTotalPhys / (1024 ** 3)
            available_ram_gb = stat.ullAvailPhys / (1024 ** 3)
        except Exception:
            pass
        # GPU detection on Windows is best-effort/untested, consistent with
        # install.ps1's existing caveat -- default to none rather than guess.

    try:
        free_disk_gb = shutil.disk_usage(str(Path.home())).free / (1024 ** 3)
    except Exception:
        free_disk_gb = None

    return {
        "total_ram_gb": round(total_ram_gb, 1),
        "available_ram_gb": round(available_ram_gb, 1),
        "gpu_present": gpu_present,
        "vram_gb": round(vram_gb, 1),
        "free_disk_gb": round(free_disk_gb, 1) if free_disk_gb is not None else None,
        "cpu_cores": os.cpu_count(),
        "os": system,
    }


def fetch_catalogue(gateway: str) -> list[dict]:
    return http_json("GET", f"{gateway}/catalogue")


def call_assign(gateway: str, hardware: dict) -> dict:
    return http_json("POST", f"{gateway}/assign", body={"hardware": hardware})


def source_to_ollama_tag(source: str) -> str | None:
    return source[len("ollama:"):] if source.startswith("ollama:") else None


def prompt_yes_no(question: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    if not sys.stdin.isatty():
        return default_yes
    answer = input(f"{question} {suffix} ").strip().lower()
    if not answer:
        return default_yes
    return answer.startswith("y")


def print_catalogue(gateway: str, hw: dict) -> None:
    catalogue = fetch_catalogue(gateway)
    headroom_ram = hw["available_ram_gb"] * 0.8
    print(dim(f"your machine: {hw['total_ram_gb']}GB RAM ({hw['available_ram_gb']}GB available), "
              f"{'GPU present' if hw['gpu_present'] else 'no GPU'}, {hw['cpu_cores']} CPU cores, {hw['os']}\n"))
    for m in catalogue:
        runnable = m["min_ram_gb"] <= headroom_ram and (not m["needs_gpu"] or hw["gpu_present"])
        mark = f"{GLYPH_DONE} runnable" if runnable else f"{GLYPH_FAILED} {red('needs more RAM/GPU')}"
        verified = blue("  [verified beats frontier in lane]") if m["verified_in_lane"] else ""
        print(f"  {m['id']:20s} {mark}{verified}")
        print(dim(f"    {m['capability_text'].strip()}"))
        print(dim(f"    tags: {', '.join(m['domain_tags'])}  ·  min RAM: {m['min_ram_gb']}GB  ·  source: {m['source']}\n"))


def resolve_model(gateway: str, args: argparse.Namespace) -> tuple[str, list[str] | None, str | None, str | None]:
    """Returns (ollama_tag, domain_tags, catalogue_id, capability_text_default)."""
    if args.model:
        catalogue = fetch_catalogue(gateway)
        match = next((m for m in catalogue if m["id"] == args.model), None)
        if match:
            ollama_tag = source_to_ollama_tag(match["source"])
            if not ollama_tag:
                die(f"'{args.model}' is an API-based catalogue entry ({match['source']}) — not something common-join can run locally.")
            return ollama_tag, match["domain_tags"], match["id"], match["capability_text"]
        # Not a catalogue id -- treat as a raw Ollama model tag, same as pre-v0.3.
        return args.model, None, None, None

    working("probing your machine's hardware...")
    hw = probe_hardware()
    print(dim(f"  {hw['total_ram_gb']}GB RAM ({hw['available_ram_gb']}GB available), "
              f"{'GPU present' if hw['gpu_present'] else 'no GPU'}, {hw['cpu_cores']} CPU cores, {hw['os']}"))

    assignment = call_assign(gateway, hw)
    print(f"{GLYPH_ROUTE} recommended   {blue(assignment['display_name'], bold=True)}")
    print(comment(assignment["reason"]))
    print()

    ollama_tag = source_to_ollama_tag(assignment["source"])
    if not ollama_tag:
        print(f"{GLYPH_FORMING} {red('the network needs this most, but it can only run as a hosted API:')}")
        print(comment(f"{', '.join(assignment['domain_tags'])} is served by {assignment['source']}, not something common-join can pull and run locally."))
        fallback = _best_local_fallback(gateway, hw, exclude_id=assignment["catalogue_id"])
        if not fallback:
            print(dim("no locally-runnable catalogue model fits this machine either."))
            print(dim("  → see other options:   common-join --list-catalogue"))
            sys.exit(0)
        print(f"{GLYPH_ROUTE} next best you can run locally   {blue(fallback['display_name'], bold=True)}")
        if not args.auto:
            if not prompt_yes_no(f"provision {fallback['display_name']} instead?"):
                print(dim("no changes made."))
                print(dim("  → pick manually:   common-join --model <id>"))
                print(dim("  → see options:     common-join --list-catalogue"))
                sys.exit(0)
        return source_to_ollama_tag(fallback["source"]), fallback["domain_tags"], fallback["id"], fallback["capability_text"]

    if not args.auto:
        if not prompt_yes_no(f"provision {assignment['display_name']} on this machine?"):
            print(dim("no changes made."))
            print(dim("  → pick manually:   common-join --model <id>"))
            print(dim("  → see options:     common-join --list-catalogue"))
            sys.exit(0)

    return ollama_tag, assignment["domain_tags"], assignment["catalogue_id"], assignment["capability_text"]


def _best_local_fallback(gateway: str, hw: dict, exclude_id: str | None) -> dict | None:
    """Best Ollama-runnable catalogue entry this hardware can fit, for when
    /assign's top pick is an API-hosted specialist common-join can't
    provision. Prefers verified-in-lane, then the smallest model (safest
    fit under headroom)."""
    catalogue = fetch_catalogue(gateway)
    headroom_ram = hw["available_ram_gb"] * 0.8
    candidates = [
        m for m in catalogue
        if m["id"] != exclude_id
        and source_to_ollama_tag(m["source"])
        and m["min_ram_gb"] <= headroom_ram
        and (not m["needs_gpu"] or hw["gpu_present"])
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda m: (not m["verified_in_lane"], m["params_b"] or 0))
    return candidates[0]


TUNNEL_HEALTH_CHECK_INTERVAL_SECONDS = 120  # quick tunnels can silently reconnect with a new hostname without the process dying


def tunnel_is_healthy(tunnel_url: str) -> bool:
    try:
        http_json("GET", f"{tunnel_url}/v1/models", timeout=8)
        return True
    except (urllib.error.URLError, socket.timeout):
        return False


def lan_ip() -> str | None:
    """This machine's address on the local network.

    Opens a UDP socket toward a public address and reads back which local
    interface the routing table chose. Nothing is actually sent -- UDP connect
    only fixes the socket's peer -- so this works with no internet access at
    all, which matters because the network it describes may be offline by
    design.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("10.255.255.255", 1))
        return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return None
    finally:
        sock.close()


def ollama_reachable_on_lan(ip: str) -> bool:
    """Can another machine on this LAN actually reach our Ollama?

    Ollama binds 127.0.0.1 by default, which is invisible to every other
    machine in the room. Registering anyway would put a node in the registry
    that health-checks green from its own machine and fails for everyone else
    -- the worst kind of broken, because the registry looks fine.
    """
    try:
        http_json("GET", f"http://{ip}:11434/v1/models", timeout=5)
        return True
    except (urllib.error.URLError, socket.timeout, OSError):
        return False


def start_tunnel() -> tuple[subprocess.Popen, str]:
    # Cloudflare's quick tunnel forwards the public tunnel hostname as the
    # Host header by default. Some Ollama versions reject any request whose
    # Host isn't localhost/127.0.0.1 (anti-DNS-rebinding protection) and
    # respond 403 -- force the header cloudflared sends to the origin so
    # that check always passes regardless of the Ollama version.
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", OLLAMA_URL, "--http-host-header", "localhost:11434"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    found_url: list[str] = []

    def read_output():
        for line in proc.stdout:
            match = TUNNEL_URL_PATTERN.search(line)
            if match and not found_url:
                found_url.append(match.group(0))

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if found_url:
            return proc, found_url[0]
        if proc.poll() is not None:
            die("cloudflared exited before a tunnel URL appeared")
        time.sleep(0.5)

    proc.terminate()
    die("timed out waiting for cloudflared to open a tunnel")


LAUNCHD_LABEL = "network.common.join"
SYSTEMD_UNIT = "common-join.service"
SCHTASKS_NAME = "CommonNetworkJoin"


def _service_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        sys.executable, os.path.abspath(__file__),
        "--gateway", args.gateway,
        "--model", args.model,
        "--name", args.name,
        "--operator", args.operator,
        "--cost", str(args.cost),
    ]
    if args.region:
        argv += ["--region", args.region]
    if args.capability:
        argv += ["--capability", args.capability]
    if args.no_update:
        argv += ["--no-update"]
    # Must be carried through, or a lab machine installed as a permanent
    # service comes back after reboot trying to open a tunnel that its own
    # network blocks -- and silently drops out of the lab.
    if args.lan:
        argv += ["--lan"]
    return argv


def _macos_plist_path() -> Path:
    d = Path.home() / "Library" / "LaunchAgents"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{LAUNCHD_LABEL}.plist"


def install_macos_service(argv: list[str]) -> None:
    plist_path = _macos_plist_path()
    log_path = Path.home() / ".common-network" / "join.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # launchd's default PATH doesn't include /usr/local/bin or /opt/homebrew/bin,
    # so ollama/cloudflared (installed there) won't resolve unless we carry over
    # the PATH from the shell that ran --permanent.
    with open(plist_path, "wb") as f:
        plistlib.dump({
            "Label": LAUNCHD_LABEL,
            "ProgramArguments": argv,
            "EnvironmentVariables": {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")},
            "RunAtLoad": True,
            "KeepAlive": True,
            "StandardOutPath": str(log_path),
            "StandardErrorPath": str(log_path),
        }, f)

    uid = os.getuid()
    subprocess.run(["launchctl", "bootout", f"gui/{uid}/{LAUNCHD_LABEL}"], capture_output=True)
    subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(plist_path)], check=True)
    print(f"{GLYPH_DONE} {blue('installed as a background service (LaunchAgent)', bold=True)} — starts at login, restarts if it crashes.")
    print(comment(f"logs: {log_path}"))
    print(comment(f"to stop: python3 {os.path.abspath(__file__)} --remove-permanent --no-update"))


def remove_macos_service() -> None:
    uid = os.getuid()
    subprocess.run(["launchctl", "bootout", f"gui/{uid}/{LAUNCHD_LABEL}"], capture_output=True)
    plist_path = _macos_plist_path()
    if plist_path.exists():
        plist_path.unlink()
    print(f"{GLYPH_DONE} removed the background service.")


def _systemd_unit_path() -> Path:
    d = Path.home() / ".config" / "systemd" / "user"
    d.mkdir(parents=True, exist_ok=True)
    return d / SYSTEMD_UNIT


def install_linux_service(argv: list[str]) -> None:
    unit_path = _systemd_unit_path()
    exec_start = " ".join(shlex.quote(a) for a in argv)
    path_env = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    unit_path.write_text(f"""[Unit]
Description=Common Network node
After=network-online.target

[Service]
Environment=PATH={path_env}
ExecStart={exec_start}
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
""")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", SYSTEMD_UNIT], check=True)
    print(f"{GLYPH_DONE} {blue('installed as a background service (systemd --user)', bold=True)} — starts at login, restarts if it crashes.")
    print(comment(f"logs: journalctl --user -u {SYSTEMD_UNIT} -f"))
    print(comment("if this is a server, also run `loginctl enable-linger $USER` so it keeps running after you log out."))
    print(comment(f"to stop: python3 {os.path.abspath(__file__)} --remove-permanent --no-update"))


def remove_linux_service() -> None:
    subprocess.run(["systemctl", "--user", "disable", "--now", SYSTEMD_UNIT], capture_output=True)
    unit_path = _systemd_unit_path()
    if unit_path.exists():
        unit_path.unlink()
    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    print(f"{GLYPH_DONE} removed the background service.")


def install_windows_service(argv: list[str]) -> None:
    cmd_str = " ".join(f'"{a}"' if " " in a else a for a in argv)
    subprocess.run([
        "schtasks", "/create", "/f", "/sc", "onlogon", "/rl", "highest",
        "/tn", SCHTASKS_NAME, "/tr", cmd_str,
    ], check=True)
    subprocess.run(["schtasks", "/run", "/tn", SCHTASKS_NAME], check=True)
    print(f"{GLYPH_DONE} {blue('installed as a scheduled task', bold=True)} — starts at login.")
    print(comment("Windows Task Scheduler doesn't auto-restart on crash like launchd/systemd do."))
    print(comment(f"to stop: python3 {os.path.abspath(__file__)} --remove-permanent --no-update"))


def remove_windows_service() -> None:
    subprocess.run(["schtasks", "/end", "/tn", SCHTASKS_NAME], capture_output=True)
    subprocess.run(["schtasks", "/delete", "/f", "/tn", SCHTASKS_NAME], capture_output=True)
    print(f"{GLYPH_DONE} removed the scheduled task.")


def install_permanent(args: argparse.Namespace) -> None:
    argv = _service_argv(args)
    system = platform.system()
    if system == "Darwin":
        install_macos_service(argv)
    elif system == "Linux":
        install_linux_service(argv)
    elif system == "Windows":
        install_windows_service(argv)
    else:
        die(f"--permanent isn't supported on {system}")


def remove_permanent() -> None:
    system = platform.system()
    if system == "Darwin":
        remove_macos_service()
    elif system == "Linux":
        remove_linux_service()
    elif system == "Windows":
        remove_windows_service()
    else:
        die(f"--permanent isn't supported on {system}")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    _enable_windows_ansi()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gateway", default=os.environ.get("COMMON_GATEWAY_URL", DEFAULT_GATEWAY), help="Gateway base URL (default: the shared Common Network gateway)")
    parser.add_argument("--model", default=None, help="Catalogue id (e.g. qwen2.5-coder-14b) or raw Ollama model tag. Default: probe hardware and ask the network what it needs most (see --auto, --list-catalogue)")
    parser.add_argument("--auto", action="store_true", help="Accept the network's recommended model automatically, no confirmation prompt")
    parser.add_argument("--list-catalogue", action="store_true", help="List catalogue models this machine can run, then exit")
    parser.add_argument("--name", default=f"{socket.gethostname()}-{os.environ.get('USER', 'node')}", help="Unique node name")
    parser.add_argument("--operator", default=os.environ.get("USER", "friend"), help="Your name")
    parser.add_argument("--region", default=None, help="Optional region hint, e.g. au-adelaide")
    parser.add_argument("--cost", type=float, default=0, help="Declared cost per 1k tokens (default: 0, it's free)")
    parser.add_argument("--capability", default=None, help="Override the auto-generated capability description")
    parser.add_argument("--no-update", action="store_true", default=bool(os.environ.get("COMMON_NO_UPDATE")), help="Skip the self-update check (useful when hacking on this script locally)")
    parser.add_argument("--lan", action="store_true", help="Join over the local network instead of a Cloudflare tunnel. For a computer lab or anywhere the gateway is on the same network — no tunnel, nothing exposed to the internet")
    parser.add_argument("--permanent", action="store_true", help="Install as a background service that starts at login/boot and restarts if it crashes, then exit")
    parser.add_argument("--remove-permanent", action="store_true", help="Remove the background service installed by --permanent, then exit")
    args = parser.parse_args()

    if not args.no_update:
        remote = fetch_update()
        if remote:
            print(f"{GLYPH_WORK} {dim('updating to the latest version...')}")
            apply_update_and_restart(remote)

    print(paper(BANNER, bold=True))

    if args.remove_permanent:
        remove_permanent()
        return

    gateway = args.gateway.rstrip("/")

    if args.list_catalogue:
        hw = probe_hardware()
        print_catalogue(gateway, hw)
        return

    ollama_tag, domain_tags, catalogue_id, capability_text_default = resolve_model(gateway, args)
    # Bake the resolution into args.model so --permanent's saved service
    # invocation re-resolves the same way on every restart -- the catalogue
    # id round-trips back through resolve_model()'s lookup branch; a raw tag
    # (no catalogue match) just flows through unchanged.
    args.model = catalogue_id or ollama_tag

    if args.permanent:
        install_permanent(args)
        return

    # LAN mode skips cloudflared entirely, so don't demand it be installed.
    check_binaries(need_tunnel=not args.lan)
    ensure_ollama_running()
    ensure_model(ollama_tag)

    tunnel_proc = None
    if args.lan:
        ip = lan_ip()
        if not ip:
            die("couldn't work out this machine's LAN address — pass a tunnel join instead (drop --lan)")
        if not ollama_reachable_on_lan(ip):
            die_with_fix(
                f"Ollama is running, but not reachable from the rest of the network on {ip}:11434.",
                "Ollama binds to localhost only by default. Restart it bound to the LAN:\n"
                "    macOS/Linux:  OLLAMA_HOST=0.0.0.0:11434 ollama serve\n"
                "    Windows:      setx OLLAMA_HOST 0.0.0.0:11434   (then restart Ollama)\n\n"
                "  If that's already set and this still fails, the network has client isolation\n"
                "  turned on — machines on the same wifi can't talk to each other. That needs\n"
                "  IT to disable it for this subnet.",
            )
        tunnel_url = f"http://{ip}:11434"
        print(f"{GLYPH_DONE} serving on the local network at {blue(tunnel_url)}")
        print(comment("no tunnel, nothing exposed to the internet — this machine is only "
                      "reachable from your own network."))
    else:
        working("opening a Cloudflare quick tunnel to your local Ollama...")
        tunnel_proc, tunnel_url = start_tunnel()
        print(f"{GLYPH_DONE} tunnel live at {blue(tunnel_url)}")

    capability_text = args.capability or capability_text_default or (
        f"{ollama_tag} running locally via Ollama, contributed by {args.operator}. Free, community-hosted."
    )

    payload = {
        "name": args.name,
        "operator": args.operator,
        "endpoint_url": f"{tunnel_url}/v1",
        "model_name": ollama_tag,
        "capability_text": capability_text,
        "region": args.region,
        "cost_per_1k": args.cost,
        "domain_tags": domain_tags,
        "catalogue_id": catalogue_id,
    }

    working(f"registering '{args.name}' with {gateway}...")
    try:
        node = http_json("POST", f"{gateway}/nodes", body=payload)
    except urllib.error.HTTPError as e:
        tunnel_proc.terminate()
        die(f"registration failed: {e.code} {e.read().decode()}")

    node_id = node["id"]
    node_token = node["node_token"]
    print(f"{GLYPH_DONE} {blue('you are live', bold=True)} · node id: {node_id}")
    print(comment("keep this window open to stay in the network — ctrl+c to leave."))

    write_identity(gateway, args.name, node_id, catalogue_id, domain_tags)

    def deregister_and_stop_tunnel():
        try:
            req = urllib.request.Request(
                f"{gateway}/nodes/{node_id}", method="DELETE", headers={"X-Common-Node-Token": node_token},
            )
            urllib.request.urlopen(req, timeout=5)
        except urllib.error.URLError:
            pass
        clear_identity()
        if tunnel_proc is not None:
            tunnel_proc.terminate()

    def cleanup(signum=None, frame=None):
        print(f"\n{GLYPH_WORK} {dim('leaving the network...')}")
        deregister_and_stop_tunnel()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    last_update_check = time.monotonic()
    last_tunnel_check = time.monotonic()
    while True:
        if tunnel_proc is not None and tunnel_proc.poll() is not None:
            print(f"{GLYPH_FORMING} {red('tunnel dropped unexpectedly.')}")
            cleanup()

        # A LAN node has no tunnel to rotate or restart, but its address can
        # still change under it -- DHCP renewal on a school network reassigns
        # addresses routinely, and a node registered at a stale IP is a node
        # the gateway will mark unhealthy for reasons nobody can see.
        if tunnel_proc is None and time.monotonic() - last_tunnel_check > TUNNEL_HEALTH_CHECK_INTERVAL_SECONDS:
            last_tunnel_check = time.monotonic()
            current = lan_ip()
            if current and f"http://{current}:11434" != tunnel_url:
                tunnel_url = f"http://{current}:11434"
                payload["endpoint_url"] = f"{tunnel_url}/v1"
                print(f"{GLYPH_FORMING} this machine's address changed — re-registering at {blue(tunnel_url)}")
                try:
                    node = http_json("POST", f"{gateway}/nodes", body=payload)
                    node_id, node_token = node["id"], node["node_token"]
                except urllib.error.HTTPError as e:
                    print(f"{GLYPH_FAILED} {red(f'failed to re-register on the new address: {e.code}')}", file=sys.stderr)

        if tunnel_proc is not None and time.monotonic() - last_tunnel_check > TUNNEL_HEALTH_CHECK_INTERVAL_SECONDS:
            last_tunnel_check = time.monotonic()
            if not tunnel_is_healthy(tunnel_url):
                print(f"{GLYPH_FORMING} {red('tunnel is no longer reachable')} (quick tunnels can silently rotate hostnames) — restarting it...")
                tunnel_proc.terminate()
                tunnel_proc, tunnel_url = start_tunnel()
                print(f"{GLYPH_DONE} tunnel live at {blue(tunnel_url)}")
                payload["endpoint_url"] = f"{tunnel_url}/v1"
                try:
                    node = http_json("POST", f"{gateway}/nodes", body=payload)
                    node_id = node["id"]
                    node_token = node["node_token"]
                    print(f"{GLYPH_DONE} re-registered '{args.name}' with the new tunnel url.")
                except urllib.error.HTTPError as e:
                    print(f"{GLYPH_FAILED} {red(f'failed to re-register after tunnel restart: {e.code} {e.read().decode()}')}", file=sys.stderr)

        if not args.no_update and time.monotonic() - last_update_check > UPDATE_CHECK_INTERVAL_SECONDS:
            last_update_check = time.monotonic()
            remote = fetch_update()
            if remote:
                print(f"\n{GLYPH_WORK} {dim('a new version is available — restarting to update...')}")
                deregister_and_stop_tunnel()
                apply_update_and_restart(remote)

        time.sleep(2)


if __name__ == "__main__":
    main()
