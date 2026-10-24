#!/usr/bin/env python3
"""Join the Common Network as a node operator.

The Common Network Alpha (v0.2.0-dev).

Copyright (C) 2026 Common AI Inc. Licensed under AGPL-3.0; see LICENSE at
https://github.com/TheCommonAI/common-network. This program comes with
ABSOLUTELY NO WARRANTY.

Runs a local model as a contributing node: fetches llama.cpp, fetches the
weights, opens a Cloudflare quick tunnel to it, registers the node with the
shared gateway, and keeps running until you press Ctrl+C (at which point it
deregisters cleanly). No account or signup needed for the tunnel.

Requires: Python 3.8+, cloudflared
(https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/).
Nothing else -- this script fetches its own model runtime (see runtime.py) and
its own weights. It used to require Ollama to be installed and running first,
which made donating a machine into installing a second product: a ~1GB
installer, a daemon with its own updater, and a private model store.

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
from __future__ import annotations

import argparse
import atexit
import importlib
import json
import os
import platform
import plistlib
import re
import secrets
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

# Sibling module: the installer puts join.py and worker.py in the same
# directory, and `python3 join.py` puts that directory on sys.path.
import worker

BANNER = r"""
 ░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓██████████████▓▒░░▒▓██████████████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░         
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░        
░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░        
░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░        
░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░        
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓██▓▒░ 
 ░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓██▓▒░ 
                                                                                                     
                                                                                                     
"""

# 127.0.0.1 instead of localhost: on Windows, localhost may resolve to ::1 (IPv6)
# while the service binds only to IPv4, causing cloudflared and self-checks to
# fail silently. Using the explicit IPv4 address works on all platforms.
#
# The model backend, which is llama-server now rather than Ollama. It keeps the
# old Ollama port on purpose, and one above it for the worker: a machine that
# already had Ollama running on 11434 does not conflict with either, and anyone
# who learned these two numbers from the old README finds them where they were.
LLAMA_PORT = 11436
LLAMA_URL = f"http://127.0.0.1:{LLAMA_PORT}"

# What actually faces the network. llama-server itself stays on 127.0.0.1: the
# worker is the only thing the tunnel (or the LAN) can reach, and it requires the
# gateway's token, serves two routes, and pins this node to one model. See
# worker.py for why that matters -- a bare llama-server has no authentication
# worth relying on, and its web UI is a file browser over the same directory the
# weights are in.
WORKER_PORT = 11435
WORKER_URL = f"http://127.0.0.1:{WORKER_PORT}"
# `api.trycloudflare.com` is Cloudflare's own endpoint for requesting a tunnel. cloudflared names it in its error
# lines (a failed first attempt on a slow or filtered network), so a first-match search published it as this
# peer's address and every request to the peer then answered HTTP 405.
TUNNEL_URL_PATTERN = re.compile(r"https://(?!api\.trycloudflare\.com)[a-zA-Z0-9.-]+\.trycloudflare\.com")

# cloudflared's own errors arrive on the same stream the tunnel URL does --
# stderr is merged into stdout in start_tunnel() -- and that stream was read for
# the URL and nothing else. So a refusal Cloudflare states in as many words
# ("429 Too Many Requests", "error code: 1015") reached the contributor as
# "cloudflared exited before a tunnel URL appeared": the symptom, with the cause
# deleted. Collapsing those two is not merely unhelpful, it is misleading, because
# they want opposite responses -- one says stop asking for a while, the other says
# the program is broken -- and the first is made worse by the retry the second
# invites.
#
# Bounded, because this accumulates somebody else's output while the process is
# alive and only the last few lines explain an exit.
TUNNEL_TAIL_LINES = 6

# Cloudflare answers quick-tunnel *creation* with 429 / error 1015 when the caller
# is rate limited, and its published guidance is to wait rather than to ask again:
# repeated attempts extend the block. cloudflared cannot be relied on to say so --
# it parses the response body as JSON whatever the status code, so the 429 arrives
# as `invalid character 'e' looking for beginning of value`
# (cloudflare/cloudflared#1302). The status code in the text is the part worth
# matching on, and it is why this reads the output rather than the exit status:
# cloudflared exits 1 for a rate limit, a bad flag and a dead DNS alike.
TUNNEL_RATE_LIMITED_PATTERN = re.compile(r"\b1015\b|\b429\b|too many requests", re.IGNORECASE)

# The default, shared Common Network gateway. Override with --gateway to
# join (or run) a different network entirely.
DEFAULT_GATEWAY = "https://gateway-production-b820.up.railway.app"

VERSION = "0.2.0-dev"
RELEASE = "The Common Network Alpha"
REPO = "TheCommonAI/common-network"
UPDATE_URL = f"https://raw.githubusercontent.com/{REPO}/main/join/join.py"

# Everything join.py drives that lives beside it, and has to be refreshed when
# join.py refreshes itself. See refresh_siblings().
SIBLING_MODULES = ("worker.py", "runtime.py")
SIBLING_UPDATE_URL = f"https://raw.githubusercontent.com/{REPO}/main/join/{{}}"

# The schemes a `--model` value may name explicitly. Anything else is read as a
# bare Ollama registry tag, which is what `--model mathstral:7b` has always
# meant. Membership here rather than a look at whether a colon is present: a raw
# tag has one too, so a colon test would misread the commonest input there is.
KNOWN_SOURCE_SCHEMES = frozenset({"ollama", "gguf"})

# How long to give llama-server to load its weights and answer /health. It only
# has to cover the very first load, because the process keeps the weights
# resident afterwards; a restart finds the file on disk and skips most of it.
# Measured on this codebase's own machine: ~1s for a 1GB model and ~21s for a
# 1.6GB one, with llama-server answering 503 "Loading model" while it works. A
# 9GB model on a cold laptop is the case this number is actually for, and the
# old backend's 53-77s first token is the only figure anyone has measured there.
# Too short and a slow machine looks broken; there is no cost to being generous,
# since it is one wait at startup.
LLAMA_STARTUP_TIMEOUT_SECONDS = 180

# Which program registered this node.
#
# The desktop app and this CLI deliberately share everything that matters --
# the same gateway, the same endpoints, the same node_token in the same
# ~/.common-network/identity.json -- so that one machine is one node however it
# joined. That is the right default and it leaves the two indistinguishable in
# the registry: `GET /nodes` describes the machine, never the program that
# spoke for it. This string is the only thing that separates them, and without
# it the gateway cannot answer "how much of the network is the app, and how
# much is the terminal".
#
# Sent as X-Common-Client on every gateway request and repeated as the `client`
# field on POST /nodes -- the same two ways common-desktop sends
# "common-desktop/<its version>". Both are self-reported and trivially
# forgeable, so this is a usage statistic and never an authorisation input:
# nothing in the gateway may branch on it.
CLIENT = f"common-cli/{VERSION}"


def _exec(argv: list[str]) -> None:
    """Replace this process with argv. Windows has no exec: os.execv there starts a new process and exits
    at once, which hands the console back to the shell while the program is still running (the prompt
    returns, output arrives later, Ctrl+C does nothing). So wait for the child and pass its exit code on."""
    if os.name != "nt":
        os.execv(argv[0], argv)
        return
    child = subprocess.Popen(argv)
    while True:
        try:
            sys.exit(child.wait())
        except KeyboardInterrupt:
            continue  # the child got Ctrl+C too; wait for it to leave cleanly


def gateway_headers(extra: dict | None = None) -> dict:
    """Headers every gateway request carries.

    Mirror of common-desktop's gatewayHeaders(). Deliberately not applied to
    local requests -- llama-server, or this node's own worker health checks --
    which have no business carrying the network's client string.
    """
    return {"X-Common-Client": CLIENT, **(extra or {})}


# The model-runtime module, imported on first use rather than at module scope.
# Worker is imported above because every release has shipped it; runtime.py
# cannot be, and the reason is worth spelling out because it is the one place
# where the upgrade path is not simply "replace the file".
#
# A machine running the release before this one updates itself with an updater
# that knows only about join.py and worker.py -- it has never heard of runtime.py,
# because runtime.py did not exist when it was written. So the first time that
# machine lands *this* file, the new join.py is on disk and runtime.py is not,
# and a bare `import runtime` at the top of the file would kill the process on
# its own self-update. Every node in the network would update itself to death,
# once, silently, on the same afternoon.
#
# Importing it where it is used, after a sibling refresh, is what makes that
# upgrade land in one step. See _require_runtime().
runtime = None


def _import_runtime():
    """`runtime.py` itself, or None — never something that merely answers to the name.

    A bare `import runtime` cannot answer "is runtime.py here?". `runtime/` is a
    real directory in this install — the one the llama.cpp build unpacks into —
    and Python imports a directory with no `__init__.py` as a *namespace
    package*. So on a machine where runtime.py was never written the import
    succeeds and hands back the directory instead, and the failure lands at the
    first call into it:

        AttributeError: module 'runtime' has no attribute 'is_installable'

    which names neither the missing file nor the directory standing in for it,
    and reads like a bug in the runtime rather than an install missing a file. A
    genuinely absent module would have said ModuleNotFoundError.

    It is a fallback, not an override, and that is worth knowing: the finder
    records the namespace portion and keeps looking, so a real runtime.py on any
    path entry wins. Which means the directory only ever answers for runtime.py
    when there is no runtime.py anywhere at all — the exact state of the install
    this was found on, and the reason "just add the file" is the whole fix.

    That is also what made the recovery in _require_runtime() unreachable: the
    ImportError it caught never arrived, so refresh_siblings() was never asked
    for runtime.py on the machines that had lost it. Absence has to be tested
    for, not inferred from whether an import raised.
    """
    try:
        mod = importlib.import_module("runtime")
    except ImportError:
        return None
    # A namespace package has no `__file__` at all, and the sibling we want is
    # the file sitting beside join.py. Both tests are needed: the path alone can
    # be satisfied by some unrelated runtime.py earlier on sys.path, and the
    # attribute alone by a directory that happens to define it.
    path = getattr(mod, "__file__", None)
    if not path or os.path.basename(path) != "runtime.py":
        return None
    return mod if hasattr(mod, "is_installable") else None


def _require_runtime():
    """The model-runtime module, fetching it first if this install predates it."""
    global runtime
    if runtime is None:
        mod = _import_runtime()
        if mod is None:
            refresh_siblings()  # fetches runtime.py; see SIBLING_MODULES
            # Both lines are load-bearing, and neither is decoration. The
            # namespace package that stood in for runtime.py is in sys.modules
            # by now, so a second import would return that same impostor; and
            # the import machinery has cached the directory listing it built
            # before runtime.py was written, so it would not see the new file
            # even if it did re-import.
            sys.modules.pop("runtime", None)
            importlib.invalidate_caches()
            mod = _import_runtime()
        if mod is None:
            die_with_fix(
                "the model runtime module is missing.",
                f"join.py needs runtime.py beside it, and could not fetch it.\n"
                f"  Check your connection, then run this again — or install both by hand\n"
                f"  from https://github.com/{REPO}/tree/main/join",
            )
        runtime = mod
    return runtime


def _enable_windows_ansi() -> None:
    if platform.system() != "Windows":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass


def _force_utf8_output() -> None:
    """Pin stdout/stderr to UTF-8 whatever the Windows code page says.

    A Windows console handles the glyph vocabulary (→ ✓ ⚠ ✗) and the banner
    fine -- Python writes it through the UTF-16 console API. A *redirected*
    stream does not: a pipe or a file gets the locale encoding, which is
    cp1252 on most installs, and cp1252 cannot encode any of those characters.
    The result is UnicodeEncodeError and a dead process.

    That is not a cosmetic problem, because the two places output is redirected
    are the two that matter most here:

      - `common join --permanent` writes the scheduled task as
        `cmd /c "... >> join.log 2>&1"`, so every run of the service sends both
        streams to a file. The service then died printing its own banner --
        before ensure_model, before the worker, before registering. A Windows
        node installed this way never came up and never installed its model.
      - `common leave` runs `--remove-permanent` with capture_output=True and
        greps the result for "Removed". The child was killed by the same
        UnicodeEncodeError printing the ✓, so the message was never written and
        the check could not succeed.

    errors="replace" is deliberate rather than strict: a glyph is decoration,
    and decoration must never be able to stop a node from starting. A stream
    that cannot be reconfigured at all (already replaced, closed, or not a
    TextIOWrapper) is left alone -- the banner is worth less than the run.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
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


def default_node_name() -> str:
    """A fresh node name: `common-node` and nine random digits.

    Not the hostname. A machine whose hostname is its LAN address, or a laptop
    named after its owner, would publish that to the whole network, and the
    gateway's node list is public. Nine digits is enough that two nodes picking
    one by chance is unlikely, and registration already refuses a name in use.
    """
    return f"common-node{secrets.randbelow(10 ** 9):09d}"


def write_identity(gateway: str, name: str, node_id: str, catalogue_id: str | None, domain_tags: list[str] | None, node_token: str | None = None) -> None:
    """Local record of the most recent node this machine registered -- read by
    `common status` / `common whoami` / `common contrib`. Not a cryptographic
    identity (see COMMON. CLI design doc's local-keypair vision) -- just a
    name, deliberately simple until that's a considered decision, not a guess.

    The node_token is kept here so a re-join can prove it owns the name: the
    gateway only lets an existing node re-register from whoever holds its
    token, and after a crash (no clean deregister) the token is the only way
    back in under the same name."""
    try:
        IDENTITY_PATH.parent.mkdir(parents=True, exist_ok=True)
        IDENTITY_PATH.write_text(json.dumps({
            "gateway": gateway, "name": name, "node_id": node_id,
            "catalogue_id": catalogue_id, "domain_tags": domain_tags,
            "node_token": node_token,
            "joined_at": time.time(),
        }))
    except OSError:
        pass  # best-effort -- never block joining the network over this


def read_identity() -> dict | None:
    try:
        return json.loads(IDENTITY_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None


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


# How long to wait on the gateway, and how hard to insist before giving up.
#
# The rule is the one `common` and `chat` already follow, for the same reason:
# the gateway must be the one to give up first, so that what a contributor sees
# is the gateway's own error rather than a bare socket timeout. `join` was the
# client that did not -- its gateway calls took http_json's 10s default, which
# is less than the gateway's database command timeout (db_command_timeout_seconds,
# 30s), let alone its read budget. A slow answer was therefore guaranteed to
# surface as
#
#     TimeoutError: The read operation timed out
#
# which says nothing about what happened.
#
# 90s, the same value as REGISTRATION_TIMEOUT, is sized against what makes the
# first call of a join slow: a gateway container that has gone to sleep accepts
# the connection at once and then takes seconds to answer. Measured here at 8.1s
# to first byte cold against 0.53s warm -- comfortably inside 90s, and still
# inside the gateway's own budget, so a gateway that is genuinely unwell still
# fails first and still says why.
#
# gateway/tests/test_timeouts.py reads this name, alongside the other two clients'.
REQUEST_TIMEOUT_SECONDS = 90

# A transport failure is usually a container still waking, and the request after
# it is served normally -- that is the pair measured above. So it is worth asking
# again, but only for a while: this caps the total time spent retrying rather
# than each attempt, so a gateway that is simply gone costs one bounded wait
# instead of one full timeout per attempt.
GATEWAY_RETRY_MAX_WAIT_SECONDS = 60.0
GATEWAY_RETRY_BACKOFF_SECONDS = 2.0


def gateway_json(method: str, url: str, body: dict | None = None, headers: dict | None = None):
    """A gateway call, retried while the failure is transport-shaped.

    An HTTPError is an answer -- the gateway replied, and asking again would
    produce the same reply -- so it is raised at once. A timeout or a dropped
    connection is the absence of an answer, which on this network usually means
    a container that has not finished starting.

    /catalogue and /assign are both reads; /assign does three SELECTs and ranks
    what it finds. A repeat therefore cannot alter anything a first call did.

    Registration deliberately does not come through here. It fails for a
    different reason -- a tunnel hostname the gateway cannot resolve yet -- and
    retries on its own terms, holding the tunnel open while it waits.
    """
    deadline = time.monotonic() + GATEWAY_RETRY_MAX_WAIT_SECONDS
    delay = GATEWAY_RETRY_BACKOFF_SECONDS
    while True:
        try:
            return http_json(method, url, body=body, headers=headers,
                             timeout=REQUEST_TIMEOUT_SECONDS)
        except urllib.error.HTTPError as e:
            if e.code == 410:
                die("the old Common gateway has been retired, so `--legacy` and --model no longer work. "
                    "Run `common join` (no flags) to join the new decentralised network on a Mac with "
                    "Apple silicon, a PC with an NVIDIA GPU, or any PC (it serves on its CPU).")
            raise  # a real answer -- and an HTTPError is a URLError, so this comes first
        except (urllib.error.URLError, socket.timeout, OSError):
            if time.monotonic() >= deadline:
                raise
            print(comment(f"the gateway did not answer; trying again in {delay:.0f}s"))
            time.sleep(delay)
            delay = min(delay * 2, 10.0)


UPDATE_CHECK_INTERVAL_SECONDS = 1800  # re-check every 30 min while a node sits running


def fetch_update() -> bytes | None:
    """Return the latest join.py source from GitHub if it differs from the local copy, else None."""
    try:
        with urllib.request.urlopen(UPDATE_URL, timeout=5) as resp:
            remote = resp.read()
    except urllib.error.HTTPError as e:
        # See the matching note in common/common.py. Offline is normal and
        # silent; a 401/403/404 means the update channel is broken for every
        # machine at once, which is worth one line rather than nothing.
        if e.code in (401, 403, 404):
            print(f"{GLYPH_FORMING} {dim(f'updates unreachable ({e.code}) — running the installed version.')}",
                  file=sys.stderr)
            print(comment(f"source: {UPDATE_URL}"), file=sys.stderr)
        return None
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


def refresh_siblings() -> bool:
    """Bring every sibling module join.py drives up to date with this script.

    join.py imports the worker and the runtime as siblings, so a self-update
    that replaced only join.py would leave a new joiner driving an old worker or
    an old runtime -- exactly the kind of version skew that shows up as a
    security fix silently not applying. A failure here must not block the join,
    because a working old sibling beats no node at all. It is reported through
    the return value, though, because the caller about to replace join.py has to
    hold off when it cannot be sure the siblings match it.

    Writes only what differs, so the common case is two small GETs per run. Note
    that this compares contents and nothing else -- there is no checksum or
    signature on this path, which is pre-existing behaviour rather than an
    oversight introduced here, and is worth knowing when reasoning about what a
    compromised raw.githubusercontent.com would be able to do.

    Called from three places: at startup, so a node left with a stale worker.py
    recovers; before join.py overwrites itself, so the next process starts with
    siblings that match it; and from _require_runtime() when runtime.py turns out
    not to be here at all. The last is what makes a version that predates
    runtime.py able to upgrade onto it.

    Returns True only when every sibling was confirmed current.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    current = True
    for name in SIBLING_MODULES:
        try:
            with urllib.request.urlopen(SIBLING_UPDATE_URL.format(name), timeout=5) as resp:
                remote = resp.read()
            if not remote.strip():
                current = False
                continue
            path = os.path.join(here, name)
            try:
                with open(path, "rb") as f:
                    if f.read() == remote:
                        continue
            except OSError:
                pass  # missing locally -- writing it below is the fix
            with open(path, "wb") as f:
                f.write(remote)
        except (urllib.error.URLError, socket.timeout, OSError):
            current = False
    return current


def apply_update_and_restart(remote: bytes) -> None:
    """Overwrite this script with `remote` and re-exec. Never returns on success.

    The caller has already brought the siblings up to date and seen
    refresh_siblings() return True; see the update checks in main().
    """
    local_path = os.path.abspath(__file__)
    try:
        with open(local_path, "wb") as f:
            f.write(remote)
    except OSError as e:
        print(f"{GLYPH_FORMING} {red(f'could not self-update ({e}), continuing with current version')}", file=sys.stderr)
        return
    _exec([sys.executable, local_path] + sys.argv[1:])


def die_with_fix(msg: str, fix: str) -> None:
    print(f"{GLYPH_FAILED} {red(msg)}", file=sys.stderr)
    print(comment(fix), file=sys.stderr)
    sys.exit(1)


def check_binaries(need_tunnel: bool = True) -> None:
    """The one external program this needs. Everything else it fetches itself.

    There is no longer a check for the model runtime here: llama.cpp is not
    something a contributor installs, it is something this script downloads, so
    a missing runtime is a job for ensure_model_runtime() rather than a reason to
    refuse to start. What is left is cloudflared, which is a real thing the user
    must have and the one a school network is most likely to block outright.

    --lan needs no tunnel, which is most of the point of it.
    """
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


def fetch_model(spec) -> Path:
    """Get this node's weights onto disk, and say so while it happens.

    Two ways in, and the quiet one matters most. A machine that has already run
    `ollama pull` for this model does not download anything -- runtime.py finds
    the same content-addressed blob in Ollama's own store and hardlinks it, which
    is why swapping runtimes costs an existing contributor nothing. The 3-9GB
    download only happens on a machine that has never had this model, and there
    the disk-space guard runs first, because finding out 6GB in that there was
    never room for it is a worse experience than being told up front.
    """
    mod = _require_runtime()
    try:
        return mod.ensure_gguf(spec)
    except mod.FetchFailed as e:
        die_with_fix(f"could not prepare the model {spec.wire_name}: {e}",
                     "Check your connection and run this again — a partly-finished\n"
                     "  download resumes where it stopped rather than starting over.")


def stop_process(proc: subprocess.Popen | None, timeout: float = 5.0) -> None:
    """Stop a child process we started, and don't leave it behind.

    Both children here -- llama-server and cloudflared -- are things this
    process is responsible for; neither is useful once this one is gone, and
    llama-server in particular *keeps the port*. An orphaned one is worse than
    untidy: the next `common join` would start a second llama-server that cannot
    bind 11436, and whichever of the two is answering /health is then a coin toss
    the new node loses silently, serving a model it did not intend to.

    escalate after a grace period rather than blocking on a process that has
    decided not to die.
    """
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass


def sleep_idle_seconds(mod) -> int:
    """How long the model stays resident with no traffic, env or runtime default.

    One reading for two consumers: llama-server is started with it, and the
    worker needs the same number to schedule its keep-warm pings inside the
    window. Both sides reading the env separately would be two parses to let
    drift apart.
    """
    raw = os.environ.get("COMMON_SLEEP_IDLE_SECONDS")
    try:
        return int(raw) if raw else mod.DEFAULT_SLEEP_IDLE_SECONDS
    except ValueError:
        return mod.DEFAULT_SLEEP_IDLE_SECONDS


def start_llama_server(gguf: Path, wire_name: str, port: int = LLAMA_PORT) -> subprocess.Popen:
    """Start llama-server on `gguf`, and block until it is actually serving.

    Returns the process, which the main loop watches and cleanup() stops.

    `wire_name` is passed as llama-server's `--alias`, so that the `model` field
    of a completion comes back as the name this node registered under rather than
    as the absolute path of the weights on this machine. See llama_server_argv.

    The wait is not a formality. Loading several gigabytes of weights and
    building the KV cache takes the better part of a minute on a laptop the first
    time, and llama-server refuses a connection until it is done. Starting the
    worker and the tunnel only once `/health` answers is what keeps the gateway
    from being handed an endpoint that 502s for the first minute of its life -- a
    window in which every health check the network runs marks this node broken,
    and a node marked broken at registration is slow to be routed to again.
    """
    mod = _require_runtime()
    try:
        binary = mod.ensure_llama_server()
    except mod.FetchFailed as e:
        die_with_fix(f"could not set up the model runtime: {e}",
                     f"Nothing was changed on this machine. Run this again to retry,\n"
                     f"  or fetch a build by hand from https://github.com/{mod.LLAMA_REPO}/releases")

    sleep_idle = sleep_idle_seconds(mod)

    argv = mod.llama_server_argv(binary, gguf, port, alias=wire_name,
                                 sleep_idle_seconds=sleep_idle)
    print(f"{GLYPH_ROUTE} {dim('starting the model runtime...')}")
    log_path = Path.home() / ".common-network" / "runtime.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("ab") as log:
            proc = subprocess.Popen(argv, stdout=log, stderr=log)
    except OSError as e:
        die_with_fix(f"couldn't start llama-server: {e}", "Run `common join` again to retry the fetch.")

    # Registered here, the moment the child exists, because the exits that leak
    # it are not the orderly ones. `die()` covers registration failures, a
    # traceback covers the rest, and neither unwinds through cleanup() -- and on
    # Windows a killed parent does not take its child with it at all. This is the
    # only hook that runs on every one of those paths.
    atexit.register(stop_process, proc)
    return proc


def wait_for_llama(proc: subprocess.Popen, gguf: Path, port: int = LLAMA_PORT) -> bool:
    """Wait for llama-server to answer its health endpoint. True if it did.

    `/health` is deliberately public and needs no API key, so this is a plain
    local request with none of the worker's ceremony. It answers 503 with a
    "loading model" body while the weights are still coming off disk, which is
    the difference between "not ready yet" and "dead" -- so it is a 503 to keep
    waiting on, not an error.

    A 200 is not on its own proof that *our* server is the one that answered.
    Nothing stops a llama-server left over from an earlier run from holding
    11436, in which case the health check passes immediately, the node registers,
    and every request is answered by a different model than the one this node
    claims to serve -- the one failure here that is both silent and wrong. So the
    200 is confirmed against /props' model_path, and a mismatch is fatal and
    says why.
    """
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + LLAMA_STARTUP_TIMEOUT_SECONDS
    last_detail = "no response"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            die_with_fix(f"the model runtime exited with code {proc.returncode}.",
                         "See ~/.common-network/runtime.log for the runtime error.")
        try:
            http_json("GET", f"{url}/health", timeout=5)
        except urllib.error.HTTPError as e:
            last_detail = f"{e.code} {e.read().decode(errors='ignore').strip()}"
        except (urllib.error.URLError, socket.timeout, OSError) as e:
            last_detail = str(e)
        else:
            try:
                props = http_json("GET", f"{url}/props", timeout=5)
            except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout, OSError):
                # /props went away between the two calls; ask again next tick.
                last_detail = "health answered but /props did not"
            else:
                serving = str(props.get("model_path") or "")
                if serving and Path(serving).resolve() != Path(gguf).resolve():
                    die_with_fix(
                        f"another model server is already using port {port}.",
                        f"It is serving:\n"
                        f"    {serving}\n"
                        f"  but this node is set up for:\n"
                        f"    {gguf}\n"
                        f"  Stop that server and run this again — otherwise this node\n"
                        f"  would register as one model while answering as another.",
                    )
                return True
        time.sleep(0.5)
    print(f"{GLYPH_FAILED} {red('the model runtime did not come up within ')}"
          f"{LLAMA_STARTUP_TIMEOUT_SECONDS}s — {dim(last_detail)}")
    return False


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
    return gateway_json("GET", f"{gateway}/catalogue", headers=gateway_headers())


def call_assign(gateway: str, hardware: dict) -> dict:
    return gateway_json("POST", f"{gateway}/assign", body={"hardware": hardware}, headers=gateway_headers())


def resolve_source(source: str):
    """Turn a catalogue `source` into something installable, or None.

    The single interpreter of the source scheme, in runtime.py beside the code
    that acts on it. Used to be a one-liner here that stripped "ollama:" and
    nothing else; the name it produced was then handed straight to Ollama. Now
    the same string is the node's public model name as well, so it is worth
    having one place that decides what a source means. See runtime.ModelSpec.
    """
    return _require_runtime().resolve_source(source)


def is_installable(source: str) -> bool:
    """Whether common-join can provision this catalogue entry itself.

    Asked far more often than the spec is needed -- for listing, for ranking,
    for the "is there a local alternative" question -- and asking it should not
    drag the spec's construction along for the ride.
    """
    return _require_runtime().is_installable(source)


def prompt_yes_no(question: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    if not sys.stdin.isatty():
        return default_yes

    # Write the question and flush it BEFORE reading, rather than handing it to
    # input() as its prompt argument.
    #
    # On Windows, input()'s own prompt bypasses Python's stdout buffer, so it
    # can appear on screen before everything printed above it. The reported
    # symptom is exactly that: the cursor lands on a fresh line, the joining
    # output arrives afterwards, and the y/n question ends up scrolled above
    # the cursor with no obvious way to answer it.
    #
    # Flushing everything first, then writing the question, then reading,
    # forces the order the user actually sees to match the order it was
    # printed.
    sys.stdout.write(f"{question} {suffix} ")
    sys.stdout.flush()
    try:
        answer = sys.stdin.readline()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if not answer:          # EOF -- stdin closed under us
        print()
        return default_yes
    answer = answer.strip().lower()
    if not answer:
        return default_yes
    return answer.startswith("y")


def print_catalogue(gateway: str, hw: dict) -> None:
    catalogue = fetch_catalogue(gateway)
    headroom_ram = hw["available_ram_gb"] * 0.8
    print(dim(f"your machine: {hw['total_ram_gb']}GB RAM ({hw['available_ram_gb']}GB available), "
              f"{'GPU present' if hw['gpu_present'] else 'no GPU'}, {hw['cpu_cores']} CPU cores, {hw['os']}\n"))

    installable_count = 0
    fits_if_freed = []

    for m in catalogue:
        # Three distinct states, not two. The old version computed "runnable"
        # from RAM and GPU alone, so an entry this tool cannot install at all
        # (an `api:` or `graph:` source, which resolve_model rejects outright)
        # displayed as "runnable" whenever its min_ram_gb was low -- and the
        # old api:cgla-legal entry declared 0GB, so it showed as the ONE
        # runnable option on a machine where nothing else fit. Advertising a
        # model and then refusing to install it is worse than not listing it.
        provisionable = is_installable(m["source"])
        fits = m["min_ram_gb"] <= headroom_ram and (not m["needs_gpu"] or hw["gpu_present"])

        if not provisionable:
            kind = m["source"].split(":", 1)[0]
            mark = f"{GLYPH_FAILED} {dim(f'not installable by common-join ({kind} source)')}"
        elif fits:
            mark = f"{GLYPH_DONE} runnable"
            installable_count += 1
        else:
            mark = f"{GLYPH_FAILED} {red('needs more RAM/GPU')}"
            if not m["needs_gpu"] and m["min_ram_gb"] <= hw["total_ram_gb"] * 0.8:
                fits_if_freed.append(m)

        verified = blue("  [verified beats frontier in lane]") if m["verified_in_lane"] else ""
        print(f"  {m['id']:20s} {mark}{verified}")
        print(dim(f"    {m['capability_text'].strip()}"))
        print(dim(f"    tags: {', '.join(m['domain_tags'])}  ·  min RAM: {m['min_ram_gb']}GB  ·  source: {m['source']}\n"))

    # "Everything failed" is not a useful place to leave someone. Almost always
    # the machine has the RAM and something else is holding it -- this box has
    # 16GB total and 4.1GB free, which fits nothing while the total would fit
    # most of the catalogue.
    if installable_count == 0:
        print(red("nothing here can run on this machine right now."))
        if fits_if_freed:
            smallest = min(fits_if_freed, key=lambda m: m["min_ram_gb"])
            need = smallest["min_ram_gb"] / 0.8
            print(comment(f"you have {hw['total_ram_gb']}GB of RAM but only "
                          f"{hw['available_ram_gb']}GB free. closing some apps would be enough — "
                          f"{smallest['id']} needs about {need:.0f}GB available."))
        else:
            print(comment("this machine is below the smallest entry in the catalogue."))


def _stronger_if_freed(catalogue: list[dict], hw: dict, chosen_id: str | None) -> dict | None:
    """The best model this machine could run if memory were freed — when that
    is materially better than what fits right now.

    The network asks `/assign` what to run, and `/assign` only knows how much
    RAM is *available*, so a 16GB laptop with a browser open is assigned a
    1.5B model and joins as a permanently weak node. Nobody is told that
    closing a few tabs would have contributed an 8B one instead.

    Free memory is the most recoverable constraint a contributor has. It is
    worth one line, because the decision it changes lasts as long as the node
    does.

    Deliberately quiet about small gains: a suggestion has to be worth acting
    on or it becomes noise that gets ignored when it matters. The bar is
    roughly double the parameters, or +3B, whichever is higher.
    """
    now = hw["available_ram_gb"] * 0.8
    potential = hw["total_ram_gb"] * 0.8

    def size(m: dict) -> float:
        return float(m["params_b"] or 0)

    chosen = next((m for m in catalogue if m["id"] == chosen_id), None)
    chosen_size = size(chosen) if chosen else 0.0

    candidates = [
        m for m in catalogue
        if is_installable(m["source"])                            # installable
        and m["min_ram_gb"] > now                                 # doesn't fit now
        and m["min_ram_gb"] <= potential                          # would fit freed
        and (not m["needs_gpu"] or hw["gpu_present"])
        and size(m) >= max(chosen_size * 2, chosen_size + 3)      # materially better
    ]
    return max(candidates, key=size) if candidates else None


def _warn_if_underpowered(gateway: str, hw: dict, chosen_id: str | None, chosen_name: str) -> None:
    try:
        catalogue = fetch_catalogue(gateway)
    except (urllib.error.URLError, socket.timeout):
        return
    better = _stronger_if_freed(catalogue, hw, chosen_id)
    if not better:
        return
    needed = better["min_ram_gb"] / 0.8
    print(f"{GLYPH_FORMING} {red('this machine can do much better than that.')}")
    print(comment(f"{better['display_name']} needs about {needed:.0f}GB free. you have "
                  f"{hw['total_ram_gb']}GB of RAM but only {hw['available_ram_gb']}GB is free "
                  f"right now, so the network is offering you a smaller model than your "
                  f"hardware deserves."))
    print(comment("a node keeps whatever model it joined with, so this is worth fixing first."))
    print(dim(f"  → close some apps, then re-run:   common join"))
    print(dim(f"  → or continue with {chosen_name} anyway"))
    print()


def resolve_model(gateway: str, args: argparse.Namespace) -> tuple[str, list[str] | None, str | None, str | None]:
    """Returns (model_source, domain_tags, catalogue_id, capability_text_default).

    The first element is the catalogue `source` string, not a tag: what to run is
    now a question with more than one kind of answer, and the thing that should
    be handed to ensure_model/start_llama_server is the spec it resolves to.
    """
    if args.model:
        catalogue = fetch_catalogue(gateway)
        match = next((m for m in catalogue if m["id"] == args.model), None)
        if match:
            if not is_installable(match["source"]):
                die(f"'{args.model}' is an API-based catalogue entry ({match['source']}) — not something common-join can run locally.")
            return match["source"], match["domain_tags"], match["id"], match["capability_text"]
        # Not a catalogue id -- treat it as a raw source string, which for a bare
        # `mathstral:7b` means "that tag in Ollama's registry", same as before.
        # Deliberately checked against the known schemes rather than for a colon:
        # a raw tag contains one too, and reading `mathstral:7b` as `scheme:path`
        # would reject the most ordinary way to name a model.
        scheme = args.model.split(":", 1)[0]
        if scheme in KNOWN_SOURCE_SCHEMES:
            source = args.model
        else:
            source = f"ollama:{args.model}"
        if not is_installable(source):
            die(f"'{args.model}' is not something common-join can install "
                f"(unrecognised source '{source}').")
        return source, None, None, None

    working("probing your machine's hardware...")
    hw = probe_hardware()
    print(dim(f"  {hw['total_ram_gb']}GB RAM ({hw['available_ram_gb']}GB available), "
              f"{'GPU present' if hw['gpu_present'] else 'no GPU'}, {hw['cpu_cores']} CPU cores, {hw['os']}"))

    assignment = call_assign(gateway, hw)
    print(f"{GLYPH_ROUTE} recommended   {blue(assignment['display_name'], bold=True)}")
    print(comment(assignment["reason"]))
    print()

    # Before anyone agrees to provision, say whether free memory is what is
    # holding this machine back. /assign cannot know -- it is only told how
    # much RAM is available, not how much exists.
    _warn_if_underpowered(gateway, hw, assignment.get("catalogue_id"),
                          assignment["display_name"])

    if not is_installable(assignment["source"]):
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
        return fallback["source"], fallback["domain_tags"], fallback["id"], fallback["capability_text"]

    if not args.auto:
        if not prompt_yes_no(f"provision {assignment['display_name']} on this machine?"):
            print(dim("no changes made."))
            print(dim("  → pick manually:   common-join --model <id>"))
            print(dim("  → see options:     common-join --list-catalogue"))
            sys.exit(0)

    return assignment["source"], assignment["domain_tags"], assignment["catalogue_id"], assignment["capability_text"]


def _best_local_fallback(gateway: str, hw: dict, exclude_id: str | None) -> dict | None:
    """Best locally-runnable catalogue entry this hardware can fit, for when
    /assign's top pick is an API-hosted specialist common-join can't
    provision. Prefers verified-in-lane, then the smallest model (safest
    fit under headroom)."""
    catalogue = fetch_catalogue(gateway)
    headroom_ram = hw["available_ram_gb"] * 0.8
    candidates = [
        m for m in catalogue
        if m["id"] != exclude_id
        and is_installable(m["source"])
        and m["min_ram_gb"] <= headroom_ram
        and (not m["needs_gpu"] or hw["gpu_present"])
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda m: (not m["verified_in_lane"], m["params_b"] or 0))
    return candidates[0]


TUNNEL_HEALTH_CHECK_INTERVAL_SECONDS = 120  # quick tunnels can silently reconnect with a new hostname without the process dying

# Registration can block while the gateway retries DNS resolution (up to ~30s).
# This timeout must exceed that window, otherwise the client aborts mid-retry.
REGISTRATION_TIMEOUT = 90  # seconds — comfortably above gateway's 30s DNS retry

# How a re-announce is retried once the gateway's copy of our address has gone
# stale. Short and bounded, unlike the initial registration's 60s, because this
# runs on the serving loop -- the same loop that notices a dead tunnel or a
# dead model, and that must not sit in a retry while it does. Giving up here is
# safe in a way it is not there: the disagreement this exists to fix outlives
# the attempt, so the next pass simply tries again.
# A restarted quick tunnel's hostname can take well over the old three attempts
# (about six seconds) to resolve, and the gateway only waits about 30s itself. So
# the schedule runs to roughly a minute: 2, 4, 8, then 10s between the rest.
REANNOUNCE_ATTEMPTS = 8
REANNOUNCE_BACKOFF_SECONDS = 2
REANNOUNCE_MAX_BACKOFF_SECONDS = 10


def tunnel_is_healthy(tunnel_url: str, worker_token: str | None, path: str = "/v1/models") -> bool:
    """Is the tunnel still carrying traffic to our worker?

    Needs the token now: the worker 401s an unauthenticated /v1/models, and a
    401 travelling back through the tunnel would prove the tunnel is fine
    while reading as a failure -- so send the credential and treat only a
    transport error as unhealthy. `path` lets the layer-network block tunnel
    (commonnet.net.tunnel) use the same check against its own route; it has no token, so pass None.
    """
    headers = {"Authorization": f"Bearer {worker_token}"} if worker_token else {}
    try:
        http_json("GET", f"{tunnel_url}{path}", timeout=8, headers=headers)
        return True
    except urllib.error.HTTPError:
        return True  # reached the worker; a status code is not a dead tunnel
    except (urllib.error.URLError, socket.timeout):
        return False


# How long to wait for a quick tunnel to become reachable before registering.
# Quick tunnels take a moment to propagate; without this check, registration
# can succeed while the tunnel is still unreachable, causing the gateway to
# mark the node unhealthy immediately.
TUNNEL_WARMUP_TIMEOUT_SECONDS = 30


def wait_for_tunnel(tunnel_url: str, worker_token: str | None, path: str = "/v1/models") -> bool:
    """Wait until the tunnel is reachable from outside, up to ~30 seconds.

    Quick tunnels take a moment after cloudflared reports the URL before they
    are actually routable. Registering before this window passes causes the
    gateway to mark the node unhealthy immediately -- it can't reach us.

    Returns True if the tunnel became reachable, False if it timed out.
    """
    deadline = time.monotonic() + TUNNEL_WARMUP_TIMEOUT_SECONDS
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        if tunnel_is_healthy(tunnel_url, worker_token, path):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        wait = min(2.0, remaining)
        if attempt == 1:
            working(f"waiting for tunnel to become reachable (up to {TUNNEL_WARMUP_TIMEOUT_SECONDS}s)...")
        time.sleep(wait)
    return False


def reannounce_target(tunnel_url: str, registered_url: str | None) -> str | None:
    """The address the gateway should have for us, or None if it already has it.

    A serving node has two addresses and they can disagree: the one it is
    answering on, and the one the gateway has on file. Only the first is
    anything this process can check from here -- which is the whole reason a
    node can sit for hours looking perfectly fine locally while every request
    the network sends it goes nowhere. A health check cannot see this
    disagreement, because the tunnel really is healthy; what is stale is the
    gateway's copy of where it points.

    Both kinds of address land here, because both move for reasons the far end
    cannot observe: a quick tunnel rotates its hostname without the cloudflared
    process dying, and a DHCP lease renewal reassigns a LAN address.
    """
    wanted = f"{tunnel_url}/v1"
    return None if wanted == registered_url else wanted


def reannounce(gateway: str, payload: dict, node_token: str, wanted: str,
               attempts: int = REANNOUNCE_ATTEMPTS,
               backoff: float = REANNOUNCE_BACKOFF_SECONDS,
               poster=None) -> tuple[dict | None, str | None]:
    """Tell the gateway where we are now, retrying a bounded number of times.

    Returns `(node, None)` on success, `(None, detail)` once every attempt has
    failed -- and a failure here is deliberately not fatal. This is the one
    case in this file where giving up is right: the disagreement this exists to
    fix is still true after a failed attempt, so the caller retries on its next
    pass rather than dying.

    That shape is the fix for a real outage. The code here used to try exactly
    once and then forget, while mutating the payload as if it had succeeded --
    so after a rotation the node stayed registered against a hostname that no
    longer existed, answering every heartbeat, until it was restarted by hand.
    The retry is not a nicety: a restarted quick tunnel is NXDOMAIN for a while,
    and the first attempt lands squarely in that window, so one retry is often
    the difference between re-announcing and never re-announcing at all.

    `poster` is injectable so the retry can be exercised without a gateway.
    """
    post = poster or http_json
    # A copy: a failed attempt must leave `payload` -- and with it the address
    # the rest of the loop believes the gateway holds -- exactly as it was.
    body = dict(payload, endpoint_url=wanted)
    detail = None
    tries = max(1, attempts)
    for attempt in range(tries):
        try:
            node = post("POST", f"{gateway}/nodes", body=body,
                        headers=gateway_headers({"X-Common-Node-Token": node_token}),
                        timeout=REGISTRATION_TIMEOUT)
            return node, None
        except urllib.error.HTTPError as e:
            detail = f"{e.code} {e.read().decode(errors='ignore').strip()}"
        except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
            detail = str(getattr(e, "reason", e))
        if attempt + 1 < tries:
            time.sleep(min(backoff * (2 ** attempt), REANNOUNCE_MAX_BACKOFF_SECONDS))
    return None, detail


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


def worker_reachable_on_lan(ip: str) -> bool:
    """Can another machine on this LAN actually reach our worker?

    Registering an address nobody else can reach would put a node in the
    registry that health-checks green from its own machine and fails for
    everyone else -- the worst kind of broken, because the registry looks fine.

    A 401 counts as reachable: unauthenticated is exactly what this probe is,
    and the worker answering at all is the thing being tested. Only a
    transport failure means the address is no good -- usually client isolation
    on the wifi, or a local firewall.
    """
    try:
        http_json("GET", f"http://{ip}:{WORKER_PORT}/v1/models", timeout=5)
        return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, socket.timeout, OSError):
        return False


def print_cloudflared_tail(tail: list[str]) -> None:
    """What cloudflared last said, as commentary. Nothing, when it said nothing."""
    for line in tail:
        if line.strip():
            print(comment(line.strip()), file=sys.stderr)


def tunnel_failure_message(tail: list[str], target_url: str = WORKER_URL) -> tuple[str, list[str]]:
    """The headline and the advice for a tunnel that never came up.

    Split out from the reporting below so the decision it encodes -- rate limited,
    or something else -- can be tested without a cloudflared to fail. `tail` is
    what cloudflared last said, oldest first.
    """
    if any(TUNNEL_RATE_LIMITED_PATTERN.search(line) for line in tail):
        return (
            "Cloudflare refused to create a quick tunnel: rate limited (429, error 1015).",
            [
                "wait before trying again — Cloudflare's guidance is that retrying extends the block",
                "a named tunnel on a free Cloudflare account has no such limit",
            ],
        )
    if not any(line.strip() for line in tail):
        return (
            "cloudflared exited before a tunnel URL appeared, and printed nothing.",
            [f"run it by hand to see what it does: cloudflared tunnel --url {target_url}"],
        )
    return ("cloudflared exited before a tunnel URL appeared.", [])


def report_tunnel_failure(tail: list[str], target_url: str = WORKER_URL) -> None:
    """Say why the tunnel did not come up, then exit.

    The tail is printed whatever the classification, because cloudflared's own
    words are the evidence for the sentence printed above them. This is also the
    only place they can be printed: the reader thread in start_tunnel() is the one
    thing reading that pipe, and it keeps a bounded tail of it and nothing else.
    """
    headline, advice = tunnel_failure_message(tail, target_url)
    print(f"{GLYPH_FAILED} {red(headline)}", file=sys.stderr)
    for line in advice:
        print(comment(line), file=sys.stderr)
    print_cloudflared_tail(tail)
    sys.exit(1)


def start_tunnel(target_url: str = WORKER_URL,
                 host_header: str = f"127.0.0.1:{WORKER_PORT}") -> tuple[subprocess.Popen, str]:
    # By default this points at the worker, never at the model backend: the worker
    # is what requires the gateway's token and refuses everything but the two
    # inference routes. The layer-network block tunnel (commonnet.net.tunnel)
    # passes its own loopback address instead, and reuses this tunnel and its health check unchanged.
    #
    # Cloudflare's quick tunnel forwards the public tunnel hostname as the Host
    # header by default, and this pins it to 127.0.0.1 instead. That used to be
    # load-bearing -- Ollama refuses requests whose Host is not localhost, as
    # anti-DNS-rebinding. llama-server has no such check, so the flag is now
    # hygiene rather than a requirement: it keeps a hostname the outside world
    # chose out of the requests this machine's backend sees, and it costs
    # nothing. Using 127.0.0.1 instead of localhost avoids IPv4/IPv6 mismatches
    # on Windows.
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", target_url, "--http-host-header", host_header],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    found_url: list[str] = []
    tail: list[str] = []

    def read_output():
        for line in proc.stdout:
            line = line.rstrip("\n")
            tail.append(line)
            if len(tail) > TUNNEL_TAIL_LINES:
                del tail[0]
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
            # The pipe closes when the process exits, but this thread may not have
            # read the last of it yet -- and an exit's explanation is always in its
            # last lines. Joining is what makes the report below worth carrying at
            # all; the timeout keeps a pipe something else is holding open from
            # hanging a join that has already failed.
            reader.join(timeout=2)
            report_tunnel_failure(tail, target_url)
        time.sleep(0.5)

    stop_process(proc)
    # Stopped first, so the pipe closes and the reader can finish, then joined for
    # the same reason as above. A tunnel that never arrives is the case where
    # cloudflared's own output is most likely to be the whole story.
    reader.join(timeout=2)
    print(f"{GLYPH_FAILED} {red('cloudflared did not open a tunnel within 30s.')}", file=sys.stderr)
    print_cloudflared_tail(tail)
    sys.exit(1)


LAUNCHD_LABEL = "network.common.join"
SYSTEMD_UNIT = "common-join.service"
SCHTASKS_NAME = "CommonNetworkJoin"


def _service_argv(args: argparse.Namespace) -> list[str]:
    if getattr(args, "network", None) is not None:
        # A layer-network service: the spec goes last, as --network=SPEC, so it can never swallow another flag.
        argv = [sys.executable, os.path.abspath(__file__)]
        if args.no_update:
            argv += ["--no-update"]
        if getattr(args, "api", None) is not None:
            argv += [f"--api={args.api}"]
        return argv + ([f"--network={args.network}"] if args.network else ["--network"])
    argv = [
        sys.executable, os.path.abspath(__file__),
        "--legacy",  # recorded, so a service never changes mode when the default does
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
    # so cloudflared (installed there) won't resolve unless we carry over the
    # PATH from the shell that ran --permanent. The model runtime does not depend
    # on this: it is fetched into ~/.common-network/runtime and spawned by
    # absolute path, which is one of the reasons it moved there.
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
    log_path = Path.home() / ".common-network" / "join.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_str = " ".join(f'"{a}"' if " " in a else a for a in argv)
    cmd_str = f'cmd /c "{cmd_str} >> "{log_path}" 2>&1"'
    result = subprocess.run([
        "schtasks", "/create", "/f", "/sc", "onlogon", "/rl", "highest",
        "/tn", SCHTASKS_NAME, "/tr", cmd_str,
    ], capture_output=True, text=True)
    if result.returncode != 0:
        if "Access is denied" in result.stderr or "access" in result.stderr.lower():
            die("creating the scheduled task requires an elevated (admin) PowerShell.")
        die(f"schtasks failed: {result.stderr.strip()}")
    subprocess.run(["schtasks", "/run", "/tn", SCHTASKS_NAME], check=True)
    print(f"{GLYPH_DONE} {blue('installed as a scheduled task', bold=True)} — starts at login.")
    print(comment(f"logs: {log_path}"))
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


def _load_layers():
    """The layers module beside join.py (fetched once if this copy predates it), or None."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        return importlib.import_module("layers")
    except ImportError:
        pass
    try:
        with urllib.request.urlopen(SIBLING_UPDATE_URL.format("layers.py"), timeout=10) as resp:
            body = resp.read()
        with open(os.path.join(here, "layers.py"), "wb") as f:
            f.write(body)
        importlib.invalidate_caches()
        return importlib.import_module("layers")
    except (OSError, ImportError):
        return None


def layer_network_supported() -> bool:
    """Can this machine serve the layer network (Apple silicon, an NVIDIA GPU, or a CPU), with layers.py available?"""
    layers = _load_layers()
    if layers is None:
        return False
    try:
        layers.platform_kind()
    except Exception:
        return False
    return True


def choose_mode(args: argparse.Namespace, supported=None) -> str:
    """'network' or 'legacy' for this invocation.

    The layer network is the default for a plain `join`. The whole-model gateway node is kept as
    `--legacy`, and anything that only means something to it (--model, --lan, --list-catalogue)
    implies it, so existing scripts and installed services (which always carry --model) do not change.
    A machine the layer network cannot use (an OS other than macOS, Windows or Linux, or no layers.py) stays on the legacy path.
    """
    if args.network is not None:
        return "network"
    if args.legacy or args.model or args.lan or args.list_catalogue or args.remove_permanent:
        return "legacy"
    return "network" if (supported or layer_network_supported)() else "legacy"


LAYER_FIRST_NOTICE = ("`common join` now joins the decentralised layer network. "
                      "To run this machine as a whole-model gateway node like before, use `common join --legacy`.")


def announce_layer_first(marker: Path | None = None, printer=print) -> bool:
    """Say, once per machine, that the default changed. Returns True if it was printed."""
    marker = marker or (Path.home() / ".common-network" / "layer-first-notice")
    if marker.exists():
        return False
    printer(f"{GLYPH_WORK} {LAYER_FIRST_NOTICE}")
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("shown\n")
    except OSError:
        pass  # saying it again next time is harmless
    return True


def run_network_member(network: str | None, node_args: list[str] | None = None) -> None:
    """Join the decentralised layer network: set up a runtime, then serve a slice of the shared model.

    layers.py (beside this file) does the work; a copy of join.py that predates it fetches it
    here, so this path never changes how a plain `join.py` behaves.
    """
    layers = _load_layers()
    if layers is None:
        die_with_fix("layers.py is missing.",
                     f"join.py needs layers.py beside it to join the layer network. Fetch it from\n"
                     f"  https://github.com/{REPO}/tree/main/join")
    try:
        code = layers.run(network or None, node_args or [])
    except layers.LayersError as e:
        die(str(e))
    except KeyboardInterrupt:
        code = 0
    sys.exit(code)


def main() -> None:
    _force_utf8_output()
    sys.stdout.reconfigure(line_buffering=True)
    _enable_windows_ansi()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gateway", default=os.environ.get("COMMON_GATEWAY_URL", DEFAULT_GATEWAY), help="Gateway base URL (default: the shared Common Network gateway)")
    parser.add_argument("--model", default=None, help="Catalogue id (e.g. qwen2.5-coder-14b) or a raw model tag (e.g. mathstral:7b). Default: probe hardware and ask the network what it needs most (see --auto, --list-catalogue)")
    # The v0.2 RPC-group prototype was removed; the layer network (--network) replaced it. These stay
    # parseable, hidden, only so that a script or service that still passes one gets a clear message.
    for removed_flag in ("--distributed-config", "--group-token", "--relay"):
        parser.add_argument(removed_flag, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--local-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--network", nargs="?", const="", default=None, metavar="URL|FILE", help="Join the decentralised layer network: serve a slice of the shared model as a peer. Optionally a seed peer's URL or a manifest file (default: COMMONNET_NETWORK or the built-in seeds). No gateway registration")
    parser.add_argument("--no-api", action="store_true", help="With the layer network: do not serve the local chat API (on by default, so `common ask` and chat apps work). Saves the ~1 GB client part of the model")
    parser.add_argument("--api", nargs="?", const=11440, default=None, type=int, metavar="PORT", help="With --network: also serve an OpenAI-compatible chat API on 127.0.0.1:PORT (default 11440) from the same process, so a desktop app or any OpenAI-style client can chat with the network. The client part of the model (about 1 GB for Qwen3-8B) loads in the background")
    parser.add_argument("--legacy", action="store_true", help="Run as a whole-model gateway node (the original network) instead of the default layer network")
    parser.add_argument("--auto", action="store_true", help="Accept the network's recommended model automatically, no confirmation prompt")
    parser.add_argument("--list-catalogue", action="store_true", help="List catalogue models this machine can run, then exit")
    # The name and operator are published to the whole network (`GET /nodes`
    # is public by design — legibility), so the defaults deliberately avoid
    # the old pattern of hostname-username, which put a donor's OS username
    # in a public registry. The random suffix keeps imaged lab machines (which
    # often share a hostname) from colliding.
    # Default is None here; resolved below to either the persisted identity's
    # name (for stable node IDs across restarts) or a fresh hostname + suffix.
    parser.add_argument("--name", default=None, help="Node name, published to the network (default: reuse previous name, or common-node + nine random digits)")
    parser.add_argument("--operator", default="friend", help="Name shown publicly as this node's operator (default: 'friend')")
    parser.add_argument("--region", default=None, help="Optional region hint, e.g. au-adelaide")
    parser.add_argument("--cost", type=float, default=0, help="Declared cost per 1k tokens (default: 0, it's free)")
    parser.add_argument("--capability", default=None, help="Override the auto-generated capability description")
    parser.add_argument("--no-update", action="store_true", default=bool(os.environ.get("COMMON_NO_UPDATE")), help="Skip the self-update check (useful when hacking on this script locally)")
    parser.add_argument("--lan", action="store_true", help="Use a LAN worker address instead of a Cloudflare tunnel; still registers with --gateway (production by default).")
    parser.add_argument("--permanent", action="store_true", help="Install as a background service that starts at login/boot and restarts if it crashes, then exit")
    parser.add_argument("--remove-permanent", action="store_true", help="Remove the background service installed by --permanent, then exit")
    args = parser.parse_args()

    # Before updates, catalogue lookup, identity reads, tunnel creation or any
    # registration path. A local test must not contact the public gateway.
    removed = [flag for flag, given in (
        ("--distributed-config", args.distributed_config is not None), ("--local-only", args.local_only),
        ("--group-token", args.group_token is not None), ("--relay", args.relay is not None),
        ("--port", args.port is not None)) if given]
    if removed:
        parser.exit(2, f"removed: {', '.join(removed)} belonged to the v0.2 RPC-group prototype, which the "
                       "layer network replaced. Use `common join --network` (see docs/layer-network.md).\n")
    if args.network is not None and (args.model or args.lan or args.list_catalogue or args.legacy
                                     or args.remove_permanent):
        parser.error("--network cannot be combined with --model, --lan, --list-catalogue, --legacy, or --remove-permanent")
    explicit_legacy = bool(args.legacy or args.model or args.lan or args.list_catalogue or args.remove_permanent)
    mode = choose_mode(args)
    defaulted = mode == "network" and args.network is None
    fell_back = mode == "legacy" and not explicit_legacy  # plain join on a machine the layer network cannot use
    if defaulted:
        args.network = ""  # the default: the built-in network
    if args.api is not None and (mode != "network" or args.remove_permanent):
        parser.error("--api is only used with the layer network (--network, or a plain join on a machine that can serve it)")
    if mode == "network" and args.api is None and not args.no_api and not args.remove_permanent:
        args.api = 11440  # on by default: `common ask`, `common chat` and chat apps need this peer's local API

    if not args.no_update:
        # A node can be left running a join.py that expects a newer worker.py than
        # the one beside it, and every start after that fails at serve() with a
        # TypeError. Refreshing here, and reloading worker in this process, is what
        # lets such a node recover without waiting for join.py itself to change.
        if refresh_siblings():
            importlib.reload(worker)
        remote = fetch_update()
        if remote and refresh_siblings():
            print(f"{GLYPH_WORK} {dim('updating to the latest version...')}")
            apply_update_and_restart(remote)

    print(paper(BANNER, bold=True))

    if args.remove_permanent:
        remove_permanent()
        return

    if args.network is not None:
        if defaulted:
            announce_layer_first()
        if args.permanent:
            install_permanent(args)
            return
        run_network_member(args.network or None, [] if args.api is None else ["--api", str(args.api)])
        return
    if fell_back:
        die("this machine cannot join the Common network: it needs macOS, Windows or Linux with Python 3.10+ "
            "(Apple silicon and NVIDIA GPUs are fastest; any other machine serves on its CPU and needs "
            "about 5 GiB of free RAM). If you are on one of those, check your connection and try again. The old whole-model gateway is retired, so there is no other mode.")

    gateway = args.gateway.rstrip("/")

    # Resolve the node name: reuse the persisted identity's name for this
    # gateway if the user didn't explicitly provide one. This keeps the node ID
    # stable across restarts and retries, instead of creating dead duplicates.
    if args.name is None:
        identity = read_identity()
        if identity and identity.get("gateway") == gateway and identity.get("name"):
            args.name = identity["name"]
        else:
            args.name = default_node_name()

    if args.list_catalogue:
        hw = probe_hardware()
        print_catalogue(gateway, hw)
        return

    model_source, domain_tags, catalogue_id, capability_text_default = resolve_model(gateway, args)
    # Bake the resolution into args.model so --permanent's saved service
    # invocation re-resolves the same way on every restart -- the catalogue
    # id round-trips back through resolve_model()'s lookup branch, and a source
    # string re-resolves through its own branch without being rewritten.
    args.model = catalogue_id or model_source or ""

    if args.permanent:
        install_permanent(args)
        return

    # LAN mode skips cloudflared entirely, so don't demand it be installed.
    check_binaries(need_tunnel=not args.lan)

    spec = resolve_source(model_source)
    if spec is None:
        die(f"can't run '{model_source}' locally (unrecognised source).")

    # The single string that is both this node's public model name and the id the
    # worker answers with. gateway/app/health.py matches `GET /v1/models`'s id
    # against the `model_name` registered below, so these two uses must stay the
    # same string -- for an `ollama:` source it is the tag with the scheme
    # stripped, which is exactly what every existing node already registers.
    wire_name = spec.wire_name

    gguf = fetch_model(spec)
    llama_proc = start_llama_server(gguf, wire_name)
    if not wait_for_llama(llama_proc, gguf):
        stop_process(llama_proc)
        die_with_fix("the model runtime never started serving.",
                     "Run `common join` again — the weights are already on disk, so a\n"
                     "  second attempt starts much faster and usually clears it.")

    # Start the worker before anything can reach this machine. It holds the
    # only credential the gateway will be given for this node, serves exactly
    # two routes, and pins the node to the model resolved above -- so llama-server
    # itself never has to be reachable from outside, on either path.
    worker_token = secrets.token_urlsafe(24)
    try:
        # The sleep window goes in too, so the worker can keep the model
        # resident for as long as this node is serving rather than letting the
        # next user after a quiet half hour pay the reload. See _keep_warm in
        # worker.py; the same number is llama-server's --sleep-idle-seconds.
        worker.serve(worker_token, wire_name, port=WORKER_PORT, upstream_url=LLAMA_URL,
                     sleep_idle_seconds=sleep_idle_seconds(_require_runtime()))
    except OSError as e:
        stop_process(llama_proc)
        die_with_fix(
            f"couldn't start the Common worker on port {WORKER_PORT}: {e}",
            f"Something else is using port {WORKER_PORT}. Stop it, or if an older\n"
            f"  `common join` is still running, close that window first.",
        )
    print(f"{GLYPH_DONE} worker listening on :{WORKER_PORT} "
          f"{comment('(token-checked, inference only)')}")

    tunnel_proc = None
    if args.lan:
        ip = lan_ip()
        if not ip:
            die("couldn't work out this machine's LAN address — pass a tunnel join instead (drop --lan)")
        # What must be reachable is the worker, not the model backend -- which
        # binds 127.0.0.1 and stays there. That is both safer and one less thing
        # for a contributor to configure than the old arrangement, where the
        # LAN instructions told them to put Ollama on every interface.
        if not worker_reachable_on_lan(ip):
            die_with_fix(
                f"the Common worker is running, but not reachable from the rest of "
                f"the network on {ip}:{WORKER_PORT}.",
                "Usually one of two things:\n"
                "    · a local firewall is blocking incoming connections on this port\n"
                "    · the network has client isolation turned on, so machines on the\n"
                "      same wifi can't talk to each other — that needs IT to disable it\n"
                "      for this subnet.",
            )
        tunnel_url = f"http://{ip}:{WORKER_PORT}"
        print(f"{GLYPH_DONE} serving on the local network at {blue(tunnel_url)}")
        print(comment("no tunnel, nothing exposed to the internet — this machine is only "
                      "reachable from your own network."))
    else:
        working("opening a Cloudflare quick tunnel to this machine's Common worker...")
        tunnel_proc, tunnel_url = start_tunnel()
        print(f"{GLYPH_DONE} Cloudflare quick tunnel live at {blue(tunnel_url)}")

        if not wait_for_tunnel(tunnel_url, worker_token):
            print(f"{GLYPH_FORMING} {red('tunnel is not reachable after 30s')} — the gateway will likely mark this node unhealthy.")
            print(comment("quick tunnels can take longer to propagate on some networks. "
                          "the registration will proceed, but expect health check failures."))

    def stop_children() -> None:
        """Stop the two processes this run started: the tunnel and the model.

        Separate from deregistering, because it is also what the failure paths
        below need -- a registration that never succeeded has nothing to
        deregister, but it has still left a llama-server holding port 11436 and
        a tunnel holding a public hostname, and both of those must go with it.
        """
        stop_process(tunnel_proc)
        stop_process(llama_proc)

    capability_text = args.capability or capability_text_default or (
        f"{wire_name} running locally via llama.cpp, contributed by {args.operator}. Free, community-hosted."
    )

    payload = {
        "name": args.name,
        "operator": args.operator,
        "endpoint_url": f"{tunnel_url}/v1",
        "model_name": wire_name,
        "capability_text": capability_text,
        "region": args.region,
        "cost_per_1k": args.cost,
        "domain_tags": domain_tags,
        "catalogue_id": catalogue_id,
        # Which program is registering, for the gateway's own reporting
        # (`nodes.client`, migration 008). The gateway also accepts this as the
        # X-Common-Client header, which every request below carries; sending
        # both is what the desktop app does, so a row looks the same whichever
        # path registered it.
        "client": CLIENT,
        # The gateway must present this to reach the worker started above.
        # Generated here rather than issued by the gateway: the worker has to
        # be running before this endpoint is worth registering.
        "worker_token": worker_token,
    }

    working(f"registering '{args.name}' with {gateway}...")
    # If this machine already registered this name before (crashed without a
    # clean deregister, say), prove ownership so the gateway lets us take the
    # name back rather than 409-ing it as a stranger's.
    identity = read_identity() or {}
    rejoin_token = (identity.get("node_token")
                    if identity.get("name") == args.name and identity.get("gateway") == gateway
                    else None)
    # gateway_headers(), not a bare dict: registration is a request like
    # any other and must carry X-Common-Client, or the operators view
    # records this node's client as blank. Empty dict rather than None so
    # the helper has something to merge into (it always adds the header).
    reg_headers = gateway_headers(
        {"X-Common-Node-Token": rejoin_token} if rejoin_token else {}
    )

    # Registration retry logic for DNS propagation delays.
    # Fresh Cloudflare quick tunnels (*.trycloudflare.com) often NXDOMAIN
    # from the gateway's perspective for a short window after coming up.
    # Retrying with the *same* tunnel URL lets DNS catch up; killing the
    # tunnel on first failure and minting a new hostname on the next
    # `common-join` just restarts the clock.
    #
    # The gateway itself retries DNS resolution for up to ~30s before
    # returning 400, so registration requests can block for a while.
    # REGISTRATION_TIMEOUT (module constant) exceeds the gateway's DNS retry
    # window so the client doesn't abort mid-retry.
    RETRY_MAX_WAIT = 60  # seconds
    RETRY_INITIAL_BACKOFF = 2  # seconds
    RETRY_MAX_BACKOFF = 10  # seconds

    node = None
    retry_elapsed = 0.0
    retry_backoff = RETRY_INITIAL_BACKOFF

    while node is None:
        try:
            node = http_json("POST", f"{gateway}/nodes", body=payload, headers=reg_headers,
                             timeout=REGISTRATION_TIMEOUT)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="ignore")

            # DNS-related failure: the gateway couldn't resolve our hostname.
            # Keep the tunnel alive and retry -- the gateway is also retrying
            # on its side, but client-side retry with the same URL helps if
            # the gateway gave up before our DNS propagated.
            if e.code == 400 and "does not resolve" in detail:
                if retry_elapsed >= RETRY_MAX_WAIT:
                    stop_children()
                    die(f"registration failed: the gateway still cannot resolve our tunnel "
                        f"hostname after {RETRY_MAX_WAIT}s. This can happen when DNS "
                        f"propagation is unusually slow.\n"
                        f"  {comment('Try again in a minute; the same hostname may work then.')}")

                print(f"{GLYPH_FORMING} {dim(f'gateway cannot resolve our hostname yet — retrying in {retry_backoff}s (waited {retry_elapsed:.0f}s / {RETRY_MAX_WAIT}s)...')}")
                time.sleep(retry_backoff)
                retry_elapsed += retry_backoff
                retry_backoff = min(retry_backoff * 2, RETRY_MAX_BACKOFF)
                continue

            # Non-DNS failure: terminate immediately.
            stop_children()
            if e.code == 409:
                die(f"registration failed: a node named '{args.name}' is already registered and "
                    f"this machine has no token proving it's ours. Pick a different name "
                    f"(--name), or run `common leave` / DELETE the old node from its own machine first.")
            die(f"registration failed: {e.code} {detail}")

        except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
            # Timeout or network error during registration. The gateway may be
            # mid-DNS-retry (blocking), or there may be transient network issues.
            # Treat like a soft failure and retry with the same tunnel URL.
            if retry_elapsed >= RETRY_MAX_WAIT:
                stop_children()
                die(f"registration failed: request timed out or network error after "
                    f"{RETRY_MAX_WAIT}s of retries. The gateway may be slow to respond "
                    f"(DNS propagation) or unreachable.\n"
                    f"  {comment('Try again in a minute.')}")

            err_desc = "timed out" if isinstance(e, (socket.timeout, TimeoutError)) else str(e.reason if hasattr(e, 'reason') else e)
            print(f"{GLYPH_FORMING} {dim(f'registration {err_desc} — retrying in {retry_backoff}s (waited {retry_elapsed:.0f}s / {RETRY_MAX_WAIT}s)...')}")
            time.sleep(retry_backoff)
            retry_elapsed += retry_backoff
            retry_backoff = min(retry_backoff * 2, RETRY_MAX_BACKOFF)
            continue

    node_id = node["id"]
    node_token = node["node_token"]
    # What the gateway has just been told, remembered for the life of this
    # process. Every change to where this machine answers has to be reconciled
    # against it -- see `reannounce_target` -- and it only ever moves once the
    # gateway has confirmed the new address, never when a request to say so
    # failed.
    registered_url = payload["endpoint_url"]
    print(f"{GLYPH_DONE} {blue('you are live', bold=True)} · node id: {node_id}")
    print(comment("keep this window open to stay in the network — ctrl+c to leave."))

    write_identity(gateway, args.name, node_id, catalogue_id, domain_tags, node_token)

    def deregister_and_stop():
        try:
            req = urllib.request.Request(
                f"{gateway}/nodes/{node_id}", method="DELETE",
                headers=gateway_headers({"X-Common-Node-Token": node_token}),
            )
            urllib.request.urlopen(req, timeout=5)
        except urllib.error.URLError:
            pass
        clear_identity()
        stop_children()

    def cleanup(signum=None, frame=None):
        print(f"\n{GLYPH_WORK} {dim('leaving the network...')}")
        deregister_and_stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    last_update_check = time.monotonic()
    last_tunnel_check = time.monotonic()
    while True:
        if tunnel_proc is not None and tunnel_proc.poll() is not None:
            print(f"{GLYPH_FORMING} {red('tunnel dropped unexpectedly.')}")
            cleanup()

        # The model backend dying is quieter than the tunnel dying and worse:
        # the worker keeps answering, the gateway keeps routing here, and every
        # request comes back a 502 from a node that still looks healthy. Say so
        # and leave, rather than sit there collecting failures -- the LaunchAgent
        # or systemd unit restarts us, and a fresh start reloads the weights.
        if llama_proc.poll() is not None:
            print(f"{GLYPH_FORMING} {red('the model runtime exited unexpectedly')} "
                  f"— code {llama_proc.returncode}. Leaving the network; restart to rejoin.")
            cleanup()

        if time.monotonic() - last_tunnel_check > TUNNEL_HEALTH_CHECK_INTERVAL_SECONDS:
            last_tunnel_check = time.monotonic()

            # Where are we serving from now? Two things can move that, and
            # neither is visible from the far end.
            #
            # A LAN node has no tunnel to rotate or restart, but its address can
            # still change under it -- DHCP renewal on a school network reassigns
            # addresses routinely, and a node registered at a stale IP is a node
            # the gateway will mark unhealthy for reasons nobody can see.
            if tunnel_proc is None:
                current = lan_ip()
                if current and f"http://{current}:{WORKER_PORT}" != tunnel_url:
                    tunnel_url = f"http://{current}:{WORKER_PORT}"
            elif not tunnel_is_healthy(tunnel_url, worker_token):
                print(f"{GLYPH_FORMING} {red('tunnel is no longer reachable')} (quick tunnels can silently rotate hostnames) — restarting it...")
                stop_process(tunnel_proc)
                tunnel_proc, tunnel_url = start_tunnel()
                print(f"{GLYPH_DONE} tunnel live at {blue(tunnel_url)}")

            # Then tell the gateway, rather than only testing ourselves.
            # Restarting the tunnel is half the job: the gateway still holds the
            # old hostname, and the check above cannot notice that, because the
            # tunnel is fine -- it is the gateway's copy of our address that is
            # stale. So the re-announce is driven by the two addresses
            # disagreeing, not by a health check that cannot see the
            # disagreement at all.
            #
            # A failed attempt leaves them disagreeing on purpose, so the next
            # pass tries again; see `reannounce` for why that is the shape.
            wanted = reannounce_target(tunnel_url, registered_url)
            if wanted is not None:
                # The wording is the pre-existing wording for each case, kept
                # exactly: the desktop app reads these lines.
                if tunnel_proc is None:
                    print(f"{GLYPH_FORMING} this machine's address changed — re-registering at {blue(tunnel_url)}")
                node, detail = reannounce(gateway, payload, node_token, wanted)
                if node is None:
                    what = "on the new address" if tunnel_proc is None else "after tunnel restart"
                    print(f"{GLYPH_FAILED} {red(f'failed to re-register {what}: {detail}')}", file=sys.stderr)
                else:
                    node_id = node["id"]
                    node_token = node["node_token"]
                    payload["endpoint_url"] = wanted
                    registered_url = wanted
                    if tunnel_proc is not None:
                        print(f"{GLYPH_DONE} re-registered '{args.name}' with the new tunnel url.")

        if not args.no_update and time.monotonic() - last_update_check > UPDATE_CHECK_INTERVAL_SECONDS:
            last_update_check = time.monotonic()
            remote = fetch_update()
            if remote and refresh_siblings():
                print(f"\n{GLYPH_WORK} {dim('a new version is available — restarting to update...')}")
                deregister_and_stop()
                apply_update_and_restart(remote)

        time.sleep(2)


if __name__ == "__main__":
    main()
