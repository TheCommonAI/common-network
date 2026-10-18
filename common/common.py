#!/usr/bin/env python3
"""COMMON. — the commons, belonging to everyone and no one.

The Common Network Alpha (v0.1.13).

Copyright (C) 2026 Common AI Inc. Licensed under AGPL-3.0; see LICENSE at
https://github.com/TheCommonAI/common-network. This program comes with
ABSOLUTELY NO WARRANTY.

Usage:
    common                         open the interactive session
    common ask "<prompt>"          route one question through the network
    common join                    put this machine on the commons as a node
    common serve <model>           contribute a specific model
    common leave                   take this machine off the commons
    common status                  this node: health, position, requests served
    common demand                  live domain coverage gaps
    common recommend               what specialist the network needs next
    common recommend --machines 20 plan a whole computer lab at once
    common peers                   connected nodes and their coverage
    common peers --all             include machines that have left
    common contrib                 your contribution ledger
    common whoami                  your node identity
    common privacy                 what the network keeps, and what it doesn't
    common config                  settings (all local, all editable)
    common report [note]           report a bad answer from the network
    common reports                 reports saved on this machine, and send them
    common test                    benchmark every node + routing, log to jsonl
    common help [verb]             help, per verb

A question is answered by the best available donated machine — `common ask`
shows you which machine answered, and why it was chosen. Composition (several
specialists answering at once, combined into one reply) is built but **off by
default in Alpha**: the network is a donation platform first and does not claim
to compose a better answer until testing/compose-test has proven it against
live models. `--compose` opts a single request in; `-v` explains the routing
either way.

`common test` sweeps every healthy node with a fixed probe set, measures
time-to-first-token / total latency / approximate tok-s per node, checks
whether routing lands each domain on a node tagged for it, and writes every
measurement to ~/.common-network/tests/<run>.jsonl. Run it on each machine
and merge the files -- each records its own client-side view of the network.
Flags: --full (3 repeats), --repeats N, --nodes-only, --routing-only, --out.

`common test --thesis` is the experiment that settles whether the network
itself is better than its best single node. It runs four arms -- single,
panel, best-member, replication -- on ground-truth cases and reports whether
composition beats the best individual specialist. This is the test to run
on the school lab if the question is "does the thesis work".

The three composition-shaped cases are scored against a rubric of independent,
lane-tagged components rather than substring presence, so an answer that
combines two specialists scores above one that emits a single specialist's
part. Scoring executes the model-written code it is judging; `--no-exec`
skips that at the cost of resolution. The report's COMPOSITION CASES block
isolates those cases and breaks each arm down by lane.

"synth" and "map" are recognised but not yet built -- see `common help synth`
/ `common help map`. This CLI checks GitHub for a newer version of itself on
every run and updates in place (pass --no-update to skip). It never overwrites
a copy that lives in a git checkout -- a clone updates with `git pull`, and
clobbering someone's working tree is not this function's job. Pass
--force-update (or COMMON_FORCE_UPDATE=1) to override, which discards local
changes. An installed copy keeps a .bak of the version it replaced.

When an answer is wrong, `common report` says so. It attaches the question and
the answer you just had, plus the routing the gateway recorded (which node
answered, the topology, the timings) and the decision id, so an operator can
open the report against the exact request. `--reason` picks from a closed list
(made-up, wrong, off-topic, truncated, refused, format, slow, other); a note is
optional but one of the two is required. Nothing is sent until you have seen a
full preview naming the gateway -- that preview, and the confirmation, are the
whole of the privacy story: you are sending your own words to the gateway, whose
operator becomes a second reader of a question the answering machine could
already read. Reports are saved locally first and sent second, so a gateway that
is unreachable loses nothing; `common reports --retry` finishes the job, and
`common reports` lists what is on this machine.
"""
import argparse
import json
import os
import platform
import random
import re
import secrets
import shlex
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import concurrent.futures
import threading
try:
    import requests
    _REQUESTS_SESSION = requests.Session()
except Exception:
    requests = None
    _REQUESTS_SESSION = None

# Not available on Windows. Only used to cap CPU/memory when executing
# model-generated code during thesis scoring; absence degrades to a plain
# wall-clock timeout rather than disabling the check.
try:
    import resource as _resource
except ImportError:  # pragma: no cover - Windows
    _resource = None

# --- Identity ---------------------------------------------------------------
# join.py owns writing/clearing this file (it runs the actual registration).
# This CLI only reads it.
IDENTITY_PATH = Path.home() / ".common-network" / "identity.json"

VERSION = "0.1.13"
RELEASE = "The Common Network Alpha"
DEFAULT_GATEWAY = "https://gateway-production-b820.up.railway.app"
REPO = "TheCommonAI/common-network"
UPDATE_URL = f"https://raw.githubusercontent.com/{REPO}/main/common/common.py"
JOIN_SCRIPT_URL = f"https://raw.githubusercontent.com/{REPO}/main/join/join.py"
INSTALL_DIR = Path.home() / ".common-network"

# The model backend a machine serves from. join.py binds llama-server to this
# port on 127.0.0.1 (`--host 127.0.0.1`), and worker.py sits in front of it on
# WORKER_PORT (11435) and forwards here.
#
# 127.0.0.1 and not `localhost`, which is what this path used to say back when
# it spoke to Ollama. On Windows `localhost` resolves to ::1 first, and the
# backend binds IPv4 only -- so the name reaches a socket nobody is listening
# on. join.py and worker.py both spell the address out for that reason, and this
# is the third place it has to be got right.
#
# Written out again rather than imported from join/, deliberately: an installed
# copy has common.py, join.py and runtime.py flat in one directory, so an import
# would resolve on a user's machine -- but this repo keeps them in sibling
# folders, so the same import would fail in this file's own tests. Restating one
# integer is the smaller cost, and all three files that name it live here.
LLAMA_PORT = 11436
LLAMA_URL = f"http://127.0.0.1:{LLAMA_PORT}"

# Which program is speaking, reported to the gateway as X-Common-Client.
#
# The desktop app and this CLI share the gateway, the endpoints and the same
# ~/.common-network/identity.json on purpose -- one machine is one node however
# it joined -- so the program that spoke for a machine is otherwise invisible in
# the gateway's records. The gateway stores this string on registration
# (`nodes.client`, migration 008) so it can answer "how much of the network is
# the app, and how much is the terminal".
#
# The same two spellings common-desktop uses: the header on every gateway
# request, and the `client` field on POST /nodes (join.py sends that one, since
# it is the file that registers). Self-reported and trivially forgeable, so it
# is a usage statistic and never an authorisation input.
CLIENT = f"common-cli/{VERSION}"

# How long to wait for an answer. Must stay ABOVE the gateway's own read
# timeout (forward_read_timeout_seconds, 180s), because the gateway's failure
# carries which node died and why, while a socket timeout here can only say
# "timed out". The inner timeout has to be the one that fires. Cold starts are
# the reason the number is large at all: a 7B model on a donated laptop takes
# ~57s to load before it emits its first token.
#
# `common test` and gateway/tests/test_timeouts.py both read this name.
REQUEST_TIMEOUT_SECONDS = 240


def gateway_headers(extra: dict | None = None) -> dict:
    """Headers every gateway request carries -- the mirror of common-desktop's
    gatewayHeaders(), so an app user and a terminal user are told apart by one
    string in one place.

    Deliberately not applied to local requests (the model backend, or a node's
    own worker): those leave this machine and have no business carrying it.
    """
    return {"X-Common-Client": CLIENT, **(extra or {})}


# How long a node that has stopped answering stays on the peer list.
#
# Under this it reads as *down*: a laptop that shut its lid, a machine
# mid-restart, a lab that closes overnight. Worth showing, because it is
# coming back. Over it, the honest reading is that whoever lent the machine
# has moved on -- an afternoon in a school lab, a friend who tried it once --
# and listing them forever makes the network look larger than it is, which is
# the one thing a contribution ledger must never do.
#
# Hidden, not deleted: `common peers --all` still shows every row the gateway
# holds, and the row itself is the gateway's to keep or prune.
PEER_GRACE_SECONDS = 24 * 60 * 60

WORDMARK = r""" ██████  ██████  ███    ███ ███    ███  ██████  ███    ██
██      ██    ██ ████  ████ ████  ████ ██    ██ ████   ██
██      ██    ██ ██ ████ ██ ██ ████ ██ ██    ██ ██ ██  ██
██      ██    ██ ██  ██  ██ ██  ██  ██ ██    ██ ██  ██ ██
 ██████  ██████  ██      ██ ██      ██  ██████  ██   ████ ·"""

LOCKUP = "common."  # for tight spaces -- period rendered in blue by print_lockup()


# --- Palette / style ---------------------------------------------------------
# Exactly the four brand colours from the COMMON. design doc. No green, no
# purple, no gradients -- restraint is the point.
PALETTE = {
    "charcoal": (0x28, 0x26, 0x24),
    "paper":    (0xED, 0xE9, 0xE1),
    "dim":      (0x8A, 0x86, 0x81),
    "blue":     (0x92, 0xB4, 0xC8),
    "red":      (0xC8, 0x44, 0x2A),
}


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

    A Windows console renders the glyph vocabulary (← → ─ │ └) and the banner
    fine -- Python reaches it through the UTF-16 console API. A *redirected*
    stream does not: a pipe or a file gets the locale encoding, which is cp1252
    on most installs, and cp1252 cannot encode any of those characters. The
    result is UnicodeEncodeError and a dead process.

    This file is where that hurt most, because two of the things it runs on
    Windows read a subprocess's redirected output:

      - `common leave` runs join.py with capture_output=True and greps the
        result for "Removed". The child was killed by the UnicodeEncodeError
        printing the ✓, so the message was never written and the check could
        never pass.
      - `common join --permanent` installs a scheduled task that sends both
        streams to join.log, so the node died printing its own banner -- before
        it ever reached the model install.

    errors="replace" is deliberate rather than strict: a glyph is decoration,
    and decoration must never be able to stop the CLI from running. A stream
    that cannot be reconfigured at all (already replaced, closed, or not a
    TextIOWrapper) is left alone -- the banner is worth less than the command.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def _color_enabled() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR") and not _ARGS_NO_COLOR[0]


_ARGS_NO_COLOR = [False]  # set from --no-color in main(); NO_COLOR env already covered above


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


# Fixed glyph vocabulary -- see design doc 1.5. Consistency over cleverness.
GLYPH_WORK = dim("·")
GLYPH_ROUTE = blue("→")
GLYPH_RECV = dim("←")
GLYPH_DONE = blue("✓")
GLYPH_FORMING = red("⚠")
GLYPH_FAILED = red("✗")


def comment(text: str) -> str:
    return dim(f"  # {text}")


def print_wordmark() -> None:
    print(paper(WORDMARK, bold=True))


def print_banner_box(subtitle: str) -> None:
    lines = WORDMARK.splitlines()
    width = max(len(l) for l in lines) + 4
    top = dim("╭" + "─" * width + "╮")
    bottom = dim("╰" + "─" * width + "╯")
    print(top)
    print(dim("│") + " " * width + dim("│"))
    for line in lines:
        pad = width - len(line) - 2
        print(dim("│") + "  " + paper(line, bold=True) + " " * pad + dim("│"))
    print(dim("│") + " " * width + dim("│"))
    sub_pad = width - len(subtitle) - 2
    print(dim("│") + "  " + dim(subtitle) + " " * max(sub_pad, 0) + dim("│"))
    print(dim("│") + " " * width + dim("│"))
    print(bottom)


# --- HTTP --------------------------------------------------------------------

def http_json(method: str, url: str, body: dict | None = None, headers: dict | None = None, timeout: float = 20.0, *, client: bool = True):
    data = json.dumps(body).encode() if body is not None else None
    # Every caller of this is either a gateway request or a local model one,
    # so the client string goes on by default and the local calls opt out with
    # client=False. Stamping it here rather than at each of the dozen call
    # sites is what makes "every gateway request carries it" true by
    # construction instead of by remembering.
    headers = gateway_headers(headers) if client else dict(headers or {})

    # Use requests.Session for connection pooling when available; fall back to
    # urllib for streaming or when requests is absent. Map exceptions to the
    # urllib.error types callers already expect.
    if _REQUESTS_SESSION is not None:
        try:
            resp = _REQUESTS_SESSION.request(method, url, json=body, headers=headers, timeout=timeout)
            if resp.status_code >= 400:
                # Build an urllib.error.HTTPError to keep callers unchanged.
                from io import BytesIO
                fp = BytesIO(resp.content or b"")
                raise urllib.error.HTTPError(url, resp.status_code, resp.reason, resp.headers, fp)
            try:
                return resp.json()
            except json.JSONDecodeError:
                # Mirror the urllib behaviour: let the caller see the decode error.
                raise
        except requests.exceptions.RequestException as e:
            raise urllib.error.URLError(str(e))

    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# --- Self-update (same pattern as join.py/chat.py) ---------------------------

def _in_git_work_tree(path: str) -> bool:
    """Is this file inside a git working tree?

    Two copies of this file exist in the world and they need opposite treatment.
    An *installed* copy lives under ~/.common-network and nobody edits it: the
    update channel exists precisely to keep it current. A *checkout* is somebody's
    work in progress, and overwriting it destroys whatever they were doing.

    Asking git is what tells them apart, and it is the right question for both.
    Note that "has uncommitted changes" is NOT the right question: a contributor
    who commits an edit to a branch and then runs the CLI loses that edit from
    the working tree just the same, and `git status` would call the tree clean.

    False whenever git is absent or the answer is unclear -- so an installed
    copy, and any machine without git, keeps self-updating exactly as before.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", os.path.dirname(path), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def self_update(force: bool = False) -> None:
    try:
        with urllib.request.urlopen(UPDATE_URL, timeout=5) as resp:
            remote = resp.read()
    except urllib.error.HTTPError as e:
        # Being offline is normal and stays silent. A 401/403/404 is not
        # transient: the update channel itself is broken -- the repo is
        # private, was renamed, or the path moved -- and every machine running
        # this CLI is frozen at whatever version it installed.
        #
        # This used to be swallowed. HTTPError is a subclass of URLError, so
        # the single `except (URLError, socket.timeout): return` below caught
        # 404s too and returned silently, which made a permanently broken
        # update channel indistinguishable from a working one. That is exactly
        # how it went unnoticed on every installed copy at once.
        if e.code in (401, 403, 404):
            print(dim(f"note: updates unreachable ({e.code}) — running the installed version."),
                  file=sys.stderr)
            print(comment(f"source: {UPDATE_URL}"), file=sys.stderr)
        return
    except (urllib.error.URLError, socket.timeout):
        return  # offline — carry on quietly with the version already here
    if not remote.strip():
        return
    local_path = os.path.abspath(__file__)
    try:
        with open(local_path, "rb") as f:
            local = f.read()
    except OSError:
        return
    if remote == local:
        return

    # Never overwrite a checkout. This is the case that lost eight edits in one
    # run: a local copy that differed from upstream was replaced silently, with
    # no warning and no backup, the first time the CLI was run from the source
    # tree. A checkout is managed by git -- `git pull` is how it updates -- and
    # this function's job is to keep *deployed* copies current.
    if _in_git_work_tree(local_path) and not force:
        print(dim("note: this is a git checkout, not an installed copy — not "
                  "self-updating."), file=sys.stderr)
        print(comment("your working tree differs from the published version. "
                      "update it with `git pull`."), file=sys.stderr)
        print(comment("to update anyway (discarding local changes): "
                      "COMMON_FORCE_UPDATE=1"), file=sys.stderr)
        return

    print(dim("updating common to the latest version..."))
    # Keep the outgoing copy. The guard above covers the checkout case, but an
    # *installed* copy can still be hand-edited by the person who installed it,
    # and a deployment that silently eats a local fix is the same bug in a
    # different directory.
    backup_path = local_path + ".bak"
    try:
        with open(backup_path, "wb") as f:
            f.write(local)
    except OSError:
        pass  # a failed backup must not block the update
    try:
        with open(local_path, "wb") as f:
            f.write(remote)
    except OSError as e:
        print(dim(f"warning: couldn't self-update ({e}), continuing with current version"))
        return
    os.execv(sys.executable, [sys.executable, local_path] + sys.argv[1:])

# (rest of file unchanged...)
