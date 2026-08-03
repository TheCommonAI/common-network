#!/usr/bin/env python3
"""COMMON. — the commons, belonging to everyone and no one.

Usage:
    common                         open the interactive session
    common ask "<prompt>"          route one question through the network
    common join                    put this machine on the commons as a node
    common serve <model>           contribute a specific model
    common leave                   take this machine off the commons
    common status                  this node: health, position, requests served
    common demand                  live domain coverage gaps
    common peers                   connected nodes and their coverage
    common contrib                 your contribution ledger
    common whoami                  your node identity
    common config                  settings (all local, all editable)
    common test                    benchmark every node + routing, log to jsonl
    common help [verb]             help, per verb

`common test` sweeps every healthy node with a fixed probe set, measures
time-to-first-token / total latency / approximate tok-s per node, checks
whether routing lands each domain on a node tagged for it, and writes every
measurement to ~/.common-network/tests/<run>.jsonl. Run it on each machine
and merge the files -- each records its own client-side view of the network.
Flags: --full (3 repeats), --repeats N, --nodes-only, --routing-only, --out.

"synth" and "map" are recognised but not yet built -- see `common help synth`
/ `common help map`. This CLI checks GitHub for a newer version of itself on
every run and updates in place (pass --no-update to skip).
"""
import argparse
import json
import os
import platform
import random
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# --- Identity ---------------------------------------------------------------
# join.py owns writing/clearing this file (it runs the actual registration).
# This CLI only reads it.
IDENTITY_PATH = Path.home() / ".common-network" / "identity.json"

DEFAULT_GATEWAY = "https://gateway-production-b820.up.railway.app"
REPO = "robot-time/common-network"
UPDATE_URL = f"https://raw.githubusercontent.com/{REPO}/main/common/common.py"
JOIN_SCRIPT_URL = f"https://raw.githubusercontent.com/{REPO}/main/join/join.py"
INSTALL_DIR = Path.home() / ".common-network"

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

def http_json(method: str, url: str, body: dict | None = None, headers: dict | None = None, timeout: float = 20.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# --- Self-update (same pattern as join.py/chat.py) ---------------------------

def self_update() -> None:
    try:
        with urllib.request.urlopen(UPDATE_URL, timeout=5) as resp:
            remote = resp.read()
    except (urllib.error.URLError, socket.timeout):
        return
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
    print(dim("updating common to the latest version..."))
    try:
        with open(local_path, "wb") as f:
            f.write(remote)
    except OSError as e:
        print(dim(f"warning: couldn't self-update ({e}), continuing with current version"))
        return
    os.execv(sys.executable, [sys.executable, local_path] + sys.argv[1:])


# --- Identity ------------------------------------------------------------

def read_identity() -> dict | None:
    try:
        return json.loads(IDENTITY_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# --- Commands ------------------------------------------------------------

def cmd_ask(gateway: str, question: str, region: str | None, model: str | None,
            local: bool, as_json: bool, quiet: bool, verbose: bool) -> None:
    if local:
        _ask_local(question, model, as_json, quiet)
        return

    node_override = None
    if model:
        try:
            nodes = http_json("GET", f"{gateway}/nodes")
        except (urllib.error.URLError, socket.timeout) as e:
            print(red(f"✗ can't reach the network."), file=sys.stderr)
            print(comment(f"{e}"), file=sys.stderr)
            sys.exit(1)
        match = next((n for n in nodes if n["healthy"] and n["model_name"] == model), None)
        if not match:
            print(red(f"✗ no healthy node is currently serving '{model}'."), file=sys.stderr)
            print(comment("refusing rather than silently rerouting to a different model."), file=sys.stderr)
            print(dim("  → see what's available:   common peers"), file=sys.stderr)
            sys.exit(1)
        node_override = match["name"]

    if not quiet:
        print(dim("plotting request into vector space"))

    body = {"model": "auto", "messages": [{"role": "user", "content": question}], "stream": True}
    headers = {"Content-Type": "application/json"}
    if region:
        headers["X-Common-Region"] = region
    if node_override:
        headers["X-Common-Node"] = node_override

    req = urllib.request.Request(f"{gateway}/v1/chat/completions", data=json.dumps(body).encode(), headers=headers, method="POST")
    start = time.monotonic()
    try:
        resp = urllib.request.urlopen(req, timeout=180)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")
        print(red("✗ the network couldn't answer that."), file=sys.stderr)
        print(comment(f"{e.code}: {detail}"), file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(red("✗ can't reach the network."), file=sys.stderr)
        print(comment("you're offline, or no peers are up in your region."), file=sys.stderr)
        print(dim(f"  → retry:            common ask \"{question}\" --retry"), file=sys.stderr)
        print(dim("  → run local only:   common ask \"...\" --local"), file=sys.stderr)
        sys.exit(1)

    node_name = resp.headers.get("X-Common-Node")
    score = resp.headers.get("X-Common-Score")

    if not quiet and not as_json:
        score_str = f"   ·   {float(score):.2f} match" if score not in (None, "forced") else ""
        weak = score not in (None, "forced") and float(score) < 0.5
        marker = f"  {GLYPH_FORMING}" if weak else ""
        print(f"{GLYPH_ROUTE} nearest node   {blue(node_name or 'unknown')}{score_str}{marker}")
        if weak:
            print(comment("nothing on the commons serves this well yet."))
        print(GLYPH_RECV + dim(" streaming"))
        print()

    full = []
    with resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
            if delta:
                if as_json:
                    full.append(delta)
                else:
                    print(paper(delta), end="", flush=True)
                    full.append(delta)
    if not as_json:
        print()

    latency_ms = int((time.monotonic() - start) * 1000)

    if as_json:
        print(json.dumps({
            "answer": "".join(full), "node": node_name, "score": score,
            "latency_ms": latency_ms,
        }))
        return

    if not quiet:
        print()
        print(dim("─" * 63))
        model_bit = f" ({model})" if model else ""
        print(dim(f"served by   {node_name}{model_bit}"))
        retention = "embedding retained for demand analytics · no raw text stored"
        print(dim(f"routed in   {latency_ms}ms   ·   {retention}   ·   no one owns this"))
        if verbose:
            print(dim(f"  score: {score}"))


def _ask_local(question: str, model: str | None, as_json: bool, quiet: bool) -> None:
    ollama_url = "http://localhost:11434"
    try:
        tags = http_json("GET", f"{ollama_url}/api/tags", timeout=5)
    except (urllib.error.URLError, socket.timeout):
        print(red("✗ can't reach Ollama on this machine."), file=sys.stderr)
        print(comment("is it installed and running? https://ollama.com/download"), file=sys.stderr)
        sys.exit(1)
    names = [m["name"] for m in tags.get("models", [])]
    if not names:
        print(red("✗ no local models available. pull one first: ollama pull llama3.2:3b"), file=sys.stderr)
        sys.exit(1)
    chosen = model or names[0]
    if not quiet:
        print(dim(f"asking {chosen} locally (never leaves this machine)"))
    body = {"model": chosen, "messages": [{"role": "user", "content": question}], "stream": False}
    try:
        result = http_json("POST", f"{ollama_url}/v1/chat/completions", body=body, timeout=180)
    except (urllib.error.URLError, socket.timeout) as e:
        print(red(f"✗ local request failed: {e}"), file=sys.stderr)
        sys.exit(1)
    answer = result["choices"][0]["message"]["content"]
    if as_json:
        print(json.dumps({"answer": answer, "node": "local", "model": chosen}))
    else:
        print(paper(answer))
        print()
        print(dim("─" * 63))
        print(dim(f"served by   local ({chosen})   ·   never left this machine"))


def cmd_peers(gateway: str, as_json: bool) -> None:
    nodes = http_json("GET", f"{gateway}/nodes")
    if as_json:
        print(json.dumps(nodes))
        return
    if not nodes:
        print(dim("no peers reachable yet."))
        print(dim("  → common join"))
        return
    healthy = sum(1 for n in nodes if n["healthy"])
    print(dim(f"{len(nodes)} peer(s)   ·   {healthy} healthy\n"))
    for n in nodes:
        badge = GLYPH_DONE if n["healthy"] else GLYPH_FAILED
        tags = ", ".join(n.get("domain_tags") or []) or "untagged"
        print(f"{badge}  {paper(n['name'])}")
        print(dim(f"   {n['model_name']}   ·   {tags}   ·   {n['avg_latency_ms']}ms avg"))


def cmd_demand(gateway: str, as_json: bool) -> None:
    nodes = http_json("GET", f"{gateway}/nodes")
    decisions = http_json("GET", f"{gateway}/decisions/recent?limit=200")

    demand: dict[str, int] = {}
    for d in decisions:
        if d.get("matched_domain"):
            demand[d["matched_domain"]] = demand.get(d["matched_domain"], 0) + 1
    coverage: dict[str, int] = {}
    for n in nodes:
        if n["healthy"] and n.get("domain_tags"):
            for t in n["domain_tags"]:
                coverage[t] = coverage.get(t, 0) + 1

    domains = sorted(set(demand) | set(coverage))
    if as_json:
        print(json.dumps({d: {"demand": demand.get(d, 0), "coverage": coverage.get(d, 0)} for d in domains}))
        return

    if not domains:
        print(dim("no domain-matched requests yet. the network hasn't seen enough traffic to show gaps."))
        return

    print(dim("domain coverage   ·   live\n"))
    max_val = max([1] + list(demand.values()) + list(coverage.values()))
    for d in domains:
        dv, cv = demand.get(d, 0), coverage.get(d, 0)
        bar_len = 20
        filled = int((dv / max_val) * bar_len) if max_val else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        status = f"{GLYPH_DONE} ok" if cv > 0 else f"{GLYPH_FORMING} forming"
        print(f"  {d:28s} {paper(bar)}   {status}")
        print(dim(f"    {dv} request(s)   ·   {cv} node(s) serving it\n"))

    forming = [d for d in domains if coverage.get(d, 0) == 0 and demand.get(d, 0) > 0]
    if forming:
        print(f"a cloud is forming in {red(forming[0])}.")
        print(comment("no node serves it well yet. join a machine into this gap:"))
        print(dim(f"  → common join --auto"))


def cmd_status(gateway: str, as_json: bool) -> None:
    identity = read_identity()
    if not identity:
        if as_json:
            print(json.dumps({"joined": False}))
            return
        print(dim("not currently on the commons."))
        print(dim("  → common join"))
        return

    try:
        nodes = http_json("GET", f"{gateway}/nodes")
    except (urllib.error.URLError, socket.timeout):
        nodes = []
    node = next((n for n in nodes if n["id"] == identity.get("node_id")), None)

    if as_json:
        print(json.dumps({"joined": True, "identity": identity, "node": node}))
        return

    since = time.time() - identity.get("joined_at", time.time())
    days = since / 86400
    print(f"node  {paper(identity['name'], bold=True)}   ·   on the commons {days:.1f} days\n")
    if node:
        badge = f"{GLYPH_DONE} healthy" if node["healthy"] else f"{GLYPH_FAILED} unhealthy"
        print(dim(f"status          {badge}"))
        print(dim(f"model           {node['model_name']}"))
        print(dim(f"avg latency     {node['avg_latency_ms']}ms"))
        print(dim(f"domain tags     {', '.join(node.get('domain_tags') or []) or 'untagged'}"))
    else:
        print(red("this node is no longer registered (deregistered or replaced)."))


def cmd_whoami(as_json: bool) -> None:
    identity = read_identity()
    if not identity:
        if as_json:
            print(json.dumps(None))
            return
        print(dim("no identity yet. join the network first:"))
        print(dim("  → common join"))
        return
    if as_json:
        print(json.dumps(identity))
        return
    print(f"node      {paper(identity['name'], bold=True)}")
    print(dim(f"gateway   {identity['gateway']}"))
    print(dim(f"node id   {identity['node_id']}"))
    print()
    print(comment("this is a name-based identity, not a cryptographic keypair yet."))
    print(comment("no account, nothing to log into -- see TODO(v0.4) in the design doc."))


def cmd_contrib(gateway: str, as_json: bool) -> None:
    identity = read_identity()
    if not identity:
        print(dim("not currently on the commons."))
        print(dim("  → common join"))
        return
    decisions = http_json("GET", f"{gateway}/decisions/recent?limit=500")
    served = [d for d in decisions if d.get("chosen_node") == identity.get("node_id")]
    ok = sum(1 for d in served if d.get("ok"))

    if as_json:
        print(json.dumps({"requests_served": len(served), "ok": ok}))
        return

    since = time.time() - identity.get("joined_at", time.time())
    days = max(since / 86400, 0.01)
    print(f"node  {paper(identity['name'], bold=True)}   ·   on the commons {days:.1f} days\n")
    print(dim(f"requests served (last 500 logged)   {len(served)}"))
    print(dim(f"successful                          {ok}/{len(served)}" if served else dim("successful                          —")))
    print()
    print(comment(f"{len(served)} questions got answered because you left your gate open."))


def cmd_config(as_json: bool, args: argparse.Namespace) -> None:
    cfg = {
        "gateway": args.gateway,
        "region": args.region or None,
        "no_color": bool(os.environ.get("NO_COLOR")),
    }
    if as_json:
        print(json.dumps(cfg))
        return
    print(dim("gateway   ") + paper(cfg["gateway"]))
    print(dim("region    ") + paper(str(cfg["region"] or "(none set)")))
    print()
    print(dim("what the network retains, in words:"))
    print(comment("every request's embedding + which node answered + latency, for"))
    print(comment("demand analytics (see `common demand`). no raw question text,"))
    print(comment("no account, no telemetry beyond that. nothing sent home beyond"))
    print(comment("what's needed to route your request and log that decision."))


def cmd_leave(gateway: str) -> None:
    identity = read_identity()
    join_py = INSTALL_DIR / "join.py"
    if join_py.exists():
        import subprocess
        print(dim("checking for a running background node service..."))
        result = subprocess.run([sys.executable, str(join_py), "--no-update", "--remove-permanent"], capture_output=True, text=True)
        if result.returncode == 0 and "Removed" in result.stdout:
            print(f"{GLYPH_DONE} done. you're off the network.")
            print(comment("nothing was kept. your keypair stays on your machine."))
            return
    if identity:
        print(dim("no background service found, and no foreground session this command can reach."))
        print(comment(f"if `common join` is running in another terminal, press Ctrl+C there to leave."))
    else:
        print(dim("not currently on the commons."))


def cmd_join_or_serve(verb: str, gateway: str, args: argparse.Namespace, extra_model: str | None = None) -> None:
    join_py = INSTALL_DIR / "join.py"
    # Always fetch the current version before delegating -- a stale local
    # copy's own self-update runs *after* argparse, so a new flag this CLI
    # relies on (e.g. --auto) would crash before join.py ever got the chance
    # to update itself. Found exactly this bug in testing.
    try:
        with urllib.request.urlopen(JOIN_SCRIPT_URL, timeout=10) as resp:
            remote = resp.read()
        join_py.parent.mkdir(parents=True, exist_ok=True)
        join_py.write_bytes(remote)
    except (urllib.error.URLError, socket.timeout) as e:
        if not join_py.exists():
            print(red(f"✗ couldn't fetch the join script: {e}"), file=sys.stderr)
            sys.exit(1)
        # Offline but we already have a copy -- use it as-is.

    if verb == "serve":
        print(dim(f"putting {extra_model} out to graze on the commons.\n"))

    argv = [sys.executable, str(join_py), "--gateway", gateway]
    if extra_model:
        argv += ["--model", extra_model, "--auto"]
    elif args.auto:
        argv += ["--auto"]
    if args.model and not extra_model:
        argv += ["--model", args.model]
    if args.region:
        argv += ["--region", args.region]

    os.execv(sys.executable, argv)


# --- Test --------------------------------------------------------------------
# `common test` is a *network* measurement, not a quality benchmark -- bench/
# owns scoring against real datasets (GSM8K/HumanEval/MMLU). This answers the
# three questions you need answered while standing in a room with several
# machines: is every node actually reachable, how fast is each one, and does
# routing land each domain on a node that claims that domain.
#
# Stdlib only, like the rest of this file, so it runs anywhere `common` does.
# Every machine can run it independently -- results carry the client identity,
# so merging the JSONL from each PC gives you each one's view of the network.

TEST_DIR = INSTALL_DIR / "tests"

# Short, deterministic, single-token-ish answers. `expect` is a smoke check
# (substring, case-insensitive), NOT a benchmark score -- a probe marked
# incorrect here means "look at this node", not "this model scores X".
PROBES = [
    {"id": "math-1",    "domain": "math",    "expect": "391",
     "prompt": "What is 17 * 23? Reply with only the number."},
    {"id": "math-2",    "domain": "math",    "expect": "5",
     "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How many cents does the ball cost? Reply with only the number."},
    {"id": "code-1",    "domain": "code",    "expect": "def is_even",
     "prompt": "Write a Python function is_even(n) that returns True when n is even. Reply with only code."},
    {"id": "code-2",    "domain": "code",    "expect": "3",
     "prompt": "In Python, what does len([1, 2, 3]) return? Reply with only the number."},
    {"id": "general-1", "domain": "general", "expect": "canberra",
     "prompt": "What is the capital of Australia? Reply with only the city name."},
    {"id": "general-2", "domain": "general", "expect": "orwell",
     "prompt": "Who wrote the novel 1984? Reply with only the author's surname."},
    {"id": "reason-1",  "domain": "general", "expect": "1",
     "prompt": "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have? Reply with only the number."},
    {"id": "legal-1",   "domain": "legal",   "expect": None,
     "prompt": "In one sentence, what is the difference between civil law and criminal law?"},
]


def _pct(values: list[int], p: int) -> int | None:
    """Linear-interpolated percentile. Report p50/p90, never means -- latency
    distributions here are long-tailed and a mean hides exactly the cold-start
    behaviour we care about."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return int(round(s[lo] + (s[hi] - s[lo]) * (k - lo)))


def _fmt_ms(ms: int | None) -> str:
    if ms is None:
        return "—"
    return f"{ms}ms" if ms < 1000 else f"{ms / 1000:.1f}s"


def _link_probe(endpoint_url: str, attempts: int = 3, timeout: float = 15.0) -> int | None:
    """Round-trip to a node's own public endpoint, doing no inference.

    This is the measurement that separates "this machine's model is slow" from
    "the link to this machine is slow" -- GET /models loads nothing and
    generates nothing, so what's left is tunnel + that operator's uplink.
    Min of N, because we want the floor of the link, not its worst sample.

    Caveat worth keeping in mind when reading the numbers: this is measured
    client->node, whereas real traffic goes gateway->node. It is a proxy for
    that leg, not the same path.
    """
    url = endpoint_url.rstrip("/") + "/models"
    best: int | None = None
    for _ in range(attempts):
        t = time.monotonic()
        try:
            with urllib.request.urlopen(url, timeout=timeout):
                pass
        except urllib.error.HTTPError:
            pass  # any HTTP answer still proves the round trip -- time it
        except (urllib.error.URLError, socket.timeout, OSError):
            continue
        ms = int((time.monotonic() - t) * 1000)
        best = ms if best is None else min(best, ms)
    return best


def _probe_once(gateway: str, prompt: str, node: str | None = None, timeout: float = 180.0,
                direct_url: str | None = None, direct_model: str | None = None) -> dict:
    """One streamed request. Always returns a measurement dict, never raises.

    Streams deliberately: with stream=False the only timing you get is total
    duration, which mostly measures how long the answer was. Time-to-first-token
    is the number that actually reflects routing + network + model load.

    `direct_url` bypasses the gateway and talks to the node's endpoint straight
    -- same prompt, same node, one hop fewer. The difference between the two
    is what the gateway costs (embed + score + proxy).
    """
    if direct_url:
        url = direct_url.rstrip("/") + "/chat/completions"
        model = direct_model or "auto"
    else:
        url = f"{gateway}/v1/chat/completions"
        model = "auto"

    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True}
    headers = {"Content-Type": "application/json"}
    if node and not direct_url:
        # Forcing a node also disables the gateway's fallback (see gateway.py),
        # which is what we want -- a failure here must surface as that node's
        # failure, not get silently masked by the runner-up.
        headers["X-Common-Node"] = node

    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    start = time.monotonic()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"http {e.code}: {e.read().decode(errors='ignore')[:200]}",
                "total_ms": int((time.monotonic() - start) * 1000)}
    except (urllib.error.URLError, socket.timeout) as e:
        return {"ok": False, "error": f"unreachable: {e}",
                "total_ms": int((time.monotonic() - start) * 1000)}

    served_by = resp.headers.get("X-Common-Node")
    score = resp.headers.get("X-Common-Score")
    ttft_ms: int | None = None
    chunks = 0
    text: list[str] = []
    gaps: list[int] = []   # inter-token arrival gaps -- the jitter signature
    last_tok: float | None = None

    try:
        with resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
                if delta:
                    now = time.monotonic()
                    if ttft_ms is None:
                        ttft_ms = int((now - start) * 1000)
                    else:
                        # A steady p50 with a fat p90/max means the link is
                        # stalling, not the model -- a model generates at a
                        # roughly constant rate, a congested uplink does not.
                        gaps.append(int((now - last_tok) * 1000))
                    last_tok = now
                    chunks += 1
                    text.append(delta)
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
        return {"ok": False, "served_by": served_by, "error": f"stream broke: {e}",
                "total_ms": int((time.monotonic() - start) * 1000), "ttft_ms": ttft_ms}

    total_ms = int((time.monotonic() - start) * 1000)
    answer = "".join(text)
    gen_ms = (total_ms - ttft_ms) if ttft_ms is not None else 0
    return {
        "ok": bool(answer),
        "served_by": served_by,
        "score": score,
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        # One SSE content delta is ~1 token for Ollama and OpenAI-compatible
        # servers, but it is an approximation -- the gateway does not yet
        # forward usage counts (TODO: add tokens_in/tokens_out to decisions).
        "out_tokens_approx": chunks,
        "tok_s_approx": round(chunks / (gen_ms / 1000.0), 2) if gen_ms > 0 else None,
        "gap_p50_ms": _pct(gaps, 50),
        "gap_p90_ms": _pct(gaps, 90),
        "gap_max_ms": max(gaps) if gaps else None,
        "answer": answer,
        "error": None if answer else "empty response",
    }


def _probe_correct(expect: str | None, answer: str) -> bool | None:
    if not expect:
        return None  # unscored probe (open-ended) -- still measured for latency
    return expect.lower() in (answer or "").lower()


def cmd_test(gateway: str, args: argparse.Namespace) -> None:
    repeats = 3 if args.full else max(1, args.repeats)
    started = time.time()
    host = socket.gethostname()
    run_id = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(started))}-{host}"

    try:
        nodes = http_json("GET", f"{gateway}/nodes")
    except (urllib.error.URLError, socket.timeout) as e:
        print(red("✗ can't reach the network."), file=sys.stderr)
        print(comment(f"{e}"), file=sys.stderr)
        sys.exit(1)

    healthy = [n for n in nodes if n["healthy"]]
    if not healthy:
        print(red("✗ no healthy nodes to test."), file=sys.stderr)
        print(dim("  → put this machine on the commons:   common join"), file=sys.stderr)
        sys.exit(1)

    tags_by_name = {n["name"]: (n.get("domain_tags") or []) for n in nodes}

    # Gateway round-trip, measured before any inference. Everything else in
    # this run sits on top of this number, so it has to be recorded separately
    # or you cannot tell gateway overhead from node slowness.
    print(f"{GLYPH_WORK} {dim('measuring gateway round-trip')}")
    rtts = []
    for _ in range(5):
        t = time.monotonic()
        try:
            http_json("GET", f"{gateway}/health", timeout=10)
            rtts.append(int((time.monotonic() - t) * 1000))
        except (urllib.error.URLError, socket.timeout):
            pass
    rtt = min(rtts) if rtts else None
    print(dim(f"  gateway rtt   {_fmt_ms(rtt)}   ·   {len(healthy)} healthy node(s)   ·   "
              f"{len(PROBES)} probe(s) × {repeats} repeat(s)"))
    print()

    # Per-node link measurement, before any inference. Two numbers per node:
    #   link_rtt_ms    -- reach the node, load nothing, generate nothing.
    #                     Pure tunnel + that operator's uplink.
    #   direct_ttft_ms -- same node, same trivial prompt, gateway bypassed.
    # Together with the gateway-routed TTFT these separate the three things
    # that a single latency number otherwise smears into one: the network to
    # that device, the model on that device, and the gateway in between.
    links: dict[str, dict] = {}
    if not args.routing_only:
        print(f"{GLYPH_WORK} {dim('measuring the link to each node (no inference)')}")
        for n in healthy:
            link_rtt = _link_probe(n["endpoint_url"])
            direct = _probe_once(gateway, "Reply with exactly: OK", timeout=120,
                                 direct_url=n["endpoint_url"], direct_model=n["model_name"])
            links[n["name"]] = {
                "link_rtt_ms": link_rtt,
                "direct_ttft_ms": direct.get("ttft_ms") if direct.get("ok") else None,
                "direct_ok": bool(direct.get("ok")),
                "direct_error": direct.get("error"),
            }
            note = "" if direct.get("ok") else red("  (direct probe failed)")
            print(dim(f"   {n['name']:<30} link {_fmt_ms(link_rtt):>7}   "
                      f"direct ttft {_fmt_ms(direct.get('ttft_ms')):>7}") + note)
        print()

    out_path = Path(args.out) if args.out else TEST_DIR / f"{run_id}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("w")

    def emit(rec: dict) -> None:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()  # flush per record so a Ctrl+C mid-run still leaves usable data

    emit({
        "type": "manifest", "run_id": run_id, "started_at": started, "gateway": gateway,
        "gateway_rtt_ms": rtt, "repeats": repeats, "links": links,
        "client": {"host": host, "os": platform.system(), "release": platform.release(),
                   "machine": platform.machine(), "python": platform.python_version()},
        "nodes": [{"name": n["name"], "model": n["model_name"], "operator": n.get("operator"),
                   "region": n.get("region"), "domain_tags": n.get("domain_tags"),
                   "healthy": n["healthy"], "avg_latency_ms": n.get("avg_latency_ms")} for n in nodes],
    })

    # Build the full work list up front, then shuffle it. Running all of node A
    # then all of node B confounds the comparison with time-of-day: home uplinks
    # are not the same at 4pm as at 9pm. Interleaving costs nothing and removes it.
    work: list[tuple[str, str | None, dict, int]] = []
    if not args.routing_only:
        for node in healthy:
            for probe in PROBES:
                for r in range(repeats):
                    work.append(("node", node["name"], probe, r))
    if not args.nodes_only:
        for probe in PROBES:
            for r in range(repeats):
                work.append(("routing", None, probe, r))
    random.Random(1234).shuffle(work)  # fixed seed: same interleaving on every machine

    results: list[dict] = []
    tty = sys.stdout.isatty()
    for i, (phase, node_name, probe, rep) in enumerate(work, 1):
        if tty:
            label = node_name or "auto-route"
            status = f"{i}/{len(work)}  {label}  {probe['id']}"
            print("\r" + GLYPH_WORK + " " + dim(f"{status:<66}"), end="", flush=True)
        m = _probe_once(gateway, probe["prompt"], node=node_name)
        rec = {
            "type": "probe", "run_id": run_id, "phase": phase, "at": time.time(),
            "forced_node": node_name, "probe_id": probe["id"], "domain": probe["domain"],
            "repeat": rep, "correct": _probe_correct(probe["expect"], m.get("answer", "")),
            "routed_tags": tags_by_name.get(m.get("served_by") or "", []),
            **{k: v for k, v in m.items() if k != "answer"},
            "answer": (m.get("answer") or "")[:600],
        }
        results.append(rec)
        emit(rec)
    if tty:
        print("\r" + " " * 78 + "\r", end="")

    fh.close()

    # --- Node sweep summary ---
    if not args.routing_only:
        print(f"{GLYPH_ROUTE} {paper('node sweep', bold=True)}")
        for node in healthy:
            rs = [r for r in results if r["phase"] == "node" and r["forced_node"] == node["name"]]
            if not rs:
                continue
            ok = [r for r in rs if r["ok"]]
            ttfts = [r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None]
            totals = [r["total_ms"] for r in ok if r.get("total_ms") is not None]
            toks = [r["tok_s_approx"] for r in ok if r.get("tok_s_approx")]
            scored = [r for r in rs if r["correct"] is not None]
            n_correct = sum(1 for r in scored if r["correct"])
            badge = GLYPH_DONE if len(ok) == len(rs) else (GLYPH_FORMING if ok else GLYPH_FAILED)
            print(f"{badge}  {paper(node['name'])}   {dim(node['model_name'])}")
            print(dim(f"   {len(ok)}/{len(rs)} ok   ·   ttft p50 {_fmt_ms(_pct(ttfts, 50))} / p90 {_fmt_ms(_pct(ttfts, 90))}"
                      f"   ·   total p50 {_fmt_ms(_pct(totals, 50))}"
                      f"   ·   ~{round(sum(toks) / len(toks), 1) if toks else '—'} tok/s"
                      f"   ·   {n_correct}/{len(scored)} correct"))

            # Split the observed TTFT into the three things it actually
            # contains, so a slow node can be blamed on the right layer.
            link = links.get(node["name"], {})
            link_rtt, direct_ttft = link.get("link_rtt_ms"), link.get("direct_ttft_ms")
            ttft_p50 = _pct(ttfts, 50)
            parts = [f"link {_fmt_ms(link_rtt)}"]
            if direct_ttft is not None and link_rtt is not None:
                parts.append(f"model ~{_fmt_ms(max(direct_ttft - link_rtt, 0))}")
            if direct_ttft is not None and ttft_p50 is not None:
                parts.append(f"gateway ~{_fmt_ms(max(ttft_p50 - direct_ttft, 0))}")
            print(dim(f"   where the time goes:   {'   ·   '.join(parts)}"))

            g50 = [r["gap_p50_ms"] for r in ok if r.get("gap_p50_ms") is not None]
            g90 = [r["gap_p90_ms"] for r in ok if r.get("gap_p90_ms") is not None]
            gmx = [r["gap_max_ms"] for r in ok if r.get("gap_max_ms") is not None]
            if g50:
                stall = max(gmx) if gmx else 0
                # max(..., 1) so a sub-millisecond p50 can't make every run
                # look like it stalled.
                jitter_note = red("   ← stalling, look at the link") if stall > 10 * max(_pct(g50, 50), 1) else ""
                print(dim(f"   token gaps:   p50 {_fmt_ms(_pct(g50, 50))}   ·   "
                          f"p90 {_fmt_ms(_pct(g90, 90))}   ·   max {_fmt_ms(stall)}") + jitter_note)

            for r in rs:
                if not r["ok"]:
                    print(comment(f"{r['probe_id']} failed: {r.get('error')}"))
        print()
        print(comment("link = client→node, no inference (a proxy for the gateway→node leg,"))
        print(comment("not the same path). model = direct-to-node ttft minus link."))
        print(comment("gateway = routed ttft minus direct ttft: embedding, scoring, proxying."))
        print()

    # --- Routing summary ---
    if not args.nodes_only:
        print(f"{GLYPH_ROUTE} {paper('routing', bold=True)}")
        routed = [r for r in results if r["phase"] == "routing"]
        by_domain: dict[str, list[dict]] = {}
        for r in routed:
            by_domain.setdefault(r["domain"], []).append(r)
        hits = misses = 0
        for domain in sorted(by_domain):
            rs = by_domain[domain]
            landed = [r for r in rs if domain in (r.get("routed_tags") or [])]
            hits += len(landed)
            misses += len(rs) - len(landed)
            mark = GLYPH_DONE if len(landed) == len(rs) else GLYPH_FORMING
            print(f"  {mark}  {domain:10s} {dim(f'{len(landed)}/{len(rs)} landed on a node tagged {domain}')}")
            for r in rs:
                if domain not in (r.get("routed_tags") or []):
                    where = r.get("served_by") or "nowhere"
                    print(comment(f"{r['probe_id']} → {where} ({', '.join(r.get('routed_tags') or []) or 'untagged'})"))
        total_routed = hits + misses
        if total_routed:
            pct = round(100 * hits / total_routed)
            print(dim(f"  routing accuracy   {hits}/{total_routed}  ({pct}%)"))
            print(comment("measured against node domain_tags, not answer quality -- "
                          "run bench/ for that"))
        print()

    ok_n = sum(1 for r in results if r["ok"])
    print(dim("─" * 63))
    print(dim(f"{len(results)} request(s)   ·   {ok_n} ok   ·   {len(results) - ok_n} failed"
              f"   ·   {int(time.time() - started)}s"))
    print(dim(f"results   {out_path}"))
    print(comment("run this on every machine, then merge the jsonl -- each one records"))
    print(comment("its own client-side view, which is where network asymmetry shows up"))

    if args.json:
        print(json.dumps({"run_id": run_id, "out": str(out_path), "gateway_rtt_ms": rtt,
                          "requests": len(results), "ok": ok_n}))


def _parse_test_flags(args: argparse.Namespace) -> argparse.Namespace:
    """The top-level parser puts everything after the verb into `rest`
    (argparse.REMAINDER), so `common test --full` never reaches it -- the flag
    lands in rest and is silently ignored. Re-parse the leftovers here so the
    flags work in the position people actually type them."""
    p = argparse.ArgumentParser(prog="common test", add_help=False)
    p.add_argument("--gateway")
    p.add_argument("--repeats", type=int)
    p.add_argument("--full", action="store_true")
    p.add_argument("--nodes-only", action="store_true")
    p.add_argument("--routing-only", action="store_true")
    p.add_argument("--out")
    p.add_argument("--json", action="store_true")
    sub, _ = p.parse_known_args(args.rest)
    for key, value in vars(sub).items():
        if value not in (None, False):  # only override what was actually passed
            setattr(args, key, value)
    return args


def cmd_help(verb: str | None) -> None:
    if verb == "synth":
        print(dim("common synth <region>"))
        print()
        print(red("not built yet."))
        print(comment("this requires real Soup-of-Experts weight merging across"))
        print(comment("nearby specialists -- explicitly out of scope until that"))
        print(comment("architecture exists (see the v0.2/v0.3 build briefs). when"))
        print(comment("it ships, it will combine real weights and validate against"))
        print(comment("held-out demand -- not simulate the result."))
        return
    if verb == "map":
        print(dim("common map"))
        print()
        print(red("not built yet."))
        print(comment("needs a new gateway endpoint exposing node/request embeddings"))
        print(comment("(none is public today) plus a 2D projection. planned, not"))
        print(comment("guessed at -- see TODO(v0.4)."))
        return
    print(__doc__)


def build_repl_help() -> str:
    return dim(
        "/ask (implicit: just type)  /join  /serve  /leave  /status\n"
        "/demand  /peers  /contrib  /whoami  /config  /test  /model  /local  /help  /exit"
    )


def interactive_session(gateway: str, args: argparse.Namespace) -> None:
    print_banner_box("the commons. belonging to everyone and no one.")
    print()
    print(dim("you're in the interactive session. type a question, or a /command."))
    print(build_repl_help())
    print()
    session_model: str | None = args.model
    session_local = args.local
    while True:
        try:
            line = input(blue("› ", bold=True)).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if not line:
            continue
        if line in ("/exit", "/quit"):
            return
        if line == "/help":
            print(build_repl_help())
            continue
        if line == "/status":
            cmd_status(gateway, False)
            continue
        if line == "/peers":
            cmd_peers(gateway, False)
            continue
        if line == "/demand":
            cmd_demand(gateway, False)
            continue
        if line == "/contrib":
            cmd_contrib(gateway, False)
            continue
        if line == "/whoami":
            cmd_whoami(False)
            continue
        if line == "/config":
            cmd_config(False, args)
            continue
        if line.startswith("/test"):
            # /test full -> the 3-repeat sweep, same as `common test --full`
            args.full = "full" in line.split()[1:]
            cmd_test(gateway, args)
            continue
        if line == "/local":
            session_local = not session_local
            print(dim(f"local-only: {'on' if session_local else 'off'}"))
            continue
        if line.startswith("/model"):
            parts = line.split(maxsplit=1)
            session_model = parts[1] if len(parts) > 1 else None
            print(dim(f"model pinned to: {session_model or '(auto)'}"))
            continue
        if line in ("/join", "/serve", "/leave"):
            print(dim(f"run this from a real terminal instead: common {line[1:]}"))
            print(comment("join/leave manage a long-running process and a background"))
            print(comment("service -- not something to do mid-session here."))
            continue
        if line.startswith("/"):
            print(dim(f"unknown command: {line}"))
            continue
        cmd_ask(gateway, line, args.region, session_model, session_local, False, False, args.verbose)
        print()


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    _enable_windows_ansi()

    parser = argparse.ArgumentParser(prog="common", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
    parser.add_argument("verb", nargs="?", default=None)
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    parser.add_argument("--gateway", default=os.environ.get("COMMON_GATEWAY_URL", DEFAULT_GATEWAY))
    parser.add_argument("--region", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--no-update", action="store_true", default=bool(os.environ.get("COMMON_NO_UPDATE")))
    # `common test` only
    parser.add_argument("--repeats", type=int, default=1, help="test: repeats per probe per node")
    parser.add_argument("--full", action="store_true", help="test: 3 repeats per probe")
    parser.add_argument("--nodes-only", action="store_true", help="test: skip the routing check")
    parser.add_argument("--routing-only", action="store_true", help="test: skip the per-node sweep")
    parser.add_argument("--out", default=None, help="test: results jsonl path")
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args()

    _ARGS_NO_COLOR[0] = args.no_color

    if not args.no_update:
        self_update()

    if args.version:
        print_wordmark()
        return

    if args.help or args.verb == "help":
        cmd_help(args.rest[0] if args.rest else None)
        return

    gateway = args.gateway.rstrip("/")

    if args.verb is None:
        interactive_session(gateway, args)
        return

    if args.verb == "ask":
        question = " ".join(args.rest)
        if not question:
            print(red("✗ ask needs a question: common ask \"...\""), file=sys.stderr)
            sys.exit(1)
        cmd_ask(gateway, question, args.region, args.model, args.local, args.json, args.quiet, args.verbose)
    elif args.verb == "peers":
        cmd_peers(gateway, args.json)
    elif args.verb == "demand":
        cmd_demand(gateway, args.json)
    elif args.verb == "status":
        cmd_status(gateway, args.json)
    elif args.verb == "whoami":
        cmd_whoami(args.json)
    elif args.verb == "contrib":
        cmd_contrib(gateway, args.json)
    elif args.verb == "config":
        cmd_config(args.json, args)
    elif args.verb == "test":
        args = _parse_test_flags(args)
        cmd_test(args.gateway.rstrip("/"), args)
    elif args.verb == "join":
        cmd_join_or_serve("join", gateway, args)
    elif args.verb == "serve":
        model = args.rest[0] if args.rest else args.model
        if not model:
            print(red("✗ serve needs a model: common serve <model>"), file=sys.stderr)
            sys.exit(1)
        cmd_join_or_serve("serve", gateway, args, extra_model=model)
    elif args.verb == "leave":
        cmd_leave(gateway)
    elif args.verb in ("synth", "map"):
        cmd_help(args.verb)
    else:
        print(red(f"✗ unknown command: {args.verb}"), file=sys.stderr)
        print(dim("  → common help"), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
