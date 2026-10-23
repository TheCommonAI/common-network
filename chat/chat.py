#!/usr/bin/env python3
"""Chat with the Common Network from your terminal.

The Common Network Alpha (v0.1.13).

Copyright (C) 2026 Common AI Inc. Licensed under AGPL-3.0; see LICENSE at
https://github.com/TheCommonAI/common-network. This program comes with
ABSOLUTELY NO WARRANTY.

Usage:
    common-chat                    # interactive chat
    common-chat "your question"    # one-shot

Every reply is followed by a line showing which node answered and where the
time went — routing, first token, total, kept as three numbers because they
answer different questions — plus what was actually retained. Transparency is
a feature of the commons, not an afterthought.

On every run it checks GitHub for a newer version of itself and updates
in place (pass --no-update to skip).
"""
import argparse
import json
import os
import platform
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

VERSION = "0.1.13"
RELEASE = "The Common Network Alpha"
DEFAULT_GATEWAY = "https://gateway-production-b820.up.railway.app"
REPO = "TheCommonAI/common-network"
UPDATE_URL = f"https://raw.githubusercontent.com/{REPO}/main/chat/chat.py"

# Which program is asking, reported to the gateway as X-Common-Client.
#
# This client never registers a node -- join.py owns that -- so the gateway
# records nothing about it today; it only reads the header on POST /nodes. It
# is sent anyway, and named "common-cli" rather than "common-chat", because the
# three terminal entry points (common, common-chat, common join) install as one
# program from one installer, and the question the field exists to answer is
# app-versus-terminal, not which terminal verb was used. The same string in all
# three files is what keeps those two buckets clean.
CLIENT = f"common-cli/{VERSION}"

# Must stay ABOVE the gateway's own read timeout (forward_read_timeout_seconds,
# 180s): the gateway's failure says which node died and why, a socket timeout
# here can only say "timed out". Cold starts are why it is large at all -- a 7B
# on a donated laptop spends ~57s loading before its first token.
#
# gateway/tests/test_timeouts.py reads this name in both clients.
REQUEST_TIMEOUT_SECONDS = 240


def gateway_headers(extra: dict | None = None) -> dict:
    """Headers every gateway request carries -- the mirror of
    common-desktop's gatewayHeaders()."""
    return {"X-Common-Client": CLIENT, **(extra or {})}


BANNER = r"""
 ░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓██████████████▓▒░░▒▓██████████████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░
░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░
░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░
░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓██▓▒░
 ░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓██▓▒░


"""


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

    A Windows console renders the glyph vocabulary (← → ⚠) and the banner
    fine -- Python reaches it through the UTF-16 console API. A *redirected*
    stream does not: a pipe or a file gets the locale encoding, which is cp1252
    on most installs, and cp1252 cannot encode any of those characters. The
    result is UnicodeEncodeError and a dead client.

    Logging a session to a file, or piping it through anything, should not be
    fatal -- and on Windows it was.

    errors="replace" is deliberate rather than strict: a glyph is decoration,
    and decoration must never be able to stop the client from running. A stream
    that cannot be reconfigured at all (already replaced, closed, or not a
    TextIOWrapper) is left alone -- the banner is worth less than the session.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


# --- Palette / style ---------------------------------------------------------
# Exactly the four brand colours from the COMMON. design doc. No green, no
# purple, no gradients -- restraint is the point.
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


# Fixed glyph vocabulary -- see design doc 1.5. Consistency over cleverness.
GLYPH_WORK = dim("·")
GLYPH_ROUTE = blue("→")
GLYPH_RECV = dim("←")
GLYPH_DONE = blue("✓")
GLYPH_FORMING = red("⚠")
GLYPH_FAILED = red("✗")


def comment(text: str) -> str:
    return dim(f"# {text}")


def self_update() -> None:
    try:
        with urllib.request.urlopen(UPDATE_URL, timeout=5) as resp:
            remote = resp.read()
    except urllib.error.HTTPError as e:
        # See the note in common/common.py: offline stays silent, but a
        # 401/403/404 means the update channel is broken everywhere at once.
        if e.code in (401, 403, 404):
            print(f"note: updates unreachable ({e.code}) — running the installed version.",
                  file=sys.stderr)
        return
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

    print(f"{GLYPH_WORK} {dim('updating to the latest version...')}")
    try:
        with open(local_path, "wb") as f:
            f.write(remote)
    except OSError as e:
        print(f"{GLYPH_FORMING} {red(f'could not self-update ({e}), continuing with current version')}", file=sys.stderr)
        return

    os.execv(sys.executable, [sys.executable, local_path] + sys.argv[1:])


def _contributor_headers(gateway: str) -> dict:
    """The node token of the machine this client runs on, if it has joined
    (see `common join`). A gateway running REQUIRE_CONTRIBUTION passes a
    request only while the machine asking is also a machine donating; on open
    gateways the header is simply ignored.

    Sent only to the gateway that issued it: --gateway can point this client
    at anyone's server, and a credential that followed the flag would be
    handed to whoever runs it."""
    try:
        with open(os.path.expanduser("~/.common-network/identity.json")) as f:
            identity = json.load(f)
        token = identity.get("node_token")
        issuer = (identity.get("gateway") or "").rstrip("/")
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}
    if not token or issuer != gateway.rstrip("/"):
        return {}
    return {"X-Common-Node-Token": token}


@dataclass
class Timings:
    """The three clocks, kept apart on purpose.

    They were one clock -- started before the request went out, stopped after
    the last token -- printed as "routed in 54346ms". Routing is the gateway
    choosing a node: tens of milliseconds. Everything else in that number was
    the node loading its model into memory, ~57s for a cold 7B on a laptop.
    Reporting the sum as routing points at the wrong machine.
    """
    routing_ms: int | None   # the gateway's own share, from a response header
    ttft_ms: int | None      # to the first token: routing + model load + prompt
    total_ms: int            # to the last token


def _fmt_ms(ms: int | None) -> str:
    if ms is None:
        return "—"
    return f"{ms}ms" if ms < 1000 else f"{ms / 1000:.1f}s"


@dataclass
class Turn:
    """One answer, and what it cost. Was a 3-tuple; named because a client that
    reaches into position 2 for a latency number is how the wrong number got
    printed under the right label."""
    text: str
    node: str | None
    timings: Timings


def _stream_chat_gateway(gateway: str, messages: list[dict], region: str | None, target_node: str | None) -> Turn:
    body = {"model": "auto", "messages": messages, "stream": True}
    headers = {"Content-Type": "application/json"}
    if region:
        headers["X-Common-Region"] = region
    if target_node:
        headers["X-Common-Node"] = target_node
    headers.update(gateway_headers(_contributor_headers(gateway)))

    req = urllib.request.Request(
        f"{gateway}/v1/chat/completions", data=json.dumps(body).encode(), headers=headers, method="POST",
    )
    start = time.monotonic()
    # See REQUEST_TIMEOUT_SECONDS: above the gateway's read timeout on purpose,
    # so the gateway's explanation is what surfaces.
    try:
        resp = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")
        if e.code == 401:
            raise RuntimeError(
                "401: this gateway answers its contributors — run `common join` on this "
                "machine first, then chat from it. (" + detail + ")"
            )
        raise RuntimeError(f"{e.code}: {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(str(e.reason))

    node = resp.headers.get("X-Common-Node")
    # X-Common-Score is deliberately not read here. It is a blend of
    # similarity, cost and latency, and the only thing this client ever did
    # with it was render it as "{n} match" -- a claim the network cannot
    # support. See the note in print_footer. `common ask -v` still shows the
    # raw value, where its meaning is understood.
    route_hdr = resp.headers.get("X-Common-Route-Ms")
    routing_ms = int(route_hdr) if (route_hdr or "").isdigit() else None
    full = []
    ttft_ms: int | None = None
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
            choices = chunk.get("choices") or [{}]
            delta = choices[0].get("delta", {}).get("content")
            if delta:
                if ttft_ms is None:
                    ttft_ms = int((time.monotonic() - start) * 1000)
                print(paper(delta), end="", flush=True)
                full.append(delta)
    print()
    timings = Timings(routing_ms=routing_ms, ttft_ms=ttft_ms,
                      total_ms=int((time.monotonic() - start) * 1000))
    return Turn(text="".join(full), node=node, timings=timings)


# --- The layer network ------------------------------------------------------
#
# A machine running `common join --network --api` serves an OpenAI-compatible chat API on loopback and writes a
# status file. Chat talks to that, not to the gateway. COMMON_LEGACY_GATEWAY=1 (or --legacy) keeps the old path.

LAYER_API_PORT = 11440


def use_gateway(args=None) -> bool:
    return bool(getattr(args, "legacy", False)) or os.environ.get("COMMON_LEGACY_GATEWAY", "").strip().lower() in ("1", "true", "yes")


def _status_path() -> Path:
    override = os.environ.get("COMMONNET_STATUS_FILE")
    return Path(override) if override else Path.home() / ".common-network" / "net" / "status.json"


def _read_status() -> dict | None:
    try:
        status = json.loads(_status_path().read_text())
    except (OSError, ValueError):
        return None
    return status if isinstance(status, dict) else None


def _layer_base(status: dict | None) -> str:
    override = os.environ.get("COMMON_LAYER_API")
    if override:
        return override.rstrip("/")
    return f"http://127.0.0.1:{((status or {}).get('api') or {}).get('port') or LAYER_API_PORT}"


def _no_peer_message(status: dict | None) -> str:
    state = (status or {}).get("state")
    if state in ("starting", "downloading", "loading", "moving"):
        return f"your peer is still getting ready ({state}); try again in a moment, or run `common status`."
    if state == "up":
        return "your peer is running without the chat API; restart it with `common join --network --api`."
    return "no peer is running on this machine. Join the network and open the chat API: `common join --network --api`."


def stream_chat_layers(messages: list[dict]) -> Turn:
    status = _read_status()
    body = {"model": "auto", "messages": messages, "stream": True}
    req = urllib.request.Request(_layer_base(status) + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    start = time.monotonic()
    try:
        resp = opener.open(req, timeout=REQUEST_TIMEOUT_SECONDS)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")
        try:
            detail = json.loads(detail)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            pass
        raise RuntimeError(f"{e.code}: {detail}")
    except (urllib.error.URLError, OSError):
        raise RuntimeError(_no_peer_message(status))
    full = []
    ttft_ms: int | None = None
    done = False
    with resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                done = True
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("error"):
                raise RuntimeError(str((chunk["error"] or {}).get("message") or chunk["error"]))
            delta = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
            if delta:
                if ttft_ms is None:
                    ttft_ms = int((time.monotonic() - start) * 1000)
                print(paper(delta), end="", flush=True)
                full.append(delta)
    print()
    if not done:
        raise RuntimeError("the answer stream ended before it was complete; a peer may have dropped out. Retry the question.")
    peer = (status or {}).get("peer")
    return Turn(text="".join(full), node=(f"the layer network, via peer {peer[:8]}" if peer else "the layer network"),
                timings=Timings(routing_ms=None, ttft_ms=ttft_ms, total_ms=int((time.monotonic() - start) * 1000)))


def stream_chat(gateway: str, messages: list[dict], region: str | None, target_node: str | None, legacy: bool = False) -> Turn:
    if legacy or use_gateway():
        return _stream_chat_gateway(gateway, messages, region, target_node)
    return stream_chat_layers(messages)


def list_nodes(gateway: str, legacy: bool = False) -> None:
    if legacy or use_gateway():
        _list_nodes_gateway(gateway)
        return
    status = _read_status()
    peers = (status or {}).get("peers") or []
    if not peers:
        print(dim("no peers visible from this machine."))
        print(dim("  → common join --network --api"))
        return
    for p in peers:
        badge = GLYPH_DONE if p.get("state") == "up" else GLYPH_FAILED
        w = p.get("window")
        print(f"{badge} {paper((p.get('peer') or '')[:8])}  {dim(f'layers {w[0]}-{w[1]}' if w else 'no layers yet')}  {dim(p.get('state') or '')}")


def print_footer(node: str | None, timings: Timings) -> None:
    print()
    print(dim("─" * 63))
    print(dim(f"served by   {node or 'unknown'}"))
    # Three numbers, not one.
    #
    # This line used to read "routed in 54346ms", and that was false twice
    # over. The clock behind it started before the request went out and stopped
    # after the last token, so it was never the routing time; and a cold 7B
    # loading its weights for ~57s was being reported as slow *routing*, which
    # points at the gateway when the wait was the node's model.
    parts = []
    if timings.routing_ms is not None:
        parts.append(f"routed in {_fmt_ms(timings.routing_ms)}")
    if timings.ttft_ms is not None:
        parts.append(f"first token {_fmt_ms(timings.ttft_ms)}")
    parts.append(f"total {_fmt_ms(timings.total_ms)}")
    print(dim("   ·   ".join(parts)))
    if (timings.ttft_ms is not None and timings.routing_ms is not None
            and timings.ttft_ms > 5000):
        # Past a few seconds the wait is the node getting its model into
        # memory, and saying so turns "this network is slow" into "this node
        # was cold" -- which is a thing the user can do something about.
        print(comment(f"   {_fmt_ms(max(timings.ttft_ms - timings.routing_ms, 0))} of that "
                      f"was {node or 'the node'} loading its model — the next one is fast."))
    # The blended score is deliberately not shown as a "match" percentage. It
    # is similarity+cost+latency, and the similarity part barely moves:
    # measured live, the gibberish "asdfgh qwerty zxcvbn" scored above a real
    # Python question. Rendering it as confidence tells the user the network
    # assessed their request when it did not. See the same note in
    # common/common.py, where this number is only shown under -v.


def one_shot(gateway: str, question: str, region: str | None, target_node: str | None, legacy: bool = False) -> None:
    messages = [{"role": "user", "content": question}]
    try:
        turn = stream_chat(gateway, messages, region, target_node, legacy)
    except RuntimeError as e:
        print(f"{GLYPH_FAILED} {red('the network could not answer that.')}", file=sys.stderr)
        print(comment(str(e)), file=sys.stderr)
        sys.exit(1)
    print_footer(turn.node, turn.timings)


def interactive(gateway: str, region: str | None, target_node: str | None, legacy: bool = False) -> None:
    print(paper(BANNER, bold=True))
    print(dim(f"common network chat — talking to {gateway}" if (legacy or use_gateway()) else "common network chat — talking to your peer's layer network (127.0.0.1)"))
    if target_node:
        print(dim(f"talking to a specific node: {target_node}"))
    print(comment("ctrl+c or ctrl+d to quit.\n"))
    messages: list[dict] = []
    while True:
        try:
            question = input(blue("you: ", bold=True)).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if not question:
            continue
        messages.append({"role": "user", "content": question})
        print(flush=True)
        try:
            turn = stream_chat(gateway, messages, region, target_node, legacy)
        except RuntimeError as e:
            print(f"{GLYPH_FAILED} {red('the network could not answer that.')}", file=sys.stderr)
            print(comment(str(e)), file=sys.stderr)
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": turn.text})
        print_footer(turn.node, turn.timings)
        print()


def _list_nodes_gateway(gateway: str) -> None:
    req = urllib.request.Request(f"{gateway}/nodes", headers=gateway_headers())
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            nodes = json.loads(resp.read().decode())
    except (urllib.error.URLError, socket.timeout) as e:
        print(f"{GLYPH_FAILED} {red('could not reach the gateway.')}", file=sys.stderr)
        print(comment(str(e)), file=sys.stderr)
        sys.exit(1)

    if not nodes:
        print(dim("no nodes registered."))
        return
    for n in nodes:
        badge = GLYPH_DONE if n["healthy"] else GLYPH_FAILED
        print(f"{badge} {paper(n['name'])}  {dim(n['model_name'])}")


def main() -> None:
    _force_utf8_output()
    sys.stdout.reconfigure(line_buffering=True)
    _enable_windows_ansi()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="*", help="Ask a one-shot question (omit for interactive chat)")
    parser.add_argument("--gateway", default=os.environ.get("COMMON_GATEWAY_URL", DEFAULT_GATEWAY), help="Gateway base URL (default: the shared Common Network gateway)")
    parser.add_argument("--region", default=None, help="Optional region hint for routing, e.g. au-adelaide")
    parser.add_argument("--node", default=None, help="Target a specific node by name instead of letting the router pick (see --list-nodes)")
    parser.add_argument("--list-nodes", action="store_true", help="List the peers this machine can see (or the gateway's nodes with --legacy), then exit")
    parser.add_argument("--legacy", action="store_true", help="Talk to the old gateway instead of the layer network (same as COMMON_LEGACY_GATEWAY=1)")
    parser.add_argument("--no-update", action="store_true", default=bool(os.environ.get("COMMON_NO_UPDATE")), help="Skip the self-update check")
    args = parser.parse_args()

    if not args.no_update:
        self_update()

    gateway = args.gateway.rstrip("/")

    if args.list_nodes:
        list_nodes(gateway, args.legacy)
        return

    if args.question:
        one_shot(gateway, " ".join(args.question), args.region, args.node, args.legacy)
    else:
        interactive(gateway, args.region, args.node, args.legacy)


if __name__ == "__main__":
    main()
