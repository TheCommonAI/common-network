#!/usr/bin/env python3
"""Chat with the Common Network from your terminal.

Usage:
    common-chat                    # interactive chat
    common-chat "your question"    # one-shot

Every reply is followed by a line showing which node answered, its
routing score, latency, and what was actually retained — transparency
is a feature of the commons, not an afterthought.

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

DEFAULT_GATEWAY = "https://gateway-production-b820.up.railway.app"
REPO = "TheCommonAI/common-network"
UPDATE_URL = f"https://raw.githubusercontent.com/{REPO}/main/chat/chat.py"

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


def stream_chat(gateway: str, messages: list[dict], region: str | None, target_node: str | None) -> tuple[str, str | None, str | None]:
    body = {"model": "auto", "messages": messages, "stream": True}
    headers = {"Content-Type": "application/json"}
    if region:
        headers["X-Common-Region"] = region
    if target_node:
        headers["X-Common-Node"] = target_node

    req = urllib.request.Request(
        f"{gateway}/v1/chat/completions", data=json.dumps(body).encode(), headers=headers, method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=180)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")
        raise RuntimeError(f"{e.code}: {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(str(e.reason))

    node = resp.headers.get("X-Common-Node")
    score = resp.headers.get("X-Common-Score")
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
            choices = chunk.get("choices") or [{}]
            delta = choices[0].get("delta", {}).get("content")
            if delta:
                print(paper(delta), end="", flush=True)
                full.append(delta)
    print()
    return "".join(full), node, score


def print_footer(node: str | None, score: str | None, latency_ms: int) -> None:
    score_str = ""
    weak = False
    if score not in (None, "forced"):
        try:
            score_str = f"   ·   {float(score):.2f} match"
            weak = float(score) < 0.5
        except ValueError:
            score_str = f"   ·   {score}"
    marker = f"  {GLYPH_FORMING}" if weak else ""
    print()
    print(dim("─" * 63))
    print(dim(f"served by   {node or 'unknown'}{score_str}{marker}"))
    retention = "embedding retained for demand analytics · no raw text stored"
    print(dim(f"routed in   {latency_ms}ms   ·   {retention}   ·   no one owns this"))


def one_shot(gateway: str, question: str, region: str | None, target_node: str | None) -> None:
    messages = [{"role": "user", "content": question}]
    start = time.monotonic()
    try:
        _, node, score = stream_chat(gateway, messages, region, target_node)
    except RuntimeError as e:
        print(f"{GLYPH_FAILED} {red('the network could not answer that.')}", file=sys.stderr)
        print(comment(str(e)), file=sys.stderr)
        sys.exit(1)
    print_footer(node, score, int((time.monotonic() - start) * 1000))


def interactive(gateway: str, region: str | None, target_node: str | None) -> None:
    print(paper(BANNER, bold=True))
    print(dim(f"common network chat — talking to {gateway}"))
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
        start = time.monotonic()
        try:
            answer, node, score = stream_chat(gateway, messages, region, target_node)
        except RuntimeError as e:
            print(f"{GLYPH_FAILED} {red('the network could not answer that.')}", file=sys.stderr)
            print(comment(str(e)), file=sys.stderr)
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": answer})
        print_footer(node, score, int((time.monotonic() - start) * 1000))
        print()


def list_nodes(gateway: str) -> None:
    req = urllib.request.Request(f"{gateway}/nodes")
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
    sys.stdout.reconfigure(line_buffering=True)
    _enable_windows_ansi()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="*", help="Ask a one-shot question (omit for interactive chat)")
    parser.add_argument("--gateway", default=os.environ.get("COMMON_GATEWAY_URL", DEFAULT_GATEWAY), help="Gateway base URL (default: the shared Common Network gateway)")
    parser.add_argument("--region", default=None, help="Optional region hint for routing, e.g. au-adelaide")
    parser.add_argument("--node", default=None, help="Target a specific node by name instead of letting the router pick (see --list-nodes)")
    parser.add_argument("--list-nodes", action="store_true", help="List registered nodes and their health, then exit")
    parser.add_argument("--no-update", action="store_true", default=bool(os.environ.get("COMMON_NO_UPDATE")), help="Skip the self-update check")
    args = parser.parse_args()

    if not args.no_update:
        self_update()

    gateway = args.gateway.rstrip("/")

    if args.list_nodes:
        list_nodes(gateway)
        return

    if args.question:
        one_shot(gateway, " ".join(args.question), args.region, args.node)
    else:
        interactive(gateway, args.region, args.node)


if __name__ == "__main__":
    main()
