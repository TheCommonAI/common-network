"""Serve one block on a Cloudflare quick tunnel and publish the tunnel URL.

    python3 -m commonnet.net.tunnel --stub --layers 0 4 --out PATH
    python3 -m commonnet.net.tunnel --block MODULE:FACTORY --layers 0 8 --out PATH

In order:
1. Picks a free loopback port and starts the block server there as a child
   process (commonnet.net.server). The model lives in that process, not in this one.
2. Waits for the server's /health to answer.
3. Starts a Cloudflare quick tunnel to it with join.start_tunnel, the call the
   worker and the group relay make. It waits for the tunnel to answer /health with
   join.wait_for_tunnel. If the tunnel is still unreachable after that warm-up, it
   stops and exits with an error rather than publish a dead URL.
4. Writes {"tunnel_url", "layers", "device"} to the --out file, atomically.
5. Keeps running. Each health interval it checks the tunnel. A tunnel that stays
   unreachable for the grace period is replaced, and the new URL is written.

Stopping: SIGTERM and Ctrl+C both unwind to the same cleanup. That stops the
tunnel, terminates the block server and writes "tunnel_url": null, so a reader
never takes a stopped URL for a live one.

No inbound ports are opened. The block server listens on 127.0.0.1 only, and the
tunnel is the only way in. The server has no authentication of its own, so the
tunnel URL is an address that anyone who learns it can call.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from commonnet.net import auth, protocol
from commonnet.net.client import BlockClient, BlockError

REPO_ROOT = Path(__file__).resolve().parents[2]
JOIN_DIR = REPO_ROOT / "join"
DEFAULT_OUT = Path.home() / ".common-network" / "commonnet" / "block-tunnel.json"
HEALTH_PATH = "/health"
SERVER_START_TIMEOUT_SECONDS = 30.0
HEALTH_CHECK_INTERVAL_SECONDS = 120.0  # quick tunnels can reconnect under a new hostname silently
TUNNEL_GRACE_SECONDS = 600.0           # unreachable this long, and the tunnel is replaced
SERVER_LOG_NAME = "block-server.log"
WARMUP_ROUNDS = 6  # x join.TUNNEL_WARMUP_TIMEOUT_SECONDS (30s) = 3 minutes for the tunnel to become routable


class LauncherError(RuntimeError):
    """The launcher could not bring the block up. Its message says why."""


def _join():
    """join/join.py, imported the way group.py imports it: join/ goes on the path first."""
    if str(JOIN_DIR) not in sys.path:
        sys.path.insert(0, str(JOIN_DIR))
    import join
    return join


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _default_open_tunnel(local_url: str, port: int):
    """A Cloudflare quick tunnel to the local server: (process, public URL)."""
    return _join().start_tunnel(local_url, f"127.0.0.1:{port}")


def _publish(path: Path, url, layers, device) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"tunnel_url": url, "layers": list(layers) if layers else None, "device": device}
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _start_server(server_args: list, port: int, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:
        return subprocess.Popen(
            [sys.executable, "-m", "commonnet.net.server", "--port", str(port), *server_args],
            cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT,
        )


def _wait_for_server(local_url: str, server: subprocess.Popen, log_path: Path, timeout: float,
                     token: str | None) -> tuple:
    client = BlockClient(token=token, retries=0)
    deadline = time.monotonic() + timeout
    while True:
        if server.poll() is not None:
            raise LauncherError(f"the block server exited with code {server.returncode}; see {log_path}")
        try:
            return client.health(local_url, timeout=2)
        except (BlockError, ValueError):
            pass
        if time.monotonic() >= deadline:
            raise LauncherError(f"the block server did not answer {HEALTH_PATH} within {timeout:.0f}s; see {log_path}")
        time.sleep(0.2)


def run(server_args: list, out_path: Path, *, token: str | None = None,
        open_tunnel=_default_open_tunnel,
        stop: threading.Event | None = None, log=print, port: int | None = None,
        health_interval: float = HEALTH_CHECK_INTERVAL_SECONDS, grace: float = TUNNEL_GRACE_SECONDS,
        server_timeout: float = SERVER_START_TIMEOUT_SECONDS) -> None:
    """Run the block and its tunnel until `stop` is set or the block dies.

    `open_tunnel(local_url, port)` returns (process, public URL). It is injectable so
    that tests run without cloudflared. `server_args` are passed to commonnet.net.server.
    `token` is the bearer token the server requires (None only for --stub). The server
    child reads the same token from the environment or from --token-file in server_args.
    """
    join = _join()
    stop = stop or threading.Event()
    port = port or free_loopback_port()
    local_url = f"http://127.0.0.1:{port}"
    log_path = out_path.parent / SERVER_LOG_NAME
    layers = None
    device = None
    server = None
    tunnel = None
    try:
        server = _start_server(server_args, port, log_path)
        layers, device = _wait_for_server(local_url, server, log_path, server_timeout, token)
        log(f"block server on {local_url}: layers [{layers[0]}, {layers[1]}) on {device}")

        tunnel, url = open_tunnel(local_url, port)
        log(f"tunnel URL: {url}")
        # A fresh quick-tunnel hostname can take minutes to resolve on some resolvers (negative
        # caching: seen >40s on a Mac using Tailscale DNS), so one 30s join warm-up is not enough.
        warm_deadline = time.monotonic() + join.TUNNEL_WARMUP_TIMEOUT_SECONDS * WARMUP_ROUNDS
        while not join.wait_for_tunnel(url, token, HEALTH_PATH):
            if time.monotonic() >= warm_deadline or tunnel.poll() is not None:
                break
        else:
            warm_deadline = None
        if warm_deadline is not None:
            raise LauncherError(
                f"the tunnel did not answer {HEALTH_PATH} within the warm-up window, so nothing was published"
            )
        _publish(out_path, url, layers, device)
        log(f"published to {out_path}")

        healthy_at = time.monotonic()
        while not stop.wait(health_interval):
            if server.poll() is not None:
                raise LauncherError(f"the block server exited with code {server.returncode}; see {log_path}")
            if tunnel.poll() is None and _healthy(join, url, token):
                healthy_at = time.monotonic()
                continue
            unreachable_for = time.monotonic() - healthy_at
            if unreachable_for < grace:
                log(f"the tunnel is not reachable yet ({unreachable_for:.0f}s); "
                    f"it is replaced after {grace:.0f}s")
                continue
            log(f"the tunnel has been unreachable for {unreachable_for:.0f}s; restarting it")
            join.stop_process(tunnel)
            tunnel, url = open_tunnel(local_url, port)
            healthy_at = time.monotonic()
            _publish(out_path, url, layers, device)
            log(f"tunnel URL: {url}")
    finally:
        join.stop_process(tunnel)
        join.stop_process(server)
        # Always written, even when nothing was published: a URL left over from an
        # earlier run must not look live.
        _publish(out_path, None, layers, device)
        log("block server and tunnel stopped")


def _healthy(join, url: str, token) -> bool:
    # The server can die mid-probe (connection reset); that is "unhealthy", and the
    # next loop iteration reports the dead server instead of this probe raising.
    try:
        return join.tunnel_is_healthy(url, token, HEALTH_PATH)
    except OSError:
        return False


def _interrupted(signum, frame) -> None:
    # SIGTERM unwinds exactly like Ctrl+C, so run()'s cleanup stops the tunnel and the server.
    raise KeyboardInterrupt


def _log(message: str) -> None:
    print(message, flush=True)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(prog="commonnet.net.tunnel", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layers", nargs=2, type=int, metavar=("START", "END"), required=True,
                        help="the half-open layer range this block serves")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--stub", action="store_true", help="serve the identity stub block (no torch)")
    source.add_argument("--block", metavar="MODULE:FACTORY", help="serve FACTORY(start, end) from MODULE")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"JSON file to write the tunnel URL to (default: {DEFAULT_OUT})")
    parser.add_argument("--token-file", metavar="PATH",
                        help=f"file holding the bearer token (or set {auth.TOKEN_ENV})")
    args = parser.parse_args(argv)

    try:
        protocol.check_layers(args.layers)
    except protocol.ProtocolError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    token = None
    if not args.stub:
        try:
            token = auth.load_token(args.token_file)
        except auth.TokenError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if token is None:
            print(f"error: a real block needs a token: set {auth.TOKEN_ENV} or pass --token-file "
                  "(--stub runs without one)", file=sys.stderr)
            return 2
    server_args = ["--layers", str(args.layers[0]), str(args.layers[1])]
    if args.stub:
        server_args += ["--stub"]
    else:
        server_args += ["--block", args.block]
        if args.token_file:
            server_args += ["--token-file", str(args.token_file)]

    print("ctrl+c or SIGTERM stops the block server and its tunnel", flush=True)
    previous = signal.signal(signal.SIGTERM, _interrupted)
    try:
        run(server_args, args.out, token=token, log=_log)
    except KeyboardInterrupt:
        print("stopping the block server and its tunnel", flush=True)
    except LauncherError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0


if __name__ == "__main__":
    sys.exit(main())
