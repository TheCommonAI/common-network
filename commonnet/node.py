"""A peer of the decentralised layer network: what `common join --network` runs. See commonnet/NETWORK.md.

    python -m commonnet.node --network https://SEED-URL        # learn the manifest from a seed peer
    python -m commonnet.node --network manifest.json --seed https://SEED-URL
    python -m commonnet.node --network manifest.json --stub --no-tunnel --advertise http://127.0.0.1:18901

It picks the window of layers the network needs most, downloads only those layers, serves them
behind a Cloudflare quick tunnel, and publishes a signed record about itself. Peers learn each other by
gossip, so there is no central server. Every 15 s it re-signs its record with its current tunnel URL
(tunnel addresses change). If a stage loses every peer and this one can be spared, it moves there.
Ctrl+C or SIGTERM publishes `leaving` and stops cleanly.

Nothing here registers with a gateway. The block token is generated per run and travels only in the signed
record, which is public to anyone on the mesh: it keeps scanners off a tunnel URL, nothing more.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import urllib.request
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from commonnet.mesh import placement
from commonnet.mesh import gossip as mesh_gossip
from commonnet.mesh.gossip import Gossiper, GossipError, PeerTable
from commonnet.mesh.manifest import Manifest, ManifestError
from commonnet.mesh.records import Identity, RecordError
from commonnet.net.server import BlockServer, StubBlock

HOME = Path(os.environ.get("COMMON_NETWORK_HOME") or Path.home() / ".common-network")
NETWORK_ENV = "COMMONNET_NETWORK"
# The operator fills in always-on peers here once there are any; --seed and the manifest add more.
DEFAULT_SEEDS: tuple[str, ...] = ()
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "networks" / "main.json"
# Seeds change without a release: this file on main lists peers that are normally up.
SEEDS_URL = "https://raw.githubusercontent.com/TheCommonAI/common-network/main/commonnet/networks/seeds.json"

RESIGN_SECONDS = 15.0
REBALANCE_SECONDS = 60.0
MOVE_COOLDOWN_SECONDS = 300.0
TUNNEL_CHECK_SECONDS = 120.0
TUNNEL_GRACE_SECONDS = 600.0
TUNNEL_WAIT_SECONDS = 180.0
VRAM_RESERVE_GIB = 1.5
MAC_MEMORY_FRACTION = 0.62
MAX_GOSSIP_RECORDS = 500
STUB_GIB = 100.0


class NodeError(RuntimeError):
    """The node cannot start or continue. The message says what to do."""


@dataclass(frozen=True)
class Hardware:
    backend: str  # "mlx" | "nf4" | "stub"
    gib: float    # memory this node offers for its layers
    name: str


def _sysctl_memsize() -> int:
    out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=10)
    return int(out.stdout.strip())


def _nvidia_free_mib() -> tuple[float, str] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=memory.free,name", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    best = None
    for line in out.stdout.strip().splitlines():
        free, _, name = line.partition(",")
        try:
            row = (float(free.strip()), name.strip())
        except ValueError:
            continue
        if best is None or row[0] > best[0]:
            best = row
    return best


def detect_hardware(system: str | None = None, machine: str | None = None, memsize=_sysctl_memsize,
                    nvidia=_nvidia_free_mib) -> Hardware:
    """Apple silicon -> mlx, NVIDIA -> nf4; anything else is refused (the network is GPU only)."""
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        gib = memsize() / 2**30 * MAC_MEMORY_FRACTION
        return Hardware("mlx", round(gib, 2), "Apple silicon")
    found = nvidia() if system in ("Windows", "Linux") else None
    if found:
        free_mib, name = found
        gib = max(0.0, free_mib / 1024 - VRAM_RESERVE_GIB)
        return Hardware("nf4", round(gib, 2), name or "NVIDIA GPU")
    raise NodeError("this machine has no supported GPU. The layer network serves on Apple silicon or NVIDIA "
                    "(nvidia-smi must work); CPU-only machines cannot join.")


def fetch_manifest(seed: str) -> Manifest:
    try:
        return mesh_gossip.fetch_manifest(seed, timeout=15)
    except (GossipError, ManifestError, ValueError, OSError) as e:
        raise NodeError(f"could not read the manifest from {seed} ({e})") from None


def fetch_default_seeds(environ=None, opener=None) -> list[str]:
    """Seed URLs listed in the repository (best effort: an empty list means this peer starts the network)."""
    environ = os.environ if environ is None else environ
    url = environ.get("COMMONNET_SEEDS_URL", SEEDS_URL)
    try:
        with (opener or urllib.request.urlopen)(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        found = [u for u in data.get("seeds", []) if isinstance(u, str) and re.match(r"^https?://", u)]
        return found
    except (OSError, ValueError, AttributeError):
        return []


def resolve_network(network: str | None, seeds: list[str], environ=None) -> tuple[Manifest, list[str]]:
    """(manifest, seed URLs) from --network: a manifest file or JSON, or a seed peer's URL."""
    environ = os.environ if environ is None else environ
    network = network or environ.get(NETWORK_ENV) or None
    seeds = list(seeds)
    if network is None:
        pool = seeds or list(DEFAULT_SEEDS)
        if pool:
            network = pool[0]
        else:
            # Nothing named: the built-in network. A peer with no seed to reach starts the network itself.
            network = str(DEFAULT_MANIFEST)
            seeds = fetch_default_seeds(environ)
    if re.match(r"^https?://", network):
        manifest = fetch_manifest(network)
        seeds = [network] + [s for s in seeds if s != network]
    else:
        try:
            manifest = Manifest.load(network)
        except ManifestError as e:
            raise NodeError(str(e)) from None
    for s in list(manifest.seeds) + list(DEFAULT_SEEDS):
        if s not in seeds:
            seeds.append(s)
    return manifest, seeds


class Node:
    """One peer. Construct, start(), later stop(). Everything slow or external is injectable for tests."""

    def __init__(self, manifest: Manifest, seeds: list[str], identity: Identity, hardware: Hardware, *,
                 port: int = 0, stub: bool = False, tunnel: bool = True, advertise: str | None = None,
                 log=print, tunnel_opener=None, block_builder=None, fetcher=None,
                 resign_s: float = RESIGN_SECONDS, rebalance_s: float = REBALANCE_SECONDS,
                 cooldown_s: float = MOVE_COOLDOWN_SECONDS, gossip_s: float | None = None, clock=time.time):
        self.manifest, self.seeds, self.identity, self.hw = manifest, list(seeds), identity, hardware
        self.port, self.stub, self.use_tunnel, self.advertise = port, stub, tunnel, advertise
        self.log = log
        self._open_tunnel = tunnel_opener
        self._build_block = block_builder
        self._fetch = fetcher
        self.resign_s, self.rebalance_s, self.cooldown_s, self.gossip_s = resign_s, rebalance_s, cooldown_s, gossip_s
        self.clock = clock
        self.table = PeerTable(manifest.id)
        self.token = secrets.token_urlsafe(24)
        self.state = "loading"
        self.window: tuple[int, int] | None = None
        self.url: str | None = None
        self.server: BlockServer | None = None
        self._block = None
        self._tunnel_proc = None
        self._tunnel_bad_since: float | None = None
        self._last_move = 0.0
        self._lock = threading.RLock()
        self._stop = threading.Event()   # set once stop() starts: loops end
        self._leave = threading.Event()  # set by a signal or by stop(): wait() returns
        self._threads: list[threading.Thread] = []
        self._gossiper = None

    # -- the mesh endpoints every peer serves, unauthenticated --------------------------------

    def handlers(self) -> dict:
        def manifest(_body):
            return 200, mesh_gossip.manifest_body(self.manifest)

        def peers(_body):
            return 200, mesh_gossip.peers_body(self.table)

        def gossip(body):
            records = body.get("records") if isinstance(body, dict) else None
            if not isinstance(records, list) or len(records) > MAX_GOSSIP_RECORDS:
                return 400, {"error": f"body must be {{\"records\": [...]}} with at most {MAX_GOSSIP_RECORDS}"}
            return 200, mesh_gossip.handle_gossip(self.table, body)

        return {("GET", "/manifest"): manifest, ("GET", "/peers"): peers, ("POST", "/gossip"): gossip}

    # -- the signed record ---------------------------------------------------------------------

    def record(self) -> dict:
        with self._lock:
            return self.identity.sign_record({
                "network": self.manifest.id, "url": self.url, "token": self.token,
                "window": None if self.window is None else list(self.window),
                "backend": self.hw.backend, "state": self.state, "gib": self.hw.gib}, now=self.clock())

    def publish(self) -> dict:
        record = self.record()
        self.table.merge(record)
        return record

    # -- life cycle ----------------------------------------------------------------------------

    def start(self) -> "Node":
        self.log(f"peer {self.identity.peer}: {self.hw.name}, offering {self.hw.gib:.1f} GiB on {self.hw.backend}")
        self._learn()
        self._gossiper = Gossiper(self.table, self.manifest, self.record, self.seeds)
        if self.gossip_s is not None:
            self._gossiper.interval = (self.gossip_s, self.gossip_s)
        self.publish()
        self._gossiper.start()
        self._spawn(self._maintain, "node-maintain")
        window = self._choose()
        self.log(f"serving layers {window[0]}-{window[1]} as {self.identity.peer}")
        self._serve(window)
        with self._lock:
            self.window, self.state = window, "up"
        self.publish()
        self._push_to_peers()
        self.log(f"up at {self.url}")
        self.log(f"you are part of the network. A friend can join through you with:  common join --network {self.url}")
        return self

    def stop(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._leave.set()
        with self._lock:
            self.state = "leaving"
        try:
            self.publish()
            self._push_to_peers()
        except Exception as e:  # leaving is best effort
            self.log(f"could not announce leaving ({type(e).__name__})")
        if self._gossiper is not None:
            self._gossiper.stop()
        self._teardown_server()
        self._close_tunnel()
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(timeout=5)

    def request_leave(self) -> None:
        self._leave.set()

    def wait(self) -> None:
        self._leave.wait()

    def _spawn(self, target, name):
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    # -- learning about peers ------------------------------------------------------------------

    def _learn(self) -> None:
        for seed in self.seeds:
            try:
                mesh_gossip.fetch_peers(self.table, seed, timeout=15)
            except (GossipError, ValueError, OSError):
                self.log(f"seed {urlsplit(seed).netloc} did not answer; continuing")

    def _push_to_peers(self) -> None:
        """Tell a few peers our latest record now rather than at the next gossip round."""
        self.table.merge(self.record())
        targets = [r["url"] for r in self.table.live() if r["peer"] != self.identity.peer and r.get("url")][:4]
        for url in targets + self.seeds[:1]:
            try:
                mesh_gossip.exchange(self.table, url, timeout=8)
            except (GossipError, ValueError, OSError):
                pass

    # -- choosing and serving a window ---------------------------------------------------------

    def _choose(self) -> tuple[int, int]:
        try:
            return placement.choose_window(self.manifest, self.table.live(), self.hw.gib, self.hw.backend,
                                           me=self.identity.peer)
        except placement.NoWindow as e:
            raise NodeError(f"{e}. This network needs at least "
                            f"{self.manifest.window_gib(*self.manifest.stages()[0], self.hw.backend):.1f} GiB free.") from None

    def _make_block(self, window):
        if self._build_block is not None:
            return self._build_block(window, self.manifest, self.hw)
        if self.stub:
            return StubBlock(*window)
        repo = self.manifest.repo(self.hw.backend)
        if not repo:
            raise NodeError(f"the manifest has no {self.hw.backend} checkpoint for {self.manifest.model.get('name')}")
        fetch = self._fetch
        local = os.environ.get("COMMONNET_LOCAL_CHECKPOINT")  # development/offline: use this directory, download nothing
        if fetch is None and local:
            fetch = lambda r, w: local  # noqa: E731
        if fetch is None:
            from commonnet import hub
            fetch = lambda r, w: hub.Hub(r).fetch(layers=tuple(w))  # noqa: E731
        self.log(f"downloading layers {window[0]}-{window[1]} of {repo} (only these)...")
        try:
            checkpoint = fetch(repo, window)
        except Exception as e:
            raise NodeError(f"the download failed ({e}); join again to resume") from None
        # Qwen activations pass float16's range on some prompts (NaN logits), so the MLX block computes in
        # bfloat16 and the hidden state between peers is float32. An explicit COMMONNET_DTYPE wins, then the
        # manifest's per-backend dtype. COMMONNET_MLX_CACHE_MB is left to the environment untouched.
        dtype = (self.manifest.model.get("dtype") or {}).get(self.hw.backend)
        if self.hw.backend == "mlx":
            dtype = dtype or "bfloat16"
        if dtype and not os.environ.get("COMMONNET_DTYPE"):
            os.environ["COMMONNET_DTYPE"] = dtype
        from commonnet.net import factory
        return factory.build_block(window[0], window[1], backend=self.hw.backend, checkpoint=checkpoint)

    def _serve(self, window) -> None:
        block = self._make_block(window)
        server = BlockServer(block, port=self.port, token=self.token, extra_handlers=self.handlers()).start()
        with self._lock:
            self._block, self.server = block, server
            self.port = server.port
        if self.use_tunnel:
            if self._tunnel_proc is None:
                self._start_tunnel()
        else:
            with self._lock:
                self.url = self.advertise or server.url

    def _teardown_server(self) -> None:
        with self._lock:
            server, block = self.server, self._block
            self.server = self._block = None
        if server is not None:
            server.stop()
        stop = getattr(block, "shutdown", None)
        if stop:
            try:
                stop()
            except Exception:
                pass

    # -- the tunnel ----------------------------------------------------------------------------

    def _tunnel_opener(self):
        if self._open_tunnel is not None:
            return self._open_tunnel
        from commonnet.net import tunnel
        return tunnel._default_open_tunnel

    def _start_tunnel(self) -> None:
        local = self.server.url
        self.log("opening a Cloudflare tunnel...")
        try:
            proc, url = self._tunnel_opener()(local, self.server.port)
        except Exception as e:
            raise NodeError(f"could not open a tunnel ({e}). Is cloudflared installed? `common join` installs it.") from None
        with self._lock:
            self._tunnel_proc, self.url = proc, url
        deadline = time.monotonic() + TUNNEL_WAIT_SECONDS
        from commonnet.net import tunnel as tunnel_mod
        while time.monotonic() < deadline and not self._stop.is_set():
            if tunnel_mod._probe(url, self.token):
                return
            self._stop.wait(3.0)
        self.log("the tunnel is not reachable yet (slow DNS here?); publishing it anyway")

    def _close_tunnel(self) -> None:
        proc, self._tunnel_proc = self._tunnel_proc, None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _check_tunnel(self) -> None:
        if not self.use_tunnel or self._tunnel_proc is None or self.state != "up":
            return
        from commonnet.net import tunnel as tunnel_mod
        if tunnel_mod._probe(self.url, self.token):
            self._tunnel_bad_since = None
            return
        now = time.monotonic()
        self._tunnel_bad_since = self._tunnel_bad_since or now
        if now - self._tunnel_bad_since >= TUNNEL_GRACE_SECONDS:
            self.log("the tunnel stayed unreachable; replacing it")
            self._close_tunnel()
            self._start_tunnel()
            self._tunnel_bad_since = None

    # -- keeping the record fresh and rebalancing ----------------------------------------------

    def _maintain(self) -> None:
        last_move_check = last_tunnel_check = time.monotonic()
        while not self._stop.wait(self.resign_s):
            try:
                self.publish()
                now = time.monotonic()
                if now - last_tunnel_check >= TUNNEL_CHECK_SECONDS:
                    last_tunnel_check = now
                    self._check_tunnel()
                if now - last_move_check >= self.rebalance_s:
                    last_move_check = now
                    self._maybe_move()
            except Exception as e:
                self.log(f"maintenance: {type(e).__name__}: {e}")

    def _mover(self, live):
        """(peer, window): the one peer that should move to restore coverage, chosen the same way by every peer.

        Every peer that could spare itself would see the same empty stage and all move at once, emptying
        the stages they left. So only the eligible peer with the lowest id moves; the others wait for the
        next round, when the stage is covered again and they have nothing to do.
        """
        eligible = []
        for r in live:
            if r.get("state") != "up" or r.get("window") is None:
                continue
            new = placement.should_move(self.manifest, live, r["peer"])
            if new is not None:
                eligible.append((r["peer"], new))
        return min(eligible) if eligible else None

    def _maybe_move(self) -> None:
        if self.state != "up" or self.clock() - self._last_move < self.cooldown_s:
            return
        chosen = self._mover(self.table.live())
        if chosen is None or chosen[0] != self.identity.peer or chosen[1] == self.window:
            return
        # Gossip lags: wait a moment, look again, and move only if the same answer still holds.
        if self._stop.wait(min(2.0, self.rebalance_s / 2)):
            return
        again = self._mover(self.table.live())
        if again is not None and again[0] == self.identity.peer and self.state == "up":
            self.move(again[1])

    def move(self, window) -> None:
        """Serve a different window: say `loading`, swap the block, say `up` again at the same address."""
        self.log(f"moving to layers {window[0]}-{window[1]} (a stage lost every peer)")
        with self._lock:
            self.state, self.window = "loading", None
        self.publish()
        self._push_to_peers()
        self._teardown_server()
        self._last_move = self.clock()
        self._serve(tuple(window))
        with self._lock:
            self.window, self.state = tuple(window), "up"
        self.publish()
        self._push_to_peers()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="commonnet.node", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network", metavar="URL|FILE", help=f"a seed peer's URL, or a manifest file (or set {NETWORK_ENV})")
    p.add_argument("--seed", action="append", default=[], metavar="URL", help="another peer to learn from (repeatable)")
    p.add_argument("--port", type=int, default=0, help="loopback port for the block server (default: any free port)")
    p.add_argument("--backend", choices=["mlx", "nf4"], help="override the detected backend")
    p.add_argument("--stage-gib", type=float, help="memory to offer, in GiB (default: detected)")
    p.add_argument("--identity-dir", type=Path, help="where this peer's key lives (default: ~/.common-network/net/<network>)")
    p.add_argument("--stub", action="store_true", help="serve an identity block with no weights (tests)")
    p.add_argument("--no-tunnel", action="store_true", help="do not open a tunnel; publish --advertise or the loopback URL")
    p.add_argument("--advertise", metavar="URL", help="with --no-tunnel: the URL to publish")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.advertise and not args.no_tunnel:
        print("error: --advertise is only for --no-tunnel", file=sys.stderr)
        return 2
    try:
        manifest, seeds = resolve_network(args.network, args.seed)
        if args.stub:
            hardware = Hardware("stub", args.stage_gib or STUB_GIB, "stub")
        else:
            hardware = detect_hardware()
            if args.backend:
                hardware = Hardware(args.backend, hardware.gib, hardware.name)
            if args.stage_gib:
                hardware = Hardware(hardware.backend, args.stage_gib, hardware.name)
        directory = args.identity_dir or HOME / "net" / re.sub(r"[^A-Za-z0-9._-]", "_", manifest.id)
        identity = Identity.load_or_create(Path(directory) / "identity.key")
    except (NodeError, RecordError, ManifestError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    node = Node(manifest, seeds, identity, hardware, port=args.port, stub=args.stub, tunnel=not args.no_tunnel,
                advertise=args.advertise)

    def leave(signum, frame):
        node.request_leave()

    signal.signal(signal.SIGTERM, leave)
    signal.signal(signal.SIGINT, leave)
    try:
        node.start()
        node.wait()
    except NodeError as e:
        print(f"error: {e}", file=sys.stderr)
        node.stop()
        return 1
    finally:
        node.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
