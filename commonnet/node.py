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
from commonnet.mesh.records import LIVE_SECONDS, Identity, RecordError
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
CPU_AVAILABLE_FRACTION = 0.6  # a CPU-only machine offers min(60% of free RAM, 50% of total RAM)
CPU_TOTAL_FRACTION = 0.5
MAX_GOSSIP_RECORDS = 500
STUB_GIB = 100.0

STATUS_ENV = "COMMONNET_STATUS_FILE"
STATUS_SECONDS = 2.0
DEFAULT_API_PORT = 11440


class NodeError(RuntimeError):
    """The node cannot start or continue. The message says what to do."""


@dataclass(frozen=True)
class Hardware:
    backend: str  # "mlx" | "nf4" | "cpu" | "stub"
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


def _memory_bytes(system: str | None = None) -> tuple[int, int] | None:
    """(total, available) RAM in bytes, or None when this OS will not say. Standard library only."""
    system = system or platform.system()
    try:
        if system == "Windows":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("tpage", ctypes.c_ulonglong),
                            ("apage", ctypes.c_ulonglong), ("tvirt", ctypes.c_ulonglong),
                            ("avirt", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong)]
            status = Status()
            status.length = ctypes.sizeof(Status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return int(status.total), int(status.avail)
        if system == "Linux":
            fields = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    key, _, rest = line.partition(":")
                    fields[key] = int(rest.split()[0]) * 1024
            return fields["MemTotal"], fields.get("MemAvailable", fields.get("MemFree", 0))
        if system == "Darwin":
            total = _sysctl_memsize()
            return total, total // 2  # Intel Macs: no cheap free-memory figure; assume half
    except (OSError, ValueError, KeyError, AttributeError, subprocess.SubprocessError):
        return None
    return None


def cpu_gib(total: float, available: float) -> float:
    """What a CPU-only machine offers: a conservative share, so the OS and the user's own work keep room."""
    return round(max(0.0, min(available * CPU_AVAILABLE_FRACTION, total * CPU_TOTAL_FRACTION)) / 2**30, 2)


def cpu_has_fast_bf16() -> bool:
    """Does this CPU multiply bfloat16 in hardware (AVX512-BF16 or AMX)? Without it bfloat16 prefill is ~20x slower than float32."""
    try:
        import torch
        cpu = torch.cpu
        return bool(getattr(cpu, "_is_avx512_bf16_supported", lambda: False)()
                    or getattr(cpu, "_is_amx_tile_supported", lambda: False)())
    except Exception:
        return False


def choose_cpu_dtype(manifest, hw: "Hardware", fast_bf16: bool) -> tuple[str, "Hardware"]:
    """('bfloat16' | 'float32', the Hardware to place with) for a CPU peer.

    Machines without fast bfloat16 compute in float32, which costs twice the memory per layer, so they offer
    half as much (in bfloat16 units) and only switch when the smallest stage still fits that way.
    """
    if fast_bf16:
        return "bfloat16", hw
    smallest = min(manifest.window_gib(a, b, "cpu") for a, b in manifest.stages())
    half = round(hw.gib / 2, 2)
    if half >= smallest:
        return "float32", Hardware(hw.backend, half, hw.name)
    return "bfloat16", hw


def detect_hardware(system: str | None = None, machine: str | None = None, memsize=_sysctl_memsize,
                    nvidia=_nvidia_free_mib, meminfo=_memory_bytes) -> Hardware:
    """Apple silicon -> mlx, NVIDIA -> nf4, anything else -> cpu (a share of RAM, bfloat16 on the CPU)."""
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
    mem = meminfo(system or platform.system())
    if not mem:
        raise NodeError("could not read how much memory this machine has, so it cannot size a CPU window. "
                        "The layer network serves on Apple silicon, NVIDIA, or a CPU with enough RAM.")
    return Hardware("cpu", cpu_gib(*mem), f"CPU ({os.cpu_count() or '?'} cores)")


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


def default_status_path(environ=None) -> Path:
    environ = os.environ if environ is None else environ
    return Path(environ.get(STATUS_ENV) or HOME / "net" / "status.json")


def friendly(message: str) -> str:
    """One sentence for a person: first letter capitalised, no trailing whitespace."""
    message = str(message).strip()
    return message[:1].upper() + message[1:] if message else "Something went wrong."


def blank_status(state: str, **fields) -> dict:
    """A status document with every field of the contract (frozen: the desktop app reads it)."""
    status = {"v": 1, "state": state, "network": "", "model": "", "peer": "", "window": None, "backend": None,
              "gib": 0.0, "url": None, "download": None, "stages": [], "coverage": [], "peers": [], "error": None,
              "api": None, "pid": os.getpid(), "ts": time.time()}
    status.update(fields)
    return status


class StatusFile:
    """Writes the status document atomically (temp file + rename) into a private directory."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, status: dict) -> None:
        data = json.dumps(status, separators=(",", ":")).encode("utf-8")
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                try:
                    os.write(fd, data)
                finally:
                    os.close(fd)
                os.replace(tmp, self.path)
            except OSError:
                pass  # a status file is a convenience; never stop a peer over it


_PROGRESS = re.compile(r"^(?P<label>.+?): (?P<done>\d+) MB of (?P<total>\d+) MB$")
_WRITTEN = re.compile(r"^(?P<label>.+?): (?P<total>\d+) MB written")


class Progress:
    """Adds up the hub's own progress lines ("model.safetensors: 1277 MB of 4394 MB") over every file it fetches."""

    def __init__(self) -> None:
        self._files: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()

    def feed(self, line: str) -> None:
        line = str(line).strip()
        found = _PROGRESS.match(line)
        if found:
            done, total = int(found["done"]), int(found["total"])
        else:
            found = _WRITTEN.match(line)
            if not found:
                return
            done = total = int(found["total"])
        with self._lock:
            self._files[found["label"]] = (done, max(done, total))

    def snapshot(self) -> dict | None:
        with self._lock:
            if not self._files:
                return None
            return {"bytes": sum(d for d, _ in self._files.values()) << 20,
                    "total": sum(t for _, t in self._files.values()) << 20}

    def reset(self) -> None:
        with self._lock:
            self._files.clear()


class Node:
    """One peer. Construct, start(), later stop(). Everything slow or external is injectable for tests."""

    def __init__(self, manifest: Manifest, seeds: list[str], identity: Identity, hardware: Hardware, *,
                 port: int = 0, stub: bool = False, tunnel: bool = True, advertise: str | None = None,
                 log=print, tunnel_opener=None, block_builder=None, fetcher=None,
                 resign_s: float = RESIGN_SECONDS, rebalance_s: float = REBALANCE_SECONDS,
                 cooldown_s: float = MOVE_COOLDOWN_SECONDS, gossip_s: float | None = None, clock=time.time,
                 status: StatusFile | None = None, status_s: float = STATUS_SECONDS,
                 api_port: int | None = None, api_backend=None,
                 fixed_window: tuple[int, int] | None = None):
        self.fixed_window = fixed_window
        self.manifest, self.seeds, self.identity, self.hw = manifest, list(seeds), identity, hardware
        self.cpu_dtype = None
        if hardware.backend == "cpu" and not os.environ.get("COMMONNET_DTYPE"):
            self.cpu_dtype, self.hw = choose_cpu_dtype(manifest, hardware, cpu_has_fast_bf16())
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
        # What the status file says; the signed record keeps its own three states (loading | up | leaving).
        self.phase = "starting"
        self._moving = False
        self.error: str | None = None
        self.progress = Progress()
        self._status, self.status_s = status, status_s
        self._status_window: tuple[int, int] | None = None
        self._api_port, self._api_backend = api_port, api_backend
        self._api: dict | None = None  # {"port": int, "ready": bool} once --api is on
        self._api_server = None
        self._api_error: str | None = None
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

    # -- the status file (read by the desktop app; see docs/layer-network.md) ------------------------

    def set_phase(self, phase: str, error: str | None = None) -> None:
        with self._lock:
            if self._moving and phase in ("downloading", "loading"):
                phase = "moving"  # a move says `moving` throughout, including its download
            self.phase = phase
            if error is not None or phase == "error":
                self.error = error
        self.write_status()

    def fail(self, message: str) -> None:
        """Record a failure the person can read; the status file then says `error` until the process ends."""
        self.set_phase("error", friendly(message))

    def status_snapshot(self) -> dict:
        with self._lock:
            phase, window, url, error = self.phase, self._status_window, self.url, self.error
            api = dict(self._api) if self._api else None
            if api is not None and self._api_error and error is None:
                error = self._api_error
        now = self.clock()
        manifest = self.manifest
        live = self.table.live()
        peers = []
        for r in self.table.export():
            age = now - r.get("ts", now)
            if r.get("state") == "leaving" or age > LIVE_SECONDS:
                continue
            peers.append({"peer": r["peer"], "window": r.get("window"), "backend": r.get("backend"),
                          "state": r.get("state"), "age_s": round(max(0.0, age), 1), "me": r["peer"] == self.identity.peer})
        peers.sort(key=lambda p: (not p["me"], p["window"] or [99, 99], p["peer"]))
        download = self.progress.snapshot() if phase in ("downloading", "moving") else None
        return {"v": 1, "state": phase, "network": manifest.id, "model": manifest.model.get("name", manifest.id),
                "peer": self.identity.peer, "window": None if window is None else list(window),
                "backend": self.hw.backend, "gib": self.hw.gib, "url": url, "download": download,
                "stages": [list(w) for w in manifest.stages()], "coverage": placement.coverage(manifest, live),
                "peers": peers, "error": error, "api": api, "pid": os.getpid(), "ts": time.time()}

    def write_status(self) -> None:
        if self._status is None:
            return
        try:
            self._status.write(self.status_snapshot())
        except Exception:  # never let a status write take a peer down
            pass

    def _status_loop(self) -> None:
        while not self._stop.wait(self.status_s):
            self.write_status()

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
        if self._api_port is not None:
            self._api = {"port": self._api_port, "ready": False}
        if self._status is not None:
            self.write_status()
            self._spawn(self._status_loop, "node-status")
        self._learn()
        self._gossiper = Gossiper(self.table, self.manifest, self.record, self.seeds)
        if self.gossip_s is not None:
            self._gossiper.interval = (self.gossip_s, self.gossip_s)
        self.publish()
        self._gossiper.start()
        self._spawn(self._maintain, "node-maintain")
        window = self._choose()
        with self._lock:
            self._status_window = window
        self.log(f"serving layers {window[0]}-{window[1]} as {self.identity.peer}")
        self._serve(window)
        with self._lock:
            self.window, self.state, self.phase = window, "up", "up"
        self.publish()
        self._push_to_peers()
        self.write_status()
        self.log(f"up at {self.url}")
        self.log(f"you are part of the network. A friend can join through you with:  common join --network {self.url}")
        if self._api_port is not None:
            self._start_api()
        return self

    def stop(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._leave.set()
        with self._lock:
            self.state = "leaving"
            if self.phase != "error":
                self.phase = "leaving"
        self.write_status()
        try:
            self.publish()
            self._push_to_peers()
        except Exception as e:  # leaving is best effort
            self.log(f"could not announce leaving ({type(e).__name__})")
        if self._gossiper is not None:
            self._gossiper.stop()
        self._teardown_api()
        self._teardown_server()
        self._close_tunnel()
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(timeout=5)
        with self._lock:
            if self.phase != "error":
                self.phase = "stopped"
            self.url = None
        self.write_status()

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
        if self.fixed_window is not None:
            window = tuple(self.fixed_window)
            if not self.manifest.is_window(*window):
                stages = ", ".join(f"{a}-{b}" for a, b in self.manifest.stages())
                raise NodeError(f"--layers {window[0]}-{window[1]} does not line up with this network's stages "
                                f"({stages}); a window starts and ends on a stage boundary.")
            need = self.manifest.window_gib(*window, self.hw.backend)
            if need > self.hw.gib:
                raise NodeError(f"layers {window[0]}-{window[1]} need {need:.1f} GiB here and this machine offers "
                                f"{self.hw.gib:.1f} GiB. Pick a shorter window.")
            return window
        try:
            return placement.choose_window(self.manifest, self.table.live(), self.hw.gib, self.hw.backend,
                                           me=self.identity.peer)
        except placement.NoWindow as e:
            raise NodeError(f"{e}. This network needs at least "
                            f"{self.manifest.window_gib(*self.manifest.stages()[0], self.hw.backend):.1f} GiB free.") from None

    def _make_block(self, window):
        if self._build_block is not None:
            self.set_phase("loading")
            return self._build_block(window, self.manifest, self.hw)
        if self.stub:
            self.set_phase("loading")
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
            fetch = lambda r, w: hub.Hub(r, log=self._hub_log).fetch(layers=tuple(w))  # noqa: E731
        self.log(f"downloading layers {window[0]}-{window[1]} of {repo} (only these)...")
        self.progress.reset()
        self.set_phase("downloading")
        try:
            checkpoint = fetch(repo, window)
        except Exception as e:
            raise NodeError(f"the download failed ({e}); join again to resume") from None
        self.set_phase("loading")
        # Qwen activations pass float16's range on some prompts (NaN logits), so the MLX block computes in
        # bfloat16 and the hidden state between peers is float32. An explicit COMMONNET_DTYPE wins, then the
        # manifest's per-backend dtype. COMMONNET_MLX_CACHE_MB is left to the environment untouched.
        dtype = (self.manifest.model.get("dtype") or {}).get(self.hw.backend)
        if self.hw.backend in ("mlx", "cpu"):
            dtype = self.cpu_dtype or dtype or "bfloat16"
        if dtype and not os.environ.get("COMMONNET_DTYPE"):
            os.environ["COMMONNET_DTYPE"] = dtype
        from commonnet.net import factory
        return factory.build_block(window[0], window[1], backend=self.hw.backend, checkpoint=checkpoint)

    def _hub_log(self, message: str) -> None:
        self.progress.feed(message)
        self.log(message)

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


    # -- the local chat API (--api): the OpenAI-compatible server, in this process --------------------

    def localize(self, transport) -> None:
        """Send requests for this node's own window to its loopback block server, not out through the tunnel."""
        with self._lock:
            own, local = self.url, (self.server.url if self.server is not None else None)
        if not own or not local:
            return

        def swap(entry):
            if isinstance(entry, tuple):
                return (local, entry[1]) if entry[0] == own else entry
            return local if entry == own else entry

        transport.urls = {name: swap(url) for name, url in transport.urls.items()}
        transport._replicas = {name: [swap(e) for e in entries] for name, entries in transport._replicas.items()}

    def _default_api_backend(self, network):
        import torch
        from commonnet.serve import NetworkBackend
        checkpoint = os.environ.get("COMMONNET_LOCAL_CHECKPOINT")
        if self.hw.backend == "mlx":
            fmt = "mlx4"  # the manifest's client_repo is the 4-bit MLX one
        else:
            # mlx4 needs Apple silicon, and the 4-bit client repo cannot be read as plain floats (its embedding is
            # packed 512 wide, not 4096). Other peers take the embedding, norm and head from their own checkpoint
            # repo, in fp16 on the CPU.
            fmt = "fp16"
            repo = self.manifest.repo(self.hw.backend)
            if checkpoint is None and repo:
                from commonnet import hub
                checkpoint = hub.Hub(repo).fetch(client=True)
        return NetworkBackend(network, checkpoint, fmt=fmt, hidden_dtype=torch.float32)

    def _start_api(self) -> None:
        from commonnet import serve
        from commonnet.netclient import Network

        node = self

        class NodeNetwork(Network):
            def transport(self, **kw):
                transport = super().transport(**kw)
                node.localize(transport)
                return transport

        # The node's own gossip keeps its table fresh, so the client side never runs a second exchange loop.
        network = NodeNetwork(seeds=self.seeds, manifest=self.manifest, table=self.table,
                              exchange=lambda table, url, timeout: None, now=self.clock)
        lazy = serve.LazyBackend(network)
        try:
            server = serve.make_server(lazy, self._api_port)
        except OSError as e:
            self._api_error = friendly(f"the chat API could not start on port {self._api_port} ({e.strerror or e})")
            self.log(self._api_error)
            self.write_status()
            return
        with self._lock:
            self._api_server = server
        self._spawn(server.serve_forever, "node-api")
        self.log(f"chat API on http://127.0.0.1:{self._api_port}/v1 (loading the client part in the background)")
        self._spawn(lambda: self._load_api_backend(lazy, network), "node-api-load")
        self.write_status()

    def _load_api_backend(self, lazy, network) -> None:
        try:
            backend = (self._api_backend or self._default_api_backend)(network)
        except (Exception, SystemExit) as e:  # NetworkBackend exits (not raises) when the manifest names no client repo
            lazy.fail(friendly(f"the chat API could not load its model part ({type(e).__name__}: {e})"))
            self._api_error = lazy.error
            self.log(lazy.error)
        else:
            lazy.set(backend)
            with self._lock:
                if self._api is not None:
                    self._api["ready"] = True
            self.log("chat API ready")
        self.write_status()

    def _teardown_api(self) -> None:
        with self._lock:
            server, self._api_server = self._api_server, None
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
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
        if self.fixed_window is not None:
            return  # pinned with --layers: this peer stays where it was told to
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
            self.state, self.window, self._status_window = "loading", None, tuple(window)
            self.phase, self._moving = "moving", True
        self.write_status()
        self.publish()
        self._push_to_peers()
        self._teardown_server()
        self._last_move = self.clock()
        self.progress.reset()
        try:
            self._serve(tuple(window))
        finally:
            self._moving = False
        with self._lock:
            self.window, self.state, self.phase = tuple(window), "up", "up"
        self.publish()
        self._push_to_peers()
        self.write_status()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="commonnet.node", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network", metavar="URL|FILE", help=f"a seed peer's URL, or a manifest file (or set {NETWORK_ENV})")
    p.add_argument("--seed", action="append", default=[], metavar="URL", help="another peer to learn from (repeatable)")
    p.add_argument("--port", type=int, default=0, help="loopback port for the block server (default: any free port)")
    p.add_argument("--backend", choices=["mlx", "nf4", "cpu"], help="override the detected backend")
    p.add_argument("--stage-gib", type=float, help="memory to offer, in GiB (default: detected)")
    p.add_argument("--layers", metavar="A-B", help="serve exactly this window of layers (for example 27-36) instead of the one the network picks; it must start and end on a stage boundary")
    p.add_argument("--identity-dir", type=Path, help="where this peer's key lives (default: ~/.common-network/net/<network>)")
    p.add_argument("--stub", action="store_true", help="serve an identity block with no weights (tests)")
    p.add_argument("--no-tunnel", action="store_true", help="do not open a tunnel; publish --advertise or the loopback URL")
    p.add_argument("--advertise", metavar="URL", help="with --no-tunnel: the URL to publish")
    p.add_argument("--status-file", type=Path, metavar="PATH",
                   help=f"where to write the status JSON the desktop app reads every 2 s (or set {STATUS_ENV}; "
                        "default: ~/.common-network/net/status.json)")
    p.add_argument("--api", nargs="?", const=DEFAULT_API_PORT, default=None, type=int, metavar="PORT",
                   help=f"also serve the OpenAI-compatible chat API on 127.0.0.1:PORT (default {DEFAULT_API_PORT}) "
                        "from this process. The client part of the model (about 1 GB for Qwen3-8B) loads in the background")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.advertise and not args.no_tunnel:
        print("error: --advertise is only for --no-tunnel", file=sys.stderr)
        return 2
    status = StatusFile(args.status_file or default_status_path())
    status.write(blank_status("starting"))
    try:
        manifest, seeds = resolve_network(args.network, args.seed)
        status.write(blank_status("starting", network=manifest.id, model=manifest.model.get("name", manifest.id)))
        if args.stub:
            hardware = Hardware("stub", args.stage_gib or STUB_GIB, "stub")
        else:
            hardware = detect_hardware()
            if args.backend:
                hardware = Hardware(args.backend, hardware.gib, hardware.name)
            if args.stage_gib:
                hardware = Hardware(hardware.backend, args.stage_gib, hardware.name)
        fixed_window = None
        if args.layers:
            m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", args.layers)
            if not m:
                raise NodeError(f"--layers wants a window like 27-36, got {args.layers!r}")
            fixed_window = (int(m.group(1)), int(m.group(2)))
        directory = args.identity_dir or HOME / "net" / re.sub(r"[^A-Za-z0-9._-]", "_", manifest.id)
        identity = Identity.load_or_create(Path(directory) / "identity.key")
    except (NodeError, RecordError, ManifestError) as e:
        print(f"error: {e}", file=sys.stderr)
        status.write(blank_status("error", error=friendly(e)))
        return 2
    node = Node(manifest, seeds, identity, hardware, port=args.port, stub=args.stub, tunnel=not args.no_tunnel,
                advertise=args.advertise, status=status, api_port=args.api, fixed_window=fixed_window)

    def leave(signum, frame):
        node.request_leave()

    signal.signal(signal.SIGTERM, leave)
    signal.signal(signal.SIGINT, leave)
    try:
        node.start()
        node.wait()
    except NodeError as e:
        print(f"error: {e}", file=sys.stderr)
        node.fail(str(e))
        node.stop()
        return 1
    finally:
        node.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
