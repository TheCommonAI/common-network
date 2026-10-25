"""Automatic ICE/STUN peer links over authenticated WebRTC data channels.

HTTPS exchanges the offer and answer; DTLS fingerprints in those descriptions
bind the direct connection to that existing authenticated signaling path. No
model traffic is relayed by this module. If ICE cannot connect, callers keep
using their existing HTTPS transport.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import importlib.util
import ipaddress
import json
import math
import re
import os
import secrets
import socket
import struct
import threading
import time

from commonnet.net.protocol import MAX_REQUEST_BYTES

CAPABILITY = "rtc_v1"
PRIVATE_CAPABILITY = "rtc_private_v1"
DEFAULT_STUN_URLS = ("stun:stun.l.google.com:19302", "stun:stun.cloudflare.com:3478",
                     "stun:stun1.l.google.com:19302", "stun:global.stun.twilio.com:3478")
STUN_PROBE_SECONDS = 1.5
STUN_CACHE_SECONDS = 300.0
SIGNAL_TIMEOUT = 30.0
OFFER_TIMEOUT = 30.0
CONNECT_TIMEOUT = 30.0
MAX_SDP_BYTES = 64 * 1024
MAX_CANDIDATES = 32
MAX_MESSAGE_BYTES = MAX_REQUEST_BYTES + 64 * 1024
FRAME_BYTES = 16 * 1024
MAX_BUFFERED_BYTES = 512 * 1024
MAX_REASSEMBLY_BYTES = 32 * 1024 * 1024
ASSEMBLY_TIMEOUT = 15.0
UNOPENED_TIMEOUT = 30.0
_HEADER = struct.Struct("!4sQII")
_MAGIC = b"CNR1"
_POST_PATHS = frozenset(("/session", "/forward", "/chain", "/close", "/truncate"))


class RtcUnavailable(RuntimeError):
    """The optional peer link is absent, failed, or exceeded a bounded wait."""


class RtcProtocolError(ValueError):
    """A malformed or oversized signaling/RPC message."""


def available() -> bool:
    """Do not import optional native dependencies on an older installation."""
    try:
        return importlib.util.find_spec("aiortc") is not None
    except (ImportError, ValueError):
        return False


def _description(value, expected_type, allow_private=False):
    """Filter candidate addresses before aiortc can resolve or contact them.

    Only literal UDP destinations are allowed. Hostnames, multicast, unspecified,
    reserved and non-public candidates cannot be supplied by a remote caller.
    Explicit private mode exists for local tests and advertised local networks.
    """
    if not isinstance(value, dict) or set(value) != {"type", "sdp"} or value["type"] != expected_type:
        raise RtcProtocolError("invalid RTC description")
    sdp = value["sdp"]
    if not isinstance(sdp, str) or len(sdp.encode("utf-8")) > MAX_SDP_BYTES or "\x00" in sdp:
        raise RtcProtocolError("invalid RTC description size")
    lines = sdp.replace("\r\n", "\n").splitlines()
    media = [line for line in lines if line.startswith("m=")]
    if len(media) != 1 or not re.fullmatch(r"m=application \d+ UDP/DTLS/SCTP webrtc-datachannel", media[0]):
        raise RtcProtocolError("RTC permits a single data channel media section")
    fingerprints = [line for line in lines if line.startswith("a=fingerprint:")]
    sha256 = re.compile(r"a=fingerprint:sha-256 (?:[0-9A-Fa-f]{2}:){31}[0-9A-Fa-f]{2}")
    if not any(sha256.fullmatch(line) for line in fingerprints):
        raise RtcProtocolError("RTC requires a SHA-256 DTLS fingerprint")
    result, candidates, kept = [], 0, 0
    for line in lines:
        if line.startswith("a=candidate:"):
            candidates += 1
            if candidates > MAX_CANDIDATES:
                raise RtcProtocolError("too many RTC candidates")
            fields = line.split()
            if len(fields) < 8 or fields[2].lower() != "udp" or fields[6] != "typ":
                continue
            try:
                address = ipaddress.ip_address(fields[4])
                port = int(fields[5])
                component = int(fields[1])
            except ValueError:
                continue
            if ("%" in fields[4] or not 1 <= port <= 65535 or component != 1
                    or address.is_unspecified or address.is_multicast):
                continue
            if not allow_private and (not address.is_global or address.is_reserved
                                      or getattr(address, "is_site_local", False)):
                continue
            kept += 1
        result.append(line)
    if not kept:
        raise RtcUnavailable("no usable public ICE candidates (STUN unreachable over UDP, or only private "
                             "addresses)")
    return {"type": expected_type, "sdp": "\r\n".join(result) + "\r\n"}


_stun_cache: dict = {}
_stun_lock = threading.Lock()


def _stun_probe(url, timeout=STUN_PROBE_SECONDS):
    """Does this STUN server answer a binding request from here? (aioice uses only ONE server.)"""
    host, _, port = url[len("stun:"):].partition("?")[0].partition(":")
    try:
        infos = socket.getaddrinfo(host, int(port or 3478), 0, socket.SOCK_DGRAM)
        family, _, _, _, addr = infos[0]
        with socket.socket(family, socket.SOCK_DGRAM) as sock:
            sock.settimeout(timeout)
            tid = os.urandom(12)
            sock.sendto(struct.pack("!HHI", 1, 0, 0x2112A442) + tid, addr)
            data = sock.recv(2048)
        return len(data) >= 20 and data[8:20] == tid
    except (OSError, ValueError, UnicodeError):
        return False


def order_stun(urls, probe=_stun_probe, now=time.monotonic):
    """(urls reordered so a reachable server comes first, True if any server answered).

    aiortc/aioice honour only the first STUN server in the configuration, so a blocked
    first choice would otherwise mean no server-reflexive candidate and a silent failure.
    """
    urls = tuple(urls)
    if not urls:
        return urls, False
    key = urls
    with _stun_lock:
        hit = _stun_cache.get(key)
        if hit is not None and now() - hit[0] < STUN_CACHE_SECONDS:
            return hit[1], hit[2]
    results = {}
    threads = [threading.Thread(target=lambda u=u: results.__setitem__(u, probe(u)), daemon=True) for u in urls]
    for t in threads:
        t.start()
    for t in threads:
        t.join(STUN_PROBE_SECONDS + 1)
    good = [u for u in urls if results.get(u)]
    ordered = tuple(good + [u for u in urls if u not in good])
    with _stun_lock:
        _stun_cache[key] = (now(), ordered, bool(good))
    return ordered, bool(good)


def _config(stun_urls, probe=True):
    from aiortc import RTCConfiguration, RTCIceServer
    urls = DEFAULT_STUN_URLS if stun_urls is None else tuple(stun_urls)
    if len(urls) > 4 or any(not isinstance(u, str) or len(u) > 256 or not u.startswith("stun:") for u in urls):
        raise RtcProtocolError("RTC STUN configuration is invalid")
    if probe and stun_urls is None:
        urls, _ = order_stun(urls)
    return RTCConfiguration(iceServers=[RTCIceServer(urls=u) for u in urls])


def join_summary(enabled: bool = True) -> str:
    """One log line for join time: can this machine make direct (webrtc) links, and with what STUN."""
    if not enabled:
        return "direct (webrtc): off (--no-rtc); peers are reached over tunnel/https"
    reason = describe_unavailable()
    if reason:
        return f"direct (webrtc): UNAVAILABLE, peers are reached over tunnel/https only: {reason}"
    import aiortc
    ordered, reachable = order_stun(DEFAULT_STUN_URLS)
    if not reachable:
        return (f"direct (webrtc): aiortc {getattr(aiortc, '__version__', '?')} present but no STUN server "
                "answered over UDP; direct links will likely fail (UDP blocked?) and fall back to tunnel/https")
    return (f"direct (webrtc): ready (aiortc {getattr(aiortc, '__version__', '?')}, STUN {ordered[0][5:]}); "
            "links are negotiated on first use, no TURN relay")


def describe_unavailable() -> str | None:
    """Why this machine cannot make direct links, or None when it can (short, no secrets)."""
    if not available():
        return "aiortc is not installed in this Python (update the runtime: pip install aiortc==1.15.0)"
    try:
        import aiortc  # noqa: F401
    except Exception as e:  # a native dependency (av, cryptography, pylibsrtp) failed to load
        return f"aiortc failed to import: {type(e).__name__}: {str(e)[:100]}"
    return None


def _encode(value):
    try:
        raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        raise RtcProtocolError("RPC message must be finite JSON") from None
    if len(raw) > MAX_MESSAGE_BYTES:
        raise RtcProtocolError("RTC RPC exceeds its size limit")
    return raw


def _request(method, path, body):
    if (method == "GET" and path in ("/health", "/health?caps=1") and body is None):
        return {"method": method, "path": path, "body": None}
    if method == "POST" and path in _POST_PATHS and isinstance(body, dict):
        return {"method": method, "path": path, "body": body}
    raise RtcProtocolError("RTC RPC path or method is not permitted")


def _response(value):
    if not isinstance(value, dict) or set(value) != {"status", "body", "server_ms"}:
        raise RtcProtocolError("invalid RTC response")
    status, duration = value["status"], value["server_ms"]
    if type(status) is not int or not 100 <= status <= 599 or not isinstance(value["body"], dict):
        raise RtcProtocolError("invalid RTC response")
    if duration is not None and (type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0):
        raise RtcProtocolError("invalid RTC timing")
    return value


class _Runtime:
    """One event loop for all optional ICE links in this process."""
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.reassembly_bytes = 0
        self.thread = threading.Thread(target=self.loop.run_forever, name="commonnet-rtc", daemon=True)
        self.thread.start()

    def run(self, coro, timeout):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise RtcUnavailable("RTC operation timed out") from None


_runtime = None
_runtime_lock = threading.Lock()


def _get_runtime():
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = _Runtime()
        return _runtime


class _Link:
    def __init__(self, pc, channel, handler=None):
        self.pc, self.channel, self.handler = pc, channel, handler
        self.runtime = _get_runtime()
        self.opened = asyncio.Event()
        self.closed = False
        self.assemblies = {}
        self.pending = {}
        self.send_lock = asyncio.Lock()
        self.request_lock = asyncio.Lock()
        self.serving = False
        channel.bufferedAmountLowThreshold = MAX_BUFFERED_BYTES // 2
        self.drained = asyncio.Event()
        self.drained.set()
        channel.on("bufferedamountlow", self.drained.set)
        channel.on("open", self.opened.set)
        channel.on("close", self.close)
        channel.on("message", self.receive)
        if channel.readyState == "open":
            self.opened.set()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.opened.set()
        self.drained.set()
        for future in self.pending.values():
            if not future.done():
                future.set_exception(RtcUnavailable("RTC channel closed"))
        for state in self.assemblies.values():
            state[2].cancel()
            self.runtime.reassembly_bytes -= len(state[1])
        self.assemblies.clear()

    def _fault(self):
        if self.closed:
            return
        self.close()
        asyncio.create_task(self.pc.close())

    def _expire(self, key):
        if key in self.assemblies:
            self._fault()

    def receive(self, frame):
        try:
            if self.closed or not isinstance(frame, bytes) or not _HEADER.size < len(frame) <= _HEADER.size + FRAME_BYTES:
                raise RtcProtocolError("invalid RTC frame")
            magic, key, total, offset = _HEADER.unpack_from(frame)
            data = frame[_HEADER.size:]
            if magic != _MAGIC or not 1 <= total <= MAX_MESSAGE_BYTES or offset + len(data) > total:
                raise RtcProtocolError("invalid RTC frame bounds")
            state = self.assemblies.get(key)
            if state is None:
                if offset != 0 or len(self.assemblies) >= 2:
                    raise RtcProtocolError("unexpected RTC frame")
                timer = self.runtime.loop.call_later(ASSEMBLY_TIMEOUT, self._expire, key)
                state = self.assemblies[key] = (total, bytearray(), timer)
            if state[0] != total or len(state[1]) != offset:
                raise RtcProtocolError("out of order RTC frame")
            if self.runtime.reassembly_bytes + len(data) > MAX_REASSEMBLY_BYTES:
                raise RtcProtocolError("RTC assembly budget exhausted")
            state[1].extend(data)
            self.runtime.reassembly_bytes += len(data)
            if len(state[1]) == total:
                del self.assemblies[key]
                state[2].cancel()
                self.runtime.reassembly_bytes -= total
                obj = json.loads(state[1].decode("utf-8"))
                if self.handler is None:
                    future = self.pending.get(key)
                    if future is None or future.done():
                        raise RtcProtocolError("unexpected RTC response")
                    future.set_result(_response(obj))
                else:
                    if self.serving:
                        raise RtcProtocolError("too many concurrent RTC requests")
                    if not isinstance(obj, dict) or set(obj) != {"method", "path", "body"}:
                        raise RtcProtocolError("invalid RTC request")
                    _request(obj["method"], obj["path"], obj["body"])
                    self.serving = True
                    asyncio.create_task(self.serve(key, obj))
        except (ValueError, TypeError, UnicodeError, struct.error, RecursionError):
            self._fault()

    async def send(self, key, obj):
        raw = _encode(obj)
        async with self.send_lock:
            for offset in range(0, len(raw), FRAME_BYTES):
                if self.closed or self.channel.readyState != "open":
                    raise RtcUnavailable("RTC channel is not open")
                while self.channel.bufferedAmount > MAX_BUFFERED_BYTES:
                    self.drained.clear()
                    if self.channel.bufferedAmount <= MAX_BUFFERED_BYTES:
                        break
                    await asyncio.wait_for(self.drained.wait(), 10)
                    if self.closed:
                        raise RtcUnavailable("RTC channel closed")
                frame = _HEADER.pack(_MAGIC, key, len(raw), offset) + raw[offset:offset + FRAME_BYTES]
                self.channel.send(frame)

    async def serve(self, key, obj):
        try:
            try:
                response = await asyncio.to_thread(self.handler, obj["method"], obj["path"], obj["body"])
                response = _response(response)
            except Exception:
                response = {"status": 500, "body": {"ok": False, "error": "RTC handler failed"}, "server_ms": None}
            await self.send(key, response)
        except Exception:
            self._fault()
        finally:
            self.serving = False

    async def request(self, method, path, body, timeout):
        obj = _request(method, path, body)
        async with self.request_lock:
            if self.closed or self.channel.readyState != "open":
                raise RtcUnavailable("RTC channel is not open")
            key = secrets.randbits(64)
            future = self.pending[key] = self.runtime.loop.create_future()
            try:
                await self.send(key, obj)
                return await asyncio.wait_for(future, timeout)
            except BaseException:
                # A timed-out request must not share its channel with later work:
                # its late reply cannot be mistaken for a fresh RPC response.
                self._fault()
                raise
            finally:
                self.pending.pop(key, None)


class RtcServer:
    def __init__(self, handler, *, allow_private=False, stun_urls=None, max_connections=16):
        if type(max_connections) is not int or not 1 <= max_connections <= 64:
            raise ValueError("RTC connection limit must be between 1 and 64")
        self.handler, self.allow_private, self.stun_urls = handler, bool(allow_private), stun_urls
        self.max_connections = max_connections
        self._pcs = set()
        self._links = {}
        self._closed = False

    def offer(self, offer, timeout=OFFER_TIMEOUT):
        checked = _description(offer, "offer", self.allow_private)
        if self._closed or not available():
            raise RtcUnavailable("RTC is unavailable")
        try:
            return _get_runtime().run(self._offer(checked), timeout)
        except (RtcUnavailable, RtcProtocolError):
            raise
        except Exception as e:
            raise RtcUnavailable(f"RTC offer failed: {type(e).__name__}") from None

    async def _offer(self, offer):
        from aiortc import RTCPeerConnection, RTCSessionDescription
        if self._closed or len(self._pcs) >= self.max_connections:
            raise RtcUnavailable("RTC connection limit reached")
        pc = RTCPeerConnection(await asyncio.to_thread(_config, self.stun_urls))
        self._pcs.add(pc)

        @pc.on("datachannel")
        def on_channel(channel):
            if channel.label != "commonnet-rpc-v1" or pc in self._links or not channel.ordered or channel.maxRetransmits is not None or channel.maxPacketLifeTime is not None:
                asyncio.create_task(pc.close())
                return
            self._links[pc] = _Link(pc, channel, self.handler)

        @pc.on("connectionstatechange")
        async def on_state():
            if pc.connectionState in ("failed", "closed"):
                link = self._links.pop(pc, None)
                if link:
                    link.close()
                self._pcs.discard(pc)
                if pc.connectionState == "failed":
                    await pc.close()

        async def expire_unopened():
            await asyncio.sleep(UNOPENED_TIMEOUT)
            if pc.connectionState != "connected":
                await pc.close()

        asyncio.create_task(expire_unopened())
        try:
            await pc.setRemoteDescription(RTCSessionDescription(**offer))
            await pc.setLocalDescription(await pc.createAnswer())
            return _description({"type": "answer", "sdp": pc.localDescription.sdp}, "answer", self.allow_private)
        except BaseException:
            await pc.close()
            self._pcs.discard(pc)
            raise

    def close(self):
        self._closed = True
        if _runtime is not None:
            _runtime.run(self._close(), 10)

    async def _close(self):
        await asyncio.gather(*(pc.close() for pc in list(self._pcs)), return_exceptions=True)
        for link in self._links.values():
            link.close()
        self._links.clear()
        self._pcs.clear()


class RtcClient:
    def __init__(self, signaler, *, allow_private=False, stun_urls=None):
        self.signaler, self.allow_private, self.stun_urls = signaler, bool(allow_private), stun_urls
        self._pc = self._link = None
        self._closed = False
        self._connect_lock = threading.Lock()

    @property
    def ready(self):
        return bool(not self._closed and self._link is not None and not self._link.closed
                    and self._pc.connectionState == "connected" and self._link.channel.readyState == "open")

    @property
    def connection_info(self):
        return {"state": self._pc.connectionState if self._pc is not None else "new",
                "ice_state": self._pc.iceConnectionState if self._pc is not None else "new",
                "candidate_policy": "private-test" if self.allow_private else "public"}

    def connect(self, timeout=CONNECT_TIMEOUT):
        if self._closed or not available():
            raise RtcUnavailable("RTC is unavailable")
        with self._connect_lock:
            if self.ready:
                return
            try:
                # The budget is for the channel to open; ICE gathering and signaling come first.
                _get_runtime().run(self._connect(timeout), timeout + SIGNAL_TIMEOUT + 15)
            except (RtcUnavailable, RtcProtocolError):
                raise
            except Exception as e:
                raise RtcUnavailable(f"RTC connection failed: {type(e).__name__}") from None

    async def _connect(self, timeout):
        from aiortc import RTCPeerConnection, RTCSessionDescription
        if self._closed:
            raise RtcUnavailable("RTC client is closed")
        if self._pc is not None:
            await self._pc.close()
        pc = self._pc = RTCPeerConnection(await asyncio.to_thread(_config, self.stun_urls))
        self._link = link = _Link(pc, pc.createDataChannel("commonnet-rpc-v1", ordered=True))

        @pc.on("connectionstatechange")
        async def on_state():
            if pc.connectionState in ("failed", "closed"):
                link.close()

        try:
            await pc.setLocalDescription(await pc.createOffer())
            offer = _description({"type": "offer", "sdp": pc.localDescription.sdp}, "offer", self.allow_private)
            answer = await asyncio.to_thread(self.signaler, offer)
            answer = _description(answer, "answer", self.allow_private)
            await pc.setRemoteDescription(RTCSessionDescription(**answer))
            try:
                await asyncio.wait_for(link.opened.wait(), timeout)
            except asyncio.TimeoutError:
                raise RtcUnavailable(f"ICE/DTLS did not connect within {timeout:g}s (ice={pc.iceConnectionState}, "
                                     f"conn={pc.connectionState}); UDP blocked or NAT too strict, no TURN") from None
            if self._closed or link.closed or not self.ready:
                raise RtcUnavailable(f"RTC data channel did not open (ice={pc.iceConnectionState}, "
                                     f"conn={pc.connectionState})")
        except BaseException:
            await pc.close()
            link.close()
            raise

    def request(self, method, path, body, timeout=60):
        if not self.ready:
            raise RtcUnavailable("RTC is not connected")
        try:
            return _get_runtime().run(self._link.request(method, path, body, timeout), timeout + 1)
        except (RtcUnavailable, RtcProtocolError):
            raise
        except Exception:
            raise RtcUnavailable("RTC request failed") from None

    def close(self):
        self._closed = True
        if _runtime is not None and self._pc is not None:
            _runtime.run(self._close(), 10)

    async def _close(self):
        if self._link is not None:
            self._link.close()
        if self._pc is not None:
            await self._pc.close()
