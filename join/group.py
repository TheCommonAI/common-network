#!/usr/bin/env python3
"""Distributed model groups: enrol a member machine and link its RPC worker.

A group's coordinator runs llama.cpp with RPC peers. Each peer is a
ggml-rpc-server on a member's machine, and llama.cpp speaks RPC to it in the
clear with no authentication of its own. Members are not asked to open ports,
forward SSH, know a LAN address or install anything extra. The rules that make
that safe:

- A member never listens on the network. Its worker agent dials out to the
  coordinator over a WebSocket (wss://) and keeps that connection open. The
  ggml-rpc-server it starts is bound to 127.0.0.1, and the agent relays bytes
  between that server and the coordinator over the outbound connection.
- The network is GPU-only. The ggml-rpc-server is started on one GPU device,
  chosen with -d from the device list it reports. A machine with no GPU device
  refuses to join, before it spends its token, and never falls back to CPU.
- The coordinator runs the relay as a server on 127.0.0.1 only, and points a
  Cloudflare quick tunnel at it: the same tunnel the worker uses (join.py's
  start_tunnel). Cloudflare terminates TLS with a publicly trusted certificate,
  so members verify it the ordinary way. Nothing here is self-signed or pinned.
- The coordinator gives each admitted member its own loopback port,
  127.0.0.1:<rpc_port>. That is the rpc_servers entry the distributed config
  lists, so the runtime and its config do not change. Only processes on the
  coordinator can reach those ports.

Admission is by one-time token. The coordinator issues a token, HMAC-signed with
the group secret, naming the group, the member, the tunnel URL the relay is
reachable at, a one-time secret and an expiry. The coordinator spends the
one-time secret on enrolment and returns a member key over the WebSocket. It
keeps only the SHA-256 of the key, and each later connection presents the key.
A token is only good while the relay that issued it is running on the same
tunnel URL; tokens name that URL, so an old token does not reach a new tunnel.

Rejoin: when the relay's tunnel URL changes, an enrolled member does not need a
new token. It rejoins with its member key from member.json, at the relay's
current address (`join.py --group-token TOKEN --relay URL`). The coordinator
checks the key against the stored SHA-256 verifier, and the member keeps its name
and rpc port. A one-time token is never accepted as a rejoin credential: a
request that carries one is refused, whether or not it is expired or used.

Trust: the tunnel is operated by Cloudflare, so Cloudflare can read the
enrolment exchange and the RPC bytes. The coordinator's checks (the token
signature, the one-time secret and the member key) decide who is admitted; the
tunnel only carries them.

Stdlib only, like the rest of the CLI.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import json
import os
import platform
import re
import secrets
import signal
import socket
import ssl
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

COMMON_DIR = Path.home() / ".common-network"
GROUPS_DIR = COMMON_DIR / "groups"

TOKEN_PREFIX = "cng1."
TOKEN_VERSION = 2
DEFAULT_TOKEN_HOURS = 24

# Member RPC ports on the coordinator. Kept clear of llama-server's own port and
# the RPC workers the manual setup uses (50051, 50052).
RPC_PORT_RANGE = range(52100, 52200)

NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")

# Where the relay answers on the tunnel. LINK_PATH upgrades to a WebSocket;
# HEALTH_PATH is what the tunnel's health check reads.
LINK_PATH = "/group/link"
HEALTH_PATH = "/group/health"
LOOPBACK_HOSTS = frozenset(("127.0.0.1", "localhost", "::1"))
RELAY_HEALTH_CHECK_SECONDS = 120
# A fresh quick tunnel's hostname can take minutes to resolve, and until it does a
# health check fails just as it would for a dead tunnel. The tunnel is replaced only
# once it has been unreachable for this long in a row, counted from its start.
RELAY_TUNNEL_GRACE_SECONDS = 600

# Frames: one kind byte, a 32-bit stream id and a 32-bit payload length.
OPEN, DATA, CLOSE, PING, PONG = 1, 2, 3, 4, 5
KINDS = frozenset((OPEN, DATA, CLOSE, PING, PONG))
FRAME_HEADER = struct.Struct("!BII")
MAX_FRAME_PAYLOAD = 1 << 20
DATA_CHUNK = 64 * 1024
PING_INTERVAL_SECONDS = 15.0
IDLE_TIMEOUT_SECONDS = 45.0
IO_TICK_SECONDS = 0.1
HANDSHAKE_TIMEOUT_SECONDS = 10.0
MAX_HANDSHAKE_BYTES = 4096
MAX_HTTP_HEAD_BYTES = 16384
RECONNECT_MAX_SECONDS = 60.0
RPC_START_TIMEOUT_SECONDS = 30.0

# WebSocket (RFC 6455): the parts this link uses.
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
WS_OP_CONT, WS_OP_TEXT, WS_OP_BINARY = 0, 1, 2
WS_OP_CLOSE, WS_OP_PING, WS_OP_PONG = 8, 9, 10
WS_MAX_PAYLOAD = 4 << 20
WS_MAX_PENDING = 4 << 20


class GroupError(Exception):
    """A problem a person can act on: a bad token, a refused enrolment, a missing file."""


class FrameError(GroupError):
    """The peer sent bytes that are not a valid frame. The link is closed."""


# --- small helpers -----------------------------------------------------------

def _check_name(value: str, what: str) -> str:
    if not isinstance(value, str) or not NAME_PATTERN.fullmatch(value):
        raise GroupError(f"{what} must be 1-40 lowercase letters, digits or hyphens, starting with a letter or digit")
    return value


def _check_tunnel_url(value) -> str:
    """A tunnel's base URL, as https://host[:port]. Plain http is only accepted to loopback.

    Plain http to loopback is for a relay on this machine, which the tests use
    with a stand-in tunnel. A member never dials a remote host without TLS.
    """
    if not isinstance(value, str):
        raise GroupError("the tunnel address is missing")
    try:
        parts = urllib.parse.urlsplit(value)
        parts.port  # raises ValueError for a bad port
    except ValueError:
        raise GroupError(f"not a tunnel address: {value!r}") from None
    if (parts.scheme not in ("https", "http") or not parts.hostname or parts.path not in ("", "/")
            or parts.query or parts.fragment or parts.username or parts.password):
        raise GroupError(f"not a tunnel address: {value!r}")
    if parts.scheme == "http" and parts.hostname not in LOOPBACK_HOSTS:
        raise GroupError(f"tunnel addresses must use https; plain http is only for loopback: {value!r}")
    return f"{parts.scheme}://{parts.netloc}"


def _close_quietly(sock) -> None:
    try:
        sock.close()
    except OSError:
        pass


def _write_private(path: Path, data: dict) -> None:
    """Write JSON atomically with owner-only permissions: keys and secrets live in these files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _group_dir(groups_dir: Path, group: str) -> Path:
    return groups_dir / _check_name(group, "group name")


def _now_iso(now: float) -> str:
    return datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- tokens ------------------------------------------------------------------

def _b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _b64d(text: str) -> bytes:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", text):
        raise GroupError("malformed group token")
    try:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError):
        raise GroupError("malformed group token") from None


def issue_token(secret: bytes, *, group: str, member: str, tunnel_url: str, expires: float,
                nonce: str | None = None) -> str:
    """A one-time enrolment token for `member`, signed with the group secret.

    `n` is the one-time secret: the coordinator spends it on enrolment, and a
    second attempt with the same token is refused.
    """
    _check_name(group, "group name")
    _check_name(member, "member name")
    payload = {"v": TOKEN_VERSION, "g": group, "m": member, "u": _check_tunnel_url(tunnel_url),
               "x": int(expires), "n": nonce or secrets.token_hex(16)}
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    signature = hmac.new(secret, body, hashlib.sha256).digest()
    return f"{TOKEN_PREFIX}{_b64e(body)}.{_b64e(signature)}"


def read_token(token: str) -> tuple[dict, bytes, bytes]:
    """Parse a token without checking its signature: (payload, signed bytes, signature).

    A member cannot check the signature; only the coordinator holds the secret.
    The member reads the fields it needs, and the coordinator checks everything.
    """
    if not isinstance(token, str) or not token.startswith(TOKEN_PREFIX):
        raise GroupError("not a group token; it should start with cng1.")
    parts = token[len(TOKEN_PREFIX):].split(".")
    if len(parts) != 2:
        raise GroupError("malformed group token")
    body, signature = _b64d(parts[0]), _b64d(parts[1])
    try:
        payload = json.loads(body)
    except ValueError:
        raise GroupError("malformed group token") from None
    if not isinstance(payload, dict) or payload.get("v") != TOKEN_VERSION:
        raise GroupError("unsupported group token version; ask the coordinator for a new one")
    for key in ("g", "m", "u", "n"):
        if not isinstance(payload.get(key), str):
            raise GroupError("malformed group token")
    if type(payload.get("x")) is not int:
        raise GroupError("malformed group token")
    _check_name(payload["g"], "group name")
    _check_name(payload["m"], "member name")
    _check_tunnel_url(payload["u"])
    return payload, body, signature


def check_token(secret: bytes, token: str, now: float) -> dict:
    """The token's payload, if it is signed by `secret` and has not expired."""
    payload, body, signature = read_token(token)
    expected = hmac.new(secret, body, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, signature):
        raise GroupError("this group token is not valid for this group")
    if payload["x"] <= now:
        raise GroupError("this group token has expired; ask the coordinator for a new one")
    return payload


# --- framing -----------------------------------------------------------------

def _check_frame(kind: int, stream: int, length: int) -> None:
    if kind not in KINDS:
        raise FrameError(f"unknown frame kind {kind}")
    if length > MAX_FRAME_PAYLOAD:
        raise FrameError(f"frame payload of {length} bytes is over the {MAX_FRAME_PAYLOAD} byte limit")
    if kind in (OPEN, DATA, CLOSE) and stream == 0:
        raise FrameError("stream 0 is reserved for control frames")
    if kind in (PING, PONG) and stream != 0:
        raise FrameError("ping and pong frames belong on stream 0")
    if kind in (OPEN, CLOSE, PING, PONG) and length:
        raise FrameError(f"frame kind {kind} carries no payload")


def encode_frame(kind: int, stream: int, payload: bytes = b"") -> bytes:
    if not 0 <= stream <= 0xFFFFFFFF:
        raise FrameError("stream id out of range")
    _check_frame(kind, stream, len(payload))
    return FRAME_HEADER.pack(kind, stream, len(payload)) + payload


class FrameDecoder:
    """Turns a byte stream into frames, however the bytes arrive.

    A frame's header is checked as soon as it is complete, before its payload is
    buffered, so a peer cannot make this allocate more than one frame's limit.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[tuple[int, int, bytes]]:
        self._buffer += data
        frames = []
        while len(self._buffer) >= FRAME_HEADER.size:
            kind, stream, length = FRAME_HEADER.unpack_from(self._buffer)
            _check_frame(kind, stream, length)
            end = FRAME_HEADER.size + length
            if len(self._buffer) < end:
                break
            payload = bytes(self._buffer[FRAME_HEADER.size:end])
            del self._buffer[:end]
            frames.append((kind, stream, payload))
        return frames


class Mux:
    """One link carrying many byte streams, driven by a single I/O loop.

    The link is any socket-like object: send, recv, settimeout and close. In
    production it is a WebSocketStream. Only run() touches it. Other threads
    queue frames with send_frame(), so the link is never read and written at the
    same moment from two threads.
    """

    def __init__(self, sock, on_open=None) -> None:
        self.sock = sock
        self.on_open = on_open  # agent side: stream id -> local socket, or None to refuse
        self.error: Exception | None = None
        self._decoder = FrameDecoder()
        self._out = bytearray()
        self._out_lock = threading.Lock()
        self._streams: dict[int, socket.socket] = {}
        self._streams_lock = threading.Lock()
        self._next_stream = 1
        self._closed = threading.Event()
        self._last_seen = time.monotonic()
        self._last_ping = time.monotonic()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def send_frame(self, kind: int, stream: int, payload: bytes = b"") -> None:
        frame = encode_frame(kind, stream, payload)
        with self._out_lock:
            self._out += frame

    def open_stream(self, local) -> int | None:
        """Coordinator side: carry a local connection over this link. None if the link is closed."""
        with self._streams_lock:
            if self.closed:
                return None
            stream = self._next_stream
            self._next_stream += 1
            self._streams[stream] = local
        self.send_frame(OPEN, stream)
        self._start_pump(stream, local)
        return stream

    def run(self) -> None:
        """Serve the link until it closes. Returns; the reason is in self.error."""
        self.sock.settimeout(IO_TICK_SECONDS)
        try:
            while not self.closed:
                self._flush()
                self._read()
                self._keepalive()
        except (OSError, GroupError) as e:
            self.error = e
        finally:
            self.close()

    def close(self) -> None:
        with self._streams_lock:
            if self._closed.is_set():
                return
            self._closed.set()
            streams = list(self._streams.values())
            self._streams.clear()
        for local in streams:
            _close_quietly(local)
        _close_quietly(self.sock)

    def _flush(self) -> None:
        with self._out_lock:
            if not self._out:
                return
            chunk = bytes(self._out[:DATA_CHUNK])
        try:
            sent = self.sock.send(chunk)
        except (socket.timeout, ssl.SSLWantReadError, ssl.SSLWantWriteError):
            return  # the same bytes are offered again next tick
        with self._out_lock:
            del self._out[:sent]

    def _read(self) -> None:
        try:
            data = self.sock.recv(DATA_CHUNK)
        except (socket.timeout, ssl.SSLWantReadError, ssl.SSLWantWriteError):
            return
        if not data:
            raise ConnectionError("the peer closed the connection")
        self._last_seen = time.monotonic()
        for kind, stream, payload in self._decoder.feed(data):
            self._dispatch(kind, stream, payload)

    def _keepalive(self) -> None:
        now = time.monotonic()
        if now - self._last_seen > IDLE_TIMEOUT_SECONDS:
            raise ConnectionError("the peer went silent")
        if now - self._last_ping >= PING_INTERVAL_SECONDS:
            self._last_ping = now
            self.send_frame(PING, 0)

    def _dispatch(self, kind: int, stream: int, payload: bytes) -> None:
        if kind == DATA:
            with self._streams_lock:
                local = self._streams.get(stream)
            if local is None:
                return  # a stream that has already closed on this side
            try:
                local.sendall(payload)
            except OSError:
                self._drop(stream, notify=True)
        elif kind == OPEN:
            local = None
            if self.on_open is not None:
                try:
                    local = self.on_open(stream)
                except OSError:
                    local = None
            if local is None:
                self.send_frame(CLOSE, stream)
                return
            with self._streams_lock:
                if self.closed:
                    _close_quietly(local)
                    return
                self._streams[stream] = local
            self._start_pump(stream, local)
        elif kind == CLOSE:
            self._drop(stream, notify=False)
        elif kind == PING:
            self.send_frame(PONG, 0)
        # PONG only refreshes _last_seen, which _read has already done.

    def _start_pump(self, stream: int, local) -> None:
        threading.Thread(target=self._pump, args=(stream, local), daemon=True).start()

    def _pump(self, stream: int, local) -> None:
        """Local bytes into frames, until the local side ends."""
        try:
            while True:
                data = local.recv(DATA_CHUNK)
                if not data:
                    break
                self.send_frame(DATA, stream, data)
        except OSError:
            pass
        finally:
            self._drop(stream, notify=True)

    def _drop(self, stream: int, notify: bool) -> None:
        with self._streams_lock:
            local = self._streams.pop(stream, None)
        if local is None:
            return
        _close_quietly(local)
        if notify:
            self.send_frame(CLOSE, stream)


# --- WebSocket link ----------------------------------------------------------

def _xor_mask(data: bytes, key: bytes) -> bytes:
    if not data:
        return b""
    stream = (key * (len(data) // 4 + 1))[:len(data)]
    return (int.from_bytes(data, "big") ^ int.from_bytes(stream, "big")).to_bytes(len(data), "big")


def _accept_key(client_key: str) -> str:
    return base64.b64encode(hashlib.sha1((client_key + WS_GUID).encode()).digest()).decode()


def _is_ws_key(value: str) -> bool:
    try:
        return len(base64.b64decode(value, validate=True)) == 16
    except (binascii.Error, ValueError):
        return False


def _read_http_head(sock) -> tuple[str, dict]:
    """The first line and the lower-cased headers of an HTTP message, read one byte at a time.

    Byte at a time so that nothing past the blank line is consumed: the bytes after
    a WebSocket upgrade belong to the link.
    """
    head = bytearray()
    while not head.endswith(b"\r\n\r\n"):
        byte = sock.recv(1)
        if not byte:
            raise GroupError("the connection closed during the handshake")
        head += byte
        if len(head) > MAX_HTTP_HEAD_BYTES:
            raise GroupError("handshake message is too long")
    lines = head[:-4].decode("latin-1").split("\r\n")
    headers = {}
    for line in lines[1:]:
        name, sep, value = line.partition(":")
        if not sep:
            raise GroupError("malformed HTTP header in the handshake")
        headers[name.strip().lower()] = value.strip()
    return lines[0], headers


def _http_reply(sock, status: str, body: bytes = b"", extra_headers: str = "") -> None:
    sock.sendall(f"HTTP/1.1 {status}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n"
                 f"{extra_headers}\r\n".encode() + body)


class WebSocketStream:
    """A byte stream carried in WebSocket frames, with the socket's own methods.

    The link above it (Mux, and the enrolment handshake) speaks send, sendall,
    recv, settimeout and close, and this presents the same methods over a
    WebSocket, so none of that framing changes. Client frames are masked and
    server frames are not, as RFC 6455 requires. A frame that breaks that rule, or
    is too large, or is a reserved kind, is refused. Bytes are a stream: message
    boundaries are not kept, because the framing above keeps its own.
    """

    def __init__(self, sock, *, client: bool) -> None:
        self.sock = sock
        self.client = client
        self._inbox = bytearray()    # bytes read from the socket that are not yet a whole frame
        self._payload = bytearray()  # decoded data, ready for recv()
        self._out = bytearray()      # encoded frames the socket has not yet taken
        self._out_lock = threading.Lock()  # guards _out and writes to the socket
        self._eof = False
        self._closing = False

    def settimeout(self, value) -> None:
        self.sock.settimeout(value)

    def gettimeout(self):
        return self.sock.gettimeout()

    def send(self, data: bytes) -> int:
        """Queue `data` as one binary frame and write what the socket will take now.

        Raises socket.timeout when the peer is not reading and too much is queued;
        the caller offers the same bytes again later.
        """
        if not data:
            return 0
        with self._out_lock:
            if len(self._out) > WS_MAX_PENDING:
                self._flush_locked()
            if len(self._out) > WS_MAX_PENDING:
                raise socket.timeout("the link is backed up")
            self._out += self._frame(WS_OP_BINARY, bytes(data))
            self._flush_locked()
        return len(data)

    def sendall(self, data: bytes) -> None:
        self.send(data)
        timeout = self.sock.gettimeout()
        deadline = time.monotonic() + (timeout if timeout is not None else HANDSHAKE_TIMEOUT_SECONDS)
        while time.monotonic() < deadline:
            with self._out_lock:
                if not self._out:
                    return
                self._flush_locked()
                if not self._out:
                    return
            time.sleep(0.01)
        raise socket.timeout("timed out sending on the link")

    def recv(self, size: int) -> bytes:
        """Up to `size` bytes of the stream. b"" once the peer has closed. Raises socket.timeout if nothing has arrived.

        One reader at a time: the link's receive side is only read from one thread.
        Sends may come from any thread.
        """
        while not self._payload:
            if self._eof:
                return b""
            with self._out_lock:
                self._flush_locked()  # pongs and close replies go out as they are queued
            self._fill()
        chunk = bytes(self._payload[:size])
        del self._payload[:size]
        return chunk

    def close(self) -> None:
        with self._out_lock:
            if not self._closing:
                self._closing = True
                self._out += self._frame(WS_OP_CLOSE, struct.pack("!H", 1000))
                try:
                    self._flush_locked()
                except OSError:
                    pass
        _close_quietly(self.sock)

    def _frame(self, opcode: int, payload: bytes) -> bytes:
        size = len(payload)
        mask_bit = 0x80 if self.client else 0
        if size < 126:
            head = bytes((0x80 | opcode, mask_bit | size))
        elif size < 1 << 16:
            head = bytes((0x80 | opcode, mask_bit | 126)) + struct.pack("!H", size)
        else:
            head = bytes((0x80 | opcode, mask_bit | 127)) + struct.pack("!Q", size)
        if not self.client:
            return head + payload
        key = os.urandom(4)
        return head + key + _xor_mask(payload, key)

    def _flush_locked(self) -> None:
        """Write queued bytes until the socket stops taking them. Caller holds _out_lock."""
        while self._out:
            try:
                sent = self.sock.send(bytes(self._out[:DATA_CHUNK]))
            except (socket.timeout, ssl.SSLWantReadError, ssl.SSLWantWriteError):
                return
            if not sent:
                return
            del self._out[:sent]

    def _fill(self) -> None:
        """Read once from the socket and decode the whole frames that arrived."""
        data = self.sock.recv(DATA_CHUNK)
        if not data:
            self._eof = True
            return
        self._inbox += data
        while True:
            frame = self._take_frame()
            if frame is None:
                return
            opcode, payload = frame
            if opcode in (WS_OP_BINARY, WS_OP_TEXT, WS_OP_CONT):
                self._payload += payload
            elif opcode == WS_OP_PING:
                with self._out_lock:
                    self._out += self._frame(WS_OP_PONG, payload)
            elif opcode == WS_OP_CLOSE:
                self._eof = True
                with self._out_lock:
                    if not self._closing:
                        self._closing = True
                        self._out += self._frame(WS_OP_CLOSE, payload[:2])
                return
            elif opcode != WS_OP_PONG:
                raise GroupError(f"unknown websocket opcode {opcode}")

    def _take_frame(self):
        """The next whole frame in the inbox as (opcode, payload), or None if it is not all here yet."""
        buf = self._inbox
        if len(buf) < 2:
            return None
        first, second = buf[0], buf[1]
        if first & 0x70:
            raise GroupError("websocket frame sets reserved bits")
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        if masked == self.client:
            raise GroupError("websocket frame masking is wrong for this side of the link")
        length = second & 0x7F
        pos = 2
        if length == 126:
            if len(buf) < 4:
                return None
            length = struct.unpack_from("!H", buf, 2)[0]
            pos = 4
        elif length == 127:
            if len(buf) < 10:
                return None
            length = struct.unpack_from("!Q", buf, 2)[0]
            pos = 10
        if length > WS_MAX_PAYLOAD:
            raise GroupError(f"websocket frame of {length} bytes is over the {WS_MAX_PAYLOAD} byte limit")
        if opcode >= 8 and (length > 125 or not first & 0x80):
            raise GroupError("websocket control frame is too long or fragmented")
        key = b""
        if masked:
            if len(buf) < pos + 4:
                return None
            key = bytes(buf[pos:pos + 4])
            pos += 4
        end = pos + length
        if len(buf) < end:
            return None
        payload = bytes(buf[pos:end])
        del buf[:end]
        return opcode, (_xor_mask(payload, key) if masked else payload)


def server_upgrade(sock, headers: dict) -> WebSocketStream:
    """Answer a WebSocket upgrade request on `sock`. Raises GroupError if the request is not one."""
    key = headers.get("sec-websocket-key", "")
    if ("websocket" not in headers.get("upgrade", "").lower()
            or "upgrade" not in headers.get("connection", "").lower()
            or headers.get("sec-websocket-version") != "13"
            or not _is_ws_key(key)):
        raise GroupError("not a WebSocket upgrade request")
    sock.sendall(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                  f"Sec-WebSocket-Accept: {_accept_key(key)}\r\n\r\n").encode())
    return WebSocketStream(sock, client=False)


def client_upgrade(sock, host: str, path: str) -> WebSocketStream:
    """Ask for a WebSocket on `sock` and check the relay agreed. Raises GroupError if it did not."""
    key = base64.b64encode(os.urandom(16)).decode()
    sock.sendall((f"GET {path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                  f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    status, headers = _read_http_head(sock)
    parts = status.split(" ")
    if len(parts) < 2 or parts[1] != "101" or headers.get("sec-websocket-accept") != _accept_key(key):
        raise GroupError(f"the relay did not accept the group link ({status}); is group.py relay running?")
    return WebSocketStream(sock, client=True)


def client_connect(tunnel_url: str, timeout: float = HANDSHAKE_TIMEOUT_SECONDS) -> WebSocketStream:
    """A WebSocket link to the relay behind `tunnel_url`.

    https uses TLS with the system's certificate authorities and the host name
    checked, which is what Cloudflare's certificate needs. Plain http is only
    accepted to a loopback address, for a relay on this machine.
    """
    base = _check_tunnel_url(tunnel_url)
    parts = urllib.parse.urlsplit(base)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    sock = socket.create_connection((parts.hostname, port), timeout=timeout)
    try:
        if parts.scheme == "https":
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=parts.hostname)
        return client_upgrade(sock, parts.netloc, LINK_PATH)
    except BaseException:
        _close_quietly(sock)
        raise


def _read_line(sock) -> bytes:
    out = bytearray()
    while True:
        byte = sock.recv(1)
        if not byte:
            raise GroupError("the connection closed during the handshake")
        if byte == b"\n":
            return bytes(out)
        out += byte
        if len(out) > MAX_HANDSHAKE_BYTES:
            raise GroupError("handshake message is too long")


def _read_json(sock) -> dict:
    try:
        message = json.loads(_read_line(sock))
    except ValueError:
        raise GroupError("the handshake message is not JSON") from None
    if not isinstance(message, dict):
        raise GroupError("the handshake message is not an object")
    return message


def _send_json(sock, message: dict) -> None:
    sock.sendall(json.dumps(message).encode() + b"\n")


# --- coordinator state -------------------------------------------------------

class GroupState:
    """The coordinator's record of one group: its secret, tunnel, members and spent tokens.

    Members map to a SHA-256 verifier of their key, never the key. Spent token
    nonces are kept until the token would have expired anyway. `tunnel_url` is set
    only while the relay runs: it is the address tokens name.
    """

    def __init__(self, path: Path, data: dict) -> None:
        self.path = path
        self.data = data
        self._lock = threading.Lock()

    @classmethod
    def load(cls, path: Path) -> "GroupState":
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            raise GroupError(f"no group at {path.parent}; create it with `group.py init`") from None
        except (OSError, ValueError) as e:
            raise GroupError(f"cannot read {path}: {e}") from None
        for key in ("group", "secret", "members", "used"):
            if key not in data:
                raise GroupError(f"{path} is missing {key}")
        data.setdefault("tunnel_url", None)
        return cls(path, data)

    @property
    def group(self) -> str:
        return self.data["group"]

    @property
    def tunnel_url(self) -> str | None:
        return self.data.get("tunnel_url")

    def save(self) -> None:
        _write_private(self.path, self.data)

    def enrol(self, token: str, *, now: float, port_range=RPC_PORT_RANGE) -> tuple[str, str, int]:
        """Spend a token: returns (member, member key, rpc port). Each token works once."""
        with self._lock:
            payload = check_token(bytes.fromhex(self.data["secret"]), token, now)
            if payload["g"] != self.group:
                raise GroupError("this token is for a different group")
            if self.tunnel_url is None:
                raise GroupError(f"the relay for {self.group} is not running; start it with `group.py relay`")
            if payload["u"] != self.tunnel_url:
                raise GroupError("this token is for a different tunnel; ask the coordinator for a new one")
            used = self.data["used"]
            for nonce in [n for n, expires in used.items() if expires <= now]:
                del used[nonce]
            if payload["n"] in used:
                raise GroupError("this group token has already been used")
            member = payload["m"]
            if member in self.data["members"]:
                raise GroupError(f"{member} is already enrolled in this group")
            taken = {entry["rpc_port"] for entry in self.data["members"].values()}
            port = next((p for p in port_range if p not in taken), None)
            if port is None:
                raise GroupError("no free RPC port left in this group's range")
            key = secrets.token_hex(32)
            self.data["members"][member] = {
                "verifier": hashlib.sha256(key.encode()).hexdigest(),
                "rpc_port": port,
                "enrolled": _now_iso(now),
            }
            used[payload["n"]] = payload["x"]
            self.save()
            return member, key, port

    def authenticate(self, member, key) -> int | None:
        """The member's rpc port if `key` is its key, else None."""
        if not isinstance(member, str) or not isinstance(key, str):
            return None
        with self._lock:
            entry = self.data["members"].get(member)
            if entry is None:
                return None
            try:
                digest = hashlib.sha256(key.encode()).hexdigest()
            except UnicodeEncodeError:
                return None
            return entry["rpc_port"] if hmac.compare_digest(digest, entry["verifier"]) else None


# --- coordinator side: relay -------------------------------------------------

class Relay:
    """The coordinator's end of every member link.

    A plain HTTP server on 127.0.0.1, which the tunnel points at. It answers a
    health check, and it upgrades LINK_PATH to a WebSocket that carries enrolment
    and member links. Each enrolled member also gets a loopback listener on its
    rpc port. A connection to that port is carried to the member over the
    member's link, or refused if the member is not connected.
    """

    def __init__(self, state_path: Path, host: str = "127.0.0.1", port: int = 0, *,
                 port_range=RPC_PORT_RANGE, log=print) -> None:
        if host != "127.0.0.1":
            raise GroupError("the relay listens on 127.0.0.1 only; the tunnel is what reaches it from outside")
        self.state = GroupState.load(state_path)
        self.host = host
        self.port = port
        self.port_range = port_range
        self.log = log
        self.address = None
        self._listener = None
        self._member_listeners: dict[str, socket.socket] = {}
        self._links: dict[str, Mux] = {}
        self._lock = threading.Lock()
        self._stopping = threading.Event()

    def start(self) -> tuple[str, int]:
        self._listener = socket.create_server((self.host, self.port))
        self.address = self._listener.getsockname()
        threading.Thread(target=self._accept, daemon=True).start()
        for member, entry in self.state.data["members"].items():
            try:
                self._bind_member(member, entry["rpc_port"])
            except GroupError as e:
                self.log(f"warning: {e}")
        return self.address

    def close(self) -> None:
        self._stopping.set()
        listeners = [self._listener, *self._member_listeners.values()]
        for listener in listeners:
            if listener is not None:
                _close_quietly(listener)
        with self._lock:
            links = list(self._links.values())
        for mux in links:
            mux.close()

    def _accept(self) -> None:
        while not self._stopping.is_set():
            try:
                conn, _ = self._listener.accept()
            except OSError:
                return
            threading.Thread(target=self._handshake, args=(conn,), daemon=True).start()

    def _handshake(self, raw) -> None:
        link = None
        try:
            raw.settimeout(HANDSHAKE_TIMEOUT_SECONDS)
            request_line, headers = _read_http_head(raw)
            method, _, rest = request_line.partition(" ")
            path = rest.split(" ", 1)[0].split("?", 1)[0]
            if method == "GET" and path == HEALTH_PATH:
                _http_reply(raw, "200 OK", b'{"ok": true}', "Content-Type: application/json\r\n")
                return
            if method != "GET" or path != LINK_PATH:
                _http_reply(raw, "404 Not Found")
                return
            try:
                link = server_upgrade(raw, headers)
            except GroupError:
                _http_reply(raw, "400 Bad Request")
                return
            self._serve(link)
        except (OSError, GroupError) as e:
            self.log(f"connection refused: {e}")
        finally:
            _close_quietly(link if link is not None else raw)

    def _serve(self, link) -> None:
        request = _read_json(link)
        op = request.get("op")
        if op == "enrol":
            try:
                member, key, port = self.state.enrol(request.get("token"), now=time.time(),
                                                     port_range=self.port_range)
            except GroupError as e:
                _send_json(link, {"ok": False, "error": str(e)})
                self.log(f"enrolment refused: {e}")
                return
            _send_json(link, {"ok": True, "member": member, "key": key, "rpc_port": port})
            self.log(f"enrolled {member} (127.0.0.1:{port})")
            try:
                self._bind_member(member, port)
            except GroupError as e:
                self.log(f"warning: {e}")
            return
        if op == "auth":
            member = request.get("member")
            port = self.state.authenticate(member, request.get("key"))
            if port is None:
                _send_json(link, {"ok": False, "error": "this member is not enrolled, or its key is wrong"})
                self.log(f"authentication refused for {member!r}")
                return
            _send_json(link, {"ok": True, "rpc_port": port})
            self._attach(member, link)
            return
        if op == "rejoin":
            # Only the member key admits a rejoin. A one-time token, fresh or
            # spent, expired or not, is refused even if a key comes with it.
            member = request.get("member")
            if "token" in request:
                _send_json(link, {"ok": False, "error": "a one-time token cannot rejoin; rejoin with the member key"})
                self.log(f"rejoin refused for {member!r}: a one-time token was presented")
                return
            port = self.state.authenticate(member, request.get("key"))
            if port is None:
                _send_json(link, {"ok": False, "error": "this member is not enrolled, or its key is wrong"})
                self.log(f"rejoin refused for {member!r}")
                return
            _send_json(link, {"ok": True, "member": member, "rpc_port": port})
            self.log(f"{member} rejoined at the current tunnel")
            return
        _send_json(link, {"ok": False, "error": "unknown request"})

    def _attach(self, member: str, link) -> None:
        mux = Mux(link)
        with self._lock:
            previous = self._links.get(member)
            self._links[member] = mux
        if previous is not None:
            previous.close()
        self.log(f"{member} connected")
        try:
            mux.run()
        finally:
            with self._lock:
                if self._links.get(member) is mux:
                    del self._links[member]
            self.log(f"{member} disconnected" + (f" ({mux.error})" if mux.error else ""))

    def _bind_member(self, member: str, port: int) -> None:
        with self._lock:
            if member in self._member_listeners:
                return
            try:
                listener = socket.create_server(("127.0.0.1", port))
            except OSError as e:
                raise GroupError(f"cannot listen on 127.0.0.1:{port} for {member} ({e})") from None
            self._member_listeners[member] = listener
        threading.Thread(target=self._accept_member, args=(member, listener), daemon=True).start()

    def _accept_member(self, member: str, listener) -> None:
        while not self._stopping.is_set():
            try:
                local, _ = listener.accept()
            except OSError:
                return
            with self._lock:
                mux = self._links.get(member)
            if mux is None or mux.open_stream(local) is None:
                _close_quietly(local)


def _open_tunnel(local_url: str, local_port: int):
    """A Cloudflare quick tunnel to the local relay, through the code the worker's tunnel uses."""
    import join  # join.py imports this module only in its member command, so there is no cycle
    return join.start_tunnel(local_url, f"127.0.0.1:{local_port}")


def _publish_tunnel(server: Relay, url: str, log) -> None:
    server.state.data["tunnel_url"] = _check_tunnel_url(url)
    server.state.save()
    log(f"tunnel URL: {url}")


def run_relay(state_path: Path, *, port: int = 0, log=print, stop: threading.Event | None = None,
              open_tunnel=_open_tunnel, port_range=RPC_PORT_RANGE) -> None:
    """Serve the group on 127.0.0.1, point a tunnel at it, and run until `stop` is set.

    The tunnel URL is written to group.json while the relay runs, so `issue` can
    name it, and is cleared on the way out. A tunnel unreachable for the whole grace
    period is restarted, which gives it a new URL: tokens issued for the old one stop
    working, so they must be issued again.
    """
    import join  # for the tunnel health check, shared with the worker
    stop = stop or threading.Event()
    server = Relay(state_path, "127.0.0.1", port, port_range=port_range, log=log)
    local_port = server.start()[1]
    local_url = f"http://127.0.0.1:{local_port}"
    log(f"relay listening on 127.0.0.1:{local_port} for group {server.state.group}")
    proc = None
    try:
        proc, url = open_tunnel(local_url, local_port)
        healthy_at = time.monotonic()
        if not join.wait_for_tunnel(url, None, HEALTH_PATH):
            log("warning: the tunnel is not reachable yet; members may not connect until it is")
        _publish_tunnel(server, url, log)
        while not stop.wait(RELAY_HEALTH_CHECK_SECONDS):
            if proc.poll() is None and join.tunnel_is_healthy(url, None, HEALTH_PATH):
                healthy_at = time.monotonic()
                continue
            unreachable_for = time.monotonic() - healthy_at
            if unreachable_for < RELAY_TUNNEL_GRACE_SECONDS:
                log(f"the tunnel is not reachable yet ({unreachable_for:.0f}s); "
                    f"it is replaced after {RELAY_TUNNEL_GRACE_SECONDS}s unreachable")
                continue
            log(f"the tunnel has been unreachable for {unreachable_for:.0f}s; restarting it")
            _stop_process(proc)
            proc, url = open_tunnel(local_url, local_port)
            healthy_at = time.monotonic()
            _publish_tunnel(server, url, log)
    finally:
        if proc is not None:
            _stop_process(proc)
        server.state.data["tunnel_url"] = None
        server.state.save()
        server.close()
        log("relay stopped")


# --- coordinator commands ----------------------------------------------------

def init_group(group: str, *, groups_dir: Path = GROUPS_DIR) -> dict:
    """Create a group and its secret. The tunnel address is set when the relay starts."""
    _check_name(group, "group name")
    directory = _group_dir(groups_dir, group)
    if (directory / "group.json").exists():
        raise GroupError(f"group {group} already exists at {directory}")
    data = {"v": TOKEN_VERSION, "group": group, "tunnel_url": None,
            "secret": secrets.token_hex(32), "members": {}, "used": {}}
    _write_private(directory / "group.json", data)
    return data


def issue(group: str, member: str, *, hours: float = DEFAULT_TOKEN_HOURS,
          groups_dir: Path = GROUPS_DIR, now: float | None = None) -> str:
    state = GroupState.load(_group_dir(groups_dir, group) / "group.json")
    _check_name(member, "member name")
    if member in state.data["members"]:
        raise GroupError(f"{member} is already enrolled in {group}")
    if state.tunnel_url is None:
        raise GroupError(f"the relay for {group} is not running; start it with: group.py relay --group {group}")
    now = time.time() if now is None else now
    return issue_token(bytes.fromhex(state.data["secret"]), group=group, member=member,
                       tunnel_url=state.tunnel_url, expires=now + hours * 3600)


# --- member side: enrolment and worker agent ---------------------------------

def member_path(groups_dir: Path, group: str) -> Path:
    return _group_dir(groups_dir, group) / "member.json"


def load_member(groups_dir: Path, group: str) -> dict | None:
    path = member_path(groups_dir, group)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        raise GroupError(f"cannot read {path}; delete it and join again") from None


def enrol(token: str, *, groups_dir: Path = GROUPS_DIR, now: float | None = None) -> dict:
    """Redeem a token with the coordinator and keep the member key on this machine."""
    payload, _, _ = read_token(token)
    now = time.time() if now is None else now
    if payload["x"] <= now:
        raise GroupError("this group token has expired; ask the coordinator for a new one")
    try:
        link = client_connect(payload["u"])
    except OSError as e:
        raise GroupError(f"cannot reach the relay this token names ({e}); if the relay has restarted, "
                         "ask the coordinator for a new token") from None
    try:
        _send_json(link, {"op": "enrol", "token": token})
        reply = _read_json(link)
    finally:
        _close_quietly(link)
    if not reply.get("ok"):
        raise GroupError(str(reply.get("error") or "the coordinator refused this token"))
    if reply.get("member") != payload["m"] or not isinstance(reply.get("key"), str):
        raise GroupError("the coordinator's reply does not match the token")
    member = {"v": TOKEN_VERSION, "group": payload["g"], "member": payload["m"],
              "tunnel_url": payload["u"], "key": reply["key"], "rpc_port": reply.get("rpc_port")}
    _write_private(member_path(groups_dir, payload["g"]), member)
    return member


def rejoin(member: dict, relay_url: str, *, groups_dir: Path = GROUPS_DIR) -> dict:
    """Prove this member's key to the relay at `relay_url`, and keep that address in member.json.

    No one-time token is involved: the member key is the credential. The member's
    name and rpc port stay as they were, so the coordinator's port mapping still
    applies. A refused key leaves member.json as it was.
    """
    url = _check_tunnel_url(relay_url)
    try:
        link = client_connect(url)
    except OSError as e:
        raise GroupError(f"cannot reach the relay at {url} ({e}); check the address the coordinator printed") from None
    try:
        _send_json(link, {"op": "rejoin", "member": member["member"], "key": member["key"]})
        reply = _read_json(link)
    finally:
        _close_quietly(link)
    if not reply.get("ok"):
        raise GroupError(str(reply.get("error") or "the coordinator refused this member"))
    if reply.get("member") != member["member"] or reply.get("rpc_port") != member.get("rpc_port"):
        raise GroupError("the coordinator's reply does not match this member")
    updated = dict(member, tunnel_url=url)
    _write_private(member_path(groups_dir, member["group"]), updated)
    return updated


def _connect_rpc(port: int):
    try:
        sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    except OSError:
        return None
    sock.settimeout(None)
    return sock


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _wait_for_listener(port: int, proc: subprocess.Popen, log_path: Path) -> None:
    deadline = time.monotonic() + RPC_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise GroupError(f"the RPC worker exited with code {proc.returncode}; see {log_path}")
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            return
        except OSError:
            time.sleep(0.2)
    raise GroupError(f"the RPC worker did not start listening on 127.0.0.1:{port}; see {log_path}")


def _stop_process(proc) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# Accelerator devices as the RPC server names them: CUDA0, MTL0 (Metal), Vulkan0 and
# so on. CPU, BLAS and RPC are not accelerators, so they never qualify.
GPU_DEVICE_PATTERN = re.compile(r"(?:CUDA|MTL|Vulkan|ROCm|HIP|SYCL|MUSA|CANN)[0-9]+")
DEVICE_LINE_PATTERN = re.compile(r"^\s*([A-Za-z][A-Za-z0-9]*):\s", re.MULTILINE)
RPC_DEVICE_LIST_TIMEOUT_SECONDS = 60
LLAMA_SERVER_NAME = "llama-server.exe" if platform.system() == "Windows" else "llama-server"


def parse_device_listing(text: str) -> list[str]:
    """The device names in a `--list-devices` listing, in the order it gives them. Pure."""
    return DEVICE_LINE_PATTERN.findall(text)


def list_rpc_devices(binary: Path) -> list[str]:
    """The devices this build reports, from `llama-server --list-devices` beside the RPC server.

    ggml-rpc-server cannot list devices: it refuses --list-devices and its --help names
    none. llama-server from the same archive lists the same local devices, and those
    are the names ggml-rpc-server takes with -d.
    """
    try:
        result = subprocess.run([str(binary.with_name(LLAMA_SERVER_NAME)), "--list-devices"],
                                stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                timeout=RPC_DEVICE_LIST_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise GroupError(f"cannot ask llama-server for the devices ({e})") from None
    if result.returncode != 0:
        raise GroupError(f"llama-server beside the RPC server could not list its devices "
                         f"(exit code {result.returncode})")
    return parse_device_listing(result.stdout + result.stderr)


def select_gpu_device(binary: Path, *, list_devices=list_rpc_devices) -> str:
    """The GPU device this member serves. Refuses when there is none: members never fall back to CPU.

    Called before the member spends its token, and before the RPC server accepts
    any connection, so a machine without a GPU never joins.
    """
    devices = list_devices(binary)
    gpus = [device for device in devices if GPU_DEVICE_PATTERN.fullmatch(device)]
    if not gpus:
        listed = ", ".join(devices) if devices else "nothing"
        raise GroupError(f"this machine has no GPU device for the group: llama-server lists {listed}. "
                         "A group member must have a GPU, and it will not fall back to CPU. Not joining.")
    return gpus[0]


def run_worker(member: dict, rpc_binary: Path, *, device: str, log_path: Path, log=print,
               stop: threading.Event | None = None) -> None:
    """Start this member's ggml-rpc-server on `device` and keep an outbound link to the coordinator.

    The RPC server stays up across reconnects, so a dropped link does not reload
    anything. Refusal of the member's key stops the agent: retrying would not
    change the answer. A relay that is not there yet is retried, as is a dropped link.
    """
    stop = stop or threading.Event()
    rpc_port = _free_loopback_port()
    with open(log_path, "ab") as log_file:
        rpc = subprocess.Popen([str(rpc_binary), "--host", "127.0.0.1", "--port", str(rpc_port), "-d", device],
                               stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT)
    try:
        _wait_for_listener(rpc_port, rpc, log_path)
        log(f"RPC worker listening on 127.0.0.1:{rpc_port}; connecting to {member['tunnel_url']}")
        delay = 1.0
        while not stop.is_set():
            if rpc.poll() is not None:
                raise GroupError(f"the RPC worker stopped (exit code {rpc.returncode}); see {log_path}")
            try:
                link = client_connect(member["tunnel_url"])
            except (OSError, GroupError) as e:
                log(f"coordinator unreachable ({e}); retrying in {delay:.0f}s")
                stop.wait(delay)
                delay = min(delay * 2, RECONNECT_MAX_SECONDS)
                continue
            try:
                _send_json(link, {"op": "auth", "member": member["member"], "key": member["key"]})
                reply = _read_json(link)
            except (OSError, GroupError) as e:
                _close_quietly(link)
                log(f"handshake failed ({e}); retrying in {delay:.0f}s")
                stop.wait(delay)
                delay = min(delay * 2, RECONNECT_MAX_SECONDS)
                continue
            if not reply.get("ok"):
                _close_quietly(link)
                raise GroupError(str(reply.get("error") or "the coordinator refused this member"))
            log("connected to the coordinator")
            delay = 1.0
            mux = Mux(link, on_open=lambda _stream: _connect_rpc(rpc_port))
            mux.run()
            if stop.is_set():
                break
            log(f"lost the coordinator ({mux.error or 'closed'}); reconnecting in {delay:.0f}s")
            stop.wait(delay)
    finally:
        _stop_process(rpc)


def join_group(token: str, *, rpc_binary: Path, groups_dir: Path = GROUPS_DIR, log=print,
               stop: threading.Event | None = None, now: float | None = None,
               list_devices=list_rpc_devices, relay_url: str | None = None) -> None:
    """Enrol with `token` if this machine is not yet a member, then run the worker.

    A machine that is already a member of the token's group rejoins with its member
    key when `relay_url` names the relay's current address; without it, the worker
    resumes at the address in member.json. The token only names the group and the
    member here: it never admits a rejoin.

    The GPU check comes first, so a machine without a GPU refuses before it
    spends the token.
    """
    payload, _, _ = read_token(token)
    group = payload["g"]
    device = select_gpu_device(rpc_binary, list_devices=list_devices)
    current = load_member(groups_dir, group)
    if current is None:
        if relay_url is not None:
            raise GroupError("--relay is for a machine already enrolled in this group; "
                             "a new member's token names the relay")
        current = enrol(token, groups_dir=groups_dir, now=now)
        log(f"enrolled as {current['member']} in group {group}")
    elif current.get("member") != payload["m"]:
        raise GroupError(f"this machine is already enrolled in group {group} as {current.get('member')}")
    elif relay_url is not None:
        current = rejoin(current, relay_url, groups_dir=groups_dir)
        log(f"rejoined group {group} as {current['member']} at {current['tunnel_url']}")
    else:
        log(f"already enrolled in group {group} as {current['member']}")
    log(f"serving GPU device {device} to group {group}")
    run_worker(current, rpc_binary, device=device, log_path=_group_dir(groups_dir, group) / "rpc-server.log",
               log=log, stop=stop)


# --- admin CLI (coordinator) -------------------------------------------------

def _interrupted(signum, frame) -> None:
    # SIGTERM unwinds exactly like Ctrl+C, so run_relay's cleanup stops the tunnel and clears its URL.
    raise KeyboardInterrupt


def _members_report(state: GroupState) -> list[str]:
    lines = []
    for name, entry in sorted(state.data["members"].items()):
        lines.append(f"{name:<24} 127.0.0.1:{entry['rpc_port']:<7} enrolled {entry.get('enrolled', '?')}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="group.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="create a group on this coordinator")
    init.add_argument("--group", required=True)

    issue_cmd = commands.add_parser("issue", help="print a one-time token for a new member (the relay must be running)")
    issue_cmd.add_argument("--group", required=True)
    issue_cmd.add_argument("--member", required=True)
    issue_cmd.add_argument("--hours", type=float, default=DEFAULT_TOKEN_HOURS)

    members = commands.add_parser("members", help="list enrolled members and their loopback RPC endpoints")
    members.add_argument("--group", required=True)

    relay = commands.add_parser("relay", help="start the local relay and its tunnel, in the foreground, and print the tunnel URL")
    relay.add_argument("--group", required=True)
    relay.add_argument("--port", type=int, default=0,
                       help="local port for the relay on 127.0.0.1 (default: any free port)")

    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            data = init_group(args.group, groups_dir=GROUPS_DIR)
            print(f"created group {data['group']}")
            print(f"next: python3 join/group.py relay --group {data['group']}")
            print(f"then: python3 join/group.py issue --group {data['group']} --member NAME")
        elif args.command == "issue":
            print(issue(args.group, args.member, hours=args.hours, groups_dir=GROUPS_DIR))
        elif args.command == "members":
            state = GroupState.load(_group_dir(GROUPS_DIR, args.group) / "group.json")
            lines = _members_report(state)
            print("\n".join(lines) if lines else "no members enrolled yet")
            endpoints = [f"127.0.0.1:{entry['rpc_port']}" for entry in state.data["members"].values()]
            print("rpc_servers: " + json.dumps(sorted(endpoints)))
        elif args.command == "relay":
            state_path = _group_dir(GROUPS_DIR, args.group) / "group.json"
            GroupState.load(state_path)  # refuse an unknown group before anything starts
            print(f"group {args.group}: ctrl+c or SIGTERM stops the relay and its tunnel")
            previous = signal.signal(signal.SIGTERM, _interrupted)
            try:
                run_relay(state_path, port=args.port)
            except KeyboardInterrupt:
                print("stopping the relay")
            finally:
                signal.signal(signal.SIGTERM, previous)
    except GroupError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (OSError, ssl.SSLError) as e:
        print(f"error: could not reach the coordinator's relay ({e})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
