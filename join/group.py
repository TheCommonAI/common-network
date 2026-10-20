#!/usr/bin/env python3
"""Distributed model groups: enrol a member machine and link its RPC worker.

A group's coordinator runs llama.cpp with RPC peers. Each peer is a
ggml-rpc-server on a member's machine, and llama.cpp speaks RPC to it in the
clear with no authentication of its own. Members are not asked to open ports,
forward SSH or install anything extra. The rules that make that safe:

- A member never listens on the network. Its worker agent dials out to the
  coordinator over TLS and keeps that connection open. The ggml-rpc-server it
  starts is bound to 127.0.0.1, and the agent relays bytes between that server
  and the coordinator over the outbound connection.
- The coordinator gives each admitted member its own loopback port,
  127.0.0.1:<rpc_port>. That is the rpc_servers entry the distributed config
  lists, so the runtime and its config do not change. Only processes on the
  coordinator can reach those ports.

Admission is by one-time token. The coordinator issues a token, HMAC-signed with
the group secret, naming the group, the member, the coordinator's address, the
pinned certificate fingerprint and an expiry. The member pins that fingerprint
when it connects, so the token cannot be redeemed at a look-alike server. The
coordinator spends the token's nonce on enrolment and returns a member key over
that TLS connection. It keeps only the SHA-256 of the key, and each later
connection presents the key over TLS.

TLS is the only secure transport used. The repo's Common worker tunnel carries
HTTP requests, not raw TCP, so it cannot carry RPC. The coordinator's
certificate is self-signed and made with the openssl command, which only the
coordinator needs. Members need nothing beyond Python.

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
import re
import secrets
import shutil
import socket
import ssl
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

COMMON_DIR = Path.home() / ".common-network"
GROUPS_DIR = COMMON_DIR / "groups"

TOKEN_PREFIX = "cng1."
TOKEN_VERSION = 1
DEFAULT_TOKEN_HOURS = 24

# Member RPC ports on the coordinator. Kept clear of llama-server's own port and
# the RPC workers the manual setup uses (50051, 50052).
RPC_PORT_RANGE = range(52100, 52200)

NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")

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
RECONNECT_MAX_SECONDS = 60.0
RPC_START_TIMEOUT_SECONDS = 30.0


class GroupError(Exception):
    """A problem a person can act on: a bad token, a refused enrolment, a missing file."""


class FrameError(GroupError):
    """The peer sent bytes that are not a valid frame. The link is closed."""


# --- small helpers -----------------------------------------------------------

def _check_name(value: str, what: str) -> str:
    if not isinstance(value, str) or not NAME_PATTERN.fullmatch(value):
        raise GroupError(f"{what} must be 1-40 lowercase letters, digits or hyphens, starting with a letter or digit")
    return value


def _split_hostport(text: str) -> tuple[str, int]:
    host, sep, port = str(text).rpartition(":")
    if not sep or not host or not port.isdigit() or not 1 <= int(port) <= 65535:
        raise GroupError(f"expected host:port, got {text!r}")
    return host, int(port)


def _close_quietly(sock) -> None:
    try:
        sock.close()
    except OSError:
        pass


def cert_fingerprint(der: bytes) -> str:
    return hashlib.sha256(der).hexdigest()


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


def issue_token(secret: bytes, *, group: str, member: str, relay: str, cert_fp: str,
                expires: float, nonce: str | None = None) -> str:
    """A one-time enrolment token for `member`, signed with the group secret."""
    _check_name(group, "group name")
    _check_name(member, "member name")
    payload = {"v": TOKEN_VERSION, "g": group, "m": member, "r": relay, "f": cert_fp,
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
        raise GroupError("unsupported group token version")
    for key in ("g", "m", "r", "f", "n"):
        if not isinstance(payload.get(key), str):
            raise GroupError("malformed group token")
    if type(payload.get("x")) is not int:
        raise GroupError("malformed group token")
    _check_name(payload["g"], "group name")
    _check_name(payload["m"], "member name")
    _split_hostport(payload["r"])
    if not re.fullmatch(r"[0-9a-f]{64}", payload["f"]):
        raise GroupError("malformed group token")
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
    """One TLS link carrying many byte streams, driven by a single I/O loop.

    Only run() touches the socket. Other threads queue frames with send_frame(),
    so the TLS object is never read and written at the same moment from two
    threads, which OpenSSL does not allow.
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


# --- TLS ---------------------------------------------------------------------

def server_context(cert: Path, key: Path) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(str(cert), str(key))
    return ctx


def client_connect(relay: str, fingerprint: str, timeout: float = HANDSHAKE_TIMEOUT_SECONDS):
    """A TLS connection to the coordinator, refused unless its certificate is the pinned one.

    Certificate chains are not checked: the pin is the trust. The token names the
    coordinator's certificate by fingerprint, and a different certificate is a
    different server, whatever its name says.
    """
    host, port = _split_hostport(relay)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    raw = socket.create_connection((host, port), timeout=timeout)
    try:
        tls = ctx.wrap_socket(raw, server_hostname=host)
    except (OSError, ssl.SSLError):
        _close_quietly(raw)
        raise
    der = tls.getpeercert(binary_form=True)
    if not der or not hmac.compare_digest(cert_fingerprint(der), fingerprint):
        _close_quietly(tls)
        raise GroupError("the coordinator's certificate does not match the group token; not connecting")
    return tls


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


def _make_certificate(cert: Path, key: Path, group: str) -> None:
    openssl = shutil.which("openssl")
    if openssl is None:
        raise GroupError("the coordinator needs the openssl command to make its certificate; "
                         "install it (macOS and most Linux systems have it) and run init again")
    subprocess.run([openssl, "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1",
                    "-nodes", "-keyout", str(key), "-out", str(cert), "-days", "3650",
                    "-subj", f"/CN=common-group-{group}"],
                   check=True, capture_output=True, timeout=60)
    os.chmod(key, 0o600)


# --- coordinator state -------------------------------------------------------

class GroupState:
    """The coordinator's record of one group: its secret, members and spent tokens.

    Members map to a SHA-256 verifier of their key, never the key. Spent token
    nonces are kept until the token would have expired anyway.
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
        for key in ("group", "relay", "cert_fp", "secret", "members", "used"):
            if key not in data:
                raise GroupError(f"{path} is missing {key}")
        return cls(path, data)

    @property
    def group(self) -> str:
        return self.data["group"]

    @property
    def relay(self) -> str:
        return self.data["relay"]

    @property
    def cert_fp(self) -> str:
        return self.data["cert_fp"]

    def save(self) -> None:
        _write_private(self.path, self.data)

    def enrol(self, token: str, *, now: float, port_range=RPC_PORT_RANGE) -> tuple[str, str, int]:
        """Spend a token: returns (member, member key, rpc port). Each token works once."""
        with self._lock:
            payload = check_token(bytes.fromhex(self.data["secret"]), token, now)
            if payload["g"] != self.group or payload["r"] != self.relay or payload["f"] != self.cert_fp:
                raise GroupError("this token is for a different group or coordinator")
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

    One TLS listener takes enrolments and member connections. Each enrolled member
    also gets a loopback listener on its rpc port. A connection to that port is
    carried to the member over the member's link, or refused if the member is
    not connected.
    """

    def __init__(self, state_path: Path, host: str, port: int, *,
                 port_range=RPC_PORT_RANGE, log=print) -> None:
        self.state = GroupState.load(state_path)
        self.host = host
        self.port = port
        self.port_range = port_range
        self.log = log
        self.ctx = server_context(state_path.parent / "cert.pem", state_path.parent / "key.pem")
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
        try:
            raw.settimeout(HANDSHAKE_TIMEOUT_SECONDS)
            tls = self.ctx.wrap_socket(raw, server_side=True)
        except (OSError, ssl.SSLError):
            _close_quietly(raw)
            return
        try:
            self._serve(tls)
        except (OSError, GroupError) as e:
            self.log(f"connection refused: {e}")
        finally:
            _close_quietly(tls)

    def _serve(self, tls) -> None:
        request = _read_json(tls)
        op = request.get("op")
        if op == "enrol":
            try:
                member, key, port = self.state.enrol(request.get("token"), now=time.time(),
                                                     port_range=self.port_range)
            except GroupError as e:
                _send_json(tls, {"ok": False, "error": str(e)})
                self.log(f"enrolment refused: {e}")
                return
            _send_json(tls, {"ok": True, "member": member, "key": key, "rpc_port": port})
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
                _send_json(tls, {"ok": False, "error": "this member is not enrolled, or its key is wrong"})
                self.log(f"authentication refused for {member!r}")
                return
            _send_json(tls, {"ok": True, "rpc_port": port})
            self._attach(member, tls)
            return
        _send_json(tls, {"ok": False, "error": "unknown request"})

    def _attach(self, member: str, tls) -> None:
        mux = Mux(tls)
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


# --- coordinator commands ----------------------------------------------------

def init_group(group: str, relay: str, *, groups_dir: Path = GROUPS_DIR) -> dict:
    """Create a group: its secret and a self-signed certificate. `relay` is the host:port members dial."""
    _check_name(group, "group name")
    host, port = _split_hostport(relay)
    relay = f"{host}:{port}"
    directory = _group_dir(groups_dir, group)
    if (directory / "group.json").exists():
        raise GroupError(f"group {group} already exists at {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    cert, key = directory / "cert.pem", directory / "key.pem"
    _make_certificate(cert, key, group)
    der = ssl.PEM_cert_to_DER_cert(cert.read_text())
    data = {"v": TOKEN_VERSION, "group": group, "relay": relay, "cert_fp": cert_fingerprint(der),
            "secret": secrets.token_hex(32), "members": {}, "used": {}}
    _write_private(directory / "group.json", data)
    return data


def issue(group: str, member: str, *, hours: float = DEFAULT_TOKEN_HOURS,
          groups_dir: Path = GROUPS_DIR, now: float | None = None) -> str:
    state = GroupState.load(_group_dir(groups_dir, group) / "group.json")
    _check_name(member, "member name")
    if member in state.data["members"]:
        raise GroupError(f"{member} is already enrolled in {group}")
    now = time.time() if now is None else now
    return issue_token(bytes.fromhex(state.data["secret"]), group=group, member=member,
                       relay=state.relay, cert_fp=state.cert_fp, expires=now + hours * 3600)


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
    tls = client_connect(payload["r"], payload["f"])
    try:
        _send_json(tls, {"op": "enrol", "token": token})
        reply = _read_json(tls)
    finally:
        _close_quietly(tls)
    if not reply.get("ok"):
        raise GroupError(str(reply.get("error") or "the coordinator refused this token"))
    if reply.get("member") != payload["m"] or not isinstance(reply.get("key"), str):
        raise GroupError("the coordinator's reply does not match the token")
    member = {"v": TOKEN_VERSION, "group": payload["g"], "member": payload["m"],
              "relay": payload["r"], "cert_fp": payload["f"], "key": reply["key"],
              "rpc_port": reply.get("rpc_port")}
    _write_private(member_path(groups_dir, payload["g"]), member)
    return member


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


def _stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_worker(member: dict, rpc_binary: Path, *, log_path: Path, log=print,
               stop: threading.Event | None = None) -> None:
    """Start this member's ggml-rpc-server and keep an outbound link to the coordinator.

    The RPC server stays up across reconnects, so a dropped link does not reload
    anything. Refusal of the member's key, or a certificate that is not the pinned
    one, stops the agent: retrying would not change either answer.
    """
    stop = stop or threading.Event()
    rpc_port = _free_loopback_port()
    with open(log_path, "ab") as log_file:
        rpc = subprocess.Popen([str(rpc_binary), "--host", "127.0.0.1", "--port", str(rpc_port)],
                               stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT)
    try:
        _wait_for_listener(rpc_port, rpc, log_path)
        log(f"RPC worker listening on 127.0.0.1:{rpc_port}; connecting to {member['relay']}")
        delay = 1.0
        while not stop.is_set():
            if rpc.poll() is not None:
                raise GroupError(f"the RPC worker stopped (exit code {rpc.returncode}); see {log_path}")
            try:
                tls = client_connect(member["relay"], member["cert_fp"])
            except OSError as e:
                log(f"coordinator unreachable ({e}); retrying in {delay:.0f}s")
                stop.wait(delay)
                delay = min(delay * 2, RECONNECT_MAX_SECONDS)
                continue
            try:
                _send_json(tls, {"op": "auth", "member": member["member"], "key": member["key"]})
                reply = _read_json(tls)
            except (OSError, GroupError) as e:
                _close_quietly(tls)
                log(f"handshake failed ({e}); retrying in {delay:.0f}s")
                stop.wait(delay)
                delay = min(delay * 2, RECONNECT_MAX_SECONDS)
                continue
            if not reply.get("ok"):
                _close_quietly(tls)
                raise GroupError(str(reply.get("error") or "the coordinator refused this member"))
            log("connected to the coordinator")
            delay = 1.0
            mux = Mux(tls, on_open=lambda _stream: _connect_rpc(rpc_port))
            mux.run()
            if stop.is_set():
                break
            log(f"lost the coordinator ({mux.error or 'closed'}); reconnecting in {delay:.0f}s")
            stop.wait(delay)
    finally:
        _stop_process(rpc)


def join_group(token: str, *, rpc_binary: Path, groups_dir: Path = GROUPS_DIR, log=print,
               stop: threading.Event | None = None, now: float | None = None) -> None:
    """Enrol with `token` if this machine is not yet a member, then run the worker."""
    payload, _, _ = read_token(token)
    group = payload["g"]
    current = load_member(groups_dir, group)
    if current is None:
        current = enrol(token, groups_dir=groups_dir, now=now)
        log(f"enrolled as {current['member']} in group {group}")
    elif current.get("member") != payload["m"]:
        raise GroupError(f"this machine is already enrolled in group {group} as {current.get('member')}")
    else:
        log(f"already enrolled in group {group} as {current['member']}")
    run_worker(current, rpc_binary, log_path=_group_dir(groups_dir, group) / "rpc-server.log",
               log=log, stop=stop)


# --- admin CLI (coordinator) -------------------------------------------------

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
    init.add_argument("--relay", required=True, help="host:port that members dial, e.g. 192.0.2.10:7443")

    issue_cmd = commands.add_parser("issue", help="print a one-time token for a new member")
    issue_cmd.add_argument("--group", required=True)
    issue_cmd.add_argument("--member", required=True)
    issue_cmd.add_argument("--hours", type=float, default=DEFAULT_TOKEN_HOURS)

    members = commands.add_parser("members", help="list enrolled members and their loopback RPC endpoints")
    members.add_argument("--group", required=True)

    relay = commands.add_parser("relay", help="serve enrolments and member links in the foreground")
    relay.add_argument("--group", required=True)
    relay.add_argument("--listen", help="host:port to bind (default: the port from init, on all interfaces)")

    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            data = init_group(args.group, args.relay)
            print(f"created group {data['group']} with relay {data['relay']}")
            print(f"certificate fingerprint {data['cert_fp']}")
            print(f"next: python3 join/group.py relay --group {data['group']}")
            print(f"then: python3 join/group.py issue --group {data['group']} --member NAME")
        elif args.command == "issue":
            print(issue(args.group, args.member, hours=args.hours))
        elif args.command == "members":
            state = GroupState.load(_group_dir(GROUPS_DIR, args.group) / "group.json")
            lines = _members_report(state)
            print("\n".join(lines) if lines else "no members enrolled yet")
            endpoints = [f"127.0.0.1:{entry['rpc_port']}" for entry in state.data["members"].values()]
            print("rpc_servers: " + json.dumps(sorted(endpoints)))
        elif args.command == "relay":
            state_path = _group_dir(GROUPS_DIR, args.group) / "group.json"
            state = GroupState.load(state_path)
            host, port = _split_hostport(args.listen or f"0.0.0.0:{_split_hostport(state.relay)[1]}")
            server = Relay(state_path, host, port)
            address = server.start()
            print(f"relay listening on {address[0]}:{address[1]} for group {state.group}; ctrl+c stops it")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("stopping the relay")
            finally:
                server.close()
    except GroupError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (OSError, ssl.SSLError) as e:
        print(f"error: could not reach the coordinator's relay ({e})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
