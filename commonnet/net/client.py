"""Client for the block server: health, open, forward and close, against a tunnel URL.

The endpoint is the block's base URL, such as the tunnel URL the launcher writes.

Retries are safe because of how the server behaves (see server.py):
- open and close are idempotent, so they retry on any transient failure.
- forward retries too. A repeated step is answered from the server's saved reply
  rather than run again, so the cache is written once.

What is retried: a refused or dropped connection, a timeout, a dropped response,
and HTTP 502, 503 and 504 (what a tunnel answers while its far end is away). What
is not retried: any other status. Those raise RemoteError at once, because
repeating a 4xx or a block failure would only repeat it.

After the last attempt the client raises Unreachable. The coordinator should treat
that as the member being down for now.
"""
from __future__ import annotations

import collections
import http.client
import json
import socket
import threading
import time
import urllib.parse

from commonnet.net import auth, dns, protocol
from commonnet.net.protocol import MAX_REQUEST_BYTES, ProtocolError, Tensor

FORWARD_TIMEOUT_SECONDS = 60.0
CONTROL_TIMEOUT_SECONDS = 10.0
DEFAULT_RETRIES = 3
RETRY_BACKOFF_SECONDS = 0.5
RETRY_MAX_BACKOFF_SECONDS = 4.0
RETRY_STATUSES = frozenset((502, 503, 504))


class BlockError(Exception):
    """Base for failures talking to a block server."""


class RemoteError(BlockError):
    """The block server answered with a status that is not retried."""

    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


class Unreachable(BlockError):
    """Every attempt failed at the transport or with a retryable status."""


def _join_url(endpoint: str, path: str) -> str:
    if not isinstance(endpoint, str) or not endpoint.startswith(("http://", "https://")):
        raise ValueError(f"endpoint must be an http(s) URL, got {endpoint!r}")
    return endpoint.rstrip("/") + path


def _reason(error: BaseException) -> str:
    return str(getattr(error, "reason", None) or error) or type(error).__name__


def _error_text(raw: bytes, reason: str) -> str:
    try:
        obj = json.loads(raw.decode("utf-8"))
        if isinstance(obj, dict) and isinstance(obj.get("error"), str):
            return obj["error"]
    except ValueError:
        pass
    return reason


def _server_ms(header) -> float | None:
    """The block's own compute time from a Server-Timing header, if it sent one."""
    for part in (header or "").split(","):
        name, _, rest = part.strip().partition(";")
        if name == "compute" and rest.startswith("dur="):
            try:
                return float(rest[4:])
            except ValueError:
                return None
    return None


def _connect_resolved(conn, tls: bool) -> None:
    """Connect to the resolved address, keeping the hostname for the Host header and SNI."""
    address = dns.resolve(conn.host, conn.port)
    if getattr(conn, "public_only", False):
        # Resolve to a numeric address once, then connect to that exact address. Validating
        # a hostname and resolving it again would leave a DNS-rebinding window.
        import ipaddress
        addresses = socket.getaddrinfo(address, conn.port, type=socket.SOCK_STREAM)
        choices = [a[4][0] for a in addresses if ipaddress.ip_address(a[4][0]).is_global
                   and not ipaddress.ip_address(a[4][0]).is_multicast]
        if len(choices) != len(addresses) or not choices:
            raise OSError("chain destination is not a public address")
        address = choices[0]
    sock = socket.create_connection((address, conn.port), conn.timeout, conn.source_address)
    if tls:
        try:
            sock = conn._context.wrap_socket(sock, server_hostname=conn.host)
        except BaseException:
            sock.close()
            raise
    conn.sock = sock


class _HTTPConnection(http.client.HTTPConnection):
    def connect(self):
        _connect_resolved(self, False)


class _HTTPSConnection(http.client.HTTPSConnection):
    def connect(self):
        _connect_resolved(self, True)


_TOKEN_FROM_ENV = object()


class BlockClient:
    """Calls block servers, keeping one connection open per endpoint between calls.

    A new connection to a tunnel costs several round trips (TCP, TLS), so the
    connection is reused for the next call. A reused connection the far side has
    closed since (idle timeout, restart) is replaced and the call sent again at once,
    without using up a retry. Safe for the same reason retries are: see the module note.

    Every call is timed into self.timings (the newest 4096): path, total_ms, whether the
    connection was reused, request and reply bytes, and server_ms (the block's own
    compute time, when the server reports it). total_ms - server_ms is the network and
    client cost of that hop.

    The token defaults to the one the server reads (COMMONNET_BLOCK_TOKEN). Pass
    token=None to send no Authorization header, which a real server refuses with 401.

    The opener ignores proxy environment variables: the endpoint is a tunnel URL,
    and a proxy in the environment must not carry it.
    """

    def __init__(self, *, token=_TOKEN_FROM_ENV, retries: int = DEFAULT_RETRIES,
                 backoff: float = RETRY_BACKOFF_SECONDS,
                 max_backoff: float = RETRY_MAX_BACKOFF_SECONDS,
                 max_request_bytes: int = MAX_REQUEST_BYTES, sleep=time.sleep,
                 pin=None, ssl_context=None, public_only: bool = False):
        """pin (default None) pins the certificate of https endpoints by sha256 fingerprint
        (commonnet/net/secure.py): a fingerprint string for every https endpoint, or a dict
        mapping "host:port" to one, which wins over the string form. An endpoint with a pin is
        never verified through the system CAs. ssl_context (default None) replaces the context
        for https endpoints without a pin. With neither, behaviour is unchanged."""
        if retries < 0:
            raise ValueError("retries must be zero or more")
        self.token = auth.load_token() if token is _TOKEN_FROM_ENV else (
            None if token is None else auth.check_token(token))
        self.retries = retries
        self.backoff = backoff
        self.max_backoff = max_backoff
        self.max_request_bytes = max_request_bytes
        self._sleep = sleep
        self._pin = pin
        self._ssl_context = ssl_context
        self._public_only = public_only
        self._pinned: dict[str, object] = {}
        self._idle: dict[tuple, http.client.HTTPConnection] = {}
        self._lock = threading.Lock()
        self.timings: collections.deque = collections.deque(maxlen=4096)

    def health(self, endpoint: str, timeout: float = CONTROL_TIMEOUT_SECONDS) -> tuple:
        """Returns (layers, device)."""
        return protocol.decode_health(self._call("GET", endpoint, "/health", None, timeout))

    def health_caps(self, endpoint: str, timeout: float = CONTROL_TIMEOUT_SECONDS) -> tuple:
        """The capability names the server advertises (empty from an old server, which ignores the query)."""
        return protocol.decode_health_caps(self._call("GET", endpoint, "/health?caps=1", None, timeout))

    def health_info(self, endpoint: str, timeout: float = CONTROL_TIMEOUT_SECONDS) -> tuple:
        """Validated (layers, device, capabilities) from one authenticated health request."""
        reply = self._call("GET", endpoint, "/health?caps=1", None, timeout)
        layers, device = protocol.decode_health(reply)
        return layers, device, protocol.decode_health_caps(reply)

    def chain(self, endpoint: str, body: dict, timeout: float = FORWARD_TIMEOUT_SECONDS) -> dict:
        """One route traversal; only token IDs and small verification metadata return."""
        from commonnet.net.chain import check_reply
        return check_reply(self._call("POST", endpoint, "/chain", body, timeout))

    def open(self, endpoint: str, session_id: str, timeout: float = CONTROL_TIMEOUT_SECONDS) -> None:
        body = protocol.encode_session_request(session_id)
        protocol.decode_ok(self._call("POST", endpoint, "/session", body, timeout))

    def close(self, endpoint: str, session_id: str, timeout: float = CONTROL_TIMEOUT_SECONDS) -> None:
        body = protocol.encode_session_request(session_id)
        protocol.decode_ok(self._call("POST", endpoint, "/close", body, timeout))

    def truncate(self, endpoint: str, session_id: str, length: int, timeout: float = CONTROL_TIMEOUT_SECONDS) -> None:
        """Cut the session back to `length` positions. Idempotent. An old server answers 404 'no endpoint'."""
        body = protocol.encode_truncate_request(session_id, length)
        protocol.decode_ok(self._call("POST", endpoint, "/truncate", body, timeout))

    def forward(self, endpoint: str, session_id: str, hidden: Tensor | None, start_pos: int,
                timeout: float = FORWARD_TIMEOUT_SECONDS, truncate_to: int | None = None, *,
                ids=None, output: str | None = None, which: str | None = None, sampling=None):
        """Returns a Tensor, or with output="ids" a list of token ids. hidden=None sends `ids` instead."""
        if ids is not None or output is not None:
            body = protocol.encode_forward_request_ex(session_id, hidden, start_pos, truncate_to, ids=ids,
                                                      output=output, which=which, sampling=sampling)
            reply = self._call("POST", endpoint, "/forward", body, timeout)
            if output is not None:
                return protocol.decode_ids_reply(reply)
            out = protocol.decode_forward_reply(reply)
            return out
        if not isinstance(hidden, Tensor):
            raise TypeError("forward takes a protocol.Tensor")
        body = protocol.encode_forward_request(session_id, hidden, start_pos, truncate_to)
        out = protocol.decode_forward_reply(self._call("POST", endpoint, "/forward", body, timeout))
        if out.shape != hidden.shape or out.dtype != hidden.dtype:
            raise ProtocolError(
                f"forward reply is shape {list(out.shape)} {out.dtype}; "
                f"sent {list(hidden.shape)} {hidden.dtype}"
            )
        return out

    def close_connections(self) -> None:
        with self._lock:
            idle, self._idle = list(self._idle.values()), {}
        for conn in idle:
            conn.close()

    def _context_for(self, host: str, port: int):
        pin = self._pin.get(f"{host}:{port}") if isinstance(self._pin, dict) else self._pin
        if pin:
            from commonnet.net import secure
            with self._lock:
                ctx = self._pinned.get(pin)
                if ctx is None:
                    ctx = self._pinned[pin] = secure.client_ssl_context(pin)
            return ctx
        return self._ssl_context  # None: http.client builds its default verifying context

    def _take(self, key: tuple, timeout: float):
        with self._lock:
            conn = self._idle.pop(key, None)
        if conn is None:
            if key[0] == "https":
                conn = _HTTPSConnection(key[1], key[2], timeout=timeout,
                                        context=self._context_for(key[1], key[2]))
            else:
                conn = _HTTPConnection(key[1], key[2], timeout=timeout)
            conn.public_only = self._public_only
            return conn, False
        conn.timeout = timeout
        try:
            if conn.sock is not None:
                conn.sock.settimeout(timeout)
        except OSError:
            conn.close()
            return self._take(key, timeout)[0], False
        return conn, True

    def _keep(self, key: tuple, conn) -> None:
        with self._lock:
            old, self._idle[key] = self._idle.get(key), conn
        if old is not None and old is not conn:
            old.close()

    def _once(self, method: str, parts, path: str, data, headers: dict, timeout: float):
        """One request, over the kept connection when there is one. Returns (status, reason, raw, reused, server_ms)."""
        key = (parts.scheme, parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        for _ in range(2):
            conn, reused = self._take(key, timeout)
            try:
                conn.request(method, target, body=data, headers=headers)
                response = conn.getresponse()
                raw = response.read()
            except (OSError, http.client.HTTPException):
                conn.close()
                if reused:
                    continue  # went stale while idle: send again on a fresh connection
                raise
            if response.will_close:
                conn.close()
            else:
                self._keep(key, conn)
            return (response.status, response.reason, raw, reused,
                    _server_ms(response.getheader("Server-Timing")))
        raise AssertionError("unreachable")

    def _call(self, method: str, endpoint: str, path: str, body: dict | None, timeout: float) -> dict:
        url = _join_url(endpoint, path)
        parts = urllib.parse.urlsplit(url)
        data = None
        headers = dict(auth.authorization_headers(self.token))
        if body is not None:
            data = json.dumps(body, separators=(",", ":")).encode("utf-8")
            if len(data) > self.max_request_bytes:
                raise ValueError(
                    f"request is {len(data)} bytes; the limit is {self.max_request_bytes}. "
                    "Send the prefill in smaller pieces."
                )
            headers["Content-Type"] = "application/json"
        attempts = self.retries + 1
        reason = ""
        for attempt in range(attempts):
            started = time.perf_counter()
            try:
                status, status_reason, raw, reused, server_ms = self._once(
                    method, parts, path, data, headers, timeout)
            except (OSError, http.client.HTTPException) as e:
                if type(e).__name__ == "PinMismatch":  # an impostor or a stale pin: retrying cannot help
                    raise Unreachable(f"{method} {url}: {e}") from None
                reason = _reason(e)
            else:
                self.timings.append({
                    "path": path, "status": status, "reused": reused,
                    "total_ms": (time.perf_counter() - started) * 1000, "server_ms": server_ms,
                    "bytes_out": len(data or b""), "bytes_in": len(raw)})
                if status >= 400:
                    message = _error_text(raw, status_reason)
                    if status not in RETRY_STATUSES:
                        raise RemoteError(status, message) from None
                    reason = f"HTTP {status}: {message}"
                else:
                    try:
                        return json.loads(raw.decode("utf-8"))
                    except ValueError:
                        raise ProtocolError(f"{url} did not answer with JSON") from None
            if attempt + 1 < attempts:
                self._sleep(min(self.backoff * (2 ** attempt), self.max_backoff))
        raise Unreachable(f"{method} {url} failed {attempts} times; last: {reason}")
