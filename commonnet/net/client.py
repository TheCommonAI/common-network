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

import http.client
import json
import time
import urllib.error
import urllib.request

from commonnet.net import auth, protocol
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


def _error_text(error: urllib.error.HTTPError) -> str:
    try:
        obj = json.loads(error.read().decode("utf-8"))
        if isinstance(obj, dict) and isinstance(obj.get("error"), str):
            return obj["error"]
    except (OSError, ValueError):
        pass
    finally:
        error.close()
    return str(error.reason)


_TOKEN_FROM_ENV = object()


class BlockClient:
    """Calls one block server per call. Holds no connection between calls.

    The token defaults to the one the server reads (COMMONNET_BLOCK_TOKEN). Pass
    token=None to send no Authorization header, which a real server refuses with 401.

    The opener ignores proxy environment variables: the endpoint is a tunnel URL,
    and a proxy in the environment must not carry it.
    """

    def __init__(self, *, token=_TOKEN_FROM_ENV, retries: int = DEFAULT_RETRIES,
                 backoff: float = RETRY_BACKOFF_SECONDS,
                 max_backoff: float = RETRY_MAX_BACKOFF_SECONDS,
                 max_request_bytes: int = MAX_REQUEST_BYTES, sleep=time.sleep):
        if retries < 0:
            raise ValueError("retries must be zero or more")
        self.token = auth.load_token() if token is _TOKEN_FROM_ENV else (
            None if token is None else auth.check_token(token))
        self.retries = retries
        self.backoff = backoff
        self.max_backoff = max_backoff
        self.max_request_bytes = max_request_bytes
        self._sleep = sleep
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def health(self, endpoint: str, timeout: float = CONTROL_TIMEOUT_SECONDS) -> tuple:
        """Returns (layers, device)."""
        return protocol.decode_health(self._call("GET", endpoint, "/health", None, timeout))

    def open(self, endpoint: str, session_id: str, timeout: float = CONTROL_TIMEOUT_SECONDS) -> None:
        body = protocol.encode_session_request(session_id)
        protocol.decode_ok(self._call("POST", endpoint, "/session", body, timeout))

    def close(self, endpoint: str, session_id: str, timeout: float = CONTROL_TIMEOUT_SECONDS) -> None:
        body = protocol.encode_session_request(session_id)
        protocol.decode_ok(self._call("POST", endpoint, "/close", body, timeout))

    def forward(self, endpoint: str, session_id: str, hidden: Tensor, start_pos: int,
                timeout: float = FORWARD_TIMEOUT_SECONDS) -> Tensor:
        if not isinstance(hidden, Tensor):
            raise TypeError("forward takes a protocol.Tensor")
        body = protocol.encode_forward_request(session_id, hidden, start_pos)
        out = protocol.decode_forward_reply(self._call("POST", endpoint, "/forward", body, timeout))
        if out.shape != hidden.shape or out.dtype != hidden.dtype:
            raise ProtocolError(
                f"forward reply is shape {list(out.shape)} {out.dtype}; "
                f"sent {list(hidden.shape)} {hidden.dtype}"
            )
        return out

    def _call(self, method: str, endpoint: str, path: str, body: dict | None, timeout: float) -> dict:
        url = _join_url(endpoint, path)
        data = None
        if body is not None:
            data = json.dumps(body, separators=(",", ":")).encode("utf-8")
            if len(data) > self.max_request_bytes:
                raise ValueError(
                    f"request is {len(data)} bytes; the limit is {self.max_request_bytes}. "
                    "Send the prefill in smaller pieces."
                )
        attempts = self.retries + 1
        reason = ""
        for attempt in range(attempts):
            request = urllib.request.Request(url, data=data, method=method)
            if data is not None:
                request.add_header("Content-Type", "application/json")
            for name, value in auth.authorization_headers(self.token).items():
                request.add_header(name, value)
            try:
                with self._opener.open(request, timeout=timeout) as response:
                    raw = response.read()
            except urllib.error.HTTPError as e:
                message = _error_text(e)
                if e.code not in RETRY_STATUSES:
                    raise RemoteError(e.code, message) from None
                reason = f"HTTP {e.code}: {message}"
            except (OSError, http.client.HTTPException) as e:
                reason = _reason(e)
            else:
                try:
                    return json.loads(raw.decode("utf-8"))
                except ValueError:
                    raise ProtocolError(f"{url} did not answer with JSON") from None
            if attempt + 1 < attempts:
                self._sleep(min(self.backoff * (2 ** attempt), self.max_backoff))
        raise Unreachable(f"{method} {url} failed {attempts} times; last: {reason}")
