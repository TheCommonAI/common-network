"""The block server: the HTTP JSON interface over one block object, on loopback.

Run it with a real block, given as MODULE:FACTORY and called as FACTORY(start, end),
or with --stub, which serves StubBlock. The stub is an identity block and needs
no torch:

    python3 -m commonnet.net.server --stub --layers 0 4 --port 0

The block object must provide:
    layers                  (start, end), a half-open range of layer indices
    device                  a string, reported by /health
    open(session_id)        create per-session state, such as a KV cache
    close(session_id)       drop that state
    forward(session_id, hidden, start_pos) -> Tensor

hidden and the return value are protocol.Tensor values (raw bytes, shape, dtype).
The output must have the same shape and dtype as the input. The server calls the
block one request at a time, so the block needs no locking of its own.

What the coordinator can rely on:
- open and close are idempotent. Opening an open session does nothing, and
  closing an unknown session is ok.
- forward needs an open session, or the answer is 404.
- A forward that repeats the last step of a session (same start_pos and same
  bytes) returns the saved reply and does not run the block again. A retry after
  a lost reply is therefore safe.
- A body over max_request_bytes is refused with 413 before it is read. A chunked
  body is refused with 411.

The server binds 127.0.0.1 only. It is reached from outside through a tunnel, and
it has no authentication of its own. See commonnet/net/tunnel.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import socketserver
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from commonnet.net import auth, protocol
from commonnet.net.chain import ChainUnavailable
from commonnet.net.protocol import MAX_REQUEST_BYTES, ProtocolError, Tensor

LOOPBACK = "127.0.0.1"
HANDLER_TIMEOUT_SECONDS = 60.0
TLS_HANDSHAKE_TIMEOUT_SECONDS = 10.0
POST_PATHS = frozenset(("/session", "/forward", "/close", "/truncate", "/chain"))


class UnknownSession(LookupError):
    """A forward for a session that is not open."""


class BlockOutputError(RuntimeError):
    """The block returned something other than a tensor of the input's shape and dtype."""


class StubBlock:
    """An identity block with no weights, for tests and for --stub.

    It records each call in `calls` as (operation, session_id, start_pos), so a
    test can see how many times the block really ran. `delay` makes forward slow
    enough for overlapping calls to show up in `overlapped`.
    """

    device = "stub"

    def __init__(self, start: int, end: int, *, delay: float = 0.0):
        self.layers = protocol.check_layers((start, end))
        self.delay = delay
        self.calls: list[tuple] = []
        self.overlapped = False
        self._open: set[str] = set()
        self._active = 0
        self._guard = threading.Lock()

    def open(self, session_id: str) -> None:
        self.calls.append(("open", session_id, None))
        self._open.add(session_id)

    def close(self, session_id: str) -> None:
        self.calls.append(("close", session_id, None))
        self._open.discard(session_id)

    def truncate(self, session_id: str, length: int) -> None:
        self.calls.append(("truncate", session_id, length))

    def forward(self, session_id: str, hidden: Tensor, start_pos: int) -> Tensor:
        if session_id not in self._open:
            raise KeyError(f"stub block has no session {session_id!r}")
        with self._guard:
            self._active += 1
            if self._active > 1:
                self.overlapped = True
        try:
            self.calls.append(("forward", session_id, start_pos))
            if self.delay:
                time.sleep(self.delay)
            return hidden
        finally:
            with self._guard:
                self._active -= 1


class _Session:
    __slots__ = ("last", "chain_last", "chain_lock")

    def __init__(self) -> None:
        self.last: tuple | None = None  # (step key, reply Tensor) of the last forward
        self.chain_last = None
        self.chain_lock = threading.Lock()


def _step_key(hidden: Tensor | None, start_pos: int, ids=None, output=None, which="last", sampling=None) -> tuple:
    """The retry/idempotency key of a step. It includes the mode: the same bytes asked for hidden states and for
    token ids (or for the last vs all positions, or another sampling) are different steps."""
    digest = hashlib.sha256()
    if hidden is not None:
        digest.update(repr(hidden.shape).encode("ascii"))
        digest.update(hidden.dtype.encode("ascii"))
        digest.update(hidden.data)
    if ids is not None:
        digest.update(b"ids:" + repr(list(ids)).encode("ascii"))
    digest.update(repr((output, which if output else None, sorted((sampling or {}).items()))).encode("ascii"))
    return (start_pos, digest.digest())


def _check_output(out, hidden: Tensor) -> None:
    if not isinstance(out, Tensor):
        raise BlockOutputError(f"block returned {type(out).__name__}, not a Tensor")
    if out.shape != hidden.shape or out.dtype != hidden.dtype:
        raise BlockOutputError(
            f"block returned shape {list(out.shape)} {out.dtype}; "
            f"input was {list(hidden.shape)} {hidden.dtype}"
        )


class BlockServer:
    """Serves one block over HTTP on 127.0.0.1.

    The session table and every call into the block share one lock. /health reads
    only the immutable layers and device, so it never waits behind a forward.
    """

    def __init__(self, block, host: str = LOOPBACK, port: int = 0, *,
                 max_request_bytes: int = MAX_REQUEST_BYTES, token: str | None = None,
                 extra_handlers: dict | None = None, ssl_context=None, allow_chain_private: bool = False):
        """token=None serves without authentication. Only tests and --stub do that;
        the command line refuses to start a real block without a token.

        extra_handlers maps ("GET" | "POST", path) to callable(body_or_None) -> (status, obj).
        They are public (no bearer token): a node uses them for the manifest and gossip, whose
        contents are signed. They share the request-size limit. The dict may be changed after
        construction.

        ssl_context (default None) serves TLS (see commonnet/net/secure.py). It is also what allows
        a host other than 127.0.0.1, for a directly reachable peer, and then a token is required."""
        if host != LOOPBACK and ssl_context is None:
            raise ValueError("the block server listens on 127.0.0.1 only; the tunnel is what reaches it")
        if host != LOOPBACK and token is None:
            raise ValueError("a block server that listens beyond loopback needs a token")
        if max_request_bytes < 1:
            raise ValueError("max_request_bytes must be positive")
        self.token = None if token is None else auth.check_token(token)
        self.block = block
        self.layers = protocol.check_layers(block.layers)
        self.device = str(block.device)
        self.max_request_bytes = max_request_bytes
        self.extra_handlers = dict(extra_handlers or {})
        # Explicit local-test opt-in. Public peers may never route requests to LAN,
        # loopback or cloud metadata services on the authenticated caller's behalf.
        self.allow_chain_private = bool(allow_chain_private)
        self._chain_clients = {}
        self._lock = threading.Lock()
        self._sessions: dict[str, _Session] = {}
        self._httpd = _Httpd((host, port), _Handler)
        self._httpd.block_server = self
        self._httpd.ssl_context = ssl_context
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    @property
    def url(self) -> str:
        if self._httpd.ssl_context is None:
            return f"http://{LOOPBACK}:{self.port}"
        return f"https://{self._httpd.server_address[0]}:{self.port}"

    def start(self) -> "BlockServer":
        """Serve in a background thread. Pair with close()."""
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="block-server", daemon=True)
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        self._httpd.serve_forever()

    def stop(self) -> None:
        """Stop serving and release the port. Safe to call after serve_forever returns."""
        if self._thread is not None:
            self._httpd.shutdown()
            self._thread.join(timeout=5)
            self._thread = None
        self._httpd.server_close()
        for client in self._chain_clients.values():
            client.close_connections()
        self._chain_clients.clear()

    def open(self, session_id: str) -> None:
        protocol.check_session_id(session_id)
        with self._lock:
            if session_id in self._sessions:
                return
            self.block.open(session_id)
            self._sessions[session_id] = _Session()

    def close(self, session_id: str) -> None:
        with self._lock:
            if session_id not in self._sessions:
                return
            self.block.close(session_id)
            del self._sessions[session_id]

    def truncate(self, session_id: str, length: int) -> None:
        """Cut the session back to `length` positions (speculative decoding's rollback).

        Idempotent: a session already at or below `length` is left alone. The saved last-step reply is dropped,
        because a forward that repeats it after a rollback must run again, not be answered from the saved reply.
        Raises NotImplementedError if the block cannot truncate."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise UnknownSession(f"session {session_id!r} is not open")
            fn = getattr(self.block, "truncate", None)
            if fn is None:
                raise NotImplementedError("this block cannot truncate a session")
            fn(session_id, length)
            session.last = None
            session.chain_last = None

    @property
    def caps(self) -> tuple:
        extra = ("chain_v1", "chain_private_v1") if self.allow_chain_private else ("chain_v1",)
        return tuple(getattr(self.block, "caps", ()) or ()) + extra + ("activation_int8_group_v1",)

    def forward(self, session_id: str, hidden: Tensor | None, start_pos: int, truncate_to: int | None = None, *,
                ids=None, output: str | None = None, which: str = "last", sampling=None):
        """Returns a Tensor, or with output="ids" a list of token ids (all positions of the step for which="all",
        else the last). hidden=None means `ids` are embedded here first (needs the ids_in capability)."""
        if ids is not None and "ids_in" not in self.caps:
            raise NotImplementedError("this block does not take token ids (no ids_in)")
        if output is not None and "head_out" not in self.caps:
            raise NotImplementedError("this block does not return token ids (no head_out)")
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise UnknownSession(f"session {session_id!r} is not open")
            key = _step_key(hidden, start_pos, ids, output, which, sampling) + (truncate_to,)
            if session.last is not None and session.last[0] == key:
                return session.last[1]
            if truncate_to is not None:
                fn = getattr(self.block, "truncate", None)
                if fn is None:
                    raise NotImplementedError("this block cannot truncate a session")
                fn(session_id, truncate_to)
            if ids is not None:
                hidden = self.block.embed(ids)
            out = self.block.forward(session_id, hidden, start_pos)
            _check_output(out, hidden)
            if output is not None:
                out = [int(t) for t in self.block.head(out, which, sampling, start_pos)]
            session.last = (key, out)
            return out


class _Httpd(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    block_server: BlockServer
    ssl_context = None

    def finish_request(self, request, client_address):
        # TLS is set up here, in the connection's own thread, so one slow or stalled handshake
        # cannot hold up accepting the next connection.
        if self.ssl_context is None:
            return super().finish_request(request, client_address)
        try:
            request.settimeout(TLS_HANDSHAKE_TIMEOUT_SECONDS)
            secured = self.ssl_context.wrap_socket(request, server_side=True)
        except OSError:  # plain HTTP to a TLS port, a scanner, a client that pinned another cert
            request.close()
            return
        try:
            super().finish_request(secured, client_address)
        finally:
            secured.close()

    def server_bind(self) -> None:
        # HTTPServer.server_bind resolves this machine's name, which can stall on a
        # loopback address. The name is never used, so skip the lookup.
        socketserver.TCPServer.server_bind(self)
        self.server_name = LOOPBACK
        self.server_port = self.server_address[1]


def _error(message: str) -> dict:
    return protocol.encode_error(message)


class _Handler(BaseHTTPRequestHandler):
    server_version = "commonnet-block/0.1"
    sys_version = ""
    # HTTP/1.1 keeps the connection open between requests, so a client pays the TCP and
    # TLS setup (several tunnel round trips) once instead of on every token. Every reply
    # carries a Content-Length, and a reply sent while a request body is still unread
    # closes the connection (see _send).
    protocol_version = "HTTP/1.1"
    timeout = HANDLER_TIMEOUT_SECONDS

    def log_message(self, format, *args):  # keep the launcher's log clean
        pass

    @property
    def _block(self) -> BlockServer:
        return self.server.block_server

    def _send(self, status: int, obj: dict) -> None:
        body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        compute_ms = getattr(self, "_compute_ms", None)
        if compute_ms is not None:
            self.send_header("Server-Timing", f"compute;dur={compute_ms:.2f}")
        if getattr(self, "_unread", False):
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        token = self._block.token
        if token is None:
            return True
        return auth.token_matches(token, self.headers.get("Authorization"))

    def _path(self) -> str:
        return self.path.split("?", 1)[0]

    def _extra(self, method: str, path: str):
        return self._block.extra_handlers.get((method, path))

    def _run_extra(self, handler, body) -> None:
        try:
            status, obj = handler(body)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            self._send(500, _error(f"handler failed: {type(e).__name__}"[:300]))
        else:
            self._send(status, obj)

    def do_GET(self) -> None:
        self._unread, self._compute_ms = False, None
        path = self._path()
        handler = self._extra("GET", path)
        if handler is not None:
            self._run_extra(handler, None)
            return
        if not self._authorized():
            self._send(401, _error("missing or wrong bearer token"))
            return
        if path == "/health":
            caps = self._block.caps if "caps=1" in self.path.partition("?")[2].split("&") else None
            self._send(200, protocol.encode_health(self._block.layers, self._block.device, caps))
        elif path in POST_PATHS:
            self._send(405, _error(f"{path} takes POST"))
        else:
            self._send(404, _error(f"no endpoint {path}"))

    def do_POST(self) -> None:
        self._unread, self._compute_ms = True, None  # a body follows until _read_body has it
        path = self._path()
        handler = self._extra("POST", path)
        if handler is not None:
            body = self._read_body()
            if body is None:
                return  # the reply has already been sent
            try:
                obj = json.loads(body.decode("utf-8"))
            except ValueError:
                self._send(400, _error("body is not JSON"))
                return
            self._run_extra(handler, obj)
            return
        if path == "/health":
            self._send(405, _error("/health takes GET"))
            return
        if path not in POST_PATHS:
            self._send(404, _error(f"no endpoint {path}"))
            return
        body = self._read_body()
        if body is None:
            return  # the reply has already been sent
        if not self._authorized():
            # The body is read first, so the refusal reaches the client intact.
            self._send(401, _error("missing or wrong bearer token"))
            return
        try:
            obj = json.loads(body.decode("utf-8"))
        except ValueError:
            self._send(400, _error("body is not JSON"))
            return
        try:
            reply = self._dispatch(path, obj)
        except ChainUnavailable as e:
            self._send(409 if e.lost else 502, _error(str(e)))
        except ProtocolError as e:
            self._send(400, _error(str(e)))
        except UnknownSession as e:
            self._send(404, _error(str(e)))
        except NotImplementedError as e:
            self._send(501, _error(str(e)))
        except Exception as e:  # the block failed, or returned something invalid
            traceback.print_exc(file=sys.stderr)
            self._send(500, _error(f"block failed: {type(e).__name__}: {e}"[:300]))
        else:
            self._send(200, reply)

    def _dispatch(self, path: str, obj) -> dict:
        if path == "/chain":
            from commonnet.net.chain import run
            started = time.perf_counter()
            reply = run(self._block, obj)
            self._compute_ms = (time.perf_counter() - started) * 1000
            return reply
        if path == "/session":
            self._block.open(protocol.decode_session_request(obj))
            return protocol.encode_ok()
        if path == "/close":
            self._block.close(protocol.decode_session_request(obj))
            return protocol.encode_ok()
        if path == "/truncate":
            session_id, length = protocol.decode_truncate_request(obj)
            self._block.truncate(session_id, length)
            return protocol.encode_ok()
        req = protocol.decode_forward_request_ex(obj)
        started = time.perf_counter()
        if req.ids is None and req.output is None:  # the original request shape: call the block as before
            out = (self._block.forward(req.session_id, req.hidden, req.start_pos) if req.truncate_to is None
                   else self._block.forward(req.session_id, req.hidden, req.start_pos, req.truncate_to))
            self._compute_ms = (time.perf_counter() - started) * 1000
            return protocol.encode_forward_reply(out)
        out = self._block.forward(req.session_id, req.hidden, req.start_pos, req.truncate_to, ids=req.ids,
                                  output=req.output, which=req.which, sampling=req.sampling)
        self._compute_ms = (time.perf_counter() - started) * 1000
        return protocol.encode_ids_reply(out) if req.output else protocol.encode_forward_reply(out)

    def _read_body(self) -> bytes | None:
        if self.headers.get("Transfer-Encoding"):
            self._send(411, _error("chunked bodies are refused; send a Content-Length"))
            return None
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._send(411, _error("Content-Length is required"))
            return None
        try:
            length = int(raw_length)
        except ValueError:
            self._send(400, _error("Content-Length is not a number"))
            return None
        if length < 0:
            self._send(400, _error("Content-Length is negative"))
            return None
        limit = self._block.max_request_bytes
        if length > limit:
            # Refused without reading the body. The connection closes after this
            # reply (see _send), so the unread bytes are never waited for.
            self._send(413, _error(f"request is {length} bytes; the limit is {limit}"))
            return None
        body = self.rfile.read(length)
        if len(body) != length:
            self._send(400, _error("body is shorter than its Content-Length"))
            return None
        self._unread = False
        return body


def _load_block(spec: str, start: int, end: int):
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise SystemExit("--block takes MODULE:FACTORY")
    factory = getattr(importlib.import_module(module_name), attribute)
    return factory(start, end)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve one block on 127.0.0.1 over HTTP JSON.")
    parser.add_argument("--port", type=int, default=0, help="loopback port (default: any free port)")
    parser.add_argument("--layers", nargs=2, type=int, metavar=("START", "END"), required=True,
                        help="the half-open layer range this block serves")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--stub", action="store_true", help="serve the identity stub block (no torch)")
    source.add_argument("--block", metavar="MODULE:FACTORY",
                        help="serve FACTORY(start, end) from MODULE")
    parser.add_argument("--max-request-bytes", type=int, default=MAX_REQUEST_BYTES)
    parser.add_argument("--token-file", metavar="PATH",
                        help=f"file holding the bearer token (or set {auth.TOKEN_ENV})")
    args = parser.parse_args(argv)

    token = None
    if not args.stub:
        try:
            token = auth.load_token(args.token_file)
        except auth.TokenError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if token is None:
            print(f"error: a real block needs a token: set {auth.TOKEN_ENV} or pass --token-file "
                  "(--stub serves without one)", file=sys.stderr)
            return 2

    start, end = args.layers
    block = StubBlock(start, end) if args.stub else _load_block(args.block, start, end)
    server = BlockServer(block, LOOPBACK, args.port, max_request_bytes=args.max_request_bytes, token=token)
    print(f"block server on {server.url} serving layers [{start}, {end}) on {server.device}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        getattr(block, "shutdown", lambda: None)()
    return 0


if __name__ == "__main__":
    sys.exit(main())
