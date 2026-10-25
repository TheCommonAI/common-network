"""A local OpenAI-compatible API in front of the layer network.

    python -m commonnet.serve --network URL|FILE [--port 11440]

    GET  /v1/models
    POST /v1/chat/completions      {"messages": [...], "max_tokens": 256, "temperature": 0.7, "top_p": 1, "stream": true}
    GET  /health                   peers online and coverage per stage

This process is the *client* of the network: it holds the embedding, final norm and head, samples the next token and sends one
token's activations through the chain of peers. It binds 127.0.0.1 only. The route is rebuilt for every request, so a request
starts on the peers that are alive now; a peer that dies mid-request is replaced by a replica (or by a newcomer it waits for).

Requests run one at a time: the client's MLX work lives on a single thread, and the peers keep one session per request.
Activations are visible to every peer on the route and the network is open, so do not send private prompts.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import queue
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT = 11440
MAX_BODY_BYTES = 1 << 20
DEFAULT_MAX_TOKENS = 256
HARD_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7   # a request with no "temperature" is sampled, so it never speculates; `common ask` sends 0
DEFAULT_DRAFT = "mlx-community/Qwen3-0.6B-4bit"
TORCH_DRAFT = "Qwen/Qwen3-0.6B"


def build_default_drafter(target_tokenizer):
    """(drafter, repo): MlxDrafter when mlx_lm imports, else TorchDrafter. Raises if neither can load."""
    try:
        import mlx_lm  # noqa: F401
    except Exception:
        from commonnet.speculative import TorchDrafter
        return TorchDrafter(TORCH_DRAFT, target_tokenizer=target_tokenizer), TORCH_DRAFT
    from commonnet.speculative import MlxDrafter
    return MlxDrafter(DEFAULT_DRAFT, target_tokenizer=target_tokenizer), DEFAULT_DRAFT
CHAIN_CAPABILITY = "chain_v1"   # a hop that can verify drafts (chain_step with drafts)


def draft_enabled() -> bool:
    """COMMONNET_DRAFT=0 (or false/off/no) turns the default-on drafter off."""
    return os.environ.get("COMMONNET_DRAFT", "1").strip().lower() not in ("0", "false", "off", "no")
DEFAULT_MAX_CONTEXT = 8192
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")


class ApiError(Exception):
    def __init__(self, status: int, message: str, kind: str = "invalid_request_error", code: str | None = None):
        super().__init__(message)
        self.status, self.message, self.kind, self.code = status, message, kind, code

    def body(self) -> dict:
        return {"error": {"message": self.message, "type": self.kind, "code": self.code}}


# --- text ---------------------------------------------------------------------------------------

class TextStream:
    """Turns token ids into text pieces, holding back an unfinished multi-byte character."""

    def __init__(self, tokenizer):
        self.tokenizer, self.ids, self.sent = tokenizer, [], 0

    def push(self, token_id: int) -> str:
        self.ids.append(token_id)
        text = self.tokenizer.decode(self.ids)
        if text.endswith("�"):
            return ""
        piece, self.sent = text[self.sent:], len(text)
        return piece

    def flush(self) -> str:
        text = self.tokenizer.decode(self.ids)
        piece, self.sent = text[self.sent:], len(text)
        return piece


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                raise ApiError(400, "only text content is supported")
            parts.append(part["text"])
        return "".join(parts)
    raise ApiError(400, "message content must be a string or a list of text parts")


def parse_request(body: dict) -> dict:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ApiError(400, "messages must be a non-empty list")
    clean = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant"):
            raise ApiError(400, "each message needs a role of system, user or assistant")
        clean.append({"role": m["role"], "content": text_of(m.get("content", ""))})

    def number(key, default, low, high, kind=float):
        value = body.get(key)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ApiError(400, f"{key} must be a number")
        if not low <= value <= high:
            raise ApiError(400, f"{key} must be between {low} and {high}")
        return kind(value)

    max_tokens = number("max_tokens", None, 1, HARD_MAX_TOKENS, int)
    if max_tokens is None:
        max_tokens = number("max_completion_tokens", DEFAULT_MAX_TOKENS, 1, HARD_MAX_TOKENS, int)
    return {"messages": clean, "max_tokens": max_tokens,
            "temperature": number("temperature", DEFAULT_TEMPERATURE, 0.0, 2.0),
            "top_p": number("top_p", 1.0, 0.0001, 1.0), "seed": number("seed", None, 0, 2 ** 63 - 1, int),
            "stream": bool(body.get("stream", False)),
            # Qwen3 opens every answer with a <think> block. On a network that runs at a few tokens a second a chat
            # wants the answer, so thinking is off unless the request turns it on (chat_template_kwargs).
            "template_kwargs": {"enable_thinking": False,
                                **(body.get("chat_template_kwargs") if isinstance(body.get("chat_template_kwargs"), dict) else {})}}


# --- backends -------------------------------------------------------------------------------------

class NetworkBackend:
    """The real backend: a Client4 on one worker thread, a fresh route per request."""

    last_spec = None      # SpecStats of the last speculative request
    last_reason = None    # why the last request did not speculate (None when it did)
    last_k = 0
    _auto_failed = False

    def __init__(self, network, checkpoint: str | None = None, *, fmt: str = "mlx4", hidden_dtype=None,
                 draft: str | None = None, draft_k: int = 4, auto_draft: bool = False):
        import torch
        from transformers import AutoTokenizer
        self.network = network
        manifest = network.manifest
        self.model_name = manifest.model.get("name", manifest.id)
        checkpoint_loader = None
        if checkpoint is None:
            from commonnet import hub
            if not manifest.client_repo:
                raise SystemExit("the manifest names no client_repo; pass --checkpoint DIR")
            source = hub.Hub(manifest.client_repo)
            metadata = getattr(source, "fetch_metadata", None)
            if metadata is None:  # older Hub adapters can still provide a full checkpoint
                checkpoint = source.fetch(client=True)
            else:
                checkpoint = metadata()
                checkpoint_loader = lambda: source.fetch(client=True)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.stop_ids = self._stop_ids()
        self._dtype = hidden_dtype or torch.float32
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="commonnet-client")

        def load():
            from commonnet.client4 import Client4
            client = Client4(checkpoint, _NoTransport(), fmt=fmt, hidden_dtype=self._dtype)
            client._endpoint_checkpoint_loader = checkpoint_loader
            return client

        self.client = self._worker.submit(load).result()
        self.draft_k, self.drafter, self.draft_note = draft_k, None, "off"
        no_spec = manifest.model.get("no_speculation")
        # Default-on speculation: the drafter loads on the first eligible request, not at startup.
        self.auto_draft = bool(auto_draft) and not draft and not no_spec and draft_enabled()
        self._auto_failed = False
        self._drafter_factory = None   # tests replace it; default builds an MlxDrafter on the worker thread
        if self.auto_draft:
            self.draft_note = f"on, loads on first greedy request ({DEFAULT_DRAFT}, k={draft_k})"
        if draft and no_spec:   # the network itself says speculation cannot work on this model
            self.draft_note = f"off: {no_spec}"[:300]
            print(f"speculative decoding is off: {self.draft_note}", file=sys.stderr, flush=True)
        elif draft:
            def load_drafter():
                from commonnet.speculative import MlxDrafter
                return MlxDrafter(draft, target_tokenizer=self.tokenizer)
            try:   # no drafter is never fatal: requests then use plain decoding
                self.drafter = self._worker.submit(load_drafter).result()
                self.draft_note = f"on ({draft}, k={draft_k})"
            except Exception as e:
                self.draft_note = f"off: {type(e).__name__}: {e}"[:300]
                print(f"speculative decoding is off: {self.draft_note}", file=sys.stderr, flush=True)

    def _stop_ids(self) -> list[int]:
        ids = {self.tokenizer.eos_token_id} - {None}
        for token in STOP_TOKENS:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0 and token_id != self.tokenizer.unk_token_id:
                ids.add(token_id)
        return sorted(ids)

    def encode_chat(self, messages, template_kwargs) -> list[int]:
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **template_kwargs)
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def stream(self, prompt_ids, max_tokens, temperature, top_p, seed):
        """An iterator of token ids with .close(). The route and transport are built on the worker thread."""
        return _WorkerStream(self._worker, self._generate, (prompt_ids, max_tokens, temperature, top_p, seed))

    def _generate(self, prompt_ids, max_tokens, temperature, top_p, seed):
        endpoints, transport = self.network.plan()
        self.client.transport = transport
        self.last_spec, self.last_reason, self.last_k = None, None, 0
        try:
            drafter = self.drafter if self.drafter is not None else self._lazy_drafter(endpoints, transport, temperature)
            if drafter is None:
                if self.last_reason is None:
                    self.last_reason = self.draft_note
                yield from self.client.generate_stream(endpoints, prompt_ids, max_tokens, temperature=temperature,
                                                       top_p=top_p, seed=seed, stop_ids=self.stop_ids)
                return
            from commonnet import speculative
            stats = speculative.SpecStats()
            self.last_spec, self.last_k = stats, self.draft_k
            yield from speculative.generate_stream(self.client, endpoints, prompt_ids, max_tokens, drafter=drafter,
                                                   k=self.draft_k, temperature=temperature, top_p=top_p, seed=seed,
                                                   stop_ids=self.stop_ids, stats=stats)
        finally:
            transport.shutdown()

    def _lazy_drafter(self, endpoints, transport, temperature):
        """The default-on drafter, or None for plain decoding. Never raises: any doubt means plain decoding.

        Eligible only when the request is greedy, every hop advertises the chain (draft-verify) capability, and the
        drafter loads (cached; a failed load is not retried). Runs on the worker thread."""
        if not self.auto_draft:
            self.last_reason = self.draft_note
            return None
        if self._auto_failed:
            self.last_reason = self.draft_note
            return None
        if temperature and temperature > 0:
            self.last_reason = "not greedy (temperature > 0): sampled requests do not speculate"
            return None
        try:
            for i, ep in enumerate(endpoints):
                if CHAIN_CAPABILITY not in transport.capabilities(ep):
                    name = ep.get("url") or ep.get("peer") if isinstance(ep, dict) else None
                    self.last_reason = f"hop {i}{' (' + str(name) + ')' if name else ''} lacks {CHAIN_CAPABILITY}"
                    return None
        except Exception as e:
            self.last_reason = f"could not read hop capabilities: {type(e).__name__}: {e}"[:300]
            return None
        try:
            repo = [DEFAULT_DRAFT]

            def build():
                d, repo[0] = build_default_drafter(self.tokenizer)
                return d
            self.drafter = (self._drafter_factory or build)()
            self.draft_note = f"on ({repo[0]}, k={self.draft_k})"
            return self.drafter
        except Exception as e:   # mlx_lm missing, out of memory, tokenizer mismatch, no network...
            self._auto_failed = True
            self.draft_note = f"off: {type(e).__name__}: {e}"[:300]
            print(f"speculative decoding is off: {self.draft_note}", file=sys.stderr, flush=True)
            return None

    def speculation(self) -> dict:
        d = self.drafter
        kind = "none" if d is None else ("mlx" if "Mlx" in type(d).__name__ else "torch" if "Torch" in type(d).__name__
                                           else type(d).__name__)
        st = self.last_spec
        last = {"k": self.last_k, "proposed": st.drafted if st else 0, "accepted": st.accepted if st else 0}
        if self.last_reason:
            last["reason"] = self.last_reason
        return {"enabled": bool(self.auto_draft or d is not None), "note": self.draft_note, "drafter": kind,
                "last_request": last}

    def health(self) -> dict:
        status = dict(self.network.status())
        status["speculation"] = self.speculation()
        return status


class LazyBackend:
    """Stands in for the real backend while its client part loads: requests get a 503 until `set()`.

    The node's --api starts the HTTP server at once and loads the embedding and head in the background, so the
    port answers (and /health says `ready: false`) instead of refusing connections for a minute.
    """

    def __init__(self, network):
        self.network = network
        self._backend = None
        self.error: str | None = None

    @property
    def ready(self) -> bool:
        return self._backend is not None

    def set(self, backend) -> None:
        self._backend = backend

    def fail(self, message: str) -> None:
        self.error = message

    def _need(self):
        if self._backend is None:
            raise ApiError(503, self.error or "The model is still loading on this machine; try again in a minute.",
                           "service_unavailable", "loading")
        return self._backend

    @property
    def model_name(self) -> str:
        if self._backend is not None:
            return self._backend.model_name
        manifest = self.network.manifest
        return manifest.model.get("name", manifest.id)

    @property
    def tokenizer(self):
        return self._need().tokenizer

    def encode_chat(self, messages, template_kwargs):
        return self._need().encode_chat(messages, template_kwargs)

    def stream(self, prompt_ids, max_tokens, temperature, top_p, seed):
        return self._need().stream(prompt_ids, max_tokens, temperature, top_p, seed)

    def health(self) -> dict:
        status = dict(self.network.status())
        status["ready"] = self.ready
        if self.error:
            status["error"] = self.error
        if self._backend is not None and hasattr(self._backend, "speculation"):
            status["speculation"] = self._backend.speculation()
        return status


class _NoTransport:
    def forward(self, *a, **k):
        raise RuntimeError("no route yet")

    def close(self, *a, **k):
        pass


_END = object()


class _WorkerStream:
    """Runs a generator on the single worker thread and hands its items to the caller through a queue."""

    def __init__(self, worker, fn, args):
        self._queue: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self._future = worker.submit(self._run, fn, args)

    def _run(self, fn, args):
        gen = fn(*args)
        try:
            for item in gen:
                if self._cancel.is_set():
                    break
                self._queue.put(item)
        except BaseException as e:  # reaches the caller through the queue
            self._queue.put(e)
        finally:
            gen.close()
            self._queue.put(_END)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is _END:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self._cancel.set()


# --- http -------------------------------------------------------------------------------------------

def _classify(error: Exception) -> ApiError:
    from commonnet.net.client import BlockError
    from commonnet.mesh.placement import NoRoute
    from commonnet.netclient import NetworkUnreachable
    if isinstance(error, ApiError):
        return error
    if isinstance(error, (NoRoute, NetworkUnreachable)):
        return ApiError(503, str(error), "service_unavailable", "no_route")
    if isinstance(error, BlockError):
        return ApiError(502, f"a peer on the route failed: {error}", "bad_gateway", "peer_failed")
    return ApiError(500, f"{type(error).__name__}: {error}", "server_error")


def make_handler(backend, *, max_context: int = DEFAULT_MAX_CONTEXT):
    class Handler(BaseHTTPRequestHandler):
        server_version = "commonnet-serve/0.1"
        sys_version = ""

        def log_message(self, format, *args):
            pass

        def _json(self, status: int, obj: dict) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/v1/models", "/models"):
                self._json(200, {"object": "list", "data": [
                    {"id": backend.model_name, "object": "model", "created": 0, "owned_by": "common-network"}]})
            elif path == "/health":
                try:
                    self._json(200, backend.health())
                except Exception as e:
                    err = _classify(e)
                    self._json(err.status, err.body())
            else:
                self._json(404, ApiError(404, f"no endpoint {path}", "not_found").body())

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path not in ("/v1/chat/completions", "/chat/completions"):
                self._json(404, ApiError(404, f"no endpoint {path}", "not_found").body())
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0 or length > MAX_BODY_BYTES:
                    raise ApiError(400, "send a JSON body of up to 1 MiB with a Content-Length")
                try:
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                except ValueError:
                    raise ApiError(400, "body is not JSON") from None
                if not isinstance(body, dict):
                    raise ApiError(400, "body must be a JSON object")
                req = parse_request(body)
                try:
                    prompt_ids = backend.encode_chat(req["messages"], req["template_kwargs"])
                except ApiError:
                    raise
                except Exception as e:
                    raise ApiError(400, f"cannot build a prompt from these messages: {e}") from None
                if len(prompt_ids) + req["max_tokens"] > max_context:
                    raise ApiError(400, f"prompt ({len(prompt_ids)} tokens) plus max_tokens ({req['max_tokens']}) is over "
                                        f"this server's context limit of {max_context}", code="context_length_exceeded")
                stream = backend.stream(prompt_ids, req["max_tokens"], req["temperature"], req["top_p"], req["seed"])
            except Exception as e:
                err = _classify(e)
                self._json(err.status, err.body())
                return
            model = body.get("model") if isinstance(body.get("model"), str) else backend.model_name
            if req["stream"]:
                # Read the first token before sending any headers: a missing route or a dead peer at the start
                # is then a proper HTTP error, not an error event inside a 200 stream.
                try:
                    first = next(stream, _END)
                except Exception as e:
                    stream.close()
                    err = _classify(e)
                    self._json(err.status, err.body())
                    return
                self._stream_reply(stream, first, prompt_ids, req, model)
            else:
                self._plain_reply(stream, prompt_ids, req, model)

        def _finish(self, count: int, req: dict, stopped: bool) -> str:
            return "length" if (count >= req["max_tokens"] and not stopped) else "stop"

        def _plain_reply(self, stream, prompt_ids, req, model):
            text = TextStream(backend.tokenizer)
            pieces, count = [], 0
            try:
                for token_id in stream:
                    count += 1
                    pieces.append(text.push(token_id))
                pieces.append(text.flush())
            except Exception as e:
                stream.close()
                err = _classify(e)
                self._json(err.status, err.body())
                return
            self._json(200, {
                "id": "chatcmpl-" + uuid.uuid4().hex[:24], "object": "chat.completion", "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(pieces)},
                             "finish_reason": self._finish(count, req, False)}],
                "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": count,
                          "total_tokens": len(prompt_ids) + count}})

        def _stream_reply(self, stream, first, prompt_ids, req, model):
            ident, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())

            def chunk(delta, finish=None):
                return {"id": ident, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

            def send(obj) -> None:
                data = b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode("utf-8")) + b"\n\n"
                self.wfile.write(data)
                self.wfile.flush()

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            text, count = TextStream(backend.tokenizer), 0
            try:
                send(chunk({"role": "assistant", "content": ""}))
                for token_id in ([] if first is _END else itertools.chain([first], stream)):
                    count += 1
                    piece = text.push(token_id)
                    if piece:
                        send(chunk({"content": piece}))
                tail = text.flush()
                if tail:
                    send(chunk({"content": tail}))
                send(chunk({}, self._finish(count, req, False)))
            except (BrokenPipeError, ConnectionResetError):
                stream.close()  # the caller went away: stop spending the network's time
                return
            except Exception as e:
                stream.close()
                try:
                    send(_classify(e).body())
                except OSError:
                    return
            try:
                send(b"[DONE]")
            except OSError:
                pass

    return Handler


def make_server(backend, port: int = DEFAULT_PORT, *, max_context: int = DEFAULT_MAX_CONTEXT) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(backend, max_context=max_context))
    server.daemon_threads = True
    return server


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="commonnet.serve", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network", default=None,
                   help="a peer URL, or a manifest file that lists seeds. Omit it (or say `dht`) to find peers through the DHT: "
                        "the built-in manifest, no seed URL; needs --bootstrap or commonnet/networks/bootstrap.json")
    p.add_argument("--bootstrap", action="append", default=[], metavar="HOST:PORT",
                   help="a DHT node to discover peers through (repeatable; experimental, see docs/layer-network.md)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--checkpoint", help="local client checkpoint (embed, final norm, head); default: download the manifest's client_repo")
    p.add_argument("--client-format", choices=["mlx4", "fp16", "fp32"], default="mlx4",
                   help="how the client's embedding and head are held; mlx4 needs Apple silicon")
    p.add_argument("--draft", nargs="?", const="mlx-community/Qwen3-0.6B-4bit", default=None, metavar="MODEL",
                   help="speculative decoding with a small local drafter (default model mlx-community/Qwen3-0.6B-4bit); "
                        "greedy requests only, off by default")
    p.add_argument("--no-draft", action="store_true", help="turn off the default-on speculative decoding (same as COMMONNET_DRAFT=0)")
    p.add_argument("--draft-k", type=int, default=4, help="tokens drafted per round (default 4; shrinks when acceptance is low)")
    p.add_argument("--max-context", type=int, default=DEFAULT_MAX_CONTEXT)
    p.add_argument("--cache-dir", help="where to remember peer addresses between runs")
    args = p.parse_args(argv)
    import torch
    from commonnet.netclient import Network, NetworkUnreachable
    try:
        network = Network.open(args.network, args.bootstrap, cache_dir=args.cache_dir)
        network.manifest  # fail now, not on the first request
        backend = NetworkBackend(network, args.checkpoint, fmt=args.client_format, hidden_dtype=torch.float32,
                                 draft=args.draft, draft_k=args.draft_k, auto_draft=not args.no_draft)
    except NetworkUnreachable as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    server = make_server(backend, args.port, max_context=args.max_context)
    if args.draft or backend.auto_draft:
        print(f"speculative decoding: {backend.draft_note}", flush=True)
    print(f"serving {backend.model_name} from the network on http://127.0.0.1:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
