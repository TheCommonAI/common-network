"""Transport adapter: the core Client's Transport protocol over HTTP block servers.

Endpoint names map to base URLs (and optionally a per-endpoint token). Tensors are
converted to protocol.Tensor (float32 stays float32, anything else goes as float16).
A long prefill is sent in sequence chunks so each request stays under the server's
request limit; the block's KV cache makes chunked prefill equal to one-shot prefill.
"""
from __future__ import annotations

import time
import uuid
import re
import os
from dataclasses import dataclass

import numpy as np
import torch

from commonnet.net.client import BlockClient, BlockError, RemoteError, Unreachable
from commonnet.net.protocol import DTYPE_WIDTH, MAX_REQUEST_BYTES, Tensor

_NP = {"float16": np.float16, "float32": np.float32}
_TORCH = {"float16": torch.float16, "float32": torch.float32}
OVERHEAD_BYTES = 4096


def to_wire(t: torch.Tensor) -> Tensor:
    name = "float32" if t.dtype == torch.float32 else "float16"
    arr = t.detach().to(device="cpu", dtype=_TORCH[name]).contiguous().numpy()
    return Tensor(tuple(arr.shape), name, arr.tobytes())


def from_wire(t: Tensor) -> torch.Tensor:
    arr = np.frombuffer(t.data, dtype=_NP[t.dtype]).reshape(t.shape)
    return torch.from_numpy(arr.copy())


@dataclass(frozen=True)
class Call:
    """One piece of a session as sent when it is not a plain hidden tensor: token ids in (hidden None) and/or token
    ids out (output="ids"). Kept in the replay history exactly as sent, so a failover replays the same mode."""
    hidden: Tensor | None = None
    ids: tuple | None = None
    output: str | None = None
    which: str = "last"
    sampling: dict | None = None

    @property
    def seq(self) -> int:
        return len(self.ids) if self.ids is not None else self.hidden.shape[1]


def _send_piece(client, url, session_id, piece, start_pos, truncate_to=None):
    kw = {} if truncate_to is None else {"truncate_to": truncate_to}
    if isinstance(piece, Call):
        return client.forward(url, session_id, piece.hidden, start_pos, ids=piece.ids, output=piece.output,
                              which=piece.which, sampling=piece.sampling, **kw)
    return client.forward(url, session_id, piece, start_pos, **kw)


DEAD_SECONDS = 60.0
REVIVE_AFTER_SECONDS = 5.0   # a server the mesh still lists is retried once it has been dead this long
WAIT_POLL_SECONDS = 1.0


class HttpTransport:
    """The core Client's Transport over HTTP block servers.

    replicas maps an endpoint name to more servers for the same layers, each a URL or a
    (URL, token) pair. When the server in use stops answering (or has lost the session),
    the transport moves to the next one, opens the session there and replays every piece
    it already sent, exactly as sent, so the new server's cache is the one the old server
    held. The call that failed is then sent again. With no replicas and no on_exhausted nothing
    is kept and a dead server raises Unreachable as before.

    on_exhausted(name) -> [(url, token), ...] makes the transport survive running out of servers.
    It must return the servers the network now lists for the SAME layers as `name`. When every
    known server of an endpoint is dead the transport calls it, adds any new servers to the
    candidates (a known server it lists again is retried once it has been dead
    REVIVE_AFTER_SECONDS) and tries them. If none works it waits and asks again, up to wait_s
    seconds, before raising Unreachable. A newcomer gets the session replayed exactly as the dead
    one held it. Because the replay needs the history, it is kept whenever on_exhausted is set.
    """

    def __init__(self, urls: dict[str, str], *, token=None, tokens: dict[str, str] | None = None,
                 replicas: dict[str, list] | None = None, max_request_bytes: int = MAX_REQUEST_BYTES,
                 now=time.monotonic, on_exhausted=None, wait_s: float = 30.0,
                 wait_sleep=time.sleep, wait_poll_s: float = WAIT_POLL_SECONDS,
                 activation_codec: str | None = None, **client_kwargs):
        self.urls = dict(urls)
        self.max_request_bytes = max_request_bytes
        self._tokens = tokens or {}
        self._default_token = token
        self._kw = client_kwargs
        # Candidate identities remain public for peer forwarding and replica
        # refresh. Only this client's own connections may use a local shortcut.
        self._local_urls: dict[str, str] = {}
        self._now = now
        self._on_exhausted = on_exhausted
        self._wait_s = wait_s
        self._wait_sleep = wait_sleep
        self._wait_poll_s = wait_poll_s
        self._last_error: Exception | None = None
        self._replicas = {n: list(r) for n, r in (replicas or {}).items()}
        self._clients: dict[tuple[str, int], BlockClient] = {}
        self._open: set[tuple[str, str]] = set()
        self._active: dict[tuple[str, str], int] = {}
        self._history: dict[tuple[str, str], list] = {}
        self._dead: dict[tuple[str, int], float] = {}
        self._truncate_ok: dict[str, bool] = {}
        self._caps: dict[str, tuple] = {}
        self._pending_cut: dict[tuple[str, str], int] = {}
        self._chain_sessions: dict[str, dict] = {}
        self._chain_caps: dict[tuple, tuple] = {}
        selected = activation_codec if activation_codec is not None else os.environ.get(
            "COMMONNET_ACTIVATION_CODEC", "int8_group")
        if selected in ("float", "uncompressed", "none"):
            self._activation_codec = "float"
        elif selected in ("int8_group", "int8_group_v1"):
            self._activation_codec = "int8_group_v1"
        else:
            raise ValueError("COMMONNET_ACTIVATION_CODEC must be int8_group or float")
        self.failovers: list[tuple[str, int, int]] = []  # (endpoint, from index, to index), for tests and logs

    def _candidates(self, name: str) -> list:
        try:
            primary = self.urls[name]
        except KeyError:
            raise KeyError(f"no URL for endpoint {name!r}") from None
        return [primary] + self._replicas.get(name, [])

    def _target(self, name: str, index: int) -> tuple[BlockClient, str]:
        entry = self._candidates(name)[index]
        url, token = entry if isinstance(entry, tuple) else (
            entry, self._tokens.get(name, self._default_token))
        if (name, index) not in self._clients:
            kw = dict(self._kw)
            if self._replicas.get(name) or self._on_exhausted:
                kw.setdefault("retries", 1)  # another server can take over, so do not wait out the full backoff
            self._clients[(name, index)] = BlockClient(
                token=token, max_request_bytes=self.max_request_bytes, **kw)
        return self._clients[(name, index)], self._local_urls.get(url, url)

    def _alive(self, name: str, index: int) -> bool:
        died = self._dead.get((name, index))
        return died is None or self._now() - died > DEAD_SECONDS

    def _first_alive(self, name: str) -> int:
        for i in range(len(self._candidates(name))):
            if self._alive(name, i):
                return i
        return 0

    def _start(self, name: str, session_id: str, index: int) -> None:
        client, url = self._target(name, index)
        client.open(url, session_id)
        self._active[(name, session_id)] = index

    @staticmethod
    def _entry_url(entry) -> str:
        return entry[0] if isinstance(entry, tuple) else entry

    def _merge_servers(self, name: str, fresh) -> int:
        """Add servers the network lists for this endpoint's layers. Returns how many are new or revived."""
        known = {self._entry_url(e): i for i, e in enumerate(self._candidates(name))}
        changed = 0
        for entry in fresh or []:
            url, token = entry if isinstance(entry, tuple) else (entry, None)
            index = known.get(url)
            if index is None:
                self._replicas.setdefault(name, []).append((url, token))
                known[url] = len(known)
                changed += 1
            elif (name, index) in self._dead and self._now() - self._dead[(name, index)] >= REVIVE_AFTER_SECONDS:
                del self._dead[(name, index)]
                changed += 1
        return changed

    def refresh_replicas(self, name: str) -> None:
        """Ask the network for the current servers of this endpoint's layers (no-op without on_exhausted)."""
        if self._on_exhausted is None:
            return
        try:
            self._merge_servers(name, self._on_exhausted(name))
        except Exception:  # a failed refresh must never break a request that may not need it
            pass

    def _try_candidates(self, name: str, session_id: str, failed: int) -> bool:
        """Try each other live server in turn. True once one holds the session; failures set _last_error."""
        key = (name, session_id)
        count = len(self._candidates(name))
        for step in range(1, count + 1):
            index = (failed + step) % count
            if index != failed and not self._alive(name, index):
                continue
            try:
                self._start(name, session_id, index)
                client, url = self._target(name, index)
                for start_pos, piece in self._history.get(key, []):
                    _send_piece(client, url, session_id, piece, start_pos)
            except BlockError as e:
                self._dead[(name, index)] = self._now()
                self._last_error = e
                continue
            self.failovers.append((name, failed, index))
            return True
        return False

    def _recover(self, name: str, session_id: str, failed: int, error: Exception) -> None:
        """Move the session to the next live server and replay what it held. Raises if none is left."""
        self._dead[(name, failed)] = self._now()
        self._last_error = error
        if self._try_candidates(name, session_id, failed):
            return
        if self._on_exhausted is None:
            raise self._last_error
        deadline = self._now() + self._wait_s
        while True:
            try:
                self._merge_servers(name, self._on_exhausted(name))
            except Exception:  # the lookup failing is not the end: keep waiting for the network
                pass
            if self._try_candidates(name, session_id, failed):
                return
            if self._now() >= deadline:
                break
            self._wait_sleep(self._wait_poll_s)
        raise Unreachable(f"no live server for {name} after waiting {self._wait_s:g}s; last error: {self._last_error}")

    def _send(self, name: str, session_id: str, piece, start_pos: int, truncate_to=None):
        key = (name, session_id)
        while True:
            index = self._active[key]
            client, url = self._target(name, index)
            try:
                return _send_piece(client, url, session_id, piece, start_pos, truncate_to)
            except (Unreachable, RemoteError) as e:
                if len(self._candidates(name)) < 2 and self._on_exhausted is None:
                    raise
                if isinstance(e, RemoteError) and e.status != 404:
                    raise  # an answer from a live server, not a lost one (404: it forgot the session)
                self._recover(name, session_id, index, e)

    # --- rollback for speculative decoding -----------------------------------------------

    @staticmethod
    def _cut_history(pieces: list, length: int) -> list:
        """The pieces of a session's history, cut to the first `length` positions (a piece may be cut in the middle)."""
        out = []
        def cut_tensor(t, keep):
            arr = np.frombuffer(t.data, dtype=_NP[t.dtype]).reshape(t.shape)[:, :keep]
            return Tensor(tuple(arr.shape), t.dtype, np.ascontiguousarray(arr).tobytes())

        for start_pos, piece in pieces:
            if start_pos >= length:
                continue
            seq = piece.seq if isinstance(piece, Call) else piece.shape[1]
            if start_pos + seq > length:
                keep = length - start_pos
                if isinstance(piece, Call):
                    piece = Call(None if piece.hidden is None else cut_tensor(piece.hidden, keep),
                                 None if piece.ids is None else piece.ids[:keep], piece.output, piece.which,
                                 piece.sampling)
                else:
                    piece = cut_tensor(piece, keep)
            out.append((start_pos, piece))
        return out

    def supports_truncate(self, endpoints) -> bool:
        """True if every server on the route has the /truncate op. An old server answers 404 'no endpoint'.

        Asks about a throwaway session: a new server says that session is not open (404), an old one has no such path."""
        import uuid
        for ep in endpoints:
            cached = self._truncate_ok.get(ep.name)
            if cached is None:
                client, url = self._target(ep.name, self._first_alive(ep.name))
                try:
                    client.truncate(url, "probe-" + uuid.uuid4().hex[:16], 0)
                    cached = True
                except RemoteError as e:
                    cached = e.status == 404 and "no endpoint" not in e.message
                except BlockError:
                    cached = False
                self._truncate_ok[ep.name] = cached
            if not cached:
                return False
        return True

    def truncate(self, endpoint, session_id: str, length: int, defer: bool = False) -> None:
        """Cut this endpoint's session back to `length` positions. Idempotent. The kept history is cut first, so a
        failover during or after this replays exactly the truncated session."""
        chain = self._chain_sessions.get(session_id)
        if chain is not None:
            from commonnet.net.protocol import check_start_pos
            check_start_pos(length)
            if endpoint.name not in {h["name"] for h in chain["hops"]}:
                return
            # The core asks once per stage. One shared cut rides through every peer.
            history = []
            for entry in chain["history"]:
                if entry["start_pos"] < length:
                    kept = dict(entry)
                    kept["ids"] = entry["ids"][:length - entry["start_pos"]]
                    history.append(kept)
            chain["history"] = history
            chain["pending_cut"] = length
            if not defer:
                hop = next(h for h in chain["hops"] if h["name"] == endpoint.name)
                client, url = self._target(hop["name"], hop["index"])
                client.truncate(url, chain["wire_id"], length)
            return
        name, key = endpoint.name, (endpoint.name, session_id)
        if key not in self._open:
            return
        if key in self._history:
            self._history[key] = self._cut_history(self._history[key], length)
        if defer:
            # ride on the next forward to this endpoint (no round trip of its own); the cut history is already what a
            # replacement server would replay, so a failover before then is still exact
            self._pending_cut[key] = length
            return
        self._pending_cut.pop(key, None)
        while True:
            index = self._active[key]
            client, url = self._target(name, index)
            try:
                client.truncate(url, session_id, length)
                return
            except (Unreachable, RemoteError) as e:
                if len(self._candidates(name)) < 2 and self._on_exhausted is None:
                    raise
                if isinstance(e, RemoteError) and e.status != 404:
                    raise
                self._recover(name, session_id, index, e)  # replays the truncated history: nothing more to cut
                return

    def capabilities(self, endpoint) -> tuple:
        """What the server for this endpoint advertises in /health (ids_in, head_out). Empty for an old server or on
        any failure, so the caller falls back to the original protocol. Cached per endpoint name."""
        cached = self._caps.get(endpoint.name)
        if cached is None:
            client, url = self._target(endpoint.name, self._first_alive(endpoint.name))
            try:
                cached = tuple(client.health_caps(url))
            except Exception:
                cached = ()
            self._caps[endpoint.name] = cached
        return cached

    def ids_route(self, endpoints) -> bool:
        """True if the first server takes token ids and the last returns them (a single server needs both)."""
        if not endpoints:
            return False
        return "ids_in" in self.capabilities(endpoints[0]) and "head_out" in self.capabilities(endpoints[-1])

    def _chain_hops(self, endpoints) -> list:
        """Check the actual selected servers, not endpoint-name capability caches."""
        from urllib.parse import urlsplit
        from commonnet.net.chain import CAPABILITY, MAX_HOPS, check_hop
        endpoints = list(endpoints)
        if not endpoints or len(endpoints) > MAX_HOPS:
            raise ValueError("chain needs a bounded, nonempty route")
        previous, hops = 0, []
        for ep in endpoints:
            if ep.start != previous or ep.end <= ep.start:
                raise ValueError("chain route must start at zero and cover contiguous layers")
            previous = ep.end
            index = self._first_alive(ep.name)
            client, url = self._target(ep.name, index)
            key = (ep.name, index, url, ep.start, ep.end)
            info = self._chain_caps.get(key)
            if info is None:
                info = client.health_info(url)
                if info[0] != (ep.start, ep.end):
                    raise ValueError("chain peer serves a different layer window")
                self._chain_caps[key] = info
            caps = info[2]
            if CAPABILITY not in caps:
                raise ValueError("a route peer does not support chain forwarding")
            wire_url = self._entry_url(self._candidates(ep.name)[index])
            parts = urlsplit(wire_url)
            pin_key = f"{parts.hostname}:{parts.port or (443 if parts.scheme == 'https' else 80)}"
            pin = client._pin.get(pin_key) if isinstance(client._pin, dict) else client._pin
            hop = {"name": ep.name, "index": index, "url": wire_url, "token": client.token,
                   "pin": pin, "layers": [ep.start, ep.end], "caps": caps}
            hops.append(hop)
        if "ids_in" not in hops[0]["caps"] or "head_out" not in hops[-1]["caps"]:
            raise ValueError("chain end peers must hold the embedding and output head")
        for current, following in zip(hops, hops[1:]):
            check_hop({k: following[k] for k in ("url", "token", "pin", "layers")},
                      allow_private="chain_private_v1" in current["caps"])
        return hops

    def chain_route(self, endpoints) -> bool:
        """Capability negotiation before generation; old or unreachable peers fall back."""
        try:
            self._chain_hops(endpoints)
            return True
        except (BlockError, ValueError, KeyError):
            return False

    @staticmethod
    def _chain_body(state, ids, start_pos, which, sampling=None, drafts=None, truncate_to=None):
        hops = state["hops"]
        wire_hops = [{k: h[k] for k in ("url", "token", "pin", "layers")} for h in hops[1:]]
        body = {"id": state["wire_id"], "ids": list(ids), "start_pos": start_pos,
                "layers": hops[0]["layers"], "route": wire_hops, "which": which}
        if sampling is not None:
            body["sampling"] = sampling
        if drafts is not None:
            body["drafts"] = list(drafts)
        if truncate_to is not None:
            body["truncate_to"] = truncate_to
        if state.get("codec", "float") != "float":
            body["activation_codec"] = state["codec"]
        return body

    def _chain_send(self, state, body):
        first = state["hops"][0]
        client, url = self._target(first["name"], first["index"])
        return client.chain(url, body)

    def _chain_restart(self, state, endpoints, error):
        """Rebuild every KV cache from the retained token-ID history after a failure.

        A new internal session ID prevents a partly completed old chain from being
        mistaken for the replay. Never switch to a hidden-state route mid-session.
        """
        old_hops, old_id = state["hops"], state["wire_id"]
        lost = isinstance(error, RemoteError) and error.status in (404, 409)
        if not lost:
            match = re.search(r"chain hop (\d+)", str(error))
            failed_layer = int(match.group(1)) if match else old_hops[0]["layers"][0]
            for hop in old_hops:
                if hop["layers"][0] == failed_layer:
                    self._dead[(hop["name"], hop["index"])] = self._now()
        for ep in endpoints:
            self.refresh_replicas(ep.name)
        fresh = self._chain_hops(endpoints)  # rejects a failover peer without the new capabilities
        if state.get("codec") == "int8_group_v1" and not all(
                "activation_int8_group_v1" in h["caps"] for h in fresh):
            raise ValueError("replacement chain peer does not support this session's activation codec")
        for hop in old_hops:
            try:
                client, url = self._target(hop["name"], hop["index"])
                client.close(url, old_id)
            except BlockError:
                pass
        state["hops"], state["wire_id"] = fresh, "chain-" + uuid.uuid4().hex
        state["pending_cut"] = None
        for entry in state["history"]:
            body = self._chain_body(state, entry["ids"], entry["start_pos"], "last")
            self._chain_send(state, body)
        for hop in fresh:
            self._active[(hop["name"], state["caller_id"])] = hop["index"]

    def chain_step(self, endpoints, session_id: str, ids, start_pos: int, *, which: str = "last",
                   sampling=None, drafts=None, truncate_to=None) -> dict:
        """IDs enter the first peer, hidden states pass between peers, IDs return.

        For speculative verification the final peer returns accepted draft IDs and
        one correction/bonus token, plus accepted and next metadata. History contains
        input IDs only, so whole-route recovery does not retain activations here.
        """
        from commonnet.net import protocol
        from commonnet.net.chain import ids as check_chain_ids
        endpoints, token_ids = list(endpoints), check_chain_ids(list(ids))
        protocol.check_session_id(session_id)
        protocol.check_start_pos(start_pos)
        if which not in ("last", "all"):
            raise ValueError("chain which must be last or all")
        if drafts is not None and (list(drafts) != token_ids[1:] or which != "all" or sampling is not None):
            raise ValueError("chain verification needs [last]+drafts and greedy output")
        state = self._chain_sessions.get(session_id)
        if state is None:
            if start_pos != 0 or any((ep.name, session_id) in self._open for ep in endpoints):
                raise ValueError("a new chain session must start at zero")
            state = {"hops": self._chain_hops(endpoints), "history": [], "pending_cut": None,
                     "wire_id": "chain-" + uuid.uuid4().hex, "caller_id": session_id, "closed": set()}
            state["codec"] = (self._activation_codec if all("activation_int8_group_v1" in h["caps"]
                             for h in state["hops"]) else "float")
            self._chain_sessions[session_id] = state
            for hop in state["hops"]:
                self._open.add((hop["name"], session_id))
                self._active[(hop["name"], session_id)] = hop["index"]
        elif [(ep.name, ep.start, ep.end) for ep in endpoints] != [
                (h["name"], *h["layers"]) for h in state["hops"]]:
            raise ValueError("a chain session cannot change its route windows")
        committed_end = (state["history"][-1]["start_pos"] + len(state["history"][-1]["ids"])) if state["history"] else 0
        if start_pos != committed_end and truncate_to is None:
            raise ValueError("chain start position differs from its retained input-ID history")
        # Qwen3's 4096-wide float32 activations for 256 positions fit under the
        # standard 8 MiB peer request limit. Long prefills are traversed in order.
        chunk = min(256, max(1, (self.max_request_bytes - OVERHEAD_BYTES) // (4096 * 4 * 4 // 3)))
        if drafts is not None and len(token_ids) > chunk:
            raise ValueError("speculative verification batch exceeds the chain request limit")
        if truncate_to is not None:
            protocol.check_start_pos(truncate_to)
            if truncate_to != start_pos or truncate_to > committed_end:
                raise ValueError("chain rollback must cut committed history to this step's start")
            # Public callers normally use truncate(..., defer=True); explicit cuts
            # obey the same history discipline before any potential route replay.
            for ep in endpoints:
                self.truncate(ep, session_id, truncate_to, defer=True)
        replies = []
        for offset in range(0, len(token_ids), chunk):
            piece = token_ids[offset:offset + chunk]
            cut = state["pending_cut"] if offset == 0 else None
            body = self._chain_body(state, piece, start_pos + offset, which, sampling, drafts, cut)
            try:
                reply = self._chain_send(state, body)
            except (Unreachable, RemoteError) as error:
                if isinstance(error, RemoteError) and error.status not in (404, 409):
                    raise
                self._chain_restart(state, endpoints, error)
                body = self._chain_body(state, piece, start_pos + offset, which, sampling, drafts)
                reply = self._chain_send(state, body)
            if drafts is not None:
                accepted = reply.get("accepted")
                if (type(accepted) is not int or accepted > len(drafts)
                        or reply["ids"][:-1] != list(drafts)[:accepted]):
                    raise protocol.ProtocolError("invalid speculative chain reply")
            state["pending_cut"] = None
            state["history"].append({"ids": list(piece), "start_pos": start_pos + offset})
            replies.append(reply)
        if drafts is not None or which == "last":
            return replies[-1]
        return {"ids": [t for reply in replies for t in reply["ids"]]}

    def forward(self, endpoint, session_id: str, hidden: torch.Tensor | None, start_pos: int, *, ids=None,
                output: str | None = None, which: str = "last", sampling=None):
        """Hidden states in and out (the original call). With ids (hidden None) the server embeds them; with
        output="ids" it returns a list of token ids, for the last position or (which="all") every position."""
        name, key = endpoint.name, (endpoint.name, session_id)
        if key not in self._open:
            self.refresh_replicas(name)
            index = self._first_alive(name)
            try:
                self._start(name, session_id, index)
            except Unreachable as e:  # an answer with an error status (401, 400...) is a live server: raise it
                if len(self._candidates(name)) < 2 and self._on_exhausted is None:
                    raise
                self._recover(name, session_id, index, e)  # nothing to replay yet: just find a live server
            self._open.add(key)
        keep = len(self._candidates(name)) > 1 or self._on_exhausted is not None
        if ids is not None:
            ids = list(ids)
            seq = len(ids)
            chunk = max(1, (self.max_request_bytes - OVERHEAD_BYTES) // 12)  # a token id is at most ~11 JSON bytes
        else:
            batch, seq, width = hidden.shape
            wire_name = "float32" if hidden.dtype == torch.float32 else "float16"
            per_token = batch * width * DTYPE_WIDTH[wire_name] * 4 // 3 + 4
            chunk = max(1, (self.max_request_bytes - OVERHEAD_BYTES) // per_token)
        outs = []
        for i in range(0, seq, chunk):
            if ids is not None:
                piece = Call(None, tuple(ids[i:i + chunk]), output, which, sampling)
            else:
                t = to_wire(hidden[:, i:i + chunk])
                piece = t if output is None else Call(t, None, output, which, sampling)
            cut = self._pending_cut.pop(key, None) if i == 0 else None
            out = self._send(name, session_id, piece, start_pos + i, cut)
            if keep:
                self._history.setdefault(key, []).append((start_pos + i, piece))
            outs.append(out if output is not None else from_wire(out))
        if output is not None:
            return outs[-1] if which == "last" else [t for o in outs for t in o]
        out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)
        return out.to(hidden.dtype) if hidden is not None else out

    def close(self, endpoint, session_id: str) -> None:
        chain = self._chain_sessions.get(session_id)
        if chain is not None:
            hop = next((h for h in chain["hops"] if h["name"] == endpoint.name), None)
            if hop is None or endpoint.name in chain["closed"]:
                return
            try:
                client, url = self._target(hop["name"], hop["index"])
                client.close(url, chain["wire_id"])
            except Exception:
                pass
            chain["closed"].add(endpoint.name)
            self._open.discard((endpoint.name, session_id))
            self._active.pop((endpoint.name, session_id), None)
            if len(chain["closed"]) == len(chain["hops"]):
                del self._chain_sessions[session_id]
            return
        key = (endpoint.name, session_id)
        if key not in self._open:
            return
        self._open.discard(key)
        self._history.pop(key, None)
        self._pending_cut.pop(key, None)
        index = self._active.pop(key, 0)
        try:
            client, url = self._target(endpoint.name, index)
            client.close(url, session_id)
        except Exception:  # best effort: the server also drops sessions with the process
            pass

    def stats(self) -> dict:
        """Per endpoint: forward calls, mean total_ms, mean server_ms, and connections reused."""
        out = {}
        for (name, index), client in self._clients.items():
            name = name if index == 0 else f"{name}#{index}"
            rows = [t for t in client.timings if t["path"] in ("/forward", "/chain") and t["status"] == 200]
            if not rows:
                continue
            served = [t["server_ms"] for t in rows if t["server_ms"] is not None]
            out[name] = {
                "calls": len(rows),
                "total_ms": sum(t["total_ms"] for t in rows) / len(rows),
                "server_ms": sum(served) / len(served) if served else None,
                "reused": sum(1 for t in rows if t["reused"]),
            }
        return out

    def shutdown(self) -> None:
        from types import SimpleNamespace
        for session_id, state in list(self._chain_sessions.items()):
            for hop in state["hops"]:
                self.close(SimpleNamespace(name=hop["name"]), session_id)
        for client in self._clients.values():
            client.close_connections()
