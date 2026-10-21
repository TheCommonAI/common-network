"""Download only the tensors a member needs from Hugging Face (stdlib urllib only).

Sharded repo: fetch model.safetensors.index.json and download only the shards holding needed tensors.
Single-file repo: read the safetensors header with a Range request, Range-download only the needed
tensors, and write a compact model.safetensors under the cache dir.

Cache: ~/.common-network/commonnet/models/<repo>/<revision>/ (override: COMMONNET_CACHE).
HF_TOKEN (env) is sent as a bearer header to the Hub host only; it is never printed.
Base URL override (for tests): COMMONNET_HF_ENDPOINT.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ENDPOINT_ENV = "COMMONNET_HF_ENDPOINT"
DEFAULT_ENDPOINT = "https://huggingface.co"
INDEX_NAME = "model.safetensors.index.json"
SINGLE_NAME = "model.safetensors"
SMALL_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
               "vocab.json", "merges.txt", "special_tokens_map.json")
TIMEOUT = 60
RETRIES = 5
CHUNK = 1 << 20


class HubError(RuntimeError):
    pass


def cache_dir(repo: str, revision: str = "main", environ=None) -> str:
    env = os.environ if environ is None else environ
    root = env.get("COMMONNET_CACHE") or os.path.expanduser("~/.common-network/commonnet/models")
    return os.path.join(root, *repo.split("/"), revision)


LAYOUTS = ("model.language_model.", "language_model.model.", "model.")  # qwen3_5 HF, qwen3_5 MLX, qwen3


def detect_layout(names) -> str:
    """Tensor-name prefix of the decoder in this checkpoint. Vision/mtp tensors never match it."""
    names = list(names)
    return next((p for p in LAYOUTS if any(n.startswith(p + "layers.") for n in names)), "model.")


def needed_prefixes(layers=None, client=False, layout="model.") -> list[str]:
    out = []
    if layers:
        out += [f"{layout}layers.{i}." for i in range(layers[0], layers[1])]
    if client:
        out += [layout + "embed_tokens.", layout + "norm.", "lm_head.", "language_model.lm_head."]
    return out


def select_names(names, prefixes) -> list[str]:
    return [n for n in names if any(n.startswith(p) for p in prefixes)]


class _SameHostAuth(urllib.request.HTTPRedirectHandler):
    """Drop the Authorization header when a redirect leaves the original host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            new.headers.pop("Authorization", None)
            new.unredirected_hdrs.pop("Authorization", None)
        return new


_opener = urllib.request.build_opener(_SameHostAuth)


class Hub:
    def __init__(self, repo, revision="main", endpoint=None, environ=None, log=None, sleep=time.sleep):
        env = os.environ if environ is None else environ
        self.repo, self.revision, self.env = repo, revision, env
        self.endpoint = (endpoint or env.get(ENDPOINT_ENV) or DEFAULT_ENDPOINT).rstrip("/")
        self._token = env.get("HF_TOKEN")
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self.sleep = sleep
        self.dir = cache_dir(repo, revision, env)

    def url(self, name):
        return f"{self.endpoint}/{self.repo}/resolve/{urllib.parse.quote(self.revision, safe='')}/{name}"

    def _request(self, name, rng=None, method="GET"):
        headers = {"User-Agent": "commonnet-hub"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        if rng:
            headers["Range"] = f"bytes={rng[0]}-{rng[1] - 1}"
        last = None
        for attempt in range(RETRIES):
            try:
                return _opener.open(urllib.request.Request(self.url(name), headers=headers, method=method), timeout=TIMEOUT)
            except urllib.error.HTTPError as e:
                if e.code in (401, 403, 404):
                    raise HubError(f"{name}: HTTP {e.code} (check repo id, revision, HF_TOKEN)") from None
                last = f"HTTP {e.code}"
            except (urllib.error.URLError, OSError) as e:
                last = type(e).__name__
            self.sleep(min(2 ** attempt, 20))
        raise HubError(f"{name}: failed after {RETRIES} attempts ({last})")

    def get_bytes(self, name, rng):
        """Exactly rng[1]-rng[0] bytes, retried; raises HubError on a size mismatch."""
        want = rng[1] - rng[0]
        for attempt in range(RETRIES):
            try:
                with self._request(name, rng) as r:
                    data = r.read()
                if len(data) == want:
                    return data
            except (OSError, http_incomplete()):
                pass
            self.sleep(min(2 ** attempt, 20))
        raise HubError(f"{name}: wrong size for bytes {rng[0]}-{rng[1]}")

    def stream_range(self, name, start, end, out, label, quiet=False):
        """Append bytes [start, end) of name to the open file `out`, resuming on drop. Returns bytes written."""
        pos, last = start, time.monotonic()
        for attempt in range(RETRIES):
            try:
                with self._request(name, (pos, end)) as r:
                    if r.status == 200 and pos > 0:
                        raise HubError(f"{name}: server ignored Range")
                    while pos < end:
                        chunk = r.read(min(CHUNK, end - pos))
                        if not chunk:
                            break
                        out.write(chunk)
                        pos += len(chunk)
                        if not quiet and time.monotonic() - last > 2:
                            last = time.monotonic()
                            self.log(f"{label}: {(pos - start) >> 20} MB of {(end - start) >> 20} MB")
                if pos == end:
                    return pos - start
            except (OSError, http_incomplete()):
                pass
            self.sleep(min(2 ** attempt, 20))
        raise HubError(f"{name}: incomplete download ({pos - start} of {end - start} bytes)")

    # ---- small files -------------------------------------------------
    def fetch_small(self, name, required=True):
        path = os.path.join(self.dir, name)
        if os.path.exists(path):
            return path
        try:
            with self._request(name) as r:
                data = r.read()
        except HubError:
            if required:
                raise
            return None
        os.makedirs(self.dir, exist_ok=True)
        _atomic_write(path, data)
        return path

    def fetch_config(self):
        self.fetch_small("config.json")
        for n in SMALL_FILES[1:]:
            self.fetch_small(n, required=False)

    # ---- whole shard (resumable, size + sha256 checked) -------------------
    def fetch_file(self, name):
        path = os.path.join(self.dir, name)
        with self._request(name, method="HEAD") as r:
            size = int(r.headers["Content-Length"])
            etag = (r.headers.get("X-Linked-Etag") or "").strip('"')
        if os.path.exists(path):
            if os.path.getsize(path) != size:
                raise HubError(f"{name}: cached file has wrong size")
            return path
        part = path + ".part"
        os.makedirs(self.dir, exist_ok=True)
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > size:
            os.remove(part)
            have = 0
        with open(part, "ab") as out:
            if have < size:
                self.stream_range(name, have, size, out, name)
        if os.path.getsize(part) != size:
            raise HubError(f"{name}: size mismatch after download")
        if re.fullmatch(r"[0-9a-f]{64}", etag):
            h = hashlib.sha256()
            with open(part, "rb") as f:
                for blk in iter(lambda: f.read(CHUNK), b""):
                    h.update(blk)
            if h.hexdigest() != etag:
                os.remove(part)
                raise HubError(f"{name}: sha256 mismatch")
        os.replace(part, path)
        return path

    # ---- single file: header + needed ranges -> compact safetensors --------
    def read_header(self, name):
        n = struct.unpack("<Q", self.get_bytes(name, (0, 8)))[0]
        if n > 100 << 20:
            raise HubError(f"{name}: implausible safetensors header size")
        return n, json.loads(self.get_bytes(name, (8, 8 + n)))

    def fetch_compact(self, name, prefixes, out_name=SINGLE_NAME):
        final = os.path.join(self.dir, out_name)
        n, header = self.read_header(name)
        meta = header.pop("__metadata__", None)
        if callable(prefixes):
            prefixes = prefixes(header)
        names = sorted(select_names(header, prefixes), key=lambda k: header[k]["data_offsets"][0])
        if not names:
            raise HubError(f"{name}: no tensors match {prefixes[:2]}...")
        new, off = {}, 0
        for k in names:
            a, b = header[k]["data_offsets"]
            new[k] = {"dtype": header[k]["dtype"], "shape": header[k]["shape"], "data_offsets": [off, off + b - a]}
            off += b - a
        if meta:
            new["__metadata__"] = meta
        hb = json.dumps(new, separators=(",", ":")).encode()
        hb += b" " * (-len(hb) % 8)
        head = struct.pack("<Q", len(hb)) + hb
        total = len(head) + off
        if os.path.exists(final):
            if os.path.getsize(final) == total:
                return final
            raise HubError(f"{out_name}: existing compact file has wrong size (different layer set?)")
        part = final + ".part"
        os.makedirs(self.dir, exist_ok=True)
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > total:
            os.remove(part)
            have = 0
        with open(part, "ab") as out:
            if have < len(head):
                out.truncate(0)
                out.write(head)
                have = len(head)
            done, base, lastlog = have - len(head), 8 + n, 0.0
            for k in names:
                a, b = header[k]["data_offsets"]
                s, e = new[k]["data_offsets"]
                if e <= done:
                    continue
                skip = max(0, done - s)
                self.stream_range(name, base + a + skip, base + b, out, out_name, quiet=True)
                done = e
                if time.monotonic() - lastlog > 2:
                    lastlog = time.monotonic()
                    self.log(f"{out_name}: {(len(head) + done) >> 20} MB of {total >> 20} MB")
        if os.path.getsize(part) != total:
            raise HubError(f"{out_name}: size mismatch after download")
        os.replace(part, final)
        self.log(f"{out_name}: {total >> 20} MB written ({len(names)} tensors)")
        return final

    # ---- top level ---------------------------------------------------
    def fetch(self, layers=None, client=False):
        """Download what is needed; returns the checkpoint directory."""
        base = self.dir
        idx_path = self.fetch_small(INDEX_NAME, required=False)
        if idx_path:
            self.fetch_config()
            with open(idx_path) as f:
                wm = json.load(f)["weight_map"]
            layout = detect_layout(wm)
            prefixes = needed_prefixes(layers, client, layout)
            shards = sorted({wm[n] for n in select_names(wm, prefixes)})
            if layout != "model.":
                # qwen3_5 repos have ~5 GB shards that also hold vision/other layers: range-download only the
                # needed tensors of each shard into a compact copy with a filtered index.
                sel = ("L%d-%d" % tuple(layers) if layers else "") + ("+client" if client else "")
                self.dir = os.path.join(base, "compact-" + sel.strip("+"))
                self.fetch_config()
                for i, sname in enumerate(shards):
                    self.log(f"shard {i + 1}/{len(shards)}: {sname} (needed tensors only)")
                    self.fetch_compact(sname, prefixes, out_name=sname)
                keep = {n: wm[n] for n in select_names(wm, prefixes)}
                _atomic_write(os.path.join(self.dir, INDEX_NAME),
                              json.dumps({"metadata": {}, "weight_map": keep}).encode())
                return self.dir
            for i, s in enumerate(shards):
                self.log(f"shard {i + 1}/{len(shards)}: {s}")
                self.fetch_file(s)
            return self.dir
        # single file: a compact checkpoint per selection, so ranges never collide
        sel = ("L%d-%d" % tuple(layers) if layers else "") + ("+client" if client else "")
        self.dir = os.path.join(base, "compact-" + sel.strip("+"))
        self.fetch_config()
        self.fetch_compact(SINGLE_NAME, lambda names: needed_prefixes(layers, client, detect_layout(names)))
        return self.dir


def http_incomplete():
    import http.client
    return http.client.IncompleteRead


def _atomic_write(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
