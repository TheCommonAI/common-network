#!/usr/bin/env python3
"""The model runtime: fetch llama.cpp, and fetch the weights it serves.

The Common Network Alpha (v0.2.0-dev).

Copyright (C) 2026 Common AI Inc. Licensed under AGPL-3.0; see LICENSE at
https://github.com/TheCommonAI/common-network. This program comes with
ABSOLUTELY NO WARRANTY.

This module replaces Ollama. Before it existed, `common join` refused to run
unless Ollama was already installed, so contributing a machine meant installing
a second product first -- a ~1GB installer, a daemon with its own updater, and a
private model store. llama.cpp publishes a 12-20MB `llama-server` binary per
platform, so the runtime becomes something this CLI fetches for itself and a
node needs nothing but Python and cloudflared.

Three things about that are worth writing down, because each one is the reason
for a decision below.

**The weights come from Ollama's registry, not from Ollama.** `registry.ollama.ai`
is a plain OCI registry, and the layer it serves for a model tag is the GGUF
itself. So the catalogue's existing `source: "ollama:<tag>"` entries stay exactly
as they are, and stay honest -- the tag names a repository in Ollama's registry,
which is read here directly. Had the registry been an Ollama-private format, the
catalogue's 17 entries would all have needed re-sourcing from Hugging Face with
new hashes, which is a great deal of work and a great deal of risk for no change
in what the network serves.

**Content addressing gives verification away for free.** The digest in the
manifest *is* the sha256 of the blob, so there is no checksum list to maintain
anywhere in this project and no way for the file and its expected hash to drift
apart. It is checked anyway (see `_download`), because a multi-gigabyte download
over a home connection is exactly where a truncated file happens, and a truncated
GGUF does not fail loudly -- it loads, and answers nonsense.

**A build number must not be pinned.** llama.cpp ships several builds a day and
deletes old ones, so `b7000` 404s today while `b6900` resolves. An installer with
a build number compiled into it stops working within months, and a node that
cannot fetch its runtime cannot come back after a reboot. `resolve_llama_build`
resolves the current build at first use, caches it, and never throws away a
runtime that already works -- see `ensure_llama_server`.

Stdlib only, like the rest of the CLI: a contributor should never have to install
anything to donate a machine.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import socket
import stat
import string
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

COMMON_DIR = Path.home() / ".common-network"
RUNTIME_DIR = COMMON_DIR / "runtime"
MODELS_DIR = COMMON_DIR / "models"
CACHE_DIR = COMMON_DIR / "cache"
DOWNLOAD_CACHE = CACHE_DIR / "downloads"
BUILD_CACHE = CACHE_DIR / "llama-build.txt"
CURRENT_RUNTIME = RUNTIME_DIR / "current"

# Ollama's registry is an ordinary OCI registry: a manifest names its layers,
# and the layer carrying the weights is the GGUF. Only these four media types
# are read; anything else in a manifest is ignored.
OLLAMA_REGISTRY = "https://registry.ollama.ai"
OCI_MANIFEST_TYPE = "application/vnd.docker.distribution.manifest.v2+json"
MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"
TEMPLATE_MEDIA_TYPE = "application/vnd.ollama.image.template"
PARAMS_MEDIA_TYPE = "application/vnd.ollama.image.params"

LLAMA_REPO = "ggml-org/llama.cpp"
RELEASES_API = f"https://api.github.com/repos/{LLAMA_REPO}/releases?per_page=1"
RELEASE_DOWNLOAD = f"https://github.com/{LLAMA_REPO}/releases/download/{{build}}/{{asset}}"

# If both the API and the cache come back empty. Deliberately not the only
# source of truth -- see the module docstring on why a pinned build rots.
FALLBACK_BUILD = "b11669"

# llama.cpp's default context is the model's own, which for most of this
# catalogue is 32k or more. That is not free: the KV cache is allocated up front
# and grows with it, and the seed file's min_ram_gb for every entry was measured
# against Ollama's much smaller default. Left at llama.cpp's default, every
# min_ram_gb in the catalogue would be an understatement, and a node would join
# a machine that then ran out of memory. 4096 keeps those figures true.
DEFAULT_CTX = 4096

# How long llama-server keeps the weights resident between requests. This is the
# successor to the `keep_alive: "30m"` hint worker.py used to stamp on every
# request, and it is a better mechanism for the same intent: the process that
# owns the weights decides when to release them, instead of another process being
# asked nicely each time.
#
# While a node is joined, this window never actually arrives: the worker pings
# the backend well inside it (see _keep_warm in worker.py), because a sleeping
# server is woken by the next user, who then pays the reload. Half an hour of
# quiet as a laptop's worth of courtesy is the right trade for the case that
# remains -- the node has left, or the worker died and the backend is lingering
# -- which is exactly when the keep-warm pings stop too.
DEFAULT_SLEEP_IDLE_SECONDS = 1800

# Streaming a 3-9GB model over a home connection, where the socket is idle far
# longer than a request ever is.
DOWNLOAD_TIMEOUT = 60

_GB = 1024 ** 3


class FetchFailed(Exception):
    """Something could not be fetched. The message is worth showing a user."""


class ModelSpec:
    """Where a catalogue `source` says a model comes from.

    `wire_name` is the load-bearing field: it is the string registered with the
    gateway as this node's `model_name` *and* the string the worker answers
    `GET /v1/models` with. The gateway's health check requires those to be equal
    (gateway/app/health.py), so one string is used twice and neither may be
    invented independently. For an `ollama:` source it is the tag with the
    scheme stripped -- byte-identical to what every existing node already
    registers, so nothing about routing or health changes.
    """

    __slots__ = ("kind", "source", "wire_name", "registry_ref", "gguf_url", "gguf_sha256")

    def __init__(self, kind, source, wire_name, registry_ref=None,
                 gguf_url=None, gguf_sha256=None):
        self.kind = kind
        self.source = source
        self.wire_name = wire_name
        self.registry_ref = registry_ref          # (namespace, name, version)
        self.gguf_url = gguf_url
        self.gguf_sha256 = gguf_sha256


def parse_ollama_tag(tag: str) -> tuple[str, str, str]:
    """Split an Ollama tag into (namespace, name, version).

    `mathstral:7b` is `library/mathstral:7b` -- an unqualified tag means the
    official namespace, exactly as it does for Ollama itself. Neither a
    namespace nor a name may contain a colon, which is what lets the first one
    in the whole string be the version separator.
    """
    head, _, version = tag.partition(":")
    if "/" in head:
        namespace, _, name = head.partition("/")
    else:
        namespace, name = "library", head
    return namespace, name, version or "latest"


def resolve_source(source: str) -> ModelSpec | None:
    """Decode a catalogue `source` into something installable, or None.

    None means "common-join cannot provision this" -- the api: and graph: kinds,
    and anything unrecognised. Callers treat it as a plain predicate
    (`is_installable`) except where the spec itself is needed.
    """
    if not isinstance(source, str):
        return None

    if source.startswith("ollama:"):
        tag = source[len("ollama:"):]
        namespace, name, version = parse_ollama_tag(tag)
        if not name:
            return None
        return ModelSpec("ollama", source, tag, registry_ref=(namespace, name, version))

    if source.startswith("gguf:"):
        url, _, digest = source[len("gguf:"):].rpartition("#")
        digest = digest.strip().lower()
        if not url or len(digest) != 64 or any(c not in string.hexdigits for c in digest):
            return None
        # Named after the file rather than the hash, because this string is
        # registered publicly as the node's model name and a 64-hex blob is not
        # something anyone can read.
        stem = Path(urllib.parse.urlparse(url).path).name
        if stem.lower().endswith(".gguf"):
            stem = stem[:-len(".gguf")]
        return ModelSpec("gguf", source, stem or f"gguf-{digest[:12]}",
                         gguf_url=url, gguf_sha256=digest)

    return None


def is_installable(source: str) -> bool:
    """Whether common-join can provision this catalogue entry itself."""
    return resolve_source(source) is not None


# --- progress ---------------------------------------------------------------

def _size(n: float | None) -> str:
    if not n:
        return "?"
    if n >= _GB:
        return f"{n / _GB:.1f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.0f} MB"
    return f"{n / 1024:.0f} KB"


class _Progress:
    """A download's progress line, in whichever shape the destination wants.

    On a terminal this is one line rewritten in place. Everywhere else it is
    nothing at all until each 10% mark, and then a single ordinary line -- and
    the distinction is not cosmetic, because the two places this output gets
    redirected are the ones that matter most: `common join --permanent` sends
    both streams to join.log on Windows, and launchd and systemd do the same on
    the other two. A progress bar written with carriage returns would fill those
    logs with one enormous line, and a node's log is read afterwards, by someone
    trying to work out why it did not come up.

    That line carries `done of total` and not only a percentage, which is the
    one thing the two shapes now share. Ten numbers on their own do not say
    what is being waited for: a reader of the log -- the person above, or the
    desktop app, which follows join.log rather than the node's streams --
    cannot tell 60% of 1.3GB from 60% of 4GB, or see that anything is moving at
    all between one mark and the next. The sizes are what make a several-minute
    wait legible in a file. The terminal line prints them already; there was no
    reason for the redirected one to keep less.
    """

    def __init__(self, what: str, total: int | None):
        self.what = what
        self.total = total
        self.tty = sys.stderr.isatty()
        self._next_emit = 10
        self._started = time.monotonic()
        self._last_paint = 0.0
        self._open = False

    def update(self, done: int) -> None:
        percent = int(done * 100 / self.total) if self.total else 0
        if self.tty:
            now = time.monotonic()
            if now - self._last_paint < 0.15 and percent < 100:
                return
            self._last_paint = now
            remaining = f"  {_size(self.total - done)} left" if self.total else ""
            sys.stderr.write(
                f"\r  {self.what}  {percent:3d}%  {_size(done)} of {_size(self.total)}{remaining}   "
            )
            sys.stderr.flush()
            self._open = True
            return
        if self.total and percent >= self._next_emit:
            self._next_emit = percent - (percent % 10) + 10
            sys.stderr.write(
                f"  {self.what}  {percent:3d}%  {_size(done)} of {_size(self.total)}\n"
            )
            sys.stderr.flush()

    def close(self) -> None:
        if self.tty and self._open:
            sys.stderr.write("\r" + " " * 78 + "\r")
            sys.stderr.flush()
            self._open = False


# --- fetching ---------------------------------------------------------------

# How many times one fetch tries before giving up, and how long it waits before
# each retry (multiplied by the attempt number). Every try resumes from the partial
# file, so a retry costs only the bytes that did not arrive. The join used to give
# up on the first dropped connection, and the whole join then had to be started
# again by hand.
DOWNLOAD_ATTEMPTS = 4
DOWNLOAD_RETRY_SECONDS = 5


def _interrupted(e: FetchFailed) -> bool:
    """True for a failure the next try can resume past, and False for one it cannot.

    A dropped or stalled connection is worth another try. A checksum or size that
    does not match is not: the file was complete and wrong, and a resume would
    only append to the wrong thing.
    """
    text = str(e)
    return "was interrupted" in text or "could not reach" in text


def _download(url: str, dest: Path, expected_sha: str | None = None,
              expected_size: int | None = None, what: str = "") -> None:
    """Fetch `url` to `dest`, retrying an interrupted attempt from where it stopped."""
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            _download_once(url, dest, expected_sha, expected_size, what)
            return
        except FetchFailed as e:
            if attempt == DOWNLOAD_ATTEMPTS or not _interrupted(e):
                raise
            print(f"  the download of {what or url} was interrupted; trying again "
                  f"from where it stopped ({attempt} of {DOWNLOAD_ATTEMPTS})",
                  file=sys.stderr)
            time.sleep(DOWNLOAD_RETRY_SECONDS * attempt)


def _download_once(url: str, dest: Path, expected_sha: str | None = None,
                   expected_size: int | None = None, what: str = "") -> None:
    """Fetch `url` to `dest`, resuming an interrupted attempt and verifying it.

    The registry answers a blob GET with a 307 to a pre-signed object-storage URL
    that expires within a day, so a resume re-issues against the registry URL and
    lets the redirect happen again. Caching the signed URL would work until it
    did not, and then fail in a way that reads as the model having vanished.

    A resumed download hashes the part it already has before appending to it,
    because the digest covers the whole file and a prefix that was never hashed
    would make the final check meaningless. That costs one read of a few
    gigabytes, which is seconds, against a check that is the only thing standing
    between a contributor and a model that loads and then answers nonsense.
    """
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    have = part.stat().st_size if part.exists() else 0
    if expected_size is not None and have > expected_size:
        have = 0  # nonsense on disk; start over rather than trust it

    headers = {"Accept-Encoding": "identity"}  # a hash needs the bytes as sent
    if have:
        headers["Range"] = f"bytes={have}-"

    hasher = hashlib.sha256()
    if have:
        try:
            with open(part, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    hasher.update(chunk)
        except OSError:
            have = 0
            hasher = hashlib.sha256()

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT)
    except urllib.error.HTTPError as e:
        raise FetchFailed(f"{url} answered {e.code}") from None
    except (urllib.error.URLError, OSError) as e:
        raise FetchFailed(f"could not reach {url} ({e})") from None

    mode = "ab" if have else "wb"
    with resp:
        # A server that ignored the Range header is sending the whole file, and
        # appending that to a partial one would produce a plausible-looking
        # corrupt file. The status says which happened.
        if have and getattr(resp, "status", 200) != 206:
            have = 0
            mode = "wb"
            hasher = hashlib.sha256()

        total = expected_size
        if total is None:
            length = resp.headers.get("Content-Length")
            total = have + int(length) if length and length.isdigit() else None

        progress = _Progress(what, total) if what else None
        if progress:
            progress.update(have)
        try:
            with open(part, mode) as out:
                while True:
                    try:
                        chunk = resp.read(1024 * 256)
                    except OSError as e:
                        # A stalled or dropped connection is the common way a
                        # first-run download on a slow network fails, and it must
                        # be said as a FetchFailed: the caller only handles that,
                        # so anything else ends the join with a traceback. The
                        # partial file is kept, so running again resumes from it.
                        raise FetchFailed(
                            f"the download of {what or url} was interrupted at "
                            f"{_size(have)} ({e}); run this again and it resumes"
                        ) from None
                    if not chunk:
                        break
                    out.write(chunk)
                    hasher.update(chunk)
                    have += len(chunk)
                    if progress:
                        progress.update(have)
                out.flush()
                # Before the rename, not after: a file that is the right name but
                # not yet on the disk is worse than one that is obviously still
                # in progress.
                os.fsync(out.fileno())
        finally:
            if progress:
                progress.close()

    digest = hasher.hexdigest()
    if expected_size is not None and have != expected_size:
        _unlink(part)
        raise FetchFailed(f"{what or 'the download'} stopped early "
                          f"({_size(have)} of {_size(expected_size)})")
    if expected_sha and digest != expected_sha:
        _unlink(part)
        raise FetchFailed(f"{what or 'the download'} did not match its recorded checksum "
                          f"(expected {expected_sha[:12]}…, got {digest[:12]}…)")

    os.replace(part, dest)


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _get(url: str, accept: str | None = None, timeout: float = 15.0):
    headers = {"User-Agent": "common-join"}
    if accept:
        headers["Accept"] = accept
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                  timeout=timeout)


def _read_text(url: str, timeout: float = 15.0) -> str:
    with _get(url, timeout=timeout) as resp:
        return resp.read().decode(errors="replace")


def fetch_manifest(namespace: str, name: str, version: str) -> dict:
    """The OCI manifest for a tag in Ollama's registry."""
    url = f"{OLLAMA_REGISTRY}/v2/{namespace}/{name}/manifests/{version}"
    try:
        with _get(url, accept=OCI_MANIFEST_TYPE) as resp:
            manifest = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise FetchFailed(
                f"Ollama's registry has no model '{namespace}/{name}:{version}'"
            ) from None
        raise FetchFailed(f"the registry answered {e.code} for {name}:{version}") from None
    except (urllib.error.URLError, OSError) as e:
        raise FetchFailed(f"could not reach {OLLAMA_REGISTRY} ({e})") from None

    if "layers" not in manifest:
        raise FetchFailed(f"the manifest for {name}:{version} named no layers")
    return manifest


def _layer(manifest: dict, media_type: str) -> tuple[str, int] | None:
    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == media_type:
            digest = str(layer.get("digest", ""))
            hex_digest = digest.split(":", 1)[-1]
            if len(hex_digest) == 64:
                return hex_digest, int(layer.get("size") or 0)
    return None


def model_layer(manifest: dict) -> tuple[str, int]:
    """(sha256 hex, size in bytes) of the layer that is the GGUF itself."""
    found = _layer(manifest, MODEL_MEDIA_TYPE)
    if found is None:
        raise FetchFailed("the registry entry carries no model weights")
    return found


def _blob_text(namespace: str, name: str, hex_digest: str) -> str | None:
    url = f"{OLLAMA_REGISTRY}/v2/{namespace}/{name}/blobs/sha256:{hex_digest}"
    try:
        with _get(url) as resp:
            return resp.read().decode(errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        return None  # metadata only -- never worth failing a join over


def template_hint(layer: str | None) -> str | None:
    """Which of llama.cpp's built-in chat templates this model wants, if any.

    The registry's template layer is a Go `text/template` -- `[INST] {{ if .System }}…`,
    `{{- range $i, $_ := .Messages }}` -- and llama.cpp's `--chat-template-file`
    wants Jinja, so the layer cannot simply be written out to disk and handed
    over; the flag would reject it. It can still be *read*, as a signature of the
    format, which is enough to name one of llama.cpp's built-ins. The names below
    are not guessed: `llama-server --help` prints the accepted list, and every one
    of these is on it.

    This is the fallback of a fallback, and the ordering in here is what decides
    it. The layer is never the thing that gets used -- `--jinja` is on by default
    and a GGUF that ships its own template is formatted from that, which is most
    of this catalogue. The layer only matters when a GGUF does not, and that case
    is dangerous precisely because it does not fail: a model formatted with the
    wrong template answers, badly, and the answer looks like a weak model rather
    than a misconfigured node. So the markers are ordered from most to least
    specific, and a model with no layer at all gets None -- an honest "unknown"
    hands the job back to the GGUF rather than forcing a wrong format on it.

    Two of these were read off the catalogue and are worth recording, because
    both go the less obvious way. `meditron:7b` layers a *chatml* template even
    though Meditron is Llama-2 derived -- that is Ollama's own choice for that tag
    and the point here is to format as it did, not as the base model's card says.
    `phi4-mini-reasoning` marks itself with `<|end|>`, not the `<|user|>` an
    earlier version of this looked for, which is why it used to return None.
    """
    if not layer:
        return None
    if "<|start_header_id|>" in layer:
        return "llama3"
    if "<|im_start|>" in layer:
        return "chatml"
    if "<start_of_turn>" in layer:
        return "gemma"
    if "<|end|>" in layer:
        return "phi4"          # Phi's own terminator, shared by phi3 and phi4
    if "<<SYS>>" in layer:
        return "llama2"        # Llama-2's system block, which mistral-v1 has not got
    if "[INST]" in layer and "[/INST]" in layer:
        return "mistral-v1"
    if "｜" in layer or "<|end▁of▁sentence|>" in layer:
        return "deepseek"
    return None


def read_params(layer: str | None) -> dict:
    """The sampling and stop defaults Ollama would have applied, as a dict.

    Small (tens of bytes) and easy to overlook, but the stop tokens in here are
    what stopped a chatty specialist running past its answer. The layer is a JSON
    object -- `{"stop": ["[INST]", "[/INST]"]}` -- and not Ollama's Modelfile
    `PARAMETER` syntax, which is what the first version of this parsed and why it
    silently found nothing in every model it looked at.

    Recorded rather than applied. The values that matter are already in the GGUF
    for llama.cpp to read, and the rest are evidence for the next person debugging
    a node. Applying them is a fast-follow: the worker already stamps `max_tokens`
    on every request and could stamp these too.
    """
    if not layer:
        return {}
    try:
        params = json.loads(layer)
    except ValueError:
        return {}
    return params if isinstance(params, dict) else {}


def _ollama_model_roots() -> list[Path]:
    roots = []
    configured = os.environ.get("OLLAMA_MODELS")
    if configured:
        roots.append(Path(configured))
    roots.append(Path.home() / ".ollama" / "models")
    return roots


def _take_from_ollama(hex_digest: str, dest: Path, size: int) -> bool:
    """Take the weights from an existing Ollama install rather than downloading.

    Ollama's store is content-addressed by exactly the digest the registry
    manifest gave us, so the file it already has is the file we want -- same
    bytes, same name. A machine that has run `ollama pull` therefore pays nothing
    to change runtimes, which matters more than it looks: the alternative is
    asking every existing contributor to spend 3-9GB again for no change in what
    the network gets.

    Hardlinked rather than copied, so the same bytes are not on disk twice. Safe
    because that store is immutable: re-pulling a digest Ollama already has is a
    no-op, and `ollama rm` unlinks, which decrements the link count and leaves
    this file intact.
    """
    for root in _ollama_model_roots():
        blob = root / "blobs" / f"sha256-{hex_digest}"
        try:
            if not blob.is_file() or blob.stat().st_size != size:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            _unlink(dest)
            try:
                os.link(blob, dest)
            except OSError:
                shutil.copyfile(blob, dest)  # another filesystem, or no hardlinks
            if dest.is_file() and dest.stat().st_size == size:
                return True
        except OSError:
            continue
    return False


def _write_meta(directory: Path, spec: ModelSpec, entry: dict) -> None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "meta.json").write_text(json.dumps({
            "source": spec.source,
            "wire_name": spec.wire_name,
            "kind": spec.kind,
            **entry,
        }, indent=2))
    except OSError:
        pass  # best effort -- it is a note for a human, not state we depend on


def _check_space(needed: int, spec: ModelSpec) -> None:
    try:
        free = shutil.disk_usage(str(MODELS_DIR.parent if MODELS_DIR.exists() else Path.home())).free
    except OSError:
        return
    if free < needed + _GB:  # a gigabyte of elbow room for the .part and the OS
        raise FetchFailed(
            f"{spec.wire_name} needs {_size(needed)} but this machine has {_size(free)} free. "
            f"Free some space and run this again"
        )


def ensure_gguf(spec: ModelSpec) -> Path:
    """The GGUF for `spec`, fetching it if this machine does not already have it."""
    if spec.kind == "ollama":
        namespace, name, version = spec.registry_ref
        manifest = fetch_manifest(namespace, name, version)
        hex_digest, size = model_layer(manifest)
        directory = MODELS_DIR / hex_digest
        gguf = directory / "model.gguf"

        # The fast path for a service restart, and the only check that is cheap
        # enough to run every time. The hash was verified when this file was
        # written; re-hashing 9GB on every join would cost more than the fetch it
        # is guarding against. Size is enough to catch the file being replaced.
        meta = directory / "meta.json"
        if gguf.is_file() and meta.is_file():
            try:
                if json.loads(meta.read_text()).get("size") == size:
                    return gguf
            except (OSError, json.JSONDecodeError):
                pass

        if _take_from_ollama(hex_digest, gguf, size):
            _write_meta(directory, spec, {"sha256": hex_digest, "size": size,
                                          "reused_from_ollama": True})
            return gguf

        _check_space(size, spec)
        blob = f"{OLLAMA_REGISTRY}/v2/{namespace}/{name}/blobs/sha256:{hex_digest}"

        template = _layer(manifest, TEMPLATE_MEDIA_TYPE)
        params = _layer(manifest, PARAMS_MEDIA_TYPE)
        template_text = _blob_text(namespace, name, template[0]) if template else None
        params_text = _blob_text(namespace, name, params[0]) if params else None

        _download(blob, gguf, expected_sha=hex_digest, expected_size=size,
                  what=f"{spec.wire_name}")
        _write_meta(directory, spec, {
            "sha256": hex_digest,
            "size": size,
            "template_hint": template_hint(template_text),
            "params": read_params(params_text),
        })
        return gguf

    # A plain GGUF at a URL, for models Ollama's registry does not carry.
    directory = MODELS_DIR / spec.gguf_sha256
    gguf = directory / "model.gguf"
    if gguf.is_file():
        try:
            if json.loads((directory / "meta.json").read_text()).get("sha256") == spec.gguf_sha256:
                return gguf
        except (OSError, json.JSONDecodeError):
            pass

    _download(spec.gguf_url, gguf, expected_sha=spec.gguf_sha256, what=spec.wire_name)
    _write_meta(directory, spec, {"sha256": spec.gguf_sha256})
    return gguf


# --- the llama.cpp runtime --------------------------------------------------

def llama_asset_name(build: str, system: str | None = None,
                     machine: str | None = None) -> str | None:
    """The release asset for this platform, or None if llama.cpp ships none.

    The CPU builds are the deliberate choice, not the fallback: they are also the
    only ones that need nothing installed alongside them. Every GPU variant binds
    to a driver or runtime -- CUDA to a toolkit version, ROCm to a kernel module,
    Vulkan to an ICD loader -- and getting that wrong turns a working node into
    one that will not start. Metal needs none of it and is compiled in, which is
    why the macOS builds are GPU builds by default.
    """
    system = system or platform.system()
    machine = machine or platform.machine()

    if system == "Darwin":
        arch = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x64", "AMD64": "x64"}.get(machine)
        return f"llama-{build}-bin-macos-{arch}.tar.gz" if arch else None
    if system == "Linux":
        arch = {"x86_64": "x64", "AMD64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(machine)
        return f"llama-{build}-bin-ubuntu-{arch}.tar.gz" if arch else None
    if system == "Windows":
        # The CPU build. A Windows machine with an NVIDIA GPU installs the CUDA
        # pair from `windows_cuda_assets` instead; see `ensure_llama_server`.
        arch = {"AMD64": "x64", "x86_64": "x64", "ARM64": "arm64", "aarch64": "arm64"}.get(machine)
        return f"llama-{build}-bin-win-cpu-{arch}.zip" if arch else None
    return None


# The CUDA builds for Windows x64, newest first. Each line is two archives: the
# llama.cpp build (llama-server, ggml-rpc-server, ggml-cuda.dll) and the CUDA
# runtime DLLs it loads. Both are extracted into one directory.
WINDOWS_CUDA_LINES = ("13.4", "12.4")
WINDOWS_CUDA_DEFAULT = "12.4"


@dataclass(frozen=True)
class NvidiaGpu:
    """An NVIDIA GPU that nvidia-smi can see.

    `driver_cuda` is the (major, minor) CUDA version the driver reports -- the
    newest runtime it can run -- or None when that could not be read.
    """
    driver_cuda: tuple[int, int] | None


def detect_nvidia_gpu() -> NvidiaGpu | None:
    """Whether an NVIDIA GPU is present, and what CUDA its driver supports.

    Presence is `nvidia-smi -L` exiting 0 with a `GPU n:` line. The plain report
    carries the driver's `CUDA Version`. Any failure to run nvidia-smi means no
    GPU; a report without a readable version means a GPU with an unknown version.
    """
    try:
        listed = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if listed.returncode != 0 or not re.search(r"^GPU \d+:", listed.stdout, re.MULTILINE):
        return None

    found = None
    try:
        report = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
        if report.returncode == 0:
            found = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", report.stdout)
    except (OSError, subprocess.SubprocessError):
        pass
    driver_cuda = (int(found.group(1)), int(found.group(2))) if found else None
    return NvidiaGpu(driver_cuda=driver_cuda)


def _cuda_version(line: str) -> tuple[int, int]:
    major, minor = line.split(".")
    return int(major), int(minor)


def windows_cuda_line(gpu: NvidiaGpu) -> str:
    """The newest CUDA build line this driver supports.

    A driver runs its own CUDA version and every older runtime, so the first line
    at or below the driver's version is the one to install. A driver whose version
    cannot be read gets the default line. A driver older than every line is refused
    rather than given the CPU build: the network is GPU-only, and a CPU worker
    would join it and be useless.
    """
    if gpu.driver_cuda is None:
        return WINDOWS_CUDA_DEFAULT
    for line in WINDOWS_CUDA_LINES:
        if gpu.driver_cuda >= _cuda_version(line):
            return line
    major, minor = gpu.driver_cuda
    raise FetchFailed(
        f"the NVIDIA driver supports CUDA {major}.{minor}, older than the "
        f"{WINDOWS_CUDA_LINES[-1]} build llama.cpp ships for Windows. "
        "Update the NVIDIA driver and run again"
    )


def windows_cuda_assets(build: str, line: str) -> tuple[str, str]:
    """The CUDA build archive and the CUDA runtime archive for one line."""
    return (f"llama-{build}-bin-win-cuda-{line}-x64.zip",
            f"cudart-llama-bin-win-cuda-{line}-x64.zip")


def llama_asset_digest(build: str, asset: str) -> str | None:
    """The sha256 GitHub records for one llama.cpp release asset, or None.

    GitHub computes and stores a digest for every release asset, and this reads
    it from the same release API that names the build. It is what the archive is
    checked against: a runtime that cannot be checked is not installed. None means
    the record is missing or the API could not be reached.
    """
    try:
        release = json.loads(_read_text(f"https://api.github.com/repos/{LLAMA_REPO}/releases/tags/{build}"))
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError):
        return None
    for entry in release.get("assets", []) if isinstance(release, dict) else []:
        if isinstance(entry, dict) and entry.get("name") == asset:
            digest = str(entry.get("digest") or "")
            if digest.startswith("sha256:") and re.fullmatch(r"[0-9a-f]{64}", digest[7:]):
                return digest[7:]
    return None


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_llama_build(force: bool = False) -> str | None:
    """The current llama.cpp build number, from cache, API or a stale pin.

    Reaches the network once and then remembers. A join that has already got a
    working runtime never calls this at all -- see `ensure_llama_server` -- so
    the cost is one request on the first run of a new machine.
    """
    if not force:
        try:
            cached = BUILD_CACHE.read_text().strip()
            if cached.startswith("b") and cached[1:].isdigit():
                return cached
        except OSError:
            pass

    build = None
    try:
        for release in json.loads(_read_text(RELEASES_API)):
            tag = str(release.get("tag_name", ""))
            # A semver release can sit at the top of this list; only b-builds
            # have the binaries.
            if tag.startswith("b") and tag[1:].isdigit():
                build = tag
                break
    except (urllib.error.HTTPError, urllib.error.URLError, OSError,
            ValueError, AttributeError):
        build = None

    if build is None:
        build = FALLBACK_BUILD

    try:
        BUILD_CACHE.parent.mkdir(parents=True, exist_ok=True)
        BUILD_CACHE.write_text(build)
    except OSError:
        pass
    return build


def _safe_member(name: str) -> str | None:
    """A member's path, made relative and checked, or None to skip it.

    An archive is a file fetched from the internet and unpacked into someone's
    home directory; a member called `../../.ssh/authorized_keys` is the whole
    reason this function exists.
    """
    cleaned = name.replace("\\", "/").lstrip("/")
    parts = [p for p in cleaned.split("/") if p not in ("", ".")]
    if not parts or ".." in parts or ":" in parts[0]:
        return None
    return "/".join(parts)


def _safe_symlink(linkname: str) -> str | None:
    """A symlink target that cannot point outside the extraction directory.

    Not a theoretical worry: an archive that creates a symlink and then writes
    through it can put a file anywhere the user can write, which is the standard
    way "just unpack this tarball" becomes an exploit. Counting the `..`s refuses
    that while still allowing the ordinary targets below.
    """
    target = linkname.replace("\\", "/")
    if not target or target.startswith("/") or ":" in target:
        return None
    depth = 0
    for part in target.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            depth -= 1
            if depth < 0:
                return None
        else:
            depth += 1
    return target


def _extract(archive: Path, into: Path) -> None:
    """Unpack a llama.cpp release into `into`.

    The two archive shapes differ in a way that matters. The tarballs wrap
    everything in one top-level directory (`llama-b11669/...`) which has to be
    stripped, or the binary ends up a level deeper than the `current` pointer
    says. The Windows zips are already flat, and stripping them would lose the
    files altogether.

    The 35 members in the `lib/` payload are the other thing that matters, and the
    first version of this got them wrong in a way worth recording. llama-server is
    linked against `@rpath/libllama-common.0.dylib` on macOS and
    `$ORIGIN/libllama.so` on Linux, and the release ships each library three
    times: the real file at `libllama-common.0.7.0.dylib`, plus two *symlinks*
    named `libllama-common.0.dylib` and `libllama-common.dylib`. Skipping symlinks
    as a safety measure -- the obvious instinct, and what this did at first --
    keeps only the versioned name, so the binary loads, is told the .0 name does
    not exist, and dies before it can print anything useful. The loader's own
    message ("Library not loaded ... tried: ... (no such file)") is the only clue,
    and on a service with no terminal there is no one to read it.

    So symlinks are recreated, with a validated relative target. That is what
    `_safe_symlink` is for: they are load-bearing here, and a plain `extractall`
    would have taken them at face value.
    """
    into.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                relative = _safe_member(member.filename)
                if relative is None:
                    continue
                target = into / relative
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
        return

    with tarfile.open(archive) as tf:
        members = [m for m in tf.getmembers() if _safe_member(m.name)]
        roots = {_safe_member(m.name).split("/", 1)[0] for m in members}
        root = roots.pop() if len(roots) == 1 else None

        for member in members:
            relative = _safe_member(member.name)
            if root is not None:
                prefix = f"{root}/"
                if not relative.startswith(prefix):
                    continue
                relative = relative[len(prefix):]
                if not relative:
                    continue
            target = into / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            # The three-names-per-library scheme described above. Recreated
            # rather than skipped, but only after the target is checked to stay
            # inside `into`.
            if member.issym():
                link = _safe_symlink(member.linkname)
                if link is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                _unlink(target)
                try:
                    os.symlink(link, target)
                except OSError:
                    pass  # Windows without developer mode; the copy below covers it
                continue

            if not member.isfile():
                continue  # no device nodes, no FIFOs

            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                continue
            with src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            os.chmod(target, member.mode | stat.S_IRUSR | stat.S_IWUSR)


def _binary_in(directory: Path) -> Path | None:
    name = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    for candidate in (directory / name, directory / "build" / "bin" / name):
        if candidate.is_file():
            return candidate
    return None


def _runnable(binary: Path) -> bool:
    """Whether this binary actually starts.

    `--version` is the cheapest thing that makes the dynamic loader resolve every
    library, which is the point: a file can exist, be executable, and still be
    unusable because the `libllama-common.0.dylib` it links against is not beside
    it. That failure is otherwise invisible until the node is asked for an answer,
    on a service with no terminal, hours later. One short-lived process per join
    is a small price for turning it into an install-time fact.
    """
    try:
        done = subprocess.run([str(binary), "--version"], timeout=30,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def _installed_binary(cuda_line: str | None = None) -> Path | None:
    """The runtime a previous run installed, if it is still there and still works.

    It must also be the flavour this machine needs now. A Windows machine that
    gains an NVIDIA GPU, or whose driver moves to a newer CUDA line, has a CPU or
    older-line install that is not what it should run any more, so it is fetched
    again rather than trusted.
    """
    try:
        key = CURRENT_RUNTIME.read_text().strip()
    except OSError:
        return None
    if not key:
        return None
    if cuda_line:
        if not key.endswith(f"-cuda-{cuda_line}"):
            return None
    elif "-cuda-" in key:
        return None
    binary = _binary_in(RUNTIME_DIR / key)
    return binary if binary is not None and _runnable(binary) else None


def _install_key(build: str, cuda_line: str | None) -> str:
    """The directory name, and the `current` pointer, for one installed runtime."""
    return f"{build}-cuda-{cuda_line}" if cuda_line else build


def _fetch_verified(build: str, asset: str) -> Path:
    """The cached or downloaded archive for `asset`, checked against GitHub's digest.

    The archive is checked on the download and on any copy already cached. No
    digest, no install: an unverified llama-server is the one thing this must
    never run.
    """
    digest = llama_asset_digest(build, asset)
    if digest is None:
        raise FetchFailed(f"no checksum for {asset} could be read from GitHub, so the download cannot be verified; try again when GitHub is reachable")
    archive = DOWNLOAD_CACHE / asset
    if archive.is_file() and _sha256_file(archive) != digest:
        _unlink(archive)
    if not archive.is_file():
        _download(RELEASE_DOWNLOAD.format(build=build, asset=asset), archive,
                  expected_sha=digest, what=f"the llama.cpp runtime ({build})")
    return archive


def _install_windows_cuda(archives: list[Path], target: Path) -> None:
    """Both CUDA archives into one directory: the build, then the runtime DLLs beside it.

    llama.cpp finds its CUDA runtime next to the executable, so the two archives
    must share a directory. They are extracted into a staging directory and renamed
    into place, so an extraction interrupted halfway is never mistaken for an install.
    """
    if _binary_in(target) is not None and (target / "ggml-cuda.dll").is_file():
        return
    staging = target.with_name(target.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    for archive in archives:
        _extract(archive, staging)
    if target.exists():
        shutil.rmtree(target)
    os.replace(staging, target)


def ensure_llama_server() -> Path:
    """A ready-to-run `llama-server`, downloading llama.cpp on first use.

    A runtime already on disk is returned as-is and never replaced because a
    newer fetch failed -- but one that will not start is replaced rather than
    trusted, because "installed" and "usable" are not the same claim and only one
    of them gets a node back after a reboot. `current` is written last, so a
    download interrupted halfway does not leave a pointer to a half-built
    directory.
    """
    system, machine = platform.system(), platform.machine()
    if llama_asset_name("b0", system, machine) is None:
        raise FetchFailed(
            f"llama.cpp publishes no build for {system} on {machine}. "
            f"See https://github.com/{LLAMA_REPO}/releases"
        )

    # Windows with an NVIDIA GPU gets the CUDA build, never the CPU one. Every
    # other machine keeps the single archive it always had.
    cuda_line = None
    if system == "Windows":
        gpu = detect_nvidia_gpu()
        if gpu is not None:
            cuda_line = windows_cuda_line(gpu)

    installed = _installed_binary(cuda_line)
    if installed is not None:
        return installed

    candidates = []
    for build in (resolve_llama_build(), FALLBACK_BUILD):
        if build and build not in candidates:
            candidates.append(build)

    failures = []
    DOWNLOAD_CACHE.mkdir(parents=True, exist_ok=True)
    for build in candidates:
        key = _install_key(build, cuda_line)
        target = RUNTIME_DIR / key
        names = (windows_cuda_assets(build, cuda_line) if cuda_line
                 else (llama_asset_name(build, system, machine),))
        try:
            archives = [_fetch_verified(build, name) for name in names]
            if cuda_line:
                _install_windows_cuda(archives, target)
            elif _binary_in(target) is None:
                _extract(archives[0], target)
            binary = _binary_in(target)
            if binary is None:
                raise FetchFailed(f"the {build} archive contained no llama-server")
            if system != "Windows":
                os.chmod(binary, 0o755)
            if not _runnable(binary):
                raise FetchFailed(f"the {key} build of llama-server would not start")
            CURRENT_RUNTIME.parent.mkdir(parents=True, exist_ok=True)
            # A file rather than a symlink: creating one on Windows needs
            # elevation, and this has to work for everyone.
            CURRENT_RUNTIME.write_text(key)
            return binary
        except FetchFailed as e:
            failures.append(f"{key}: {e}")
            for name in names:
                _unlink(DOWNLOAD_CACHE / name)

    raise FetchFailed("could not fetch a llama.cpp runtime. " + " · ".join(failures))


RPC_SERVER_NAME = "ggml-rpc-server.exe" if platform.system() == "Windows" else "ggml-rpc-server"


def ensure_rpc_server() -> Path:
    """The `ggml-rpc-server` that ships beside the installed llama-server.

    llama.cpp's release workflow builds with GGML_RPC=ON and packs the whole build
    directory, so the RPC worker is in the same archive as llama-server. A group
    member runs this binary; nothing else is fetched or built.
    """
    binary = ensure_llama_server()
    for candidate in (binary.parent / RPC_SERVER_NAME, binary.parent / "build" / "bin" / RPC_SERVER_NAME):
        if candidate.is_file():
            # A CUDA install whose GPU backend is missing would run the worker on
            # the CPU. This network does not take CPU workers, so refuse it here.
            if "-cuda-" in binary.parent.name and not (binary.parent / "ggml-cuda.dll").is_file():
                raise FetchFailed(f"the installed CUDA build in {binary.parent} has no ggml-cuda.dll, "
                                  f"so {RPC_SERVER_NAME} would run on the CPU. Delete {CURRENT_RUNTIME} and run again")
            return candidate
    raise FetchFailed(f"the installed llama.cpp build has no {RPC_SERVER_NAME}; "
                      f"it was built without RPC support. Delete {CURRENT_RUNTIME} and run again to fetch the current build")


def llama_server_argv(binary, gguf, port: int, *,
                      alias: str | None = None,
                      ctx: int = DEFAULT_CTX,
                      parallel: int = 1,
                      sleep_idle_seconds: int = DEFAULT_SLEEP_IDLE_SECONDS,
                      chat_template: str | None = None,
                      extra: list[str] | None = None) -> list[str]:
    """The command line for this node's `llama-server`.

    Bound to 127.0.0.1 for the same reason Ollama was: the worker is the only
    thing the tunnel or the LAN can reach, and it is the thing holding the
    gateway's token. llama-server has no authentication of its own worth relying
    on, and it does not need any while nothing outside this machine can open a
    connection to it.

    `-np 1` and an explicit `-c` are about memory, not speed. Each parallel slot
    allocates its own KV cache, and the context size is what the cache is sized
    from -- so both of these decide whether the catalogue's `min_ram_gb` figures
    are true. `-ngl` is deliberately absent: llama.cpp's default is `auto`, which
    offloads as far as the hardware allows and already does the right thing on
    Apple's unified memory, on a discrete GPU and on a CPU-only box alike.

    `alias` is not cosmetic. Left unset, the `model` field of every completion
    -- and the `id` in the backend's own /v1/models -- is the path to the
    weights, so a caller in another country is handed
    `/Users/someone/.common-network/models/<sha>/model.gguf` in the reply. The
    worker streams the response through untouched and could not rewrite that
    field without parsing the SSE stream, which is exactly what it must not do.
    Naming the alias here, where the string already exists, is the only place it
    can be fixed.
    """
    argv = [
        str(binary),
        "-m", str(gguf),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-c", str(ctx),
        "-np", str(parallel),
        # The successor to worker.py's `keep_alive` stamp: the process that owns
        # the weights decides when to give them back. 1800s because a node that
        # has run `common join` has already decided its hardware is for this, and
        # the alternative -- Ollama's five-minute default -- meant every idle node
        # was cold, and the first question of the day paid 53-77s for it.
        "--sleep-idle-seconds", str(sleep_idle_seconds),
        "--no-webui",
    ]
    if alias:
        argv += ["--alias", alias]
    if chat_template:
        argv += ["--chat-template", chat_template]
    if extra:
        argv += list(extra)
    return argv


# How long an RPC endpoint gets to accept a TCP connection before it counts as
# unreachable. A live worker, or its SSH tunnel, answers in milliseconds.
RPC_REACHABILITY_TIMEOUT_SECONDS = 2.0

# llama-server's words when a remote RPC peer drops mid-run. Either one in the
# runtime log means the runtime exited because a worker went away.
RPC_WORKER_LOST_MARKERS = ("Remote RPC server crashed", "send failed")


def split_hostport(endpoint: str) -> tuple[str, int]:
    """The (host, port) of a host:port endpoint. Pure: raises ValueError if malformed."""
    host, sep, port = str(endpoint).rpartition(":")
    if not sep or not host or not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError(f"expected host:port, got {endpoint!r}")
    return host, int(port)


def unreachable_rpc_servers(endpoints, timeout=RPC_REACHABILITY_TIMEOUT_SECONDS):
    """Return the host:port endpoints that do not accept a TCP connection in time.

    This only proves something is listening on the port. It does not speak RPC,
    so a worker that is up but wedged still passes.
    """
    unreachable = []
    for endpoint in endpoints:
        host, port = endpoint.rsplit(":", 1)
        try:
            socket.create_connection((host, int(port)), timeout=timeout).close()
        except OSError:
            unreachable.append(endpoint)
    return unreachable


def missing_worker_in_log(log_tail):
    """True if a runtime log tail shows the runtime died because an RPC worker went away.

    Pure: it reads only the text it is given, so the caller decides how to read
    the log and what to do with the answer.
    """
    if isinstance(log_tail, str):
        log_tail = log_tail.splitlines()
    return any(marker in line for line in log_tail for marker in RPC_WORKER_LOST_MARKERS)


def cpu_reporting_devices(devices_text, devices):
    """The configured RPC devices whose discovery line names a CPU backend.

    The network is GPU-only, so a CPU backend behind any configured device is a
    refusal. A CPU token is matched as a whole word after the device's colon, so
    a hostname such as cpu-host does not count. Pure: it reads only the text.
    """
    reporting = []
    for name, rest in re.findall(r"^\s*(RPC[0-9]+):(.*)$", devices_text, re.MULTILINE):
        if name in devices and re.search(r"(?<![A-Za-z0-9])CPU[0-9]*(?![A-Za-z0-9])", rest):
            reporting.append(name)
    return reporting


class DistributedConfig:
    """Operator-managed v0.2 model group; llama.cpp owns verification and KV state.

    Paths resolve relative to the config, never the caller's working directory.
    RPC is only suitable for trusted hosts over a private transport.
    """

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        try:
            data = json.loads(self.path.read_text())
            allowed = {"version", "binary", "target_model", "draft_model", "model_name",
                       "rpc_servers", "devices", "tensor_split", "draft_device",
                       "draft_tokens", "context", "parallel"}
            if not isinstance(data, dict) or set(data) - allowed:
                raise ValueError("unknown configuration fields or invalid object")
            if type(data.get("version")) is not int or data["version"] != 2:
                raise ValueError("version must be 2")
            def file(key):
                value = data[key]
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{key} must be a file path")
                result = (self.path.parent / Path(value).expanduser()).resolve()
                if not result.is_file():
                    raise ValueError(f"{key} does not exist: {result}")
                return result
            self.binary = file("binary")
            self.target = file("target_model")
            self.draft = file("draft_model") if "draft_model" in data and data["draft_model"] is not None else None
            self.name = data["model_name"]
            if not isinstance(self.name, str) or not self.name.strip() or any(ord(c) < 32 for c in self.name):
                raise ValueError("model_name must be a nonempty public model alias")
            self.peers = data["rpc_servers"]
            if not isinstance(self.peers, list) or len(self.peers) < 2 or len(set(self.peers)) != len(self.peers):
                raise ValueError("rpc_servers must contain at least two distinct host:port endpoints")
            import re
            for peer in self.peers:
                if not isinstance(peer, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]+:[0-9]+", peer):
                    raise ValueError("RPC endpoints must be host:port (IPv4 or hostname)")
                if not 1 <= int(peer.rsplit(":", 1)[1]) <= 65535:
                    raise ValueError("RPC port must be between 1 and 65535")
            self.devices = data["devices"]
            if (not isinstance(self.devices, list) or len(self.devices) < 2
                    or len(set(self.devices)) != len(self.devices)
                    or any(not isinstance(d, str) or not re.fullmatch(r"RPC[0-9]+", d) for d in self.devices)):
                raise ValueError("devices must explicitly select distinct RPC devices, e.g. RPC0 and RPC1")
            self.split = data.get("tensor_split", [1] * len(self.devices))
            if (not isinstance(self.split, list) or len(self.split) != len(self.devices)
                    or any(type(v) not in (int, float) or not 0 < v < float("inf") for v in self.split)):
                raise ValueError("tensor_split must provide a positive finite weight per device")
            self.draft_device = data.get("draft_device", "none")
            if not isinstance(self.draft_device, str) or not re.fullmatch(r"none|[a-zA-Z]+[0-9]+", self.draft_device):
                raise ValueError("draft_device must be a single local device or none")
            for key, default, low, high in (("draft_tokens", 4, 1, 32), ("context", 4096, 128, 1048576), ("parallel", 1, 1, 64)):
                value = data.get(key, default)
                if type(value) is not int or not low <= value <= high:
                    raise ValueError(f"{key} must be an integer between {low} and {high}")
                setattr(self, key, value)
            if self.context % self.parallel or self.context // self.parallel < 128:
                raise ValueError("context must divide evenly into parallel slots, with at least 128 tokens per slot")
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise FetchFailed(f"invalid distributed config {self.path}: {e}") from e

    def extra_argv(self):
        args = ["--rpc", ",".join(self.peers), "--device", ",".join(self.devices),
                "--split-mode", "layer", "--tensor-split", ",".join(map(str, self.split)),
                "--n-gpu-layers", "999", "--fit", "off"]
        if self.draft:
            args += ["--spec-type", "draft-simple", "--spec-draft-model", str(self.draft),
                     "--spec-draft-n-max", str(self.draft_tokens),
                     "--spec-draft-device", self.draft_device,
                     "--spec-draft-ngl", "0" if self.draft_device == "none" else "999"]
        else:
            args += ["--spec-type", "none"]
        return args

    def argv(self, port=11436, sleep_idle_seconds=DEFAULT_SLEEP_IDLE_SECONDS):
        return llama_server_argv(self.binary, self.target, port, alias=self.name,
                                 ctx=self.context, parallel=self.parallel,
                                 sleep_idle_seconds=sleep_idle_seconds, extra=self.extra_argv())

    def preflight(self):
        """Check the installed build supports this protocol before registering."""
        try:
            result = subprocess.run([str(self.binary), "--help"], capture_output=True,
                                    text=True, timeout=20)
            help_text = result.stdout + result.stderr
            required = ["--rpc", "--device", "--tensor-split", "--fit"]
            if self.draft:
                required += ["--spec-type", "--spec-draft-model", "--spec-draft-n-max",
                             "--spec-draft-device", "--spec-draft-ngl"]
            if result.returncode or any(flag not in help_text for flag in required):
                raise FetchFailed("llama-server lacks required RPC/speculative flags; use a matching RPC-enabled build on all hosts")
            unreachable = unreachable_rpc_servers(self.peers)
            if unreachable:
                raise FetchFailed(f"RPC worker not accepting connections at {', '.join(unreachable)}; start that worker and its tunnel, then retry")
            result = subprocess.run([str(self.binary), "--rpc", ",".join(self.peers), "--list-devices"],
                                    capture_output=True, text=True, timeout=60)
            devices_text = result.stdout + result.stderr
            import re
            discovered = dict(re.findall(r"^\s*(RPC[0-9]+):\s+(\S+)\s+\(", devices_text, re.MULTILINE))
            if (result.returncode or any(device not in discovered for device in self.devices)
                    or {discovered.get(device) for device in self.devices} != set(self.peers)):
                raise FetchFailed("RPC device discovery failed or a configured device is unavailable; check peer processes, tunnels, and matching runtime builds")
            cpu_devices = cpu_reporting_devices(devices_text, self.devices)
            if cpu_devices:
                raise FetchFailed(f"RPC device(s) {', '.join(cpu_devices)} report a CPU backend; this network is GPU-only, so start a GPU RPC server (CUDA, Vulkan or Metal) on that member")
            # RPC workers are listed too, but they are remote: a draft device must be local.
            local_devices = {name for name in re.findall(r"^\s*([A-Za-z]+[0-9]+):", devices_text, re.MULTILINE)
                             if not name.startswith("RPC")}
            if self.draft and self.draft_device != "none" and self.draft_device not in local_devices:
                raise FetchFailed(f"local draft device {self.draft_device} is unavailable")
        except (OSError, subprocess.SubprocessError) as e:
            raise FetchFailed(f"cannot run distributed runtime: {e}") from e


def main() -> int:
    """Inspect what this module would do, without doing any of it.

    `common join` is a long-running foreground process that provisions several
    gigabytes before it says anything useful, which is a bad place to debug from.
    This is the same code with the side effects left out.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="common-runtime", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", help="resolve a catalogue source string and describe it")
    parser.add_argument("--manifest", help="fetch the registry manifest for an ollama:<tag>")
    parser.add_argument("--build", action="store_true", help="resolve the current llama.cpp build")
    parser.add_argument("--asset", action="store_true", help="the release asset for this platform")
    parser.add_argument("--argv", metavar="SOURCE", help="print the llama-server command line")
    parser.add_argument("--distributed-config", help="validate a v0.2 config and print its command")
    parser.add_argument("--rpc-worker", help="run an installed ggml-rpc-server on loopback (use SSH forwarding for peers)")
    parser.add_argument("--rpc-port", type=int, default=50052)
    args = parser.parse_args()

    if args.rpc_worker:
        if not 1 <= args.rpc_port <= 65535:
            parser.error("--rpc-port must be between 1 and 65535")
        binary = Path(args.rpc_worker).expanduser().resolve()
        if not binary.is_file():
            parser.error(f"RPC binary does not exist: {binary}")
        os.execv(str(binary), [str(binary), "--host", "127.0.0.1", "--port", str(args.rpc_port)])

    if args.distributed_config:
        try:
            config = DistributedConfig(args.distributed_config)
            config.preflight()
        except FetchFailed as e:
            parser.error(str(e))
        print(json.dumps(config.argv(), indent=2))
        return 0

    if args.source:
        spec = resolve_source(args.source)
        print(f"{args.source} -> {spec}" if spec is None else
              f"{args.source}\n  kind       {spec.kind}\n  wire name  {spec.wire_name}\n"
              f"  registry   {spec.registry_ref}\n  gguf       {spec.gguf_url}\n"
              f"  sha256     {spec.gguf_sha256}")
        return 0

    if args.manifest:
        namespace, name, version = parse_ollama_tag(args.manifest)
        manifest = fetch_manifest(namespace, name, version)
        hex_digest, size = model_layer(manifest)
        print(f"{namespace}/{name}:{version}\n  weights  sha256:{hex_digest}  {_size(size)}")
        for media, label in ((TEMPLATE_MEDIA_TYPE, "template"), (PARAMS_MEDIA_TYPE, "params")):
            found = _layer(manifest, media)
            if found:
                text = _blob_text(namespace, name, found[0])
                print(f"  {label:8s} {len(text or '')} bytes  hint={template_hint(text)}")
        return 0

    if args.build:
        print(resolve_llama_build())
        return 0

    if args.asset:
        build = resolve_llama_build()
        print(llama_asset_name(build) or f"no build for {platform.system()}/{platform.machine()}")
        return 0

    if args.argv:
        spec = resolve_source(args.argv)
        if spec is None:
            print(f"not installable: {args.argv}", file=sys.stderr)
            return 1
        print(" ".join(llama_server_argv("llama-server", "model.gguf", 11436)))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
