#!/usr/bin/env python3
"""Tests for join.py, worker.py and runtime.py.

Two kinds of thing live here.

The first is configuration constants that encode a Windows-specific decision --
an address that must be an address and not a name -- and the node identity that
has to survive a restart. Those were the original subject of this file.

The second is the model-runtime seam: what a catalogue `source` means, what
command line it turns into, and whether the node's public model name is still
the string it has always been. That last one is the reason this file exists in
its current shape. `gateway/app/health.py` marks a node healthy only when the
`id` in its `GET /v1/models` is exactly equal to the `model_name` it registered,
and the gateway routes by that name; so the name is not decoration, it is the
key the whole routing layer is built on. Swapping Ollama for llama.cpp touches
every path that produces that string, and a mistake in it would not raise
anywhere -- it would quietly take every node in the fleet out of the routing
table at once. `wire_name` is asserted against its literal historic value below
for that reason, and the assertion is meant to be annoying to change.

Everything here is offline. No gateway, no registry, no network, no model: the
fixtures are minimal manifests, and the template-layer samples are the marker
sequences observed in the real catalogue's registry entries rather than whole
templates. `import runtime` must therefore stay free of import-time side
effects, which is why the module does all of its work inside functions.
"""
import contextlib
import importlib
import io
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import join
import runtime
import worker

FAILURES = []
CHECKS = [0]


def check(name: str, got, want) -> None:
    CHECKS[0] += 1
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL  {name}  (got {got!r}, want {want!r})")
    else:
        print(f"  ok    {name}")


def section(title: str) -> None:
    print(f"\n{title}")


# --- addresses and timeouts -------------------------------------------------

section("IPv4 addresses (Windows localhost resolves to IPv6, breaking tunnels)")
check("LLAMA_URL uses 127.0.0.1, not localhost",
      "127.0.0.1" in join.LLAMA_URL and "localhost" not in join.LLAMA_URL, True)
check("WORKER_URL uses 127.0.0.1, not localhost",
      "127.0.0.1" in join.WORKER_URL and "localhost" not in join.WORKER_URL, True)
check("the model backend and the worker are on different ports",
      join.LLAMA_PORT != join.WORKER_PORT, True)

section("tunnel health check timeout is reasonable")
check("TUNNEL_WARMUP_TIMEOUT_SECONDS is ~30s",
      25 <= join.TUNNEL_WARMUP_TIMEOUT_SECONDS <= 35, True)


# --- the wire name, which the gateway routes on ------------------------------

section("the node's public model name is unchanged by the runtime swap")

# The literal, spelled out rather than computed, because its whole job is to be
# the same string that nodes have been registering since before this change. If
# this fails, every node whose catalogue entry is `ollama:<that tag>` will
# register a name the gateway's health check does not recognise, and routing
# will drop it. Think hard before "fixing" this line.
check("ollama:mathstral:7b -> mathstral:7b",
      runtime.resolve_source("ollama:mathstral:7b").wire_name, "mathstral:7b")
check("ollama:mightykatun/qwen2.5-math:7b keeps its namespace",
      runtime.resolve_source("ollama:mightykatun/qwen2.5-math:7b").wire_name,
      "mightykatun/qwen2.5-math:7b")
check("the scheme is stripped and nothing else",
      runtime.resolve_source("ollama:phi4-mini-reasoning:latest").wire_name,
      "phi4-mini-reasoning:latest")

section("an Ollama tag splits into registry coordinates")
check("a bare tag means the library namespace",
      runtime.parse_ollama_tag("mathstral:7b"), ("library", "mathstral", "7b"))
check("a namespaced tag keeps its namespace",
      runtime.parse_ollama_tag("mightykatun/qwen2.5-math:7b"),
      ("mightykatun", "qwen2.5-math", "7b"))
check("a tag with no version means latest",
      runtime.parse_ollama_tag("llama3.2"), ("library", "llama3.2", "latest"))

section("what common-join can and cannot provision")
for source in ("api:openai/gpt-4", "graph:something", "", "ollama:"):
    check(f"{source!r} is not installable",
          runtime.resolve_source(source), None)
for bad in ("gguf:https://example.com/m.gguf",                 # no digest
            "gguf:https://example.com/m.gguf#" + "a" * 63,    # short digest
            "gguf:https://example.com/m.gguf#" + "z" * 64):   # not hex
    check(f"{bad[:34]!r}... is rejected", runtime.resolve_source(bad), None)


# --- the manifest parser -----------------------------------------------------

section("reading a registry manifest")

# Minimal but structurally real: the same three layers a catalogue model
# actually carries, with the media types that identify them.
_DIGEST = "a" * 64
MANIFEST = {
    "schemaVersion": 2,
    "layers": [
        {"mediaType": runtime.TEMPLATE_MEDIA_TYPE, "digest": "sha256:" + "b" * 64, "size": 1200},
        {"mediaType": runtime.PARAMS_MEDIA_TYPE, "digest": "sha256:" + "c" * 64, "size": 44},
        {"mediaType": runtime.MODEL_MEDIA_TYPE, "digest": "sha256:" + _DIGEST, "size": 4113288768},
    ],
}
check("the model layer is found by media type, not by position",
      runtime.model_layer(MANIFEST), (_DIGEST, 4113288768))
check("its digest is returned without the sha256: prefix",
      runtime.model_layer(MANIFEST)[0][:2], "aa")
try:
    runtime.model_layer({"layers": []})
    check("a manifest with no weights raises FetchFailed", "no raise", "FetchFailed")
except runtime.FetchFailed:
    check("a manifest with no weights raises FetchFailed", "FetchFailed", "FetchFailed")
check("the template layer is found separately",
      runtime._layer(MANIFEST, runtime.TEMPLATE_MEDIA_TYPE)[0], "b" * 64)
check("a media type that is absent returns None, not a crash",
      runtime._layer(MANIFEST, "application/vnd.ollama.image.license"), None)


# --- template hints ----------------------------------------------------------

section("a template hint points at a built-in llama.cpp chat template")

# The registry's template layer is Go, which llama.cpp cannot load -- so the
# layer is only ever read as a signature. These are the marker sequences that
# distinguish the families, taken from the real entries in the catalogue.
HINTS = [
    ("llama3",     "{% for m in .Messages %}<|start_header_id|>{{ m.Role }}<|end_header_id|>"),
    ("chatml",     "<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>"),
    ("gemma",      "<start_of_turn>user\n{{ .Prompt }}<end_of_turn>"),
    ("phi4",       "<|user|>{{ .Prompt }}<|end|>"),
    ("llama2",     "<<SYS>>{{ .System }}<</SYS>>[INST] {{ .Prompt }} [/INST]"),
    ("mistral-v1", "[INST] {{ .Prompt }} [/INST]"),
    ("deepseek",   "{{ .Prompt }}<|end▁of▁sentence|>"),
]
for want, layer in HINTS:
    check(f"{want:11} is recognised", runtime.template_hint(layer), want)
check("an empty layer means no hint rather than a wrong one",
      runtime.template_hint(""), None)
check("an absent layer means no hint", runtime.template_hint(None), None)
check("an unrecognised template says so instead of guessing",
      runtime.template_hint("{{ .Prompt }} please answer nicely"), None)


# --- params ------------------------------------------------------------------

section("parameters the registry carries travel with the model")
check("stop tokens are read as JSON",
      runtime.read_params('{"stop": ["[INST]", "[/INST]"]}'),
      {"stop": ["[INST]", "[/INST]"]})
check("num_ctx is read too",
      runtime.read_params('{"num_ctx": 2048, "stop": ["<|im_start|>"]}')["num_ctx"], 2048)
check("an absent layer gives an empty dict", runtime.read_params(None), {})
check("a layer that is not JSON gives an empty dict, not an exception",
      runtime.read_params("PARAMETER stop [INST]"), {})
check("JSON that is not an object gives an empty dict",
      runtime.read_params('["not", "an", "object"]'), {})


# --- the llama.cpp command line ----------------------------------------------

section("the llama-server command line")

check("all six published targets are named",
      [runtime.llama_asset_name("b1", s, m) for s, m in (
          ("Darwin", "arm64"), ("Darwin", "x86_64"),
          ("Linux", "x86_64"), ("Linux", "aarch64"),
          ("Windows", "AMD64"), ("Windows", "ARM64"))],
      ["llama-b1-bin-macos-arm64.tar.gz", "llama-b1-bin-macos-x64.tar.gz",
       "llama-b1-bin-ubuntu-x64.tar.gz", "llama-b1-bin-ubuntu-arm64.tar.gz",
       "llama-b1-bin-win-cpu-x64.zip", "llama-b1-bin-win-cpu-arm64.zip"])
check("a platform llama.cpp does not build for returns None",
      runtime.llama_asset_name("b1", "Plan9", "mips"), None)

ARGV = runtime.llama_server_argv("/r/llama-server", "/m/model.gguf", 11436,
                                 alias="mathstral:7b")
check("the weights are named with -m", ARGV[1:3], ["-m", "/m/model.gguf"])
check("it binds localhost and does not face the network",
      ARGV[ARGV.index("--host") + 1], "127.0.0.1")
check("the context is explicit, so min_ram_gb stays honest",
      ARGV[ARGV.index("-c") + 1], "4096")
check("one slot, because each slot allocates its own KV cache",
      ARGV[ARGV.index("-np") + 1], "1")
check("idle unload is the process's own decision, not a hint per request",
      ARGV[ARGV.index("--sleep-idle-seconds") + 1], "1800")
check("the web UI is off -- it is a file browser over the weights directory",
      "--no-webui" in ARGV, True)
check("the model answers under its registered name, not its path",
      ARGV[ARGV.index("--alias") + 1], "mathstral:7b")
check("-ngl is absent: llama.cpp's default is already auto",
      "-ngl" in ARGV or "--n-gpu-layers" in ARGV, False)
check("no chat template is passed when there is no hint",
      "--chat-template" in ARGV, False)
_HINTED = runtime.llama_server_argv("/r/l", "/m/g", 11436, chat_template="llama3")
check("a hint becomes a built-in --chat-template",
      _HINTED[_HINTED.index("--chat-template") + 1], "llama3")


# --- the worker's ceiling on one generation ----------------------------------

section("the worker's max_tokens ceiling")

check("a sane request is taken at its word", worker.bounded_max_tokens(64, 1024), 64)
check("a request for less than the ceiling is honoured",
      worker.bounded_max_tokens(1000, 1024), 1000)
check("more than the ceiling is capped", worker.bounded_max_tokens(10 ** 9, 1024), 1024)
check("an absent value gets the ceiling", worker.bounded_max_tokens(None, 1024), 1024)
check("zero gets the ceiling -- 0 means 'unset', not 'generate nothing'",
      worker.bounded_max_tokens(0, 1024), 1024)
check("-1 gets the ceiling -- it is the unbounded case, and the bug",
      worker.bounded_max_tokens(-1, 1024), 1024)
check("a negative float gets the ceiling",
      worker.bounded_max_tokens(-0.5, 1024), 1024)
check("a bool gets the ceiling, because True is not a token count",
      worker.bounded_max_tokens(True, 1024), 1024)
check("False gets the ceiling too, not zero",
      worker.bounded_max_tokens(False, 1024), 1024)
check("a string gets the ceiling rather than raising in the request path",
      worker.bounded_max_tokens("500", 1024), 1024)
check("a float is floored to an int", worker.bounded_max_tokens(12.9, 1024), 12)

section("the worker's default ceiling is the documented one")
check("DEFAULT_MAX_TOKENS is 1024", worker.DEFAULT_MAX_TOKENS, 1024)
check("the upstream default is the model backend's port, on 127.0.0.1",
      worker.DEFAULT_UPSTREAM, "http://127.0.0.1:11436")


section("the worker keeps its config tidy")
_cfg = worker.WorkerConfig("t", "m", "http://127.0.0.1:11436/")
check("a trailing slash on the upstream is trimmed once, here",
      _cfg.upstream_url, "http://127.0.0.1:11436")
check("the model name is kept exactly as given", _cfg.model, "m")


section("the sibling modules that must be refreshed together")
check("runtime.py is in the self-update list",
      "runtime.py" in join.SIBLING_MODULES, True)
check("worker.py still is", "worker.py" in join.SIBLING_MODULES, True)


# --- what a failed tunnel is allowed to say ---------------------------------

section("a rate-limited quick tunnel is named as one")

# The real lines, copied from cloudflared 2026.7.1 rather than invented. The
# second is where the status code is, and the third is the JSON-parse failure
# that hides it -- cloudflared parses the error body as JSON whatever the status
# code is (cloudflare/cloudflared#1302), so "429 Too Many Requests" arrives as
# "invalid character 'e' looking for beginning of value" and the honest part is
# buried in the middle of somebody else's log line. A test written against a
# paraphrase of this would pass while the live text went unmatched.
_RATE_LIMITED = [
    "2026-10-16T23:07:55Z INF Requesting new quick Tunnel on trycloudflare.com...",
    "2026-10-16T23:07:55Z ERR Error unmarshaling QuickTunnel response: error code: 1015",
    ' error="invalid character \'e\' looking for beginning of value" status_code="429 Too Many Requests"',
    "failed to unmarshal quick Tunnel: invalid character 'e' looking for beginning of value",
]
_headline, _advice = join.tunnel_failure_message(_RATE_LIMITED)
check("the headline says rate limited, and not that cloudflared simply died",
      "rate limited" in _headline and "exited" not in _headline, True)
check("the status code is in it", "429" in _headline, True)
check("the advice is to wait", any("wait" in a for a in _advice), True)
# The one wrong reading to guard against: a rate limit is Cloudflare declining
# this machine, not the Common Network being down, and the difference decides
# whether the contributor waits or goes looking for a fault that is not there.
check("it does not blame the network", "network" not in _headline.lower(), True)


section("a bare status code is enough, in either spelling")
for _line in ("error code: 1015", "status_code=429", "429 Too Many Requests"):
    check(f"{_line!r} reads as rate limited",
          "rate limited" in join.tunnel_failure_message([_line])[0], True)


section("a tunnel that died of something else is not called a rate limit")
_OTHER = [
    "ERR Failed to dial edge: dial tcp 198.41.200.13:7844: connect: connection refused",
]
_other_headline, _other_advice = join.tunnel_failure_message(_OTHER)
check("the headline is the plain one",
      _other_headline, "cloudflared exited before a tunnel URL appeared.")
check("and nothing is claimed about rate limits",
      "rate limited" not in _other_headline, True)
check("a tail that said something gets no run-it-by-hand hint",
      _other_advice, [])


section("cloudflared printing nothing says so, and says what to run")
_silent_headline, _silent_advice = join.tunnel_failure_message([])
check("the headline admits it said nothing",
      _silent_headline, "cloudflared exited before a tunnel URL appeared, and printed nothing.")
check("the advice names the command to run by hand",
      any("cloudflared tunnel --url" in a for a in _silent_advice), True)
check("whitespace counts as nothing too",
      join.tunnel_failure_message(["", "   "])[0], _silent_headline)
check("the kept tail is bounded, and small enough to read",
      2 <= join.TUNNEL_TAIL_LINES <= 20, True)


# --- a directory must not stand in for the module ---------------------------
# This is not a hypothetical. An install was found with `runtime/` (the
# unpacked llama.cpp build) and no runtime.py beside it, and because Python
# imports a directory with no __init__.py as a namespace package, `import
# runtime` succeeded and every call into it died with
#
#     AttributeError: module 'runtime' has no attribute 'is_installable'
#
# A message naming neither the missing file nor the directory standing in for
# it. Worse, it made join.py's own repair unreachable: _require_runtime()
# catches ImportError to decide whether to fetch the file, and no ImportError
# ever came. Both directions are checked below, because "refuse the impostor"
# is only half a fix -- the other half is accepting the real file afterwards.

section("a runtime/ directory must not pass as a missing runtime.py")

_saved_path = list(sys.path)
_saved_runtime = sys.modules.pop("runtime", None)
_here = Path(__file__).resolve().parent
_scratch = tempfile.mkdtemp(prefix="common-runtime-impostor-")
try:
    # The shape the llama.cpp build unpacks into: a versioned directory, no
    # __init__.py anywhere, sitting where runtime.py should be.
    (Path(_scratch) / "runtime" / "b11700").mkdir(parents=True)
    # This file's own directory comes off the path for the duration, and that is
    # not tidiness -- it is the condition being reproduced. A namespace package
    # is a *fallback*: PathFinder notes the namespace portion and keeps
    # searching, and a real runtime.py on any later path entry wins outright. So
    # a `runtime/` directory only ever answers for runtime.py when there is no
    # runtime.py anywhere on the path at all, which is precisely the state the
    # affected install was in. Leaving this directory on the path would test the
    # opposite claim -- that a directory wins -- and it does not, so the test
    # would fail while the bug stayed unfixed.
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _here]
    sys.path.insert(0, _scratch)
    importlib.invalidate_caches()
    check("a bare directory named runtime/ is refused",
          join._import_runtime(), None)

    # The other direction, and the one that makes the fetch in
    # _require_runtime() a repair rather than a second way to fail: write the
    # file beside the directory and it is the file that is imported.
    shutil.copyfile(Path(__file__).resolve().parent / "runtime.py",
                    Path(_scratch) / "runtime.py")
    sys.modules.pop("runtime", None)
    importlib.invalidate_caches()
    _real = join._import_runtime()
    check("the file is accepted once it is written beside the directory",
          _real is not None and Path(_real.__file__).name == "runtime.py", True)
    check("and it is the real module, with the attribute the call sites want",
          hasattr(_real, "is_installable"), True)
finally:
    sys.path[:] = _saved_path
    sys.modules.pop("runtime", None)
    if _saved_runtime is not None:
        sys.modules["runtime"] = _saved_runtime
    importlib.invalidate_caches()
    shutil.rmtree(_scratch, ignore_errors=True)

check("the module the rest of this file uses is still the real one",
      join._import_runtime() is runtime, True)


section("a redirected download says how far along it is, and how big")

# The other half of what changed when the model download moved into the join.
# The desktop app used to get byte-level progress from Ollama's pull stream; it
# now follows `join.log` by byte offset, and this line is the only thing in that
# log that says the download is moving. A bare percentage -- which is what the
# redirected branch used to write -- leaves a multi-gigabyte wait readable as
# ten numbers with no idea what is being waited for, so the sizes are the load-
# bearing part and not decoration.
#
# Forced to non-tty rather than left to wherever this test is run from: the
# branch under test is the one a pipe, launchd and join.log all take, and a
# terminal would exercise the other one.
_total = 100 * 1024 * 1024
_sink = io.StringIO()
with contextlib.redirect_stderr(_sink):
    prog = runtime._Progress("model", _total)
    prog.tty = False
    for _pct in (0, 9, 10, 25, 100):
        prog.update(_total * _pct // 100)
    prog.close()
_marks = [line.split() for line in _sink.getvalue().splitlines() if line.strip()]

check("one line per 10% mark, and not one per update", len(_marks), 3)
check("each line names the download, the percentage and both sizes",
      _marks[0], ["model", "10%", "10", "MB", "of", "100", "MB"])
check("the last mark is the whole file, not a rounded-off 99%",
      _marks[-1], ["model", "100%", "100", "MB", "of", "100", "MB"])

# And the terminal branch has to stay what the docstring says it is: one line
# repainted, never a series. A newline here is what would turn join.log into a
# file of thousands of lines, which is the reason the two shapes exist at all.
_tty_sink = io.StringIO()
with contextlib.redirect_stderr(_tty_sink):
    prog = runtime._Progress("model", _total)
    prog.tty = True
    prog.update(0)
    prog.update(_total)
    prog.close()
_tty_out = _tty_sink.getvalue()
check("a terminal still gets one repainted line and no newlines",
      (_tty_out.count("\n"), _tty_out.count("\r") > 1), (0, True))


print()
if FAILURES:
    print(f"{len(FAILURES)} of {CHECKS[0]} FAILED\n")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print(f"all {CHECKS[0]} tests passed")
