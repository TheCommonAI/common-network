#!/usr/bin/env python3
"""Tests for common.py -- the reporting half, and the seams it is built on.

There has never been a harness for this file. `gateway/tests/test_timeouts.py`
reads it as source text to check one constant and never imports it, so every
other assertion in this repository about the CLI was made by hand, once, in a
terminal. That is why one bug in particular is pinned below rather than
remarked on.

The bug worth naming: `sanitise()` on the gateway is a strict allowlist, and the
diagnostics this CLI attaches are filtered through it on arrival. A key the
allowlist does not know is dropped **silently** -- `routingMs` was sent by this
file for a day and never stored anywhere. Nothing raised, nothing logged, and
the report looked complete from both ends. `test_the_diagnostics_keys_survive_
the_gateways_allowlist` is the inverse guard: it asserts the exact set of keys
this CLI sends is the set the gateway keeps, so adding one on a whim fails here
instead of disappearing in production.

Everything here is offline and hermetic. The one outbound path taken is to
`127.0.0.1:1`, which refuses instantly, so the failure branches are exercised
without a network and without waiting. `INSTALL_DIR` is redirected to a temp
directory for the whole run: these tests write reports, and they must not write
them into the user's real `~/.common-network`.

    python3 test_common.py
"""
import contextlib
import http.server
import importlib.util
import io
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Imported rather than exec'd, because the first thing that has to be true is
# that importing this file is safe: `common.py` must reach its `__main__` guard
# without touching the network, the disk or sys.argv. If that ever stops being
# true, every line below fails at once, which is the right time to find out.
import common  # noqa: E402

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


# --- the sandbox -------------------------------------------------------------
#
# A module global, because that is how the CLI reads it -- `INSTALL_DIR` is a
# constant every command closes over. Redirecting the three names is the whole
# fixture: no file this run writes can land in the real ~/.common-network.

_SANDBOX = Path(tempfile.mkdtemp(prefix="common-cli-test-"))
common.INSTALL_DIR = _SANDBOX
common.REPORTS_DIR = _SANDBOX / "reports"
common.LAST_EXCHANGE_PATH = _SANDBOX / "last-exchange.json"

DEAD_GATEWAY = "http://127.0.0.1:1"  # refuses instantly; no timeout to wait on

STASH = {
    "schema": 1,
    "created_unix": int(time.time()),
    "local": False,
    "gateway": "https://gateway.example",
    "question": "What is the capital of South Australia?",
    "answer": "The capital of South Australia is Sydney, founded in 1836.",
    "answer_truncated": False,
    "request": {"region": None, "node": None, "model": None, "compose": None},
    "timings": {"routing_ms": 412, "ttft_ms": 8100, "total_ms": 12400},
    "routing": {
        "node": "au-adelaide-laptop", "score": 0.5741, "forced": False,
        "topology": "single", "candidates": 3, "margin_over_runner_up": 0.0321,
        "compose_reason": "one domain", "checks_run": 4, "checks_failed": 0,
        "routing_ms": 412,
        "decision_id": "8f14e45f-ceea-467a-9e4b-1a2b3c4d5e6f",
    },
}

OPTS = {"reason": "made-up", "category": "chat", "rating": None, "contact": None}


class Args:
    """Stands in for argparse's Namespace, with everything unset."""
    reason = category = rating = contact = None
    yes = save_only = json = unsent = retry = False


class Cfg:
    """The two fields `cmd_config` reads, and nothing else."""
    gateway = "https://gw.example"
    region = None


def build(**overrides) -> dict:
    stash = dict(STASH)
    stash.update(overrides.pop("stash", {}))
    opts = dict(OPTS)
    opts.update(overrides.pop("opts", {}))
    return common._build_report(stash, overrides.pop("note", "it said Sydney"),
                                opts, overrides.pop("gateway", STASH["gateway"]))


# --- the report id -----------------------------------------------------------

section("the report id is the gateway's primary key, so it has to be the right shape")
_id = common._new_report_id()
check("it matches the format the gateway validates", bool(common.REPORT_ID_RE.fullmatch(_id)), True)
check("and is exactly that many characters", len(_id), len("COMMON-20261017-1432-A1B2C3D4"))
check("it starts with COMMON-", _id.startswith("COMMON-"), True)
# Four random bytes, not a counter: two reports filed in the same minute must
# not collide, and a counter would also leak how many this machine had filed.
check("two ids in the same minute differ", common._new_report_id() != common._new_report_id(), True)
# The loop exists so a collision with a *stored* file cannot happen by
# construction. Asserted by creating the collision rather than trusting it.
_saved_id = common._new_report_id()
common._report_path(_saved_id).parent.mkdir(parents=True, exist_ok=True)
common._report_path(_saved_id).write_text("{}")
check("an id whose file already exists is not handed out again",
      common._new_report_id() == _saved_id, False)
check("a malformed id does not match", bool(common.REPORT_ID_RE.fullmatch("COMMON-2026-1432-A1B2C3D4")), False)
check("nor does a lowercase tail", bool(common.REPORT_ID_RE.fullmatch("COMMON-20261017-1432-a1b2c3d4")), False)


# --- flags, which arrive unparsed because of argparse.REMAINDER ---------------

section("flags after the verb (REMAINDER hands them over untouched)")
note, opts, err = common._parse_report_flags(["it said Sydney", "--reason", "wrong"])
check("the note keeps its spaces", note, "it said Sydney")
check("the value flag is picked out", opts["reason"], "wrong")
check("nothing errored", err, None)

note, opts, err = common._parse_report_flags(["--reason=made-up", "--save-only"])
check("--flag=value works too", (note, opts["reason"], opts["save_only"]), ("", "made-up", True))

note, opts, err = common._parse_report_flags(["--reason", "wrong", "it", "typed", "at", "the", "start"])
check("only the whole token is a flag; the note is whatever is left", note, "it typed at the start")
check("and a value flag consumes the token after it, wherever it appears",
      common._parse_report_flags(["it", "typed", "--reason", "at", "the"])[0], "it typed the")
check("and --reason with no value is an error, not a silent default",
      common._parse_report_flags(["--reason"])[2], "--reason needs a value")

note, opts, _ = common._parse_report_flags(["-y", "note"])
check("-y is the short --yes", opts["yes"], True)

# The documented cost of the whole approach, asserted so it stays documented
# behaviour rather than becoming a surprise.
check("a quoted flag inside a note is still a flag (the documented caveat)",
      common._parse_report_flags(["I typed --reason"] )[1]["reason"], None)


# --- the closed reason list --------------------------------------------------

section("the reason is a closed list (a countable field, not free text)")
for good in common.REPORT_REASONS:
    opts, err = common._validate_report_options(
        {"reason": good, "category": None, "rating": None, "contact": None})
    check(f"'{good}' is accepted", (opts["reason"], err), (good, None))
for bad in ("hallucinated", "madeup", "wrong-answer", "3", "", "made-up-wrong"):
    _opts, err = common._validate_report_options(
        {"reason": bad, "category": None, "rating": None, "contact": None})
    check(f"{bad!r} is refused", err is not None, True)
# Forgiving about how it is typed, strict about what it means: a trailing space
# or a shout is a typing difference, not a different meaning.
_opts, err = common._validate_report_options(
    {"reason": "Made_Up", "category": None, "rating": None, "contact": None})
check("case, underscores and spaces are normalised, not rejected", (_opts["reason"], err), ("made-up", None))
for typed, wanted in (("MADE UP", "made-up"), ("wrong ", "wrong"), ("  Slow  ", "slow")):
    _opts, err = common._validate_report_options(
        {"reason": typed, "category": None, "rating": None, "contact": None})
    check(f"{typed!r} is read as '{wanted}'", (_opts["reason"], err), (wanted, None))

_opts, err = common._validate_report_options(
    {"reason": None, "category": "nonsense", "rating": None, "contact": None})
check("an unknown category is refused", err is not None, True)
for rating, wanted in ((1, None), (5, None), (0, "err"), (6, "err"), ("three", "err"), (None, None)):
    _opts, err = common._validate_report_options(
        {"reason": None, "category": None, "rating": rating, "contact": None})
    check(f"rating {rating!r} -> {'accepted' if wanted is None else 'refused'}",
          err is None, wanted is None)
_opts, err = common._validate_report_options(
    {"reason": None, "category": None, "rating": None, "contact": "x" * 201})
check("an over-long contact is refused rather than truncated", err is not None, True)
check("the reason list is the gateway's list",
      list(common.REPORT_REASONS), "made-up wrong off-topic truncated refused format slow other".split())


# --- the wire body: a strict subset, built field by field --------------------

section("the wire body is a strict subset of what is stored")
wire = common._report_wire_body(build())
check("the id travels", wire["id"].startswith("COMMON-"), True)
check("mode is problem for a bad answer", wire["mode"], "problem")
check("the reason travels", wire["reason"], "made-up")
check("the exchange travels", wire["exchange"]["question"], STASH["question"])
check("the answer is byte-identical", wire["exchange"]["answer"], STASH["answer"])

# These are this machine's business. `_report_wire_body` builds its output
# field by field rather than deleting from a copy, so this cannot regress by
# someone adding a field to `_build_report`.
LOCAL_ONLY = ("schema", "created_at", "created_unix", "asked_unix", "gateway",
              "client", "send", "note")
check("nothing local-only reaches the network",
      [key for key in LOCAL_ONLY if key in wire], [])
check("and the wire body is exactly these keys",
      sorted(wire), sorted(["id", "mode", "category", "reason", "text", "rating",
                            "contact", "diagnostics", "exchange"]))

# The landmine. The gateway's `_valid_routing` expects `compose_reason` to be an
# *object*; the CLI holds the same fact as the response header, which is a
# *string*. Sending it would be a 400 rather than a drop, so it must not be sent
# -- and it must not be sent even though the field is right there in the stash.
routing = wire["exchange"]["routing"]
check("the header's compose_reason (a string) is not sent as the gateway's object",
      "compose_reason" in routing, False)
check("nor is the local 'forced' marker", "forced" in routing, False)
check("checks_run travels", routing["checks_run"], 4)
check("the decision id travels", routing["decision_id"], STASH["routing"]["decision_id"])
check("the numeric score travels as a number", routing["score"], 0.5741)

# A decision id the gateway cannot parse is a 400, so it is filtered here.
_bad = build(stash={"routing": {"node": "n", "decision_id": "not-a-uuid"}})
check("a malformed decision id is dropped rather than posted",
      "decision_id" in common._report_wire_body(_bad)["exchange"]["routing"], False)
_feedback = build(stash={"routing": {"node": "n", "panel": ["a", "b"], "topology": "panel"}})
check("a panel travels as a list",
      common._report_wire_body(_feedback)["exchange"]["routing"]["panel"], ["a", "b"])
check("an empty panel is not sent as an empty list",
      "panel" in common._report_wire_body(build(stash={"routing": {"panel": []}}))["exchange"].get("routing", {}),
      False)
check("a bool is not a number here",
      "score" in common._report_wire_body(build(stash={"routing": {"score": True}}))["exchange"].get("routing", {}),
      False)
check("a routing object with nothing recognisable in it is omitted entirely",
      "routing" in common._report_wire_body(build(stash={"routing": {"nonsense": 1}}))["exchange"], False)
check("but the question and answer still travel without it",
      common._report_wire_body(build(stash={"routing": {}}))["exchange"]["answer"] != "", True)


# --- diagnostics, and the allowlist that eats them silently ------------------

section("the diagnostics this CLI sends are the ones the gateway keeps")
_diag = build()["diagnostics"]
# Copied from gateway/app/client_observability.py -- `version` by its regex,
# `platform`/`arch` from ENUMS, and the two timings from NUMBERS. Pinned as
# literals on purpose: the whole failure mode is that a key not on this list is
# dropped without a word, so a change here has to be a deliberate one.
KEPT_BY_THE_GATEWAY = {"version", "platform", "arch", "ttftMs", "totalMs"}
check("every key sent is a key kept",
      sorted(set(_diag) - KEPT_BY_THE_GATEWAY), [])
check("and the set is exactly this", sorted(_diag),
      sorted(["version", "platform", "arch", "ttftMs", "totalMs"]))
check("the version is this CLI's", _diag["version"], common.VERSION)
check("the client version regex on the gateway accepts it",
      bool(__import__("re").fullmatch(r"\d+\.\d+\.\d+(?:-[a-z0-9.]+)?", _diag["version"])), True)
check("the platform is an enum value the gateway knows",
      _diag["platform"] in ("win32", "darwin", "linux"), True)
check("the arch too", _diag["arch"] in ("x64", "arm64", "ia32"), True)
# `routingMs` was here and vanished on arrival. Asserted absent, because the
# routing time is carried once, on the routing object, where it survives.
check("routing time is not sent twice (it is dropped there, kept there)",
      "routingMs" in _diag, False)
check("no timings, no keys", common._diagnostics({"timings": {}}).get("ttftMs"), None)


# --- truncation --------------------------------------------------------------

section("caps are the gateway's caps, applied the way the gateway measures")
# The answer is capped in BYTES: 11k emoji is over 32KiB while looking short.
_text, cut = common._truncate_bytes("\U0001f600" * 11000, common.ANSWER_MAX_BYTES)
check("multi-byte text is measured in bytes", cut, True)
check("and the cut never leaves half a character behind",
      len(_text.encode("utf-8")) <= common.ANSWER_MAX_BYTES, True)
check("a short answer is untouched", common._truncate_bytes("hello", 32768), ("hello", False))
# The question is capped in CHARACTERS by the gateway, so it is clipped in them.
check("the question is capped in characters",
      common._truncate_chars("q" * 4001, common.QUESTION_MAX_CHARS)[1], True)
check("and 4000 characters of emoji stays 4000 characters",
      len(common._truncate_chars("\U0001f600" * 4001, common.QUESTION_MAX_CHARS)[0]), 4000)
check("an over-long question is flagged in the record",
      build(stash={"question": "q" * 4001})["exchange"]["question_truncated"], True)
check("an over-long answer is flagged in the record",
      build(stash={"answer": "a" * 40000})["exchange"]["answer_truncated"], True)
check("the cap on the answer and the gateway's agree",
      common.ANSWER_MAX_BYTES, 32768)


# --- the preview -------------------------------------------------------------

section("the preview says what is leaving, and to whom")
_preview = common._report_preview(build())
for phrase in (
    STASH["gateway"],                                        # the exact gateway
    STASH["routing"]["decision_id"],                         # what to join against
    STASH["question"],                                       # in full
    STASH["answer"],                                         # in full
    "au-adelaide-laptop",                                    # which node answered
    "it said Sydney",                                        # the note
    "no files, no other",                                    # what is NOT attached
    "could already read your question",                      # who reads it already
    "makes its operator a second reader",                    # and who the report adds
):
    check(f"the preview contains {phrase[:34]!r}", phrase in _preview, True)
check("an empty answer is called out rather than shown as blank",
      "returned nothing" in common._report_preview(build(stash={"answer": ""})), True)
check("a truncated answer says so",
      "clipped at 32768 bytes" in common._report_preview(build(stash={"answer": "a" * 40000})), True)
check("an unrecorded decision says so plainly",
      "not recorded by the gateway" in common._report_preview(build(stash={"routing": {}})), True)
check("and with no routing at all there is nothing to be 'answered by'",
      "answered by" in common._report_preview(build(stash={"routing": {}})), False)
check("a stale exchange is flagged as stale",
      "more than an hour old" in
      common._report_preview(build(stash={"created_unix": int(time.time()) - 7200})), True)
check("a fresh one is not", "more than an hour old" in _preview, False)


# --- the store ---------------------------------------------------------------

section("one file per report; saved is not the same as sent")
_path = common._write_json_atomic(common._report_path("COMMON-20261017-0900-DEADBEEF"), build())
check("the file is named for the id", _path.name, "COMMON-20261017-0900-DEADBEEF.json")
check("no .part file is left behind", list(_path.parent.glob("*.part")), [])
check("and it reads back", common._read_report(_path)["exchange"]["question"], STASH["question"])
check("a truncated file is skipped rather than fatal",
      common._read_report(_path.parent / "COMMON-20261017-0900-TRUNC.json"), None)
(_path.parent / "COMMON-20261017-0900-JUNK.json").write_text("{not json")
_listed = [p.name for p, _ in common._list_reports()]
check("and it does not take the listing down with it",
      "COMMON-20261017-0900-JUNK.json" in _listed, False)
check("the good one is still listed",
      "COMMON-20261017-0900-DEADBEEF.json" in _listed, True)

# The `sent` flip is the whole state machine on this side.
_record = common._read_report(_path)
check("a fresh report is unsent", _record["send"]["sent"], False)
check("with no attempts recorded", _record["send"]["attempts"], 0)
check("and no error", _record["send"]["last_error"], None)

_sent = build()
# Reports store second-resolution creation times; make this fixture explicitly
# newer instead of relying on execution crossing a wall-clock second boundary.
_sent["created_unix"] = max(r.get("created_unix", 0) for _, r in common._list_reports()) + 1
_sent["send"].update({"sent": True, "attempts": 1})
common._write_json_atomic(common._report_path(_sent["id"]), _sent)
check("a sent report reads back as sent",
      common._read_report(common._report_path(_sent["id"]))["send"]["sent"], True)
check("newest first",
      [r["id"] for _, r in common._list_reports()][0], _sent["id"])

# Pruning mirrors the server's 90 days -- and only for what has been sent.
old = build()
old["id"] = "COMMON-20260101-0900-0000000A"
old["created_at"] = "2026-01-01T09:00:00Z"
old["send"]["sent"] = True
common._write_json_atomic(common._report_path(old["id"]), old)
old_unsent = build(stash={"created_unix": 0})
old_unsent["id"] = "COMMON-20260101-0900-0000000B"
old_unsent["created_at"] = "2026-01-01T09:00:00Z"
common._write_json_atomic(common._report_path(old_unsent["id"]), old_unsent)
common._prune_reports()
_names = [p.name for p, _ in common._list_reports()]
check("a sent report older than 90 days is pruned",
      "COMMON-20260101-0900-0000000A.json" in _names, False)
check("an unsent one is kept however old it is -- it is the only copy",
      "COMMON-20260101-0900-0000000B.json" in _names, True)


# --- the commands, against a gateway that is not there ----------------------

section("--json stdout stays parseable (the app reads it)")
for path in common.REPORTS_DIR.glob("*.json"):
    path.unlink()
common._write_exchange_stash({**{k: v for k, v in STASH.items() if k != "schema"}})

_out, _err = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(_out), contextlib.redirect_stderr(_err):
    try:
        common.cmd_report(DEAD_GATEWAY, True, ["it said Sydney", "--reason", "made-up", "--yes"], Args())
    except SystemExit as exit_code:
        _code = exit_code.code
parsed = json.loads(_out.getvalue())
check("an unreachable gateway still exits non-zero", _code, 1)
check("and stdout is one JSON object, not a preview in front of one",
      (parsed["saved"], parsed["sent"]), (True, False))
check("the report is on disk anyway (write-first is the offline decision)",
      len(list(common.REPORTS_DIR.glob("*.json"))), 1)
# The error travels in the object the caller is parsing, not on the stream it
# is not. Under --json the preview goes to stderr and the result to stdout, so
# an error that only reached stderr would look like a process that was killed.
check("the failure is explained inside the parsed object",
      "refused" in (parsed["error"] or ""), True)
check("and no .part file survives the failure", list(common.REPORTS_DIR.glob("*.part")), [])

section("--save-only writes nothing to the network and points at the retry")
for path in common.REPORTS_DIR.glob("*.json"):
    path.unlink()
_out, _err = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(_out), contextlib.redirect_stderr(_err):
    common.cmd_report(DEAD_GATEWAY, True, ["x", "--reason", "slow", "--save-only"], Args())
parsed = json.loads(_out.getvalue())
check("it is saved", parsed["saved"], True)
check("and not sent", parsed["sent"], False)
check("the file agrees", json.loads(list(common.REPORTS_DIR.glob('*.json'))[0].read_text())["send"]["sent"], False)
check("the preview was still shown, on stderr",
      "Report a bad answer" in _err.getvalue(), True)

section("declining keeps nothing at all")
for path in common.REPORTS_DIR.glob("*.json"):
    path.unlink()
_out, _err = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(_out), contextlib.redirect_stderr(_err):
    common.cmd_report(DEAD_GATEWAY, True, ["x", "--reason", "slow"], Args())
parsed = json.loads(_out.getvalue())
check("a decline is reported as a decline", parsed["reason"], "declined")
check("nothing was written -- that is what declining means",
      list(common.REPORTS_DIR.glob("*.json")), [])

section("there has to be something to report, and something to say")
common.LAST_EXCHANGE_PATH.unlink(missing_ok=True)
_out = io.StringIO()
with contextlib.redirect_stdout(_out), contextlib.redirect_stderr(io.StringIO()):
    try:
        common.cmd_report(DEAD_GATEWAY, True, ["x", "--reason", "slow", "--yes"], Args())
    except SystemExit as exit_code:
        _code = exit_code.code
check("no prior ask -> refuses, exit 1", (_code, json.loads(_out.getvalue())["error"]),
      (1, "nothing to report"))

# The tombstone. Without it, `common ask --local` followed by `common report`
# would offer to send the *previous* network question -- a report about an
# answer the user is not even looking at.
common._write_exchange_stash({"local": True, "gateway": None, "question": "local q",
                              "answer": "local a", "answer_truncated": False,
                              "timings": {}, "routing": None})
_out = io.StringIO()
with contextlib.redirect_stdout(_out), contextlib.redirect_stderr(io.StringIO()):
    try:
        common.cmd_report(DEAD_GATEWAY, True, ["x", "--reason", "slow", "--yes"], Args())
    except SystemExit as exit_code:
        _code = exit_code.code
check("a --local answer is refused, with the real reason",
      (_code, json.loads(_out.getvalue())["error"]),
      (1, "that answer never left this machine"))

common._write_exchange_stash({k: v for k, v in STASH.items() if k != "schema"})
_out = io.StringIO()
with contextlib.redirect_stdout(_out), contextlib.redirect_stderr(io.StringIO()):
    try:
        common.cmd_report(DEAD_GATEWAY, True, [], Args())
    except SystemExit as exit_code:
        _code = exit_code.code
check("no note and no reason -> refuses rather than filing noise",
      (_code, json.loads(_out.getvalue())["error"]), (1, "nothing to say"))

# An empty value is a typed flag, so it is answered as a bad reason with the
# list attached -- more use than swallowing it and complaining about silence.
_out = io.StringIO()
with contextlib.redirect_stdout(_out), contextlib.redirect_stderr(io.StringIO()):
    try:
        common.cmd_report(DEAD_GATEWAY, True, ["--reason", ""], Args())
    except SystemExit as exit_code:
        _code = exit_code.code
check("an empty --reason is answered with the list",
      "made-up, wrong" in (json.loads(_out.getvalue())["error"] or ""), True)

_out = io.StringIO()
with contextlib.redirect_stdout(_out), contextlib.redirect_stderr(io.StringIO()):
    try:
        common.cmd_report(DEAD_GATEWAY, True, ["x", "--reason", "hallucinated"], Args())
    except SystemExit as exit_code:
        _code = exit_code.code
check("an unknown reason is refused before anything is written", _code, 1)
check("and nothing was written", list(common.REPORTS_DIR.glob("*.json")), [])

# --- what `common privacy` and `common config` now claim ---------------------
#
# Both used to say, in different words, that a question is never stored and
# that nothing beyond routing is sent home. That stopped being true the moment
# `common report` existed, and it is the kind of sentence nobody re-reads --
# it was written when it was correct and it stayed on screen afterwards. These
# do not pin the wording; they pin that the qualification is still there.
section("the privacy claims account for a report")

_out = io.StringIO()
with contextlib.redirect_stdout(_out):
    common.cmd_privacy(True)
facts = json.loads(_out.getvalue())
check("the question is no longer flatly 'not stored'",
      facts["request_text"] == "not stored", False)
check("and the exception names the command that makes it",
      "common report" in facts["request_text"], True)
check("a reported answer is its own listed fact",
      "reported_answer" in facts, True)

_out = io.StringIO()
with contextlib.redirect_stdout(_out):
    common.cmd_privacy(False)
_printed = _out.getvalue()
check("the screen version says a report is read by a second reader",
      "gateway" in _printed and "second reader" in _printed, True)
check("and tells the reader what a report contains",
      "common report" in _printed, True)

_out = io.StringIO()
with contextlib.redirect_stdout(_out):
    common.cmd_config(False, Cfg())
_config = _out.getvalue()
check("config no longer claims nothing is sent home, unqualified",
      "nothing sent home beyond" in _config and "common report" in _config, True)


section("--local talks to the model backend, and to nothing else")

# The seam this section exists for. `--local` promises the prompt never leaves
# this machine, and the process answering it changed underneath: it used to be
# Ollama on 11434, and it is now the llama.cpp server join.py binds to 11436.
# Both are OpenAI-shaped, and that is exactly what makes a wrong port here quiet
# rather than loud -- pointed at the old one, this keeps working and keeps
# talking to a process the network did not start. So the port is asserted, not
# assumed, and the request is served by a real socket rather than compared to a
# string in this file.
check("the backend port is the one join.py binds", common.LLAMA_PORT, 11436)
check("and it is written as an address, not as `localhost`",
      common.LLAMA_URL, "http://127.0.0.1:11436")

# Read as text rather than imported, the way gateway/tests/test_timeouts.py reads
# its own constant: join.py pulls worker.py and runtime.py in on use, and a test
# about which port three files share must not need a working install to say so.
# The duplication is deliberate (see the constant's comment in common.py), and
# this is the check that keeps it honest.
_join_src = (Path(__file__).resolve().parent.parent / "join" / "join.py").read_text()
_worker_src = (Path(__file__).resolve().parent.parent / "join" / "worker.py").read_text()
check("the three files that name this port still agree",
      (f"LLAMA_PORT = {common.LLAMA_PORT}" in _join_src,
       f"http://127.0.0.1:{common.LLAMA_PORT}" in _worker_src),
      (True, True))

# A real socket, because the request itself is what has to be asserted: that it
# carries no keep_alive -- Ollama's per-request idle-unload instruction, which
# llama.cpp ignores *silently*, the worst way for it to be wrong -- that it
# carries no X-Common-Client, and that it reaches the two OpenAI routes rather
# than Ollama's /api/tags. One recorder covers all three, and every one of them
# would have failed against the shape this replaced.
_seen: list = []


class _Backend(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):  # the run's own output is the report
        pass

    def _record(self, body):
        _seen.append({
            "method": self.command,
            "path": self.path,
            "headers": {k.lower(): v for k, v in self.headers.items()},
            "body": body,
        })

    def do_GET(self):
        self._record(None)
        payload = json.dumps({"data": [{"id": "qwen3-1.7b"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        self._record(json.loads(raw) if raw else None)
        # HTTP/1.0, which is this handler's default: the connection closes after
        # the body, so the client's line iteration ends there instead of waiting
        # on a length nobody sent.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')
        self.wfile.write(b"data: [DONE]\n\n")


_backend = http.server.HTTPServer(("127.0.0.1", 0), _Backend)
threading.Thread(target=_backend.serve_forever, daemon=True).start()
# An ephemeral port on purpose. Hardcoding 11436 would make this test fail
# whenever the person running it happens to be serving, which is the one state
# it is most likely to be run in.
_real_llama_url = common.LLAMA_URL
common.LLAMA_URL = f"http://127.0.0.1:{_backend.server_address[1]}"
try:
    _out = io.StringIO()
    with contextlib.redirect_stdout(_out), contextlib.redirect_stderr(io.StringIO()):
        common._ask_local("what is a node?", None, True, True)
    _asked = json.loads(_out.getvalue())
finally:
    common.LLAMA_URL = _real_llama_url
    _backend.shutdown()

check("the answer comes back", _asked["answer"], "hi")
check("--local reports no node, because no node answered", _asked["node"], "local")
check("the probe is an OpenAI listing, not Ollama's /api/tags",
      [_s["path"] for _s in _seen][:1], ["/v1/models"])
check("the ask is the OpenAI chat route",
      [_s["path"] for _s in _seen][1:], ["/v1/chat/completions"])
check("the model came from that listing rather than being defaulted",
      _seen[1]["body"]["model"], "qwen3-1.7b")
check("the answer is requested as a stream", _seen[1]["body"]["stream"], True)
check("no keep_alive -- that field is Ollama's, and llama.cpp ignores it",
      "keep_alive" in _seen[1]["body"], False)
check("and neither request identifies the caller to the gateway",
      ["x-common-client" in _s["headers"] for _s in _seen], [False, False])


# --- ask --json streams a single node's answer, never a panel's -------------
#
# The desktop app runs `common ask --json` for every question and has nothing
# to show until this process exits, which on a cold model is tens of seconds
# of a blank bubble. The fix is NDJSON: one `{"delta": "..."}` line per token as
# it arrives from the gateway, flushed immediately, so a reader of the pipe
# sees each word the moment this process does -- followed by the one `{"answer":
# ...}` summary line this always ended on, unchanged, so a caller that only
# reads the last line still gets exactly what it got before.
#
# Deliberately not for a panel or a composed answer: the gateway's own
# `_sse_from_text` synthesises that text whole before the first byte goes out
# (see its docstring), so there is no real generation to narrate -- chunking a
# complete string into fake deltas here would be theatre, reporting a stream
# that never happened. `X-Common-Topology` is what tells this file which case
# it is in, and the two bodies below exist to prove both sides of that branch.
def _sse_server(topology: str, panel: str | None = None):
    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("X-Common-Node", "friends-mac")
            self.send_header("X-Common-Topology", topology)
            self.send_header("X-Common-Candidates", "1")
            if panel is not None:
                self.send_header("X-Common-Panel", panel)
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n')
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n')
            self.wfile.write(b"data: [DONE]\n\n")

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _run_ask(server) -> list[str]:
    gateway = f"http://127.0.0.1:{server.server_address[1]}"
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            common.cmd_ask(gateway, "what is a node?", None, None, False, True, False, False)
    finally:
        server.shutdown()
    return [line for line in out.getvalue().splitlines() if line.strip().startswith("{")]


_single_lines = _run_ask(_sse_server("single"))
_deltas = [json.loads(line) for line in _single_lines[:-1]]
_summary = json.loads(_single_lines[-1])
check("a single-node answer streams one line per delta",
      [d.get("delta") for d in _deltas], ["a", "b"])
check("and ends on the one summary line this always ended on",
      _summary["answer"], "ab")

_panel_lines = _run_ask(_sse_server("panel", panel="friends-mac, guzzler"))
check("a panel's answer is not streamed as deltas -- it was never generated as one",
      len(_panel_lines), 1)
check("the panel's one line is still the complete summary",
      json.loads(_panel_lines[0])["answer"], "ab")


print()
if FAILURES:
    print(f"{len(FAILURES)} of {CHECKS[0]} FAILED\n")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print(f"all {CHECKS[0]} tests passed")
