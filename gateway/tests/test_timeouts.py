"""The timeout budget — who may wait for whom, and how long.

A configuration relationship, asserted because a configuration relationship is
what broke, and nothing in the suite could have caught it. Every number here is
defensible on its own; only their ordering was wrong.

`forward_timeout_seconds = 60.0` was one value covering connect AND the wait for
response headers. A cold 7B on this laptop takes ~57s to load (the number is in
config.py, under compose_member_timeout_seconds). So a request to a cold node
spent 57s loading, tripped a 60s ceiling it had no way to know about, and then
retried the *backup* — a different, lower-ranked node — for up to another 60s.
Both reported symptoms came out of that one number: "sometimes randomly it's
super slow", and an answer from the wrong specialist. A routing bug and a
latency bug with a shared cause, and neither was in the router.

Checked here:

  1. The read timeout clears a cold start with real margin. Not "greater than
     57" — that is a 3-second margin on a measurement taken from the *fast*
     machine in the fleet, which is the mistake that caused this.
  2. Connect is short and separate, so a node that is down fails fast instead
     of holding a request open for the read timeout.
  3. The panel's per-member cutoff still exceeds a cold start, or every panel
     held with a cold member would drop it.
  4. Both clients wait longer than the gateway, so the informative timeout is
     the one that fires. Read from the client sources, because the invariant
     spans files and nothing else would notice it drifting.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from app.config import settings  # noqa: E402

FAILURES = []
REPO = HERE.parent.parent

# Imported rather than read, because the number below is a constant in the
# worker and this check is about the pair. A parse failure would be a failure
# too, but there is no reason to invite one: the worker is stdlib-only and
# imports with no side effects.
sys.path.insert(0, str(REPO / "join"))
import worker  # noqa: E402


def check(name: str, got, want) -> None:
    if got != want:
        FAILURES.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  FAIL  {name}  (got {got!r}, want {want!r})")
    else:
        print(f"  ok    {name}")


# The measured cold start that every number below is sized against. Measured on
# the development laptop, which is the fast case — donor machines may be slower,
# so the margins are multiples of it rather than increments.
MEASURED_COLD_START_S = 57.0

print("\na cold start must fit inside the read timeout, with margin")
check("a cold start fits at least 3x over",
      settings.forward_read_timeout_seconds >= 3 * MEASURED_COLD_START_S, True)
# 3x is the whole point. 60s was 1.05x -- nominally enough, actually a coin
# toss once prompt evaluation and the tunnel are added on top of the load.
check("the old value would not have passed this",
      60.0 >= 3 * MEASURED_COLD_START_S, False)

print("\nconnect is separate, and short")
check("connect is under the read timeout",
      settings.forward_connect_timeout_seconds < settings.forward_read_timeout_seconds, True)
# A node that is up accepts a socket in milliseconds. Anything still connecting
# after this long is down, and a request should stop waiting rather than occupy
# a slot for the full read timeout.
check("a dead node fails fast", settings.forward_connect_timeout_seconds <= 15.0, True)
check("write does not outlast read",
      settings.forward_write_timeout_seconds <= settings.forward_read_timeout_seconds, True)

print("\nthe panel cutoff still clears a cold start")
# Deliberately shorter than the read timeout -- a member that has not answered
# is dropped so the others proceed -- but it must still clear a cold start, or
# a panel held with one cold member loses it every time and silently degrades
# to an unaggregated answer.
check("a panel member may take longer than a cold start",
      settings.compose_member_timeout_seconds > MEASURED_COLD_START_S, True)
check("but is still cut off before the single-route read timeout",
      settings.compose_member_timeout_seconds < settings.forward_read_timeout_seconds, True)

print("\na bounded generation still fits inside the read timeout")
# The worker now caps every request at DEFAULT_MAX_TOKENS, because nothing
# upstream bounded generation at all: on 2026-10-13 a node whose model failed to
# emit EOS ran to the context window and took 4m24s to return repetition salad.
# The cap was chosen against this relationship rather than picked round, so if
# either number moves the pair has to be re-checked instead of silently drifting.
MEASURED_TOKENS_PER_SECOND = 13.0
check("a capped worst case lands inside the read timeout",
      worker.DEFAULT_MAX_TOKENS / MEASURED_TOKENS_PER_SECOND
      < settings.forward_read_timeout_seconds, True)
# And generous enough not to truncate real answers, which stop on their own at
# 20-200 tokens. This is the check that fails if someone "tightens" the cap.
check("the cap is well clear of an ordinary answer",
      worker.DEFAULT_MAX_TOKENS >= 4 * 200, True)

print("\nclients outwait the gateway, so the gateway's error is the one seen")
# Read from source rather than imported: these clients are installed as
# standalone scripts (they self-update by overwriting their own file), so
# importing them is not how they are ever used. A parse failure is itself a
# failure -- it means the constant moved and this check went silently blind.
CONSTANT = "REQUEST_TIMEOUT_SECONDS"
for rel in ("common/common.py", "chat/chat.py"):
    path = REPO / rel
    match = re.search(rf"^{CONSTANT}\s*=\s*(\d+)", path.read_text(), re.MULTILINE)
    check(f"{rel} declares {CONSTANT}", match is not None, True)
    if match is None:
        continue
    seconds = int(match.group(1))
    check(f"{rel} waits longer than the gateway",
          seconds > settings.forward_read_timeout_seconds, True)
    # And is actually used, not just declared. A constant nothing references
    # would make the two checks above pass while the client kept its old value.
    uses = len(re.findall(rf"urlopen\([^)]*timeout={CONSTANT}\)", path.read_text()))
    check(f"{rel} applies it to every long request", uses >= 1, True)

# join is the third client, and the one the rule above was missing.
#
# Its gateway calls took http_json's 10s default, which is under the gateway's
# own database command timeout -- so a slow catalogue read could only ever
# surface as a client socket timeout, whatever the gateway knew and could have
# said. The first call of a join is exactly where that lands: a Railway container
# that has gone to sleep accepts the connection at once and answers seconds
# later, measured at 8.1s to first byte cold against 0.53s warm. 10s clears
# neither a slower wake nor a slower machine.
rel = "join/join.py"
source = (REPO / rel).read_text()
match = re.search(rf"^{CONSTANT}\s*=\s*(\d+)", source, re.MULTILINE)
check(f"{rel} declares {CONSTANT}", match is not None, True)
if match is not None:
    seconds = int(match.group(1))
    # Held against the database budget rather than the read budget: /catalogue
    # and /assign are served from the catalogue tables and are never forwarded
    # to a node, so the gateway's own timeout on them is a query, not a cold
    # model. It still has to be shorter than the read budget, so that a gateway
    # which is genuinely unwell is the one to give up.
    check(f"{rel} waits longer than the gateway's database",
          seconds > settings.db_command_timeout_seconds, True)
    check(f"{rel} still lets the gateway fail first",
          seconds < settings.forward_read_timeout_seconds, True)

# Declared is not applied. Both of the calls a join makes before it registers
# must go through the timeout and the retry, because those two are the whole
# fix -- leaving one on http_json's default would keep the original failure and
# pass every check above.
for fn in ("fetch_catalogue", "call_assign"):
    body = re.search(rf"^def {fn}\(.*?(?=^def |\Z)", source, re.MULTILINE | re.DOTALL)
    check(f"{rel} {fn} goes through the retrying helper",
          body is not None and "gateway_json(" in body.group(0), True)
    check(f"{rel} {fn} no longer takes the 10s default",
          body is not None and "http_json(" not in body.group(0), True)

# And the retrying is bounded, so a gateway that is simply gone costs a wait
# with a ceiling rather than one full timeout per attempt.
check(f"{rel} caps the time it spends retrying",
      re.search(r"^GATEWAY_RETRY_MAX_WAIT_SECONDS\s*=\s*[\d.]+", source, re.MULTILINE) is not None, True)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED\n")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all timeout tests passed")
