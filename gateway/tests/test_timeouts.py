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

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED\n")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all timeout tests passed")
