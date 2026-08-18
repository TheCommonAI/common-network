#!/usr/bin/env python3
"""Run every gateway test suite. `python tests/run_all.py` from `gateway/`.

No pytest, no database, no network beyond localhost, no embedding model — the
suites stub what they need. If this passes, the composition gate, the verifier
and the panel transport are all behaving.
"""
import subprocess
import sys
from pathlib import Path

TESTS = ["test_verify.py", "test_compose.py", "test_panel_flow.py"]

here = Path(__file__).resolve().parent
failed = []

for name in TESTS:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    result = subprocess.run([sys.executable, str(here / name)], cwd=here.parent)
    if result.returncode != 0:
        failed.append(name)

print(f"\n{'=' * 60}")
if failed:
    print(f"FAILED: {', '.join(failed)}")
    sys.exit(1)
print(f"all {len(TESTS)} suites passed")
