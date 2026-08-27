"""Isolated execution for model-generated code.

Runs in a fresh subprocess (own memory space, own crash domain) with a wall-clock
timeout and a resource cap on CPU time / memory. This is process-level isolation,
not container-level -- it stops a hang or a crash from taking down the bench run,
but a determined malicious payload could still touch the filesystem or network.
That's an acceptable bar for scoring completions from known, trusted models in a
controlled benchmark run. TODO(v0.3): move to a container (e.g. Docker) sandbox
if this ever runs untrusted, user-submitted code.
"""
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 10
MEMORY_LIMIT_BYTES = 512 * 1024 * 1024  # 512MB


def _limit_resources():
    """Best-effort caps. Never raise: anything thrown in a preexec_fn comes back
    as SubprocessError, which run_program reports as a failed program -- so a
    limit that cannot be applied would score every correct completion as wrong.

    RLIMIT_AS is the specific hazard and is Linux-only here: CPython on macOS
    reserves more address space than a 512MB cap allows and dies before running
    a line, so on a Mac this made run_program return False for *every* input,
    correct or not. The wall-clock timeout in run_program is what actually
    bounds a runaway program.
    """
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (DEFAULT_TIMEOUT_SECONDS, DEFAULT_TIMEOUT_SECONDS))
    except (ValueError, OSError):
        pass
    if sys.platform.startswith("linux"):
        try:
            resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
        except (ValueError, OSError):
            pass


def run_program(source: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> tuple[bool, str]:
    """Run a self-contained Python program. Returns (ok, error_message)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(source)
        path = f.name

    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout,
            preexec_fn=_limit_resources if sys.platform != "win32" else None,
        )
        if result.returncode != 0:
            return False, (result.stderr or result.stdout or "non-zero exit").strip()[-2000:]
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    except Exception as e:
        return False, str(e)
    finally:
        Path(path).unlink(missing_ok=True)
