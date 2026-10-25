"""Join the decentralised layer network: make sure a runtime exists, then run commonnet.node.

    python3 join.py --network                 # the default network's seeds
    python3 join.py --network URL|FILE        # a seed peer's URL, or a manifest file

Standard library only, so join.py can import it before anything is installed.

A runtime is a Python that can import commonnet and its dependencies (cryptography, torch, and
mlx or bitsandbytes; a machine with neither Apple silicon nor an NVIDIA GPU serves on the CPU with
plain torch). If the interpreter running join.py already has them, it is used. Otherwise a
virtual environment is made under ~/.common-network/runtime-venv and the pinned requirements for this
platform are installed into it. commonnet itself comes from a checkout (COMMONNET_SRC) or from the
repository tarball for COMMONNET_REF, unpacked once into ~/.common-network/commonnet-src.

The Windows/Linux NVIDIA path (CUDA torch + bitsandbytes) is written from the documentation and has
not been run on a machine with a GPU. The CPU path (CPU torch, bfloat16) is tested on a Mac's CPU only,
not on Windows. The macOS Apple-silicon path is the tested one.

Nothing here runs unless --network was given: join.py without it behaves as it always has.
"""
from __future__ import annotations

import io
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = "TheCommonAI/common-network"
HOME_DIR = Path(os.environ.get("COMMON_NETWORK_HOME") or Path.home() / ".common-network")
RUNTIME_VENV = HOME_DIR / "runtime-venv"
SOURCE_DIR = HOME_DIR / "commonnet-src"
# Pin this to a release tag (for example v0.3.3) when cutting one; "main" follows the repository.
DEFAULT_REF = "main"
TARBALL_URL = "https://github.com/" + REPO + "/archive/{ref}.tar.gz"
DEFAULT_NETWORK_ENV = "COMMONNET_NETWORK"
IMPORT_CHECK = "import commonnet, cryptography, torch, numpy, safetensors, transformers, aiortc"
# The backend's own modules too: a Python with torch but no mlx (or no bitsandbytes) cannot serve a block.
BACKEND_IMPORTS = {"mac-arm64": "import mlx.core, mlx_lm", "nvidia": "import bitsandbytes", "cpu": ""}

# Pinned, so a join today and a join next month install the same thing. Versions are the ones
# the tests ran with (see commonnet/tests).
MAC_REQUIREMENTS = (
    "mlx==0.32.4", "mlx-lm==0.32.2", "numpy>=2.0", "torch==2.11.0", "transformers==5.8.0",
    "safetensors>=0.7.0", "tokenizers>=0.22", "cryptography>=46.0", "aiortc==1.15.0",
)
NVIDIA_REQUIREMENTS = (  # UNTESTED: no Windows/CUDA machine was available
    "numpy>=2.0", "transformers==5.8.0", "safetensors>=0.7.0", "tokenizers>=0.22",
    "cryptography>=46.0", "bitsandbytes>=0.45", "accelerate>=1.0", "aiortc==1.15.0",
)
CPU_REQUIREMENTS = (  # no mlx, no bitsandbytes; torch comes from the CPU wheel index (small, no CUDA libraries)
    "numpy>=2.0", "transformers==5.8.0", "safetensors>=0.7.0", "tokenizers>=0.22", "cryptography>=46.0", "aiortc==1.15.0",
)
CPU_TORCH_REQUIREMENT = "torch==2.11.0"
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"  # Windows and Linux; macOS wheels come from PyPI
CUDA_TORCH_REQUIREMENT = "torch==2.11.0"
CUDA_TORCH_INDEX = "https://download.pytorch.org/whl/cu128"  # UNTESTED


class LayersError(RuntimeError):
    """Something stopped the join. The message says what to do about it."""


def has_nvidia(which=shutil.which, run=subprocess.run) -> bool:
    exe = which("nvidia-smi")
    if not exe:
        return False
    try:
        out = run([exe, "-L"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and "GPU" in (out.stdout or "")


def platform_kind(system: str | None = None, machine: str | None = None, nvidia: bool | None = None) -> str:
    """'mac-arm64' (MLX), 'nvidia' (NF4 on CUDA), or 'cpu' (bfloat16 torch on the CPU) for everything else."""
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return "mac-arm64"
    if system in ("Windows", "Linux"):
        if nvidia is None:
            nvidia = has_nvidia()
        if nvidia:
            return "nvidia"
    if system in ("Windows", "Linux", "Darwin"):
        return "cpu"
    raise LayersError(f"{system}/{machine} is not supported. The layer network serves on Apple silicon, NVIDIA GPUs, "
                      "or the CPU of a Windows, Linux or Intel Mac machine.")


def requirements(kind: str) -> tuple[list[str], list[str]]:
    """(plain requirements, extra pip arguments for the torch install) for a platform kind."""
    if kind == "mac-arm64":
        return list(MAC_REQUIREMENTS), []
    if kind == "nvidia":
        return list(NVIDIA_REQUIREMENTS), [CUDA_TORCH_REQUIREMENT, "--index-url", CUDA_TORCH_INDEX]
    if kind == "cpu":
        # macOS has no separate CPU wheel index; the default wheels are CPU builds there
        index = ["--index-url", CPU_TORCH_INDEX] if platform.system() in ("Windows", "Linux") else []
        return list(CPU_REQUIREMENTS), [CPU_TORCH_REQUIREMENT, *index]
    raise LayersError(f"unknown platform kind {kind!r}")


def venv_python(venv: Path = RUNTIME_VENV) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _source_prefix(source: Path | None) -> str:
    # -m and -c put the working directory before PYTHONPATH. Select the downloaded
    # code explicitly while keeping cwd unchanged for relative manifest paths.
    return f"import sys; sys.path.insert(0, {str(source.resolve())!r}); " if source is not None else ""


def can_import(python, source: Path | None = None, run=subprocess.run, kind: str | None = None) -> bool:
    """Can `python` import commonnet and its dependencies (with `source` on the path, if given)?

    With `kind`, the platform's backend modules must import as well."""
    env = dict(os.environ)
    if source is not None:
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(source), env.get("PYTHONPATH", "")) if p)
    try:
        check = _source_prefix(source) + IMPORT_CHECK + ("; " + BACKEND_IMPORTS[kind] if BACKEND_IMPORTS.get(kind) else "")
        out = run([str(python), "-c", check], capture_output=True, env=env, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0


def _check_package(root: Path) -> Path:
    if not (root / "commonnet" / "__init__.py").is_file() or not (root / "commonnet" / "node.py").is_file():
        raise LayersError(f"{root} does not hold a commonnet package with node.py; "
                          "set COMMONNET_SRC to a checkout of the repository")
    return root


def ensure_source(env=None, ref: str | None = None, dest: Path = SOURCE_DIR, urlopen=urllib.request.urlopen,
                  dry_run: bool = False, log=print) -> Path:
    """The directory that holds the `commonnet` package: COMMONNET_SRC, or a tarball of the repository."""
    env = os.environ if env is None else env
    given = env.get("COMMONNET_SRC")
    if given:
        return _check_package(Path(given).expanduser())
    ref = ref or env.get("COMMONNET_REF") or DEFAULT_REF
    marker = dest / ".ref"
    if (dest / "commonnet" / "node.py").is_file() and marker.is_file() and marker.read_text().strip() == ref \
            and ref not in ("main", "master"):
        return dest  # a pinned ref never changes; a branch is re-fetched
    url = TARBALL_URL.format(ref=ref)
    if dry_run:
        log(f"would download {url} into {dest}")
        return dest
    log(f"downloading the Common Network code ({ref})...")
    try:
        with urlopen(url, timeout=60) as resp:
            blob = resp.read()
    except OSError as e:
        raise LayersError(f"could not download {url} ({e}). Check your connection, or set COMMONNET_SRC to a "
                          "checkout of the repository.") from None
    staging = dest.with_name(dest.name + ".new")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            prefix = None
            for member in tar.getmembers():
                parts = Path(member.name).parts
                if len(parts) < 2 or parts[1] not in ("commonnet", "join"):  # join/ holds the tunnel opener
                    continue
                if member.name.startswith(("/", "..")) or ".." in parts or not (member.isfile() or member.isdir()):
                    continue  # refuse absolute paths, traversal, links and devices
                prefix = parts[0]
                target = staging.joinpath(*parts[1:])
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(tar.extractfile(member).read())
            if prefix is None:
                raise LayersError("the downloaded archive has no commonnet/ package")
        if not (staging / "join" / "join.py").is_file():
            raise LayersError("the downloaded archive has no join/join.py (the tunnel opener)")
        _check_package(staging)
        (staging / ".ref").write_text(ref)
    except (tarfile.TarError, OSError) as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise LayersError(f"the downloaded archive could not be unpacked ({e})") from None
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(staging, dest)
    return dest


def ensure_runtime(kind: str, *, source: Path, current_python=None, venv: Path = RUNTIME_VENV,
                   run=subprocess.run, dry_run: bool = False, log=print) -> str:
    """A Python that can run commonnet.node: this interpreter if it can, else the runtime venv (made if needed)."""
    current_python = current_python or sys.executable
    if can_import(current_python, source, run, kind):
        return str(current_python)
    python = venv_python(venv)
    if python.is_file() and can_import(python, source, run, kind):
        return str(python)
    plain, torch_args = requirements(kind)
    steps = []
    if not python.is_file():
        steps.append(([str(current_python), "-m", "venv", str(venv)], "creating a Python environment"))
    steps.append(([str(python), "-m", "pip", "install", "--upgrade", "pip"], "preparing pip"))
    if torch_args:
        steps.append(([str(python), "-m", "pip", "install", *torch_args], "installing torch (large)"))
    steps.append(([str(python), "-m", "pip", "install", *plain], "installing the model runtime (large, once)"))
    for command, what in steps:
        log(f"{what}...")
        if dry_run:
            log("  would run: " + " ".join(command))
            continue
        result = run(command)
        if getattr(result, "returncode", 0) != 0:
            raise LayersError(f"{what} failed (exit {result.returncode}). Fix the error above and run the same "
                              "command again; it resumes where it stopped.")
    if not dry_run and not can_import(python, source, run, kind):
        raise LayersError("the runtime installed but commonnet still cannot be imported; "
                          f"try deleting {venv} and joining again")
    return str(python)


def build_command(python: str, network: str | None, extra: list[str] | None = None,
                  *, source: Path | None = None) -> list[str]:
    """The command that runs the node. `network` may be None to use COMMONNET_NETWORK or the default seeds."""
    command = [str(python), "-m", "commonnet.node"]
    if source is not None:
        command = [str(python), "-c", _source_prefix(source)
                   + "import runpy; runpy.run_module('commonnet.node', run_name='__main__', alter_sys=True)"]
    if network:
        command += ["--network", network]
    return command + list(extra or [])


def run(network: str | None, extra: list[str] | None = None, *, dry_run: bool = False, env=None,
        runner=subprocess.Popen, run_cmd=subprocess.run, log=print, nvidia: bool | None = None) -> int:
    """Set everything up and run the node in the foreground. Returns its exit code."""
    env = dict(os.environ if env is None else env)
    kind = platform_kind(nvidia=nvidia)
    source = ensure_source(env, dry_run=dry_run, log=log)
    python = ensure_runtime(kind, source=source, run=run_cmd, dry_run=dry_run, log=log)
    command = build_command(python, network, extra, source=source)
    if dry_run:
        log("would run: " + " ".join(command))
        return 0
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(source), env.get("PYTHONPATH", "")) if p)
    env["PYTHONUNBUFFERED"] = "1"  # progress must show as it happens, not when the node exits
    child = runner(command, env=env)

    def forward(signum, frame):
        try:
            child.send_signal(signal.SIGTERM if os.name != "nt" else signal.SIGINT)
        except OSError:
            pass

    previous = {}
    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                previous[sig] = signal.signal(sig, forward)
            except ValueError:  # not the main thread
                pass
    try:
        return child.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
