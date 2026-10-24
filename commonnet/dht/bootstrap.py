"""Where a node starts from: commonnet/networks/bootstrap.json plus COMMONNET_DHT_BOOTSTRAP."""
from __future__ import annotations

import json
import os
from pathlib import Path

ENV_VAR = "COMMONNET_DHT_BOOTSTRAP"
DEFAULT_FILE = Path(__file__).resolve().parent.parent / "networks" / "bootstrap.json"


def parse_addr(text, allow_zero_port: bool = False) -> tuple[str, int]:
    """'host:port' (or a [host, port] pair) to (host, port). Raises ValueError if it is not one."""
    if isinstance(text, (list, tuple)) and len(text) == 2:
        host, port = text
    elif isinstance(text, str) and ":" in text:
        host, _, port = text.strip().rpartition(":")
    else:
        raise ValueError(f"not a host:port address: {text!r}")
    host = str(host).strip("[]")
    try:
        port = int(port)
    except (TypeError, ValueError):
        raise ValueError(f"bad port in {text!r}") from None
    if not host or not (0 if allow_zero_port else 1) <= port < 65536:
        raise ValueError(f"not a host:port address: {text!r}")
    return host, port


def load_bootstrap(path=None, env=None) -> list[tuple[str, int]]:
    """Addresses from the file (a JSON list, or {"bootstrap": [...]}) then the env var (comma separated).

    Bad entries are skipped, a missing or unreadable file is an empty list: a node with nothing to
    contact simply becomes the first node of its network. Duplicates are dropped, order is kept.
    """
    env = os.environ if env is None else env
    raw: list = []
    try:
        data = json.loads(Path(path or DEFAULT_FILE).read_text())
        raw.extend(data.get("bootstrap", []) if isinstance(data, dict) else data)
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    raw.extend(p for p in env.get(ENV_VAR, "").split(",") if p.strip())
    out: list[tuple[str, int]] = []
    for item in raw:
        try:
            addr = parse_addr(item)
        except ValueError:
            continue
        if addr not in out:
            out.append(addr)
    return out
