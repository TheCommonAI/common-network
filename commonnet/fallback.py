"""Choose a locally supportable Bonsai network when the requested model has no live route."""
from __future__ import annotations

import re
from pathlib import Path

_DEFAULT_GATEWAY = object()


class FallbackUnavailable(RuntimeError):
    """No locally fitting Bonsai network has enough live peers to serve a route."""


def local_hardware():
    from commonnet.node import detect_hardware
    return detect_hardware()


def _bonsai_parameter_count(manifest) -> float:
    """Read the nominal parameter count from catalogue model metadata (for example, 'Bonsai 2 9B')."""
    name = str(manifest.model.get("name", ""))
    counts = re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)\s*B\b", name, flags=re.IGNORECASE)
    return float(counts[-1]) if counts else 0.0


def available_bonsai_networks(network_type, gateway=_DEFAULT_GATEWAY):
    """Load Bonsai manifests and discover peers in each model's independent network namespace."""
    from commonnet.mesh import directory
    from commonnet.node import NETWORK_MODELS
    from commonnet.netclient import NetworkUnreachable

    networks_dir = Path(__file__).resolve().parent / "networks"
    if gateway is _DEFAULT_GATEWAY:
        gateway = directory.gateway_url(default=directory.DEFAULT_GATEWAY)
    candidates = []
    for name, filename in NETWORK_MODELS.items():
        if not name.startswith("bonsai-"):
            continue
        candidate = network_type.from_spec(str(networks_dir / filename), gateway=gateway)
        try:
            candidate.refresh(force=True)
        except NetworkUnreachable:
            continue
        candidates.append(candidate)
    return candidates


def select_network_with_bonsai_fallback(primary, candidates, hardware, coverage_check=None):
    """Keep a covered network; otherwise choose the largest locally fitting, covered Bonsai network."""
    from commonnet.mesh.assign import AssignmentError
    from commonnet.netclient import NetworkUnreachable, NoLayersServed
    from commonnet.node import NodeError

    check = coverage_check or (lambda network: network.plan())
    try:
        primary.refresh(force=True)
    except NetworkUnreachable:
        pass
    try:
        check(primary)
        return primary, False
    except (AssignmentError, NetworkUnreachable, NoLayersServed) as primary_error:
        try:
            hardware = hardware() if callable(hardware) else hardware
        except NodeError as e:
            raise FallbackUnavailable(
                f"{primary_error}; cannot detect local capacity for a Bonsai fallback ({e})") from None
        candidates = candidates() if callable(candidates) else candidates
        fitting = []
        for candidate in candidates:
            manifest = candidate.manifest
            if manifest.id == primary.manifest.id:
                continue
            if not str(manifest.model.get("name", "")).lower().startswith("ternary bonsai"):
                continue
            if not manifest.repo(hardware.backend) or not manifest.windows_fitting(hardware.gib, hardware.backend):
                continue
            live = candidate.table.live()
            if any(record.get("network") != manifest.id for record in live):
                continue
            fitting.append(candidate)

        fitting.sort(key=lambda candidate: (-_bonsai_parameter_count(candidate.manifest), candidate.manifest.id))
        for candidate in fitting:
            try:
                check(candidate)
                return candidate, True
            except (AssignmentError, NetworkUnreachable, NoLayersServed):
                continue

        if not fitting:
            reason = (f"no Bonsai model/window fits {hardware.name} "
                      f"({hardware.gib:.1f} GiB on {hardware.backend})")
        else:
            reason = "no locally fitting Bonsai network has enough live peers to cover its model"
        raise FallbackUnavailable(f"{primary_error}; {reason}") from None
