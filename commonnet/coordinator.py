"""Coordinator: read member endpoints, check coverage, and generate through them.

    python -m commonnet.coordinator --endpoints members.json --checkpoint DIR --prompt TEXT

--endpoints is a JSON file holding a list of {"url", "layers": [start, end], "token_file"},
or a comma-separated list of URL@START:END (token from COMMONNET_BLOCK_TOKEN).

    python -m commonnet.coordinator --network URL|FILE --prompt TEXT

    python -m commonnet.coordinator --network URL|FILE --plan

--network uses the live layer network: the route, replicas and failover come from the peers it lists.
If that network has no complete route, the coordinator tries the largest Bonsai network with live coverage
and a window that fits the local machine. The client then holds its embedding and head as mlx4 and
exchanges float32 hidden states with the peers.
Prints the generated text only. It never prints the bearer token or token ids.
--plan prints the layer assignment the live peers add up to (commonnet.mesh.assign) as JSON and generates nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

from commonnet.client import BlockEndpoint, Client
from commonnet.net import auth
from commonnet.net.transport import HttpTransport


class EndpointError(ValueError):
    pass


def load_endpoints(spec: str) -> list[dict]:
    if os.path.exists(spec):
        with open(spec) as f:
            raw = json.load(f)
    else:
        raw = []
        for part in spec.split(","):
            try:
                url, rng = part.strip().rsplit("@", 1)
                s, e = rng.split(":")
                raw.append({"url": url, "layers": [int(s), int(e)]})
            except ValueError:
                raise EndpointError(f"not a file, and {part!r} is not URL@START:END") from None
    if not isinstance(raw, list) or not raw:
        raise EndpointError("endpoints must be a non-empty JSON list")
    for i, m in enumerate(raw):
        if not isinstance(m, dict) or not isinstance(m.get("url"), str) or not m["url"] \
                or not isinstance(m.get("layers"), list) or len(m["layers"]) != 2 \
                or not all(type(x) is int for x in m["layers"]):
            raise EndpointError(f"member {i}: needs a url and layers [start, end]")
    return raw


def build_route(members: list[dict], num_layers: int):
    """Sort by start, refuse gaps, overlaps and a wrong total. Returns (endpoints, urls, tokens)."""
    ordered = sorted(members, key=lambda m: m["layers"][0])
    route = [BlockEndpoint(f"member{i}", m["layers"][0], m["layers"][1]) for i, m in enumerate(ordered)]
    try:
        Client.check_route(route, num_layers)
    except ValueError as e:
        raise EndpointError(f"layer coverage is wrong: {e}") from None
    urls, tokens = {}, {}
    for ep, m in zip(route, ordered):
        urls[ep.name] = m["url"]
        tokens[ep.name] = auth.load_token(m.get("token_file"))
        if tokens[ep.name] is None:
            raise EndpointError(f"no token for {ep.name}: give token_file or set {auth.TOKEN_ENV}")
    return route, urls, tokens


def generate(members, checkpoint, prompt_ids, max_new_tokens, fmt="fp32", transport_kwargs=None,
             hidden_dtype=None):
    from commonnet.client4 import Client4
    from commonnet.weights import load_text_config
    n = load_text_config(checkpoint).num_hidden_layers
    route, urls, tokens = build_route(members, n)
    transport = HttpTransport(urls, tokens=tokens, **(transport_kwargs or {}))
    extra = {} if hidden_dtype is None else {"hidden_dtype": hidden_dtype}
    client = Client4(checkpoint, transport, fmt=fmt, **extra)
    return client.generate(route, prompt_ids, max_new_tokens)


def generate_network(network, checkpoint, prompt_ids, max_new_tokens, fmt="mlx4", hidden_dtype=None, **sampling):
    """Generate over the live layer network: route, replicas and failover come from the peers the network lists.

    Peers run the MLX block in bfloat16 and exchange float32 hidden states (float16 overflows to NaN on some Qwen prompts),
    so the client uses float32 unless told otherwise.
    """
    import torch
    from commonnet.client4 import Client4
    endpoints, transport = network.plan()
    client = Client4(checkpoint, transport, fmt=fmt, hidden_dtype=hidden_dtype or torch.float32)
    return client.generate(endpoints, prompt_ids, max_new_tokens, **sampling)


def print_plan(spec: str) -> int:
    """The assignment the live network at `spec` adds up to: every peer's window, the chain, and the idle peers."""
    from commonnet.mesh import assign
    from commonnet.netclient import Network, NetworkUnreachable
    try:
        network = Network.from_spec(spec)
        original = network
        network, fallback = select_network_with_bonsai_fallback(
            network,
            lambda: _available_bonsai_networks(Network),
            _local_hardware,
            coverage_check=lambda candidate: assign.plan(candidate.manifest, candidate.table.live()),
        )
        result = assign.plan(network.manifest, network.table.live())
    except (NetworkUnreachable, assign.AssignmentError, FallbackUnavailable) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    payload = result.to_dict()
    if fallback:
        payload.update({
            "selected_network": network.manifest.id,
            "selected_model": network.manifest.model.get("name", network.manifest.id),
            "fallback_from": original.manifest.id,
        })
    print(json.dumps(payload, indent=2))
    return 0


class FallbackUnavailable(RuntimeError):
    """No locally fitting Bonsai network has enough live peers to serve a route."""


def _local_hardware():
    from commonnet.node import detect_hardware
    return detect_hardware()


def _bonsai_parameter_count(manifest) -> float:
    """Read the nominal parameter count from the catalogue model name (for example, 'Bonsai 2 9B')."""
    name = str(manifest.model.get("name", ""))
    counts = re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)\s*B\b", name, flags=re.IGNORECASE)
    return float(counts[-1]) if counts else 0.0


def _available_bonsai_networks(network_type):
    """Load each catalogue Bonsai network and learn peers from its own model-id namespace."""
    from pathlib import Path
    from commonnet.mesh import directory
    from commonnet.node import NETWORK_MODELS
    from commonnet.netclient import NetworkUnreachable

    networks_dir = Path(__file__).resolve().parent / "networks"
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

    check = coverage_check or (lambda network: network.plan())
    try:
        primary.refresh(force=True)
    except NetworkUnreachable:
        pass
    try:
        check(primary)
        return primary, False
    except (AssignmentError, NetworkUnreachable, NoLayersServed) as primary_error:
        from commonnet.node import NodeError
        try:
            hardware = hardware() if callable(hardware) else hardware
        except NodeError as e:
            raise FallbackUnavailable(f"{primary_error}; cannot detect local capacity for a Bonsai fallback ({e})") from None
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="commonnet.coordinator", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoints", help="a members.json file, or URL@START:END,... (a fixed route)")
    p.add_argument("--network", metavar="URL|FILE",
                   help="use the live layer network instead of a fixed route: a peer URL, or a manifest file that lists seeds; "
                        "if it has no complete route, try the largest locally fitting Bonsai network")
    p.add_argument("--checkpoint", help="local checkpoint directory")
    p.add_argument("--repo", help="Hugging Face repo id; downloads only embed, final norm and lm_head")
    p.add_argument("--revision", default="main")
    p.add_argument("--prompt")
    p.add_argument("--plan", action="store_true",
                   help="with --network: print the layer assignment the live peers add up to (JSON) and exit")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--client-format", choices=["fp32", "fp16", "mlx4"], default=None,
                   help="default: fp16 with --endpoints, mlx4 with --network")
    p.add_argument("--hidden-dtype", choices=["float16", "float32"], default=None,
                   help="what crosses between members (mlx4 client); float32 with COMMONNET_DTYPE=bfloat16 "
                        "on the members matches a bfloat16 model exactly. Default: float16 with --endpoints, "
                        "float32 with --network")
    args = p.parse_args(argv)
    if bool(args.endpoints) == bool(args.network):
        print("error: give exactly one of --endpoints and --network", file=sys.stderr)
        return 2
    if args.plan:
        return print_plan(args.network)
    if args.prompt is None:
        print("error: --prompt is required", file=sys.stderr)
        return 2
    if args.client_format is None:
        args.client_format = "mlx4" if args.network else "fp16"
    if args.hidden_dtype is None:
        args.hidden_dtype = "float32" if args.network else "float16"
    try:
        network = None
        if args.network:
            from commonnet.netclient import Network
            network = Network.from_spec(args.network)
            if not args.checkpoint and not args.repo:
                original = network
                network, fallback = select_network_with_bonsai_fallback(
                    network, lambda: _available_bonsai_networks(Network), _local_hardware)
                if fallback:
                    print(f"using {network.manifest.model.get('name', network.manifest.id)} "
                          f"({network.manifest.id}) because {original.manifest.id} lacks peer coverage",
                          file=sys.stderr)
            if not args.checkpoint and not args.repo:
                args.repo = network.manifest.client_repo
        if not args.checkpoint:
            if not args.repo:
                print("error: give --checkpoint DIR or --repo REPO", file=sys.stderr)
                return 2
            from commonnet import hub
            args.checkpoint = hub.Hub(args.repo, args.revision).fetch(client=True)
        from transformers import AutoTokenizer
        from commonnet.net.client import BlockError
        tok = AutoTokenizer.from_pretrained(args.checkpoint)
        ids = tok(args.prompt)["input_ids"]
        import torch
        if network is not None:
            out = generate_network(network, args.checkpoint, ids, args.max_new_tokens, args.client_format,
                                   hidden_dtype=getattr(torch, args.hidden_dtype))
        else:
            members = load_endpoints(args.endpoints)
            out = generate(members, args.checkpoint, ids, args.max_new_tokens, args.client_format,
                           hidden_dtype=getattr(torch, args.hidden_dtype))
    except RuntimeError as e:
        kind = type(e).__name__
        if kind in ("NetworkUnreachable", "NoLayersServed", "FallbackUnavailable"):
            print(f"error: {e}", file=sys.stderr)
            return 1
        if kind != "HubError":
            raise
        print(f"error: download failed: {e}", file=sys.stderr)
        return 1
    except (EndpointError, auth.TokenError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    except BlockError as e:
        print(f"error: member call failed: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        if type(e).__name__ != "ManifestError":
            raise
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(tok.decode(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
