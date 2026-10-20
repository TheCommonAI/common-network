#!/usr/bin/env python3
"""Write a distributed model group's network.json from its enrolled members.

The coordinator's distributed config lists the RPC endpoints the target is split
across, one device per endpoint. Writing that list by hand is what this replaces:

- rpc_servers is this coordinator's own RPC worker (127.0.0.1:50051 by default),
  then the loopback port of each member enrolled in the group, in port order.
- devices names them RPC0, RPC1, ... in the same order. llama-server numbers its
  --rpc peers that way, and runtime.preflight checks that mapping.
- tensor_split gives each device its weight. The trial default is [0.65, 0.35]:
  the coordinator's Mac (16 GB unified memory, Metal working set about 12.7 GB)
  takes about 10.4 GB of the 16 GB target, and Patrick's 8 GB NVIDIA card takes
  about 5.6 GB, which leaves room for KV cache and compute buffers.

The trial is exactly two devices. The network is GPU-only. Each endpoint's backend (MTL, CUDA or Vulkan) is declared
on the command line, and any CPU device, any endpoint without a GPU, or a drafter
on the CPU is refused. runtime preflight then checks that discovery reports no
CPU backend for any configured device.

Everything else comes from the template, join/distributed.trial-32b.json. The
result is checked with runtime.DistributedConfig before it replaces an existing
network.json, so a refused or invalid group never leaves a broken config behind.

This only reads group.json. It does not change group state, open a socket, or
start a process. Stdlib only, like the rest of the CLI.

Usage:
  python3 join/group_config.py --group NAME --target TARGET.gguf \\
      --drafter DRAFT.gguf --binary /path/to/llama-server [--draft-tokens 4]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import group
import runtime

TEMPLATE = Path(__file__).resolve().with_name("distributed.trial-32b.json")
DEFAULT_LOCAL_WORKER = "127.0.0.1:50051"
DEFAULT_TENSOR_SPLIT = (0.65, 0.35)
TRIAL_ENDPOINTS = 2


class ConfigError(Exception):
    pass


# --- reading the group ------------------------------------------------------

def member_endpoints(groups_dir: Path, group_name: str) -> list[str]:
    """Loopback RPC endpoints of the members enrolled in a group, in port order."""
    try:
        state = group.GroupState.load(group._group_dir(groups_dir, group_name) / "group.json")
    except group.GroupError as e:
        raise ConfigError(str(e)) from None
    ports = sorted(entry["rpc_port"] for entry in state.data["members"].values())
    return [f"127.0.0.1:{port}" for port in ports]


def load_template(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise ConfigError(f"cannot read template {path}: {e}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"template {path} must be a JSON object")
    return data


# --- building and writing the config ----------------------------------------

GPU_BACKEND = re.compile(r"(MTL|CUDA|Vulkan)[0-9]+")


def check_gpu_only(backends: list[str], endpoints: list[str], draft_device: str) -> None:
    """Refuse any CPU device, and any endpoint whose backend is not a GPU.

    backends[i] names the device behind endpoints[i]. A llama.cpp RPC endpoint
    cannot say what it runs, so the operator declares it and runtime preflight
    checks the discovery output against it. Refused: a device whose name starts
    with CPU, a backend that is not MTL, CUDA or Vulkan, and a drafter on the CPU.
    """
    if len(backends) != len(endpoints):
        raise ConfigError(f"each endpoint needs a declared GPU backend: got {len(backends)} "
                          f"for {len(endpoints)} endpoint(s)")
    for endpoint, backend in zip(endpoints, backends):
        if backend.startswith("CPU"):
            raise ConfigError(f"refusing device {backend} behind {endpoint}: this network is "
                              "GPU-only, and a CPU device would run part of the model on the CPU")
        if not GPU_BACKEND.fullmatch(backend):
            raise ConfigError(f"endpoint {endpoint} has no GPU device (backend {backend!r}); "
                              "every endpoint must be MTL, CUDA or Vulkan, because the network is GPU-only")
    if not GPU_BACKEND.fullmatch(draft_device):
        raise ConfigError(f"draft_device {draft_device!r} is not a GPU; the drafter must run on a "
                          "GPU (MTL, CUDA or Vulkan), because 'none' drafts on the CPU")


def build_config(template: dict, *, endpoints: list[str], backends: list[str], binary: str,
                 target: str, drafter: str, draft_tokens: int | None = None,
                 split: tuple[float, ...] = DEFAULT_TENSOR_SPLIT) -> dict:
    """The network.json contents for these RPC endpoints. Pure: touches no files.

    The trial is exactly two devices: the coordinator's Mac and Patrick's machine.
    Fewer than two is refused because the target is split across the devices, and
    runtime.DistributedConfig rejects a single device for the same reason.
    """
    check_gpu_only(backends, endpoints, template.get("draft_device", "none"))
    if len(endpoints) < TRIAL_ENDPOINTS:
        raise ConfigError(
            f"refusing to write a config with {len(endpoints)} RPC endpoint(s); it needs at least "
            f"{TRIAL_ENDPOINTS}. The target is split across the devices, so one device is not a "
            "distributed run. Enrol a member first: `group.py issue`, then the member runs "
            "`join.py --group-token`, and `group.py relay` is running so its port is live.")
    if len(endpoints) != TRIAL_ENDPOINTS:
        raise ConfigError(f"this trial config is for exactly {TRIAL_ENDPOINTS} endpoints "
                          f"(the coordinator and one member); got {len(endpoints)}")
    if len(set(endpoints)) != len(endpoints):
        raise ConfigError(f"duplicate RPC endpoint in {', '.join(endpoints)}")
    if len(split) != len(endpoints):
        raise ConfigError(f"tensor_split needs one weight per device: {len(split)} weight(s) "
                          f"for {len(endpoints)} device(s)")
    for weight in split:
        if (type(weight) not in (int, float) or not 0 < weight < float("inf")):
            raise ConfigError(f"tensor_split weights must be positive finite numbers; got {weight!r}")
    config = dict(template)
    config.update({
        "binary": binary,
        "target_model": target,
        "draft_model": drafter,
        "rpc_servers": list(endpoints),
        "devices": [f"RPC{index}" for index in range(len(endpoints))],
        "tensor_split": list(split),
    })
    if draft_tokens is not None:
        config["draft_tokens"] = draft_tokens
    return config


def write_config(config: dict, out: Path) -> None:
    """Write network.json atomically, after runtime.DistributedConfig accepts it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(config, indent=2) + "\n")
    try:
        runtime.DistributedConfig(tmp)
    except runtime.FetchFailed as e:
        tmp.unlink(missing_ok=True)
        raise ConfigError(str(e)) from None
    os.replace(tmp, out)


# --- command line -----------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="group_config.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", required=True, help="group name created with group.py init")
    parser.add_argument("--target", required=True, help="path to the target GGUF on this coordinator")
    parser.add_argument("--drafter", required=True, help="path to the draft GGUF; it runs on this coordinator")
    parser.add_argument("--binary", required=True, help="RPC-enabled llama-server on this coordinator")
    parser.add_argument("--local-worker", default=DEFAULT_LOCAL_WORKER,
                        help="this coordinator's own RPC worker (default: %(default)s)")
    parser.add_argument("--local-backend", default="MTL0",
                        help="GPU device behind the local worker: MTL0, CUDA0 or Vulkan0 (default: %(default)s)")
    parser.add_argument("--member-backend", default=None,
                        help="GPU device behind each member's RPC server: CUDA0 or Vulkan0, whichever "
                             "Patrick's build reports. Required when members are enrolled.")
    parser.add_argument("--draft-tokens", type=int, default=None,
                        help="draft length; the template value (4) if omitted")
    parser.add_argument("--tensor-split", default=",".join(str(w) for w in DEFAULT_TENSOR_SPLIT),
                        help="one weight per device, comma separated, in RPC0,RPC1 order (default: %(default)s)")
    parser.add_argument("--template", default=str(TEMPLATE), help="base config (default: %(default)s)")
    parser.add_argument("--out", help="output path (default: the group's network.json)")
    parser.add_argument("--groups-dir", default=str(group.GROUPS_DIR), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        groups_dir = Path(args.groups_dir).expanduser()
        out = (Path(args.out).expanduser() if args.out
               else group._group_dir(groups_dir, args.group) / "network.json")
        group._split_hostport(args.local_worker)
        members = member_endpoints(groups_dir, args.group)
        if members and not args.member_backend:
            raise ConfigError("name the GPU backend of the member RPC servers with --member-backend, "
                              "CUDA0 or Vulkan0, whichever Patrick's build reports")
        endpoints = [args.local_worker, *members]
        backends = [args.local_backend, *([args.member_backend] * len(members))]
        try:
            split = tuple(float(w) for w in args.tensor_split.split(","))
        except ValueError:
            raise ConfigError(f"--tensor-split must be comma-separated numbers, got {args.tensor_split!r}") from None
        config = build_config(
            load_template(Path(args.template).expanduser()),
            endpoints=endpoints,
            backends=backends,
            binary=str(Path(args.binary).expanduser().resolve()),
            target=str(Path(args.target).expanduser().resolve()),
            drafter=str(Path(args.drafter).expanduser().resolve()),
            draft_tokens=args.draft_tokens,
            split=split,
        )
        write_config(config, out)
    except (ConfigError, group.GroupError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    print(f"rpc_servers: {json.dumps(config['rpc_servers'])}")
    print("restart the coordinator to load it: the split is fixed when the runtime starts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
