"""Node-count milestones select a shared model network, never a private group."""
from __future__ import annotations

from dataclasses import replace


def add_node_count_flags(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--nodes", type=int, choices=range(1, 8), metavar="1-7",
                       help="planned node count: 1-2 Qwen3-8B, 3-6 Qwen3.5-35B-A3B, 7 Qwen3.5-122B-A10B")
    for count in range(1, 8):
        group.add_argument(f"--{count}", dest="nodes", action="store_const", const=count,
                           help=f"same as --nodes {count}")


def model_for_nodes(count: int) -> str:
    if type(count) is not int or not 1 <= count <= 7:
        raise ValueError("node count must be between 1 and 7")
    return "qwen3-8b" if count <= 2 else "qwen3.5-35b-a3b" if count <= 6 else "qwen3.5-122b-a10b"


def placement_manifest(manifest, count):
    """Limit each planned contributor's slice; extra 35B peers provide replicas.

    Stable network IDs let all peers serving the same model discover each other.
    The hint only changes local placement, never model weights or the wire protocol.
    """
    model_for_nodes(count)  # validate
    limit = 36 if count == 1 else 18 if count == 2 else 16 if count <= 6 else 7
    return replace(manifest, model={**manifest.model, "max_window_layers": limit, "planned_nodes": count})
