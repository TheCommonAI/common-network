"""Split planner: which decoder layers each member serves, decided by memory.

The route is two members. The first member (the Mac, which also holds the
client's embedding and head) serves layers [0, a). The second (the PC) serves
[a, L). The planner tries every split point a and keeps the feasible ones. Each
member's bytes are its fixed weights, plus per layer its weights and its KV
cache for the planned context. A split is feasible when both members stay under
their budgets. Among feasible splits it keeps the one with the most headroom on
the tighter member. If no split is feasible it raises PlanError and says how
many layers each member could hold.

Budgets are in GiB (2**30 bytes). The 32B constants below were read from the
safetensors headers of two checkpoints. They were not estimated:
  mlx-community/Qwen3-32B-4bit   rev bcaaf7f5  (18.43 GB, 4 shards)
  unsloth/Qwen3-32B-bnb-4bit     rev 7f721e74  (19.21 GB, 4 shards)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

GiB = 1 << 30
SUPPORTED_DEVICES = ("metal", "cuda")

# Qwen3-32B architecture (from config.json of both checkpoints).
QWEN3_32B_LAYERS = 64
QWEN3_32B_KV_HEADS = 8
QWEN3_32B_HEAD_DIM = 128

# Weight bytes, from the safetensors headers.
MLX4_LAYER_BYTES = 274_289_152  # one decoder layer, 4-bit affine MLX (group 64)
NF4_LAYER_BYTES = 251_561_339  # one decoder layer, NF4 (bitsandbytes), largest layer
MLX4_CLIENT_BYTES = 875_161_600  # embed + lm_head at 4 bit + final norm
FP16_CLIENT_BYTES = 3_111_659_520  # embed + lm_head at 16 bit + final norm


class PlanError(ValueError):
    """The members cannot host the model under their budgets, or a member is refused."""


def kv_bytes_per_token_per_layer(kv_heads: int, head_dim: int, dtype_bytes: int = 2) -> int:
    """Keys and values for one token in one layer, stored in fp16 by default."""
    return 2 * kv_heads * head_dim * dtype_bytes


@dataclass(frozen=True)
class Member:
    """One machine's memory. budget_bytes covers its fixed weights, its layers and their KV cache."""

    name: str
    device: str  # "metal" (Mac) or "cuda" (PC). Anything else, including cpu, is refused.
    budget_bytes: int
    layer_bytes: int  # weight bytes of one decoder layer in this member's format
    fixed_bytes: int = 0  # weights that live here outside its layers (the client, on the Mac)

    def __post_init__(self):
        if self.device not in SUPPORTED_DEVICES:
            raise PlanError(
                f"member {self.name!r} is on device {self.device!r}; only {SUPPORTED_DEVICES} "
                "are accepted (a CPU-only member is refused)"
            )
        if self.budget_bytes <= 0 or self.layer_bytes <= 0 or self.fixed_bytes < 0:
            raise PlanError(f"member {self.name!r} has a non-positive budget or layer size")


@dataclass(frozen=True)
class Assignment:
    name: str
    start: int
    end: int
    used_bytes: int
    budget_bytes: int

    @property
    def headroom_bytes(self) -> int:
        return self.budget_bytes - self.used_bytes


@dataclass(frozen=True)
class SplitPlan:
    num_layers: int
    context_tokens: int
    assignments: tuple[Assignment, ...]

    def route(self) -> list[tuple[str, int, int]]:
        return [(a.name, a.start, a.end) for a in self.assignments]


def member_bytes(member: Member, layers: int, kv_per_token_per_layer: int, context_tokens: int) -> int:
    """Bytes a member needs to serve `layers` layers at this context length."""
    return member.fixed_bytes + layers * (member.layer_bytes + kv_per_token_per_layer * context_tokens)


def max_layers(member: Member, kv_per_token_per_layer: int, context_tokens: int) -> int:
    """How many layers the member could serve alone, ignoring the other member."""
    per_layer = member.layer_bytes + kv_per_token_per_layer * context_tokens
    spare = member.budget_bytes - member.fixed_bytes
    return max(0, spare // per_layer)


def plan_split(
    num_layers: int,
    first: Member,
    second: Member,
    *,
    kv_bytes_per_token_per_layer: int,
    context_tokens: int,
) -> SplitPlan:
    """Choose the split point a. `first` serves [0, a) and `second` serves [a, num_layers).

    Raises PlanError if any member is refused, if there is no feasible split, or if the
    context length is not positive. The error message gives the layer capacities.
    """
    if num_layers < 2:
        raise PlanError("need at least two layers to split")
    if context_tokens <= 0:
        raise PlanError("context_tokens must be positive")
    if kv_bytes_per_token_per_layer <= 0:
        raise PlanError("kv_bytes_per_token_per_layer must be positive")

    kv, ctx = kv_bytes_per_token_per_layer, context_tokens
    best: tuple[int, int, int, int] | None = None  # (headroom, a, used_first, used_second)
    for a in range(1, num_layers):
        used1 = member_bytes(first, a, kv, ctx)
        used2 = member_bytes(second, num_layers - a, kv, ctx)
        if used1 > first.budget_bytes or used2 > second.budget_bytes:
            continue
        headroom = min(first.budget_bytes - used1, second.budget_bytes - used2)
        if best is None or headroom > best[0]:
            best = (headroom, a, used1, used2)

    if best is None:
        cap1 = max_layers(first, kv, ctx)
        cap2 = max_layers(second, kv, ctx)
        raise PlanError(
            f"no split of {num_layers} layers fits at context {ctx}: {first.name} can hold at most "
            f"{cap1} layers and {second.name} at most {cap2} ({cap1 + cap2} in total, "
            f"short by {max(0, num_layers - cap1 - cap2)} layers)"
        )

    _, a, used1, used2 = best
    return SplitPlan(
        num_layers=num_layers,
        context_tokens=ctx,
        assignments=(
            Assignment(first.name, 0, a, used1, first.budget_bytes),
            Assignment(second.name, a, num_layers, used2, second.budget_bytes),
        ),
    )


def plan_qwen3_32b(
    mac_budget_gib: float = 11.0,
    pc_budget_gib: float = 6.5,
    *,
    context_tokens: int = 2048,
    client: str = "mlx4",
    mac_reserve_gib: float = 0.0,
    pc_reserve_gib: float = 0.0,
    kv_dtype_bytes: int = 2,
) -> SplitPlan:
    """Plan the 32B split from the two budgets. The client is 'mlx4' (4-bit) or 'fp16'.

    reserve_gib is memory the budget does not count on, such as a CUDA context, and is
    subtracted before the plan is made. It defaults to 0 so the plan uses the stated budgets.
    """
    client_bytes = {"mlx4": MLX4_CLIENT_BYTES, "fp16": FP16_CLIENT_BYTES}.get(client)
    if client_bytes is None:
        raise PlanError(f"unknown client format {client!r}")
    kv = kv_bytes_per_token_per_layer(QWEN3_32B_KV_HEADS, QWEN3_32B_HEAD_DIM, kv_dtype_bytes)
    mac = Member(
        "mac",
        "metal",
        int((mac_budget_gib - mac_reserve_gib) * GiB),
        MLX4_LAYER_BYTES,
        fixed_bytes=client_bytes,
    )
    pc = Member("pc", "cuda", int((pc_budget_gib - pc_reserve_gib) * GiB), NF4_LAYER_BYTES)
    return plan_split(
        QWEN3_32B_LAYERS, mac, pc, kv_bytes_per_token_per_layer=kv, context_tokens=context_tokens
    )


def _format(plan: SplitPlan) -> str:
    lines = []
    for a in plan.assignments:
        lines.append(
            f"  {a.name:<4} layers [{a.start:>2}, {a.end:>2})  n={a.end - a.start:>2}  "
            f"used {a.used_bytes / GiB:6.3f} GiB of {a.budget_bytes / GiB:5.2f}  "
            f"headroom {a.headroom_bytes / GiB:+6.3f} GiB"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Plan the Qwen3-32B split from memory budgets.")
    p.add_argument("--mac-gib", type=float, default=11.0)
    p.add_argument("--pc-gib", type=float, default=6.5)
    p.add_argument("--ctx", type=int, default=2048, help="context tokens the KV cache must hold")
    p.add_argument("--client", choices=["mlx4", "fp16"], default="mlx4")
    args = p.parse_args(argv)
    try:
        plan = plan_qwen3_32b(args.mac_gib, args.pc_gib, context_tokens=args.ctx, client=args.client)
    except PlanError as e:
        print(f"REFUSED: {e}")
        return 1
    print(f"FITS: {QWEN3_32B_LAYERS} layers, context {plan.context_tokens}, client {args.client}")
    print(_format(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
